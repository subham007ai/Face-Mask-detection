import json
import os
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator

ROOT = Path(__file__).parent.parent.resolve()
DATA_ROOT = ROOT / "data" / "test"
MODEL_DIR = ROOT / "avijit_task"
SREYAN_DIR = ROOT / "sreyan"

SREYAN_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

def main():
    print("=" * 60)
    print("[*] INITIATING MEMBER 3 EVALUATION PIPELINE (MULTI-MODEL)")
    print("=" * 60)

    model_paths = list(MODEL_DIR.glob("*.h5"))
    if not model_paths:
        print(f"ERROR: No model files found in {MODEL_DIR}")
        return

    print(f"[*] Found {len(model_paths)} models to evaluate.")

    print(f"[*] Loading test dataset from: {DATA_ROOT}")
    test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    
    test_data = test_datagen.flow_from_directory(
        directory=str(DATA_ROOT),
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=False, 
    )

    true_classes = test_data.classes
    class_indices = test_data.class_indices
    class_labels = [k for k, v in sorted(class_indices.items(), key=lambda item: item[1])]

    all_metrics = {}
    
    plt.figure(1, figsize=(10, 8))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Receiver Operating Characteristic (ROC) Comparison')
    plt.grid(alpha=0.3)

    plt.figure(2, figsize=(10, 8))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall (PR) Curve Comparison')
    plt.grid(alpha=0.3)

    colors = ['darkorange', 'purple', 'green', 'red', 'blue']

    for i, model_path in enumerate(model_paths):
        model_name = model_path.stem.replace("mask_model_", "")
        if model_name == "mask_model":
             model_name = "EfficientNetB0"
             
        print(f"\n{'-' * 40}")
        print(f"[*] Evaluating Model: {model_name}")
        print(f"{'-' * 40}")
        
        tf.keras.backend.clear_session()
        
        model = tf.keras.models.load_model(model_path)
        
        print(f"    - Generating probability distributions...")
        probabilities = model.predict(test_data, verbose=1).flatten()
        predicted_classes = (probabilities > 0.5).astype(int)

        print(f"    - Computing metrics...")
        acc = accuracy_score(true_classes, predicted_classes)
        precision = precision_score(true_classes, predicted_classes)
        recall = recall_score(true_classes, predicted_classes)
        f1 = f1_score(true_classes, predicted_classes)
        roc_auc = roc_auc_score(true_classes, probabilities)

        print(f"    - Accuracy:  {acc:.4f}")
        print(f"    - Precision: {precision:.4f}")
        print(f"    - Recall:    {recall:.4f}")
        print(f"    - F1-Score:  {f1:.4f}")
        print(f"    - ROC-AUC:   {roc_auc:.4f}")

        all_metrics[model_name] = {
            "accuracy": float(acc),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "roc_auc": float(roc_auc)
        }

        fpr, tpr, _ = roc_curve(true_classes, probabilities)
        plt.figure(1)
        plt.plot(fpr, tpr, color=colors[i % len(colors)], lw=2, label=f'{model_name} (AUC = {roc_auc:.2f})')

        precisions, recalls, _ = precision_recall_curve(true_classes, probabilities)
        pr_auc = auc(recalls, precisions)
        plt.figure(2)
        plt.plot(recalls, precisions, color=colors[i % len(colors)], lw=2, label=f'{model_name} (AUC = {pr_auc:.2f})')

    plt.figure(1)
    plt.legend(loc="lower right")
    roc_path = SREYAN_DIR / "roc_curve_comparison.png"
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n[*] Combined ROC Curve saved to {roc_path}")

    plt.figure(2)
    plt.legend(loc="lower left")
    pr_path = SREYAN_DIR / "pr_curve_comparison.png"
    plt.savefig(pr_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[*] Combined PR Curve saved to {pr_path}")

    report_path = SREYAN_DIR / "all_models_evaluation_metrics.json"
    with open(report_path, "w") as f:
        json.dump({
            "class_mapping": class_indices,
            "models": all_metrics
        }, f, indent=4)
    print(f"[*] Saved all metrics to {report_path}")

    print("=" * 60)
    print("[DONE] MULTI-MODEL EVALUATION PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)

if __name__ == "__main__":
    main()
