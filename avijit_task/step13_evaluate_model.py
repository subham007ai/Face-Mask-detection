"""Step 13 – Evaluate the trained models on the test set.

Team Member 3 (or 4): Evaluation / Inference
- Scans directory for mask_model_*.h5
- Fetches the unmodified test data generator
- Predicts on the held-out test dataset for each model
- Generates a Confusion Matrix and Classification Report for each
- Saves the evaluation metrics to reports/
- Prints a final comparison summary
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from step10_validation_split_generators import build_generators

ROOT = Path(__file__).parent.resolve()
REPORTS_DIR = ROOT / "reports"


def main() -> None:
    # Find all h5 models
    model_paths = list(ROOT.glob("mask_model_*.h5"))
    if not model_paths:
        print("No models found! Please run step12_train_model.py first to train the models.")
        return

    print("Loading test data generator...")
    # We only need the test generator
    _, _, test_data = build_generators()
    
    true_classes = test_data.classes
    class_indices = test_data.class_indices
    class_labels = [k for k, v in sorted(class_indices.items(), key=lambda item: item[1])]

    results_summary = []

    for model_path in model_paths:
        # Extract architecture name, e.g., mask_model_EfficientNetB0.h5 -> EfficientNetB0
        arch_name = model_path.stem.replace("mask_model_", "")
        if arch_name == "mask_model":
            # Fallback for the older naming convention
            arch_name = "EfficientNetB0"
        
        print(f"\n{'='*50}")
        print(f"🔍 EVALUATING MODEL: {arch_name}")
        print(f"{'='*50}")
        
        tf.keras.backend.clear_session()
        
        print(f"Loading trained model from {model_path.name}...")
        model = tf.keras.models.load_model(model_path)

        print(f"Evaluating {arch_name} on test data...")
        eval_metrics = model.evaluate(test_data, verbose=1)
        
        print(f"Generating predictions for {arch_name}...")
        predictions = model.predict(test_data, verbose=1)
        predicted_classes = (predictions > 0.5).astype(int).flatten()
        
        cm = confusion_matrix(true_classes, predicted_classes)
        cr_dict = classification_report(
            true_classes, 
            predicted_classes, 
            target_names=class_labels, 
            output_dict=True
        )

        report = {
            "model_name": arch_name,
            "test_loss": float(eval_metrics[0]),
            "test_accuracy": float(eval_metrics[1]),
            "confusion_matrix": {
                f"True {class_labels[0]}": int(cm[0][0]),
                f"False {class_labels[1]}": int(cm[0][1]),
                f"False {class_labels[0]}": int(cm[1][0]),
                f"True {class_labels[1]}": int(cm[1][1]),
                "raw_matrix": cm.tolist()
            },
            "classification_report": cr_dict
        }

        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        report_file = REPORTS_DIR / f"step13_evaluation_report_{arch_name}.json"
        report_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Report saved to: {report_file}")
        
        # Save to summary
        results_summary.append({
            "Architecture": arch_name,
            "Accuracy": report["test_accuracy"],
            "Loss": report["test_loss"]
        })

    # Print Final Summary Table
    print("\n\n" + "#"*60)
    print("🏆 FINAL MODEL COMPARISON SUMMARY 🏆".center(60))
    print("#"*60)
    print(f"{'Architecture':<20} | {'Test Accuracy':<15} | {'Test Loss':<15}")
    print("-" * 60)
    
    # Sort by accuracy descending
    results_summary.sort(key=lambda x: x["Accuracy"], reverse=True)
    
    for res in results_summary:
        acc_str = f"{res['Accuracy']*100:.2f}%"
        loss_str = f"{res['Loss']:.4f}"
        print(f"{res['Architecture']:<20} | {acc_str:<15} | {loss_str:<15}")
    
    print("-" * 60)
    if results_summary:
        best_model = results_summary[0]['Architecture']
        print(f"🎯 WINNER: {best_model} is the best performing model!")
    print("#"*60 + "\n")


if __name__ == "__main__":
    main()
