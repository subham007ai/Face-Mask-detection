"""Step 12 – CNN training with multiple backbones.

Team Member 2: Machine Learning Engineer
- Iterates over 3 architectures: EfficientNetB0, MobileNetV2, ResNet50
- Custom head: GlobalAveragePooling2D → Dropout → Dense (sigmoid)
- Binary crossentropy + Adam
- Offline training using augmented data from step10 generators
- Saves final weights to mask_model_{Arch}.h5
"""

from __future__ import annotations

import json
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0, MobileNetV2, ResNet50
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model

from step10_validation_split_generators import build_generators


ROOT = Path(__file__).parent.resolve()
REPORTS_DIR = ROOT / "reports"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 25

ARCHITECTURES = {
    "EfficientNetB0": EfficientNetB0,
    "MobileNetV2": MobileNetV2,
    "ResNet50": ResNet50
}

def build_model(architecture_name: str) -> Model:
    """Construct model with custom classification head.

    Transfer learning strategy:
    - Base (ImageNet weights) is FROZEN — only the custom head trains.
    - This preserves learned low-level features and prevents overfitting
      on a small dataset. Unfreeze base layers in a second fine-tuning
      pass once the head has converged.
    """
    inputs = Input(shape=(*IMAGE_SIZE, 3))

    base_model_class = ARCHITECTURES[architecture_name]

    base = base_model_class(include_top=False, weights="imagenet", input_tensor=inputs)

    base.trainable = False
    print(f"[INFO] {architecture_name} base frozen — {len(base.layers)} layers locked.")

    x = GlobalAveragePooling2D()(base.output)
    x = Dropout(0.5)(x)
    outputs = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=inputs, outputs=outputs, name=architecture_name)
    return model


def compile_model(model: Model) -> None:
    """Compile with binary crossentropy and Adam optimizer."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )


def train_model(model: Model, train_data, val_data, class_weights: dict) -> tf.keras.callbacks.History:
    """Execute offline training loop."""
    return model.fit(
        train_data,
        validation_data=val_data,
        epochs=EPOCHS,
        class_weight=class_weights,
        verbose=1,
    )


def build_report(architecture_name: str, history: tf.keras.callbacks.History, train_data, val_data, model_path: Path) -> dict:
    """Build training summary report."""
    return {
        "model": architecture_name,
        "architecture": {
            "base": f"{architecture_name} (ImageNet weights, include_top=False)",
            "head": ["GlobalAveragePooling2D", "Dropout(0.5)", "Dense(1, sigmoid)"],
        },
        "compilation": {
            "optimizer": "Adam (lr=1e-4)",
            "loss": "binary_crossentropy",
            "metrics": ["accuracy"],
        },
        "training": {
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "train_samples": int(train_data.samples),
            "val_samples": int(val_data.samples),
        },
        "final_metrics": {
            "train_accuracy": float(history.history["accuracy"][-1]),
            "train_loss": float(history.history["loss"][-1]),
            "val_accuracy": float(history.history["val_accuracy"][-1]),
            "val_loss": float(history.history["val_loss"][-1]),
        },
        "saved_model": str(model_path),
    }


def main() -> None:
    print("Building generators...")
    train_data, val_data, _ = build_generators()

    print("Computing class weights...")
    from step11_class_weights import get_class_weights
    class_weights = get_class_weights()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    for arch_name in ARCHITECTURES.keys():
        print(f"\n{'='*50}")
        print(f"STARTING TRAINING FOR: {arch_name}")
        print(f"{'='*50}\n")
        
        tf.keras.backend.clear_session()
        
        model = build_model(arch_name)
        compile_model(model)
        
        print(f"Training {arch_name} for {EPOCHS} epochs...")
        history = train_model(model, train_data, val_data, class_weights)

        model_path = ROOT / f"mask_model_{arch_name}.h5"
        print(f"\nSaving {arch_name} model weights...")
        model.save(model_path)
        print(f"Model saved to {model_path}")

        report = build_report(arch_name, history, train_data, val_data, model_path)
        report_path = REPORTS_DIR / f"step12_training_report_{arch_name}.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"REPORT={report_path}")

    print("\nAll 3 models trained successfully!")

if __name__ == "__main__":
    main()
