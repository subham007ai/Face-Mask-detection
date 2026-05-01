"""Step 10 validation-aware generator setup.

This adds a validation split from the training data only.

Rules enforced:
- Validation comes only from `data/train`
- Test data remains untouched until final evaluation
- Training uses augmentation
- Validation and test use rescaling only
- Batch size is 32 and class mode is binary
"""

from __future__ import annotations

import json
from pathlib import Path

from tensorflow.keras.preprocessing.image import ImageDataGenerator


ROOT = Path(r"D:\4thSemproject")
DATA_ROOT = ROOT / "data"
REPORT_PATH = ROOT / "reports" / "step10_validation_split_report.json"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 20260501
VALIDATION_SPLIT = 0.15


def build_generators():
    """Build train, validation, and test generators."""
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=20,
        zoom_range=0.15,
        horizontal_flip=True,
        validation_split=VALIDATION_SPLIT,
    )

    val_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        validation_split=VALIDATION_SPLIT,
    )

    test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_data = train_datagen.flow_from_directory(
        directory=str(DATA_ROOT / "train"),
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        subset="training",
        shuffle=True,
        seed=SEED,
    )

    val_data = val_datagen.flow_from_directory(
        directory=str(DATA_ROOT / "train"),
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        subset="validation",
        shuffle=False,
        seed=SEED,
    )

    test_data = test_datagen.flow_from_directory(
        directory=str(DATA_ROOT / "test"),
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=False,
    )

    return train_data, val_data, test_data


def build_report(train_data, val_data, test_data) -> dict:
    """Capture generator settings and verify a single batch from each split."""
    train_batch_x, train_batch_y = next(train_data)
    val_batch_x, val_batch_y = next(val_data)
    test_batch_x, test_batch_y = next(test_data)

    return {
        "seed": SEED,
        "validation_split": VALIDATION_SPLIT,
        "batch_size": BATCH_SIZE,
        "target_size": list(IMAGE_SIZE),
        "class_mode": "binary",
        "class_indices": dict(train_data.class_indices),
        "train": {
            "samples": int(train_data.samples),
            "batch_shape": list(train_batch_x.shape),
            "label_shape": list(train_batch_y.shape),
            "augmentation": {
                "rotation_range": 20,
                "zoom_range": 0.15,
                "horizontal_flip": True,
            },
            "rescale": "1./255",
        },
        "validation": {
            "samples": int(val_data.samples),
            "batch_shape": list(val_batch_x.shape),
            "label_shape": list(val_batch_y.shape),
            "augmentation": None,
            "rescale": "1./255",
        },
        "test": {
            "samples": int(test_data.samples),
            "batch_shape": list(test_batch_x.shape),
            "label_shape": list(test_batch_y.shape),
            "augmentation": None,
            "rescale": "1./255",
        },
    }


def main() -> None:
    train_data, val_data, test_data = build_generators()
    report = build_report(train_data, val_data, test_data)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"REPORT={REPORT_PATH}")


if __name__ == "__main__":
    main()
