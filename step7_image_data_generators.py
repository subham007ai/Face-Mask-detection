"""Step 7 Keras-style ImageDataGenerator setup.

This script wires the final `data/train` and `data/test` directories into
training and test generators with:
- target size 224x224
- batch size 32
- binary class mode
- augmentation only for the training split
- rescaling only for the test split

The script writes a JSON report and verifies one batch from each generator
so the configuration is auditable.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator


ROOT = Path(__file__).parent.resolve()
DATA_ROOT = ROOT / "data"
REPORT_PATH = ROOT / "reports" / "step7_generator_report.json"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 20260501


def build_generators():
    """Build training and test ImageDataGenerator iterators."""
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=20,
        zoom_range=0.15,
        horizontal_flip=True,
    )

    test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_generator = train_datagen.flow_from_directory(
        directory=str(DATA_ROOT / "train"),
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=True,
        seed=SEED,
    )

    test_generator = test_datagen.flow_from_directory(
        directory=str(DATA_ROOT / "test"),
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=False,
    )

    return train_generator, test_generator


def build_report(train_generator, test_generator) -> dict:
    """Capture the final generator configuration and a small sanity check."""
    train_batch_x, train_batch_y = next(train_generator)
    test_batch_x, test_batch_y = next(test_generator)

    return {
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "target_size": list(IMAGE_SIZE),
        "class_mode": "binary",
        "train": {
            "samples": int(train_generator.samples),
            "class_indices": dict(train_generator.class_indices),
            "batch_shape": list(train_batch_x.shape),
            "label_shape": list(train_batch_y.shape),
            "augmentation": {
                "rotation_range": 20,
                "zoom_range": 0.15,
                "horizontal_flip": True,
            },
            "rescale": "1./255",
        },
        "test": {
            "samples": int(test_generator.samples),
            "class_indices": dict(test_generator.class_indices),
            "batch_shape": list(test_batch_x.shape),
            "label_shape": list(test_batch_y.shape),
            "augmentation": None,
            "rescale": "1./255",
        },
    }


def main() -> None:
    train_generator, test_generator = build_generators()
    report = build_report(train_generator, test_generator)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"REPORT={REPORT_PATH}")


if __name__ == "__main__":
    main()
