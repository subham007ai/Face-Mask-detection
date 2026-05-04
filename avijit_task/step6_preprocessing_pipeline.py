"""Step 6 preprocessing pipeline for the face mask dataset.

This module keeps the split data on disk unchanged and provides
framework-agnostic preprocessing utilities for downstream training.

Behavior:
- Train: resize to 224x224, rescale to [0, 1], apply random rotation,
  random zoom, and horizontal flip.
- Test: resize to 224x224, rescale to [0, 1], no augmentation.

The script can also generate a small preview image and a JSON report
so the step is auditable.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Iterator, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).parent.resolve()
DATA_ROOT = ROOT / "data"
REPORT_PATH = ROOT / "reports" / "step6_pipeline_report.json"
PREVIEW_PATH = ROOT / "reports" / "step6_preview.png"

IMAGE_SIZE = (224, 224)
SEED = 20260501
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASSES = ("WithMask", "WithoutMask")


def list_samples(split: str) -> List[Tuple[Path, str]]:
    """Return (path, label) pairs for a split in deterministic order."""
    split_root = DATA_ROOT / split
    samples: List[Tuple[Path, str]] = []
    for label in CLASSES:
        for path in sorted((split_root / label).glob("*")):
            if path.is_file() and path.suffix.lower() in EXTENSIONS:
                samples.append((path, label))
    return samples


def load_rgb(path: Path) -> Image.Image:
    """Load an image as RGB."""
    with Image.open(path) as img:
        return img.convert("RGB")


def resize_to_target(image: Image.Image) -> Image.Image:
    """Resize image to the model input size."""
    return ImageOps.fit(
        image,
        IMAGE_SIZE,
        method=Image.Resampling.LANCZOS,
        centering=(0.5, 0.5),
    )


def rescale_array(image: Image.Image) -> np.ndarray:
    """Convert to float32 array in [0, 1]."""
    return np.asarray(image, dtype=np.float32) / 255.0


def _zoom_image(image: Image.Image, zoom: float) -> Image.Image:
    """Apply a center zoom while keeping the final size fixed."""
    width, height = image.size
    if math.isclose(zoom, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        return image

    if zoom > 1.0:
        new_w = max(1, int(round(width * zoom)))
        new_h = max(1, int(round(height * zoom)))
        enlarged = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        left = max(0, (new_w - width) // 2)
        top = max(0, (new_h - height) // 2)
        return enlarged.crop((left, top, left + width, top + height))

    # zoom < 1.0
    new_w = max(1, int(round(width * zoom)))
    new_h = max(1, int(round(height * zoom)))
    shrunk = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), (0, 0, 0))
    left = (width - new_w) // 2
    top = (height - new_h) // 2
    canvas.paste(shrunk, (left, top))
    return canvas


def augment_train_image(image: Image.Image, rng: random.Random) -> Image.Image:
    """Apply training-only augmentation before rescaling."""
    image = resize_to_target(image)

    if rng.random() < 0.5:
        image = ImageOps.mirror(image)

    angle = rng.uniform(-20.0, 20.0)
    image = image.rotate(
        angle,
        resample=Image.Resampling.BILINEAR,
        expand=False,
        fillcolor=(0, 0, 0),
    )

    zoom = rng.uniform(0.85, 1.15)
    image = _zoom_image(image, zoom)
    return image


def preprocess_test_image(image: Image.Image) -> Image.Image:
    """Apply test-time standardization only."""
    return resize_to_target(image)


def iter_batches(
    split: str,
    batch_size: int,
    augment: bool = False,
    seed: int = SEED,
    shuffle: bool = True,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Yield batches of standardized image arrays and integer labels."""
    samples = list_samples(split)
    rng = random.Random(seed)
    if shuffle:
        rng.shuffle(samples)

    for start in range(0, len(samples), batch_size):
        batch = samples[start : start + batch_size]
        batch_x = []
        batch_y = []
        for path, label in batch:
            image = load_rgb(path)
            image = augment_train_image(image, rng) if augment else preprocess_test_image(image)
            batch_x.append(rescale_array(image))
            batch_y.append(1 if label == "WithMask" else 0)
        yield np.stack(batch_x, axis=0), np.asarray(batch_y, dtype=np.int64)


def build_report() -> dict:
    """Build a summary of the step 6 preprocessing setup."""
    train_samples = list_samples("train")
    test_samples = list_samples("test")

    return {
        "seed": SEED,
        "image_size": list(IMAGE_SIZE),
        "train_samples": len(train_samples),
        "test_samples": len(test_samples),
        "train_class_counts": {
            "WithMask": sum(1 for _, label in train_samples if label == "WithMask"),
            "WithoutMask": sum(1 for _, label in train_samples if label == "WithoutMask"),
        },
        "test_class_counts": {
            "WithMask": sum(1 for _, label in test_samples if label == "WithMask"),
            "WithoutMask": sum(1 for _, label in test_samples if label == "WithoutMask"),
        },
        "transforms": {
            "train": {
                "resize": "224x224",
                "rescale": "1./255",
                "augmentation": {
                    "rotation_degrees": [-20, 20],
                    "zoom_range": [0.85, 1.15],
                    "horizontal_flip": True,
                },
            },
            "test": {
                "resize": "224x224",
                "rescale": "1./255",
                "augmentation": None,
            },
        },
    }


def _make_preview_tile(image: Image.Image, width: int = 280, height: int = 210) -> Image.Image:
    """Resize an image for display while preserving aspect ratio."""
    return ImageOps.contain(image, (width, height))


def create_preview() -> None:
    """Create a visual preview of training augmentation and test standardization."""
    rng = random.Random(SEED)
    train_samples = list_samples("train")
    test_samples = list_samples("test")

    # Keep the preview small but representative.
    train_pick = rng.sample(train_samples, 4)
    test_pick = rng.sample(test_samples, 4)
    rows = [("train", path, label) for path, label in train_pick] + [
        ("test", path, label) for path, label in test_pick
    ]

    tile_w = 300
    tile_h = 220
    pad = 12
    label_h = 54
    sheet_w = pad * 3 + tile_w * 2
    sheet_h = pad * (len(rows) + 1) + (tile_h + label_h) * len(rows)
    sheet = Image.new("RGB", (sheet_w, sheet_h), "white")
    draw = ImageDraw.Draw(sheet)

    try:
        font = ImageFont.truetype("arial.ttf", 15)
        small = ImageFont.truetype("arial.ttf", 12)
    except Exception:
        font = ImageFont.load_default()
        small = ImageFont.load_default()

    for idx, (split, path, label) in enumerate(rows):
        y0 = pad + idx * (tile_h + label_h + pad)
        original = load_rgb(path)
        if split == "train":
            transformed = augment_train_image(original, rng)
            title = "train"
            right_label = "augmented + resized + rescaled"
        else:
            transformed = preprocess_test_image(original)
            title = "test"
            right_label = "resized + rescaled only"

        left_tile = _make_preview_tile(original, width=tile_w, height=tile_h)
        right_tile = _make_preview_tile(transformed, width=tile_w, height=tile_h)
        left_canvas = Image.new("RGB", (tile_w, tile_h), "#f3f3f3")
        right_canvas = Image.new("RGB", (tile_w, tile_h), "#f3f3f3")
        left_canvas.paste(left_tile, ((tile_w - left_tile.width) // 2, (tile_h - left_tile.height) // 2))
        right_canvas.paste(right_tile, ((tile_w - right_tile.width) // 2, (tile_h - right_tile.height) // 2))

        x_left = pad
        x_right = pad * 2 + tile_w
        sheet.paste(left_canvas, (x_left, y0))
        sheet.paste(right_canvas, (x_right, y0))

        draw.rectangle([x_left, y0 + tile_h, x_left + tile_w, y0 + tile_h + label_h], outline="black", width=1)
        draw.rectangle([x_right, y0 + tile_h, x_right + tile_w, y0 + tile_h + label_h], outline="black", width=1)
        draw.text((x_left + 6, y0 + tile_h + 4), f"{idx + 1}. {title} original", fill="black", font=small)
        draw.text((x_left + 6, y0 + tile_h + 22), f"{label} | {path.name}", fill="black", font=small)
        draw.text((x_right + 6, y0 + tile_h + 4), f"{idx + 1}. {right_label}", fill="black", font=small)
        draw.text((x_right + 6, y0 + tile_h + 22), f"target: 224x224 | 1./255", fill="black", font=small)

    sheet.save(PREVIEW_PATH)


def main() -> None:
    report = build_report()
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    create_preview()
    print(json.dumps(report, indent=2))
    print(f"REPORT={REPORT_PATH}")
    print(f"PREVIEW={PREVIEW_PATH}")


if __name__ == "__main__":
    main()
