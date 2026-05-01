"""Step 11 – Compute class weights for imbalanced training data.

This step calculates balanced class weights from the training generator
built in Step 10.  The weights compensate for any class imbalance so the
model does not become biased toward the majority class.

Usage in model training:
    from step11_class_weights import get_class_weights

    class_weights = get_class_weights()   # e.g. {0: 1.12, 1: 0.90}
    model.fit(..., class_weight=class_weights)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.utils.class_weight import compute_class_weight

from step10_validation_split_generators import build_generators


ROOT = Path(r"D:\4thSemproject")
REPORT_PATH = ROOT / "reports" / "step11_class_weights_report.json"


def _compute_weights(train_data) -> dict:
    """Compute balanced class weights from the training generator.

    Parameters
    ----------
    train_data : DirectoryIterator
        The training generator returned by ``build_generators()``.

    Returns
    -------
    dict
        Mapping of integer class index to its computed weight,
        e.g. ``{0: 1.12, 1: 0.90}``.
    """
    # train_data.classes contains the integer label for every sample
    classes = np.unique(train_data.classes)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=train_data.classes,
    )

    return dict(enumerate(weights))


def get_class_weights() -> dict:
    """Public helper – returns class weights ready for ``model.fit()``.

    Example
    -------
    >>> from step11_class_weights import get_class_weights
    >>> model.fit(X, y, class_weight=get_class_weights())
    """
    train_data, _, _ = build_generators()
    return _compute_weights(train_data)


def build_report(train_data, class_weights: dict) -> dict:
    """Build an auditable report for step 11."""
    class_indices = dict(train_data.class_indices)  # e.g. {'WithMask': 1, 'WithoutMask': 0}

    # Per-class sample counts
    unique, counts = np.unique(train_data.classes, return_counts=True)
    class_counts = {int(k): int(v) for k, v in zip(unique, counts)}

    return {
        "class_indices": class_indices,
        "total_train_samples": int(train_data.samples),
        "per_class_sample_counts": {
            cls: class_counts.get(idx, 0)
            for cls, idx in class_indices.items()
        },
        "class_weights": {int(k): round(float(v), 6) for k, v in class_weights.items()},
        "usage": "Pass the class_weights dict to model.fit(class_weight=class_weights)",
    }


def main() -> None:
    train_data, _, _ = build_generators()
    class_weights = _compute_weights(train_data)
    report = build_report(train_data, class_weights)

    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nComputed class weights: {class_weights}")
    print(f"REPORT={REPORT_PATH}")


if __name__ == "__main__":
    main()
