"""Exploratory Data Analysis for the face mask dataset.

This script analyses the final prepared dataset (data/train and data/test)
and produces a multi-panel PDF report plus a JSON summary.

Analyses performed
------------------
1. Class distribution  – bar chart of per-class counts for each split.
2. Sample image grid   – representative thumbnails per class (train + test).
3. Resolution analysis – width and height histograms across classes.
4. Pixel channel stats – per-channel (R, G, B) mean and std for each class.
5. Blur score dist.    – Laplacian variance distribution per class.

All figures are saved to reports/eda_*.png and a summary is written to
reports/step_eda_report.json.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

ROOT      = Path(r"D:\4thSemproject")
DATA_ROOT = ROOT / "data"
REPORT_DIR = ROOT / "reports"

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASSES    = ("WithMask", "WithoutMask")
SPLITS     = ("train", "test")
SEED       = 20260501

# Matplotlib palette (calm, professional)
PALETTE = {
    "WithMask":    "#2EB872",   # teal-green
    "WithoutMask": "#E05A5A",   # muted red
}

FIGURE_DPI = 140

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def list_samples(split: str) -> List[Tuple[Path, str]]:
    """Return (path, label) pairs for a split in deterministic order."""
    split_root = DATA_ROOT / split
    samples: List[Tuple[Path, str]] = []
    for label in CLASSES:
        class_dir = split_root / label
        if not class_dir.is_dir():
            continue
        for path in sorted(class_dir.glob("*")):
            if path.is_file() and path.suffix.lower() in EXTENSIONS:
                samples.append((path, label))
    return samples


def load_pil(path: Path) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("RGB")


def laplacian_blur_score(path: Path) -> float:
    """Return Laplacian variance (higher = sharper)."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    return float(cv2.Laplacian(img, cv2.CV_64F).var())


# ---------------------------------------------------------------------------
# Analysis collectors
# ---------------------------------------------------------------------------

def collect_all(
    samples: List[Tuple[Path, str]],
    sample_limit: int = 5000,
) -> Dict:
    """Walk samples and collect resolution, channel stats, and blur scores.

    If a split has more than *sample_limit* images we take a stratified
    random sub-sample to keep runtime reasonable.
    """
    rng = random.Random(SEED)

    # Stratified sub-sample per class
    by_class: Dict[str, List[Path]] = {c: [] for c in CLASSES}
    for path, label in samples:
        by_class[label].append(path)

    per_class_limit = sample_limit // len(CLASSES)
    selected: List[Tuple[Path, str]] = []
    for label, paths in by_class.items():
        pick = rng.sample(paths, min(len(paths), per_class_limit))
        selected.extend((p, label) for p in pick)

    rng.shuffle(selected)

    widths:  Dict[str, List[int]]   = {c: [] for c in CLASSES}
    heights: Dict[str, List[int]]   = {c: [] for c in CLASSES}
    means:   Dict[str, List[float]] = {c: [] for c in CLASSES}  # per-image mean across channels
    stds:    Dict[str, List[float]] = {c: [] for c in CLASSES}
    ch_means: Dict[str, Dict[str, List[float]]] = {
        c: {"R": [], "G": [], "B": []} for c in CLASSES
    }
    blurs:   Dict[str, List[float]] = {c: [] for c in CLASSES}

    total = len(selected)
    for i, (path, label) in enumerate(selected, 1):
        if i % 500 == 0 or i == total:
            print(f"  collecting ... {i}/{total}", flush=True)
        try:
            img = load_pil(path)
        except Exception:
            continue

        w, h = img.size
        widths[label].append(w)
        heights[label].append(h)

        arr = np.asarray(img, dtype=np.float32) / 255.0  # H×W×3
        means[label].append(float(arr.mean()))
        stds[label].append(float(arr.std()))
        for ci, ch in enumerate(["R", "G", "B"]):
            ch_means[label][ch].append(float(arr[:, :, ci].mean()))

        blurs[label].append(laplacian_blur_score(path))

    return {
        "widths": widths,
        "heights": heights,
        "means": means,
        "stds": stds,
        "ch_means": ch_means,
        "blurs": blurs,
        "sampled": {label: len(widths[label]) for label in CLASSES},
    }


# ---------------------------------------------------------------------------
# Figure 1 – Class distribution
# ---------------------------------------------------------------------------

def fig_class_distribution(all_splits: Dict[str, List[Tuple[Path, str]]]) -> Path:
    """Bar chart of per-class counts for train and test splits."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), dpi=FIGURE_DPI)
    fig.suptitle("Class Distribution", fontsize=15, fontweight="bold", y=1.01)

    for ax, split in zip(axes, SPLITS):
        samples = all_splits[split]
        counts = {c: sum(1 for _, l in samples if l == c) for c in CLASSES}
        colors = [PALETTE[c] for c in CLASSES]
        bars = ax.bar(CLASSES, [counts[c] for c in CLASSES], color=colors,
                      edgecolor="white", linewidth=0.8, width=0.55)
        for bar, count in zip(bars, [counts[c] for c in CLASSES]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 20,
                f"{count:,}",
                ha="center", va="bottom", fontsize=11, fontweight="bold",
            )
        total = sum(counts.values())
        ax.set_title(f"{split.capitalize()}  (n={total:,})", fontsize=12, pad=8)
        ax.set_ylabel("Image count", fontsize=10)
        ax.set_ylim(0, max(counts.values()) * 1.18)
        ax.tick_params(axis="x", labelsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.yaxis.grid(True, linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)

    fig.tight_layout()
    out = REPORT_DIR / "eda_01_class_distribution.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {out.name}")
    return out


# ---------------------------------------------------------------------------
# Figure 2 – Sample image grid
# ---------------------------------------------------------------------------

def fig_sample_grid(all_splits: Dict[str, List[Tuple[Path, str]]]) -> Path:
    """5x4 grid: 5 images per class x 2 classes x 2 splits (train + test)."""
    rng = random.Random(SEED + 1)
    N_COLS = 5
    THUMB = 112  # px thumbnail size

    rows: List[Tuple[str, str, List[Path]]] = []
    for split in SPLITS:
        by_class = {c: [] for c in CLASSES}
        for path, label in all_splits[split]:
            by_class[label].append(path)
        for cls in CLASSES:
            pick = rng.sample(by_class[cls], min(N_COLS, len(by_class[cls])))
            rows.append((split, cls, pick))

    n_rows = len(rows)
    fig, axes = plt.subplots(n_rows, N_COLS,
                             figsize=(N_COLS * 1.4, n_rows * 1.55),
                             dpi=FIGURE_DPI)
    fig.suptitle("Sample Image Grid  (train & test, both classes)",
                 fontsize=13, fontweight="bold", y=1.01)

    for row_idx, (split, cls, paths) in enumerate(rows):
        for col_idx in range(N_COLS):
            ax = axes[row_idx][col_idx]
            if col_idx < len(paths):
                img = load_pil(paths[col_idx]).resize((THUMB, THUMB), Image.LANCZOS)
                ax.imshow(np.asarray(img))
            else:
                ax.set_facecolor("#e0e0e0")
            ax.axis("off")
            if col_idx == 0:
                ax.set_ylabel(
                    f"{split}\n{cls}",
                    fontsize=8, rotation=0, labelpad=58,
                    va="center", ha="right",
                    color=PALETTE[cls], fontweight="bold",
                )

    fig.tight_layout(pad=0.4)
    out = REPORT_DIR / "eda_02_sample_grid.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {out.name}")
    return out


# ---------------------------------------------------------------------------
# Figure 3 – Resolution histograms (width & height)
# ---------------------------------------------------------------------------

def fig_resolution(collected: Dict) -> Path:
    """Overlapping histograms of original image width and height per class."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=FIGURE_DPI)
    fig.suptitle("Image Resolution Distribution (before resizing)",
                 fontsize=13, fontweight="bold", y=1.01)

    for ax, dim, key in zip(axes, ["Width", "Height"], ["widths", "heights"]):
        for cls in CLASSES:
            vals = collected[key][cls]
            if not vals:
                continue
            ax.hist(
                vals, bins=50, alpha=0.65, color=PALETTE[cls],
                label=f"{cls}  (mean={np.mean(vals):.0f}px)",
                edgecolor="none",
            )
        ax.set_title(f"Image {dim}", fontsize=11)
        ax.set_xlabel(f"{dim} (px)", fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.legend(fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.yaxis.grid(True, linestyle="--", alpha=0.35)
        ax.set_axisbelow(True)

    fig.tight_layout()
    out = REPORT_DIR / "eda_03_resolution.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {out.name}")
    return out


# ---------------------------------------------------------------------------
# Figure 4 – Pixel channel statistics
# ---------------------------------------------------------------------------

def fig_channel_stats(collected: Dict) -> Path:
    """Grouped bar chart of per-channel mean (R/G/B) for each class."""
    channels = ["R", "G", "B"]
    ch_colors = ["#e05a5a", "#6abf69", "#5a9ce0"]
    x = np.arange(len(CLASSES))
    width = 0.22
    offsets = [-width, 0, width]

    fig, ax = plt.subplots(figsize=(9, 5), dpi=FIGURE_DPI)
    fig.suptitle("Per-Channel Mean Pixel Intensity (R / G / B)",
                 fontsize=13, fontweight="bold")

    for i, (ch, col, off) in enumerate(zip(channels, ch_colors, offsets)):
        ch_vals = [
            np.mean(collected["ch_means"][cls][ch]) if collected["ch_means"][cls][ch] else 0
            for cls in CLASSES
        ]
        ch_errs = [
            np.std(collected["ch_means"][cls][ch]) if collected["ch_means"][cls][ch] else 0
            for cls in CLASSES
        ]
        bars = ax.bar(
            x + off, ch_vals, width,
            label=f"Channel {ch}",
            color=col, alpha=0.82,
            edgecolor="white", linewidth=0.6,
            yerr=ch_errs, capsize=4, error_kw={"linewidth": 1.2},
        )
        for bar, v in zip(bars, ch_vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.004,
                f"{v:.3f}",
                ha="center", va="bottom", fontsize=7.5,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES, fontsize=11)
    ax.set_ylabel("Mean pixel value  [0, 1]", fontsize=10)
    ax.set_ylim(0, 0.75)
    ax.legend(fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)

    fig.tight_layout()
    out = REPORT_DIR / "eda_04_channel_stats.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {out.name}")
    return out


# ---------------------------------------------------------------------------
# Figure 5 – Blur score distribution
# ---------------------------------------------------------------------------

def fig_blur(collected: Dict) -> Path:
    """Histogram + box plot of Laplacian blur variance per class,
    with a vertical reference line at the Step-3 quality threshold (var=65).
    """
    BLUR_THRESHOLD = 65  # Step 3 quality filter: images below this were removed

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), dpi=FIGURE_DPI)
    fig.suptitle(
        "Blur Score Distribution  (Laplacian Variance, higher = sharper)",
        fontsize=12, fontweight="bold", y=1.01,
    )

    # --- left: overlapping histograms ---
    ax = axes[0]
    for cls in CLASSES:
        vals = collected["blurs"][cls]
        if not vals:
            continue
        # cap extreme outliers for readability
        cap = np.percentile(vals, 98)
        capped = [min(v, cap) for v in vals]
        ax.hist(capped, bins=60, alpha=0.60, color=PALETTE[cls],
                label=f"{cls}  (mean={np.mean(vals):.1f})", edgecolor="none")

    # Step 3 quality threshold
    ax.axvline(
        x=BLUR_THRESHOLD, color="#e53935", linewidth=1.6,
        linestyle="--", zorder=5,
    )
    ymax = ax.get_ylim()[1]
    ax.text(
        BLUR_THRESHOLD + 4, ymax * 0.92,
        f"Step-3 threshold\n(var = {BLUR_THRESHOLD})",
        color="#e53935", fontsize=8, va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#e53935",
                  alpha=0.85, linewidth=0.8),
    )

    ax.set_title("Histogram (98th-pct capped)", fontsize=10)
    ax.set_xlabel("Laplacian variance", fontsize=9)
    ax.set_ylabel("Count", fontsize=9)
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)

    # --- right: box plot ---
    ax = axes[1]
    data_to_plot = [collected["blurs"][cls] for cls in CLASSES]
    bp = ax.boxplot(
        data_to_plot,
        patch_artist=True,
        notch=False,
        medianprops={"color": "white", "linewidth": 2},
        flierprops={"marker": ".", "markersize": 2, "alpha": 0.3},
        showfliers=True,
    )
    for patch, cls in zip(bp["boxes"], CLASSES):
        patch.set_facecolor(PALETTE[cls])
        patch.set_alpha(0.75)
    # threshold reference on box plot
    ax.axhline(
        y=BLUR_THRESHOLD, color="#e53935", linewidth=1.4,
        linestyle="--", zorder=5, label=f"Threshold (var={BLUR_THRESHOLD})",
    )
    ax.legend(fontsize=8, loc="upper left")
    ax.set_xticks([1, 2])
    ax.set_xticklabels(CLASSES, fontsize=10)
    ax.set_ylabel("Laplacian variance", fontsize=9)
    ax.set_title("Box Plot", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)

    fig.tight_layout()
    out = REPORT_DIR / "eda_05_blur.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {out.name}")
    return out


# ---------------------------------------------------------------------------
# JSON report
# ---------------------------------------------------------------------------

def build_json_report(
    all_splits: Dict[str, List[Tuple[Path, str]]],
    train_collected: Dict,
    test_collected: Dict,
    figures: List[Path],
    elapsed: float,
) -> dict:
    def stats(vals: List[float]) -> dict:
        if not vals:
            return {}
        a = np.array(vals)
        return {
            "n": len(vals),
            "mean": round(float(a.mean()), 4),
            "std":  round(float(a.std()),  4),
            "min":  round(float(a.min()),  4),
            "p25":  round(float(np.percentile(a, 25)), 4),
            "p50":  round(float(np.percentile(a, 50)), 4),
            "p75":  round(float(np.percentile(a, 75)), 4),
            "max":  round(float(a.max()),  4),
        }

    report: dict = {
        "elapsed_seconds": round(elapsed, 1),
        "figures": [str(f.name) for f in figures],
    }

    for split, collected in [("train", train_collected), ("test", test_collected)]:
        split_data: dict = {}
        samples = all_splits[split]
        for cls in CLASSES:
            split_data[cls] = {
                "total_images": sum(1 for _, l in samples if l == cls),
                "analysed": collected["sampled"].get(cls, 0),
                "width":  stats(collected["widths"][cls]),
                "height": stats(collected["heights"][cls]),
                "blur":   stats(collected["blurs"][cls]),
                "channel_mean": {
                    ch: round(float(np.mean(collected["ch_means"][cls][ch])), 4)
                    if collected["ch_means"][cls][ch] else None
                    for ch in ["R", "G", "B"]
                },
                "channel_std": {
                    ch: round(float(np.std(collected["ch_means"][cls][ch])), 4)
                    if collected["ch_means"][cls][ch] else None
                    for ch in ["R", "G", "B"]
                },
            }
        report[split] = split_data

    return report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    t0 = time.perf_counter()

    print("=" * 60)
    print("  EDA - Face Mask Detection Dataset")
    print("=" * 60)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 0. Load file lists ──────────────────────────────────────────
    print("\n[0] Loading file lists ...")
    all_splits: Dict[str, List[Tuple[Path, str]]] = {}
    for split in SPLITS:
        samples = list_samples(split)
        all_splits[split] = samples
        for cls in CLASSES:
            n = sum(1 for _, l in samples if l == cls)
            print(f"    {split}/{cls}: {n:,} images")

    # ── 1. Class distribution ───────────────────────────────────────
    print("\n[1] Class distribution ...")
    f1 = fig_class_distribution(all_splits)

    # ── 2. Sample grid ──────────────────────────────────────────────
    print("\n[2] Sample image grid ...")
    f2 = fig_sample_grid(all_splits)

    # ── 3–5. Pixel-level stats (train) ─────────────────────────────
    print("\n[3-5] Collecting pixel stats for train split ...")
    train_collected = collect_all(all_splits["train"])

    print("\n[3-5] Collecting pixel stats for test split ...")
    test_collected = collect_all(all_splits["test"])

    # ── 3. Resolution ───────────────────────────────────────────────
    print("\n[3] Resolution histograms ...")
    # Merge train + test for the resolution figure (train dominates visually)
    merged_res = {
        key: {
            cls: train_collected[key][cls] + test_collected[key][cls]
            for cls in CLASSES
        }
        for key in ("widths", "heights")
    }
    # re-use the same structure collect_all returns
    merged_for_res = {**train_collected, "widths": merged_res["widths"], "heights": merged_res["heights"]}
    f3 = fig_resolution(merged_for_res)

    # ── 4. Channel stats ────────────────────────────────────────────
    print("\n[4] Channel statistics (train) ...")
    f4 = fig_channel_stats(train_collected)

    # ── 5. Blur distribution ────────────────────────────────────────
    print("\n[5] Blur score distribution (train) ...")
    f5 = fig_blur(train_collected)

    # ── JSON report ─────────────────────────────────────────────────
    elapsed = time.perf_counter() - t0
    figures = [f1, f2, f3, f4, f5]
    report = build_json_report(all_splits, train_collected, test_collected, figures, elapsed)

    report_path = REPORT_DIR / "step_eda_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    print("  EDA complete")
    print(f"  Elapsed : {elapsed:.1f}s")
    print(f"  Report  : {report_path}")
    for f in figures:
        print(f"  Figure  : {f}")
    print("=" * 60)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
