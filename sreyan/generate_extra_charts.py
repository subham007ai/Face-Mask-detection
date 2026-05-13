"""
Extra Charts Generator — Face Mask Detection Project
=====================================================
Generates 4 additional publication-quality charts from the real
evaluation metrics stored in all_models_evaluation_metrics.json.

Charts produced:
  1. grouped_bar_chart.png      — side-by-side metric comparison
  2. radar_chart.png            — spider/radar plot per model
  3. metrics_heatmap.png        — colour-coded metric vs model matrix
  4. error_analysis.png         — misclassification count per model

Run:
    python sreyan/generate_extra_charts.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

SREYAN_DIR  = Path(__file__).parent.resolve()
METRICS_FILE = SREYAN_DIR / "all_models_evaluation_metrics.json"

with open(METRICS_FILE) as f:
    data = json.load(f)

models_data = data["models"]
model_names = list(models_data.keys())
METRIC_KEYS  = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
METRIC_LABELS = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]

values = np.array([
    [models_data[m][k] for k in METRIC_KEYS]
    for m in model_names
])

PALETTE    = ["#1565C0", "#E65100", "#2E7D32"]
BG_COLOR   = "#ffffff"
GRID_COLOR = "#dddddd"
TEXT_COLOR = "#1a1a1a"

plt.rcParams.update({
    "figure.facecolor": BG_COLOR,
    "axes.facecolor":   BG_COLOR,
    "axes.edgecolor":   "#aaaaaa",
    "axes.labelcolor":  TEXT_COLOR,
    "xtick.color":      TEXT_COLOR,
    "ytick.color":      TEXT_COLOR,
    "text.color":       TEXT_COLOR,
    "grid.color":       GRID_COLOR,
    "font.family":      "DejaVu Sans",
})

TEST_SIZE = 1726

def plot_grouped_bar():
    fig, ax = plt.subplots(figsize=(13, 7), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    n_models  = len(model_names)
    n_metrics = len(METRIC_KEYS)
    x         = np.arange(n_metrics)
    width     = 0.22
    offsets   = np.linspace(-(n_models - 1) / 2, (n_models - 1) / 2, n_models) * width

    for i, (model, color) in enumerate(zip(model_names, PALETTE)):
        bars = ax.bar(
            x + offsets[i], values[i] * 100,
            width=width - 0.02,
            color=color, alpha=0.88,
            label=model, zorder=3,
        )
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2, h + 0.08,
                f"{h:.2f}%", ha="center", va="bottom",
                fontsize=7.5, color=TEXT_COLOR, fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(METRIC_LABELS, fontsize=12)
    ax.set_ylim(99.0, 101.5)
    ax.set_ylabel("Score (%)", fontsize=12)
    ax.set_title(
        "Model Performance Comparison — All Metrics",
        fontsize=14, fontweight="bold", pad=15, color=TEXT_COLOR,
    )
    ax.legend(fontsize=11, framealpha=0.2, edgecolor=GRID_COLOR)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)

    plt.tight_layout()
    out = SREYAN_DIR / "grouped_bar_chart.png"
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()
    print(f"[*] Saved: {out}")


def plot_radar():
    angles = np.linspace(0, 2 * np.pi, len(METRIC_KEYS), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(
        figsize=(8, 8), subplot_kw={"polar": True}, facecolor=BG_COLOR,
    )
    ax.set_facecolor(BG_COLOR)
    ax.spines["polar"].set_color(GRID_COLOR)

    for i, (model, color) in enumerate(zip(model_names, PALETTE)):
        v = values[i].tolist() + [values[i][0]]
        ax.plot(angles, v, color=color, linewidth=2, label=model)
        ax.fill(angles, v, color=color, alpha=0.12)

    ax.set_thetagrids(np.degrees(angles[:-1]), METRIC_LABELS, fontsize=11)
    ax.set_ylim(0.98, 1.005)
    ax.set_yticks([0.985, 0.990, 0.995, 1.000])
    ax.set_yticklabels(["98.5%", "99.0%", "99.5%", "100%"], fontsize=8)
    ax.yaxis.grid(True, linestyle="--", alpha=0.35)
    ax.xaxis.grid(True, linestyle="--", alpha=0.35)

    ax.set_title(
        "Radar Chart — Multi-Model Metric Comparison",
        fontsize=14, fontweight="bold", pad=20, color=TEXT_COLOR,
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1),
              fontsize=11, framealpha=0.2, edgecolor=GRID_COLOR)

    plt.tight_layout()
    out = SREYAN_DIR / "radar_chart.png"
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()
    print(f"[*] Saved: {out}")


def plot_heatmap():
    fig, ax = plt.subplots(figsize=(10, 4.5), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    vmin, vmax = 0.98, 1.001
    im = ax.imshow(values, cmap="YlGn", aspect="auto", vmin=vmin, vmax=vmax)

    ax.set_xticks(range(len(METRIC_LABELS)))
    ax.set_xticklabels(METRIC_LABELS, fontsize=12)
    ax.set_yticks(range(len(model_names)))
    ax.set_yticklabels(model_names, fontsize=12)

    for row in range(len(model_names)):
        for col in range(len(METRIC_KEYS)):
            val = values[row, col]
            ax.text(
                col, row, f"{val * 100:.3f}%",
                ha="center", va="center",
                fontsize=11, fontweight="bold",
                color="black" if val > 0.995 else "white",
            )

    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.ax.yaxis.set_tick_params(color=TEXT_COLOR)
    cbar.outline.set_edgecolor(GRID_COLOR)

    ax.set_title(
        "Metrics Heatmap — Model × Metric",
        fontsize=14, fontweight="bold", pad=12, color=TEXT_COLOR,
    )

    plt.tight_layout()
    out = SREYAN_DIR / "metrics_heatmap.png"
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()
    print(f"[*] Saved: {out}")


def plot_error_analysis():
    errors = [round((1.0 - models_data[m]["accuracy"]) * TEST_SIZE)
              for m in model_names]
    correct = [TEST_SIZE - e for e in errors]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=BG_COLOR)

    ax = axes[0]
    ax.set_facecolor(BG_COLOR)
    x = np.arange(len(model_names))

    bars_ok  = ax.bar(x, correct, color="#4CAF50", alpha=0.85, label="Correct", zorder=3)
    bars_err = ax.bar(x, errors,  bottom=correct,
                      color="#f44336", alpha=0.85, label="Errors", zorder=3)

    for bar, e, c in zip(bars_ok, errors, correct):
        ax.text(bar.get_x() + bar.get_width() / 2, c / 2,
                f"{c}", ha="center", va="center",
                fontsize=12, fontweight="bold", color="white")
    for bar, e, c in zip(bars_err, errors, correct):
        ax.text(bar.get_x() + bar.get_width() / 2, c + e / 2 + 1,
                f"{e} err", ha="center", va="center",
                fontsize=11, fontweight="bold", color="white")

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=11)
    ax.set_ylabel("Image Count", fontsize=11)
    ax.set_title(
        f"Correct vs Misclassified\n(Total test images: {TEST_SIZE:,})",
        fontsize=13, fontweight="bold", color=TEXT_COLOR,
    )
    ax.legend(fontsize=11, framealpha=0.2)
    ax.yaxis.grid(True, linestyle="--", alpha=0.3, zorder=0)

    ax2 = axes[1]
    ax2.set_facecolor(BG_COLOR)

    bars_h = ax2.barh(model_names, errors, color=PALETTE, alpha=0.85, zorder=3)
    for bar, e in zip(bars_h, errors):
        ax2.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                 f"  {e} images", va="center", fontsize=12, color=TEXT_COLOR)

    ax2.set_xlabel("Number of Misclassified Images", fontsize=11)
    ax2.set_title(
        "Misclassification Count (Zoomed)",
        fontsize=13, fontweight="bold", color=TEXT_COLOR,
    )
    ax2.xaxis.grid(True, linestyle="--", alpha=0.3, zorder=0)
    ax2.set_xlim(0, max(errors) + 12 if max(errors) > 0 else 5)

    plt.suptitle(
        "Error Analysis — Face Mask Detection Models",
        fontsize=15, fontweight="bold", color=TEXT_COLOR, y=1.02,
    )
    plt.tight_layout()
    out = SREYAN_DIR / "error_analysis.png"
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()
    print(f"[*] Saved: {out}")


if __name__ == "__main__":
    print("=" * 60)
    print("[*] GENERATING EXTRA CHARTS FROM REAL METRICS")
    print("=" * 60)
    plot_grouped_bar()
    plot_radar()
    plot_heatmap()
    plot_error_analysis()
    print("=" * 60)
    print("[DONE] All 4 charts saved to sreyan/")
    print("=" * 60)
