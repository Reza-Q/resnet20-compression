"""Result plots: pruning comparison, Pareto fronts and QAT comparison.

Each function saves one PNG to ``cfg.plots_dir`` and returns its path. Matplotlib's
non-interactive ``Agg`` backend is used throughout, so these also run headless (CI, a
plain terminal, a Docker container with no display).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from .config import Config

logger = logging.getLogger(__name__)

DPI = 300


def _savefig(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved plot: %s", path.name)
    return path


def plot_pruning_comparison(
    cfg: Config,
    results_a: dict[float, dict[str, Any]],
    results_b: dict[float, dict[str, Any]],
    baseline_acc: float,
    baseline_size_mb: float,
) -> Path:
    """4-panel figure: accuracy, compression ratio, model size, and the KD benefit."""
    rates = sorted(cfg.pruning_rates)
    rate_pcts = [r * 100 for r in rates]

    a_accs = [results_a[r]["final_accuracy"] for r in rates]
    b_accs = [results_b[r]["final_accuracy"] for r in rates]
    a_comps = [results_a[r]["compression_ratio"] for r in rates]
    b_comps = [results_b[r]["compression_ratio"] for r in rates]
    a_sizes = [results_a[r]["final_size_mb"] for r in rates]
    b_sizes = [results_b[r]["final_size_mb"] for r in rates]

    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(rate_pcts, a_accs, "o-", label="A: WITH KD", linewidth=2.5, markersize=8, color="blue")
    ax1.plot(rate_pcts, b_accs, "s-", label="B: WITHOUT KD", linewidth=2.5, markersize=8, color="red")
    ax1.axhline(y=baseline_acc, color="green", linestyle="--", label="Baseline", linewidth=2)
    ax1.axhline(
        y=baseline_acc - cfg.acceptable_acc_drop,
        color="orange",
        linestyle="--",
        label=f"{cfg.acceptable_acc_drop:.1f}% Drop",
        linewidth=1.5,
    )
    ax1.set_xlabel("Target Pruning Rate (%)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Accuracy (%)", fontsize=12, fontweight="bold")
    ax1.set_title("Accuracy vs Pruning Rate", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(rate_pcts)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(rate_pcts, a_comps, "o-", label="A: WITH KD", linewidth=2.5, markersize=8, color="blue")
    ax2.plot(rate_pcts, b_comps, "s-", label="B: WITHOUT KD", linewidth=2.5, markersize=8, color="red")
    ax2.set_xlabel("Target Pruning Rate (%)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Compression Ratio", fontsize=12, fontweight="bold")
    ax2.set_title("Compression Ratio vs Pruning Rate", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(rate_pcts)

    ax3 = fig.add_subplot(gs[1, 0])
    width = 0.35
    x = np.arange(len(rates))
    ax3.bar(x - width / 2, a_sizes, width, label="A: WITH KD", alpha=0.8, color="skyblue")
    ax3.bar(x + width / 2, b_sizes, width, label="B: WITHOUT KD", alpha=0.8, color="lightcoral")
    ax3.axhline(y=baseline_size_mb, color="green", linestyle="--", label="Baseline", linewidth=2)
    ax3.set_xlabel("Target Pruning Rate (%)", fontsize=12, fontweight="bold")
    ax3.set_ylabel("Model Size (MB)", fontsize=12, fontweight="bold")
    ax3.set_title("Model Size Comparison", fontsize=14, fontweight="bold")
    ax3.set_xticks(x)
    ax3.set_xticklabels([f"{int(r)}%" for r in rate_pcts])
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3, axis="y")

    ax4 = fig.add_subplot(gs[1, 1])
    kd_benefits = [a_accs[i] - b_accs[i] for i in range(len(rates))]
    colors = ["green" if b > 0 else "red" for b in kd_benefits]
    bars = ax4.bar(rate_pcts, kd_benefits, color=colors, alpha=0.7, edgecolor="black", linewidth=1.5)
    ax4.axhline(y=0, color="black", linestyle="-", linewidth=1)
    ax4.set_xlabel("Target Pruning Rate (%)", fontsize=12, fontweight="bold")
    ax4.set_ylabel("KD Benefit (% Accuracy Improvement)", fontsize=12, fontweight="bold")
    ax4.set_title("Knowledge Distillation Benefit", fontsize=14, fontweight="bold")
    ax4.grid(True, alpha=0.3, axis="y")
    ax4.set_xticks(rate_pcts)
    for bar, benefit in zip(bars, kd_benefits):
        height = bar.get_height()
        ax4.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{benefit:.2f}%",
            ha="center",
            va="bottom" if benefit > 0 else "top",
            fontsize=9,
            fontweight="bold",
        )

    return _savefig(fig, cfg.plots_dir / "plot1_pruning_comparison.png")


def plot_pareto_fronts(
    cfg: Config,
    results_a: dict[float, dict[str, Any]],
    results_b: dict[float, dict[str, Any]],
    results_c: dict[str, Any],
    baseline_acc: float,
) -> Path:
    """Two-panel figure: pruning Pareto front, and the full pipeline's Pareto path."""
    rates = sorted(cfg.pruning_rates)
    rate_pcts = [r * 100 for r in rates]
    a_accs = [results_a[r]["final_accuracy"] for r in rates]
    b_accs = [results_b[r]["final_accuracy"] for r in rates]
    a_comps = [results_a[r]["compression_ratio"] for r in rates]
    b_comps = [results_b[r]["compression_ratio"] for r in rates]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax1 = axes[0]
    ax1.scatter(
        a_comps, a_accs, s=200, marker="o", c=rate_pcts, cmap="viridis",
        edgecolors="blue", linewidth=2, label="A: WITH KD", alpha=0.8,
    )
    ax1.scatter(
        b_comps, b_accs, s=200, marker="s", c=rate_pcts, cmap="plasma",
        edgecolors="red", linewidth=2, label="B: WITHOUT KD", alpha=0.8,
    )
    for i, rate in enumerate(rates):
        ax1.annotate(
            f"{int(rate * 100)}%", (a_comps[i], a_accs[i]),
            textcoords="offset points", xytext=(0, 10), ha="center", fontsize=8,
        )
        ax1.annotate(
            f"{int(rate * 100)}%", (b_comps[i], b_accs[i]),
            textcoords="offset points", xytext=(0, -15), ha="center", fontsize=8,
        )
    ax1.axhline(
        y=baseline_acc - cfg.acceptable_acc_drop, color="orange",
        linestyle="--", linewidth=2, alpha=0.7, label="Acceptable Drop",
    )
    ax1.set_xlabel("Compression Ratio", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Accuracy (%)", fontsize=13, fontweight="bold")
    ax1.set_title("Pareto Front: Pruning Scenarios", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    best_rate = results_c["base_info"]["rate"]
    best_results = results_a if results_c["base_info"]["scenario"] == "A" else results_b
    best_pruned_acc = best_results[best_rate]["final_accuracy"]
    best_pruned_comp = best_results[best_rate]["compression_ratio"]

    stages_acc = [baseline_acc, best_pruned_acc, results_c["int8"]["accuracy"], results_c["int4"]["accuracy"]]
    stages_comp = [1.0, best_pruned_comp, results_c["int8"]["compression_ratio"], results_c["int4"]["compression_ratio"]]
    stages_labels = ["Baseline", "Pruned\n(Best)", "INT8+KD", "INT4+KD"]
    stages_colors = ["green", "blue", "orange", "red"]

    ax2.plot(stages_comp, stages_acc, "o-", linewidth=3, markersize=12, color="purple", alpha=0.5, label="Pipeline WITH KD")
    for comp, acc, label, color in zip(stages_comp, stages_acc, stages_labels, stages_colors):
        ax2.scatter([comp], [acc], s=250, marker="o", color=color, edgecolors="black", linewidth=2, zorder=5)
        ax2.annotate(
            f"{label}\n{acc:.1f}%\n{comp:.1f}x", (comp, acc),
            textcoords="offset points", xytext=(0, 20), ha="center", fontsize=9, fontweight="bold",
            bbox={"boxstyle": "round,pad=0.5", "facecolor": color, "alpha": 0.3},
        )
    ax2.axhline(
        y=baseline_acc - cfg.acceptable_acc_drop, color="orange",
        linestyle="--", linewidth=2, alpha=0.7, label="Acceptable Drop",
    )
    ax2.set_xlabel("Compression Ratio", fontsize=13, fontweight="bold")
    ax2.set_ylabel("Accuracy (%)", fontsize=13, fontweight="bold")
    ax2.set_title("Complete Compression Pipeline", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    return _savefig(fig, cfg.plots_dir / "plot2_pareto_fronts.png")


def _qat_bar_panel(ax, categories: Sequence[str], accs: Sequence[float], comps: Sequence[float], title: str) -> None:
    x = np.arange(len(categories))
    width = 0.35
    ax_twin = ax.twinx()

    bars1 = ax.bar(x - width / 2, accs, width, label="Accuracy (%)", alpha=0.8, color="skyblue", edgecolor="black", linewidth=1.5)
    bars2 = ax_twin.bar(x + width / 2, comps, width, label="Compression", alpha=0.8, color="lightcoral", edgecolor="black", linewidth=1.5)

    ax.set_ylabel("Accuracy (%)", fontsize=12, fontweight="bold")
    ax_twin.set_ylabel("Compression Ratio", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend(loc="upper left", fontsize=10)
    ax_twin.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    for bar, acc in zip(bars1, accs):
        ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), f"{acc:.2f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
    for bar, comp in zip(bars2, comps):
        ax_twin.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), f"{comp:.2f}x", ha="center", va="bottom", fontsize=10, fontweight="bold")


def plot_qat_comparison(cfg: Config, results_c: dict[str, Any], results_d: dict[str, Any]) -> Path:
    """Two-panel figure comparing INT8 and INT4 QAT, each with vs. without KD."""
    categories = ["C: WITH KD", "D: WITHOUT KD"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    _qat_bar_panel(
        axes[0], categories,
        [results_c["int8"]["accuracy"], results_d["int8"]["accuracy"]],
        [results_c["int8"]["compression_ratio"], results_d["int8"]["compression_ratio"]],
        "INT8 Quantization: KD vs No-KD",
    )
    _qat_bar_panel(
        axes[1], categories,
        [results_c["int4"]["accuracy"], results_d["int4"]["accuracy"]],
        [results_c["int4"]["compression_ratio"], results_d["int4"]["compression_ratio"]],
        "INT4 Quantization: KD vs No-KD",
    )

    fig.tight_layout()
    return _savefig(fig, cfg.plots_dir / "plot3_qat_comparison.png")
