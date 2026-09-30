"""Final summary table, key insights and recommendations.

Mirrors the original notebook's closing cell: one row per model (baseline, every
pruning rate of scenarios A/B, and every bit-width of scenarios C/D), a handful of
headline numbers, and plain-language recommendations for which pipeline to use.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from .config import Config

logger = logging.getLogger(__name__)

SUMMARY_COLUMNS = [
    "Model",
    "Scenario",
    "Pruning",
    "Quantization",
    "Accuracy (%)",
    "Acc Drop (%)",
    "Params",
    "Size (MB)",
    "Compression",
]


def build_summary_table(
    cfg: Config,
    baseline_acc: float,
    baseline_params: int,
    baseline_size_mb: float,
    results_a: dict[float, dict[str, Any]],
    results_b: dict[float, dict[str, Any]],
    results_c: dict[str, Any],
    results_d: dict[str, Any],
) -> pd.DataFrame:
    """One row per model produced by the pipeline, as a :class:`pandas.DataFrame`."""
    rows: list[dict[str, str]] = [
        {
            "Model": "Baseline",
            "Scenario": "-",
            "Pruning": "-",
            "Quantization": "FP32",
            "Accuracy (%)": f"{baseline_acc:.2f}",
            "Acc Drop (%)": "0.00",
            "Params": f"{baseline_params:,}",
            "Size (MB)": f"{baseline_size_mb:.3f}",
            "Compression": "1.00x",
        }
    ]

    for rate in sorted(cfg.pruning_rates):
        for label, tag, results in (("A", "A (KD)", results_a), ("B", "B (No-KD)", results_b)):
            entry = results[rate]
            rows.append(
                {
                    "Model": f"{label}-{int(rate * 100)}%",
                    "Scenario": tag,
                    "Pruning": f"{int(rate * 100)}%",
                    "Quantization": "FP32",
                    "Accuracy (%)": f"{entry['final_accuracy']:.2f}",
                    "Acc Drop (%)": f"{entry['accuracy_drop']:.2f}",
                    "Params": f"{entry['final_params']:,}",
                    "Size (MB)": f"{entry['final_size_mb']:.3f}",
                    "Compression": f"{entry['compression_ratio']:.2f}x",
                }
            )

    best_rate = results_c["base_info"]["rate"]
    for model_label, tag, results in (("C", "C (QAT+KD)", results_c), ("D", "D (QAT-NoKD)", results_d)):
        for bit_width in ("int8", "int4"):
            entry = results[bit_width]
            rows.append(
                {
                    "Model": f"{model_label}-{bit_width.upper()}",
                    "Scenario": tag,
                    "Pruning": f"{int(best_rate * 100)}%",
                    "Quantization": bit_width.upper(),
                    "Accuracy (%)": f"{entry['accuracy']:.2f}",
                    "Acc Drop (%)": f"{entry['accuracy_drop']:.2f}",
                    "Params": f"{entry['parameters']:,}",
                    "Size (MB)": f"{entry['size_mb']:.3f}",
                    "Compression": f"{entry['compression_ratio']:.2f}x",
                }
            )

    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def compute_insights(
    cfg: Config,
    results_a: dict[float, dict[str, Any]],
    results_b: dict[float, dict[str, Any]],
    results_c: dict[str, Any],
    results_d: dict[str, Any],
) -> dict[str, float]:
    """The headline numbers used in the final report."""
    rates = cfg.pruning_rates
    avg_kd_benefit = sum(results_a[r]["final_accuracy"] - results_b[r]["final_accuracy"] for r in rates) / len(rates)
    return {
        "avg_kd_benefit_pruning": avg_kd_benefit,
        "kd_benefit_int8": results_c["int8"]["accuracy"] - results_d["int8"]["accuracy"],
        "kd_benefit_int4": results_c["int4"]["accuracy"] - results_d["int4"]["accuracy"],
        "max_compression_int8": results_c["int8"]["compression_ratio"],
        "max_compression_int4": results_c["int4"]["compression_ratio"],
    }


def format_final_report(
    cfg: Config,
    summary: pd.DataFrame,
    insights: dict[str, float],
    results_a: dict[float, dict[str, Any]],
    results_b: dict[float, dict[str, Any]],
    results_c: dict[str, Any],
    results_d: dict[str, Any],
) -> str:
    """Render the summary table, key insights and recommendations as one text report."""
    best_rate = results_c["base_info"]["rate"]
    best_scenario = results_c["base_info"]["scenario"]
    best_results = results_a if best_scenario == "A" else results_b

    lines = [
        "=" * 100,
        "COMPLETE RESULTS SUMMARY",
        "=" * 100,
        summary.to_string(index=False),
        "",
        "KEY INSIGHTS",
        "-" * 100,
        f"1. Average KD benefit while pruning: +{insights['avg_kd_benefit_pruning']:.2f}%",
        f"2. KD benefit at INT8 QAT: +{insights['kd_benefit_int8']:.2f}%",
        f"3. KD benefit at INT4 QAT: +{insights['kd_benefit_int4']:.2f}%",
        f"4. Maximum compression (INT8+KD): {insights['max_compression_int8']:.2f}x",
        f"5. Maximum compression (INT4+KD): {insights['max_compression_int4']:.2f}x",
        "",
        "BEST MODELS",
        "-" * 100,
        (f"Best pruned model (FP32): Scenario {best_scenario} ({'WITH' if best_scenario == 'A' else 'WITHOUT'} KD) at {int(best_rate * 100)}%"
        f" -- accuracy {best_results[best_rate]['final_accuracy']:.2f}%, compression {best_results[best_rate]['compression_ratio']:.2f}x"),
        (f"Best quantized model (INT8): Scenario C (QAT+KD) -- accuracy {results_c['int8']['accuracy']:.2f}%,"
        f" compression {results_c['int8']['compression_ratio']:.2f}x, drop {results_c['int8']['accuracy_drop']:.2f}%"),
        (f"Best quantized model (INT4): Scenario C (QAT+KD) -- accuracy {results_c['int4']['accuracy']:.2f}%,"
        f" compression {results_c['int4']['compression_ratio']:.2f}x, drop {results_c['int4']['accuracy_drop']:.2f}%"),
        "",
        "RECOMMENDATIONS",
        "-" * 100,
        (f"1. Maximum accuracy retention: use Scenario A (pruning with KD), ~{insights['avg_kd_benefit_pruning']:.1f}%"
        " better accuracy than without KD."),
        f"2. Maximum compression: pruning with KD -> QAT with KD -> INT4, reaching {insights['max_compression_int4']:.1f}x.",
        (f"3. Balanced: pruning with KD -> QAT with KD -> INT8, reaching {insights['max_compression_int8']:.1f}x compression"
        f" with a {results_c['int8']['accuracy_drop']:.1f}% accuracy drop."),
        "",
        "OUTPUT FILES",
        "-" * 100,
        f"{cfg.base_dir}/",
        "  checkpoints/            baseline + teacher ensemble",
        f"  {cfg.scenario_dir('A').name}/    {len(results_a)} pruned models (with KD)",
        f"  {cfg.scenario_dir('B').name}/      {len(results_b)} pruned models (without KD)",
        f"  {cfg.scenario_dir('C').name}/ 2 quantized models (INT8, INT4)",
        f"  {cfg.scenario_dir('D').name}/   2 quantized models (INT8, INT4)",
        "  plots/                  3 result figures",
        "  baseline_metrics.json",
        "  teacher_metrics.json",
        "  best_model_for_qat.json",
        "  complete_results_summary.csv",
        "=" * 100,
    ]
    return "\n".join(lines)


def save_summary_csv(cfg: Config, summary: pd.DataFrame) -> Path:
    path = cfg.base_dir / "complete_results_summary.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(path, index=False)
    logger.info("Saved: %s", path.name)
    return path
