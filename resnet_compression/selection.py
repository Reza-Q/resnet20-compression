"""Select the best pruned model (across scenarios A and B) to carry into QAT."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import Config
from .models import ResNet, build_model_from_state_dict
from .utils import load_json, load_torch_file, rate_tag, save_json

logger = logging.getLogger(__name__)


@dataclass
class BestModelInfo:
    """Which scenario/rate produced the model carried forward into QAT."""

    scenario: str  # "A" or "B"
    rate: float
    accuracy: float
    compression: float
    params: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "rate": self.rate,
            "accuracy": self.accuracy,
            "compression": self.compression,
            "params": self.params,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BestModelInfo:
        return cls(
            scenario=data["scenario"],
            rate=data["rate"],
            accuracy=data["accuracy"],
            compression=data["compression"],
            params=data["params"],
        )


def _best_model_path(cfg: Config) -> Path:
    return cfg.base_dir / "best_model_for_qat.json"


def select_best_model(
    cfg: Config,
    results_a: dict[float, dict[str, Any]],
    results_b: dict[float, dict[str, Any]],
    baseline_acc: float,
) -> BestModelInfo:
    """Pick the highest-compression model whose accuracy drop is acceptable.

    Ties and the case where no model meets ``cfg.acceptable_acc_drop`` are handled the
    same way as the original notebook: among rates that qualify, keep the highest
    compression ratio (scenario A is only preferred over B because it is checked
    first in the loop, i.e. on an exact tie A wins); otherwise fall back to the rate
    with the smallest accuracy drop across both scenarios.

    Resumable: reuses ``best_model_for_qat.json`` when it already exists and
    ``cfg.force`` is ``False``.
    """
    path = _best_model_path(cfg)
    if path.exists() and not cfg.force:
        logger.info("Loading existing best-model selection from %s", path.name)
        return BestModelInfo.from_dict(load_json(path))

    best_scenario = None
    best_rate = None
    best_compression = 0.0

    for rate in sorted(cfg.pruning_rates):
        for scenario, results in (("A", results_a), ("B", results_b)):
            entry = results[rate]
            if (
                baseline_acc - entry["final_accuracy"] <= cfg.acceptable_acc_drop
                and entry["compression_ratio"] > best_compression
            ):
                best_compression = entry["compression_ratio"]
                best_scenario = scenario
                best_rate = rate

    if best_scenario is None:
        logger.warning("No model met the %.1f%% accuracy-drop threshold; picking the smallest drop", cfg.acceptable_acc_drop)
        candidates = [
            ("A", rate, results_a[rate]["accuracy_drop"]) for rate in cfg.pruning_rates
        ] + [("B", rate, results_b[rate]["accuracy_drop"]) for rate in cfg.pruning_rates]
        best_scenario, best_rate, _ = min(candidates, key=lambda item: item[2])
        best_compression = (results_a if best_scenario == "A" else results_b)[best_rate]["compression_ratio"]

    results = results_a if best_scenario == "A" else results_b
    info = BestModelInfo(
        scenario=best_scenario,
        rate=best_rate,
        accuracy=results[best_rate]["final_accuracy"],
        compression=results[best_rate]["compression_ratio"],
        params=results[best_rate]["final_params"],
    )
    logger.info(
        "Best model for QAT: scenario %s, rate %.0f%% (accuracy=%.2f%%, compression=%.2fx)",
        info.scenario,
        info.rate * 100,
        info.accuracy,
        info.compression,
    )
    save_json(info.to_dict(), path)
    return info


def load_pruned_model_for_qat(cfg: Config, info: BestModelInfo) -> ResNet:
    """Load the pruned model chosen by :func:`select_best_model` from disk.

    The model's (possibly non-uniform) per-block widths are recovered directly from the
    checkpoint's tensor shapes via :func:`resnet_compression.models.infer_block_channels`,
    so no separate record of the pruned architecture needs to be kept alongside it.
    """
    model_path = cfg.scenario_dir(info.scenario) / f"final_model_{rate_tag(info.rate)}_rate.pth"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Expected pruned model at {model_path}, but it does not exist. "
            "Run the pruning scenarios before selecting a model for QAT."
        )
    state_dict = load_torch_file(model_path, map_location="cpu")
    model = build_model_from_state_dict(state_dict)
    logger.info("Loaded pruned model for QAT from %s", model_path.name)
    return model
