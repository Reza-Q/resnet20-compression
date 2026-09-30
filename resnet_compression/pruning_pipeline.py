"""Iterative pruning scenarios ("A": with KD, "B": without KD).

Both scenarios repeat the same loop for every configured pruning rate: prune a
percentage of the *current* model's filters, fine-tune, and stop on an accuracy-drop
threshold, a maximum iteration count, or repeated no-op pruning. The only difference
between scenarios is whether fine-tuning distills from the teacher ensemble.

The two scenarios are implemented as a single parameterized function so that each
scenario only ever writes to its own :meth:`Config.scenario_dir`. In the original
notebook, scenario B's intermediate checkpoints were accidentally written into
scenario A's directory (a copy-pasted path constant); see ``NOTES.md`` for details.
Because there is only one code path here, that class of bug cannot recur.
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Sequence
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import Config
from .pruning import prune_filters
from .training import evaluate, train_model, train_with_kd
from .utils import (
    count_conv_filters,
    count_parameters,
    get_model_size_mb,
    load_json,
    rate_tag,
    save_checkpoint,
    save_json,
)

logger = logging.getLogger(__name__)

#: Consecutive low-yield iterations that stop pruning at a given rate.
MAX_CONSECUTIVE_STALLS = 3
#: An iteration that prunes less than this fraction of the current model is a "stall".
MIN_MEANINGFUL_PRUNE_FRACTION = 0.005
#: Hard stop: never let cumulative pruning exceed this fraction of the original params.
MAX_CUMULATIVE_PRUNE_FRACTION = 0.95


def _stop_reason(consecutive_stalls: int, last_drop: float, cumulative_pruned: float, cfg: Config) -> str:
    if consecutive_stalls >= MAX_CONSECUTIVE_STALLS:
        return "no_pruning"
    if last_drop > cfg.acceptable_acc_drop:
        return "accuracy_drop"
    if cumulative_pruned > MAX_CUMULATIVE_PRUNE_FRACTION:
        return "extreme_pruning"
    return "max_iterations"


def _prune_at_rate(
    cfg: Config,
    scenario: str,
    rate: float,
    baseline_model: nn.Module,
    baseline_acc: float,
    baseline_params: int,
    use_kd: bool,
    teachers: Sequence[nn.Module],
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    """Run the iterative prune + fine-tune loop for a single target rate."""
    current_model = copy.deepcopy(baseline_model)
    history: list[dict[str, Any]] = []
    saved_checkpoints: dict[str, Any] = {}
    next_checkpoint = cfg.checkpoint_interval
    cumulative_pruned = 0.0
    consecutive_stalls = 0
    iteration = 0

    while iteration < cfg.prune_iterations:
        iteration += 1
        params_before = count_parameters(current_model)

        pruned_model, num_pruned = prune_filters(current_model, rate)
        pruned_model = pruned_model.to(device)

        params_after = count_parameters(pruned_model)
        actual_pruned_this_iter = 1.0 - (params_after / params_before)
        cumulative_remaining = params_after / baseline_params
        cumulative_pruned = 1.0 - cumulative_remaining

        if num_pruned == 0 or actual_pruned_this_iter < MIN_MEANINGFUL_PRUNE_FRACTION:
            consecutive_stalls += 1
        else:
            consecutive_stalls = 0

        if consecutive_stalls >= MAX_CONSECUTIVE_STALLS:
            logger.info(
                "[%s %.0f%%] stopping: %d consecutive iterations with no meaningful pruning",
                scenario,
                rate * 100,
                consecutive_stalls,
            )
            break

        if use_kd:
            pruned_model, acc_after, _ = train_with_kd(
                pruned_model,
                teachers,
                train_loader,
                test_loader,
                epochs=cfg.finetune_epochs,
                device=device,
                lr=0.001,
                temperature=cfg.kd_temperature,
                alpha=cfg.kd_alpha,
                desc=f"{scenario}-{rate * 100:.1f}%-iter{iteration}",
            )
        else:
            pruned_model, acc_after, _ = train_model(
                pruned_model,
                train_loader,
                test_loader,
                epochs=cfg.finetune_epochs,
                lr=0.001,
                device=device,
                model_name=f"{scenario}-{rate * 100:.1f}%-iter{iteration}",
            )
        pruned_model = pruned_model.to(device)
        acc_drop = baseline_acc - acc_after

        history.append(
            {
                "iteration": iteration,
                "accuracy": acc_after,
                "accuracy_drop": acc_drop,
                "params": params_after,
                "cumulative_pruned_pct": cumulative_pruned * 100,
                "cumulative_remaining_pct": cumulative_remaining * 100,
                "filters_pruned_this_iter": num_pruned,
                "actual_pruned_pct_this_iter": actual_pruned_this_iter * 100,
            }
        )
        current_model = pruned_model
        logger.info(
            "[%s %.0f%%] iter %d: acc=%.2f%% (drop=%.2f%%), params=%s (%.1f%% pruned)",
            scenario,
            rate * 100,
            iteration,
            acc_after,
            acc_drop,
            f"{params_after:,}",
            cumulative_pruned * 100,
        )

        if cumulative_pruned >= next_checkpoint:
            checkpoint_key = f"{round(next_checkpoint * 100)}pct"
            checkpoint_path = cfg.scenario_dir(scenario) / (
                f"checkpoint_{rate_tag(rate)}_rate_{checkpoint_key}_pruned.pth"
            )
            save_checkpoint(current_model.cpu().state_dict(), checkpoint_path)
            saved_checkpoints[checkpoint_key] = {
                "iteration": iteration,
                "accuracy": acc_after,
                "cumulative_pruned": cumulative_pruned * 100,
                "params": params_after,
            }
            current_model = current_model.to(device)
            next_checkpoint += cfg.checkpoint_interval

        if acc_drop > cfg.acceptable_acc_drop or cumulative_pruned > MAX_CUMULATIVE_PRUNE_FRACTION:
            break

    current_model = current_model.to(device)
    final_acc = evaluate(current_model, test_loader, device)
    current_model = current_model.cpu()

    final_params = count_parameters(current_model)
    final_size = get_model_size_mb(current_model)
    baseline_size = get_model_size_mb(baseline_model.cpu())
    final_param_reduction = 1.0 - (final_params / baseline_params)
    last_drop = history[-1]["accuracy_drop"] if history else 0.0

    result = {
        "target_prune_rate_per_iter": rate,
        "iterations_completed": len(history),
        "stop_reason": _stop_reason(consecutive_stalls, last_drop, cumulative_pruned, cfg),
        "final_accuracy": final_acc,
        "final_params": final_params,
        "final_filters": count_conv_filters(current_model),
        "final_size_mb": final_size,
        "compression_ratio": baseline_size / final_size,
        "param_reduction_pct": final_param_reduction * 100,
        "params_remaining_pct": (1 - final_param_reduction) * 100,
        "accuracy_drop": baseline_acc - final_acc,
        "pruning_history": history,
        "saved_checkpoints": saved_checkpoints,
    }

    model_path = cfg.scenario_dir(scenario) / f"final_model_{rate_tag(rate)}_rate.pth"
    save_checkpoint(current_model.state_dict(), model_path)
    return result


def run_pruning_scenario(
    cfg: Config,
    scenario: str,
    use_kd: bool,
    baseline_model: nn.Module,
    baseline_acc: float,
    baseline_params: int,
    teachers: Sequence[nn.Module],
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
) -> dict[float, dict[str, Any]]:
    """Run the iterative pruning scenario for every rate in ``cfg.pruning_rates``.

    Resumable: if ``cfg.results_path(scenario)`` already exists and ``cfg.force`` is
    ``False``, its contents are loaded and returned without retraining anything.
    """
    results_path = cfg.results_path(scenario)
    if results_path.exists() and not cfg.force:
        logger.info("Scenario %s: loading existing results from %s", scenario, results_path.name)
        return {float(k): v for k, v in load_json(results_path).items()}

    logger.info(
        "Scenario %s (%s): running %d pruning rates",
        scenario,
        "with KD" if use_kd else "without KD",
        len(cfg.pruning_rates),
    )
    results: dict[float, dict[str, Any]] = {}
    for rate in cfg.pruning_rates:
        results[rate] = _prune_at_rate(
            cfg,
            scenario,
            rate,
            baseline_model,
            baseline_acc,
            baseline_params,
            use_kd,
            teachers,
            train_loader,
            test_loader,
            device,
        )

    save_json({str(k): v for k, v in results.items()}, results_path)
    return results
