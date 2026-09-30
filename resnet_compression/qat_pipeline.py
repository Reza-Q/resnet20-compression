"""Quantization-aware training scenarios ("C": with KD, "D": without KD).

Each scenario takes the pruned model selected by
:func:`resnet_compression.selection.select_best_model`, then runs QAT twice — once with
an INT8 qconfig, once with the custom INT4 qconfig — fine-tuning and converting each to a
genuinely quantized model. See :mod:`resnet_compression.quantization` for the bug this
fixes relative to the original notebook (INT8 and INT4 preparation are now independent).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import Config
from .models import ResNet
from .quantization import QATResult, convert_to_quantized, prepare_qat_model, train_qat
from .training import evaluate
from .utils import (
    count_quantized_parameters,
    get_quantized_model_size_mb,
    load_json,
    save_json,
)

logger = logging.getLogger(__name__)

BIT_WIDTHS = ("int8", "int4")


def _run_one_bit_width(
    cfg: Config,
    scenario: str,
    bit_width: str,
    pruned_model: ResNet,
    baseline_acc: float,
    baseline_size_mb: float,
    use_kd: bool,
    teachers: Sequence[nn.Module],
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
) -> QATResult:
    backend = cfg.qat_backend
    qat_model = prepare_qat_model(pruned_model, qconfig_type=bit_width, backend=backend)

    acc_before = evaluate(qat_model.to(device), test_loader, device)
    logger.info("[%s %s] accuracy before QAT fine-tuning: %.2f%%", scenario, bit_width, acc_before)

    qat_model, _, history = train_qat(
        qat_model,
        train_loader,
        test_loader,
        epochs=cfg.qat_epochs,
        device=device,
        lr=cfg.qat_lr,
        teachers=teachers if use_kd else (),
        temperature=cfg.kd_temperature,
        alpha=cfg.kd_alpha,
        desc=f"{scenario}-{bit_width.upper()}",
    )

    quantized = convert_to_quantized(qat_model)
    device_cpu = torch.device("cpu")
    final_acc = evaluate(quantized, test_loader, device_cpu)
    params = count_quantized_parameters(quantized)
    size_mb = get_quantized_model_size_mb(quantized)

    result = QATResult(
        bit_width=bit_width,
        accuracy=final_acc,
        parameters=params,
        size_mb=size_mb,
        compression_ratio=baseline_size_mb / size_mb,
        accuracy_drop=baseline_acc - final_acc,
        history=history,
    )
    logger.info(
        "[%s %s] accuracy=%.2f%% (drop=%.2f%%), size=%.3f MB, compression=%.2fx",
        scenario,
        bit_width,
        result.accuracy,
        result.accuracy_drop,
        result.size_mb,
        result.compression_ratio,
    )

    suffix = "with_kd" if use_kd else "no_kd"
    model_path = cfg.scenario_dir(scenario) / f"best_model_{bit_width}_{suffix}.pth"
    torch.save(quantized.state_dict(), model_path)
    return result


def run_qat_scenario(
    cfg: Config,
    scenario: str,
    use_kd: bool,
    pruned_model: ResNet,
    base_info: dict[str, Any],
    baseline_acc: float,
    baseline_size_mb: float,
    teachers: Sequence[nn.Module],
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    """Run INT8 then INT4 QAT for one scenario ("C" or "D").

    Resumable like :func:`resnet_compression.pruning_pipeline.run_pruning_scenario`.
    Fake-quantized fine-tuning runs on ``device`` like the earlier stages, but every
    *converted*, genuinely quantized model always runs on the CPU: PyTorch's quantized
    kernels (fbgemm/qnnpack/onednn) are CPU-only, so :func:`convert_to_quantized` and the
    accuracy measured right after it never use ``device``.
    """
    results_path = cfg.results_path(scenario)
    if results_path.exists() and not cfg.force:
        logger.info("Scenario %s: loading existing results from %s", scenario, results_path.name)
        return load_json(results_path)

    logger.info("Scenario %s (%s): running INT8 then INT4 QAT", scenario, "with KD" if use_kd else "without KD")
    results: dict[str, Any] = {}
    for bit_width in BIT_WIDTHS:
        outcome = _run_one_bit_width(
            cfg,
            scenario,
            bit_width,
            pruned_model,
            baseline_acc,
            baseline_size_mb,
            use_kd,
            teachers,
            train_loader,
            test_loader,
            device,
        )
        results[bit_width] = outcome.to_dict()

    results["base_info"] = base_info
    save_json(results, results_path)
    return results
