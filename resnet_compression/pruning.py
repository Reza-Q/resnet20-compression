"""Structural filter pruning for CIFAR-style ResNets.

Filters are ranked by the L1 norm of their weights. The lowest-ranked filters of the two
3x3 convolutions in every residual block are removed and a new, physically smaller
network is built; the surviving weights are copied into it (including the matching
batch-norm statistics and the input channels of the following layers).

Each block is pruned on its own, so the residual (identity) connections of the pruned
network are *not* channel-aligned with the branch they are added to. The network is
therefore not function-preserving right after pruning and needs fine-tuning.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch import nn

from .models import STAGE_NAMES, ResNet, build_resnet_with_channels

logger = logging.getLogger(__name__)

#: A convolution is never shrunk to fewer than this many filters. If the requested
#: pruning would go below it, the convolution is left untouched.
MIN_FILTERS = 2

PruningPlan = dict[str, dict[str, Any]]

_BN_TENSORS = ("weight", "bias", "running_mean", "running_var")


def compute_filter_importance(model: nn.Module) -> dict[str, torch.Tensor]:
    """L1 norm of every filter of every ``Conv2d``, keyed by module name."""
    importance: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                importance[name] = module.weight.abs().sum(dim=(1, 2, 3))
    return importance


def _block_conv_names(model: nn.Module) -> list[str]:
    """Names of the two 3x3 convolutions of every residual block, in network order."""
    names: list[str] = []
    for stage in STAGE_NAMES:
        for block_index in range(len(getattr(model, stage))):
            names.append(f"{stage}.{block_index}.conv1")
            names.append(f"{stage}.{block_index}.conv2")
    return names


def make_pruning_plan(model: nn.Module, prune_percentage: float) -> tuple[PruningPlan, int]:
    """Decide which filters of every block convolution to keep.

    ``int(out_channels * prune_percentage)`` filters are removed from each convolution
    (rounding down), so small rates may not prune narrow layers at all.

    Returns:
        ``(plan, total_pruned)`` where ``plan[conv_name]`` holds ``keep_indices`` (sorted
        indices of the surviving filters), ``num_kept`` and ``num_pruned``, and
        ``total_pruned`` is the number of removed filters summed over all convolutions.
    """
    importance = compute_filter_importance(model)
    modules = dict(model.named_modules())

    plan: PruningPlan = {}
    total_pruned = 0
    for name in _block_conv_names(model):
        current_channels = modules[name].out_channels
        num_to_prune = int(current_channels * prune_percentage)
        num_to_keep = current_channels - num_to_prune

        if num_to_keep < MIN_FILTERS:
            num_to_keep, num_to_prune = current_channels, 0

        if num_to_prune > 0:
            _, order = torch.sort(importance[name], descending=True)
            keep_indices = order[:num_to_keep].sort()[0]
        else:
            keep_indices = torch.arange(current_channels)

        plan[name] = {
            "keep_indices": keep_indices,
            "num_kept": num_to_keep,
            "num_pruned": num_to_prune,
        }
        total_pruned += num_to_prune

    return plan, total_pruned


def _slice_conv(
    weight: torch.Tensor, keep_out: torch.Tensor, keep_in: torch.Tensor | None
) -> torch.Tensor:
    """Select output filters (and optionally input channels) of a convolution weight."""
    pruned = weight[keep_out]
    if keep_in is not None:
        pruned = pruned[:, keep_in]
    return pruned.clone()


def _copy_batch_norm(
    dst: dict[str, torch.Tensor],
    src: dict[str, torch.Tensor],
    name: str,
    keep: torch.Tensor | None = None,
) -> None:
    """Copy (and optionally channel-select) all tensors of a batch-norm layer."""
    for key in _BN_TENSORS:
        tensor = src[f"{name}.{key}"]
        dst[f"{name}.{key}"] = (tensor if keep is None else tensor[keep]).clone()
    dst[f"{name}.num_batches_tracked"] = src[f"{name}.num_batches_tracked"].clone()


def build_pruned_model(original: ResNet, plan: PruningPlan) -> ResNet:
    """Build the narrower network described by ``plan`` and copy the surviving weights.

    The width of every block is the number of filters kept in its ``conv2``; ``conv1``
    of the same block uses that width as well.

    Note:
        When the first block of ``layer1`` becomes narrower than the 16-channel stem, it
        needs a 1x1 projection shortcut that the unpruned network does not have. That
        new shortcut starts from PyTorch's default initialization.
    """
    src = original.state_dict()
    block_channels = [
        [plan[f"{stage}.{b}.conv2"]["num_kept"] for b in range(len(getattr(original, stage)))]
        for stage in STAGE_NAMES
    ]
    pruned = build_resnet_with_channels(block_channels, src["linear.weight"].shape[0])
    dst = pruned.state_dict()

    # The stem convolution is never pruned.
    dst["conv1.weight"] = src["conv1.weight"].clone()
    _copy_batch_norm(dst, src, "bn1")

    # Kept channels of the previous block's output = kept input channels of this block.
    previous_keep: torch.Tensor | None = None
    for stage in STAGE_NAMES:
        for block_index in range(len(getattr(original, stage))):
            prefix = f"{stage}.{block_index}"
            keep1 = plan[f"{prefix}.conv1"]["keep_indices"]
            keep2 = plan[f"{prefix}.conv2"]["keep_indices"]

            dst[f"{prefix}.conv1.weight"] = _slice_conv(
                src[f"{prefix}.conv1.weight"], keep1, previous_keep
            )
            _copy_batch_norm(dst, src, f"{prefix}.bn1", keep1)
            dst[f"{prefix}.conv2.weight"] = _slice_conv(
                src[f"{prefix}.conv2.weight"], keep2, keep1
            )
            _copy_batch_norm(dst, src, f"{prefix}.bn2", keep2)

            shortcut_key = f"{prefix}.shortcut.0.weight"
            if shortcut_key in src:
                dst[shortcut_key] = _slice_conv(src[shortcut_key], keep2, previous_keep)
                _copy_batch_norm(dst, src, f"{prefix}.shortcut.1", keep2)

            previous_keep = keep2

    # Classifier: keep the input features that come from the surviving channels.
    dst["linear.weight"] = src["linear.weight"][:, previous_keep].clone()
    dst["linear.bias"] = src["linear.bias"].clone()

    pruned.load_state_dict(dst)
    return pruned


def prune_filters(model: ResNet, prune_percentage: float) -> tuple[nn.Module, int]:
    """Prune ``prune_percentage`` of the filters of every block convolution.

    The input ``model`` is moved to the CPU. If ``prune_percentage`` is outside (0, 1)
    the model is returned unchanged.

    Returns:
        ``(pruned_model, number_of_removed_filters)``.
    """
    if prune_percentage <= 0 or prune_percentage >= 1:
        return model, 0

    model = model.cpu()
    plan, total_pruned = make_pruning_plan(model, prune_percentage)
    return build_pruned_model(model, plan), total_pruned
