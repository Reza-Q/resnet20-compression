"""Network architectures: CIFAR-style ResNets, WideResNet and (pruned) variable-width ResNets.

Pruned networks are ordinary :class:`ResNet` instances built with per-block channel
counts, so the same class serves the baseline, the pruned models and the models used
for quantization-aware training.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torch.ao.nn.quantized import FloatFunctional

from .config import NUM_CLASSES

#: Output width of the three ResNet stages before pruning.
STAGE_WIDTHS = (16, 32, 64)
#: Names of the three stages in a :class:`ResNet`.
STAGE_NAMES = ("layer1", "layer2", "layer3")


class BasicBlock(nn.Module):
    """Two 3x3 convolutions with an identity or 1x1-projection shortcut.

    The residual addition uses ``FloatFunctional`` so that the block can be quantized
    (it behaves exactly like ``torch.add`` in floating point and adds no parameters).
    """

    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.skip_add = FloatFunctional()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.skip_add.add(out, self.shortcut(x))
        return F.relu(out)


class ResNet(nn.Module):
    """CIFAR-style ResNet with three stages of ``BasicBlock``s.

    Args:
        block: residual block class.
        num_blocks: number of blocks in each stage, e.g. ``[3, 3, 3]`` for ResNet20.
        num_classes: number of output classes.
        block_channels: optional output width of every block, one list per stage
            (``[[16, 16, 16], [32, 32, 32], [64, 64, 64]]`` by default). Pruned models
            use narrower widths.
    """

    def __init__(
        self,
        block: type[BasicBlock],
        num_blocks: Sequence[int],
        num_classes: int = NUM_CLASSES,
        block_channels: Sequence[Sequence[int]] | None = None,
    ) -> None:
        super().__init__()
        if block_channels is None:
            block_channels = [[width] * n for width, n in zip(STAGE_WIDTHS, num_blocks)]
        if [len(stage) for stage in block_channels] != list(num_blocks):
            raise ValueError("block_channels must provide one width per block in every stage")

        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, block_channels[0], stride=1)
        self.layer2 = self._make_layer(block, block_channels[1], stride=2)
        self.layer3 = self._make_layer(block, block_channels[2], stride=2)
        self.linear = nn.Linear(block_channels[2][-1] * block.expansion, num_classes)

    def _make_layer(
        self, block: type[BasicBlock], planes_per_block: Sequence[int], stride: int
    ) -> nn.Sequential:
        strides = [stride] + [1] * (len(planes_per_block) - 1)
        layers = []
        for planes, block_stride in zip(planes_per_block, strides):
            layers.append(block(self.in_planes, planes, block_stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        return self.linear(out)


def resnet20(num_classes: int = NUM_CLASSES) -> ResNet:
    return ResNet(BasicBlock, [3, 3, 3], num_classes)


def resnet32(num_classes: int = NUM_CLASSES) -> ResNet:
    return ResNet(BasicBlock, [5, 5, 5], num_classes)


def resnet56(num_classes: int = NUM_CLASSES) -> ResNet:
    return ResNet(BasicBlock, [9, 9, 9], num_classes)


# ----------------------------------------------------------------- WideResNet ------
class WideBasicBlock(nn.Module):
    """Pre-activation residual block of a WideResNet."""

    def __init__(self, in_planes: int, planes: int, dropout_rate: float, stride: int = 1) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=False)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)
        return out


class WideResNet(nn.Module):
    """WideResNet-``depth``-``widen_factor`` for 32x32 inputs."""

    def __init__(
        self,
        depth: int = 28,
        widen_factor: int = 10,
        dropout_rate: float = 0.3,
        num_classes: int = NUM_CLASSES,
    ) -> None:
        super().__init__()
        if (depth - 4) % 6 != 0:
            raise ValueError("WideResNet depth must be of the form 6n + 4")
        n = (depth - 4) // 6
        k = widen_factor
        stages = [16, 16 * k, 32 * k, 64 * k]

        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, stages[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._wide_layer(stages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(stages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(stages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(stages[3], momentum=0.9)
        self.linear = nn.Linear(stages[3], num_classes)

    def _wide_layer(
        self, planes: int, num_blocks: int, dropout_rate: float, stride: int
    ) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for block_stride in strides:
            layers.append(WideBasicBlock(self.in_planes, planes, dropout_rate, block_stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        return self.linear(out)


def wide_resnet28_10(num_classes: int = NUM_CLASSES) -> WideResNet:
    return WideResNet(depth=28, widen_factor=10, dropout_rate=0.3, num_classes=num_classes)


def wide_resnet10_1(num_classes: int = NUM_CLASSES) -> WideResNet:
    """Very small WideResNet, only used as a stand-in teacher in smoke tests."""
    return WideResNet(depth=10, widen_factor=1, dropout_rate=0.3, num_classes=num_classes)


# ------------------------------------------------- variable-width (pruned) models ------
def build_resnet_with_channels(
    block_channels: Sequence[Sequence[int]], num_classes: int = NUM_CLASSES
) -> ResNet:
    """Build a ResNet whose blocks have the given output widths (one list per stage)."""
    num_blocks = [len(stage) for stage in block_channels]
    return ResNet(BasicBlock, num_blocks, num_classes, block_channels)


def infer_block_channels(state_dict: dict[str, torch.Tensor]) -> list[list[int]]:
    """Read the width of every block (output channels of ``conv2``) from a state dict."""
    block_channels: list[list[int]] = []
    for stage in STAGE_NAMES:
        widths: list[int] = []
        while f"{stage}.{len(widths)}.conv2.weight" in state_dict:
            widths.append(state_dict[f"{stage}.{len(widths)}.conv2.weight"].shape[0])
        if not widths:
            raise ValueError(f"State dict contains no blocks for '{stage}'")
        block_channels.append(widths)
    return block_channels


def build_model_from_state_dict(state_dict: dict[str, torch.Tensor]) -> ResNet:
    """Rebuild a (pruned) ResNet from a ``state_dict`` and load its weights."""
    num_classes = state_dict["linear.weight"].shape[0]
    model = build_resnet_with_channels(infer_block_channels(state_dict), num_classes)
    model.load_state_dict(state_dict)
    return model


# ------------------------------------------------------------- teacher registry ------
#: (checkpoint file name, model factory, display name) of the teacher ensemble.
TEACHER_SPECS = [
    ("teacher_resnet32.pth", resnet32, "ResNet32"),
    ("teacher_resnet56.pth", resnet56, "ResNet56"),
    ("teacher_wideresnet2810.pth", wide_resnet28_10, "WideResNet-28-10"),
]

#: Lightweight replacement used when ``Config.smoke_test`` is set.
SMOKE_TEACHER_SPECS = [
    ("teacher_resnet32.pth", resnet32, "ResNet32"),
    ("teacher_resnet56.pth", resnet56, "ResNet56"),
    ("teacher_wideresnet10_1.pth", wide_resnet10_1, "WideResNet-10-1"),
]
