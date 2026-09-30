"""Quantization-aware training (QAT).

Wraps a floating-point :class:`~resnet_compression.models.ResNet` with quant/dequant
stubs, fake-quantizes it for either INT8 (a standard backend qconfig) or INT4 (a custom
symmetric weight / affine activation qconfig), fine-tunes it, and converts the result to
a genuinely quantized model for size and accuracy measurement.

Bug fixed relative to the original notebook
--------------------------------------------
The notebook prepared INT8 and INT4 QAT by calling its ``prepare_qat_model(best_model, ...)``
helper *twice on the very same in-memory model*, without copying it. That helper builds a
thin wrapper around the model's existing submodules (it assigns ``self.conv1 = base_model.conv1``
etc., which does not copy anything) and then calls ``torch.quantization.prepare_qat(...,
inplace=True)``, which recurses through the module tree and gives every submodule a
``.qconfig`` attribute plus fake-quantize hooks.

Because the second call reused the *same* submodule objects the first call had already
mutated, and ``prepare_qat``'s internal qconfig propagation keeps a module's existing
``.qconfig`` if it already has one, every internal conv/activation kept the INT8 range from
the first call; only the freshly constructed top-level input stub picked up the INT4 range.
In other words, the notebook's "INT4" model was an INT8 model with an INT4 input stub, not a
genuinely INT4 network. This was confirmed empirically by re-running the notebook's own
functions on a shared model and inspecting ``quant_min``/``quant_max`` of the fake-quantizers
before and after the second call.

The fix here is to deep-copy the floating-point model inside :func:`prepare_qat_model` before
wrapping it, so that preparing one bit-width never leaves state behind for another.
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import torch
import torch.ao.quantization as tq
import torch.nn.functional as F
from torch import nn, optim
from torch.ao.quantization import DeQuantStub, QuantStub
from torch.utils.data import DataLoader
from tqdm import tqdm

from .models import ResNet
from .training import ensemble_distillation_loss, evaluate

logger = logging.getLogger(__name__)

QConfigType = Literal["int8", "int4"]


class QATWrapper(nn.Module):
    """Adds quant/dequant stubs around a :class:`~resnet_compression.models.ResNet`."""

    def __init__(self, base_model: ResNet) -> None:
        super().__init__()
        self.quant = QuantStub()
        self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.linear = base_model.linear
        self.dequant = DeQuantStub()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quant(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = F.avg_pool2d(x, 8)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return self.dequant(x)


def int4_qconfig() -> tq.QConfig:
    """Symmetric 4-bit weights (``[-8, 7]``), affine 4-bit activations (``[0, 15]``)."""
    return tq.QConfig(
        activation=tq.FakeQuantize.with_args(
            observer=tq.MovingAverageMinMaxObserver,
            quant_min=0,
            quant_max=15,
            dtype=torch.quint8,
            qscheme=torch.per_tensor_affine,
            reduce_range=False,
        ),
        weight=tq.FakeQuantize.with_args(
            observer=tq.MovingAverageMinMaxObserver,
            quant_min=-8,
            quant_max=7,
            dtype=torch.qint8,
            qscheme=torch.per_tensor_symmetric,
            reduce_range=False,
        ),
    )


def prepare_qat_model(model: ResNet, qconfig_type: QConfigType = "int8", backend: str = "fbgemm") -> QATWrapper:
    """Wrap a copy of ``model`` and insert fake-quantization modules for QAT.

    ``model`` is deep-copied first (see the module docstring): the returned wrapper never
    shares, or mutates, state with ``model`` or with a wrapper built from an earlier call.
    """
    qat_model = QATWrapper(copy.deepcopy(model))
    qat_model.train()
    qat_model.qconfig = (
        tq.get_default_qat_qconfig(backend) if qconfig_type == "int8" else int4_qconfig()
    )
    qat_model = qat_model.cpu()
    tq.prepare_qat(qat_model, inplace=True)
    return qat_model


def convert_to_quantized(qat_model: QATWrapper) -> nn.Module:
    """Fold the fake-quantizers of a trained QAT model into a real quantized model."""
    qat_model = qat_model.eval().cpu()
    return tq.convert(qat_model, inplace=False)


# ------------------------------------------------------------------------ training ------
def train_qat(
    student_qat: QATWrapper,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int,
    device: torch.device,
    lr: float = 1e-4,
    teachers: Sequence[nn.Module] = (),
    temperature: float = 4.0,
    alpha: float = 0.7,
    desc: str = "QAT",
) -> tuple[QATWrapper, float, dict[str, list[float]]]:
    """Fine-tune a QAT-prepared model, optionally against a teacher ensemble (KD).

    Pass ``teachers=()`` (the default) for plain fine-tuning; a non-empty sequence enables
    KD, matching ``alpha * soft_loss + (1 - alpha) * hard_loss``. Uses Adam, as in the
    original QAT fine-tuning (SGD is used for the earlier, full-precision stages).

    Returns ``(student_qat, best_test_accuracy, history)``.
    """
    student_qat = student_qat.to(device)
    use_kd = len(teachers) > 0
    for teacher in teachers:
        teacher.eval()
        teacher.to(device)

    optimizer = optim.Adam(student_qat.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    history: dict[str, list[float]] = {"train_loss": [], "test_acc": []}

    for epoch in range(epochs):
        student_qat.train()
        total_loss = 0.0
        progress = tqdm(train_loader, desc=f"{desc} Epoch {epoch + 1}/{epochs}", leave=False)
        for data, target in progress:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = student_qat(data)

            if use_kd:
                with torch.no_grad():
                    teacher_outputs = [teacher(data) for teacher in teachers]
                loss_hard = criterion(output, target)
                loss_soft = ensemble_distillation_loss(output, teacher_outputs, temperature)
                loss = alpha * loss_soft + (1 - alpha) * loss_hard
            else:
                loss = criterion(output, target)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            progress.set_postfix({"loss": f"{loss.item():.3f}"})

        scheduler.step()
        test_acc = evaluate(student_qat, test_loader, device)
        history["train_loss"].append(total_loss / len(train_loader))
        history["test_acc"].append(test_acc)
        best_acc = max(best_acc, test_acc)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(
                "Epoch %d/%d: Test=%.2f%%, Best=%.2f%%", epoch + 1, epochs, test_acc, best_acc
            )

    return student_qat, best_acc, history


@dataclass
class QATResult:
    """Outcome of preparing, fine-tuning, converting and evaluating one bit-width."""

    bit_width: str  # "int8" or "int4"
    accuracy: float
    parameters: int
    size_mb: float
    compression_ratio: float
    accuracy_drop: float
    history: dict[str, list[float]]

    def to_dict(self) -> dict[str, object]:
        return {
            "accuracy": self.accuracy,
            "parameters": self.parameters,
            "size_mb": self.size_mb,
            "compression_ratio": self.compression_ratio,
            "accuracy_drop": self.accuracy_drop,
        }
