"""Training and evaluation loops, including ensemble knowledge distillation (KD)."""

from __future__ import annotations

import logging
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------- basic loops ------
def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    desc: str = "Training",
) -> tuple[float, float]:
    """Train ``model`` for one epoch with hard labels; returns ``(mean loss, accuracy %)``."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    progress = tqdm(loader, desc=desc, leave=False)
    for data, target in progress:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += output.argmax(dim=1).eq(target).sum().item()
        total += target.size(0)
        progress.set_postfix({"loss": f"{loss.item():.3f}", "acc": f"{100.0 * correct / total:.2f}%"})

    return total_loss / len(loader), 100.0 * correct / total


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Top-1 accuracy (%) of ``model`` on ``loader``."""
    model.eval()
    correct = 0
    total = 0
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        correct += output.argmax(dim=1).eq(target).sum().item()
        total += target.size(0)
    return 100.0 * correct / total


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
    model_name: str = "Model",
) -> tuple[nn.Module, float, dict[str, list[float]]]:
    """Train with SGD + cosine annealing.

    Returns:
        ``(model, best_test_accuracy, history)``. The returned model holds the weights of
        the *last* epoch; ``best_test_accuracy`` is the best accuracy seen on the test
        set during training.
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0
    history: dict[str, list[float]] = {"train_loss": [], "train_acc": [], "test_acc": []}

    logger.info("Training %s ...", model_name)
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, desc=f"Epoch {epoch + 1}"
        )
        test_acc = evaluate(model, test_loader, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)
        best_acc = max(best_acc, test_acc)

        if (epoch + 1) % 20 == 0 or epoch == 0:
            logger.info(
                "Epoch %d/%d: Loss=%.4f, Train=%.2f%%, Test=%.2f%%, Best=%.2f%%",
                epoch + 1,
                epochs,
                train_loss,
                train_acc,
                test_acc,
                best_acc,
            )

    return model, best_acc, history


# ------------------------------------------------------------------ distillation ------
def distillation_loss(
    student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float = 4.0
) -> torch.Tensor:
    """KL divergence between temperature-softened student and teacher distributions."""
    soft_student = F.log_softmax(student_logits / temperature, dim=1)
    soft_teacher = F.softmax(teacher_logits / temperature, dim=1)
    return F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (temperature**2)


def ensemble_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits_list: Sequence[torch.Tensor],
    temperature: float = 4.0,
) -> torch.Tensor:
    """Distillation loss against the average logits of an ensemble of teachers."""
    mean_teacher_logits = torch.mean(torch.stack(list(teacher_logits_list)), dim=0)
    return distillation_loss(student_logits, mean_teacher_logits, temperature)


def train_epoch_kd(
    student: nn.Module,
    teachers: Sequence[nn.Module],
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion_hard: nn.Module,
    device: torch.device,
    temperature: float,
    alpha: float,
    desc: str = "KD",
) -> float:
    """One epoch of KD training: ``alpha * soft loss + (1 - alpha) * hard loss``.

    Returns the mean training loss. Teachers must already be on ``device`` and in eval mode.
    """
    student.train()
    total_loss = 0.0

    progress = tqdm(loader, desc=desc, leave=False)
    for data, target in progress:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        student_output = student(data)

        with torch.no_grad():
            teacher_outputs = [teacher(data) for teacher in teachers]

        loss_hard = criterion_hard(student_output, target)
        loss_soft = ensemble_distillation_loss(student_output, teacher_outputs, temperature)
        loss = alpha * loss_soft + (1 - alpha) * loss_hard

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        progress.set_postfix({"loss": f"{loss.item():.3f}"})

    return total_loss / len(loader)


def train_with_kd(
    student: nn.Module,
    teachers: Sequence[nn.Module],
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int,
    device: torch.device,
    lr: float = 0.001,
    temperature: float = 4.0,
    alpha: float = 0.7,
    desc: str = "KD",
) -> tuple[nn.Module, float, dict[str, list[float]]]:
    """Fine-tune ``student`` against an ensemble of teachers (SGD + cosine annealing).

    Returns ``(student, best_test_accuracy, history)``; like :func:`train_model`, the
    returned student holds the weights of the last epoch.
    """
    student = student.to(device)
    for teacher in teachers:
        teacher.eval()
        teacher.to(device)

    optimizer = optim.SGD(student.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion_hard = nn.CrossEntropyLoss()

    best_acc = 0.0
    history: dict[str, list[float]] = {"train_loss": [], "test_acc": []}

    for epoch in range(epochs):
        train_loss = train_epoch_kd(
            student,
            teachers,
            train_loader,
            optimizer,
            criterion_hard,
            device,
            temperature,
            alpha,
            desc=f"{desc} Epoch {epoch + 1}/{epochs}",
        )
        scheduler.step()
        test_acc = evaluate(student, test_loader, device)

        history["train_loss"].append(train_loss)
        history["test_acc"].append(test_acc)
        best_acc = max(best_acc, test_acc)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(
                "Epoch %d/%d: Loss=%.4f, Test=%.2f%%, Best=%.2f%%",
                epoch + 1,
                epochs,
                train_loss,
                test_acc,
                best_acc,
            )

    return student, best_acc, history
