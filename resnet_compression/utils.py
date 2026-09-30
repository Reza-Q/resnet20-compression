"""Shared helpers: logging, seeding, checkpoint / JSON I/O and model statistics."""

from __future__ import annotations

import json
import logging
import pickle
import random
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

logger = logging.getLogger(__name__)

PACKAGE_LOGGER_NAME = "resnet_compression"


# --------------------------------------------------------------------- logging ------
def setup_logging(log_file: Path | None = None, level: int = logging.INFO) -> None:
    """Configure the package logger (console output and an optional log file)."""
    package_logger = logging.getLogger(PACKAGE_LOGGER_NAME)
    package_logger.setLevel(level)
    package_logger.handlers.clear()
    package_logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S")
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    package_logger.addHandler(console)

    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setFormatter(formatter)
        package_logger.addHandler(file_handler)


def log_section(title: str, char: str = "=", width: int = 80) -> None:
    """Log a titled separator block."""
    logger.info(char * width)
    logger.info(title)
    logger.info(char * width)


# ------------------------------------------------------------ reproducibility ------
def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch (CPU and CUDA) random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(preference: str = "auto") -> torch.device:
    """Resolve a device string; ``"auto"`` selects CUDA when it is available."""
    if preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(preference)


# ------------------------------------------------------------------------ I/O ------
def save_checkpoint(obj: Any, path: Path) -> bool:
    """Save ``obj`` with ``torch.save``; returns ``True`` on success."""
    path = Path(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(obj, path)
        logger.info("Saved: %s", path.name)
        return True
    except Exception as exc:  # noqa: BLE001 - report any I/O problem and carry on
        logger.error("Could not save %s: %s", path, exc)
        return False


def load_torch_file(path: Path, map_location: Any = "cpu") -> Any:
    """Load a file written by ``torch.save``.

    Tries the safe ``weights_only`` mode first. Files that contain arbitrary Python
    objects are loaded with a warning; only do this for files you created yourself.
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except pickle.UnpicklingError:
        logger.warning("%s is not a plain weights file; loading with weights_only=False", path)
        return torch.load(path, map_location=map_location, weights_only=False)


def load_checkpoint(path: Path, device: Any = "cpu") -> tuple[Any | None, bool]:
    """Load a checkpoint if it exists; returns ``(object, success)``."""
    path = Path(path)
    if not path.exists():
        return None, False
    try:
        obj = load_torch_file(path, map_location=device)
        logger.info("Loaded: %s", path.name)
        return obj, True
    except Exception as exc:  # noqa: BLE001
        logger.error("Could not load %s: %s", path, exc)
        return None, False


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def save_json(data: Any, path: Path) -> None:
    """Write ``data`` as indented JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, default=_json_default)
    logger.info("Saved: %s", path.name)


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


# ----------------------------------------------------------- naming conventions ------
def rate_tag(rate: float) -> str:
    """File-name tag for a pruning rate, e.g. ``0.02 -> "2pct"``."""
    return f"{rate * 100:g}pct"


# ------------------------------------------------------------ model statistics ------
def count_parameters(model: nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_conv_filters(model: nn.Module) -> int:
    """Total number of convolutional filters (output channels of all ``Conv2d`` layers)."""
    return sum(m.out_channels for m in model.modules() if isinstance(m, nn.Conv2d))


def get_model_size_mb(model: nn.Module) -> float:
    """In-memory size of parameters and buffers in MiB."""
    param_bytes = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.nelement() * b.element_size() for b in model.buffers())
    return (param_bytes + buffer_bytes) / (1024**2)


def get_quantized_model_size_mb(model: nn.Module) -> float:
    """Size in MiB of a (quantized) model's ``state_dict`` serialized to disk."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "model.pth"
        torch.save(model.state_dict(), path)
        return path.stat().st_size / (1024**2)


def count_quantized_parameters(model: nn.Module) -> int:
    """Count all tensor elements of a quantized model (parameters and stored buffers)."""
    parameter_names = {name for name, _ in model.named_parameters()}
    total = sum(p.numel() for p in model.parameters())
    for key, value in model.state_dict().items():
        if isinstance(value, torch.Tensor) and key not in parameter_names:
            total += value.numel()
    return total if total > 0 else count_parameters(model)
