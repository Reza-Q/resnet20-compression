"""Shared pytest fixtures: small deterministic models and data for fast CPU tests."""

from __future__ import annotations

import pytest
import torch

from resnet_compression.config import Config
from resnet_compression.models import resnet20


@pytest.fixture(autouse=True)
def _deterministic():
    torch.manual_seed(0)
    yield


@pytest.fixture()
def tiny_config(tmp_path) -> Config:
    cfg = Config(output_dir=str(tmp_path / "experiment"), data_dir=str(tmp_path / "data"))
    cfg.apply_smoke_test()
    cfg.make_dirs()
    return cfg


@pytest.fixture()
def small_resnet20() -> torch.nn.Module:
    return resnet20(num_classes=10)
