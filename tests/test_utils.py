import torch
from torch import nn

from resnet_compression.utils import (
    count_conv_filters,
    count_parameters,
    get_device,
    get_model_size_mb,
    load_checkpoint,
    load_json,
    rate_tag,
    save_checkpoint,
    save_json,
    set_seed,
)


def test_set_seed_makes_torch_rand_reproducible():
    set_seed(123)
    a = torch.rand(5)
    set_seed(123)
    b = torch.rand(5)
    assert torch.equal(a, b)


def test_get_device_auto_and_explicit():
    assert get_device("cpu") == torch.device("cpu")
    resolved = get_device("auto")
    assert resolved.type in ("cpu", "cuda")


def test_rate_tag_formatting():
    assert rate_tag(0.02) == "2pct"
    assert rate_tag(0.1) == "10pct"
    assert rate_tag(0.005) == "0.5pct"


def test_count_parameters_and_filters_on_a_tiny_conv_net():
    net = nn.Sequential(nn.Conv2d(3, 4, 3, bias=False), nn.Conv2d(4, 5, 3, bias=False))
    assert count_conv_filters(net) == 9  # 4 + 5 output channels
    expected_params = (3 * 4 * 3 * 3) + (4 * 5 * 3 * 3)  # no bias terms
    assert count_parameters(net) == expected_params


def test_get_model_size_mb_is_positive_and_scales_with_width():
    small = nn.Conv2d(3, 4, 3)
    large = nn.Conv2d(3, 40, 3)
    assert get_model_size_mb(small) > 0
    assert get_model_size_mb(large) > get_model_size_mb(small)


def test_checkpoint_roundtrip(tmp_path):
    path = tmp_path / "ckpt.pth"
    payload = {"a": torch.tensor([1, 2, 3]), "b": 42}

    assert save_checkpoint(payload, path) is True
    loaded, ok = load_checkpoint(path)
    assert ok is True
    assert loaded["b"] == 42
    assert torch.equal(loaded["a"], payload["a"])


def test_load_checkpoint_missing_file_returns_false(tmp_path):
    loaded, ok = load_checkpoint(tmp_path / "missing.pth")
    assert loaded is None
    assert ok is False


def test_json_roundtrip_with_numpy_scalars(tmp_path):
    import numpy as np

    path = tmp_path / "results.json"
    data = {"accuracy": np.float64(91.5), "count": np.int64(7), "nested": {"x": [1, 2, 3]}}
    save_json(data, path)
    loaded = load_json(path)
    assert loaded == {"accuracy": 91.5, "count": 7, "nested": {"x": [1, 2, 3]}}
