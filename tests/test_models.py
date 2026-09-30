import torch

from resnet_compression.models import (
    build_model_from_state_dict,
    build_resnet_with_channels,
    infer_block_channels,
    resnet20,
    resnet32,
    resnet56,
    wide_resnet10_1,
    wide_resnet28_10,
)

BATCH, CHANNELS, SIZE = 2, 3, 32


def _forward_shape_ok(model: torch.nn.Module, num_classes: int) -> bool:
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(BATCH, CHANNELS, SIZE, SIZE))
    return tuple(out.shape) == (BATCH, num_classes)


def test_resnet20_forward_shape():
    assert _forward_shape_ok(resnet20(num_classes=100), 100)


def test_resnet32_and_56_forward_shape():
    assert _forward_shape_ok(resnet32(num_classes=100), 100)
    assert _forward_shape_ok(resnet56(num_classes=100), 100)


def test_wide_resnet_forward_shape():
    assert _forward_shape_ok(wide_resnet10_1(num_classes=100), 100)


def test_wide_resnet28_10_forward_shape():
    assert _forward_shape_ok(wide_resnet28_10(num_classes=10), 10)


def test_resnet20_parameter_count_matches_known_value():
    # A standard CIFAR ResNet20 (3 stages x 3 blocks, widths 16/32/64) for 100 classes
    # has 278,324 trainable parameters; regressions in the architecture would change this.
    model = resnet20(num_classes=100)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n_params == 278_324


def test_rejects_mismatched_block_channels():
    import pytest

    from resnet_compression.models import BasicBlock, ResNet

    with pytest.raises(ValueError):
        # num_blocks says 3 blocks per stage, but layer1's channel list only has 2.
        ResNet(BasicBlock, [3, 3, 3], block_channels=[[16, 16], [32, 32, 32], [64, 64, 64]])


def test_infer_block_channels_roundtrip_uniform_width():
    channels = [[12, 12, 12], [20, 20, 20], [40, 40, 40]]
    model = build_resnet_with_channels(channels, num_classes=100)
    inferred = infer_block_channels(model.state_dict())
    assert inferred == channels


def test_infer_block_channels_roundtrip_nonuniform_width():
    # infer_block_channels reads each block's own width, so it must also work when
    # blocks within a stage differ (pruning happens to keep them equal in practice,
    # but the reader itself makes no such assumption).
    channels = [[10, 14, 12], [22, 18, 20], [40, 36, 44]]
    model = build_resnet_with_channels(channels, num_classes=100)
    inferred = infer_block_channels(model.state_dict())
    assert inferred == channels


def test_build_model_from_state_dict_reproduces_forward_pass():
    original = build_resnet_with_channels([[14] * 3, [28] * 3, [50] * 3], num_classes=100)
    rebuilt = build_model_from_state_dict(original.state_dict())
    original.eval()
    rebuilt.eval()
    x = torch.randn(BATCH, CHANNELS, SIZE, SIZE)
    with torch.no_grad():
        assert torch.equal(original(x), rebuilt(x))
