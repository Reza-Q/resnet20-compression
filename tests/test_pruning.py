import torch

from resnet_compression.models import resnet20
from resnet_compression.pruning import MIN_FILTERS, make_pruning_plan, prune_filters
from resnet_compression.utils import count_conv_filters, count_parameters


def test_noop_outside_open_interval():
    model = resnet20(num_classes=10)
    for rate in (0.0, 1.0, -0.1, 1.5):
        pruned, num_pruned = prune_filters(model, rate)
        assert num_pruned == 0
        assert pruned is model


def test_pruning_reduces_parameters_and_filters():
    model = resnet20(num_classes=10)
    params_before = count_parameters(model)
    filters_before = count_conv_filters(model)

    pruned, num_pruned = prune_filters(model, 0.2)

    assert num_pruned > 0
    assert count_parameters(pruned) < params_before
    # Total filter count is not simply `filters_before - num_pruned`: num_pruned only
    # counts the two 3x3 convs of each block, but pruning also resizes (or, if a
    # stage's first block newly needs a projection, creates) 1x1 shortcut convs to keep
    # the residual add shape-consistent. Those filters move independently, so only the
    # overall decrease is guaranteed, not an exact filter-count bookkeeping identity.
    assert count_conv_filters(pruned) < filters_before


def test_forward_pass_shape_is_preserved_after_pruning():
    model = resnet20(num_classes=10)
    pruned, _ = prune_filters(model, 0.3)
    pruned.eval()
    with torch.no_grad():
        out = pruned(torch.randn(2, 3, 32, 32))
    assert tuple(out.shape) == (2, 10)


def test_every_block_conv_respects_minimum_filter_count():
    model = resnet20(num_classes=10)
    # A very high rate would remove almost everything without the MIN_FILTERS floor.
    plan, _ = make_pruning_plan(model, 0.99)
    assert all(entry["num_kept"] >= MIN_FILTERS for entry in plan.values())


def test_iterative_pruning_is_monotonically_non_increasing():
    model = resnet20(num_classes=10)
    sizes = [count_parameters(model)]
    current = model
    for _ in range(8):
        current, _ = prune_filters(current, 0.1)
        sizes.append(count_parameters(current))
    assert all(sizes[i] >= sizes[i + 1] for i in range(len(sizes) - 1))


def test_pruning_keeps_the_highest_l1_norm_filters():
    model = resnet20(num_classes=10)
    # Make layer1.0.conv1's filter importances unambiguous: give filter 0 a tiny norm
    # and every other filter a large, distinct norm.
    conv = model.layer1[0].conv1
    with torch.no_grad():
        conv.weight.zero_()
        conv.weight[0] = 1e-6
        for i in range(1, conv.out_channels):
            conv.weight[i] = float(i + 1)

    plan, _ = make_pruning_plan(model, 0.2)
    kept = plan["layer1.0.conv1"]["keep_indices"].tolist()
    assert 0 not in kept
