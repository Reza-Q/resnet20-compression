import torch
from torch.utils.data import DataLoader, TensorDataset

from resnet_compression.models import build_resnet_with_channels
from resnet_compression.quantization import (
    convert_to_quantized,
    int4_qconfig,
    prepare_qat_model,
    train_qat,
)


def _tiny_model():
    # A tiny but structurally normal 3-stage ResNet, fast enough to prepare/convert in a test.
    return build_resnet_with_channels([[4, 4, 4], [8, 8, 8], [12, 12, 12]], num_classes=5)


def test_prepare_qat_model_does_not_mutate_the_source_model():
    model = _tiny_model()
    state_before = {k: v.clone() for k, v in model.state_dict().items()}

    prepare_qat_model(model, qconfig_type="int8")

    for key, value in model.state_dict().items():
        assert torch.equal(value, state_before[key]), f"{key} was mutated by prepare_qat_model"


def test_int8_and_int4_preparations_are_independent():
    """Regression test for the notebook bug described in quantization.py's module docstring.

    Preparing INT8 and then INT4 QAT from the same source model must not leave the INT4
    model with any INT8-range fake-quantizers: every internal conv, not just the input
    stub, must reflect the qconfig it was actually prepared with.
    """
    model = _tiny_model()

    qat_int8 = prepare_qat_model(model, qconfig_type="int8", backend="fbgemm")
    qat_int4 = prepare_qat_model(model, qconfig_type="int4")

    # The two wrappers must not share any submodules (each call deep-copies its input).
    assert qat_int8.layer2[0].conv1 is not qat_int4.layer2[0].conv1

    int8_weight_fq = qat_int8.layer2[0].conv1.weight_fake_quant
    int4_weight_fq = qat_int4.layer2[0].conv1.weight_fake_quant
    int4_act_fq = qat_int4.layer2[0].conv1.activation_post_process

    assert (int8_weight_fq.quant_min, int8_weight_fq.quant_max) == (-128, 127)
    assert (int4_weight_fq.quant_min, int4_weight_fq.quant_max) == (-8, 7)
    assert (int4_act_fq.quant_min, int4_act_fq.quant_max) == (0, 15)


def test_int4_qconfig_ranges():
    qconfig = int4_qconfig()
    weight_fq = qconfig.weight()
    activation_fq = qconfig.activation()
    assert (weight_fq.quant_min, weight_fq.quant_max) == (-8, 7)
    assert (activation_fq.quant_min, activation_fq.quant_max) == (0, 15)


def test_qat_train_and_convert_runs_end_to_end():
    model = _tiny_model()
    qat_model = prepare_qat_model(model, qconfig_type="int8")

    x = torch.randn(16, 3, 32, 32)
    y = torch.randint(0, 5, (16,))
    loader = DataLoader(TensorDataset(x, y), batch_size=4)

    qat_model, best_acc, history = train_qat(
        qat_model, loader, loader, epochs=1, device=torch.device("cpu"), lr=1e-3
    )
    assert 0.0 <= best_acc <= 100.0
    assert len(history["train_loss"]) == 1

    quantized = convert_to_quantized(qat_model)
    quantized.eval()
    with torch.no_grad():
        out = quantized(x)
    assert tuple(out.shape) == (16, 5)
