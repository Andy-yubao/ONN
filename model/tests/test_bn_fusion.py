"""Tests for BatchNorm fusion — numeric equivalence, structure, and round-trip.

These tests use only synthetic random tensors (no MNIST download) and run on
CPU.  Coverage complements ``test_quantization.py`` with a focus on the
delivery-candidate ``BaselineCNN``.
"""

import copy

import pytest
import torch
from torch import nn

from onn_model.models.baseline_cnn import BaselineCNN
from onn_model.quantization import (
    check_bn_fusion_error,
    fuse_conv_bn_eval,
    fuse_model_bn,
    load_bn_fused_model,
)

DEVICE = torch.device("cpu")
FUSION_THRESHOLD = 1e-4


# --------------------------------------------------------------------------
#  Single Conv2d + BatchNorm2d fusion
# --------------------------------------------------------------------------

def _make_conv_bn(bias: bool) -> tuple:
    torch.manual_seed(0)
    conv = nn.Conv2d(3, 8, kernel_size=3, padding=1, bias=bias)
    bn = nn.BatchNorm2d(8)
    # Non-trivial BN parameters
    bn.weight.data.uniform_(0.8, 1.4)
    bn.bias.data.uniform_(-0.2, 0.2)
    bn.running_mean.data.uniform_(-0.1, 0.1)
    bn.running_var.data.uniform_(0.8, 1.6)
    conv.eval()
    bn.eval()
    return conv, bn


class TestSingleConvBNFusion:
    def test_equivalent_no_bias(self):
        conv, bn = _make_conv_bn(bias=False)
        x = torch.randn(4, 3, 16, 16)
        fused = fuse_conv_bn_eval(conv, bn)
        err = (bn(conv(x)) - fused(x)).abs().max().item()
        assert err < FUSION_THRESHOLD, f"max err = {err:.3e}"

    def test_equivalent_with_bias(self):
        conv, bn = _make_conv_bn(bias=True)
        x = torch.randn(4, 3, 16, 16)
        fused = fuse_conv_bn_eval(conv, bn)
        err = (bn(conv(x)) - fused(x)).abs().max().item()
        assert err < FUSION_THRESHOLD, f"max err = {err:.3e}"

    def test_fused_conv_carries_bias(self):
        conv, bn = _make_conv_bn(bias=False)
        fused = fuse_conv_bn_eval(conv, bn)
        assert fused.bias is not None
        assert fused.bias.shape == (8,)

    def test_preserves_conv_geometry(self):
        conv, bn = _make_conv_bn(bias=False)
        fused = fuse_conv_bn_eval(conv, bn)
        assert fused.kernel_size == conv.kernel_size
        assert fused.stride == conv.stride
        assert fused.padding == conv.padding
        assert fused.dilation == conv.dilation
        assert fused.groups == conv.groups
        assert fused.weight.shape == conv.weight.shape


# --------------------------------------------------------------------------
#  BaselineCNN full-model fusion
# --------------------------------------------------------------------------

def _baseline_cnn_eval() -> BaselineCNN:
    torch.manual_seed(1)
    model = BaselineCNN()
    model.eval()
    return model


def _layer_outputs(model: nn.Module, x: torch.Tensor) -> dict:
    """Run a forward pass capturing each functional stage output (by position)."""
    captured = {}
    targets = {
        "stem_relu": model.stem[-1],
        "pool1": model.pool1,
        "conv2_relu": model.conv2[-1],
        "pool2": model.pool2,
        "conv3_relu": model.conv3[-1],
        "pool": model.pool,
        "logits": model.fc,
    }
    handles = []

    def _hook(key):
        def _fn(_m, _i, out):
            captured[key] = out.detach().clone()

        return _fn

    for key, mod in targets.items():
        handles.append(mod.register_forward_hook(_hook(key)))
    try:
        model(x)
    finally:
        for h in handles:
            h.remove()
    return captured


def _count_type(model: nn.Module, t) -> int:
    return sum(1 for m in model.modules() if isinstance(m, t))


class TestBaselineCNNBNFusion:
    @pytest.fixture
    def x(self):
        torch.manual_seed(2)
        return torch.randn(4, 1, 28, 28)

    def test_fused_model_has_no_batchnorm(self, x):
        model = _baseline_cnn_eval()
        fused = fuse_model_bn(model)
        assert _count_type(fused, nn.BatchNorm2d) == 0
        assert _count_type(fused, nn.Conv2d) == 3
        assert _count_type(fused, nn.Linear) == 1
        # original still has its three BatchNorm layers
        assert _count_type(model, nn.BatchNorm2d) == 3

    def test_output_error_below_threshold(self, x):
        model = _baseline_cnn_eval()
        fused = fuse_model_bn(model)
        result = check_bn_fusion_error(model, fused, x)
        assert result["output_ok"], f"max err = {result['max_error']:.3e}"
        assert result["max_error"] < FUSION_THRESHOLD

    def test_layerwise_error_below_threshold(self, x):
        model = _baseline_cnn_eval()
        fused = fuse_model_bn(model)
        orig_out = _layer_outputs(model, x.clone())
        fused_out = _layer_outputs(fused, x.clone())
        assert set(orig_out) == set(fused_out)
        for key in orig_out:
            a, b = orig_out[key], fused_out[key]
            assert tuple(a.shape) == tuple(b.shape), f"{key}: shape mismatch"
            assert torch.isfinite(a).all() and torch.isfinite(b).all(), f"{key}: NaN/Inf"
            err = (a - b).abs().max().item()
            assert err < FUSION_THRESHOLD, f"{key}: max err = {err:.3e}"

    def test_original_model_untouched(self):
        model = _baseline_cnn_eval()
        before = copy.deepcopy(model.state_dict())
        _ = fuse_model_bn(model)
        for k, v in before.items():
            assert torch.equal(v, model.state_dict()[k]), f"param {k} changed"

    def test_fused_model_eval_and_output_shape(self, x):
        model = _baseline_cnn_eval()
        fused = fuse_model_bn(model)
        # Fusion uses BN running stats (eval semantics); fused model must be eval
        assert not fused.training
        assert fused(x).shape == (4, 10)


# --------------------------------------------------------------------------
#  Save / reload round-trip
# --------------------------------------------------------------------------

class TestFusedCheckpointRoundTrip:
    def test_save_and_reload_roundtrip(self, tmp_path):
        torch.manual_seed(3)
        model = _baseline_cnn_eval()
        fused = fuse_model_bn(model)
        x = torch.randn(8, 1, 28, 28)

        ckpt_path = str(tmp_path / "fused.pt")
        torch.save(
            {
                "model_name": "BaselineCNN",
                "bn_fused": True,
                "dtype": "float32",
                "input_shape": [1, 28, 28],
                "state_dict": fused.state_dict(),
            },
            ckpt_path,
        )

        loaded, meta = load_bn_fused_model(ckpt_path, BaselineCNN, DEVICE)
        assert meta["model_name"] == "BaselineCNN"
        assert meta["bn_fused"] is True
        assert _count_type(loaded, nn.BatchNorm2d) == 0

        with torch.no_grad():
            out_before = fused(x.clone())
            out_after = loaded(x.clone())
        assert out_before.shape == (8, 10)
        assert torch.equal(out_before, out_after), "round-trip outputs differ"

    def test_strict_load_rejects_bad_state_dict(self, tmp_path):
        fused = fuse_model_bn(_baseline_cnn_eval())
        ckpt_path = str(tmp_path / "bad.pt")
        # Save a state dict with a random extra key — strict load must fail
        sd = fused.state_dict()
        sd["bogus_key"] = torch.zeros(1)
        torch.save({"state_dict": sd}, ckpt_path)
        with pytest.raises(RuntimeError):
            load_bn_fused_model(ckpt_path, BaselineCNN, DEVICE)
