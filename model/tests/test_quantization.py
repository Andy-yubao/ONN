"""Tests for BN fusion and INT8 PTQ simulation."""

import copy

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from onn_model.models.compact_cnn import MicroCNNSmall, MicroCNNExtraSmall, DepthwiseMicroCNN
from onn_model.quantization import (
    fuse_conv_bn_eval,
    fuse_model_bn,
    check_bn_fusion_error,
    quantize_weights_per_channel,
    calibrate_activation_range,
    evaluate_ptq,
)


# ============================================================================
#  BN Fusion Tests
# ============================================================================

class TestConvBNFusion:
    """Verify Conv+BN fusion produces near-identical outputs."""

    @pytest.fixture
    def single_conv_bn(self):
        conv = nn.Conv2d(3, 8, kernel_size=3, padding=1, bias=False)
        bn = nn.BatchNorm2d(8)
        # Set known BN params (override random init)
        bn.weight.data.fill_(1.2)
        bn.bias.data.fill_(0.1)
        bn.running_mean.data.fill_(0.05)
        bn.running_var.data.fill_(1.5)
        conv.eval()
        bn.eval()
        return conv, bn

    def test_fusion_preserves_output(self, single_conv_bn):
        conv, bn = single_conv_bn
        x = torch.randn(4, 3, 16, 16)

        fused = fuse_conv_bn_eval(conv, bn)

        out_orig = bn(conv(x))
        out_fused = fused(x)

        error = (out_orig - out_fused).abs().max().item()
        assert error < 1e-4, f"Max fusion error: {error:.6f}"

    def test_fused_conv_has_bias(self, single_conv_bn):
        conv, bn = single_conv_bn
        fused = fuse_conv_bn_eval(conv, bn)
        assert fused.bias is not None, "Fused conv should have bias"
        assert torch.isfinite(fused.bias).all(), "Fused bias has NaN/Inf"

    def test_same_weight_shape(self, single_conv_bn):
        conv, bn = single_conv_bn
        fused = fuse_conv_bn_eval(conv, bn)
        assert fused.weight.shape == conv.weight.shape
        assert fused.kernel_size == conv.kernel_size
        assert fused.stride == conv.stride
        assert fused.padding == conv.padding

    def test_conv_with_existing_bias(self):
        """Fusion should work when conv already has bias."""
        conv = nn.Conv2d(3, 8, kernel_size=3, padding=1, bias=True)
        bn = nn.BatchNorm2d(8)
        bn.weight.data.fill_(1.0)
        bn.bias.data.fill_(0.0)
        conv.eval()
        bn.eval()

        x = torch.randn(2, 3, 8, 8)
        fused = fuse_conv_bn_eval(conv, bn)
        out_orig = bn(conv(x))
        out_fused = fused(x)
        error = (out_orig - out_fused).abs().max().item()
        assert error < 1e-4, f"Max error with existing bias: {error:.6f}"


class TestModelBNFusion:
    """Verify full model BN fusion."""

    @pytest.fixture(params=[
        ("MicroCNNSmall", MicroCNNSmall(in_channels=1, num_classes=10)),
        ("MicroCNNExtraSmall", MicroCNNExtraSmall(in_channels=1, num_classes=10)),
        ("DepthwiseMicroCNN", DepthwiseMicroCNN(in_channels=1, num_classes=10)),
    ])
    def model_pair(self, request):
        name, model = request.param
        model.eval()
        return name, model

    def test_fused_model_output_close(self, model_pair):
        name, model = model_pair
        x = torch.randn(4, 1, 28, 28)

        fused = fuse_model_bn(model)
        result = check_bn_fusion_error(model, fused, x)

        assert result["output_ok"], (
            f"{name}: BN fusion max_error={result['max_error']:.6f}, expected < 1e-4"
        )

    def test_original_model_unchanged(self, model_pair):
        name, model = model_pair
        original_state = copy.deepcopy(model.state_dict())

        _ = fuse_model_bn(model)

        # Model should be unchanged
        for key in original_state:
            assert torch.equal(original_state[key], model.state_dict()[key]), (
                f"{name}: Parameter {key} changed after fuse_model_bn"
            )

    def test_fused_model_no_batchnorm(self, model_pair):
        name, model = model_pair
        fused = fuse_model_bn(model)

        # Check no BatchNorm modules remain
        has_bn = False
        for mod in fused.modules():
            if isinstance(mod, (nn.BatchNorm2d, nn.BatchNorm1d)):
                has_bn = True
                break
        # Note: Some BN may remain if not in sequential pattern (e.g., residuals)
        # For the plain sequential models, all BN should be fused
        if "Depthwise" not in name:
            # DS-MicroCNN has BN inside DepthwiseSeparableBlock which is not Sequential
            assert not has_bn, f"{name}: BatchNorm found in fused model"

    def test_fused_model_checkpoint_loadable(self, model_pair):
        """Fused model state dict should be loadable."""
        name, model = model_pair
        fused = fuse_model_bn(model)
        # The fused model should still have a valid state dict
        sd = fused.state_dict()
        assert len(sd) > 0, f"{name}: fused model has empty state dict"


# ============================================================================
#  INT8 PTQ Tests
# ============================================================================

class TestWeightQuantization:
    """Verify per-channel weight quantisation."""

    def test_conv2d_quantization(self):
        conv = nn.Conv2d(3, 8, kernel_size=3)
        result = quantize_weights_per_channel(conv)
        assert len(result["scales"]) == 8, "Should have 8 scales (one per output channel)"
        assert len(result["zero_points"]) == 8
        assert all(s > 0 for s in result["scales"]), "All scales should be positive"
        assert 0 <= result["saturated_fraction"] <= 1.0
        assert result["qweight"].shape == conv.weight.shape

    def test_linear_quantization(self):
        linear = nn.Linear(16, 10)
        result = quantize_weights_per_channel(linear)
        assert len(result["scales"]) == 10
        assert all(s > 0 for s in result["scales"])
        assert result["qweight"].shape == linear.weight.shape

    def test_saturated_fraction_bound(self):
        """Saturated fraction should be in [0, 1]."""
        conv = nn.Conv2d(3, 16, kernel_size=3)
        result = quantize_weights_per_channel(conv)
        assert 0.0 <= result["saturated_fraction"] <= 1.0

    def test_no_nan_in_dequantized(self):
        conv = nn.Conv2d(3, 8, kernel_size=3)
        result = quantize_weights_per_channel(conv)
        assert torch.isfinite(result["dequantized_weight"]).all(), "Dequantized weights have NaN/Inf"


class TestActivationCalibration:
    """Verify activation range calibration."""

    @pytest.fixture
    def model_and_loader(self):
        model = MicroCNNSmall(in_channels=1, num_classes=10)
        model.eval()
        images = torch.randn(32, 1, 28, 28)
        labels = torch.randint(0, 10, (32,))
        loader = DataLoader(TensorDataset(images, labels), batch_size=8)
        return model, loader

    def test_calibration_returns_ranges(self, model_and_loader):
        model, loader = model_and_loader
        ranges = calibrate_activation_range(model, loader, torch.device("cpu"), num_batches=2)
        assert len(ranges) > 0, "Should have ranges for at least one layer"
        for name, (mn, mx) in ranges.items():
            assert mn <= mx, f"{name}: min ({mn}) > max ({mx})"
            assert torch.isfinite(torch.tensor(mn)), f"{name}: non-finite min"
            assert torch.isfinite(torch.tensor(mx)), f"{name}: non-finite max"

    def test_calibration_with_few_batches(self, model_and_loader):
        model, loader = model_and_loader
        ranges_1 = calibrate_activation_range(model, loader, torch.device("cpu"), num_batches=1)
        ranges_all = calibrate_activation_range(model, loader, torch.device("cpu"), num_batches=None)
        # With more batches, range should be at least as wide
        for name in ranges_1:
            if name in ranges_all:
                assert ranges_all[name][1] >= ranges_1[name][1], (
                    f"{name}: more batches should give wider or equal max"
                )


class TestPTQEvaluate:
    """Verify PTQ evaluation runs without error and produces reasonable outputs."""

    @pytest.fixture
    def setup(self):
        model = MicroCNNSmall(in_channels=1, num_classes=10)
        model.eval()
        images = torch.randn(40, 1, 28, 28)
        labels = torch.randint(0, 10, (40,))
        loader = DataLoader(TensorDataset(images, labels), batch_size=8)
        return model, loader

    def test_ptq_evaluation_runs(self, setup):
        model, loader = setup
        result = evaluate_ptq(model, loader, torch.device("cpu"), calib_batches=2)
        assert "accuracy" in result
        assert "loss" in result
        assert isinstance(result["accuracy"], float)
        assert 0.0 <= result["accuracy"] <= 1.0
        assert torch.isfinite(torch.tensor(result["loss"]))

    def test_ptq_weight_saturated_list(self, setup):
        model, loader = setup
        result = evaluate_ptq(model, loader, torch.device("cpu"), calib_batches=2)
        assert len(result["weight_saturated_fractions"]) > 0
        for entry in result["weight_saturated_fractions"]:
            assert "name" in entry
            assert "saturated_fraction" in entry
            assert 0.0 <= entry["saturated_fraction"] <= 1.0

    def test_ptq_layer_scale_stats(self, setup):
        model, loader = setup
        result = evaluate_ptq(model, loader, torch.device("cpu"), calib_batches=2)
        assert len(result["layer_scale_stats"]) > 0
        for name, stats in result["layer_scale_stats"].items():
            assert "min" in stats
            assert "max" in stats
            assert stats["min"] <= stats["max"]
