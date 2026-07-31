"""Test model output shapes, residual blocks, gradient flow, and compact models."""

import pytest
import torch

from onn_model.models.baseline_cnn import BaselineCNN
from onn_model.models.tiny_resnet import TinyResNet, BasicBlock
from onn_model.models.compact_cnn import MicroCNNSmall, MicroCNNExtraSmall, DepthwiseMicroCNN, DepthwiseSeparableBlock


# ============================================================================
#  BaselineCNN
# ============================================================================

class TestBaselineCNN:
    @pytest.fixture
    def model(self):
        return BaselineCNN(num_classes=10)

    @pytest.fixture
    def input_batch(self):
        return torch.randn(2, 1, 28, 28)

    def test_output_shape(self, model, input_batch):
        output = model(input_batch)
        assert output.shape == (2, 10), f"Expected (2, 10), got {output.shape}"

    def test_forward_backward(self, model, input_batch):
        criterion = torch.nn.CrossEntropyLoss()
        output = model(input_batch)
        loss = criterion(output, torch.randint(0, 10, (2,)))
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}"
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        optimizer.step()
        assert torch.isfinite(loss).all(), "Loss is not finite"

    def test_no_nan_or_inf_output(self, model, input_batch):
        output = model(input_batch)
        assert torch.isfinite(output).all(), "Output contains NaN or Inf"


# ============================================================================
#  TinyResNet
# ============================================================================

class TestTinyResNet:
    @pytest.fixture
    def model(self):
        return TinyResNet(num_classes=10)

    @pytest.fixture
    def input_batch(self):
        return torch.randn(2, 1, 28, 28)

    def test_output_shape(self, model, input_batch):
        output = model(input_batch)
        assert output.shape == (2, 10), f"Expected (2, 10), got {output.shape}"

    def test_forward_backward(self, model, input_batch):
        criterion = torch.nn.CrossEntropyLoss()
        output = model(input_batch)
        loss = criterion(output, torch.randint(0, 10, (2,)))
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}"
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        optimizer.step()
        assert torch.isfinite(loss).all(), "Loss is not finite"

    def test_no_nan_or_inf_output(self, model, input_batch):
        output = model(input_batch)
        assert torch.isfinite(output).all(), "Output contains NaN or Inf"


# ============================================================================
#  BasicBlock
# ============================================================================

class TestBasicBlock:
    def test_same_shape(self):
        block = BasicBlock(16, 16, stride=1)
        x = torch.randn(2, 16, 14, 14)
        out = block(x)
        assert out.shape == (2, 16, 14, 14), f"Expected (2, 16, 14, 14), got {out.shape}"

    def test_stride_2(self):
        block = BasicBlock(16, 32, stride=2)
        x = torch.randn(2, 16, 14, 14)
        out = block(x)
        assert out.shape == (2, 32, 7, 7), f"Expected (2, 32, 7, 7), got {out.shape}"

    def test_channel_change(self):
        block = BasicBlock(16, 32, stride=1)
        x = torch.randn(2, 16, 14, 14)
        out = block(x)
        assert out.shape == (2, 32, 14, 14), f"Expected (2, 32, 14, 14), got {out.shape}"

    def test_shortcut_gradient(self):
        block = BasicBlock(16, 32, stride=2)
        x = torch.randn(2, 16, 14, 14, requires_grad=True)
        out = block(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None, "Input gradient is None"
        assert torch.isfinite(x.grad).all(), "Input gradient contains NaN/Inf"
        shortcut_has_grad = False
        for p in block.shortcut.parameters():
            if p.requires_grad and p.grad is not None:
                shortcut_has_grad = True
                break
        assert shortcut_has_grad, "Shortcut path has no gradients"


# ============================================================================
#  Compact Model Tests
# ============================================================================

class TestMicroCNNSmall:
    @pytest.fixture
    def model(self):
        return MicroCNNSmall(in_channels=1, num_classes=10)

    @pytest.fixture
    def input_batch(self):
        return torch.randn(2, 1, 28, 28)

    def test_output_shape(self, model, input_batch):
        output = model(input_batch)
        assert output.shape == (2, 10), f"Expected (2, 10), got {output.shape}"

    def test_forward_backward(self, model, input_batch):
        criterion = torch.nn.CrossEntropyLoss()
        output = model(input_batch)
        loss = criterion(output, torch.randint(0, 10, (2,)))
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}"
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        optimizer.step()
        assert torch.isfinite(loss).all(), "Loss is not finite"

    def test_no_nan_or_inf_output(self, model, input_batch):
        output = model(input_batch)
        assert torch.isfinite(output).all(), "Output contains NaN or Inf"

    def test_get_activity_targets(self, model):
        targets = model.get_activity_targets()
        expected_keys = {"stem_output", "conv2_output", "conv3_output"}
        assert expected_keys.issubset(targets.keys()), f"Expected {expected_keys}, got {set(targets.keys())}"
        for key, mod in targets.items():
            assert isinstance(mod, torch.nn.ReLU), f"{key} should be ReLU, got {type(mod)}"

    def test_configurable_channels(self):
        model_1ch = MicroCNNSmall(in_channels=1, num_classes=10)
        out_1ch = model_1ch(torch.randn(2, 1, 28, 28))
        assert out_1ch.shape == (2, 10)
        # Verify stem conv changes channels
        first_conv = model_1ch.stem[0]
        assert first_conv.in_channels == 1
        assert first_conv.out_channels == 8

    def test_configurable_classes(self):
        model_3cls = MicroCNNSmall(in_channels=1, num_classes=3)
        out = model_3cls(torch.randn(2, 1, 28, 28))
        assert out.shape == (2, 3), f"Expected (2, 3), got {out.shape}"


class TestMicroCNNExtraSmall:
    @pytest.fixture
    def model(self):
        return MicroCNNExtraSmall(in_channels=1, num_classes=10)

    @pytest.fixture
    def input_batch(self):
        return torch.randn(2, 1, 28, 28)

    def test_output_shape(self, model, input_batch):
        output = model(input_batch)
        assert output.shape == (2, 10), f"Expected (2, 10), got {output.shape}"

    def test_forward_backward(self, model, input_batch):
        criterion = torch.nn.CrossEntropyLoss()
        output = model(input_batch)
        loss = criterion(output, torch.randint(0, 10, (2,)))
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}"
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        optimizer.step()
        assert torch.isfinite(loss).all(), "Loss is not finite"

    def test_no_nan_or_inf_output(self, model, input_batch):
        output = model(input_batch)
        assert torch.isfinite(output).all(), "Output contains NaN or Inf"

    def test_get_activity_targets(self, model):
        targets = model.get_activity_targets()
        expected_keys = {"stem_output", "conv2_output", "conv3_output"}
        assert expected_keys.issubset(targets.keys())

    def test_configurable_channels(self):
        model_3ch = MicroCNNExtraSmall(in_channels=3, num_classes=10)
        out = model_3ch(torch.randn(2, 3, 28, 28))
        assert out.shape == (2, 10)
        assert model_3ch.stem[0].in_channels == 3


class TestDepthwiseMicroCNN:
    @pytest.fixture
    def model(self):
        return DepthwiseMicroCNN(in_channels=1, num_classes=10)

    @pytest.fixture
    def input_batch(self):
        return torch.randn(2, 1, 28, 28)

    def test_output_shape(self, model, input_batch):
        output = model(input_batch)
        assert output.shape == (2, 10), f"Expected (2, 10), got {output.shape}"

    def test_forward_backward(self, model, input_batch):
        criterion = torch.nn.CrossEntropyLoss()
        output = model(input_batch)
        loss = criterion(output, torch.randint(0, 10, (2,)))
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}"
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        optimizer.step()
        assert torch.isfinite(loss).all(), "Loss is not finite"

    def test_no_nan_or_inf_output(self, model, input_batch):
        output = model(input_batch)
        assert torch.isfinite(output).all(), "Output contains NaN or Inf"

    def test_get_activity_targets(self, model):
        targets = model.get_activity_targets()
        expected_keys = {"stem_output", "ds_block1.output", "ds_block2.output"}
        assert expected_keys.issubset(targets.keys()), f"Expected {expected_keys}, got {set(targets.keys())}"

    def test_configurable_channels(self):
        model_3ch = DepthwiseMicroCNN(in_channels=3, num_classes=10)
        out = model_3ch(torch.randn(2, 3, 28, 28))
        assert out.shape == (2, 10)
        assert model_3ch.stem[0].in_channels == 3

    def test_depthwise_separable_block(self):
        block = DepthwiseSeparableBlock(8, 16, stride=2)
        x = torch.randn(2, 8, 14, 14)
        out = block(x)
        assert out.shape == (2, 16, 7, 7), f"Expected (2, 16, 7, 7), got {out.shape}"

        block_1 = DepthwiseSeparableBlock(16, 16, stride=1)
        x = torch.randn(2, 16, 7, 7)
        out = block_1(x)
        assert out.shape == (2, 16, 7, 7), f"Expected (2, 16, 7, 7), got {out.shape}"


# ============================================================================
#  Model Parameter & MAC Tests
# ============================================================================

class TestModelSizeRelations:
    """Verify relative sizes of models."""

    def test_micro_cnn_s_smaller_than_baseline(self):
        baseline = BaselineCNN()
        micro_s = MicroCNNSmall()
        b_params = sum(p.numel() for p in baseline.parameters())
        s_params = sum(p.numel() for p in micro_s.parameters())
        assert s_params < b_params, (
            f"MicroCNN-S ({s_params}) should have fewer params than BaselineCNN ({b_params})"
        )

    def test_micro_cnn_xs_smaller_than_micro_s(self):
        micro_s = MicroCNNSmall()
        micro_xs = MicroCNNExtraSmall()
        s_params = sum(p.numel() for p in micro_s.parameters())
        xs_params = sum(p.numel() for p in micro_xs.parameters())
        assert xs_params < s_params, (
            f"MicroCNN-XS ({xs_params}) should have fewer params than MicroCNN-S ({s_params})"
        )

    @pytest.fixture
    def dummy(self):
        return torch.randn(1, 1, 28, 28)

    def test_profile_basic(self, dummy):
        """All models should profile with positive MACs and params."""
        from onn_model.profiling import profile_model

        models = [
            ("MicroCNNSmall", MicroCNNSmall()),
            ("MicroCNNExtraSmall", MicroCNNExtraSmall()),
            ("DepthwiseMicroCNN", DepthwiseMicroCNN()),
        ]
        for name, model in models:
            result = profile_model(model, (1, 1, 28, 28))
            assert result["total_params"] > 0, f"{name} params should be > 0"
            assert result["total_macs"] > 0, f"{name} MACs should be > 0"
            assert len(result["layer_details"]) > 0, f"{name} should have layer details"
            assert result["model_name"] == name, f"Expected model_name={name}, got {result['model_name']}"

    def test_fpga_proxy_present(self):
        """FPGA proxy fields should be present in profile output."""
        from onn_model.profiling import profile_model

        model = MicroCNNSmall()
        result = profile_model(model, (1, 1, 28, 28))
        fpga = result.get("fpga_proxy", {})
        assert "total_weight_bytes_int8" in fpga
        assert "peak_activation_bytes_int8" in fpga
        assert "estimated_ping_pong_activation_bytes_int8" in fpga
        assert fpga["total_weight_bytes_int8"] > 0
        assert fpga["peak_activation_bytes_int8"] > 0

    def test_ds_micro_cnn_macs_less_than_micro_s(self):
        """DS-MicroCNN should have fewer MACs than MicroCNN-S."""
        from onn_model.profiling import profile_model

        s_result = profile_model(MicroCNNSmall(), (1, 1, 28, 28))
        ds_result = profile_model(DepthwiseMicroCNN(), (1, 1, 28, 28))
        assert ds_result["total_macs"] < s_result["total_macs"], (
            f"DS-MicroCNN MACs ({ds_result['total_macs']}) should be < "
            f"MicroCNN-S MACs ({s_result['total_macs']})"
        )


# ============================================================================
#  Model Counts
# ============================================================================

class TestModelCounts:
    def test_baseline_cnn_reasonable_size(self):
        model = BaselineCNN()
        total = sum(p.numel() for p in model.parameters())
        assert 10_000 < total < 500_000, f"BaselineCNN params {total} out of expected range"

    def test_tiny_resnet_reasonable_size(self):
        model = TinyResNet()
        total = sum(p.numel() for p in model.parameters())
        assert 10_000 < total < 500_000, f"TinyResNet params {total} out of expected range"

    def test_micro_cnn_s_reasonable_size(self):
        model = MicroCNNSmall()
        total = sum(p.numel() for p in model.parameters())
        assert 1_000 < total < 100_000, f"MicroCNN-S params {total} out of expected range"

    def test_micro_cnn_xs_reasonable_size(self):
        model = MicroCNNExtraSmall()
        total = sum(p.numel() for p in model.parameters())
        assert 500 < total < 50_000, f"MicroCNN-XS params {total} out of expected range"

    def test_ds_micro_cnn_reasonable_size(self):
        model = DepthwiseMicroCNN()
        total = sum(p.numel() for p in model.parameters())
        assert 500 < total < 50_000, f"DS-MicroCNN params {total} out of expected range"
