"""Test model output shapes, residual blocks, and gradient flow."""

import pytest
import torch

from onn_model.models.baseline_cnn import BaselineCNN
from onn_model.models.tiny_resnet import TinyResNet, BasicBlock


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
        # Check gradients exist and are finite
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}"
        # Check parameters update
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        optimizer.step()
        # Loss was finite
        assert torch.isfinite(loss).all(), "Loss is not finite"

    def test_no_nan_or_inf_output(self, model, input_batch):
        output = model(input_batch)
        assert torch.isfinite(output).all(), "Output contains NaN or Inf"


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


class TestBasicBlock:
    """Verify residual block shapes and shortcut correctness."""

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
        """Confirm shortcut path has gradients."""
        block = BasicBlock(16, 32, stride=2)
        x = torch.randn(2, 16, 14, 14, requires_grad=True)
        out = block(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None, "Input gradient is None"
        assert torch.isfinite(x.grad).all(), "Input gradient contains NaN/Inf"
        # Check shortcut conv has gradients
        shortcut_has_grad = False
        for p in block.shortcut.parameters():
            if p.requires_grad and p.grad is not None:
                shortcut_has_grad = True
                break
        assert shortcut_has_grad, "Shortcut path has no gradients"


class TestModelCounts:
    """Sanity checks on model sizes."""

    def test_baseline_cnn_reasonable_size(self):
        model = BaselineCNN()
        total = sum(p.numel() for p in model.parameters())
        assert 10_000 < total < 500_000, f"BaselineCNN params {total} out of expected range"

    def test_tiny_resnet_reasonable_size(self):
        model = TinyResNet()
        total = sum(p.numel() for p in model.parameters())
        assert 10_000 < total < 500_000, f"TinyResNet params {total} out of expected range"
