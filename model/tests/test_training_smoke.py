"""Smoke tests for training pipeline and activation analysis.

These use synthetic data to avoid MNIST download dependency.
"""

import tempfile
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from onn_model.engine import (
    train_one_epoch,
    validate_one_epoch,
    run_experiment,
)
from onn_model.models.baseline_cnn import BaselineCNN
from onn_model.models.tiny_resnet import TinyResNet
from onn_model.activity import analyze_activations
from onn_model.profiling import profile_model


@pytest.fixture
def synthetic_loader():
    """Create 32 synthetic MNIST-like samples."""
    images = torch.randn(32, 1, 28, 28)
    labels = torch.randint(0, 10, (32,))
    dataset = TensorDataset(images, labels)
    return DataLoader(dataset, batch_size=8, shuffle=True)


class TestTrainingSmoke:
    def test_baseline_cnn_train_one_epoch(self, synthetic_loader):
        model = BaselineCNN()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        loss, acc = train_one_epoch(model, synthetic_loader, criterion, optimizer, torch.device("cpu"))
        assert torch.isfinite(torch.tensor(loss)), f"Loss not finite: {loss}"
        assert 0.0 <= acc <= 1.0, f"Accuracy out of range: {acc}"

    def test_tiny_resnet_train_one_epoch(self, synthetic_loader):
        model = TinyResNet()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        loss, acc = train_one_epoch(model, synthetic_loader, criterion, optimizer, torch.device("cpu"))
        assert torch.isfinite(torch.tensor(loss)), f"Loss not finite: {loss}"
        assert 0.0 <= acc <= 1.0, f"Accuracy out of range: {acc}"

    def test_validate_one_epoch(self, synthetic_loader):
        model = BaselineCNN()
        criterion = nn.CrossEntropyLoss()
        loss, acc = validate_one_epoch(model, synthetic_loader, criterion, torch.device("cpu"))
        assert torch.isfinite(torch.tensor(loss)), f"Loss not finite: {loss}"
        assert 0.0 <= acc <= 1.0, f"Accuracy out of range: {acc}"


class TestExperimentSmoke:
    """Run mini experiments with synthetic data to exercise the full pipeline."""

    def _make_minimal_config(self, model_name: str):
        return {
            "model": model_name,
            "seed": 42,
            "batch_size": 8,
            "num_workers": 0,
            "epochs": 2,
            "loss": "CrossEntropyLoss",
            "optimizer": "AdamW",
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "scheduler": "CosineAnnealingLR",
            "device": "cpu",
            "data_root": str(tempfile.mkdtemp()),  # won't be used — overridden via synthetic data
        }

    def test_baseline_cnn_smoke_experiment(self):
        config = self._make_minimal_config("BaselineCNN")
        config["epochs"] = 2
        config["batch_size"] = 8
        # Override: use a small temp dir for data so it doesn't crash on load
        with tempfile.TemporaryDirectory() as tmp:
            config["data_root"] = tmp
            state = run_experiment(config, smoke_test=True)
        # Verify state populated
        assert len(state.history["epoch"]) > 0, "No training history"
        assert state.best_val_acc >= 0.0, "Invalid best val acc"
        assert state.best_epoch >= 1, "Best epoch not recorded"

    def test_tiny_resnet_smoke_experiment(self):
        config = self._make_minimal_config("TinyResNet")
        with tempfile.TemporaryDirectory() as tmp:
            config["data_root"] = tmp
            state = run_experiment(config, smoke_test=True)
        assert len(state.history["epoch"]) > 0, "No training history"
        assert state.best_val_acc >= 0.0


class TestActivationAnalysisSmoke:
    def test_activations_no_side_effects(self, synthetic_loader):
        """Verify activation analysis doesn't change model parameters."""
        model = BaselineCNN()
        initial_params = [p.clone() for p in model.parameters()]

        results = analyze_activations(model, synthetic_loader, torch.device("cpu"))

        for p_orig, p_now in zip(initial_params, model.parameters()):
            assert torch.equal(p_orig, p_now), "Model parameters changed after activation analysis"

    def test_activations_output_shape(self, synthetic_loader):
        model = BaselineCNN()
        results = analyze_activations(model, synthetic_loader, torch.device("cpu"))
        assert len(results) > 0, "No layers analyzed"
        for name, data in results.items():
            for ch in data["channels"]:
                assert "mean_abs_activation" in ch
                assert "nonzero_ratio" in ch
                assert "zero_ratio" in ch
            assert data["channels"][-1]["channel"] == len(data["channels"]) - 1


class TestProfilingSmoke:
    def test_baseline_cnn_profile(self):
        model = BaselineCNN()
        result = profile_model(model, (1, 1, 28, 28))
        assert result["total_params"] > 0
        assert result["total_macs"] > 0
        assert len(result["layer_details"]) > 0
        assert result["model_name"] == "BaselineCNN"

    def test_tiny_resnet_profile(self):
        model = TinyResNet()
        result = profile_model(model, (1, 1, 28, 28))
        assert result["total_params"] > 0
        assert result["total_macs"] > 0
        assert len(result["layer_details"]) > 0
        assert result["model_name"] == "TinyResNet"
