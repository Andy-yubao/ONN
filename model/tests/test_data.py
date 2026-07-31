"""Test data splitting — sizes, overlap, reproducibility, split_seed/run_seed.

NOTE: These tests require MNIST to be downloaded (or already present).
They are skipped if MNIST is not available.
"""

import tempfile
from pathlib import Path

import pytest
import torch

from onn_model.data import split_train_val, get_mnist_dataset, get_train_val_loaders


@pytest.fixture(scope="module")
def mnist_dataset():
    """Use a temp dir to avoid downloading multiple times.

    Skip if MNIST download fails (e.g., no network, no local cache).
    """
    with tempfile.TemporaryDirectory() as tmp:
        try:
            dataset = get_mnist_dataset(root=tmp, train=True, download=True)
            yield dataset
        except Exception as e:
            pytest.skip(f"MNIST not available: {e}")


def test_split_sizes(mnist_dataset):
    train_sub, val_sub = split_train_val(mnist_dataset, train_size=55000, val_size=5000, seed=42)
    assert len(train_sub) == 55000, f"Train size expected 55000, got {len(train_sub)}"
    assert len(val_sub) == 5000, f"Val size expected 5000, got {len(val_sub)}"
    assert len(train_sub) + len(val_sub) == 60000


def test_no_overlap(mnist_dataset):
    train_sub, val_sub = split_train_val(mnist_dataset, seed=42)
    train_ids = set(train_sub.indices)
    val_ids = set(val_sub.indices)
    overlap = train_ids & val_ids
    assert len(overlap) == 0, f"Train and val overlap: {overlap}"


def test_reproducibility(mnist_dataset):
    train_1, val_1 = split_train_val(mnist_dataset, seed=42)
    train_2, val_2 = split_train_val(mnist_dataset, seed=42)
    assert train_1.indices == train_2.indices, "Train split not reproducible"
    assert val_1.indices == val_2.indices, "Val split not reproducible"


def test_different_seed_different_split(mnist_dataset):
    train_1, _ = split_train_val(mnist_dataset, seed=42)
    train_2, _ = split_train_val(mnist_dataset, seed=99)
    assert train_1.indices != train_2.indices, "Different seeds should give different splits"


# ============================================================================
#  split_seed / run_seed Tests
# ============================================================================

class TestSplitRunSeed:
    """Verify split_seed and run_seed operate independently."""

    @pytest.fixture(scope="class")
    def data_root(self):
        """Download MNIST once for all tests in this class."""
        with tempfile.TemporaryDirectory() as tmp:
            try:
                # Trigger download
                get_mnist_dataset(root=tmp, train=True, download=True)
                yield tmp
            except Exception as e:
                pytest.skip(f"MNIST not available: {e}")

    def test_same_split_seed_same_split(self, data_root):
        """Same split_seed should produce identical train/val indices."""
        _, val_1, _ = get_train_val_loaders(
            data_root, batch_size=128, num_workers=0,
            split_seed=42, run_seed=42,
        )
        _, val_2, _ = get_train_val_loaders(
            data_root, batch_size=128, num_workers=0,
            split_seed=42, run_seed=99,  # different run_seed
        )
        # The val dataset should be the same subset
        assert val_1.dataset.indices == val_2.dataset.indices, (
            "Same split_seed should give same val indices regardless of run_seed"
        )

    def test_different_split_seed_different_split(self, data_root):
        """Different split_seed should give different train/val indices."""
        _, val_1, _ = get_train_val_loaders(
            data_root, batch_size=128, num_workers=0,
            split_seed=42, run_seed=42,
        )
        _, val_2, _ = get_train_val_loaders(
            data_root, batch_size=128, num_workers=0,
            split_seed=99, run_seed=42,
        )
        assert val_1.dataset.indices != val_2.dataset.indices, (
            "Different split_seed should give different val indices"
        )

    def test_same_run_seed_same_shuffle_order(self, data_root):
        """Same run_seed should produce same DataLoader shuffle order.

        This test checks the first batch order for determinism.
        """
        # Use a small batch size to see ordering differences
        train_1, _, _ = get_train_val_loaders(
            data_root, batch_size=32, num_workers=0,
            split_seed=42, run_seed=42,
        )
        train_2, _, _ = get_train_val_loaders(
            data_root, batch_size=32, num_workers=0,
            split_seed=42, run_seed=42,  # same run_seed
        )

        # Get first batch indices (using the underlying dataset)
        batch_1 = next(iter(train_1))
        batch_2 = next(iter(train_2))
        # The shuffled order should be the same
        # (We compare data hash since TensorDataset hashing is deterministic)
        assert torch.equal(batch_1[0], batch_2[0]), (
            "Same run_seed should produce same shuffle order"
        )

    def test_split_seed_ignored_by_train_shuffle(self, data_root):
        """Changing run_seed should change shuffle order even with same split."""
        train_1, _, _ = get_train_val_loaders(
            data_root, batch_size=32, num_workers=0,
            split_seed=42, run_seed=42,
        )
        train_2, _, _ = get_train_val_loaders(
            data_root, batch_size=32, num_workers=0,
            split_seed=42, run_seed=99,  # different run_seed
        )

        batch_1 = next(iter(train_1))
        batch_2 = next(iter(train_2))
        # The shuffled order should differ (very likely)
        # It's theoretically possible (1/(32!)) but practically impossible
        same = torch.equal(batch_1[0], batch_2[0])
        assert not same, "Different run_seed should (practically) give different shuffle order"

    def test_legacy_seed_backward_compat(self, data_root):
        """Legacy 'seed' parameter should set both split and run seeds."""
        train_legacy, val_legacy, _ = get_train_val_loaders(
            data_root, batch_size=128, num_workers=0,
            seed=42,
        )
        train_new, val_new, _ = get_train_val_loaders(
            data_root, batch_size=128, num_workers=0,
            split_seed=42, run_seed=42,
        )

        # Same split
        assert val_legacy.dataset.indices == val_new.dataset.indices, (
            "Legacy seed=42 should produce same split as split_seed=42"
        )
        # Note: shuffle order may differ because legacy didn't pass generator
        # So we only compare split indices for backward compat
