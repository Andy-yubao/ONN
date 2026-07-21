"""Test data splitting — sizes, overlap, reproducibility."""

import tempfile
from pathlib import Path

import pytest
import torch

from onn_model.data import split_train_val, get_mnist_dataset


@pytest.fixture(scope="module")
def mnist_dataset():
    """Use a temp dir to avoid downloading multiple times."""
    with tempfile.TemporaryDirectory() as tmp:
        dataset = get_mnist_dataset(root=tmp, train=True, download=True)
        yield dataset


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
