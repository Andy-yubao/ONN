"""Data pipeline for MNIST with fixed train/val split."""

from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import transforms
from torchvision.datasets import MNIST

# MNIST canonical statistics
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081

# Fixed split sizes
TRAIN_SIZE = 55000
VAL_SIZE = 5000
TEST_SIZE = 10000


def _get_transform(add_augment: bool = False) -> transforms.Compose:
    """Return the standard preprocessing transform.

    First version uses only ToTensor + Normalize for a clean baseline.
    Data augmentation can be added later via ``add_augment=True``.
    """
    t = [transforms.ToTensor(), transforms.Normalize(MNIST_MEAN, MNIST_STD)]
    if add_augment:
        # Placeholder — augmentation not used for first baseline
        pass
    return transforms.Compose(t)


def get_mnist_dataset(
    root: str,
    train: bool = True,
    download: bool = True,
    add_augment: bool = False,
) -> MNIST:
    """Download (if needed) and load the MNIST dataset.

    Parameters
    ----------
    root : str
        Directory to store / read the data.
    train : bool
        ``True`` for the training split (60k), ``False`` for the test split (10k).
    download : bool
        Whether to download if not found locally.
    add_augment : bool
        Not yet implemented; reserved for future augmentation.

    Returns
    -------
    MNIST
        The raw dataset (unsplit for training, or the full test set).
    """
    return MNIST(
        root=root,
        train=train,
        download=download,
        transform=_get_transform(add_augment=add_augment),
    )


def split_train_val(
    dataset: MNIST,
    train_size: int = TRAIN_SIZE,
    val_size: int = VAL_SIZE,
    seed: int = 42,
) -> Tuple[Subset, Subset]:
    """Split the 60k training MNIST set into train (55k) and val (5k).

    The split is deterministic for a given ``seed``.  The validation set
    is never mixed with the test set.

    Parameters
    ----------
    dataset : MNIST
        The full 60k training MNIST dataset.
    train_size : int
        Number of training samples (default 55_000).
    val_size : int
        Number of validation samples (default 5_000).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    (train_subset, val_subset)
    """
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_size, val_size], generator=generator)


def get_train_val_loaders(
    root: str,
    batch_size: int = 128,
    num_workers: int = 0,
    seed: int = 42,
    download: bool = True,
    add_augment: bool = False,
    pin_memory: bool = True,
    *,
    split_seed: Optional[int] = None,
    run_seed: Optional[int] = None,
    worker_generator: Optional[torch.Generator] = None,
) -> Tuple[DataLoader, DataLoader, Dataset]:
    """Create training and validation DataLoaders.

    Parameters
    ----------
    root : str
        Path where MNIST data is stored.
    batch_size : int
    num_workers : int
    seed : int
        Legacy: sets both split_seed and run_seed when split_seed/run_seed
        are not provided.  Deprecated — use split_seed and run_seed instead.
    download : bool
    add_augment : bool
    pin_memory : bool
    split_seed : int or None
        Controls train/val split reproducibility (new preferred API).
        When provided, ``seed`` is ignored for splitting.
    run_seed : int or None
        Controls DataLoader shuffle randomness (new preferred API).
        When provided, ``seed`` is ignored for DataLoader shuffle.
    worker_generator : torch.Generator or None
        Generator for DataLoader shuffle, created from ``run_seed`` if not provided.

    Returns
    -------
    (train_loader, val_loader, full_train_dataset)
    """
    # Determine split_seed and run_seed (new API takes precedence)
    effective_split_seed = split_seed if split_seed is not None else seed
    effective_run_seed = run_seed if run_seed is not None else seed

    full_train = get_mnist_dataset(root, train=True, download=download, add_augment=add_augment)
    train_subset, val_subset = split_train_val(full_train, seed=effective_split_seed)

    g = worker_generator if worker_generator is not None else torch.Generator().manual_seed(effective_run_seed)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=g,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, full_train


def get_test_loader(
    root: str,
    batch_size: int = 128,
    num_workers: int = 0,
    download: bool = True,
    pin_memory: bool = True,
) -> DataLoader:
    """Create the test DataLoader (official 10k MNIST test set)."""
    test_dataset = get_mnist_dataset(root, train=False, download=download)
    return DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
