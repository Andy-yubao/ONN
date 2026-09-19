"""Shared, frozen data and seed policy for matched 8x8 experiments."""

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

PROTOCOL_ID = "mnist8x8-sequential-v1"
SEEDS = (7, 17, 27)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_loaders(data_dir: Path, batch_size: int, num_workers: int):
    transform = transforms.Compose([
        transforms.Resize((8, 8), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
    ])
    train_full = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    common = dict(batch_size=batch_size, num_workers=num_workers,
                  pin_memory=torch.cuda.is_available())
    return (
        DataLoader(Subset(train_full, range(55_000)), shuffle=True, **common),
        DataLoader(Subset(train_full, range(55_000, 60_000)), shuffle=False, **common),
        DataLoader(test_set, shuffle=False, **common),
    )
