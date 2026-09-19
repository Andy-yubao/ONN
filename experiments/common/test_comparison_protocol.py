"""Check real MNIST splits/preprocessing and repeatable random streams."""

from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import RandomSampler, SequentialSampler

from .comparison_protocol import SEEDS, make_loaders, set_seed


def test_fixed_split_and_input_contract():
    data_dir = Path(__file__).resolve().parents[2] / "data"
    train, validation, test = make_loaders(data_dir, 256, 0)
    assert SEEDS == (7, 17, 27)
    assert train.dataset.indices == range(0, 55_000)
    assert validation.dataset.indices == range(55_000, 60_000)
    assert len(test.dataset) == 10_000
    assert isinstance(train.sampler, RandomSampler)
    assert isinstance(validation.sampler, SequentialSampler)
    assert isinstance(test.sampler, SequentialSampler)
    for dataset in (train.dataset, validation.dataset, test.dataset):
        image, label = dataset[0]
        assert image.shape == (1, 8, 8)
        assert image.dtype == torch.float32
        assert 0 <= image.min() <= image.max() <= 1
        assert 0 <= label < 10


def test_seed_repeats_all_random_streams():
    def draw(seed):
        set_seed(seed)
        return random.random(), np.random.random(), torch.rand(4)
    first, repeat, other = draw(7), draw(7), draw(17)
    assert first[:2] == repeat[:2]
    assert torch.equal(first[2], repeat[2])
    assert not torch.equal(first[2], other[2])
    assert torch.backends.cudnn.deterministic
    assert not torch.backends.cudnn.benchmark
