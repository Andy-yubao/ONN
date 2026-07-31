"""Model definitions for MNIST baselines and compact candidates."""
from __future__ import annotations

from .baseline_cnn import BaselineCNN
from .tiny_resnet import TinyResNet, BasicBlock
from .compact_cnn import MicroCNNSmall, MicroCNNExtraSmall, DepthwiseMicroCNN, DepthwiseSeparableBlock

__all__ = [
    "BaselineCNN",
    "TinyResNet",
    "BasicBlock",
    "MicroCNNSmall",
    "MicroCNNExtraSmall",
    "DepthwiseMicroCNN",
    "DepthwiseSeparableBlock",
]
