"""Tiny-ResNet — a shallow residual network for MNIST and FPGA deployment.

Architecture:
  Stem: Conv(1→16, 3×3) → BN → ReLU
  Stage 1: BasicBlock(16→16, s=1), BasicBlock(16→16, s=1)
  Stage 2: BasicBlock(16→32, s=2), BasicBlock(32→32, s=1)
  AdaptiveAvgPool → Linear(32→10)

BasicBlock:
  Conv3×3 → BN → ReLU → Conv3×3 → BN
  Shortcut: identity (same shape) or Conv1×1 → BN (shape change)
  Output: (main + shortcut) → ReLU
"""

from typing import Any, Optional

import torch
from torch import nn


class BasicBlock(nn.Module):
    """Residual block with two 3×3 convolutions.

    Parameters
    ----------
    in_channels : int
    out_channels : int
    stride : int
        Stride of the first convolution.  ``stride=2`` halves spatial size.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection
        self.shortcut: nn.Module
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)
        return out


class TinyResNet(nn.Module):
    """Shallow residual network for MNIST.

    Designed for FPGA deployment: small channel counts, no 7×7 convolutions,
    no initial MaxPool, global average pooling instead of large FC layers.
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.num_classes = num_classes

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )

        # Stage 1: 16 → 16, spatial 28×28
        self.stage1 = nn.Sequential(
            BasicBlock(16, 16, stride=1),
            BasicBlock(16, 16, stride=1),
        )

        # Stage 2: 16 → 32, spatial 14×14
        self.stage2 = nn.Sequential(
            BasicBlock(16, 32, stride=2),
            BasicBlock(32, 32, stride=1),
        )

        # Head
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, num_classes)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Kaiming normal init for Conv, constant for BN."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)         # [N, 16, 28, 28]
        x = self.stage1(x)       # [N, 16, 28, 28]
        x = self.stage2(x)       # [N, 32, 14, 14]
        x = self.pool(x)         # [N, 32, 1, 1]
        x = torch.flatten(x, 1)
        x = self.fc(x)           # [N, 10]
        return x


def tiny_resnet_config() -> dict[str, Any]:
    """Return default config for Tiny-ResNet training."""
    return {
        "model": "TinyResNet",
        "seed": 42,
        "batch_size": 128,
        "num_workers": 0,
        "epochs": 20,
        "loss": "CrossEntropyLoss",
        "optimizer": "AdamW",
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "scheduler": "CosineAnnealingLR",
        "device": "auto",
        "data_root": "model/data",
    }
