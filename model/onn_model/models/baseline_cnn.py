"""BaselineCNN — a lightweight, FPGA-friendly CNN for MNIST.

Architecture:
  Conv(1→16, 3×3) → BN → ReLU → MP(2×2)
  Conv(16→32, 3×3) → BN → ReLU → MP(2×2)
  Conv(32→32, 3×3) → BN → ReLU
  AdaptiveAvgPool(1) → Linear(32→10)
"""

from typing import Any

import torch
from torch import nn


class BaselineCNN(nn.Module):
    """Lightweight plain CNN baseline without residual connections."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.num_classes = num_classes

        self.stem = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)       # [N, 16, 28, 28]
        x = self.pool1(x)      # [N, 16, 14, 14]

        x = self.conv2(x)      # [N, 32, 14, 14]
        x = self.pool2(x)      # [N, 32, 7, 7]

        x = self.conv3(x)      # [N, 32, 7, 7]

        x = self.pool(x)       # [N, 32, 1, 1]
        x = torch.flatten(x, 1)
        x = self.fc(x)         # [N, 10]
        return x


def baseline_cnn_config() -> dict[str, Any]:
    """Return default config for BaselineCNN training."""
    return {
        "model": "BaselineCNN",
        "seed": 42,
        "batch_size": 128,
        "num_workers": 0,
        "epochs": 15,
        "loss": "CrossEntropyLoss",
        "optimizer": "AdamW",
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "scheduler": "CosineAnnealingLR",
        "device": "auto",
        "data_root": "model/data",
    }
