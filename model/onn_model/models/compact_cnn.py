"""Compact CNN family — MicroCNN-S, MicroCNN-XS, DS-MicroCNN.

Designed for FPGA deployment on Cyclone IV EP4CE10F17C8.
All models support ``in_channels`` and ``num_classes`` for future compatibility.
"""

from typing import Any

import torch
from torch import nn


class MicroCNNSmall(nn.Module):
    """MicroCNN-S: primary compact deployment candidate.

    Architecture (channels halved vs BaselineCNN):
      Conv 3×3, 1→8 → BN → ReLU → MP(2×2)   [8×14×14]
      Conv 3×3, 8→16 → BN → ReLU → MP(2×2)  [16×7×7]
      Conv 3×3, 16→16 → BN → ReLU            [16×7×7]
      AdaptiveAvgPool → Linear(16→10)

    Expected: ~3.8K params, ~0.40M MACs.
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 10) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, num_classes)

    def get_activity_targets(self) -> dict[str, nn.Module]:
        """Return post-activation modules for channel activity analysis."""
        return {
            "stem_output": self.stem[-1],
            "conv2_output": self.conv2[-1],
            "conv3_output": self.conv3[-1],
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)        # [N, 8, 28, 28]
        x = self.pool1(x)       # [N, 8, 14, 14]

        x = self.conv2(x)       # [N, 16, 14, 14]
        x = self.pool2(x)       # [N, 16, 7, 7]

        x = self.conv3(x)       # [N, 16, 7, 7]

        x = self.pool(x)        # [N, 16, 1, 1]
        x = torch.flatten(x, 1)
        x = self.fc(x)          # [N, num_classes]
        return x


class MicroCNNExtraSmall(nn.Module):
    """MicroCNN-XS: aggressive low-resource candidate.

    Architecture (channels aggressively reduced):
      Conv 3×3, 1→4 → BN → ReLU → MP(2×2)   [4×14×14]
      Conv 3×3, 4→8 → BN → ReLU → MP(2×2)   [8×7×7]
      Conv 3×3, 8→8 → BN → ReLU              [8×7×7]
      AdaptiveAvgPool → Linear(8→10)

    Expected: ~1.0K params, ~0.113M MACs.
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 10) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(8, num_classes)

    def get_activity_targets(self) -> dict[str, nn.Module]:
        """Return post-activation modules for channel activity analysis."""
        return {
            "stem_output": self.stem[-1],
            "conv2_output": self.conv2[-1],
            "conv3_output": self.conv3[-1],
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)        # [N, 4, 28, 28]
        x = self.pool1(x)       # [N, 4, 14, 14]

        x = self.conv2(x)       # [N, 8, 14, 14]
        x = self.pool2(x)       # [N, 8, 7, 7]

        x = self.conv3(x)       # [N, 8, 7, 7]

        x = self.pool(x)        # [N, 8, 1, 1]
        x = torch.flatten(x, 1)
        x = self.fc(x)          # [N, num_classes]
        return x


class DepthwiseMicroCNN(nn.Module):
    """DS-MicroCNN: depthwise-separable CNN for extreme MAC reduction.

    Architecture:
      Stem: Conv 3×3, 1→8 → BN → ReLU → MP(2×2)   [8×14×14]
      DS-Block1: DW(8→8) → BN → ReLU → PW(8→16) → BN → ReLU → MP(2×2)  [16×7×7]
      DS-Block2: DW(16→16) → BN → ReLU → PW(16→16) → BN → ReLU          [16×7×7]
      AdaptiveAvgPool → Linear(16→10)

    Expected: ~1.0K params, ~0.12M MACs.

    NOTE: Depthwise convolutions have lower theoretical MACs but may not
    achieve proportional energy savings on Cyclone IV due to DSP under-
    utilisation, pointwise conv overhead, and limited M9K memory bandwidth.
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 10) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
        )
        self.stem_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Depthwise-Separable Block 1
        self.ds_block1 = DepthwiseSeparableBlock(8, 16, stride=2)

        # Depthwise-Separable Block 2 (no spatial downsampling)
        self.ds_block2 = DepthwiseSeparableBlock(16, 16, stride=1)

        # Head
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, num_classes)

    def get_activity_targets(self) -> dict[str, nn.Module]:
        """Return post-activation modules for channel activity analysis.

        Returns the outputs of the main stages: stem, ds_block1 output,
        ds_block2 output.  Internal depthwise outputs are available as
        optional debug targets via ``named_modules()``.
        """
        return {
            "stem_output": self.stem[-1],
            "ds_block1.output": self.ds_block1.relu_out,
            "ds_block2.output": self.ds_block2.relu_out,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)            # [N, 8, 28, 28]
        x = self.stem_pool(x)       # [N, 8, 14, 14]

        x = self.ds_block1(x)       # [N, 16, 7, 7]
        x = self.ds_block2(x)       # [N, 16, 7, 7]

        x = self.pool(x)            # [N, 16, 1, 1]
        x = torch.flatten(x, 1)
        x = self.fc(x)              # [N, num_classes]
        return x


class DepthwiseSeparableBlock(nn.Module):
    """Depthwise-separable convolution block.

    Depthwise 3×3 → BN → ReLU → Pointwise 1×1 → BN → ReLU
    With optional MaxPool at the end (for spatial downsampling).
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=1,
            padding=1, groups=in_channels, bias=False,
        )
        self.bn_dw = nn.BatchNorm2d(in_channels)
        self.relu_dw = nn.ReLU(inplace=True)

        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, bias=False,
        )
        self.bn_pw = nn.BatchNorm2d(out_channels)
        self.relu_out = nn.ReLU(inplace=True)

        self.pool: nn.Module
        if stride == 2:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        else:
            self.pool = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.bn_dw(x)
        x = self.relu_dw(x)

        x = self.pointwise(x)
        x = self.bn_pw(x)
        x = self.relu_out(x)

        x = self.pool(x)
        return x
