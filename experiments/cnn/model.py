"""First CNN candidate with the small SNN's convolution dimensions."""

import torch
from torch import nn


class MatchedConvSmall(nn.Module):
    """8x8 image -> Conv16/ReLU -> Conv32 stride2/ReLU -> Linear10."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=1, bias=True)
        self.readout = nn.Linear(512, 10, bias=True)
        self.relu = nn.ReLU()
        nn.init.kaiming_uniform_(self.conv1.weight, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.conv2.weight, nonlinearity="relu")
        nn.init.xavier_uniform_(self.readout.weight)
        for layer in (self.conv1, self.conv2, self.readout):
            nn.init.zeros_(layer.bias)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1:] != (1, 8, 8):
            raise ValueError("images must have shape [B, 1, 8, 8]")
        hidden = self.relu(self.conv1(images))
        hidden = self.relu(self.conv2(hidden))
        return self.readout(hidden.flatten(1))

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def dense_equivalent_macs_per_sample(self) -> int:
        return 8 * 8 * 16 * 9 + 4 * 4 * 32 * 16 * 9 + 512 * 10
