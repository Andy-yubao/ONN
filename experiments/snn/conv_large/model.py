"""Compact 8x8 convolutional SNN with a fixed, minimal leak."""

from __future__ import annotations

import torch
from torch import nn
import snntorch as snn
from snntorch import surrogate


class ConvSNN(nn.Module):
    """1x8x8 -> 32C -> stride-2 64C -> 128 LIF -> 10 membrane logits."""

    def __init__(
        self,
        conv1_channels: int = 32,
        conv2_channels: int = 64,
        hidden_size: int = 128,
        threshold: float = 1.0,
    ) -> None:
        super().__init__()
        spike_grad = surrogate.fast_sigmoid(slope=5)
        beta = 0.9
        self.conv1 = nn.Conv2d(1, conv1_channels, kernel_size=3, padding=1, bias=False)
        self.if1 = snn.Leaky(
            beta=beta, threshold=threshold, spike_grad=spike_grad, reset_mechanism="zero"
        )
        self.conv2 = nn.Conv2d(
            conv1_channels, conv2_channels, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.if2 = snn.Leaky(
            beta=beta, threshold=threshold, spike_grad=spike_grad, reset_mechanism="zero"
        )
        self.fc1 = nn.Linear(conv2_channels * 4 * 4, hidden_size, bias=False)
        self.if3 = snn.Leaky(
            beta=beta, threshold=threshold, spike_grad=spike_grad, reset_mechanism="zero"
        )
        self.fc2 = nn.Linear(hidden_size, 10, bias=False)

        nn.init.kaiming_uniform_(self.conv1.weight, nonlinearity="linear")
        nn.init.kaiming_uniform_(self.conv2.weight, nonlinearity="linear")
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity="linear")
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, first_spike_times: torch.Tensor, time_steps: int) -> torch.Tensor:
        """Consume [B,1,8,8] integer latency maps and return [B,10] logits."""
        batch_size = first_spike_times.shape[0]
        dtype = self.conv1.weight.dtype
        mem1 = self.conv1.weight.new_zeros(batch_size, self.conv1.out_channels, 8, 8)
        mem2 = self.conv1.weight.new_zeros(batch_size, self.conv2.out_channels, 4, 4)
        mem3 = self.conv1.weight.new_zeros(batch_size, self.fc1.out_features)
        output_mem = self.conv1.weight.new_zeros(batch_size, 10)

        for step in range(time_steps):
            input_spikes = (first_spike_times == step).to(dtype)
            spikes1, mem1 = self.if1(self.conv1(input_spikes), mem1)
            spikes2, mem2 = self.if2(self.conv2(spikes1), mem2)
            spikes3, mem3 = self.if3(self.fc1(spikes2.flatten(1)), mem3)
            output_mem = output_mem + self.fc2(spikes3)

        return output_mem

    def dense_equivalent_macs_per_step(self) -> int:
        conv1 = 8 * 8 * self.conv1.out_channels * self.conv1.in_channels * 3 * 3
        conv2 = 4 * 4 * self.conv2.out_channels * self.conv2.in_channels * 3 * 3
        linear1 = self.fc1.in_features * self.fc1.out_features
        linear2 = self.fc2.in_features * self.fc2.out_features
        return conv1 + conv2 + linear1 + linear2
