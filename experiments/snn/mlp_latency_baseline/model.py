"""Minimal fully connected IF-SNN used by the experiment."""

from __future__ import annotations

import torch
from torch import nn
import snntorch as snn
from snntorch import surrogate


class MinimalIFSNN(nn.Module):
    """64 -> 128 IF neurons -> 10 non-leaky membrane readout neurons."""

    def __init__(self, hidden_size: int = 128, threshold: float = 1.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(64, hidden_size, bias=False)
        self.fc2 = nn.Linear(hidden_size, 10, bias=False)
        # snn.Leaky with beta=1 is an IF neuron (no leak).
        self.hidden_if = snn.Leaky(
            beta=1.0,
            threshold=threshold,
            spike_grad=surrogate.fast_sigmoid(slope=5),
            reset_mechanism="zero",
        )
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity="linear")
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, first_spike_times: torch.Tensor, time_steps: int) -> torch.Tensor:
        batch_size = first_spike_times.shape[0]
        hidden_mem = self.fc1.weight.new_zeros(batch_size, self.fc1.out_features)
        output_mem = self.fc1.weight.new_zeros(batch_size, 10)

        for step in range(time_steps):
            input_spikes = (first_spike_times == step).to(self.fc1.weight.dtype)
            hidden_spikes, hidden_mem = self.hidden_if(self.fc1(input_spikes), hidden_mem)
            output_mem = output_mem + self.fc2(hidden_spikes)

        return output_mem
