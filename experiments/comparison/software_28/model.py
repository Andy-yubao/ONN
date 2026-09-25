"""Shape-matched convolutional SNN and CNN for 8x8/28x28 MNIST."""

from __future__ import annotations

import torch
from torch import nn
from snntorch import surrogate

from experiments.snn.conv_small.model import _spatial_fanout, normalized_temporal_weights


class DecayConvSNN(nn.Module):
    """Same-step convolutional SNN with subtract reset and optional membrane leak."""

    def __init__(self, image_size: int, first_stride: int, membrane_decay: float = 1.0):
        super().__init__()
        if image_size not in (8, 28) or first_stride not in (1, 2):
            raise ValueError("unsupported image shape")
        if not 0.0 < membrane_decay <= 1.0:
            raise ValueError("membrane_decay must be in (0, 1]")
        self.image_size = image_size
        self.first_stride = first_stride
        self.membrane_decay = float(membrane_decay)
        side1 = (image_size + 2 - 3) // first_stride + 1
        side2 = (side1 + 2 - 3) // 2 + 1
        self.side1, self.side2 = side1, side2
        self.conv1 = nn.Conv2d(1, 16, 3, stride=first_stride, padding=1, bias=False)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=1, bias=False)
        self.readout = nn.Linear(32 * side2 * side2, 10, bias=False)
        self.spike_fn = surrogate.fast_sigmoid(slope=5)
        nn.init.kaiming_uniform_(self.conv1.weight, nonlinearity="linear")
        nn.init.kaiming_uniform_(self.conv2.weight, nonlinearity="linear")
        nn.init.xavier_uniform_(self.readout.weight)
        self.register_buffer("fanout1", _spatial_fanout(image_size, image_size, kernel_size=3,
                             stride=first_stride, padding=1), persistent=False)
        self.register_buffer("fanout2", _spatial_fanout(side1, side1, kernel_size=3,
                             stride=2, padding=1), persistent=False)

    def forward(self, times: torch.Tensor, *, collect_stats: bool = False,
                return_event_proxy: bool = False):
        if times.ndim != 4 or times.shape[1:] != (1, self.image_size, self.image_size):
            raise ValueError("invalid input latency shape")
        if collect_stats and return_event_proxy:
            raise ValueError("stats and event proxy are mutually exclusive")
        batch = times.shape[0]
        mem1 = self.conv1.weight.new_zeros(batch, 16, self.side1, self.side1)
        mem2 = self.conv1.weight.new_zeros(batch, 32, self.side2, self.side2)
        logits = self.conv1.weight.new_zeros(batch, 10)
        proxy = self.conv1.weight.new_zeros(())
        counts = {name: 0 for name in ("input_spikes", "layer1_spikes", "layer2_spikes",
                                      "effective_synaptic_additions")}
        weights = normalized_temporal_weights(4, 0.5, device=times.device,
                                              dtype=self.conv1.weight.dtype)
        for step in range(4):
            input_spikes = (times == step).to(self.conv1.weight.dtype)
            integrated1 = self.membrane_decay * mem1 + self.conv1(input_spikes)
            spikes1 = self.spike_fn(integrated1 - 1.0)
            mem1 = integrated1 - spikes1.detach()
            integrated2 = self.membrane_decay * mem2 + self.conv2(spikes1)
            spikes2 = self.spike_fn(integrated2 - 1.0)
            mem2 = integrated2 - spikes2.detach()
            logits = logits + weights[step] * self.readout(spikes2.flatten(1))
            if return_event_proxy or collect_stats:
                work1 = (input_spikes * self.fanout1[None, None] * 16).sum()
                work2 = (spikes1 * self.fanout2[None, None] * 32).sum()
                work3 = spikes2.sum() * 10
                proxy = proxy + work2 + work3
                if collect_stats:
                    counts["input_spikes"] += int(input_spikes.sum().item())
                    counts["layer1_spikes"] += int(spikes1.sum().item())
                    counts["layer2_spikes"] += int(spikes2.sum().item())
                    counts["effective_synaptic_additions"] += int((work1 + work2 + work3).item())
        logits = logits / 4.0
        if return_event_proxy:
            return logits, proxy / batch
        if collect_stats:
            return logits, counts
        return logits

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())

    def dense_macs(self):
        per_step = self.side1**2 * 16 * 9 + self.side2**2 * 32 * 16 * 9 + self.side2**2 * 32 * 10
        return per_step * 4


class MatchedCNN(nn.Module):
    """Same convolution widths, strides and readout width as DecayConvSNN."""

    def __init__(self, image_size: int, first_stride: int):
        super().__init__()
        side1 = (image_size + 2 - 3) // first_stride + 1
        side2 = (side1 + 2 - 3) // 2 + 1
        self.side1, self.side2 = side1, side2
        self.conv1 = nn.Conv2d(1, 16, 3, stride=first_stride, padding=1, bias=False)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=1, bias=False)
        self.readout = nn.Linear(32 * side2 * side2, 10, bias=False)
        nn.init.kaiming_uniform_(self.conv1.weight, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.conv2.weight, nonlinearity="relu")
        nn.init.xavier_uniform_(self.readout.weight)

    def forward(self, images):
        return self.readout(torch.relu(self.conv2(torch.relu(self.conv1(images)))).flatten(1))

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())

    def dense_macs(self):
        return self.side1**2 * 16 * 9 + self.side2**2 * 32 * 16 * 9 + self.side2**2 * 32 * 10
