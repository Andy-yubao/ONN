"""Matched IF/LIF ablation of the frozen 8x8 small SNN."""

from __future__ import annotations

from typing import Callable

import torch
from torch import nn
from snntorch import surrogate


def normalized_temporal_weights(
    time_steps: int,
    beta: float,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return gentle early-time weights with the baseline total scale.

    For T=4 the unnormalized shape is ``1 + beta * (4 - t)`` for
    one-based ``t``.  Every beta is rescaled to sum to 10, so beta=1 keeps
    the historical 4:3:2:1 accumulated-membrane readout exactly.
    """
    if time_steps <= 0:
        raise ValueError("time_steps must be positive")
    if beta < 0:
        raise ValueError("beta must be non-negative")
    reverse_steps = torch.arange(time_steps - 1, -1, -1, device=device, dtype=dtype)
    raw = 1.0 + float(beta) * reverse_steps
    baseline_sum = time_steps * (time_steps + 1) / 2.0
    return raw * (baseline_sum / raw.sum())


def subtract_if(
    current: torch.Tensor,
    membrane: torch.Tensor,
    threshold: float,
    spike_fn: Callable[[torch.Tensor], torch.Tensor],
    decay: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Strict-threshold IF/LIF update, with leak before the input current."""
    integrated = decay * membrane + current
    spikes = spike_fn(integrated - threshold)
    next_membrane = integrated - threshold * spikes.detach()
    return spikes, next_membrane


def _spatial_fanout(
    height: int,
    width: int,
    *,
    kernel_size: int,
    stride: int,
    padding: int,
) -> torch.Tensor:
    """Count valid output positions reached by one input spatial position."""
    output_height = (height + 2 * padding - kernel_size) // stride + 1
    output_width = (width + 2 * padding - kernel_size) // stride + 1
    fanout = torch.zeros(height, width, dtype=torch.float32)
    for input_row in range(height):
        for input_col in range(width):
            count = 0
            for output_row in range(output_height):
                row_start = output_row * stride - padding
                row_valid = row_start <= input_row < row_start + kernel_size
                for output_col in range(output_width):
                    col_start = output_col * stride - padding
                    col_valid = col_start <= input_col < col_start + kernel_size
                    count += int(row_valid and col_valid)
            fanout[input_row, input_col] = count
    return fanout


class DeviceLIFConvSmall(nn.Module):
    """1x8x8 -> Conv16/LIF -> Conv32 stride2/LIF -> Linear10 readout.

    The hidden recurrence is kept explicit so its numerical contract is the
    documented same-step integrate, strict threshold, emit, subtract update.
    """

    def __init__(
        self,
        threshold: float = 1.0,
        readout_mode: str = "final_membrane",
        temporal_beta: float = 1.0,
        decay: float = 1.0,
    ) -> None:
        super().__init__()
        if readout_mode not in {"final_membrane", "accumulated_membrane"}:
            raise ValueError("readout_mode must be final_membrane or accumulated_membrane")
        if temporal_beta < 0:
            raise ValueError("temporal_beta must be non-negative")
        if not 0.0 < decay <= 1.0:
            raise ValueError("decay must be in (0, 1]")
        self.threshold = float(threshold)
        self.decay = float(decay)
        self.readout_mode = readout_mode
        self.temporal_beta = float(temporal_beta)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.readout = nn.Linear(32 * 4 * 4, 10, bias=False)
        self.spike_fn = surrogate.fast_sigmoid(slope=5)

        nn.init.kaiming_uniform_(self.conv1.weight, nonlinearity="linear")
        nn.init.kaiming_uniform_(self.conv2.weight, nonlinearity="linear")
        nn.init.xavier_uniform_(self.readout.weight)

        self.register_buffer(
            "conv1_spatial_fanout",
            _spatial_fanout(8, 8, kernel_size=3, stride=1, padding=1),
            persistent=False,
        )
        self.register_buffer(
            "conv2_spatial_fanout",
            _spatial_fanout(8, 8, kernel_size=3, stride=2, padding=1),
            persistent=False,
        )

    def forward(
        self,
        first_spike_times: torch.Tensor,
        time_steps: int = 24,
        *,
        collect_stats: bool = False,
        return_event_proxy: bool = False,
    ) -> (
        torch.Tensor
        | tuple[torch.Tensor, dict[str, int]]
        | tuple[torch.Tensor, torch.Tensor]
    ):
        """Consume [B,1,8,8] latency maps and return [B,10] logits.

        ``collect_stats`` adds the actual valid-connection event work count;
        it is disabled during training to avoid needless synchronization.
        """
        if first_spike_times.ndim != 4 or first_spike_times.shape[1:] != (1, 8, 8):
            raise ValueError("first_spike_times must have shape [B, 1, 8, 8]")
        if time_steps <= 0:
            raise ValueError("time_steps must be positive")

        batch_size = first_spike_times.shape[0]
        dtype = self.conv1.weight.dtype
        device = first_spike_times.device
        mem1 = self.conv1.weight.new_zeros(batch_size, 16, 8, 8)
        mem2 = self.conv1.weight.new_zeros(batch_size, 32, 4, 4)
        readout_mem = self.conv1.weight.new_zeros(batch_size, 10)
        weighted_logits = self.conv1.weight.new_zeros(batch_size, 10)
        event_proxy = self.conv1.weight.new_zeros(())
        temporal_weights = normalized_temporal_weights(
            time_steps,
            self.temporal_beta,
            device=device,
            dtype=dtype,
        )
        synaptic_additions = torch.zeros((), device=device, dtype=torch.float64)
        input_events = torch.zeros((), device=device, dtype=torch.float64)
        layer1_events = torch.zeros((), device=device, dtype=torch.float64)
        layer2_events = torch.zeros((), device=device, dtype=torch.float64)

        for step in range(time_steps):
            input_spikes = (first_spike_times == step).to(dtype)
            spikes1, mem1 = subtract_if(
                self.conv1(input_spikes), mem1, self.threshold, self.spike_fn, self.decay
            )
            spikes2, mem2 = subtract_if(
                self.conv2(spikes1), mem2, self.threshold, self.spike_fn, self.decay
            )
            current = self.readout(spikes2.flatten(1))
            readout_mem = readout_mem + current
            if self.readout_mode == "accumulated_membrane":
                weighted_logits = weighted_logits + temporal_weights[step] * current

            if return_event_proxy:
                event_proxy = event_proxy + (
                    spikes1
                    * self.conv2_spatial_fanout.to(dtype=dtype)[None, None, :, :]
                    * 32
                ).sum()
                event_proxy = event_proxy + spikes2.sum() * 10

            if collect_stats:
                input_events = input_events + input_spikes.detach().sum(dtype=torch.float64)
                layer1_events = layer1_events + spikes1.detach().sum(dtype=torch.float64)
                layer2_events = layer2_events + spikes2.detach().sum(dtype=torch.float64)
                synaptic_additions = synaptic_additions + (
                    input_spikes.detach()
                    * self.conv1_spatial_fanout.to(dtype=dtype)[None, None, :, :]
                    * 16
                ).sum()
                synaptic_additions = synaptic_additions + (
                    spikes1.detach()
                    * self.conv2_spatial_fanout.to(dtype=dtype)[None, None, :, :]
                    * 32
                ).sum()
                synaptic_additions = synaptic_additions + spikes2.detach().sum() * 10

        logits = (weighted_logits / float(time_steps)
                  if self.readout_mode == "accumulated_membrane" else readout_mem)
        if collect_stats and return_event_proxy:
            raise ValueError("collect_stats and return_event_proxy are mutually exclusive")
        if return_event_proxy:
            return logits, event_proxy / float(batch_size)
        if not collect_stats:
            return logits
        stats = {
            "effective_synaptic_additions": int(synaptic_additions.item()),
            "input_spike_events": int(input_events.item()),
            "layer1_spike_events": int(layer1_events.item()),
            "layer2_spike_events": int(layer2_events.item()),
        }
        return logits, stats

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def dense_equivalent_macs_per_step(self) -> int:
        conv1 = 8 * 8 * 16 * 1 * 3 * 3
        conv2 = 4 * 4 * 32 * 16 * 3 * 3
        readout = 512 * 10
        return conv1 + conv2 + readout

    def hidden_state_count(self) -> int:
        return 16 * 8 * 8 + 32 * 4 * 4
