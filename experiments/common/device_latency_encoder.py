"""Photoconductive time-to-first-spike encoder for 8x8 MNIST images."""

from __future__ import annotations

import math

import torch


def quantize_first_spike_times(times: torch.Tensor, source_steps: int, target_steps: int) -> torch.Tensor:
    """Compress zero-based TTFS bins without dropping any existing spike."""
    if not 1 <= target_steps <= source_steps:
        raise ValueError("target_steps must be between 1 and source_steps")
    quantized = torch.div(times * target_steps, source_steps, rounding_mode="floor")
    return torch.where(times >= 0, quantized, torch.full_like(times, -1))


def quantile_boundaries_from_histogram(
    latency_histogram: torch.Tensor, target_steps: int
) -> tuple[int, ...]:
    """Return fixed inclusive upper cut points for equal-mass temporal bins.

    Discrete latency levels cannot be split, so bins are only approximately
    equal when one source latency contains many events. When enough occupied
    source levels exist, every target bin receives at least one such level.
    """
    if latency_histogram.ndim != 1 or latency_histogram.numel() == 0:
        raise ValueError("latency_histogram must be a non-empty 1D tensor")
    if not 1 <= target_steps <= latency_histogram.numel():
        raise ValueError("target_steps must not exceed the source histogram length")
    if (latency_histogram < 0).any():
        raise ValueError("latency_histogram counts must be non-negative")
    total = int(latency_histogram.sum().item())
    if total <= 0:
        raise ValueError("latency_histogram must contain at least one event")
    histogram = latency_histogram.to(torch.int64)
    cumulative = histogram.cumsum(0)
    occupied = torch.nonzero(histogram > 0, as_tuple=False).flatten().tolist()
    if len(occupied) < target_steps:
        raise ValueError("not enough occupied source latency levels for non-empty quantile bins")
    boundaries: list[int] = []
    previous_occupied_position = -1
    for bin_index in range(1, target_steps):
        target_count = total * bin_index / target_steps
        first_position = previous_occupied_position + 1
        last_position = len(occupied) - (target_steps - bin_index) - 1
        candidate_positions = range(first_position, last_position + 1)
        best_position = min(
            candidate_positions,
            key=lambda position: abs(int(cumulative[occupied[position]].item()) - target_count),
        )
        boundary = occupied[best_position]
        boundaries.append(boundary)
        previous_occupied_position = best_position
    return tuple(boundaries)


def quantile_map_first_spike_times(
    times: torch.Tensor, boundaries: tuple[int, ...]
) -> torch.Tensor:
    """Map valid source latencies through fixed monotonic quantile cut points."""
    cuts = torch.tensor(boundaries, dtype=times.dtype, device=times.device)
    mapped = torch.bucketize(times, cuts, right=False)
    return torch.where(times >= 0, mapped, torch.full_like(times, -1))


class DeviceLatencyEncoder:
    """Encode each pixel as zero or one threshold-crossing spike."""

    def __init__(
        self,
        *,
        g0: float = 0.10,
        alpha: float = 0.90,
        tau: float = 5.0,
        g_threshold: float = 0.35,
        time_steps: int = 24,
        quantized_time_steps: int | None = None,
        time_mapping: str = "linear",
        quantile_boundaries: tuple[int, ...] | None = None,
    ) -> None:
        if not g0 < g_threshold < g0 + alpha:
            raise ValueError("g_threshold must lie between g0 and g0 + alpha")
        self.g0 = g0
        self.alpha = alpha
        self.tau = tau
        self.g_threshold = g_threshold
        if time_steps <= 0:
            raise ValueError("time_steps must be positive")
        if quantized_time_steps is not None and not 1 <= quantized_time_steps <= time_steps:
            raise ValueError("quantized_time_steps must be between 1 and time_steps")
        if time_mapping not in {"linear", "quantile"}:
            raise ValueError("time_mapping must be linear or quantile")
        self.source_time_steps = time_steps
        self.time_steps = time_steps if quantized_time_steps is None else quantized_time_steps
        self.time_mapping = time_mapping
        if time_mapping == "quantile":
            if quantile_boundaries is None or len(quantile_boundaries) != self.time_steps - 1:
                raise ValueError("quantile mapping needs target_steps - 1 boundaries")
            if any(left > right for left, right in zip(quantile_boundaries, quantile_boundaries[1:])):
                raise ValueError("quantile boundaries must be non-decreasing")
            if any(boundary < 0 or boundary >= self.source_time_steps for boundary in quantile_boundaries):
                raise ValueError("quantile boundaries must be valid source latency indices")
            self.quantile_boundaries = tuple(int(value) for value in quantile_boundaries)
        else:
            if quantile_boundaries is not None:
                raise ValueError("quantile_boundaries are only valid for quantile mapping")
            self.quantile_boundaries = None

    @property
    def minimum_firing_intensity(self) -> float:
        """Smallest intensity whose asymptotic conductance reaches threshold."""
        return (self.g_threshold - self.g0) / self.alpha

    def first_spike_times(self, pixels: torch.Tensor) -> torch.Tensor:
        """Return integer times in [0, T-1], with -1 for pixels that never fire.

        ``pixels`` may have any shape. Values are expected in [0, 1]. Time step
        Without quantization, zero corresponds to physical time t=1. With
        quantization, validity is still decided in the original source window;
        only valid first-spike times are mapped to the output bins.
        """
        pixels = pixels.clamp(0.0, 1.0)
        delta = self.g_threshold - self.g0
        can_fire = self.alpha * pixels > delta
        safe_pixels = pixels.clamp_min(torch.finfo(pixels.dtype).eps)
        ratio = (delta / (self.alpha * safe_pixels)).clamp(max=1.0 - 1e-7)
        continuous_time = -self.tau * torch.log1p(-ratio)
        discrete_time = torch.ceil(continuous_time).to(torch.long) - 1
        valid = can_fire & (discrete_time >= 0) & (discrete_time < self.source_time_steps)
        times = torch.where(valid, discrete_time, torch.full_like(discrete_time, -1))
        if self.time_steps == self.source_time_steps and self.time_mapping == "linear":
            return times
        if self.time_mapping == "linear":
            return quantize_first_spike_times(times, self.source_time_steps, self.time_steps)
        return quantile_map_first_spike_times(times, self.quantile_boundaries or ())

    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        """Return a boolean spike tensor with layout [T, ..., 64]."""
        times = self.first_spike_times(pixels)
        steps = torch.arange(self.time_steps, device=pixels.device)
        view_shape = (self.time_steps,) + (1,) * times.ndim
        return steps.view(view_shape) == times.unsqueeze(0)

    def conductance(self, pixels: torch.Tensor, step: int) -> torch.Tensor:
        """Evaluate G(t) at one one-based physical time step."""
        t = float(step + 1)
        return self.g0 + self.alpha * pixels * (1.0 - math.exp(-t / self.tau))
