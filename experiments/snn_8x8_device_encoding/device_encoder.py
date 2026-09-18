"""Photoconductive time-to-first-spike encoder for 8x8 MNIST images."""

from __future__ import annotations

import math

import torch


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
    ) -> None:
        if not g0 < g_threshold < g0 + alpha:
            raise ValueError("g_threshold must lie between g0 and g0 + alpha")
        self.g0 = g0
        self.alpha = alpha
        self.tau = tau
        self.g_threshold = g_threshold
        self.time_steps = time_steps

    @property
    def minimum_firing_intensity(self) -> float:
        """Smallest intensity whose asymptotic conductance reaches threshold."""
        return (self.g_threshold - self.g0) / self.alpha

    def first_spike_times(self, pixels: torch.Tensor) -> torch.Tensor:
        """Return integer times in [0, T-1], with -1 for pixels that never fire.

        ``pixels`` may have any shape. Values are expected in [0, 1]. Time step
        zero corresponds to physical time t=1 in the discretized device model.
        """
        pixels = pixels.clamp(0.0, 1.0)
        delta = self.g_threshold - self.g0
        can_fire = self.alpha * pixels > delta
        safe_pixels = pixels.clamp_min(torch.finfo(pixels.dtype).eps)
        ratio = (delta / (self.alpha * safe_pixels)).clamp(max=1.0 - 1e-7)
        continuous_time = -self.tau * torch.log1p(-ratio)
        discrete_time = torch.ceil(continuous_time).to(torch.long) - 1
        valid = can_fire & (discrete_time >= 0) & (discrete_time < self.time_steps)
        return torch.where(valid, discrete_time, torch.full_like(discrete_time, -1))

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
