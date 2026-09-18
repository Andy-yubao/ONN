"""Fixed photoconductive first-spike encoder for the 8x8 SNN experiments."""

from __future__ import annotations

import torch


class DeviceLatencyEncoder:
    """Convert each 8x8 pixel into at most one first-spike event."""

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
        if time_steps <= 0:
            raise ValueError("time_steps must be positive")
        self.g0 = g0
        self.alpha = alpha
        self.tau = tau
        self.g_threshold = g_threshold
        self.time_steps = time_steps

    @property
    def minimum_firing_intensity(self) -> float:
        return (self.g_threshold - self.g0) / self.alpha

    def first_spike_times(self, pixels: torch.Tensor) -> torch.Tensor:
        """Return latency indices in [0, T-1], or -1 for no event."""
        pixels = pixels.clamp(0.0, 1.0)
        delta = self.g_threshold - self.g0
        can_fire = self.alpha * pixels > delta
        safe_pixels = pixels.clamp_min(torch.finfo(pixels.dtype).eps)
        ratio = (delta / (self.alpha * safe_pixels)).clamp(max=1.0 - 1e-7)
        continuous_time = -self.tau * torch.log1p(-ratio)
        discrete_time = torch.ceil(continuous_time).to(torch.long) - 1
        valid = can_fire & (discrete_time >= 0) & (discrete_time < self.time_steps)
        return torch.where(valid, discrete_time, torch.full_like(discrete_time, -1))

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Encode [B, 1, 8, 8] images as [T, B, 1, 8, 8] binary spikes."""
        times = self.first_spike_times(images)
        steps = torch.arange(self.time_steps, device=images.device)
        return steps.view(self.time_steps, 1, 1, 1, 1) == times.unsqueeze(0)
