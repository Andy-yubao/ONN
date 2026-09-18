"""Lightweight numerical contract for the shared device encoder."""

from __future__ import annotations

import math

import torch

from .device_latency_encoder import DeviceLatencyEncoder


def test_device_latency_encoder_contract() -> None:
    encoder = DeviceLatencyEncoder()
    assert math.isclose(encoder.minimum_firing_intensity, 0.25 / 0.90)

    pixels = torch.tensor([0.0, 0.20, 0.30, 0.60, 1.0], dtype=torch.float32)
    times = encoder.first_spike_times(pixels)

    assert times.shape == pixels.shape
    assert times[0].item() == -1
    assert times[1].item() == -1
    assert 0 <= times[2].item() < encoder.time_steps
    assert times[4].item() <= times[3].item()

    spikes = encoder.encode(pixels)
    assert spikes.shape == (encoder.time_steps, pixels.numel())
    assert spikes.sum(dim=0).tolist() == [0, 0, 1, 1, 1]
