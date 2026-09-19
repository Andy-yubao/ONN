"""Compatibility and spike-preservation checks for linear TTFS compression."""

import pytest
import torch

from .device_latency_encoder import (
    DeviceLatencyEncoder,
    quantile_boundaries_from_histogram,
    quantile_map_first_spike_times,
    quantize_first_spike_times,
)
from experiments.snn.conv_small.model import DeviceIFConvSmall


def original_24_times(pixels):
    """Frozen reference of the pre-quantization default device calculation."""
    pixels = pixels.clamp(0, 1)
    delta = 0.35 - 0.10
    can_fire = 0.90 * pixels > delta
    safe = pixels.clamp_min(torch.finfo(pixels.dtype).eps)
    ratio = (delta / (0.90 * safe)).clamp(max=1.0 - 1e-7)
    discrete = torch.ceil(-5.0 * torch.log1p(-ratio)).long() - 1
    valid = can_fire & (discrete >= 0) & (discrete < 24)
    return torch.where(valid, discrete, torch.full_like(discrete, -1))


@pytest.mark.parametrize("target", [24, 12, 8, 5, 4, 3])
def test_spike_mask_count_monotonicity_and_shape(target):
    generator = torch.Generator().manual_seed(17)
    pixels = torch.cat((torch.rand(4096, generator=generator),
                        torch.tensor([0., .2, .2777778, .279, .281, .3, 1.])))
    pixels = pixels.sort().values
    original = original_24_times(pixels)
    encoder = DeviceLatencyEncoder(quantized_time_steps=target)
    times = encoder.first_spike_times(pixels)
    assert times.shape == pixels.shape
    assert torch.equal(times >= 0, original >= 0)
    assert torch.equal(encoder.encode(pixels).sum(0), (original >= 0).long())
    assert encoder.encode(pixels).shape == (target, pixels.numel())
    assert (times[times >= 0] < target).all()
    # Sorted intensity means decreasing latency; ties after quantization are OK.
    firing_times = times[times >= 0]
    assert (firing_times[1:] <= firing_times[:-1]).all()
    if target == 24:
        assert torch.equal(times, original)
        assert torch.equal(DeviceLatencyEncoder().first_spike_times(pixels), original)


@pytest.mark.parametrize("target,mac", [(12, 1056768), (8, 704512), (5, 440320),
                                         (4, 352256), (3, 264192)])
def test_all_time_bins_and_model_forward(target, mac):
    original = torch.arange(-1, 24)
    mapped = quantize_first_spike_times(original, 24, target)
    assert mapped[0] == -1
    assert mapped[1] == 0 and mapped[-1] == target - 1
    assert (mapped[2:] >= mapped[1:-1]).all()
    assert torch.equal(mapped[1:].unique(), torch.arange(target))
    encoder = DeviceLatencyEncoder(quantized_time_steps=target)
    pixels = torch.rand(2, 1, 8, 8, generator=torch.Generator().manual_seed(17))
    model = DeviceIFConvSmall()
    logits = model(encoder.first_spike_times(pixels), target)
    assert logits.shape == (2, 10) and torch.isfinite(logits).all()
    assert model.parameter_count() == 9872
    assert model.dense_equivalent_macs_per_step() == 88064
    assert model.dense_equivalent_macs_per_step() * target == mac


@pytest.mark.parametrize("target", [0, -1, 25])
def test_reject_invalid_target(target):
    with pytest.raises(ValueError):
        DeviceLatencyEncoder(quantized_time_steps=target)


@pytest.mark.parametrize("target", [4, 5, 8])
def test_fixed_quantile_mapping_is_monotonic_and_spike_preserving(target):
    histogram = torch.tensor([40, 120, 80, 35, 20, 10, 5, 3] + [0] * 16)
    boundaries = quantile_boundaries_from_histogram(histogram, target)
    source = torch.arange(-1, 24)
    mapped = quantile_map_first_spike_times(source, boundaries)
    assert len(boundaries) == target - 1
    assert mapped[0] == -1
    assert torch.equal(mapped >= 0, source >= 0)
    assert (mapped[2:] >= mapped[1:-1]).all()
    assert mapped.max() < target
    encoder = DeviceLatencyEncoder(
        quantized_time_steps=target,
        time_mapping="quantile",
        quantile_boundaries=boundaries,
    )
    pixels = torch.rand(128, generator=torch.Generator().manual_seed(17))
    times = encoder.first_spike_times(pixels)
    assert torch.equal(encoder.encode(pixels).sum(0), (times >= 0).long())


def test_quantile_encoder_requires_fixed_training_boundaries():
    with pytest.raises(ValueError, match="boundaries"):
        DeviceLatencyEncoder(quantized_time_steps=4, time_mapping="quantile")
