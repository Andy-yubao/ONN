"""Focused numerical checks for the implemented architecture."""

from __future__ import annotations

import torch

from .model import DeviceIFConvSmall, subtract_if

def test_subtract_if_uses_strict_threshold_and_keeps_residual() -> None:
    model = DeviceIFConvSmall()
    current = torch.tensor([[1.0, 1.1, -0.2]])
    membrane = torch.zeros_like(current)
    spikes, next_membrane = subtract_if(current, membrane, 1.0, model.spike_fn)

    assert torch.equal(spikes, torch.tensor([[0.0, 1.0, 0.0]]))
    assert torch.allclose(next_membrane, torch.tensor([[1.0, 0.1, -0.2]]))

    spikes, next_membrane = subtract_if(
        torch.tensor([[-0.2, 0.5, 0.0]]), next_membrane, 1.0, model.spike_fn
    )
    assert torch.equal(spikes, torch.tensor([[0.0, 0.0, 0.0]]))
    assert torch.allclose(next_membrane, torch.tensor([[0.8, 0.6, -0.2]]))


def test_shape_parameter_and_mac_contract() -> None:
    model = DeviceIFConvSmall()
    times = torch.full((2, 1, 8, 8), -1, dtype=torch.long)
    logits = model(times, 24)

    assert logits.shape == (2, 10)
    assert model.parameter_count() == 9_872
    assert model.hidden_state_count() == 1_536
    assert model.dense_equivalent_macs_per_step() == 88_064


def test_workload_is_zero_for_no_input_events() -> None:
    model = DeviceIFConvSmall()
    times = torch.full((2, 1, 8, 8), -1, dtype=torch.long)
    logits, stats = model(times, 24, collect_stats=True)

    assert torch.equal(logits, torch.zeros_like(logits))
    assert stats == {
        "effective_synaptic_additions": 0,
        "input_spike_events": 0,
        "layer1_spike_events": 0,
        "layer2_spike_events": 0,
    }
