"""Focused numerical checks for the implemented architecture."""

from __future__ import annotations

import torch

from .model import DeviceIFConvSmall, normalized_temporal_weights, subtract_if

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


def test_final_membrane_readout_weights_all_output_currents_equally() -> None:
    model = DeviceIFConvSmall()
    with torch.no_grad():
        model.conv1.weight.fill_(0.25)
        model.conv2.weight.fill_(0.02)
        model.readout.weight.fill_(0.01)
    times = torch.full((2, 1, 8, 8), -1, dtype=torch.long)
    times[0, :, 1:7, 1:7] = 0
    times[1, :, 1:7, 1:7] = 5
    currents = []
    handle = model.readout.register_forward_hook(
        lambda _module, _inputs, output: currents.append(output.detach().clone())
    )
    try:
        with torch.no_grad():
            logits = model(times, 8)
    finally:
        handle.remove()
    assert logits.shape == (2, 10)
    assert len(currents) == 8
    assert logits.abs().sum() > 0
    assert torch.allclose(logits, torch.stack(currents).sum(0))
    old_logits = sum((8 - step) * current for step, current in enumerate(currents)) / 8
    assert not torch.allclose(logits, old_logits)
    assert model.parameter_count() == 9872
    assert model.dense_equivalent_macs_per_step() * 8 == 704512


def test_accumulated_membrane_mode_preserves_time_weighted_baseline() -> None:
    model = DeviceIFConvSmall(readout_mode="accumulated_membrane")
    with torch.no_grad():
        model.conv1.weight.fill_(0.25)
        model.conv2.weight.fill_(0.02)
        model.readout.weight.fill_(0.01)
    times = torch.full((2, 1, 8, 8), -1, dtype=torch.long)
    times[0, :, 1:7, 1:7] = 0
    times[1, :, 1:7, 1:7] = 5
    currents = []
    handle = model.readout.register_forward_hook(
        lambda _module, _inputs, output: currents.append(output.detach().clone())
    )
    try:
        with torch.no_grad():
            logits = model(times, 8)
    finally:
        handle.remove()
    expected = sum((8 - step) * current for step, current in enumerate(currents)) / 8
    assert torch.allclose(logits, expected)
    assert not torch.allclose(logits, torch.stack(currents).sum(0))


def test_temporal_weight_normalization_preserves_scale() -> None:
    expected = {
        1.0: torch.tensor([4.0, 3.0, 2.0, 1.0]),
        0.5: torch.tensor([3.5714285, 2.8571429, 2.1428571, 1.4285715]),
        0.25: torch.tensor([3.1818182, 2.7272727, 2.2727273, 1.8181818]),
    }
    for beta, target in expected.items():
        weights = normalized_temporal_weights(4, beta)
        assert torch.allclose(weights, target, atol=1e-6)
        assert torch.allclose(weights.sum(), torch.tensor(10.0))


def test_temporal_beta_one_is_mathematically_identical_to_old_accumulation() -> None:
    model = DeviceIFConvSmall(readout_mode="accumulated_membrane", temporal_beta=1.0)
    times = torch.randint(-1, 4, (3, 1, 8, 8))
    currents = []
    handle = model.readout.register_forward_hook(
        lambda _module, _inputs, output: currents.append(output.detach().clone())
    )
    try:
        with torch.no_grad():
            logits = model(times, 4)
    finally:
        handle.remove()
    old_accumulation = torch.zeros_like(currents[0])
    readout_mem = torch.zeros_like(currents[0])
    for current in currents:
        readout_mem = readout_mem + current
        old_accumulation = old_accumulation + readout_mem
    assert torch.allclose(logits, old_accumulation / 4.0, atol=1e-6)


def test_event_proxy_backpropagates_without_changing_forward_contract() -> None:
    model = DeviceIFConvSmall(readout_mode="accumulated_membrane", temporal_beta=0.5)
    times = torch.randint(-1, 4, (2, 1, 8, 8))
    logits, raw_event_proxy = model(times, 4, return_event_proxy=True)
    assert logits.shape == (2, 10)
    assert raw_event_proxy.ndim == 0
    assert raw_event_proxy.requires_grad
    (logits.square().mean() + 0.01 * raw_event_proxy / 20_876.389236363637).backward()
    assert model.conv1.weight.grad is not None
    assert model.conv2.weight.grad is not None


def test_zero_event_lambda_matches_unregularized_loss_and_gradients() -> None:
    baseline = DeviceIFConvSmall(readout_mode="accumulated_membrane", temporal_beta=0.5)
    zero_lambda = DeviceIFConvSmall(readout_mode="accumulated_membrane", temporal_beta=0.5)
    zero_lambda.load_state_dict(baseline.state_dict())
    times = torch.randint(-1, 4, (2, 1, 8, 8))
    labels = torch.tensor([1, 7])
    criterion = torch.nn.CrossEntropyLoss()
    baseline_loss = criterion(baseline(times, 4), labels)
    logits, event_proxy = zero_lambda(times, 4, return_event_proxy=True)
    regularized_loss = criterion(logits, labels) + 0.0 * event_proxy
    baseline_loss.backward()
    regularized_loss.backward()
    assert torch.equal(baseline_loss, regularized_loss)
    for left, right in zip(baseline.parameters(), zero_lambda.parameters()):
        assert torch.allclose(left.grad, right.grad)


def test_rejects_unknown_readout_mode() -> None:
    import pytest
    with pytest.raises(ValueError, match="readout_mode"):
        DeviceIFConvSmall(readout_mode="unknown")
