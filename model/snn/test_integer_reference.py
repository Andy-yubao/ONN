"""Focused contract tests for frozen-SNN PTQ and integer inference."""

from __future__ import annotations

import torch

from model.snn.integer_reference import IntegerSNNReference, IntegerWidths
from model.snn.quantization import (
    quantize_weight_symmetric,
    round_half_away_from_zero,
    saturate_signed,
)


def _state(fill: float = 0.0) -> dict[str, torch.Tensor]:
    return {
        "conv1.weight": torch.full((16, 1, 3, 3), fill),
        "conv2.weight": torch.full((32, 16, 3, 3), fill),
        "readout.weight": torch.full((10, 512), fill),
    }


def test_rounding_boundaries_and_signed_negative_values() -> None:
    values = torch.tensor([-2.5, -1.5, -0.5, -0.49, 0.49, 0.5, 1.5, 2.5])
    expected = torch.tensor([-3.0, -2.0, -1.0, 0.0, 0.0, 1.0, 2.0, 3.0])
    assert torch.equal(round_half_away_from_zero(values), expected)


def test_saturation_boundaries() -> None:
    values = torch.tensor([-200, -128, -127, 0, 126, 127, 200], dtype=torch.int64)
    assert torch.equal(
        saturate_signed(values, 8),
        torch.tensor([-128, -128, -127, 0, 126, 127, 127], dtype=torch.int64),
    )


def test_weight_quantization_is_deterministic_and_uses_no_zero_point() -> None:
    weight = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0])
    left, left_spec = quantize_weight_symmetric(weight, "w")
    right, right_spec = quantize_weight_symmetric(weight, "w")
    assert torch.equal(left, right)
    assert left_spec == right_spec
    assert left_spec.zero_point == 0


def test_threshold_crossing_is_strict_and_reset_subtracts_threshold() -> None:
    current = torch.tensor([[4, 5, 6, -3]], dtype=torch.int64)
    membrane = torch.zeros_like(current)
    spikes, post, integrated = IntegerSNNReference._if_step(current, membrane, 5, 8, 8)
    assert torch.equal(integrated, current)
    assert torch.equal(spikes, torch.tensor([[False, False, True, False]]))
    assert torch.equal(post, torch.tensor([[4, 5, 1, -3]], dtype=torch.int64))


def test_beta_is_readout_coefficients_not_membrane_shift() -> None:
    assert IntegerSNNReference.temporal_coefficients == (5, 4, 3, 2)


def test_one_timestep_and_minimal_layer_trace() -> None:
    state = _state()
    state["conv1.weight"][:, :, 1, 1] = 2.0
    state["conv2.weight"][:, 0, 1, 1] = 2.0
    state["readout.weight"].fill_(2.0)
    reference = IntegerSNNReference(state)
    times = torch.full((1, 1, 8, 8), -1, dtype=torch.long)
    times[0, 0, 2, 2] = 0
    logits, trace = reference.forward(times, return_trace=True)
    assert logits.shape == (1, 10)
    assert len(trace["conv1_current"]) == 4
    assert trace["conv1_current"][0][0, 0, 2, 2] == (
        reference.weights["conv1.weight"][0, 0, 1, 1]
        << reference.state_fractional_guard_bits
    )
    assert trace["spikes1"][0][0, 0, 2, 2]
    assert trace["spikes2"][0].any()
    reconstructed = sum(
        coefficient * current
        for coefficient, current in zip(
            reference.temporal_coefficients, trace["readout_current"]
        )
    )
    assert torch.equal(logits, reconstructed)


def test_complete_inference_is_repeatable() -> None:
    generator = torch.Generator().manual_seed(7)
    state = {
        "conv1.weight": torch.randn(16, 1, 3, 3, generator=generator),
        "conv2.weight": torch.randn(32, 16, 3, 3, generator=generator),
        "readout.weight": torch.randn(10, 512, generator=generator),
    }
    reference = IntegerSNNReference(state)
    times = torch.randint(-1, 4, (2, 1, 8, 8), generator=generator)
    assert torch.equal(reference.forward(times), reference.forward(times))


def test_supported_guard_bits_have_worst_case_safe_widths() -> None:
    assert vars(IntegerWidths.for_guard_bits(8)) == {
        "conv1_current": 20, "mem1": 22, "conv2_current": 24, "mem2": 26,
        "readout_current": 18, "weighted_logits": 21,
    }
    assert vars(IntegerWidths.for_guard_bits(4)) == {
        "conv1_current": 16, "mem1": 18, "conv2_current": 20, "mem2": 22,
        "readout_current": 18, "weighted_logits": 21,
    }
    assert vars(IntegerWidths.for_guard_bits(2, readout_current=17)) == {
        "conv1_current": 14, "mem1": 16, "conv2_current": 18, "mem2": 20,
        "readout_current": 17, "weighted_logits": 21,
    }


def test_instrumentation_reports_ranges_and_no_saturation_for_minimal_case() -> None:
    reference = IntegerSNNReference(_state(), guard_bits=2)
    times = torch.full((1, 1, 8, 8), -1, dtype=torch.long)
    logits, metrics = reference.forward(times, return_metrics=True)
    assert torch.equal(logits, torch.zeros_like(logits))
    assert sum(metrics["saturations"].values()) == 0
    assert metrics["ranges"]["conv1_current"] == {"min": 0, "max": 0}
    assert metrics["threshold_margin"]["lif1"]["max"] < 0
