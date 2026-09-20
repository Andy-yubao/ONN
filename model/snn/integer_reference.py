"""Bit-accurate integer core for the frozen T=4 IF-SNN.

The encoder boundary is an integer latency map ``[B,1,8,8]`` with values
``-1`` or ``0..3``.  All operations after that boundary are integer.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch.nn import functional as F

from .quantization import LayerQuantization, quantize_weight_symmetric, saturate_signed


@dataclass(frozen=True)
class IntegerWidths:
    conv1_current: int = 16
    mem1: int = 18
    conv2_current: int = 20
    mem2: int = 22
    readout_current: int = 17
    weighted_logits: int = 21

    @classmethod
    def for_guard_bits(
        cls,
        guard_bits: int,
        *,
        readout_current: int = 18,
    ) -> "IntegerWidths":
        """Return worst-case-safe widths for the supported guard-bit candidates."""
        hidden = {
            8: (20, 22, 24, 26),
            4: (16, 18, 20, 22),
            2: (14, 16, 18, 20),
        }
        if guard_bits not in hidden:
            raise ValueError("guard_bits must be one of 8, 4, or 2")
        return cls(*hidden[guard_bits], readout_current, 21)


class IntegerSNNReference:
    """Integer implementation of the frozen spike core, suitable for RTL comparison."""

    time_steps = 4
    # FP32 beta=0.5 weights are [25,20,15,10]/7.  Division by the common
    # positive factors 7 and T does not change argmax, leaving [5,4,3,2].
    temporal_coefficients = (5, 4, 3, 2)
    state_fractional_guard_bits = 4

    def __init__(
        self,
        state_dict: dict[str, torch.Tensor],
        *,
        widths: IntegerWidths | None = None,
        guard_bits: int = 4,
    ) -> None:
        if guard_bits not in {8, 4, 2}:
            raise ValueError("guard_bits must be one of 8, 4, or 2")
        self.state_fractional_guard_bits = guard_bits
        self.widths = widths or IntegerWidths.for_guard_bits(guard_bits)
        self.weights: dict[str, torch.Tensor] = {}
        self.specs: dict[str, LayerQuantization] = {}
        for name in ("conv1.weight", "conv2.weight", "readout.weight"):
            quantized, spec = quantize_weight_symmetric(
                state_dict[name].cpu(), name, power_of_two=False
            )
            self.weights[name] = quantized.to(torch.int64)
            self.specs[name] = spec
        self.thresholds = {
            name: int(
                math.floor(
                    1.0 / (spec.scale / (1 << self.state_fractional_guard_bits))
                    + 0.5
                )
            )
            for name, spec in self.specs.items()
            if name in {"conv1.weight", "conv2.weight"}
        }

    def quantization_metadata(self) -> dict:
        return {
            "weights": {name: spec.to_dict() for name, spec in self.specs.items()},
            "threshold_integer": dict(self.thresholds),
            "threshold_unsigned_bits": {
                name: value.bit_length() for name, value in self.thresholds.items()
            },
            "widths": vars(self.widths),
            "temporal_coefficients": list(self.temporal_coefficients),
            "hidden_state_guard_bits": self.state_fractional_guard_bits,
            "rounding": "round-to-nearest, ties-away-from-zero (weights and positive thresholds)",
            "overflow": "saturate after every current, membrane update, and readout accumulation",
            "threshold_compare": "strict greater-than",
            "reset": "subtract one quantized threshold after a spike",
        }

    @staticmethod
    def _if_step_instrumented(
        current: torch.Tensor,
        membrane: torch.Tensor,
        threshold: int,
        current_bits: int,
        membrane_bits: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, int]]:
        current_raw = current
        current = saturate_signed(current_raw, current_bits)
        integrated_raw = membrane + current
        integrated = saturate_signed(integrated_raw, membrane_bits)
        spikes = integrated > threshold
        post_reset_raw = integrated - spikes.to(torch.int64) * threshold
        post_reset = saturate_signed(post_reset_raw, membrane_bits)

        def clipped(values: torch.Tensor, bits: int) -> int:
            lower = -(1 << (bits - 1))
            upper = (1 << (bits - 1)) - 1
            return int(((values < lower) | (values > upper)).sum().item())

        margin = integrated - threshold
        metrics = {
            "current_saturations": clipped(current_raw, current_bits),
            "integrated_saturations": clipped(integrated_raw, membrane_bits),
            "post_reset_saturations": clipped(post_reset_raw, membrane_bits),
            "current_min": int(current_raw.min().item()),
            "current_max": int(current_raw.max().item()),
            "integrated_min": int(integrated_raw.min().item()),
            "integrated_max": int(integrated_raw.max().item()),
            "post_reset_min": int(post_reset_raw.min().item()),
            "post_reset_max": int(post_reset_raw.max().item()),
            "threshold_margin_min": int(margin.min().item()),
            "threshold_margin_max": int(margin.max().item()),
            "threshold_margin_min_abs": int(margin.abs().min().item()),
            "threshold_margin_abs_le_1": int((margin.abs() <= 1).sum().item()),
        }
        return spikes, post_reset, integrated, metrics

    @staticmethod
    def _if_step(
        current: torch.Tensor,
        membrane: torch.Tensor,
        threshold: int,
        current_bits: int,
        membrane_bits: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        current = saturate_signed(current, current_bits)
        integrated = saturate_signed(membrane + current, membrane_bits)
        spikes = integrated > threshold
        post_reset = saturate_signed(
            integrated - spikes.to(torch.int64) * threshold, membrane_bits
        )
        return spikes, post_reset, integrated

    def forward(
        self,
        first_spike_times: torch.Tensor,
        *,
        return_trace: bool = False,
        return_metrics: bool = False,
    ):
        if first_spike_times.ndim != 4 or first_spike_times.shape[1:] != (1, 8, 8):
            raise ValueError("first_spike_times must have shape [B,1,8,8]")
        times = first_spike_times.to(device="cpu", dtype=torch.int64)
        batch = times.shape[0]
        mem1 = torch.zeros(batch, 16, 8, 8, dtype=torch.int64)
        mem2 = torch.zeros(batch, 32, 4, 4, dtype=torch.int64)
        weighted_logits = torch.zeros(batch, 10, dtype=torch.int64)
        trace: dict[str, list[torch.Tensor]] = {
            "conv1_current": [], "mem1_integrated": [], "mem1_post_reset": [],
            "spikes1": [], "conv2_current": [], "mem2_integrated": [],
            "mem2_post_reset": [], "spikes2": [], "readout_current": [],
            "weighted_logits": [],
        }
        metrics = {
            "saturations": {
                "conv1_current": 0, "mem1_integrated": 0, "mem1_post_reset": 0,
                "conv2_current": 0, "mem2_integrated": 0, "mem2_post_reset": 0,
                "readout_current": 0, "weighted_logits": 0,
            },
            "ranges": {},
            "threshold_margin": {
                "lif1": {"min": None, "max": None, "min_abs": None, "abs_le_1": 0},
                "lif2": {"min": None, "max": None, "min_abs": None, "abs_le_1": 0},
            },
        }

        def update_range(name: str, values: torch.Tensor) -> None:
            low, high = int(values.min().item()), int(values.max().item())
            if name not in metrics["ranges"]:
                metrics["ranges"][name] = {"min": low, "max": high}
            else:
                metrics["ranges"][name]["min"] = min(metrics["ranges"][name]["min"], low)
                metrics["ranges"][name]["max"] = max(metrics["ranges"][name]["max"], high)

        def update_if_metrics(layer: str, prefix: str, values: dict[str, int]) -> None:
            metrics["saturations"][f"{prefix}_current"] += values["current_saturations"]
            metrics["saturations"][f"mem{layer}_integrated"] += values["integrated_saturations"]
            metrics["saturations"][f"mem{layer}_post_reset"] += values["post_reset_saturations"]
            margin = metrics["threshold_margin"][f"lif{layer}"]
            margin["min"] = values["threshold_margin_min"] if margin["min"] is None else min(
                margin["min"], values["threshold_margin_min"]
            )
            margin["max"] = values["threshold_margin_max"] if margin["max"] is None else max(
                margin["max"], values["threshold_margin_max"]
            )
            margin["min_abs"] = values["threshold_margin_min_abs"] if margin["min_abs"] is None else min(
                margin["min_abs"], values["threshold_margin_min_abs"]
            )
            margin["abs_le_1"] += values["threshold_margin_abs_le_1"]

        for step, coefficient in enumerate(self.temporal_coefficients):
            input_spikes = (times == step).to(torch.int64)
            current1 = F.conv2d(input_spikes, self.weights["conv1.weight"], padding=1)
            current1 = current1 << self.state_fractional_guard_bits
            if return_metrics:
                spikes1, mem1, integrated1, metrics1 = self._if_step_instrumented(
                    current1, mem1, self.thresholds["conv1.weight"],
                    self.widths.conv1_current, self.widths.mem1,
                )
            else:
                spikes1, mem1, integrated1 = self._if_step(
                    current1, mem1, self.thresholds["conv1.weight"],
                    self.widths.conv1_current, self.widths.mem1,
                )
                metrics1 = None
            current2 = F.conv2d(
                spikes1.to(torch.int64), self.weights["conv2.weight"],
                stride=2, padding=1,
            )
            current2 = current2 << self.state_fractional_guard_bits
            if return_metrics:
                spikes2, mem2, integrated2, metrics2 = self._if_step_instrumented(
                    current2, mem2, self.thresholds["conv2.weight"],
                    self.widths.conv2_current, self.widths.mem2,
                )
            else:
                spikes2, mem2, integrated2 = self._if_step(
                    current2, mem2, self.thresholds["conv2.weight"],
                    self.widths.conv2_current, self.widths.mem2,
                )
                metrics2 = None
            readout_raw = (
                spikes2.flatten(1).to(torch.int64)
                @ self.weights["readout.weight"].t()
            )
            readout_current = saturate_signed(
                readout_raw, self.widths.readout_current
            )
            weighted_raw = weighted_logits + coefficient * readout_current
            weighted_logits = saturate_signed(
                weighted_raw, self.widths.weighted_logits,
            )
            if return_metrics:
                assert metrics1 is not None and metrics2 is not None
                update_if_metrics("1", "conv1", metrics1)
                update_if_metrics("2", "conv2", metrics2)
                readout_lower = -(1 << (self.widths.readout_current - 1))
                readout_upper = (1 << (self.widths.readout_current - 1)) - 1
                score_lower = -(1 << (self.widths.weighted_logits - 1))
                score_upper = (1 << (self.widths.weighted_logits - 1)) - 1
                metrics["saturations"]["readout_current"] += int(
                    ((readout_raw < readout_lower) | (readout_raw > readout_upper)).sum().item()
                )
                metrics["saturations"]["weighted_logits"] += int(
                    ((weighted_raw < score_lower) | (weighted_raw > score_upper)).sum().item()
                )
                for name, value in (
                    ("conv1_current", current1), ("mem1_integrated", integrated1),
                    ("mem1_post_reset", mem1), ("conv2_current", current2),
                    ("mem2_integrated", integrated2), ("mem2_post_reset", mem2),
                    ("readout_current", readout_raw), ("weighted_logits", weighted_raw),
                ):
                    update_range(name, value)
            if return_trace:
                values = (
                    current1, integrated1, mem1, spikes1, current2, integrated2,
                    mem2, spikes2, readout_current, weighted_logits,
                )
                for key, value in zip(trace, values):
                    trace[key].append(value.clone())

        if return_trace and return_metrics:
            return weighted_logits, trace, metrics
        if return_trace:
            return weighted_logits, trace
        if return_metrics:
            return weighted_logits, metrics
        return weighted_logits
