"""Explicit quantization primitives shared by PTQ and the integer reference."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch


def round_half_away_from_zero(values: torch.Tensor) -> torch.Tensor:
    """Round to nearest, resolving exact half cases away from zero."""
    return torch.sign(values) * torch.floor(torch.abs(values) + 0.5)


def saturate_signed(values: torch.Tensor, bits: int) -> torch.Tensor:
    """Clamp an integer tensor to the signed two's-complement range."""
    if bits < 2:
        raise ValueError("signed values need at least two bits")
    lower = -(1 << (bits - 1))
    upper = (1 << (bits - 1)) - 1
    return values.clamp(lower, upper)


@dataclass(frozen=True)
class LayerQuantization:
    """Per-tensor symmetric INT8 weight quantization metadata."""

    name: str
    scale: float
    fractional_bits: int | None
    qmin: int = -127
    qmax: int = 127
    zero_point: int = 0
    rounding: str = "round-to-nearest, ties-away-from-zero"

    def to_dict(self) -> dict:
        return asdict(self)


def _weight_scale(weight: torch.Tensor, *, power_of_two: bool) -> tuple[float, int | None]:
    maximum = float(weight.detach().abs().max().item())
    if not math.isfinite(maximum) or maximum <= 0.0:
        return 1.0, 0 if power_of_two else None
    if power_of_two:
        fractional_bits = math.floor(math.log2(127.0 / maximum))
        return math.ldexp(1.0, -fractional_bits), fractional_bits
    return maximum / 127.0, None


def quantize_weight_symmetric(
    weight: torch.Tensor,
    name: str,
    *,
    power_of_two: bool = False,
) -> tuple[torch.Tensor, LayerQuantization]:
    """Return signed INT8 weights and explicit per-tensor metadata."""
    scale, fractional_bits = _weight_scale(weight, power_of_two=power_of_two)
    quantized = round_half_away_from_zero(weight.detach().to(torch.float64) / scale)
    quantized = quantized.clamp(-127, 127).to(torch.int8)
    return quantized, LayerQuantization(name, scale, fractional_bits)


def dequantize_weight(weight: torch.Tensor, spec: LayerQuantization) -> torch.Tensor:
    """Dequantize an INT8 tensor for the weight-only fake-quant baseline."""
    return weight.to(torch.float32) * spec.scale
