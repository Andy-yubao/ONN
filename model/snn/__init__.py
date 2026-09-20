"""Deployment-oriented PTQ and integer reference for the frozen T=4 SNN."""

from .integer_reference import IntegerSNNReference
from .quantization import (
    LayerQuantization,
    dequantize_weight,
    quantize_weight_symmetric,
    round_half_away_from_zero,
    saturate_signed,
)

__all__ = [
    "IntegerSNNReference",
    "LayerQuantization",
    "dequantize_weight",
    "quantize_weight_symmetric",
    "round_half_away_from_zero",
    "saturate_signed",
]
