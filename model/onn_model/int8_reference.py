"""Pure-integer reference inference for the scheme-A quantised BaselineCNN.

This module re-implements the W8A8 inference path of :mod:`onn_model.int8_ptq`
but as a *truly integer* forward pass: the convolution / pool / fc chain never
dequantises back to float to continue computing.  Float values are only
touched at three points, exactly as the phase spec allows:

  1. at build time, to read the frozen scales (``candidate_quant_config.json``);
  2. at build time, to derive the fixed-point requantisation multipliers;
  3. at the very end, to dequantise the INT32 logits for error reporting.

Integer conventions (frozen "方案 A" config, per
``experiments/model_deployment/baseline_cnn_int8_ptq/candidate_quant_config.json``):

    weights      signed INT8, symmetric, range [-127, 127], zero_point = 0
    input        signed INT8, range [-128, 127], zero_point = 0
    ReLU output  UINT8,     range [  0, 255], zero_point = 0
    bias         INT32,     quantised at bias_scale = input_scale * weight_scale
    accumulator  INT32
    requant      acc * multiplier >> shift   (INT64 intermediate,
                   round-half-away-from-zero, saturate to UINT8)
    MaxPool      direct max over UINT8 (scale preserved, never re-quantised)
    GAP          integer sum / 49 with round-half-away-from-zero
    logits       INT32 accumulator, argmax directly (no final quantise)

Observable integer nodes produced by :meth:`Int8Reference.infer` (``trace``):

    input_q, conv1_acc, stem_q, pool1_q,
    conv2_acc, conv2_q, pool2_q,
    conv3_acc, conv3_q, gap_q, fc_acc

Semantic note on the GAP / fc scales:

The integer GAP keeps the *conv3* integer units (``round(sum/49)``), so the
fc's input-activation scale is ``conv3_relu``'s scale and the fc bias is
quantised at ``s_conv3 * s_w_fc``.  The fake-quant reference instead
re-quantises the average to the ``pool`` scale (``s_pool``).  Both are
self-consistent approximations of the FP32 model; they differ in GAP rounding
granularity and in the fc bias scale.  This is the one place where the integer
reference deliberately deviates from the fake-quant forward.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from onn_model.int8_ptq import (
    INT8_MAX,
    INT8_MIN,
    INT32_MAX,
    INT32_MIN,
    UINT8_MAX,
    UINT8_MIN,
    WEIGHT_MAX,
    WEIGHT_MIN,
    WEIGHT_LAYERS,
    WEIGHT_LAYER_ACCESSORS,
    quantize_signed,
    round_half_away_from_zero,
)

# ---------------------------------------------------------------------------
# Integer node contract
# ---------------------------------------------------------------------------

# Integer nodes produced by the reference forward, in forward order.
TRACE_NODES: Tuple[str, ...] = (
    "input_q",
    "conv1_acc",
    "stem_q",
    "pool1_q",
    "conv2_acc",
    "conv2_q",
    "pool2_q",
    "conv3_acc",
    "conv3_q",
    "gap_q",
    "fc_acc",
)

# dtype + [min, max] range contract for every integer node.
NODE_SPEC: Dict[str, Tuple[torch.dtype, int, int]] = {
    "input_q": (torch.int8, INT8_MIN, INT8_MAX),
    "conv1_acc": (torch.int32, INT32_MIN, INT32_MAX),
    "stem_q": (torch.uint8, UINT8_MIN, UINT8_MAX),
    "pool1_q": (torch.uint8, UINT8_MIN, UINT8_MAX),
    "conv2_acc": (torch.int32, INT32_MIN, INT32_MAX),
    "conv2_q": (torch.uint8, UINT8_MIN, UINT8_MAX),
    "pool2_q": (torch.uint8, UINT8_MIN, UINT8_MAX),
    "conv3_acc": (torch.int32, INT32_MIN, INT32_MAX),
    "conv3_q": (torch.uint8, UINT8_MIN, UINT8_MAX),
    "gap_q": (torch.uint8, UINT8_MIN, UINT8_MAX),
    "fc_acc": (torch.int32, INT32_MIN, INT32_MAX),
}

# Layers with integer accumulators, in forward order (the fc is a matmul).
ACCUMULATOR_LAYERS: Tuple[str, ...] = ("stem_conv", "conv2", "conv3", "fc")

# Which activation position feeds each weight layer in the *integer* reference.
# The fc input is gap_q, which lives in conv3 units (see module docstring).
INT_LAYER_INPUT_ACTIVATION: Dict[str, str] = {
    "stem_conv": "input",
    "conv2": "stem_relu",   # pool1_q preserves stem_relu scale
    "conv3": "conv2_relu",  # pool2_q preserves conv2_relu scale
    "fc": "conv3_relu",     # gap_q stays in conv3 units
}

# Activation position a conv layer requantises *to* (its UINT8 output).
LAYER_OUTPUT_ACTIVATION: Dict[str, str] = {
    "stem_conv": "stem_relu",
    "conv2": "conv2_relu",
    "conv3": "conv3_relu",
}

# GAP spatial divisor: conv3 output is 7 x 7.
GAP_DIVISOR = 49

# Requantisation multiplier must fit a signed 31-bit magnitude.
MULTIPLIER_LIMIT = 2**31


def _resolve_module(model: nn.Module, accessor: str) -> nn.Module:
    """Resolve a short accessor like ``stem[-1]`` on the fused model."""
    name, _, idx = accessor.partition("[")
    mod = getattr(model, name)
    if idx:
        return mod[int(idx.rstrip("]"))]
    return mod


# ---------------------------------------------------------------------------
# Integer primitives
# ---------------------------------------------------------------------------


def round_shift(x: torch.Tensor, shift: int) -> torch.Tensor:
    """Round-half-away-from-zero right shift of an integer tensor.

    ``x`` is an INT64 tensor (``accumulator * multiplier``).  Returns
    ``round_half_away_from_zero(x / 2**shift)`` as INT64, computed on the
    magnitude so no two's-complement edge cases arise (e.g. ``-4 >> 1``).
    """
    if shift <= 0:
        return x
    ax = x.abs()
    q = (ax + (1 << (shift - 1))) >> shift
    return torch.sign(x) * q


def round_div_away(a: torch.Tensor, d: int) -> torch.Tensor:
    """Integer division of a signed tensor by ``d``, rounding half away from zero."""
    ax = a.abs()
    q = (ax + (d >> 1)) // d
    return torch.sign(a) * q


def _round_py(v: float) -> int:
    """Round-half-away-from-zero of a non-negative float to an int."""
    return int(math.floor(v + 0.5))


def real_multiplier_to_fixed(real_multiplier: float) -> Tuple[int, int, float]:
    """Approximate a positive real multiplier as ``multiplier / 2**shift``.

    Returns ``(multiplier, shift, relative_error)`` with

      * ``multiplier`` a positive integer < ``2**31`` (signed 31-bit magnitude),
      * ``shift`` a non-negative integer,
      * ``multiplier / 2**shift ≈ real_multiplier``.

    ``shift`` is chosen so ``multiplier`` is as large as possible below
    ``2**31``, maximising precision (the loop starts at 0 and advances until the
    multiplier would cross the limit; the last representable shift wins).
    """
    assert real_multiplier > 0, f"real multiplier must be positive, got {real_multiplier}"
    if real_multiplier >= MULTIPLIER_LIMIT:
        raise ValueError(
            f"real multiplier {real_multiplier:.6e} >= 2**31 is not representable "
            "as multiplier / 2**shift"
        )
    s = 0
    while s <= 63:
        m = _round_py(real_multiplier * (1 << s))
        if m >= MULTIPLIER_LIMIT:
            break
        s += 1
    if s == 0:
        raise ValueError(
            f"real multiplier {real_multiplier:.6e} already >= 2**31 at shift 0"
        )
    s -= 1
    m = _round_py(real_multiplier * (1 << s))
    approx = m / (1 << s)
    relative_error = abs(approx - real_multiplier) / real_multiplier
    return m, s, relative_error


def quantize_weight_with_scale(w: torch.Tensor, scale: float) -> torch.Tensor:
    """Symmetric INT8 weight quantisation with a *frozen* scale.

    Same rule as the PTQ phase (round-half-away-from-zero, range [-127, 127]),
    but the scale comes from the frozen config instead of being recomputed.
    """
    q = round_half_away_from_zero(w / scale)
    q = q.clamp(WEIGHT_MIN, WEIGHT_MAX).to(torch.int8)
    return q


def quantize_bias_int32(b: torch.Tensor, bias_scale: float) -> Tuple[torch.Tensor, int]:
    """Quantise a fused bias into the INT32 accumulator.

    ``bias_scale = input_scale * weight_scale``.  Returns
    ``(q_int32, out_of_range_count)`` where the count reports elements that
    would fall outside INT32 *before* saturating (saturate, never wrap).
    """
    q = round_half_away_from_zero(b.double() / bias_scale)
    out_of_range = int(((q < INT32_MIN) | (q > INT32_MAX)).sum().item())
    q = q.clamp(INT32_MIN, INT32_MAX).to(torch.int32)
    return q, out_of_range


def check_accumulator(
    acc64: torch.Tensor, layer: str, acc_stats: Dict[str, Any]
) -> torch.Tensor:
    """Verify an INT64 accumulator fits INT32; saturate if not (no wraparound).

    Records per-layer running min / max / overflow count for reporting.
    """
    acc_stats["overflow_count"][layer] += int(
        ((acc64 < INT32_MIN) | (acc64 > INT32_MAX)).sum().item()
    )
    acc_min = float(acc64.min().item())
    acc_max = float(acc64.max().item())
    acc_stats["min"][layer] = min(acc_stats["min"][layer], acc_min)
    acc_stats["max"][layer] = max(acc_stats["max"][layer], acc_max)
    acc = acc64.clamp(INT32_MIN, INT32_MAX).to(torch.int32)
    return acc


# ---------------------------------------------------------------------------
# The reference model
# ---------------------------------------------------------------------------


class Int8Reference:
    """Pure-integer W8A8 forward for the BN-fused BaselineCNN (scheme A).

    ``config`` is the frozen quant config (``candidate_quant_config.json``):
    it must carry ``activation_scale`` (input / stem_relu / conv2_relu /
    conv3_relu) and ``weight_scale`` (stem_conv / conv2 / conv3 / fc).
    """

    def __init__(self, model: nn.Module, config: Dict[str, Any]) -> None:
        model.eval()
        self.act_scale = dict(config["activation_scale"])
        self.weight_scale = dict(config["weight_scale"])
        self.s_in = self.act_scale["input"]

        self.wq: Dict[str, torch.Tensor] = {}
        self.qb: Dict[str, torch.Tensor] = {}
        self.bias_out_of_int32: Dict[str, int] = {}

        for name, accessor in WEIGHT_LAYER_ACCESSORS.items():
            mod = _resolve_module(model, accessor)
            # -- weights (frozen per-tensor scale) --
            self.wq[name] = quantize_weight_with_scale(
                mod.weight.detach(), self.weight_scale[name]
            )
            # -- bias (input_scale * weight_scale) --
            act_scale = self.act_scale[INT_LAYER_INPUT_ACTIVATION[name]]
            bias_scale = act_scale * self.weight_scale[name]
            qb, out_of_range = quantize_bias_int32(mod.bias.detach(), bias_scale)
            self.qb[name] = qb
            self.bias_out_of_int32[name] = out_of_range

        # -- fixed-point requantisation for the three convs --
        self.requant: Dict[str, Dict[str, Any]] = {}
        for name in ("stem_conv", "conv2", "conv3"):
            s_in = self.act_scale[INT_LAYER_INPUT_ACTIVATION[name]]
            s_w = self.weight_scale[name]
            s_out = self.act_scale[LAYER_OUTPUT_ACTIVATION[name]]
            r = s_in * s_w / s_out
            m, shift, rel_err = real_multiplier_to_fixed(r)
            self.requant[name] = {
                "layer": name,
                "real_multiplier": r,
                "multiplier": m,
                "shift": shift,
                "relative_error": rel_err,
                "s_in": s_in,
                "s_w": s_w,
                "s_out": s_out,
            }

    # -- helpers -----------------------------------------------------------

    def _requantize_uint8(self, acc: torch.Tensor, rq: Dict[str, Any]) -> torch.Tensor:
        """Requantise an INT32 accumulator into UINT8 via multiplier/shift."""
        prod = acc.to(torch.int64) * rq["multiplier"]  # INT64 intermediate
        q = round_shift(prod, rq["shift"])
        return q.clamp(UINT8_MIN, UINT8_MAX).to(torch.uint8)

    def _maxpool_uint8(self, q: torch.Tensor) -> torch.Tensor:
        """Integer 2x2/stride-2 MaxPool directly on UINT8 (scale preserved)."""
        return F.max_pool2d(q.to(torch.int32), kernel_size=2, stride=2).to(torch.uint8)

    def _conv_acc(
        self, x: torch.Tensor, w: torch.Tensor, qb: torch.Tensor, layer: str, stats: Dict[str, Any]
    ) -> torch.Tensor:
        """Integer conv: INT64 MAC host, verified + saturated to INT32."""
        acc64 = F.conv2d(x.to(torch.int64), w.to(torch.int64), stride=1, padding=1)
        acc64 = acc64 + qb.to(torch.int64).view(1, -1, 1, 1)
        return check_accumulator(acc64, layer, stats)

    def _linear_acc(
        self, x: torch.Tensor, w: torch.Tensor, qb: torch.Tensor, layer: str, stats: Dict[str, Any]
    ) -> torch.Tensor:
        """Integer linear (fc): INT64 MAC host, verified + saturated to INT32.

        The fc is a matmul over the flattened GAP output.  The dot product is
        accumulated on an INT64 host so a true overflow is detected *before*
        any INT32 wraparound could corrupt the value; ``check_accumulator``
        then records the real pre-saturation range and saturates to INT32.
        """
        acc64 = x.to(torch.int64) @ w.to(torch.int64).t()
        acc64 = acc64 + qb.to(torch.int64).view(1, -1)
        return check_accumulator(acc64, layer, stats)

    # -- forward -----------------------------------------------------------

    @torch.no_grad()
    def infer(
        self, x_norm: torch.Tensor, return_trace: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, Any]]]:
        """Pure-integer W8A8 forward of the fused BaselineCNN.

        Parameters
        ----------
        x_norm : torch.Tensor
            Normalised FP32 input ``[N, 1, 28, 28]`` (same preprocessing as
            training: ToTensor -> Normalize(0.1307, 0.3081)).
        return_trace : bool
            When True, returns a ``trace`` dict with the observable integer
            nodes (see ``TRACE_NODES``) plus per-batch accumulator stats.

        Returns
        -------
        (prediction, logits, trace)
            prediction : INT64 argmax over ``fc_acc`` (never requantised);
            logits     : FP32 dequantised logits (reporting only);
            trace      : dict of integer node tensors + ``"acc_stats"``.
        """
        acc_stats: Dict[str, Any] = {
            "overflow_count": {l: 0 for l in ACCUMULATOR_LAYERS},
            "min": {l: float("inf") for l in ACCUMULATOR_LAYERS},
            "max": {l: float("-inf") for l in ACCUMULATOR_LAYERS},
        }

        # ---- input (signed INT8) ----
        q = quantize_signed(x_norm, self.s_in)  # int8

        # ---- stem: conv -> requant (UINT8 ReLU) -> integer MaxPool ----
        acc1 = self._conv_acc(q, self.wq["stem_conv"], self.qb["stem_conv"], "stem_conv", acc_stats)
        stem_q = self._requantize_uint8(acc1, self.requant["stem_conv"])
        pool1_q = self._maxpool_uint8(stem_q)

        # ---- conv2: conv -> requant (UINT8 ReLU) -> integer MaxPool ----
        acc2 = self._conv_acc(pool1_q, self.wq["conv2"], self.qb["conv2"], "conv2", acc_stats)
        conv2_q = self._requantize_uint8(acc2, self.requant["conv2"])
        pool2_q = self._maxpool_uint8(conv2_q)

        # ---- conv3: conv -> requant (UINT8 ReLU) ----
        acc3 = self._conv_acc(pool2_q, self.wq["conv3"], self.qb["conv3"], "conv3", acc_stats)
        conv3_q = self._requantize_uint8(acc3, self.requant["conv3"])

        # ---- integer GAP: sum / 49 with round-half-away-from-zero ----
        gap_sum = conv3_q.to(torch.int32).sum(dim=(2, 3))  # [N, 32]
        gap_q = round_div_away(gap_sum, GAP_DIVISOR)
        gap_q = gap_q.clamp(UINT8_MIN, UINT8_MAX).to(torch.uint8)

        # ---- fc: UINT8 x INT8 -> INT32 logits (no requant, no dequant) ----
        # INT64 MAC host: a true overflow is seen before any INT32 wraparound;
        # check_accumulator records the real range and saturates to INT32.
        fc_acc = self._linear_acc(gap_q, self.wq["fc"], self.qb["fc"], "fc", acc_stats)

        prediction = fc_acc.argmax(dim=1)  # INT64
        logits = fc_acc.float() * (self.weight_scale["fc"] * self.act_scale["conv3_relu"])

        if not return_trace:
            return prediction, logits, None

        trace = {
            "input_q": q,
            "conv1_acc": acc1,
            "stem_q": stem_q,
            "pool1_q": pool1_q,
            "conv2_acc": acc2,
            "conv2_q": conv2_q,
            "pool2_q": pool2_q,
            "conv3_acc": acc3,
            "conv3_q": conv3_q,
            "gap_q": gap_q,
            "fc_acc": fc_acc,
            "acc_stats": acc_stats,
        }
        return prediction, logits, trace
