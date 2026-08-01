"""Spec-compliant INT8 PTQ primitives and W8A8 simulation for BaselineCNN.

This module implements the *numerical rules* of the INT8 PTQ feasibility study
(see ``experiments/model_deployment/baseline_cnn_int8_ptq/README.md``):

    quantise    : q = round(x / scale)
    rounding    : round-half-away-from-zero
    overflow    : saturate (clamp) — integer wraparound is never allowed
    weights     : signed INT8, symmetric, range [-127, 127], zero_point = 0
    activations : signed [-128, 127] (network input),
                  unsigned [0, 255] (ReLU / GAP output), zero_point = 0
    accumulator : INT32

Everything here is a *software* W8A8 simulation (fake quantisation):

    FP32 -> quantise integer -> dequantise FP32 -> continue

and does **not** claim FPGA bit-accuracy.

The target is the BN-fused ``BaselineCNN`` (seed 43 delivery candidate).
TinyResNet and the M2 compact models are out of scope for this phase.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Numerical rules (spec section 4)
# ---------------------------------------------------------------------------

INT8_MIN, INT8_MAX = -128, 127
WEIGHT_MIN, WEIGHT_MAX = -127, 127
UINT8_MIN, UINT8_MAX = 0, 255
INT32_MIN, INT32_MAX = -(2**31), 2**31 - 1
_EPS = 1e-12

# Activation positions measured during calibration, in forward order.
ACTIVATION_POSITIONS: Tuple[str, ...] = (
    "input",
    "stem_relu",
    "pool1",
    "conv2_relu",
    "pool2",
    "conv3_relu",
    "pool",
    "logits",
)

# Positions whose *integer* values pass through MaxPool unchanged: they keep
# the scale of their upstream ReLU output (max-pool never re-quantises).
MAXPOOL_PASSTHROUGH: Dict[str, str] = {
    "pool1": "stem_relu",
    "pool2": "conv2_relu",
}

# Calibration position -> module accessor on the fused BaselineCNN.
LAYER_ACCESSORS: Dict[str, str] = {
    "stem_relu": "stem[-1]",
    "pool1": "pool1",
    "conv2_relu": "conv2[-1]",
    "pool2": "pool2",
    "conv3_relu": "conv3[-1]",
    "pool": "pool",
    "logits": "fc",
}

# Fused-model layer accessors for the three convs and the fc.
WEIGHT_LAYERS: Tuple[str, ...] = ("stem_conv", "conv2", "conv3", "fc")
WEIGHT_LAYER_ACCESSORS: Dict[str, str] = {
    "stem_conv": "stem[0]",
    "conv2": "conv2[0]",
    "conv3": "conv3[0]",
    "fc": "fc",
}

# Activation position feeding each weight layer (its input activation scale).
LAYER_INPUT_ACTIVATION: Dict[str, str] = {
    "stem_conv": "input",
    "conv2": "stem_relu",    # pooled stem_relu, scale preserved
    "conv3": "conv2_relu",   # pooled conv2_relu, scale preserved
    "fc": "pool",
}


def _resolve_module(model: nn.Module, accessor: str) -> nn.Module:
    """Resolve a short accessor like ``stem[-1]`` on the model."""
    name, _, idx = accessor.partition("[")
    mod = getattr(model, name)
    if idx:
        return mod[int(idx.rstrip("]"))]
    return mod


# ---------------------------------------------------------------------------
# Quantisation primitives
# ---------------------------------------------------------------------------


def round_half_away_from_zero(x: torch.Tensor) -> torch.Tensor:
    """Round half away from zero (0.5 -> 1, -0.5 -> -1).

    ``torch.round`` is round-half-to-even, which the phase spec forbids.
    """
    return torch.sign(x) * torch.floor(x.abs() + 0.5)


def quantize_signed(x: torch.Tensor, scale: float) -> torch.Tensor:
    """Symmetrically quantise to signed INT8, saturating.

    Returns an ``int8`` tensor; ``zero_point = 0`` always.
    """
    q = round_half_away_from_zero(x / scale)
    q = q.clamp(INT8_MIN, INT8_MAX)
    return q.to(torch.int8)


def quantize_unsigned(x: torch.Tensor, scale: float) -> torch.Tensor:
    """Quantise non-negative values to UINT8 [0, 255], saturating.

    ``zero_point = 0``.  Values outside the range clip to 0 / 255 (saturate,
    no wraparound).
    """
    q = round_half_away_from_zero(x / scale)
    q = q.clamp(UINT8_MIN, UINT8_MAX)
    return q.to(torch.uint8)


def dequantize(q: torch.Tensor, scale: float) -> torch.Tensor:
    """Dequantise an integer tensor back to FP32 (fake quantise)."""
    return q.float() * scale


def quantize_weight_tensor(
    w: torch.Tensor, qmin: int = WEIGHT_MIN, qmax: int = WEIGHT_MAX
) -> Dict[str, Any]:
    """Per-tensor symmetric INT8 weight quantisation (scheme A).

    ``scale = max_abs / 127``.  Returns ``q`` (int8), ``scale`` (float) and
    saturation statistics, where *saturated* means a quantised magnitude of
    exactly 127 (the range edge).
    """
    scale = w.abs().max().item() / 127.0
    if scale < _EPS:
        scale = _EPS
    q = round_half_away_from_zero(w / scale)
    q = q.clamp(qmin, qmax).to(torch.int8)
    saturated = q.abs() >= qmax
    count = int(saturated.sum().item())
    return {
        "q": q,
        "scale": float(scale),
        "saturated": saturated,
        "saturated_count": count,
        "saturated_fraction": count / w.numel(),
    }


def quantize_weight_per_channel(
    w: torch.Tensor, dim: int = 0, qmin: int = WEIGHT_MIN, qmax: int = WEIGHT_MAX
) -> Dict[str, Any]:
    """Per-output-channel symmetric INT8 weight quantisation (scheme B).

    ``w`` is ``(Cout, ...)``.  Returns ``q`` (int8), ``scales`` (Cout-vector)
    and saturation statistics.
    """
    out_channels = w.shape[dim]
    w2d = w.reshape(out_channels, -1)
    scales = torch.clamp(w2d.abs().max(dim=1).values / 127.0, min=_EPS)
    q = round_half_away_from_zero(w2d / scales[:, None])
    q = q.clamp(qmin, qmax).to(torch.int8)
    q = q.reshape(w.shape)
    saturated = q.abs() >= qmax
    count = int(saturated.sum().item())
    return {
        "q": q,
        "scales": scales,
        "saturated": saturated,
        "saturated_count": count,
        "saturated_fraction": count / w.numel(),
    }


def quantize_bias(
    b: torch.Tensor, weight_scale: Any, act_scale: float
) -> torch.Tensor:
    """Quantise a bias into the INT32 accumulator (as float64 integer values).

    ``weight_scale`` is either a scalar (per-tensor weights) or a
    per-output-channel vector.  Returns a 1-D integer-valued tensor.
    """
    s = weight_scale * act_scale
    return round_half_away_from_zero(b.double() / s).clamp(INT32_MIN, INT32_MAX)


# ---------------------------------------------------------------------------
# Activation calibration
# ---------------------------------------------------------------------------


def get_calibration_loader(
    data_root: str,
    num_samples: int = 2048,
    seed: int = 12345,
    batch_size: int = 256,
) -> Tuple[DataLoader, List[int]]:
    """Build a fixed calibration set from the MNIST *training* split.

    Uses a fixed random seed and returns the exact sampled indices so the
    results are reproducible.  The test set is never used to derive scales.
    """
    from onn_model.data import get_mnist_dataset
    from torch.utils.data import Subset

    dataset = get_mnist_dataset(data_root, train=True, download=True)
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:num_samples].tolist()
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0)
    return loader, indices


def _percentiles(values: torch.Tensor, qs: Sequence[float]) -> List[float]:
    """Exact percentile values via numpy.

    ``torch.quantile`` errors with "input tensor is too large" above roughly
    2**24 elements, which the largest calibration position (stem_relu, ~25.7M)
    exceeds, so the sort is delegated to numpy.
    """
    v = values.detach().float().contiguous().view(-1)
    return [float(x) for x in np.quantile(v.numpy(), list(qs))]


def activation_stats(values: torch.Tensor) -> Dict[str, Any]:
    """One position's calibration statistics over all collected samples."""
    v = values.detach().float().flatten()
    numel = v.numel()
    pct = _percentiles(v, [0.99, 0.999, 0.9999])
    pct_abs = _percentiles(v.abs(), [0.99, 0.999, 0.9999])
    return {
        "min": float(v.min().item()),
        "max": float(v.max().item()),
        "max_abs": float(v.abs().max().item()),
        "mean": float(v.mean().item()),
        "std": float(v.std().item()),
        "p99": pct[0],
        "p99.9": pct[1],
        "p99.99": pct[2],
        "p99_abs": pct_abs[0],
        "p99.9_abs": pct_abs[1],
        "p99.99_abs": pct_abs[2],
        "numel": numel,
    }


def calibrate_activations(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> Dict[str, torch.Tensor]:
    """Collect raw activation values at the 8 calibration positions.

    Returns ``{position: tensor}`` with values concatenated over all
    calibration batches (loader is not shuffled, order is deterministic).
    """
    captured: Dict[str, List[torch.Tensor]] = {p: [] for p in ACTIVATION_POSITIONS}
    handles: List[Any] = []

    def _input_hook(_mod, inp, _out):
        captured["input"].append(inp[0].detach().clone())

    def _make_hook(key: str):
        def _hook(_mod, _inp, out):
            captured[key].append(out.detach().clone())

        return _hook

    handles.append(_resolve_module(model, "stem[0]").register_forward_hook(_input_hook))
    for key, accessor in LAYER_ACCESSORS.items():
        handles.append(_resolve_module(model, accessor).register_forward_hook(_make_hook(key)))

    try:
        for images, _labels in loader:
            model(images.to(device))
    finally:
        for h in handles:
            h.remove()

    return {p: torch.cat(v, dim=0) for p, v in captured.items() if v}


def build_activation_scales(
    calib: Dict[str, torch.Tensor],
    clip_percentile: Optional[float] = None,
) -> Dict[str, float]:
    """Derive per-tensor activation scales from calibration data.

    Signed input uses ``max_abs / 127``; UINT8 (ReLU / GAP) positions use
    ``max / 255``.  When ``clip_percentile`` is given the scale is derived from
    that percentile instead of the full range (saturation absorbs the tail).
    MaxPool positions inherit their upstream ReLU scale.
    """
    stats = {p: activation_stats(v) for p, v in calib.items()}
    scales: Dict[str, float] = {}
    for p in ACTIVATION_POSITIONS:
        if p == "logits":
            continue  # final output kept FP32, never quantised
        if p in MAXPOOL_PASSTHROUGH:
            scales[p] = scales[MAXPOOL_PASSTHROUGH[p]]
            continue
        if p == "input":
            if clip_percentile is None:
                ref = stats[p]["max_abs"]
            else:
                ref = stats[p][f"p{clip_percentile}_abs"]
            scales[p] = max(ref / 127.0, _EPS)
        else:
            if clip_percentile is None:
                ref = stats[p]["max"]
            else:
                ref = stats[p][f"p{clip_percentile}"]
            scales[p] = max(ref / 255.0, _EPS)
    return scales


# ---------------------------------------------------------------------------
# Weight statistics / quantisation
# ---------------------------------------------------------------------------


def weight_stats(model: nn.Module) -> Dict[str, Any]:
    """Per-layer overall + per-output-channel weight statistics."""
    stats: Dict[str, Any] = {}
    for name, accessor in WEIGHT_LAYER_ACCESSORS.items():
        w = _resolve_module(model, accessor).weight.detach()
        per_ch = w.reshape(w.shape[0], -1).abs().max(dim=1).values.tolist()
        stats[name] = {
            "shape": list(w.shape),
            "min": float(w.min().item()),
            "max": float(w.max().item()),
            "max_abs": float(w.abs().max().item()),
            "per_output_channel_max_abs": [float(v) for v in per_ch],
        }
    return stats


def build_weight_config(
    model: nn.Module,
    weight_scheme: str,
    fc_scheme: Optional[str] = None,
) -> Dict[str, Any]:
    """Quantise every weight layer (and its bias) under the chosen scheme.

    ``weight_scheme`` applies to the three convs; ``fc_scheme`` overrides the
    fc when given (scheme B tests both per-tensor and per-output-channel on
    the fc).  Returns a dict per layer with ``q`` (int8), ``weight_scale``
    (scalar or Cout-vector), ``q_bias`` (INT32-valued) and saturation stats.
    """
    cfg: Dict[str, Any] = {}
    for name, accessor in WEIGHT_LAYER_ACCESSORS.items():
        mod = _resolve_module(model, accessor)
        scheme = fc_scheme if (name == "fc" and fc_scheme is not None) else weight_scheme
        w = mod.weight.detach()
        if scheme == "per-output-channel":
            r = quantize_weight_per_channel(w)
            wscale: Any = r["scales"]
            method = "per-output-channel"
        else:
            r = quantize_weight_tensor(w)
            wscale = r["scale"]
            method = "per-tensor"
        cfg[name] = {
            "method": method,
            "q": r["q"],
            "weight_scale": wscale,
            "saturated_count": r["saturated_count"],
            "saturated_fraction": r["saturated_fraction"],
            "bias": mod.bias.detach(),
        }
    return cfg


def layer_input_activation_scale(layer: str, act_scales: Dict[str, float]) -> float:
    """Activation scale of the integer values feeding ``layer``."""
    return act_scales[LAYER_INPUT_ACTIVATION[layer]]


def finalize_weight_config(
    weight_cfg: Dict[str, Any], act_scales: Dict[str, float]
) -> Dict[str, Any]:
    """Quantise each layer's bias using its own input-activation scale."""
    for layer, entry in weight_cfg.items():
        act_scale = layer_input_activation_scale(layer, act_scales)
        entry["q_bias"] = quantize_bias(entry["bias"], entry["weight_scale"], act_scale)
        entry.pop("bias", None)
    return weight_cfg


# ---------------------------------------------------------------------------
# INT32 accumulator simulation
# ---------------------------------------------------------------------------


def _accumulate_int32(
    acc: torch.Tensor, layer: str, info: Dict[str, Any]
) -> torch.Tensor:
    """Verify an integer accumulator fits INT32; saturate if it does not."""
    max_abs = float(acc.abs().max().item())
    overflow = int(((acc < INT32_MIN) | (acc > INT32_MAX)).sum().item())
    acc = acc.clamp(INT32_MIN, INT32_MAX)
    info["accumulator_max_abs"][layer] = max_abs
    info["accumulator_overflow"][layer] = overflow
    return acc


# ---------------------------------------------------------------------------
# W8A8 forward simulation (fused BaselineCNN only)
# ---------------------------------------------------------------------------


@torch.no_grad()
def w8a8_forward(
    model: nn.Module,
    x: torch.Tensor,
    act_scales: Dict[str, float],
    weight_cfg: Dict[str, Any],
    return_trace: bool = False,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Simulate one W8A8 forward pass of the fused BaselineCNN.

    Returns ``(logits, info)`` where ``info`` carries per-position activation
    saturation counts and INT32-accumulator statistics for this batch.

    When ``return_trace`` is True, ``info["trace"]`` additionally holds the
    intermediate integer nodes with the same names as the pure-integer
    reference (:mod:`onn_model.int8_reference`): ``input_q``, ``conv1_acc``,
    ``stem_q``, ``pool1_q``, ``conv2_acc``, ``conv2_q``, ``pool2_q``,
    ``conv3_acc``, ``conv3_q``, ``gap_q``, ``fc_acc``.  Accumulator tensors
    are float64 integer-valued (the fake-quant host); activation tensors are
    the actual int8 / uint8 quantised values.
    """
    info: Dict[str, Any] = {
        "activation_saturation": {
            p: {"low": 0, "high": 0} for p in ACTIVATION_POSITIONS if p != "logits"
        },
        "activation_numel": {p: 0 for p in ACTIVATION_POSITIONS if p != "logits"},
        "accumulator_max_abs": {},
        "accumulator_overflow": {},
        "trace": {},
    }
    x = x.to(torch.float32)

    def _track(p: str, q: torch.Tensor) -> None:
        # Low = negative edge (input only) / dead neuron (UINT8 q==0); benign.
        # High = true clipping at the positive edge (q==255 for UINT8, q==127
        # for the signed input) — the quantity that matters for accuracy.
        info["activation_numel"][p] += q.numel()
        if p == "input":
            lo, hi = (q == INT8_MIN), (q == INT8_MAX)
        else:
            lo, hi = (q == UINT8_MIN), (q == UINT8_MAX)
        info["activation_saturation"][p]["low"] += int(lo.sum().item())
        info["activation_saturation"][p]["high"] += int(hi.sum().item())

    # ---- input (signed INT8) ----
    s_in = act_scales["input"]
    q = quantize_signed(x, s_in)
    _track("input", q)
    if return_trace:
        info["trace"]["input_q"] = q

    def _dequant_scale(w: Dict[str, Any]) -> Any:
        """Per-channel scale reshaped for a conv (Cout -> Cout,1,1); scalar unchanged."""
        s = w["weight_scale"] * s_a
        if isinstance(s, torch.Tensor):
            return s.view(-1, 1, 1)
        return s

    # ---- stem: conv -> ReLU -> UINT8, then MaxPool ----
    s_a = s_in
    w = weight_cfg["stem_conv"]
    acc = F.conv2d(q.to(torch.float64), w["q"].to(torch.float64), stride=1, padding=1)
    acc = acc + w["q_bias"].to(torch.float64).view(1, -1, 1, 1)
    acc = _accumulate_int32(acc, "stem_conv", info)
    if return_trace:
        info["trace"]["conv1_acc"] = acc
    y = relu_fp(acc * _dequant_scale(w))
    s_next = act_scales["stem_relu"]
    q_stem = quantize_unsigned(y, s_next)
    _track("stem_relu", q_stem)
    if return_trace:
        info["trace"]["stem_q"] = q_stem
    q = F.max_pool2d(q_stem.float(), 2).to(torch.uint8)
    _track("pool1", q)
    if return_trace:
        info["trace"]["pool1_q"] = q

    # ---- conv2: conv -> ReLU -> UINT8, then MaxPool ----
    s_a = s_next  # stem_relu scale preserved through pool1
    w = weight_cfg["conv2"]
    acc = F.conv2d(q.to(torch.float64), w["q"].to(torch.float64), stride=1, padding=1)
    acc = acc + w["q_bias"].to(torch.float64).view(1, -1, 1, 1)
    acc = _accumulate_int32(acc, "conv2", info)
    if return_trace:
        info["trace"]["conv2_acc"] = acc
    y = relu_fp(acc * _dequant_scale(w))
    s_next = act_scales["conv2_relu"]
    q_conv2 = quantize_unsigned(y, s_next)
    _track("conv2_relu", q_conv2)
    if return_trace:
        info["trace"]["conv2_q"] = q_conv2
    q = F.max_pool2d(q_conv2.float(), 2).to(torch.uint8)
    _track("pool2", q)
    if return_trace:
        info["trace"]["pool2_q"] = q

    # ---- conv3: conv -> ReLU -> UINT8 ----
    s_a = s_next  # conv2_relu scale preserved through pool2
    w = weight_cfg["conv3"]
    acc = F.conv2d(q.to(torch.float64), w["q"].to(torch.float64), stride=1, padding=1)
    acc = acc + w["q_bias"].to(torch.float64).view(1, -1, 1, 1)
    acc = _accumulate_int32(acc, "conv3", info)
    if return_trace:
        info["trace"]["conv3_acc"] = acc
    y = relu_fp(acc * _dequant_scale(w))
    s_conv3 = act_scales["conv3_relu"]
    q_conv3 = quantize_unsigned(y, s_conv3)
    _track("conv3_relu", q_conv3)
    if return_trace:
        info["trace"]["conv3_q"] = q_conv3

    # ---- GAP: average the dequantised conv3 output, quantise to UINT8 ----
    s_pool = act_scales["pool"]
    a_gap = F.adaptive_avg_pool2d(dequantize(q_conv3, s_conv3), 1).flatten(1)
    q_gap = quantize_unsigned(a_gap, s_pool)
    _track("pool", q_gap)
    if return_trace:
        info["trace"]["gap_q"] = q_gap

    # ---- fc: integer matmul, output kept FP32 (logits) ----
    w = weight_cfg["fc"]
    acc = q_gap.to(torch.float64) @ w["q"].to(torch.float64).t()
    acc = acc + w["q_bias"].to(torch.float64)
    acc = _accumulate_int32(acc, "fc", info)
    if return_trace:
        info["trace"]["fc_acc"] = acc
    logits = acc * (w["weight_scale"] * s_pool)
    return logits, info


def relu_fp(x: torch.Tensor) -> torch.Tensor:
    return x.clamp(min=0.0)


# ---------------------------------------------------------------------------
# Evaluation on a loader
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate_w8a8(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    act_scales: Dict[str, float],
    weight_cfg: Dict[str, Any],
    max_mismatches: int = 20,
) -> Dict[str, Any]:
    """Evaluate W8A8 over a loader, comparing against the FP32 fused model.

    Returns accuracy, accuracy drop vs FP32, prediction agreement, per-position
    activation saturation, accumulator stats, logits errors and the first
    ``max_mismatches`` changed samples.
    """
    sat_totals = {p: {"low": 0, "high": 0} for p in ACTIVATION_POSITIONS if p != "logits"}
    sat_numel = {p: 0 for p in ACTIVATION_POSITIONS if p != "logits"}
    acc_overflow = {layer: 0 for layer in WEIGHT_LAYERS}
    acc_max_abs = {layer: 0.0 for layer in WEIGHT_LAYERS}

    total = agree = correct_fp = correct_q = 0
    sum_logits_err = 0.0
    max_logits_err = 0.0
    err_count = 0
    nan_inf = False
    mismatches: List[Dict[str, Any]] = []
    idx = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        out_fp32 = model(images)
        out_q, info = w8a8_forward(model, images, act_scales, weight_cfg)
        out_q = out_q.to(torch.float32)

        nan_inf = nan_inf or bool((~torch.isfinite(out_q)).any().item())

        pred_fp32 = out_fp32.argmax(dim=1)
        pred_q = out_q.argmax(dim=1)
        correct_fp += (pred_fp32 == labels).sum().item()
        correct_q += (pred_q == labels).sum().item()
        total += images.size(0)

        eq = pred_fp32 == pred_q
        agree += eq.sum().item()

        err = (out_fp32 - out_q).abs()
        max_logits_err = max(max_logits_err, float(err.max().item()))
        sum_logits_err += float(err.sum().item())
        err_count += err.numel()

        for p in sat_totals:
            sat_totals[p]["low"] += info["activation_saturation"][p]["low"]
            sat_totals[p]["high"] += info["activation_saturation"][p]["high"]
            sat_numel[p] += info["activation_numel"][p]
        for layer in acc_overflow:
            acc_overflow[layer] += info["accumulator_overflow"][layer]
            acc_max_abs[layer] = max(acc_max_abs[layer], info["accumulator_max_abs"][layer])

        for i in (~eq).nonzero(as_tuple=True)[0]:
            if len(mismatches) >= max_mismatches:
                break
            mismatches.append(
                {
                    "index": idx + i.item(),
                    "true_label": int(labels[i].item()),
                    "fp32_pred": int(pred_fp32[i].item()),
                    "int8_pred": int(pred_q[i].item()),
                    "fp32_logits": [float(v) for v in out_fp32[i]],
                    "int8_logits": [float(v) for v in out_q[i]],
                    "max_logits_err": float(err[i].max().item()),
                }
            )
        idx += images.size(0)

    acc_fp = correct_fp / total if total else 0.0
    acc_q = correct_q / total if total else 0.0
    return {
        "total": total,
        "fp32_accuracy": acc_fp,
        "w8a8_accuracy": acc_q,
        "accuracy_drop_pp": (acc_fp - acc_q) * 100.0,
        "agreement": agree,
        "disagree": total - agree,
        "agreement_rate": agree / total if total else 0.0,
        "max_logits_abs_err": max_logits_err,
        "mean_logits_abs_err": sum_logits_err / err_count if err_count else 0.0,
        "nan_or_inf": nan_inf,
        "activation_saturation": {
            p: {
                "low_count": sat_totals[p]["low"],
                "high_count": sat_totals[p]["high"],
                "numel": sat_numel[p],
                "low_fraction": sat_totals[p]["low"] / sat_numel[p] if sat_numel[p] else 0.0,
                "high_fraction": sat_totals[p]["high"] / sat_numel[p] if sat_numel[p] else 0.0,
            }
            for p in sat_totals
        },
        "accumulator": {
            layer: {"max_abs": acc_max_abs[layer], "overflow_count": acc_overflow[layer]}
            for layer in WEIGHT_LAYERS
        },
        "mismatches": mismatches,
    }
