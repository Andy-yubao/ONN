"""Export the frozen pure-integer BaselineCNN as FPGA hardware parameters.

This module turns the frozen ``Int8Reference`` (see :mod:`onn_model.int8_reference`)
into the parameter package and ModelSim golden vectors consumed by the FPGA
phase.  It writes **no RTL** and **no UART** — only data:

    fpga/baseline_cnn/
    ├── params/
    │   ├── manifest.json             all file hashes, scales, layouts
    │   ├── baseline_cnn_params.vh    scalar constants only (no weights)
    │   ├── checksums.sha256
    │   ├── weights/                  *.mem ($readmemh) + *.mif (Quartus)
    │   └── biases/                   *.mem + *.mif
    ├── sim/vectors/
    │   ├── smoke/                    one sample per digit (input_q + fc_acc)
    │   └── golden_trace/             full per-layer trace of the digit-8 sample
    └── docs/data_format.md           bit layout / addressing reference

Parameter provenance
--------------------
Every numeric parameter is taken **verbatim** from the frozen implementation:

  * ``Int8Reference.wq``  — INT8 weights (signed, symmetric, [-127, 127]);
  * ``Int8Reference.qb``  — INT32 biases (quantised at input_scale * weight_scale);
  * ``Int8Reference.requant`` — multiplier / shift for the three convs;
  * ``candidate_quant_config.json`` — the frozen activation / weight scales.

No quantisation formula is re-derived here and no scale is recomputed.

File formats
------------
* weights   : 8-bit two's-complement, one 2-digit hex value per line;
* biases    : 32-bit two's-complement, one 8-digit hex value per line;
* addresses : 0, 1, 2, … contiguous; no comment text that would trip
  ``$readmemh`` (``.mem``) or Quartus MIF parsing.
* conv weights are laid out **OIHW** (``addr = ((oc*Cin + ic)*Kh + ky)*Kw + kx``),
  fc weights **OI** (``addr = o*F + f``), feature maps **CHW**
  (``addr = (c*H + y)*W + x``).

Determinism
-----------
The export is deterministic: no timestamps, no absolute paths, no RNG.  Two
runs produce byte-identical files (this is enforced by the verification
script).  All paths inside exported files are relative to ``fpga/baseline_cnn/``.

Usage (from the repository root):

    python model/export_baseline_cnn_hardware.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))

from onn_model.data import get_test_loader
from onn_model.int8_ptq import (
    INT32_MAX,
    INT32_MIN,
    UINT8_MAX,
    UINT8_MIN,
    WEIGHT_LAYERS,
    quantize_signed,
)
from onn_model.int8_reference import (
    ACCUMULATOR_LAYERS,
    GAP_DIVISOR,
    INT_LAYER_INPUT_ACTIVATION,
    Int8Reference,
    LAYER_OUTPUT_ACTIVATION,
    NODE_SPEC,
    TRACE_NODES,
    check_accumulator,
    round_div_away,
    round_shift,
)
from onn_model.models.baseline_cnn import BaselineCNN
from onn_model.quantization import load_bn_fused_model

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_ARTIFACT = REPO_ROOT / "model" / "artifacts" / "baseline_cnn_seed43_bn_fused_fp32.pt"
DEFAULT_CONFIG = (
    REPO_ROOT
    / "experiments"
    / "model_deployment"
    / "baseline_cnn_int8_ptq"
    / "candidate_quant_config.json"
)
DEFAULT_DATA_ROOT = REPO_ROOT / "model" / "data"
DEFAULT_OUT_DIR = REPO_ROOT / "fpga" / "baseline_cnn"

# ---------------------------------------------------------------------------
# Bit widths / geometry of the frozen design
# ---------------------------------------------------------------------------

WEIGHT_BITS = 8
BIAS_BITS = 32

# File-name stems (layer key -> file stem).
WEIGHT_FILE_NAME: Dict[str, str] = {
    "stem_conv": "stem_weight",
    "conv2": "conv2_weight",
    "conv3": "conv3_weight",
    "fc": "fc_weight",
}
BIAS_FILE_NAME: Dict[str, str] = {
    "stem_conv": "stem_bias",
    "conv2": "conv2_bias",
    "conv3": "conv3_bias",
    "fc": "fc_bias",
}

# Expected element counts / shapes of the frozen network (asserted on export).
EXPECTED_WEIGHT_NUMEL: Dict[str, int] = {
    "stem_conv": 16 * 1 * 3 * 3,  # 144
    "conv2": 32 * 16 * 3 * 3,  # 4608
    "conv3": 32 * 32 * 3 * 3,  # 9216
    "fc": 10 * 32,  # 320
}
EXPECTED_BIAS_NUMEL: Dict[str, int] = {
    "stem_conv": 16,
    "conv2": 32,
    "conv3": 32,
    "fc": 10,
}
EXPECTED_WEIGHT_SHAPE: Dict[str, List[int]] = {
    "stem_conv": [16, 1, 3, 3],
    "conv2": [32, 16, 3, 3],
    "conv3": [32, 32, 3, 3],
    "fc": [10, 32],
}
EXPECTED_BIAS_SHAPE: Dict[str, List[int]] = {
    "stem_conv": [16],
    "conv2": [32],
    "conv3": [32],
    "fc": [10],
}

INPUT_SHAPE = [1, 28, 28]
KERNEL = 3
PADDING = 1
STRIDE = 1

# Per-layer input/output shapes (CHW, excluding batch) of the frozen network.
LAYER_INOUT: Dict[str, Dict[str, List[int]]] = {
    "stem_conv": {"input": [1, 28, 28], "output": [16, 28, 28]},
    "conv2": {"input": [16, 14, 14], "output": [32, 14, 14]},
    "conv3": {"input": [32, 7, 7], "output": [32, 7, 7]},
    "fc": {"input": [32], "output": [10]},
}

# Node dtype -> (hex digit count, signed interpretation on read-back).
# int8 / int32 nodes are signed two's complement; uint8 feature maps are
# stored as plain 8-bit patterns (00..FF) and read back unsigned.
def _node_format(node: str) -> Tuple[int, bool]:
    dtype = NODE_SPEC[node][0]
    if dtype == torch.int8:
        return 2, True
    if dtype == torch.uint8:
        return 2, False
    if dtype == torch.int32:
        return 8, True
    raise ValueError(f"unexpected node dtype {dtype} for {node}")


def _hex_digits_for_node(node: str) -> int:
    return _node_format(node)[0]


# ---------------------------------------------------------------------------
# Two's-complement <-> hex
# ---------------------------------------------------------------------------


def to_twos_complement_hex(value: int, bits: int) -> str:
    """Return a zero-padded uppercase hex string of ``value`` as a ``bits``-bit
    two's-complement *pattern* (e.g. -5 in 8 bits -> ``FB``, 4_000_000_000 in
    32 bits -> ``EE6B2800``).

    ``value`` must fit in ``bits``: signed values in ``[-2^(bits-1), 2^(bits-1)-1]``
    or unsigned values in ``[0, 2^bits-1]``.  A value outside both ranges would
    silently wrap, which the frozen design forbids.
    """
    assert bits % 4 == 0
    lo, hi = -(1 << (bits - 1)), (1 << bits) - 1
    assert lo <= value <= hi, f"{value} not representable as a {bits}-bit pattern"
    mask = (1 << bits) - 1
    return f"{(value & mask):0{bits // 4}X}"


def parse_twos_complement_hex(text: str, bits: int) -> int:
    """Signed inverse of :func:`to_twos_complement_hex` (e.g. ``FB`` -> -5)."""
    v = int(text.strip(), 16)
    sign_bit = 1 << (bits - 1)
    if v & sign_bit:
        v -= 1 << bits
    return v


def parse_unsigned_hex(text: str, bits: int) -> int:
    """Unsigned inverse of :func:`to_twos_complement_hex` (``FF`` -> 255)."""
    return int(text.strip(), 16)


def int_to_hex_pair(value: int, bits: int) -> Tuple[str, int]:
    """Round-trip a signed integer through hex (helper used by tests)."""
    return to_twos_complement_hex(value, bits), parse_twos_complement_hex(
        to_twos_complement_hex(value, bits), bits
    )


# ---------------------------------------------------------------------------
# .mem / .mif writers and readers
# ---------------------------------------------------------------------------


def _write_text(path: Path, text: str) -> None:
    """Write text with explicit LF newlines (no CRLF translation).

    ``Path.write_text`` on Windows writes ``\\r\\n``, which would make the file
    bytes — and therefore every SHA256 in the manifest / checksums — platform
    dependent.  LF is used everywhere so the hashes are stable across machines
    and git checkouts.
    """
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def write_mem(path: Path, values: List[int], bits: int) -> None:
    """Write a ``$readmemh``-friendly file: one hex value per line, no comments."""
    lines = [to_twos_complement_hex(v, bits) for v in values]
    _write_text(path, "\n".join(lines) + "\n")


def write_mif(path: Path, values: List[int], bits: int, name: str) -> None:
    """Write a Quartus MIF (one ``addr : hex;`` entry per address)."""
    width = bits // 4
    lines = [
        f"-- {name}: {bits}-bit two's complement, {len(values)} entries",
        f"WIDTH = {bits};",
        f"DEPTH = {len(values)};",
        "ADDRESS_RADIX = DEC;",
        "DATA_RADIX = HEX;",
        "CONTENT",
        "BEGIN",
    ]
    for addr, v in enumerate(values):
        lines.append(f"{addr} : {v & ((1 << bits) - 1):0{width}X};")
    lines.append("END;")
    _write_text(path, "\n".join(lines) + "\n")


def read_mem(path: Path, bits: int, *, unsigned: bool = False) -> List[int]:
    """Parse a ``.mem`` file back into integers (address = index).

    ``unsigned=True`` reads the patterns as ``[0, 2^bits-1]`` (for uint8
    feature maps); otherwise they are interpreted as signed two's complement.

    Blank lines and ``//`` comments (which ``$readmemh`` accepts) are skipped.
    """
    parse = parse_unsigned_hex if unsigned else parse_twos_complement_hex
    values: List[int] = []
    for raw in path.read_text(encoding="ascii").splitlines():
        line = raw.split("//", 1)[0].strip()
        if not line:
            continue
        values.append(parse(line, bits))
    return values


def read_mif(path: Path, bits: int, *, unsigned: bool = False) -> Tuple[List[int], List[int]]:
    """Parse the Quartus MIF produced by :func:`write_mif`.

    Returns ``(addresses, values)``.  Addresses must be contiguous
    ``0..n-1`` in order; anything else raises ``ValueError``.
    """
    parse = parse_unsigned_hex if unsigned else parse_twos_complement_hex
    addresses: List[int] = []
    values: List[int] = []
    in_content = False
    for raw in path.read_text(encoding="ascii").splitlines():
        line = raw.split("--", 1)[0].strip()
        if not line:
            continue
        up = line.upper()
        if up.startswith("CONTENT"):
            in_content = True
            continue
        if up.startswith("BEGIN"):
            continue
        if up.startswith("END"):
            break
        if in_content and ":" in line:
            addr_s, _, rest = line.partition(":")
            data_s = rest.split(";", 1)[0].strip()
            addr_s = addr_s.strip()
            if addr_s.startswith("["):
                lo_s, _, hi_s = addr_s.strip("[]").partition("..")
                lo, hi = int(lo_s), int(hi_s)
                v = parse(data_s, bits)
                for a in range(lo, hi + 1):
                    addresses.append(a)
                    values.append(v)
            else:
                addresses.append(int(addr_s))
                values.append(parse(data_s, bits))
    if addresses != list(range(len(addresses))):
        raise ValueError(
            f"{path.name}: MIF addresses not contiguous 0..n-1 "
            f"(got first={addresses[:5] if addresses else 'empty'}, n={len(addresses)})"
        )
    return addresses, values


# ---------------------------------------------------------------------------
# Layout flattening (row-major = the documented address formulas)
# ---------------------------------------------------------------------------


def conv_weight_address(oc: int, ic: int, ky: int, kx: int, cin: int, kh: int, kw: int) -> int:
    """OIHW address: ``addr = ((oc*Cin + ic)*Kh + ky)*Kw + kx``."""
    return ((oc * cin + ic) * kh + ky) * kw + kx


def fc_weight_address(o: int, f: int, in_features: int) -> int:
    """OI address: ``addr = o*F + f``."""
    return o * in_features + f


def feature_map_address(c: int, y: int, x: int, h: int, w: int) -> int:
    """CHW address: ``addr = (c*H + y)*W + x``."""
    return (c * h + y) * w + x


def flatten_conv_weight_oihw(w: torch.Tensor) -> torch.Tensor:
    """Flatten a ``[Cout, Cin, Kh, Kw]`` conv weight in OIHW order.

    A row-major tensor already satisfies the OIHW formula, so this is a
    contiguous reshape; the formula is asserted by :mod:`onn_model.tests.test_hardware_export`.
    """
    assert w.ndim == 4, f"conv weight must be 4-D, got {tuple(w.shape)}"
    return w.contiguous().reshape(-1)


def flatten_fc_weight_oi(w: torch.Tensor) -> torch.Tensor:
    """Flatten a ``[O, F]`` fc weight in OI order (row-major)."""
    assert w.ndim == 2, f"fc weight must be 2-D, got {tuple(w.shape)}"
    return w.contiguous().reshape(-1)


def flatten_chw(t: torch.Tensor) -> torch.Tensor:
    """Flatten a ``[C, H, W]`` feature map (or ``[1, C, H, W]``) in CHW order."""
    if t.ndim == 4:
        assert t.shape[0] == 1, "only batch-1 tensors are exported"
        t = t[0]
    assert t.ndim == 3, f"expected [C, H, W], got {tuple(t.shape)}"
    return t.contiguous().reshape(-1)


# ---------------------------------------------------------------------------
# HardwareModel: forward rebuilt purely from exported parameters
# ---------------------------------------------------------------------------


class HardwareModel:
    """Pure-integer forward rebuilt **only** from exported hardware parameters.

    Constructed from the parsed ``.mem``/``.mif`` weight and bias files plus
    the scalar constants in ``manifest.json`` (multipliers, shifts, input
    scale) — never from the FP32 checkpoint.  The verification script uses
    this class to prove the export is self-sufficient: re-running the whole
    10,000-image test set through it must match ``Int8Reference`` prediction
    for prediction, with every golden-trace node bit-identical.

    The forward reproduces ``Int8Reference.infer`` step for step so the two
    paths share the exact integer semantics (INT64 MAC host, requantise with
    multiplier/shift, integer MaxPool, integer GAP sum/49, fc argmax on the
    INT32 accumulator).
    """

    def __init__(
        self,
        *,
        wq: Dict[str, torch.Tensor],
        qb: Dict[str, torch.Tensor],
        requant: Dict[str, Dict[str, int]],
        s_in: float,
    ) -> None:
        for layer in WEIGHT_LAYERS:
            assert layer in wq and layer in qb, layer
        for layer in ("stem_conv", "conv2", "conv3"):
            assert layer in requant, layer
        self.wq = {n: wq[n] for n in WEIGHT_LAYERS}
        self.qb = {n: qb[n] for n in WEIGHT_LAYERS}
        # only the three convs carry a requantisation multiplier (the fc is
        # argmax directly on the INT32 accumulator, never requantised)
        self.requant = {n: dict(requant[n]) for n in ("stem_conv", "conv2", "conv3")}
        self.s_in = s_in

    def _requantize_uint8(self, acc: torch.Tensor, layer: str) -> torch.Tensor:
        rq = self.requant[layer]
        prod = acc.to(torch.int64) * rq["multiplier"]  # INT64 intermediate
        q = round_shift(prod, rq["shift"])
        return q.clamp(UINT8_MIN, UINT8_MAX).to(torch.uint8)

    def _maxpool_uint8(self, q: torch.Tensor) -> torch.Tensor:
        return F.max_pool2d(q.to(torch.int32), kernel_size=2, stride=2).to(torch.uint8)

    def _conv_acc(self, x: torch.Tensor, layer: str, stats: Dict[str, Any]) -> torch.Tensor:
        acc64 = F.conv2d(x.to(torch.int64), self.wq[layer].to(torch.int64), stride=1, padding=1)
        acc64 = acc64 + self.qb[layer].to(torch.int64).view(1, -1, 1, 1)
        return check_accumulator(acc64, layer, stats)

    def _linear_acc(self, x: torch.Tensor, layer: str, stats: Dict[str, Any]) -> torch.Tensor:
        acc64 = x.to(torch.int64) @ self.wq[layer].to(torch.int64).t()
        acc64 = acc64 + self.qb[layer].to(torch.int64).view(1, -1)
        return check_accumulator(acc64, layer, stats)

    @torch.no_grad()
    def infer(
        self, x_norm: torch.Tensor, return_trace: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Same integer forward as ``Int8Reference.infer``; returns
        ``(prediction, trace)`` (no FP32 logits are needed by hardware)."""
        acc_stats: Dict[str, Any] = {
            "overflow_count": {l: 0 for l in ACCUMULATOR_LAYERS},
            "min": {l: float("inf") for l in ACCUMULATOR_LAYERS},
            "max": {l: float("-inf") for l in ACCUMULATOR_LAYERS},
        }
        q = quantize_signed(x_norm, self.s_in)

        acc1 = self._conv_acc(q, "stem_conv", acc_stats)
        stem_q = self._requantize_uint8(acc1, "stem_conv")
        pool1_q = self._maxpool_uint8(stem_q)

        acc2 = self._conv_acc(pool1_q, "conv2", acc_stats)
        conv2_q = self._requantize_uint8(acc2, "conv2")
        pool2_q = self._maxpool_uint8(conv2_q)

        acc3 = self._conv_acc(pool2_q, "conv3", acc_stats)
        conv3_q = self._requantize_uint8(acc3, "conv3")

        gap_sum = conv3_q.to(torch.int32).sum(dim=(2, 3))
        gap_q = round_div_away(gap_sum, GAP_DIVISOR)
        gap_q = gap_q.clamp(UINT8_MIN, UINT8_MAX).to(torch.uint8)

        fc_acc = self._linear_acc(gap_q, "fc", acc_stats)
        prediction = fc_acc.argmax(dim=1)

        if not return_trace:
            return prediction
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
        return prediction, trace


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, obj: Any) -> None:
    _write_text(path, json.dumps(obj, indent=2, sort_keys=True) + "\n")


# ---------------------------------------------------------------------------
# Smoke / golden-trace sample selection
# ---------------------------------------------------------------------------


def _select_smoke_samples(
    ref: Int8Reference, test_loader
) -> Dict[int, Dict[str, Any]]:
    """Pick, for each digit 0..9, the *smallest* MNIST test index that the
    integer reference classifies correctly (prediction == label == digit)."""
    found: Dict[int, Dict[str, Any]] = {}
    idx = 0
    for images, labels in test_loader:
        pred, _, _ = ref.infer(images, return_trace=False)
        labels = labels.tolist()
        pred = pred.tolist()
        for i in range(len(labels)):
            d = labels[i]
            if d in found:
                continue
            if pred[i] == d:
                found[d] = {
                    "test_index": idx + i,
                    "image": images[i].clone(),
                    "true_label": d,
                    "prediction": d,
                }
        if len(found) == 10:
            break
        idx += len(labels)
    if len(found) != 10:
        missing = sorted(set(range(10)) - set(found))
        raise RuntimeError(f"could not find a correct digit sample for {missing}")
    return found


def _write_smoke_vector(sample_dir: Path, trace: Dict[str, Any]) -> Dict[str, Any]:
    sample_dir.mkdir(parents=True, exist_ok=True)
    input_q = flatten_chw(trace["input_q"])  # [1,28,28] -> 784, 2-digit hex
    fc_acc = trace["fc_acc"].reshape(-1)  # [1,10] -> 10, 8-digit hex
    write_mem(sample_dir / "input_q.mem", input_q.tolist(), 8)
    write_mem(sample_dir / "fc_acc.mem", fc_acc.tolist(), 32)
    return {
        "input_q": {
            "file": "input_q.mem",
            "bit_width": 8,
            "numel": int(input_q.numel()),
            "sha256": sha256_file(sample_dir / "input_q.mem"),
        },
        "fc_acc": {
            "file": "fc_acc.mem",
            "bit_width": 32,
            "numel": int(fc_acc.numel()),
            "sha256": sha256_file(sample_dir / "fc_acc.mem"),
        },
    }


def _write_golden_trace(
    trace_dir: Path, ref: Int8Reference, image: torch.Tensor
) -> Tuple[Dict[str, Any], torch.Tensor]:
    """Write the full per-layer trace of one sample; return (node meta, prediction)."""
    trace_dir.mkdir(parents=True, exist_ok=True)
    pred, _, trace = ref.infer(image.unsqueeze(0), return_trace=True)
    meta: Dict[str, Any] = {}
    for node in TRACE_NODES:
        t = trace[node]
        bits = _hex_digits_for_node(node) * 4
        flat = t.reshape(-1).tolist()
        write_mem(trace_dir / f"{node}.mem", flat, bits)
        meta[node] = {
            "file": f"{node}.mem",
            "shape": list(t.shape),
            "dtype": str(t.dtype),
            "bit_width": bits,
            "numel": int(t.numel()),
            "min": int(t.min().item()),
            "max": int(t.max().item()),
            "sha256": sha256_file(trace_dir / f"{node}.mem"),
        }
    prediction = int(pred[0].item())
    _write_text(trace_dir / "prediction.txt", f"{prediction}\n")
    return meta, prediction


# ---------------------------------------------------------------------------
# baseline_cnn_params.vh
# ---------------------------------------------------------------------------


def _params_header(requant: Dict[str, Dict[str, Any]]) -> str:
    """Scalar-only Verilog parameter header (no weight arrays)."""
    def rq(line: str) -> Dict[str, Any]:
        return requant[line]

    def mult_hex(layer: str) -> str:
        m = rq(layer)["multiplier"]
        return f"32'h{m:08X}"

    lines = [
        "// baseline_cnn_params.vh - BaselineCNN (scheme A) hardware parameter header.",
        "// Generated by model/export_baseline_cnn_hardware.py. Do not hand-edit.",
        "// Every multiplier is positive and fits a signed 32-bit value (31-bit magnitude).",
        "",
        "// ---- input geometry ----",
        "localparam IN_CH = 1;",
        "localparam IN_H  = 28;",
        "localparam IN_W  = 28;",
        "",
        "// ---- conv / pool geometry (3x3, pad 1, stride 1; 2x2 maxpool) ----",
        "localparam KERNEL       = 3;",
        "localparam PADDING      = 1;",
        "localparam STEM_OC      = 16;",
        "localparam STEM_H       = 28;",
        "localparam STEM_W       = 28;",
        "localparam STEM_POOL_H  = 14;",
        "localparam STEM_POOL_W  = 14;",
        "localparam CONV2_OC     = 32;",
        "localparam CONV2_H      = 14;",
        "localparam CONV2_W      = 14;",
        "localparam CONV2_POOL_H = 7;",
        "localparam CONV2_POOL_W = 7;",
        "localparam CONV3_OC     = 32;",
        "localparam CONV3_H      = 7;",
        "localparam CONV3_W      = 7;",
        "",
        "// ---- global average pool ----",
        "localparam GAP_CH       = 32;   // conv3 output channels (GAP input)",
        "localparam GAP_DIVISOR  = 49;   // integer sum / 49, round-half-away-from-zero",
        "",
        "// ---- fc ----",
        "localparam FC_OC  = 10;",
        "localparam FC_IC  = 32;   // = GAP_CH",
        "",
        "// ---- fixed-point requantisation (acc * mult) >> shift -------",
        f"localparam STEM_REQUANT_MULT  = {mult_hex('stem_conv')};  // {rq('stem_conv')['multiplier']}",
        f"localparam STEM_REQUANT_SHIFT = {rq('stem_conv')['shift']};",
        f"localparam CONV2_REQUANT_MULT = {mult_hex('conv2')};  // {rq('conv2')['multiplier']}",
        f"localparam CONV2_REQUANT_SHIFT = {rq('conv2')['shift']};",
        f"localparam CONV3_REQUANT_MULT = {mult_hex('conv3')};  // {rq('conv3')['multiplier']}",
        f"localparam CONV3_REQUANT_SHIFT = {rq('conv3')['shift']};",
        "",
        "// ---- memory depths (weights OIHW / fc OI; biases linear) ----",
        "localparam STEM_WEIGHT_DEPTH  = 144;",
        "localparam CONV2_WEIGHT_DEPTH = 4608;",
        "localparam CONV3_WEIGHT_DEPTH = 9216;",
        "localparam FC_WEIGHT_DEPTH    = 320;",
        "localparam STEM_BIAS_DEPTH    = 16;",
        "localparam CONV2_BIAS_DEPTH   = 32;",
        "localparam CONV3_BIAS_DEPTH   = 32;",
        "localparam FC_BIAS_DEPTH      = 10;",
        "",
        "// ---- integer range bounds (clamp-saturate, no wraparound) ----",
        "localparam ACC_INT32_MIN = -32'h80000000;",
        "localparam ACC_INT32_MAX =  32'h7FFFFFFF;",
        "localparam UINT8_MAX_VAL = 8'hFF;",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Top-level export
# ---------------------------------------------------------------------------


def export_all(
    ref: Int8Reference,
    config: Dict[str, Any],
    test_loader,
    out_dir: Path,
) -> List[str]:
    """Write the complete export tree under ``out_dir``.

    Returns the sorted list of generated file paths relative to ``out_dir``.
    Fully deterministic: identical content on every run, no absolute paths.
    """
    out_dir = Path(out_dir)
    weights_dir = out_dir / "params" / "weights"
    biases_dir = out_dir / "params" / "biases"
    smoke_dir = out_dir / "sim" / "vectors" / "smoke"
    trace_dir = out_dir / "sim" / "vectors" / "golden_trace"
    for d in (weights_dir, biases_dir, smoke_dir, trace_dir):
        d.mkdir(parents=True, exist_ok=True)

    # ---- validate frozen geometry ---------------------------------------
    for layer in WEIGHT_LAYERS:
        w = ref.wq[layer]
        b = ref.qb[layer]
        assert tuple(w.shape) == tuple(EXPECTED_WEIGHT_SHAPE[layer]), (
            f"{layer} weight shape {tuple(w.shape)} != {EXPECTED_WEIGHT_SHAPE[layer]}"
        )
        assert tuple(b.shape) == tuple(EXPECTED_BIAS_SHAPE[layer]), (
            f"{layer} bias shape {tuple(b.shape)} != {EXPECTED_BIAS_SHAPE[layer]}"
        )
        assert w.numel() == EXPECTED_WEIGHT_NUMEL[layer]
        assert b.numel() == EXPECTED_BIAS_NUMEL[layer]
        assert w.dtype == torch.int8
        assert b.dtype == torch.int32

    # ---- 1. select smoke samples (digit 8 sample drives the golden trace) ----
    smoke = _select_smoke_samples(ref, test_loader)

    # ---- 2. weights / biases (.mem + .mif) -------------------------------
    weight_meta: Dict[str, Any] = {}
    bias_meta: Dict[str, Any] = {}
    for layer in WEIGHT_LAYERS:
        w = ref.wq[layer]
        b = ref.qb[layer]
        stem = WEIGHT_FILE_NAME[layer]
        bstem = BIAS_FILE_NAME[layer]
        w_flat = flatten_conv_weight_oihw(w) if w.ndim == 4 else flatten_fc_weight_oi(w)
        b_flat = b.contiguous().reshape(-1)
        write_mem(weights_dir / f"{stem}.mem", w_flat.tolist(), WEIGHT_BITS)
        write_mif(weights_dir / f"{stem}.mif", w_flat.tolist(), WEIGHT_BITS, stem)
        write_mem(biases_dir / f"{bstem}.mem", b_flat.tolist(), BIAS_BITS)
        write_mif(biases_dir / f"{bstem}.mif", b_flat.tolist(), BIAS_BITS, bstem)
        weight_meta[layer] = {
            "file": f"params/weights/{stem}.mem",
            "mif": f"params/weights/{stem}.mif",
            "shape": list(w.shape),
            "layout": "OIHW" if w.ndim == 4 else "OI",
            "bit_width": WEIGHT_BITS,
            "numel": int(w.numel()),
            "sha256": sha256_file(weights_dir / f"{stem}.mem"),
        }
        bias_meta[layer] = {
            "file": f"params/biases/{bstem}.mem",
            "mif": f"params/biases/{bstem}.mif",
            "shape": list(b.shape),
            "bit_width": BIAS_BITS,
            "numel": int(b.numel()),
            "sha256": sha256_file(biases_dir / f"{bstem}.mem"),
        }

    # ---- 3. golden trace (full per-layer trace of the digit-8 sample) ------
    trace_node_meta, trace_pred = _write_golden_trace(trace_dir, ref, smoke[8]["image"])

    # ---- 4. smoke vectors (input_q + fc_acc for each digit) ----------------
    smoke_samples_meta: List[Dict[str, Any]] = []
    for d in range(10):
        s = smoke[d]
        _, _, trace = ref.infer(s["image"].unsqueeze(0), return_trace=True)
        files = _write_smoke_vector(
            smoke_dir / f"digit{d}_idx{s['test_index']:05d}", trace
        )
        pred_d = int(trace["fc_acc"].argmax(dim=1)[0].item())
        smoke_samples_meta.append(
            {
                "digit": d,
                "test_index": s["test_index"],
                "true_label": s["true_label"],
                "prediction": pred_d,
                "dir": f"digit{d}_idx{s['test_index']:05d}",
                **files,
            }
        )

    # ---- 5. smoke / trace manifests ---------------------------------------
    smoke_manifest = {
        "count": len(smoke_samples_meta),
        "selection_rule": (
            "smallest MNIST test index whose digit d is classified correctly by "
            "the frozen integer reference (prediction == label == d)"
        ),
        "input_preprocessing": (
            "PC-side: MNIST ToTensor + Normalize(0.1307,0.3081) then quantise to "
            "INT8 (input_q). FPGA v1 does no float normalisation."
        ),
        "samples": smoke_samples_meta,
    }
    _write_json(smoke_dir / "smoke_manifest.json", smoke_manifest)

    trace_manifest = {
        "sample_index": smoke[8]["test_index"],
        "true_label": 8,
        "prediction": trace_pred,
        "source": "digit-8 smoke sample",
        "nodes": trace_node_meta,
        "prediction_file": "prediction.txt",
    }
    _write_json(trace_dir / "trace_manifest.json", trace_manifest)

    # ---- 6. params header (.vh) --------------------------------------------
    vh_text = _params_header(ref.requant)
    vh_path = out_dir / "params" / "baseline_cnn_params.vh"
    _write_text(vh_path, vh_text)

    # ---- 7. top-level manifest ----------------------------------------------
    manifest = _build_manifest(
        ref=ref,
        config=config,
        weight_meta=weight_meta,
        bias_meta=bias_meta,
        smoke_manifest_file="sim/vectors/smoke/smoke_manifest.json",
        trace_manifest_file="sim/vectors/golden_trace/trace_manifest.json",
        trace_node_meta=trace_node_meta,
        trace_pred=trace_pred,
        trace_sample_index=smoke[8]["test_index"],
        vh_sha256=sha256_file(vh_path),
        vh_file="params/baseline_cnn_params.vh",
        smoke_count=len(smoke_samples_meta),
    )
    manifest_path = out_dir / "params" / "manifest.json"
    _write_json(manifest_path, manifest)

    # ---- 8. checksums (all generated files, excluding itself) ----------------
    generated = _collect_generated(out_dir)
    checksum_lines = [
        f"{sha256_file(out_dir / rel)}  {rel}" for rel in generated
    ]
    _write_text(out_dir / "params" / "checksums.sha256", "\n".join(checksum_lines) + "\n")
    generated.append("params/checksums.sha256")
    generated.sort()

    return generated


def _build_manifest(
    *,
    ref: Int8Reference,
    config: Dict[str, Any],
    weight_meta: Dict[str, Any],
    bias_meta: Dict[str, Any],
    smoke_manifest_file: str,
    trace_manifest_file: str,
    trace_node_meta: Dict[str, Any],
    trace_pred: int,
    trace_sample_index: int,
    vh_sha256: str,
    vh_file: str,
    smoke_count: int,
) -> Dict[str, Any]:
    act_scale = config["activation_scale"]
    weight_scale = config["weight_scale"]

    requant_out: Dict[str, Any] = {}
    for layer in ("stem_conv", "conv2", "conv3"):
        rq = ref.requant[layer]
        requant_out[layer] = {
            "multiplier": int(rq["multiplier"]),
            "shift": int(rq["shift"]),
            "bit_width": 32,
            "signed": True,
            "real_multiplier": float(rq["real_multiplier"]),
            "relative_error": float(rq["relative_error"]),
        }

    layers_out: Dict[str, Any] = {}
    for layer in WEIGHT_LAYERS:
        in_pos = INT_LAYER_INPUT_ACTIVATION[layer]
        entry = {
            "input_shape": LAYER_INOUT[layer]["input"],
            "output_shape": LAYER_INOUT[layer]["output"],
            "weight_scale": float(weight_scale[layer]),
            "activation_input_scale": float(act_scale[in_pos]),
            "weight_file": weight_meta[layer]["file"],
            "bias_file": bias_meta[layer]["file"],
        }
        if layer != "fc":
            out_pos = LAYER_OUTPUT_ACTIVATION[layer]
            entry.update(
                {
                    "activation_output_scale": float(act_scale[out_pos]),
                    "kernel": [KERNEL, KERNEL],
                    "padding": PADDING,
                    "stride": STRIDE,
                    "requant": requant_out[layer],
                }
            )
        else:
            entry["argmax_directly_on_int32_accumulator"] = True
        layers_out[layer] = entry

    return {
        "format": {
            "name": "baseline_cnn_hardware_params",
            "version": 1,
            "model": "BaselineCNN",
            "scheme": "scheme_a",
            "source": "Int8Reference (frozen pure-integer reference)",
        },
        "layouts": {
            "conv_weight": "OIHW",
            "fc_weight": "OI",
            "feature_map": "CHW",
        },
        "dtypes": {
            "weight": {"name": "sint8", "bit_width": 8, "min": -127, "max": 127, "zero_point": 0},
            "input": {"name": "sint8", "bit_width": 8, "min": -128, "max": 127, "zero_point": 0},
            "relu_output": {"name": "uint8", "bit_width": 8, "min": 0, "max": 255, "zero_point": 0},
            "bias": {"name": "sint32", "bit_width": 32, "min": INT32_MIN, "max": INT32_MAX},
            "accumulator": "int32",
        },
        "network": {
            "input_shape": INPUT_SHAPE,
            "input_scale": float(act_scale["input"]),
            "input_preprocessing": (
                "PC-side: MNIST ToTensor + Normalize(0.1307, 0.3081) then quantise "
                "to INT8 (input_q); FPGA v1 performs no float normalisation"
            ),
            "layers": layers_out,
            "maxpool": {"type": "2x2 stride-2", "scale_preserved": True},
            "gap": {
                "type": "integer_sum_div",
                "divisor": GAP_DIVISOR,
                "rounding": "round-half-away-from-zero",
                "output_unit": "conv3_relu",
            },
            "fc_input_activation": "conv3_relu",
            "rounding": "round-half-away-from-zero",
            "saturation": "clamp-saturate-no-wraparound",
            "accumulator_requant_order": "acc * multiplier (INT64) -> round-half-away-from-zero right shift -> saturate to UINT8",
        },
        "weights": weight_meta,
        "biases": bias_meta,
        "params_header": {"file": vh_file, "sha256": vh_sha256},
        "smoke_vectors": {
            "manifest": smoke_manifest_file,
            "count": smoke_count,
        },
        "golden_trace": {
            "sample_index": trace_sample_index,
            "true_label": 8,
            "prediction": trace_pred,
            "manifest": trace_manifest_file,
            "nodes": trace_node_meta,
        },
        "checksums": "params/checksums.sha256",
    }


def _collect_generated(out_dir: Path) -> List[str]:
    """All generated data files under ``out_dir`` (relative posix paths), sorted.

    ``checksums.sha256`` is added afterwards by :func:`export_all`.  ``docs/``
    is authored documentation (``data_format.md``) rather than generated data,
    so it is deliberately excluded from the controlled-file set.
    """
    rels: List[str] = []
    for p in out_dir.rglob("*"):
        if p.is_file() and p.name != "checksums.sha256":
            rel = p.relative_to(out_dir).as_posix()
            if rel.startswith("docs/"):
                continue
            rels.append(rel)
    return sorted(rels)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export BaselineCNN integer parameters and golden vectors"
    )
    parser.add_argument("--artifact", type=str, default=str(DEFAULT_ARTIFACT))
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--data-root", type=str, default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args()

    device = torch.device("cpu")
    model, meta = load_bn_fused_model(args.artifact, BaselineCNN, device)
    model.eval()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    ref = Int8Reference(model, config)
    test_loader = get_test_loader(root=args.data_root, batch_size=128, num_workers=0)

    print("=" * 78)
    print("BaselineCNN hardware parameter export")
    print("=" * 78)
    print(f"artifact : {args.artifact}")
    print(f"config   : {args.config}")
    print(f"model    : {meta['model_name']}")

    generated = export_all(ref, config, test_loader, Path(args.out_dir))
    print(f"\nexported {len(generated)} files under {Path(args.out_dir)}")
    for rel in generated:
        print(f"  {rel}")
    print("\nexport OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
