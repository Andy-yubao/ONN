"""Stem convolution engine contract tests.

Validates the frozen stem golden vectors under ``fpga/baseline_cnn/`` against
the scheme-A Int8Reference semantics that ``rtl/stem_conv_serial.v`` must
reproduce bit-exactly.  The Questa testbench checks the RTL itself; this test
independently checks the *vectors, address formulas and padding semantics* so a
broken engine, a stale vector, or a drifted reference each fail in exactly one
place.

Frozen contract (docs/rtl_microarchitecture.md, data_format.md):
* weights:   16x1x3x3 = 144, OIHW  ``addr = ((oc*1+ic)*3+ky)*3+kx``
* inputs:    1x28x28  = 784, CHW   ``addr = y*28+x``
* outputs:   16x28x28 = 12544, CHW  ``addr = (oc*28+y)*28+x``
* acc(oc,y,x) = sum_{ky,kx} signed8(input)*signed8(weight) + signed32(bias)
* stem_q     = requantize(acc, 0x7A999012, 38)   (round-half-away-from-zero)
* padding:   out-of-bounds taps contribute 0; corners 4 / edges 6 / centre 9

Pure Python integer math - no torch, no repo imports - so it runs anywhere.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PARAMS = REPO_ROOT / "fpga" / "baseline_cnn" / "params"
TRACE = REPO_ROOT / "fpga" / "baseline_cnn" / "sim" / "vectors" / "golden_trace"

IN_H = IN_W = 28
STEM_OC = 16
KERNEL = 3
PADDING = 1
OUT_H = OUT_W = 28
IN_NUMEL = IN_H * IN_W
OUT_NUMEL = STEM_OC * OUT_H * OUT_W
STEM_WEIGHT_NUMEL = 144
STEM_BIAS_NUMEL = 16

STEM_MULT = 2056884242          # 32'h7A999012
STEM_SHIFT = 38


def _read_mem(path: Path, width: int) -> list[int]:
    """Read a ``.mem`` file (one hex word per line) into ints."""
    if not path.is_file():
        raise FileNotFoundError(f"golden vector missing: {path}")
    values = []
    for line in path.read_text(encoding="ascii").splitlines():
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        values.append(int(line, 16))
    return values


def _signed(value: int, bits: int) -> int:
    """Interpret a ``bits``-wide two's-complement bit pattern as signed."""
    value &= (1 << bits) - 1
    if value >= 1 << (bits - 1):
        value -= 1 << bits
    return value


# ---- frozen address formulas ----
def in_addr(y: int, x: int) -> int:
    return y * IN_W + x


def weight_addr(oc: int, ic: int, ky: int, kx: int) -> int:
    return ((oc * 1 + ic) * KERNEL + ky) * KERNEL + kx


def out_addr(oc: int, y: int, x: int) -> int:
    return (oc * OUT_H + y) * OUT_W + x


def requant(acc: int, multiplier: int, shift: int) -> int:
    """Round-half-away-from-zero right shift of acc*multiplier, UINT8 clamp."""
    product = acc * multiplier
    ax = abs(product)
    rounded = (ax + (1 << (shift - 1))) >> shift if shift > 0 else ax
    value = rounded if product >= 0 else -rounded
    return max(0, min(255, value))


def _load_golden():
    """Load and signed-interpret all stem golden vectors once."""
    input_q = [_signed(v, 8) for v in _read_mem(TRACE / "input_q.mem", 8)]
    weights = [_signed(v, 8) for v in _read_mem(PARAMS / "weights" / "stem_weight.mem", 8)]
    biases = [_signed(v, 32) for v in _read_mem(PARAMS / "biases" / "stem_bias.mem", 32)]
    conv1_acc = [_signed(v, 32) for v in _read_mem(TRACE / "conv1_acc.mem", 32)]
    stem_q = _read_mem(TRACE / "stem_q.mem", 8)
    return input_q, weights, biases, conv1_acc, stem_q


def _conv_acc(input_q, weights, biases, oc: int, y: int, x: int) -> int:
    """acc(oc,y,x) with exact padding semantics (out-of-bounds taps -> 0)."""
    acc = 0
    for ky in range(KERNEL):
        for kx in range(KERNEL):
            iy, ix = y + ky - PADDING, x + kx - PADDING
            if 0 <= iy < IN_H and 0 <= ix < IN_W:
                acc += input_q[in_addr(iy, ix)] * weights[weight_addr(oc, 0, ky, kx)]
    return acc + biases[oc]


def test_golden_element_counts() -> None:
    """input 784 / weight 144 / bias 16 / acc 12544 / stem_q 12544."""
    input_q, weights, biases, conv1_acc, stem_q = _load_golden()
    assert len(input_q) == IN_NUMEL
    assert len(weights) == STEM_WEIGHT_NUMEL
    assert len(biases) == STEM_BIAS_NUMEL
    assert len(conv1_acc) == OUT_NUMEL
    assert len(stem_q) == OUT_NUMEL


def test_address_formulas_bijective() -> None:
    """OIHW weight addresses cover 0..143; CHW output addresses cover 0..12543."""
    seen_w = set()
    for oc in range(STEM_OC):
        for ic in range(1):  # Cin = 1
            for ky in range(KERNEL):
                for kx in range(KERNEL):
                    seen_w.add(weight_addr(oc, ic, ky, kx))
    assert seen_w == set(range(STEM_WEIGHT_NUMEL))

    seen_o = set()
    for oc in range(STEM_OC):
        for y in range(OUT_H):
            for x in range(OUT_W):
                seen_o.add(out_addr(oc, y, x))
    assert seen_o == set(range(OUT_NUMEL))          # contiguous 0..12543


def test_valid_tap_counts() -> None:
    """Padding: corners 4, edges 6, centre 9 valid taps."""
    def valid_taps(y: int, x: int) -> int:
        n = 0
        for ky in range(KERNEL):
            for kx in range(KERNEL):
                iy, ix = y + ky - PADDING, x + kx - PADDING
                if 0 <= iy < IN_H and 0 <= ix < IN_W:
                    n += 1
        return n

    assert valid_taps(0, 0) == 4      # top-left corner
    assert valid_taps(0, 27) == 4     # top-right corner
    assert valid_taps(27, 0) == 4     # bottom-left corner
    assert valid_taps(27, 27) == 4    # bottom-right corner
    assert valid_taps(0, 14) == 6     # top edge
    assert valid_taps(14, 0) == 6     # left edge
    assert valid_taps(14, 14) == 9    # centre


def test_stem_requant_constants_match_params_header() -> None:
    """stem multiplier/shift agree with manifest.json and baseline_cnn_params.vh."""
    manifest = json.loads((PARAMS / "manifest.json").read_text(encoding="utf-8"))
    rq = manifest["network"]["layers"]["stem_conv"]["requant"]
    assert rq["multiplier"] == STEM_MULT
    assert rq["shift"] == STEM_SHIFT
    vh = (PARAMS / "baseline_cnn_params.vh").read_text(encoding="utf-8")
    assert "STEM_REQUANT_MULT  = 32'h7A999012" in vh
    assert "STEM_REQUANT_SHIFT = 38" in vh


def test_signed_int8_interpretation() -> None:
    """Weights and inputs are two's-complement S8; spot-check frozen bytes."""
    raw_in = _read_mem(TRACE / "input_q.mem", 8)
    raw_wt = _read_mem(PARAMS / "weights" / "stem_weight.mem", 8)
    assert _signed(raw_in[0], 8) == -19      # 0xED
    assert _signed(raw_wt[0], 8) == -74      # 0xB6
    # every weight/input stays inside S8
    input_q = [_signed(v, 8) for v in raw_in]
    weights = [_signed(v, 8) for v in raw_wt]
    assert all(-128 <= v <= 127 for v in input_q + weights)


def test_full_convolution_matches_golden() -> None:
    """All 12544 outputs: acc == conv1_acc AND requant(acc) == stem_q."""
    input_q, weights, biases, conv1_acc, stem_q = _load_golden()
    for oc in range(STEM_OC):
        for y in range(OUT_H):
            for x in range(OUT_W):
                idx = out_addr(oc, y, x)
                acc = _conv_acc(input_q, weights, biases, oc, y, x)
                assert acc == conv1_acc[idx], (
                    f"acc[{idx}] oc={oc} y={y} x={x}: got {acc}, golden {conv1_acc[idx]}")
                q = requant(acc, STEM_MULT, STEM_SHIFT)
                assert q == stem_q[idx], (
                    f"stem_q[{idx}] oc={oc} y={y} x={x}: got {q}, golden {stem_q[idx]}")
