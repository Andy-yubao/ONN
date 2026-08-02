"""Shared conv2/conv3 serial engine contract tests.

Validates the frozen conv2/conv3 golden vectors under ``fpga/baseline_cnn/``
against the scheme-A Int8Reference semantics that ``rtl/conv_u8_serial.v`` must
reproduce bit-exactly.  The Questa testbenches (tb_conv2_pool2 / tb_conv3_serial)
check the RTL itself; this test independently checks the *vectors, the OIHW/CHW
address formulas, the padding semantics, the UINT8 x SINT8 interpretation and
the requant constants* so a broken engine, a stale vector or a drifted reference
each fail in exactly one place.

Frozen contract (docs/rtl_microarchitecture.md, data_format.md, manifest.json):
* conv2: pool1_q 16x14x14 (UINT8) -> Conv2d 16->32, 3x3, pad1
         -> conv2_acc / conv2_q 32x14x14   (multiplier 1097020857, shift 38)
* conv3: pool2_q 32x7x7   (UINT8) -> Conv2d 32->32, 3x3, pad1
         -> conv3_acc / conv3_q 32x7x7     (multiplier 1298974956, shift 38)
* weights OIHW  ``addr = ((oc*Cin+ic)*3+ky)*3+kx``
* inputs/outputs CHW ``addr = (oc*H+y)*W+x``
* acc(oc,y,x) = sum_{ic,ky,kx} UINT8(input) * SINT8(weight) + SINT32(bias)
* q = requantize(acc, layer_mult, 38)   (round-half-away-from-zero)
* padding: out-of-bounds taps contribute 0; corners 4 / edges 6 / centre 9
* pool2_q[oc][py][px] = max of the 2x2 window over conv2_q

Pure Python integer math - no torch, no repo imports - so it runs anywhere.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PARAMS = REPO_ROOT / "fpga" / "baseline_cnn" / "params"
TRACE = REPO_ROOT / "fpga" / "baseline_cnn" / "sim" / "vectors" / "golden_trace"

KERNEL = 3
PADDING = 1
SHIFT = 38

# layer -> (Cin, H, W, Cout, weight_numel, out_numel, multiplier)
GEOM = {
    "conv2": (16, 14, 14, 32, 4608, 6272, 1097020857),
    "conv3": (32, 7, 7, 32, 9216, 1568, 1298974956),
}


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


# ---- frozen address formulas (per layer) ----
def cin_of(layer: str) -> int:
    return GEOM[layer][0]


def hw_of(layer: str) -> int:
    return GEOM[layer][1]   # H == W for both layers


def weight_addr(layer: str, oc: int, ic: int, ky: int, kx: int) -> int:
    cin, _, _, _, _, _, _ = GEOM[layer]
    return ((oc * cin + ic) * KERNEL + ky) * KERNEL + kx


def in_addr(layer: str, ic: int, y: int, x: int) -> int:
    h, w = hw_of(layer), hw_of(layer)
    return (ic * h + y) * w + x


def out_addr(layer: str, oc: int, y: int, x: int) -> int:
    h, w = hw_of(layer), hw_of(layer)
    return (oc * h + y) * w + x


def requant(acc: int, multiplier: int, shift: int) -> int:
    """Round-half-away-from-zero right shift of acc*multiplier, UINT8 clamp."""
    product = acc * multiplier
    ax = abs(product)
    rounded = (ax + (1 << (shift - 1))) >> shift if shift > 0 else ax
    value = rounded if product >= 0 else -rounded
    return max(0, min(255, value))


def _load_conv(layer: str):
    """Load input / weight / bias / acc / q for one layer."""
    in_file = "pool1_q.mem" if layer == "conv2" else "pool2_q.mem"
    in_fm = _read_mem(TRACE / in_file, 8)                        # UINT8, unsigned
    weights = [_signed(v, 8) for v in _read_mem(PARAMS / "weights" / f"{layer}_weight.mem", 8)]
    biases = [_signed(v, 32) for v in _read_mem(PARAMS / "biases" / f"{layer}_bias.mem", 32)]
    acc = [_signed(v, 32) for v in _read_mem(TRACE / f"{layer}_acc.mem", 32)]
    q = _read_mem(TRACE / f"{layer}_q.mem", 8)
    return in_fm, weights, biases, acc, q


def _conv_acc(layer: str, in_fm, weights, biases, oc: int, y: int, x: int) -> int:
    """acc(oc,y,x) with exact padding semantics (out-of-bounds taps -> 0)."""
    cin = cin_of(layer)
    h, w = hw_of(layer), hw_of(layer)
    acc = 0
    for ic in range(cin):
        for ky in range(KERNEL):
            for kx in range(KERNEL):
                iy, ix = y + ky - PADDING, x + kx - PADDING
                if 0 <= iy < h and 0 <= ix < w:
                    acc += in_fm[in_addr(layer, ic, iy, ix)] * weights[weight_addr(layer, oc, ic, ky, kx)]
    return acc + biases[oc]


# ---------------------------------------------------------------------------
def test_golden_element_counts() -> None:
    """pool1 3136 / conv2_acc+conv2_q 6272 / pool2 1568 / conv3_acc+conv3_q 1568."""
    assert len(_read_mem(TRACE / "pool1_q.mem", 8)) == 3136
    assert len(_read_mem(TRACE / "pool2_q.mem", 8)) == 1568
    for layer in ("conv2", "conv3"):
        _, _, _, acc, q = _load_conv(layer)
        assert len(acc) == GEOM[layer][5]
        assert len(q) == GEOM[layer][5]


def test_conv2_weight_layout_bijective() -> None:
    """conv2 OIHW weight addresses cover 0..4607."""
    seen = set()
    for oc in range(32):
        for ic in range(16):
            for ky in range(KERNEL):
                for kx in range(KERNEL):
                    seen.add(weight_addr("conv2", oc, ic, ky, kx))
    assert seen == set(range(4608))


def test_conv3_weight_layout_bijective() -> None:
    """conv3 OIHW weight addresses cover 0..9215."""
    seen = set()
    for oc in range(32):
        for ic in range(32):
            for ky in range(KERNEL):
                for kx in range(KERNEL):
                    seen.add(weight_addr("conv3", oc, ic, ky, kx))
    assert seen == set(range(9216))


def test_output_chw_bijective() -> None:
    """conv2 out 0..6271, conv3 out 0..1567; input CHW also bijective."""
    for layer in ("conv2", "conv3"):
        cout = GEOM[layer][3]           # position 3 = Cout (position 0 is Cin)
        out_numel = GEOM[layer][5]
        h, w = hw_of(layer), hw_of(layer)
        seen_o = {out_addr(layer, oc, y, x)
                  for oc in range(cout) for y in range(h) for x in range(w)}
        assert seen_o == set(range(out_numel)), layer
        cin = cin_of(layer)
        seen_i = {in_addr(layer, ic, y, x)
                  for ic in range(cin) for y in range(h) for x in range(w)}
        assert seen_i == set(range(cin * h * w)), layer


def test_valid_tap_counts() -> None:
    """Padding: corners 4, edges 6, centre 9 valid taps for both layers."""
    def valid_taps(layer: str, y: int, x: int) -> int:
        h, w = hw_of(layer), hw_of(layer)
        n = 0
        for ky in range(KERNEL):
            for kx in range(KERNEL):
                iy, ix = y + ky - PADDING, x + kx - PADDING
                if 0 <= iy < h and 0 <= ix < w:
                    n += 1
        return n

    for layer in ("conv2", "conv3"):
        h = hw_of(layer)
        assert valid_taps(layer, 0, 0) == 4          # top-left corner
        assert valid_taps(layer, 0, h - 1) == 4      # top-right corner
        assert valid_taps(layer, h - 1, 0) == 4      # bottom-left corner
        assert valid_taps(layer, h - 1, h - 1) == 4  # bottom-right corner
        assert valid_taps(layer, 0, 1) == 6          # top edge
        assert valid_taps(layer, 1, 0) == 6          # left edge
        assert valid_taps(layer, 1, 1) == 9          # centre


def test_uint8_input_sint8_weight_interpretation() -> None:
    """Inputs are UINT8 [0,255]; weights/bias are two's-complement S8/S32."""
    in2 = _read_mem(TRACE / "pool1_q.mem", 8)
    in3 = _read_mem(TRACE / "pool2_q.mem", 8)
    assert all(0 <= v <= 255 for v in in2 + in3), "conv inputs must be unsigned 0..255"
    for layer in ("conv2", "conv3"):
        _, weights, biases, _, _ = _load_conv(layer)
        assert all(-128 <= v <= 127 for v in weights), f"{layer} weights must fit S8"
        assert all(-(1 << 31) <= v <= (1 << 31) - 1 for v in biases), f"{layer} biases must fit S32"


def test_requant_constants_match_manifest() -> None:
    """conv2/conv3 multiplier/shift agree with manifest.json."""
    manifest = json.loads((PARAMS / "manifest.json").read_text(encoding="utf-8"))
    for layer in ("conv2", "conv3"):
        rq = manifest["network"]["layers"][layer]["requant"]
        assert rq["multiplier"] == GEOM[layer][6]
        assert rq["shift"] == SHIFT


def _spot_points(layer: str):
    h = hw_of(layer)
    return [(0, 0, 0), (0, 0, h - 1), (0, h - 1, 0), (0, h - 1, h - 1),
            (0, 0, 1), (0, 1, 0), (0, 1, 1), (0, h // 2, h // 2),
            (31, h - 1, h - 1), (1, h // 2, h // 2)]


def test_conv2_spot_outputs() -> None:
    """Hand-recomputed conv2 output points (corners/edges/centre/last)."""
    in_fm, weights, biases, acc, q = _load_conv("conv2")
    mult = GEOM["conv2"][6]
    for oc, y, x in _spot_points("conv2"):
        idx = out_addr("conv2", oc, y, x)
        a = _conv_acc("conv2", in_fm, weights, biases, oc, y, x)
        assert a == acc[idx], f"conv2 acc[{idx}] oc={oc} y={y} x={x}: got {a}, golden {acc[idx]}"
        assert requant(a, mult, SHIFT) == q[idx], f"conv2 q[{idx}] oc={oc} y={y} x={x}"


def test_conv3_spot_outputs() -> None:
    """Hand-recomputed conv3 output points (corners/edges/centre/last)."""
    in_fm, weights, biases, acc, q = _load_conv("conv3")
    mult = GEOM["conv3"][6]
    for oc, y, x in _spot_points("conv3"):
        idx = out_addr("conv3", oc, y, x)
        a = _conv_acc("conv3", in_fm, weights, biases, oc, y, x)
        assert a == acc[idx], f"conv3 acc[{idx}] oc={oc} y={y} x={x}: got {a}, golden {acc[idx]}"
        assert requant(a, mult, SHIFT) == q[idx], f"conv3 q[{idx}] oc={oc} y={y} x={x}"


def test_full_conv2_recompute_matches_golden() -> None:
    """All 6272 conv2 outputs: acc == conv2_acc AND requant(acc) == conv2_q."""
    in_fm, weights, biases, acc, q = _load_conv("conv2")
    mult = GEOM["conv2"][6]
    h = hw_of("conv2")
    for oc in range(32):
        for y in range(h):
            for x in range(h):
                idx = out_addr("conv2", oc, y, x)
                a = _conv_acc("conv2", in_fm, weights, biases, oc, y, x)
                assert a == acc[idx], f"conv2 acc[{idx}] oc={oc} y={y} x={x}"
                assert requant(a, mult, SHIFT) == q[idx], f"conv2 q[{idx}] oc={oc} y={y} x={x}"


def test_full_conv3_recompute_matches_golden() -> None:
    """All 1568 conv3 outputs: acc == conv3_acc AND requant(acc) == conv3_q."""
    in_fm, weights, biases, acc, q = _load_conv("conv3")
    mult = GEOM["conv3"][6]
    h = hw_of("conv3")
    for oc in range(32):
        for y in range(h):
            for x in range(h):
                idx = out_addr("conv3", oc, y, x)
                a = _conv_acc("conv3", in_fm, weights, biases, oc, y, x)
                assert a == acc[idx], f"conv3 acc[{idx}] oc={oc} y={y} x={x}"
                assert requant(a, mult, SHIFT) == q[idx], f"conv3 q[{idx}] oc={oc} y={y} x={x}"


def test_full_pool2_recompute_matches_golden() -> None:
    """All 1568 pool2_q outputs: 2x2 max over conv2_q == pool2_q bit-exactly."""
    conv2_q = _read_mem(TRACE / "conv2_q.mem", 8)
    pool2_q = _read_mem(TRACE / "pool2_q.mem", 8)
    assert len(pool2_q) == 1568
    for oc in range(32):
        for py in range(7):
            for px in range(7):
                idx = out_addr("conv3", oc, py, px)     # 32x7x7 CHW
                y0, x0 = 2 * py, 2 * px
                window = [conv2_q[out_addr("conv2", oc, y0 + dy, x0 + dx)]
                          for dy in range(2) for dx in range(2)]
                got = max(window)
                assert got == pool2_q[idx], (
                    f"pool2_q[{idx}] oc={oc} py={py} px={px}: got {got}, golden {pool2_q[idx]}")


def test_conv_p3_requant_and_emit_contract() -> None:
    """Shared conv captures explicit post-bias S64 and valid-qualifies EMIT."""
    src = (REPO_ROOT / "fpga" / "baseline_cnn" / "rtl" /
           "conv_u8_serial.v").read_text(encoding="utf-8")
    assert "wire signed [63:0] post_bias_acc64 = acc64 + bias64_w;" in src
    assert "token_acc64 <= post_bias_acc64;" in src
    assert "requantize_u8_pipe #(.SHIFT(CONV_SHIFT)) u_req_pipe" in src
    assert "req_in_valid = (state == S_SAT32)" in src
    assert "wire emit_valid = (state == S_EMIT) && req_out_valid;" in src
    assert "acc_valid = emit_valid" in src
    assert "q_valid   = emit_valid" in src
    assert "acc_addr  = emit_valid ? token_addr : 13'd0" in src
    assert "q_addr    = emit_valid ? token_addr : 13'd0" in src
    for name, value in (("S_SAT32", 4), ("S_MUL", 5),
                        ("S_ROUND_SHIFT_SAT", 6), ("S_EMIT", 7), ("S_DONE", 8)):
        assert f"localparam {name}" in src and f"4'd{value}" in src

    tb2 = (REPO_ROOT / "fpga" / "baseline_cnn" / "tb" /
           "tb_conv2_pool2.v").read_text(encoding="utf-8")
    tb3 = (REPO_ROOT / "fpga" / "baseline_cnn" / "tb" /
           "tb_conv3_serial.v").read_text(encoding="utf-8")
    tb_gap = (REPO_ROOT / "fpga" / "baseline_cnn" / "tb" /
              "tb_conv3_gap.v").read_text(encoding="utf-8")
    assert "first_q_cyc - start_cyc != 149" in tb2 and "940801" in tb2
    assert "first_q_cyc - start_cyc == 293" in tb3 and "460993" in tb3
    assert "C3GAP_ALL_PASS" in tb_gap and "last_gap_cyc != last_q_cyc + 1" in tb_gap
    assert "post_bias_bad" in tb2 + tb3 + tb_gap
    assert "emit_valid_bad" in tb2 + tb3 + tb_gap
    assert "!u_conv.req_out_valid" in tb2 and "!u_conv.req_out_valid" in tb3
