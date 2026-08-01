"""RTL golden-vector contract tests.

Validates the frozen FPGA golden vectors under ``fpga/baseline_cnn/`` against
the scheme-A Int8Reference semantics that the RTL modules ``requantize_u8``
and ``gap_div49`` must reproduce bit-exactly (the Questa testbenches check the
RTL itself; this test independently checks the *vectors* against the frozen
semantics, so a broken RTL, a stale vector, or a drifted reference each fail
in exactly one place).

* requant: ``q = saturate_uint8(round_half_away_from_zero((acc * mult) >> shift))``
* gap:     ``q = saturate_uint8((sum + 24) / 49)``   (sum >= 0)

Pure Python integer math - no torch, no repo imports - so it runs anywhere.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PARAMS = REPO_ROOT / "fpga" / "baseline_cnn" / "params"
TRACE = REPO_ROOT / "fpga" / "baseline_cnn" / "sim" / "vectors" / "golden_trace"

GAP_DIVISOR = 49
REQUANT_LAYERS = {
    "stem_conv": ("conv1_acc", "stem_q"),
    "conv2": ("conv2_acc", "conv2_q"),
    "conv3": ("conv3_acc", "conv3_q"),
}


def _read_mem(path: Path, width: int) -> list[int]:
    """Read a golden ``.mem`` file (one hex word per line) into ints."""
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


def requantize(acc: int, multiplier: int, shift: int) -> int:
    """Round-half-away-from-zero right shift of ``acc * multiplier``, UINT8 clamp."""
    product = acc * multiplier
    ax = abs(product)
    if shift > 0:
        rounded = (ax + (1 << (shift - 1))) >> shift
    else:
        rounded = ax  # shift == 0: identity (no rounding term), per reference
    value = rounded if product >= 0 else -rounded
    return max(0, min(255, value))


def gap_div(sum_: int) -> int:
    """Integer sum/49 with round-half-away-from-zero, UINT8 clamp."""
    value = (sum_ + (GAP_DIVISOR >> 1)) // GAP_DIVISOR
    return max(0, min(255, value))


def test_requant_golden_vectors() -> None:
    """conv1_acc->stem_q, conv2_acc->conv2_q, conv3_acc->conv3_q (20384 total)."""
    manifest = json.loads((PARAMS / "manifest.json").read_text(encoding="utf-8"))
    expected_total = 0
    for layer, (acc_file, q_file) in REQUANT_LAYERS.items():
        rq = manifest["network"]["layers"][layer]["requant"]
        multiplier, shift = rq["multiplier"], rq["shift"]
        accs = [_signed(v, 32) for v in _read_mem(TRACE / f"{acc_file}.mem", 32)]
        qs = _read_mem(TRACE / f"{q_file}.mem", 8)
        assert len(accs) == len(qs), f"{layer}: acc/q length mismatch"
        for i, (a, q) in enumerate(zip(accs, qs)):
            got = requantize(a, multiplier, shift)
            assert got == q, f"{layer}[{i}]: acc={a} mult={multiplier} shift={shift} -> {got}, golden {q}"
        expected_total += len(accs)
    assert expected_total == 12544 + 6272 + 1568


def test_gap_exhaustive() -> None:
    """sum = 0..12495 matches the exact integer division reference."""
    for s in range(0, 12496):
        assert gap_div(s) == (s + (GAP_DIVISOR >> 1)) // GAP_DIVISOR


def test_gap_golden_trace() -> None:
    """Per-channel sums of conv3_q (32 x 7x7) -> gap_div -> gap_q golden."""
    conv3_q = _read_mem(TRACE / "conv3_q.mem", 8)
    gap_q = _read_mem(TRACE / "gap_q.mem", 8)
    assert len(conv3_q) == 32 * 49
    assert len(gap_q) == 32
    for c in range(32):
        s = sum(conv3_q[c * 49:(c + 1) * 49])
        assert gap_div(s) == gap_q[c], f"channel {c}: sum={s} -> {gap_div(s)}, golden {gap_q[c]}"
