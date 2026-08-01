"""Streaming 2x2 MaxPool contract tests.

Validates the pool golden vectors under ``fpga/baseline_cnn/`` against the
frozen scheme-A MaxPool semantics that ``rtl/maxpool2x2_stream.v`` must
reproduce bit-exactly:

    pool_q[oc][py][px] = max(q[oc][2py][2px], q[oc][2py][2px+1],
                              q[oc][2py+1][2px], q[oc][2py+1][2px+1])

The Questa testbench checks the RTL itself; this test independently checks the
*vectors, the CHW layout / pool-address mapping and the full 3136-element
recomputation* so a broken pool, a stale vector or a drifted reference each
fail in exactly one place.

Frozen contract (docs/rtl_microarchitecture.md, data_format.md):
* stem_q : 16x28x28 = 12544, CHW   ``addr = (oc*28+y)*28+x``
* pool1_q: 16x14x14 = 3136,  CHW   ``addr = (oc*14+py)*14+px``
* pooling is 2x2 stride 2 on unsigned UINT8, no requant (scale preserved)
* outputs stay in UINT8 [0,255]

Pure Python integer math - no torch, no repo imports - so it runs anywhere.
"""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TRACE = REPO_ROOT / "fpga" / "baseline_cnn" / "sim" / "vectors" / "golden_trace"

STEM_OC = 16
IN_H = IN_W = 28
POOL_H = POOL_W = 14
STEM_NUMEL = STEM_OC * IN_H * IN_W        # 12544
POOL_NUMEL = STEM_OC * POOL_H * POOL_W    # 3136


def _read_mem(path: Path) -> list[int]:
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


# ---- frozen address formulas ----
def stem_addr(oc: int, y: int, x: int) -> int:
    return (oc * IN_H + y) * IN_W + x


def pool_addr(oc: int, py: int, px: int) -> int:
    return (oc * POOL_H + py) * POOL_W + px


def _window(stem_q: list[int], oc: int, py: int, px: int) -> list[int]:
    """The four stem_q taps under the (oc,py,px) 2x2 window."""
    y0, x0 = 2 * py, 2 * px
    return [stem_q[stem_addr(oc, y0 + dy, x0 + dx)] for dy in range(2) for dx in range(2)]


def _pool(stem_q: list[int], oc: int, py: int, px: int) -> int:
    return max(_window(stem_q, oc, py, px))


def _load():
    stem_q = _read_mem(TRACE / "stem_q.mem")
    pool1_q = _read_mem(TRACE / "pool1_q.mem")
    return stem_q, pool1_q


def test_pool1_q_element_counts() -> None:
    """stem_q 12544 inputs, pool1_q 3136 outputs."""
    stem_q, pool1_q = _load()
    assert len(stem_q) == STEM_NUMEL
    assert len(pool1_q) == POOL_NUMEL


def test_chw_layout_and_pool_address_mapping() -> None:
    """stem CHW addr = (oc*28+y)*28+x; pool CHW addr = (oc*14+py)*14+px."""
    assert stem_addr(0, 0, 0) == 0
    assert stem_addr(1, 0, 0) == 784
    assert stem_addr(15, 27, 27) == 12543
    assert pool_addr(0, 0, 0) == 0
    assert pool_addr(1, 0, 0) == 196
    assert pool_addr(15, 13, 13) == 3135
    # pool addresses are contiguous and bijective over 0..3135
    seen = {pool_addr(oc, py, px)
            for oc in range(STEM_OC)
            for py in range(POOL_H)
            for px in range(POOL_W)}
    assert seen == set(range(POOL_NUMEL))
    # a pool window's top-left tap maps onto the stem CHW space correctly:
    # stem_addr(0, 2*py, 2*px) == (0*28 + 2*py)*28 + 2*px
    assert stem_addr(0, 0, 0) == 0                      # pool(0,0) top-left tap
    assert stem_addr(0, 26, 26) == 754                  # pool(0,13,13) top-left tap


def test_spot_windows() -> None:
    """Four corners + the last window are recomputed by hand and matched."""
    stem_q, pool1_q = _load()
    spots = [(0, 0, 0), (0, 0, 13), (0, 13, 0), (0, 13, 13), (15, 13, 13)]
    for oc, py, px in spots:
        idx = pool_addr(oc, py, px)
        assert pool1_q[idx] == _pool(stem_q, oc, py, px), (
            f"pool1_q[{idx}] oc={oc} py={py} px={px}: got {_pool(stem_q, oc, py, px)}, "
            f"golden {pool1_q[idx]}")


def test_all_values_in_uint8_range() -> None:
    """Pooling is a max of UINT8 -> outputs stay in [0,255]."""
    _, pool1_q = _load()
    assert all(0 <= v <= 255 for v in pool1_q)


def test_full_pool_recompute_matches_golden() -> None:
    """All 3136 outputs: max of the 4 window taps == pool1_q bit-exactly."""
    stem_q, pool1_q = _load()
    for oc in range(STEM_OC):
        for py in range(POOL_H):
            for px in range(POOL_W):
                idx = pool_addr(oc, py, px)
                got = _pool(stem_q, oc, py, px)
                assert got == pool1_q[idx], (
                    f"pool1_q[{idx}] oc={oc} py={py} px={px}: got {got}, golden {pool1_q[idx]}")
