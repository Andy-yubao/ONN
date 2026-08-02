"""Complete BaselineCNN compute-core contract tests.

Validates the frozen end-to-end golden vectors
(``input_q -> conv1_acc -> stem_q -> pool1_q -> conv2_acc -> conv2_q -> pool2_q
 -> conv3_acc -> conv3_q -> gap_q -> fc_acc -> prediction``) against the
scheme-A Int8Reference semantics that ``rtl/baseline_cnn_core.v`` must reproduce
bit-exactly.  ``tb_baseline_cnn_core.v`` checks the RTL itself; this test
independently checks:

* the golden node element counts (784/12544/3136/6272/1568/1568/32/10/1);
* a full 32-channel GAP recompute from conv3_q (round-half-away-from-zero /49);
* a full 10-class FC recompute from gap_q (S64 accumulation, no requant);
* the signed argmax prediction with the "tie -> smaller class" rule, including
  the hand-built 5/7 tie vector (gap[0]=82 -> both 6891 -> prediction 5);
* the controller stage order (IDLE..STEM..CONV2..CONV3..FC..DONE);
* the pool2 RAM capacity/addresses (1568 cells, ADDR_W 11, addresses 0..1567);
* the storage limit: no complete stem_q / conv2_q / conv3_q RAM anywhere;
* the corrected multiplier hex comments (32'h416335B9 / 32'h4D6CC8EC);
* the unselected weight-ROM address gating (wt2_read / wt3_read);
* the full-network theoretical output counts.

Pure Python integer math - no torch, no repo imports - so it runs anywhere.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PARAMS = REPO_ROOT / "fpga" / "baseline_cnn" / "params"
TRACE = REPO_ROOT / "fpga" / "baseline_cnn" / "sim" / "vectors" / "golden_trace"
RTL = REPO_ROOT / "fpga" / "baseline_cnn" / "rtl"

SHIFT = 38
MULT = {  # conv2 / conv3 (and hex, frozen in baseline_cnn_params.vh)
    "conv2": (1097020857, 0x416335B9),
    "conv3": (1298974956, 0x4D6CC8EC),
}
GAP_DIV = 49


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
    value &= (1 << bits) - 1
    if value >= 1 << (bits - 1):
        value -= 1 << bits
    return value


def _read_rtl(name: str) -> str:
    return (RTL / name).read_text(encoding="utf-8")


def _signed_argmax(values: list[int]) -> int:
    """Signed argmax with ties resolved to the smaller class index."""
    best, pred = values[0], 0
    for i in range(1, len(values)):
        if values[i] > best:      # strictly greater: an equal score keeps i-1
            best, pred = values[i], i
    return pred


# ---------------------------------------------------------------------------
def test_golden_node_element_counts() -> None:
    """Every golden node has the frozen element count."""
    assert len(_read_mem(TRACE / "input_q.mem", 8)) == 784
    assert len(_read_mem(TRACE / "conv1_acc.mem", 32)) == 12544
    assert len(_read_mem(TRACE / "stem_q.mem", 8)) == 12544
    assert len(_read_mem(TRACE / "pool1_q.mem", 8)) == 3136
    assert len(_read_mem(TRACE / "conv2_acc.mem", 32)) == 6272
    assert len(_read_mem(TRACE / "conv2_q.mem", 8)) == 6272
    assert len(_read_mem(TRACE / "pool2_q.mem", 8)) == 1568
    assert len(_read_mem(TRACE / "conv3_acc.mem", 32)) == 1568
    assert len(_read_mem(TRACE / "conv3_q.mem", 8)) == 1568
    assert len(_read_mem(TRACE / "gap_q.mem", 8)) == 32
    assert len(_read_mem(TRACE / "fc_acc.mem", 32)) == 10


def test_full_network_output_counts() -> None:
    """The full network's per-stage output counts are the frozen numbers."""
    stages = [
        ("input -> conv1_acc / stem_q", 784, 12544),
        ("stem_q -> pool1_q", 12544, 3136),
        ("pool1_q -> conv2", 3136, 6272),
        ("conv2_q -> pool2_q", 6272, 1568),
        ("pool2_q -> conv3", 1568, 1568),
        ("conv3_q -> gap_q", 1568, 32),
        ("gap_q -> fc_acc", 32, 10),
        ("fc_acc -> prediction", 10, 1),
    ]
    for name, n_in, n_out in stages:
        assert n_out == {
            "input -> conv1_acc / stem_q": 12544,
            "stem_q -> pool1_q": 3136,
            "pool1_q -> conv2": 6272,
            "conv2_q -> pool2_q": 1568,
            "pool2_q -> conv3": 1568,
            "conv3_q -> gap_q": 32,
            "gap_q -> fc_acc": 10,
            "fc_acc -> prediction": 1,
        }[name], name
        assert n_in > 0 and n_out > 0


# ---------------------------------------------------------------------------
def test_gap_full_32channel_recompute() -> None:
    """gap_q[c] == round_half_away_from_zero(sum(conv3_q[c]) / 49), all 32."""
    conv3_q = _read_mem(TRACE / "conv3_q.mem", 8)
    gap_q = _read_mem(TRACE / "gap_q.mem", 8)
    assert len(conv3_q) == 1568 and len(gap_q) == 32
    for c in range(32):
        s = sum(conv3_q[c * 49:(c + 1) * 49])       # CHW: channel-major, 49 per channel
        got = (s + (GAP_DIV >> 1)) // GAP_DIV       # floor((s+24)/49), s >= 0
        assert got == gap_q[c], f"gap_q[{c}]: sum={s} got {got}, golden {gap_q[c]}"


def test_gap_sum_fits_14bits() -> None:
    """Every per-channel GAP sum fits the 14-bit accumulator (<= 12495)."""
    conv3_q = _read_mem(TRACE / "conv3_q.mem", 8)
    for c in range(32):
        s = sum(conv3_q[c * 49:(c + 1) * 49])
        assert 0 <= s <= 12495, f"channel {c} sum {s} exceeds 14-bit range"


# ---------------------------------------------------------------------------
def test_fc_full_10class_recompute() -> None:
    """fc_acc[o] == sum_f gap_q[f]*fc_weight[o*32+f] + fc_bias[o], all 10."""
    gap_q = _read_mem(TRACE / "gap_q.mem", 8)
    fc_acc = [_signed(v, 32) for v in _read_mem(TRACE / "fc_acc.mem", 32)]
    weights = [_signed(v, 8) for v in _read_mem(PARAMS / "weights" / "fc_weight.mem", 8)]
    biases = [_signed(v, 32) for v in _read_mem(PARAMS / "biases" / "fc_bias.mem", 32)]
    assert len(weights) == 320 and len(biases) == 10 and len(gap_q) == 32
    for o in range(10):
        acc = sum(gap_q[f] * weights[o * 32 + f] for f in range(32)) + biases[o]
        # exact S64 accumulation never exceeds INT32 for these weights
        assert -(1 << 31) <= acc <= (1 << 31) - 1, f"fc_acc[{o}] {acc} exceeds INT32"
        assert acc == fc_acc[o], f"fc_acc[{o}]: got {acc}, golden {fc_acc[o]}"


def test_prediction_is_signed_argmax_of_fc_acc() -> None:
    """prediction == signed argmax over the golden fc_acc (golden trace = 8)."""
    fc_acc = [_signed(v, 32) for v in _read_mem(TRACE / "fc_acc.mem", 32)]
    pred = int((TRACE / "prediction.txt").read_text(encoding="ascii").strip(), 10)
    assert _signed_argmax(fc_acc) == pred == 8


def test_tie_rule_selects_smaller_class() -> None:
    """Hand-built tie (gap[0]=82) makes classes 5 and 7 tie at 6891 -> pred 5."""
    gap_q = [0] * 32
    gap_q[0] = 82
    weights = [_signed(v, 8) for v in _read_mem(PARAMS / "weights" / "fc_weight.mem", 8)]
    biases = [_signed(v, 32) for v in _read_mem(PARAMS / "biases" / "fc_bias.mem", 32)]
    acc = [sum(gap_q[f] * weights[o * 32 + f] for f in range(32)) + biases[o]
           for o in range(10)]
    # the tie really is a tie and strictly dominates every other class
    assert acc[5] == acc[7] == 6891
    assert all(acc[c] < 6891 for c in (0, 1, 2, 3, 4, 6, 8, 9))
    assert _signed_argmax(acc) == 5            # smaller class index wins the tie
    assert _signed_argmax(acc) != 7


# ---------------------------------------------------------------------------
def test_controller_stage_order() -> None:
    """baseline_cnn_core.v: stages defined 0..9 and FSM advances in order."""
    src = _read_rtl("baseline_cnn_core.v")
    names = [
        "ST_IDLE", "ST_STEM_START", "ST_STEM_WAIT", "ST_CONV2_START",
        "ST_CONV2_WAIT", "ST_CONV3_START", "ST_CONV3_WAIT", "ST_FC_START",
        "ST_FC_WAIT", "ST_DONE",
    ]
    # localparams must be assigned 0..9 in order
    for i, name in enumerate(names):
        assert re.search(rf"localparam {name}\s*=\s*4'd{i}", src), (
            f"{name} must be localparam 4'd{i}")
    # the FSM case must list the stages in the frozen order
    case_positions = [src.index(f"begin : stage_{s}" if False else f"{s}: begin") for s in
                      ("ST_STEM_START", "ST_STEM_WAIT", "ST_CONV2_START",
                       "ST_CONV2_WAIT", "ST_CONV3_START", "ST_CONV3_WAIT",
                       "ST_FC_START", "ST_FC_WAIT", "ST_DONE")]
    assert case_positions == sorted(case_positions)
    # frozen transition checks
    assert "if (stem_done_w) state <= ST_CONV2_START;" in src
    assert "if (pool2_done_w) state <= ST_CONV3_START;" in src
    assert "if (gap_done_w) state <= ST_FC_START;" in src
    assert "if (fc_done_w)" in src
    assert "state <= ST_DONE;" in src
    assert "done <= 1'b1;" in src
    assert "state <= ST_IDLE;" in src


def test_co_start_and_co_done_rules() -> None:
    """conv2+pool2 and conv3+GAP are started together; done partners are 1:1."""
    src = _read_rtl("baseline_cnn_core.v")
    assert "conv_start   = (state == ST_CONV2_START) || (state == ST_CONV3_START)" in src
    assert "pool2_start  = (state == ST_CONV2_START)" in src
    assert "gap_start    = (state == ST_CONV3_START)" in src
    assert "fc_start     = (state == ST_FC_START)" in src
    # GAP input gated to the conv3 phase only
    assert "gap_in_valid = conv_q_valid_w && conv3_phase" in src


def test_pool2_ram_capacity_and_addresses() -> None:
    """pool2 RAM: 1568 cells, 11-bit address (0..1567), the only conv FM store."""
    src = _read_rtl("baseline_cnn_core.v")
    assert re.search(r"sync_ram_u8 #\(\.DEPTH\(1568\), \.ADDR_W\(11\)\) u_pool2_ram", src)
    assert ".raddr (pool2_raddr)" in src
    assert "pool2_raddr = conv3_phase ? conv_fm_raddr_w[10:0] : 11'd0" in src
    # addresses strictly 0..1567 is covered by the golden recompute below


def test_no_complete_feature_map_rams() -> None:
    """No full stem_q (12544) / conv2_q (6272) / conv3_q (1568) RAM anywhere."""
    pipe = _read_rtl("stem_pool1_pipeline.v")
    core = _read_rtl("baseline_cnn_core.v")
    # the pipeline turns OFF the stem output RAM
    assert "STORE_OUTPUT_RAM(0)" in pipe
    # no RAM with a full stem_q / conv2_q / conv3_q depth in either file
    assert ".DEPTH(12544)" not in pipe and ".DEPTH(12544)" not in core
    assert ".DEPTH(6272)" not in pipe and ".DEPTH(6272)" not in core
    # the only sync_ram_u8 in the core is the pool2_q RAM (1568); conv3_q is
    # streamed straight into the GAP, conv2_q straight into pool2
    core_rams = re.findall(r"sync_ram_u8 #\(\.DEPTH\((\d+)\)", core)
    assert core_rams == ["1568"], f"core must hold only pool2 RAM, got {core_rams}"
    # conv_q never writes any RAM (the engine exposes q_valid/q_value streams only)
    assert "u_pool2_ram" in core and "u_gap" in core and "u_fc" in core


def test_multiplier_hex_comments() -> None:
    """conv_u8_serial.v multiplier hex comments match the params header."""
    src = _read_rtl("conv_u8_serial.v")
    header = (PARAMS / "baseline_cnn_params.vh").read_text(encoding="utf-8")
    assert "32'h416335B9" in src and "32'h416335B9" in header     # conv2
    assert "32'h4D6CC8EC" in src and "32'h4D6CC8EC" in header     # conv3
    # the OLD wrong hex must not be present
    assert "4162F7B9" not in src and "4D6C5DEC" not in src


def test_unselected_weight_rom_gating() -> None:
    """conv_u8_serial.v gates each weight ROM by the selected layer."""
    src = _read_rtl("conv_u8_serial.v")
    assert "wire wt2_read = issuing && !layer;" in src
    assert "wire wt3_read = issuing &&  layer;" in src
    assert ".addr(wt2_read ? wt2_addr : 13'd0)" in src
    assert ".addr(wt3_read ? wt3_addr : 14'd0)" in src
    # conv2's addr space (max 4751 under a conv3 run) is never issued to the ROM
    assert ".DEPTH(C2_WT_D), .ADDR_W(13)" in src
    assert ".DEPTH(C3_WT_D), .ADDR_W(14)" in src


def test_gap_fc_rtl_interfaces() -> None:
    """The new RTL modules expose the frozen interfaces (port names/widths)."""
    gap = _read_rtl("gap_stream_u8.v")
    assert "module gap_stream_u8" in gap
    assert "input  wire       in_valid" in gap
    assert "input  wire [7:0] in_q" in gap
    assert "output reg  [4:0] out_channel" in gap
    assert "output reg  [7:0] out_q" in gap
    assert "gap_div49 u_gap" in gap          # reuses the verified division
    fc = _read_rtl("fc_argmax_serial.v")
    assert "module fc_argmax_serial" in fc
    assert "input  wire              gap_we" in fc
    assert "input  wire [4:0]        gap_waddr" in fc
    assert "output reg signed [31:0] fc_acc" in fc
    assert "output reg  [3:0]        prediction" in fc
    # tie rule: class 0 seeds unconditionally; later classes only on strict >
    assert "best       <= acc32;" in fc
    assert "acc32 > best" in fc
    assert ">=" not in re.sub(r"//.*", "", fc)


def test_a_plus_p3_end_to_end_cycle_contract() -> None:
    """Full-core and board TBs assert the new 1,590,315-cycle schedule."""
    core_tb = (REPO_ROOT / "fpga" / "baseline_cnn" / "tb" /
               "tb_baseline_cnn_core.v").read_text(encoding="utf-8")
    board_tb = (REPO_ROOT / "fpga" / "baseline_cnn" / "tb" /
                "tb_ac620_cnn_selftest.v").read_text(encoding="utf-8")
    assert "core_done_cyc - start_cyc != 1590315" in core_tb
    assert "retime_bad" in core_tb
    assert "done_cyc - start_cyc != 1590315" in board_tb
    assert "AC620_CNN_SELFTEST_PASS ALL_PASS" in board_tb
    assert "1529163" not in core_tb and "1529163" not in board_tb
