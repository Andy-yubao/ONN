"""Tests for the pure-integer reference inference (``int8_reference``).

Covers the required points of the integer-reference phase:

  * positive / negative round-half-away-from-zero
  * multiplier / shift fixed-point approximation
  * signed right-shift rounding
  * INT32 bias quantisation
  * small hand-computed convolution
  * UINT8 ReLU saturation
  * UINT8 MaxPool
  * integer divide-by-49 average pooling
  * INT32 overflow detection (saturate, no wraparound)
  * end-to-end determinism
  * every intermediate node keeps an integer dtype / in-range

All tests use synthetic data on CPU; no MNIST download.
"""

import torch
import torch.nn.functional as F

from onn_model.int8_ptq import INT32_MAX, INT32_MIN, UINT8_MAX, UINT8_MIN, quantize_signed
from onn_model.int8_reference import (
    Int8Reference,
    NODE_SPEC,
    TRACE_NODES,
    WEIGHT_LAYER_ACCESSORS,
    _resolve_module,
    check_accumulator,
    quantize_bias_int32,
    quantize_weight_with_scale,
    real_multiplier_to_fixed,
    round_div_away,
    round_shift,
)
from onn_model.models.baseline_cnn import BaselineCNN
from onn_model.quantization import fuse_model_bn

DEVICE = torch.device("cpu")


def _fused_model():
    torch.manual_seed(0)
    return fuse_model_bn(BaselineCNN())


def _synth_config(model):
    """Build a frozen-config-style dict from a (random) fused model."""
    ws = {}
    for name, accessor in WEIGHT_LAYER_ACCESSORS.items():
        w = _resolve_module(model, accessor).weight
        ws[name] = w.abs().max().item() / 127.0
    return {
        "activation_scale": {
            "input": 0.05,
            "stem_relu": 0.05,
            "conv2_relu": 0.05,
            "conv3_relu": 0.05,
        },
        "weight_scale": ws,
    }


def _make_reference(model):
    return Int8Reference(model, _synth_config(model))


# ---------------------------------------------------------------------------
#  Rounding rules
# ---------------------------------------------------------------------------


class TestRounding:
    def test_quantize_signed_rounds_half_away_from_zero(self):
        # 0.5 -> 1, -0.5 -> -1, 2.5 -> 3, -2.5 -> -3 (never half-to-even)
        q = quantize_signed(torch.tensor([0.5, -0.5, 1.5, -1.5, 2.5, -2.5]), scale=1.0)
        assert q.tolist() == [1, -1, 2, -2, 3, -3]

    def test_round_div_away_pos_neg(self):
        a = torch.tensor([73, 74, 49, 24, 25, -73, -74, 0, 98])
        r = round_div_away(a, 49)
        assert r.tolist() == [1, 2, 1, 0, 1, -1, -2, 0, 2]

    def test_round_div_away_even_divisor(self):
        a = torch.tensor([3, 4, -3, -4, 0])
        assert round_div_away(a, 4).tolist() == [1, 1, -1, -1, 0]


# ---------------------------------------------------------------------------
#  Fixed-point multiplier / shift
# ---------------------------------------------------------------------------


class TestFixedPoint:
    def test_multiplier_shift_approximation(self):
        for r in (7.4828e-3, 3.9905e-3, 4.7257e-3, 0.5, 0.12345, 0.999999):
            m, s, rel = real_multiplier_to_fixed(r)
            assert 0 < m < 2**31, f"multiplier {m} out of signed 31-bit magnitude"
            assert s >= 0
            approx = m / (1 << s)
            assert abs(approx - r) / r < 1e-6, f"r={r} approx={approx}"
            assert rel < 1e-6

    def test_multiplier_uses_high_precision(self):
        m, s, _ = real_multiplier_to_fixed(4.7257e-3)
        assert m > 2**30  # ~31-bit magnitude: precision maximised

    def test_rejects_too_large_real_multiplier(self):
        import pytest

        with pytest.raises(ValueError):
            real_multiplier_to_fixed(2**31)


# ---------------------------------------------------------------------------
#  Signed right-shift rounding
# ---------------------------------------------------------------------------


class TestRoundShift:
    def test_round_shift_half_away_from_zero(self):
        x = torch.tensor([3, -3, 4, -4, 5, -5, 1, -1], dtype=torch.int64)
        # 3/2=1.5->2, 4/2=2, 5/2=2.5->3, 1/2=0.5->1
        assert round_shift(x, 1).tolist() == [2, -2, 2, -2, 3, -3, 1, -1]

    def test_round_shift_larger_shift(self):
        x = torch.tensor([7, -7, 8, -8, 9, -9, 10, -10], dtype=torch.int64)
        # /4: 1.75->2, 2.0, 2.25->2, 2.5->3
        assert round_shift(x, 2).tolist() == [2, -2, 2, -2, 2, -2, 3, -3]

    def test_round_shift_zero_shift_identity(self):
        x = torch.tensor([-100, 100], dtype=torch.int64)
        assert torch.equal(round_shift(x, 0), x)


# ---------------------------------------------------------------------------
#  Weight / bias quantisation
# ---------------------------------------------------------------------------


class TestBiasQuantization:
    def test_bias_quantised_to_int32(self):
        torch.manual_seed(0)
        b = torch.randn(16) * 0.5
        wscale, ascale = 1.0 / 127.0, 0.02
        q, out_of_range = quantize_bias_int32(b, wscale * ascale)
        assert q.dtype == torch.int32
        assert q.min().item() >= INT32_MIN and q.max().item() <= INT32_MAX
        assert out_of_range == 0
        # round-trips within half a bias-scale unit
        assert (q.double() * (wscale * ascale) - b).abs().max().item() <= wscale * ascale

    def test_bias_out_of_range_reported(self):
        b = torch.tensor([1e12, -1e12, 3.0])
        q, out_of_range = quantize_bias_int32(b, 1.0)
        assert out_of_range == 2
        assert q[0].item() == INT32_MAX and q[1].item() == INT32_MIN
        assert q[2].item() == 3

    def test_weight_quantise_with_frozen_scale(self):
        torch.manual_seed(0)
        w = torch.randn(4, 3, 3, 3)
        scale = w.abs().max().item() / 127.0
        q = quantize_weight_with_scale(w, scale)
        assert q.dtype == torch.int8
        assert q.min().item() >= -127 and q.max().item() <= 127
        # peak magnitude maps to exactly +-127
        assert q.abs().max().item() == 127


# ---------------------------------------------------------------------------
#  Small hand-computed convolution
# ---------------------------------------------------------------------------


def _manual_conv2d(x, w, pad=1):
    """Exact integer conv2d (padding=pad) computed with plain Python ints.

    ``x`` and ``w`` are nested lists (from ``tensor.tolist()``); shapes are
    derived with ``len`` so the cross-check never reuses torch's conv.
    """
    cin, h, wd = len(x), len(x[0]), len(x[0][0])
    cout = len(w)
    kh, kw = len(w[0][0]), len(w[0][0][0])
    # padding expands the output: H_out = H + 2*pad - Kh + 1
    h_out, w_out = h + 2 * pad - kh + 1, wd + 2 * pad - kw + 1
    out = []
    for co in range(cout):
        plane = []
        for y in range(h_out):
            row = []
            for xx in range(w_out):
                acc = 0
                for ci in range(cin):
                    for i in range(kh):
                        for j in range(kw):
                            iy, ix = y + i - pad, xx + j - pad
                            if 0 <= iy < h and 0 <= ix < wd:
                                acc += x[ci][iy][ix] * w[co][ci][i][j]
                row.append(acc)
            plane.append(row)
        out.append(plane)
    return out


def _manual_linear(x, w, qb):
    """Exact integer fc: x [N, F] @ w^T [O, F] + bias [O], plain Python ints.

    ``x``, ``w``, ``qb`` are nested lists / lists from ``tensor.tolist()``;
    shapes are derived with ``len`` so the cross-check never reuses torch's matmul.
    """
    out = []
    for n in range(len(x)):
        row = []
        for o in range(len(w)):
            acc = 0
            for f in range(len(w[o])):
                acc += x[n][f] * w[o][f]
            row.append(acc + qb[o])
        out.append(row)
    return out


class TestIntegerConv:
    def test_small_hand_convolution_matches_manual(self):
        x = torch.tensor(
            [[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]],
            dtype=torch.int32,
        )  # [1, 4, 4]
        w = torch.tensor(
            [
                [[[1, 0], [0, -1]]],  # output channel 0
                [[[2, 3], [-1, 1]]],  # output channel 1
            ],
            dtype=torch.int32,
        )  # [2, 1, 2, 2]
        qb = torch.tensor([7, -5], dtype=torch.int32)

        ref = _make_reference(_fused_model())
        stats = {
            "overflow_count": {"probe": 0},
            "min": {"probe": float("inf")},
            "max": {"probe": float("-inf")},
        }
        acc = ref._conv_acc(x.unsqueeze(0), w, qb, "probe", stats)
        # 4x4 input, 2x2 kernel, padding=1 -> 5x5 output
        assert acc.shape == (1, 2, 5, 5)
        assert stats["overflow_count"]["probe"] == 0

        manual = torch.tensor(_manual_conv2d(x.tolist(), w.tolist(), pad=1))
        expected = manual + qb.view(2, 1, 1)
        assert torch.equal(acc[0], expected)


# ---------------------------------------------------------------------------
#  UINT8 ReLU saturation / MaxPool / GAP
# ---------------------------------------------------------------------------


class TestUint8Ops:
    def test_requant_saturates_to_uint8(self):
        ref = _make_reference(_fused_model())
        rq = {"multiplier": 2**30, "shift": 30}  # multiplier / 2**shift == 1.0
        acc = torch.tensor(
            [[[300.0], [-5.0], [100.0], [255.0], [256.0], [0.0]]],
            dtype=torch.int32,
        )
        q = ref._requantize_uint8(acc, rq)
        assert q.dtype == torch.uint8
        # acc shape [1, 6, 1]: 300 -> 255 (sat), -5 -> 0, 100 -> 100,
        # 255 -> 255, 256 -> 255 (sat), 0 -> 0
        assert q.tolist() == [[[255], [0], [100], [255], [255], [0]]]

    def test_uint8_maxpool_picks_max(self):
        ref = _make_reference(_fused_model())
        q = torch.tensor(
            [[[1, 9, 2, 3], [7, 4, 5, 6], [8, 2, 0, 1], [3, 4, 9, 2]]],
            dtype=torch.uint8,
        )
        m = ref._maxpool_uint8(q)
        assert m.dtype == torch.uint8
        assert m.tolist() == [[[9, 6], [8, 9]]]

    def test_gap_divide_by_49(self):
        ref = _make_reference(_fused_model())
        # 7x7 UINT8 planes: sum 74 -> round(74/49)=2; sum 25 -> 1; all-7 -> 7
        plane = torch.zeros(2, 3, 7, 7, dtype=torch.uint8)
        plane[0, 0, 0, 0] = 74
        plane[0, 1, 0, 0] = 25
        plane[0, 2] = 7  # 49 elements x 7 = 343 -> 7
        plane[1, 0] = 0
        gap_sum = plane.to(torch.int32).sum(dim=(2, 3))
        gap_q = round_div_away(gap_sum, 49).clamp(UINT8_MIN, UINT8_MAX).to(torch.uint8)
        assert gap_q.tolist() == [[2, 1, 7], [0, 0, 0]]
        assert gap_q.max().item() <= UINT8_MAX


# ---------------------------------------------------------------------------
#  INT32 overflow detection
# ---------------------------------------------------------------------------


class TestOverflow:
    def test_accumulator_overflow_detected_and_saturated(self):
        x = torch.tensor([[[[1_000_000]]]], dtype=torch.int64)  # [1,1,1,1]
        w = torch.tensor([[[[3_000]]]], dtype=torch.int64)  # product 3e9 > INT32
        stats = {
            "overflow_count": {"probe": 0},
            "min": {"probe": float("inf")},
            "max": {"probe": float("-inf")},
        }
        acc64 = F.conv2d(x, w)  # [1,1,1,1] = 3e9
        acc = check_accumulator(acc64, "probe", stats)
        assert stats["overflow_count"]["probe"] == 1
        assert acc.item() == INT32_MAX  # saturated, not wrapped
        assert stats["min"]["probe"] == 3e9 and stats["max"]["probe"] == 3e9

    def test_accumulator_within_int32_no_overflow(self):
        x = torch.tensor([[[[100]]]], dtype=torch.int64)
        w = torch.tensor([[[[200]]]], dtype=torch.int64)
        stats = {"overflow_count": {"probe": 0}, "min": {"probe": float("inf")}, "max": {"probe": float("-inf")}}
        acc = check_accumulator(F.conv2d(x, w), "probe", stats)
        assert stats["overflow_count"]["probe"] == 0
        assert acc.item() == 20_000


# ---------------------------------------------------------------------------
#  FC (linear) accumulator: INT64 host, no INT32 wraparound
# ---------------------------------------------------------------------------


class TestFcAccumulator:
    """The fc is a matmul; its accumulator must run on an INT64 host so a real
    overflow is seen and saturated *before* any INT32 wraparound (see
    ``Int8Reference._linear_acc``)."""

    def test_fc_matches_manual_integer_dot_product(self):
        ref = _make_reference(_fused_model())
        # gap_q-like input [2, 32] uint8 against the model's real fc weights
        x = torch.randint(0, 256, (2, 32), dtype=torch.uint8)
        stats = {
            "overflow_count": {"fc": 0},
            "min": {"fc": float("inf")},
            "max": {"fc": float("-inf")},
        }
        acc = ref._linear_acc(x, ref.wq["fc"], ref.qb["fc"], "fc", stats)
        assert acc.dtype == torch.int32
        assert acc.shape == (2, 10)
        assert stats["overflow_count"]["fc"] == 0
        manual = torch.tensor(
            _manual_linear(x.tolist(), ref.wq["fc"].tolist(), ref.qb["fc"].tolist())
        )
        assert torch.equal(acc, manual)

    def test_fc_overflow_saturates_without_int32_wraparound(self):
        ref = _make_reference(_fused_model())
        # x [2, 2] @ w^T [2, 2] + bias [2]: every dot product exceeds INT32.
        x = torch.tensor([[2_000_000, 2_000_000], [-2_000_000, -2_000_000]], dtype=torch.int64)
        w = torch.tensor([[2000, 0], [0, 2000]], dtype=torch.int64)
        qb = torch.tensor([1000, -1000], dtype=torch.int64)
        stats = {
            "overflow_count": {"fc": 0},
            "min": {"fc": float("inf")},
            "max": {"fc": float("-inf")},
        }
        acc = ref._linear_acc(x, w, qb, "fc", stats)

        # true INT64 sums before saturation
        true_sums = torch.tensor(
            _manual_linear(x.tolist(), w.tolist(), qb.tolist()), dtype=torch.int64
        )
        assert true_sums.tolist() == [
            [4_000_001_000, 3_999_999_000],
            [-3_999_999_000, -4_000_001_000],
        ]
        # every element overflows INT32
        assert stats["overflow_count"]["fc"] == 4
        # recorded min/max are the true pre-saturation INT64 values
        assert stats["min"]["fc"] == -4_000_001_000.0
        assert stats["max"]["fc"] == 4_000_001_000.0
        # saturated to the INT32 bounds, never wrapped
        expected = torch.tensor(
            [[INT32_MAX, INT32_MAX], [INT32_MIN, INT32_MIN]], dtype=torch.int32
        )
        assert torch.equal(acc, expected)
        # explicit no-wraparound: the INT32-wrapped values must not be returned
        wrapped = torch.remainder(true_sums + 2**31, 2**32) - 2**31
        assert not torch.equal(acc.to(torch.int64), wrapped)


# ---------------------------------------------------------------------------
#  End-to-end reference forward
# ---------------------------------------------------------------------------


class TestReferenceForward:
    def _ref(self):
        return _make_reference(_fused_model())

    def test_all_nodes_integer_dtype_and_range(self):
        ref = self._ref()
        x = torch.randn(4, 1, 28, 28)
        _, _, trace = ref.infer(x)
        for node in TRACE_NODES:
            t = trace[node]
            expected_dtype, lo, hi = NODE_SPEC[node]
            assert t.dtype == expected_dtype, f"{node}: dtype {t.dtype} != {expected_dtype}"
            assert t.min().item() >= lo, f"{node}: min {t.min().item()} < {lo}"
            assert t.max().item() <= hi, f"{node}: max {t.max().item()} > {hi}"
        assert trace["acc_stats"]["overflow_count"] == {
            "stem_conv": 0, "conv2": 0, "conv3": 0, "fc": 0,
        }

    def test_deterministic(self):
        ref = self._ref()
        x = torch.randn(8, 1, 28, 28)
        p1, l1, t1 = ref.infer(x)
        p2, l2, t2 = ref.infer(x)
        assert torch.equal(p1, p2)
        assert torch.equal(l1, l2)
        for node in TRACE_NODES:
            assert torch.equal(t1[node], t2[node]), f"{node} not identical across runs"

    def test_prediction_is_argmax_of_int32_fc_acc(self):
        ref = self._ref()
        x = torch.randn(6, 1, 28, 28)
        pred, _, trace = ref.infer(x)
        assert torch.equal(pred, trace["fc_acc"].argmax(dim=1))
        assert pred.dtype == torch.int64
        assert pred.shape == (6,)

    def test_logits_dequant_only_at_output(self):
        ref = self._ref()
        x = torch.randn(2, 1, 28, 28)
        _, logits, trace = ref.infer(x)
        scale = ref.weight_scale["fc"] * ref.act_scale["conv3_relu"]
        assert torch.allclose(logits, trace["fc_acc"].float() * scale)

    def test_conv_nodes_exact_integer_sum(self):
        # input_q / weights are integers: the conv accumulator must equal the
        # exact integer convolution (cross-checked with int64 F.conv2d).
        ref = self._ref()
        x = torch.randn(3, 1, 28, 28)
        _, _, trace = ref.infer(x)
        # conv1: manual re-run on the traced integers
        manual = F.conv2d(
            trace["input_q"].to(torch.int64),
            ref.wq["stem_conv"].to(torch.int64),
            stride=1, padding=1,
        ) + ref.qb["stem_conv"].to(torch.int64).view(1, -1, 1, 1)
        assert torch.equal(trace["conv1_acc"].to(torch.int64), manual)
