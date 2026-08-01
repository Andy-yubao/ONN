"""Tests for the spec-compliant INT8 PTQ primitives and W8A8 simulation.

Covers the six required points (spec section 7):

  * round-half-away-from-zero
  * signed and unsigned saturation
  * per-tensor weight quantisation
  * per-output-channel weight quantisation
  * quantisation determinism
  * INT32 accumulator arithmetic (no Python-integer rules, saturate not wrap)

All tests use synthetic data on CPU; no MNIST download.
"""

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from onn_model.int8_ptq import (
    _accumulate_int32,
    build_activation_scales,
    build_weight_config,
    calibrate_activations,
    finalize_weight_config,
    quantize_signed,
    quantize_unsigned,
    quantize_weight_per_channel,
    quantize_weight_tensor,
    round_half_away_from_zero,
    w8a8_forward,
)
from onn_model.models.baseline_cnn import BaselineCNN
from onn_model.quantization import fuse_model_bn

DEVICE = torch.device("cpu")


# ---------------------------------------------------------------------------
#  Rounding rule
# ---------------------------------------------------------------------------

class TestRounding:
    def test_round_half_away_from_zero(self):
        x = torch.tensor([0.5, -0.5, 1.5, -1.5, 2.5, -2.5, 2.4, -2.4, 0.4, -0.4, 0.0])
        expected = torch.tensor([1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 2.0, -2.0, 0.0, 0.0, 0.0])
        assert torch.equal(round_half_away_from_zero(x), expected)

    def test_differs_from_torch_round_half_even(self):
        # torch.round(0.5) == 0 (half-to-even); our rule must give 1
        x = torch.tensor([0.5, 2.5])
        assert torch.equal(round_half_away_from_zero(x), torch.tensor([1.0, 3.0]))
        assert not torch.equal(round_half_away_from_zero(x), torch.round(x))

    def test_round_half_away_does_not_use_python_int(self):
        x = torch.tensor([1e8, -1e8], dtype=torch.float64)
        q = round_half_away_from_zero(x)
        assert q.dtype == torch.float64
        assert torch.equal(q, x)  # huge exact magnitudes pass through unchanged


# ---------------------------------------------------------------------------
#  Saturation rules
# ---------------------------------------------------------------------------

class TestSaturation:
    def test_signed_clamps_to_int8(self):
        q = quantize_signed(
            torch.tensor([1000.0, -1000.0, 127.0, -128.0, 0.0, 3.6, -3.6, 0.5]), scale=1.0
        )
        assert q.dtype == torch.int8
        vals = q.tolist()
        assert vals == [127, -128, 127, -128, 0, 4, -4, 1]
        # no value outside [-128, 127]
        assert all(-128 <= v <= 127 for v in vals)

    def test_unsigned_clamps_to_uint8(self):
        q = quantize_unsigned(
            torch.tensor([300.0, -50.0, 0.0, 100.0, 254.6, 255.4, 0.4]), scale=1.0
        )
        assert q.dtype == torch.uint8
        vals = q.tolist()
        assert vals == [255, 0, 0, 100, 255, 255, 0]
        assert all(0 <= v <= 255 for v in vals)

    def test_saturate_not_wrap_signed(self):
        # a value far beyond the range must clamp, not wrap mod 256
        q = quantize_signed(torch.tensor([300.0, -300.0]), scale=1.0)
        assert q.tolist() == [127, -128]

    def test_saturate_not_wrap_unsigned(self):
        q = quantize_unsigned(torch.tensor([300.0, -300.0]), scale=1.0)
        assert q.tolist() == [255, 0]

    def test_scale_division_applied(self):
        # x / scale = 0.5 / 2.0 = 0.25 -> rounds to 0
        assert quantize_unsigned(torch.tensor([0.5]), scale=2.0).item() == 0
        # x / scale = 1.0 / 2.0 = 0.5 -> rounds half away from zero to 1
        assert quantize_unsigned(torch.tensor([1.0]), scale=2.0).item() == 1


# ---------------------------------------------------------------------------
#  Weight quantisation
# ---------------------------------------------------------------------------

class TestWeightQuantizationSpec:
    def test_per_tensor_scale_and_edge(self):
        w = torch.tensor([[1.0, -2.0, 3.0], [4.0, 0.5, -0.5]])
        r = quantize_weight_tensor(w)
        assert r["scale"] == 4.0 / 127.0
        assert r["q"].max().item() == 127          # peak maps to +127
        assert r["q"].min().item() >= -127          # never -128
        assert r["q"].max().item() <= 127
        # dequant round-trips within half a ULP
        deq = r["q"].float() * r["scale"]
        assert (deq - w).abs().max().item() < r["scale"]

    def test_per_tensor_range_never_uses_minus_128(self):
        w = torch.tensor([[-127.0, -127.0, -1.0]])
        r = quantize_weight_tensor(w)
        assert r["q"].min().item() == -127

    def test_per_channel_scales_and_saturation(self):
        w = torch.tensor([[1.0, 2.0], [100.0, -50.0]])  # 2 output channels
        r = quantize_weight_per_channel(w)
        assert torch.allclose(r["scales"], torch.tensor([2.0, 100.0]) / 127.0)
        # each channel's peak magnitude maps to exactly +127
        assert r["q"][0, 1].item() == 127  # 2.0 -> 127
        assert r["q"][1, 0].item() == 127  # 100 -> 127
        assert r["saturated_count"] >= 2
        assert 0.0 <= r["saturated_fraction"] <= 1.0

    def test_per_channel_shape_and_range(self):
        torch.manual_seed(0)
        w = torch.randn(8, 3, 3, 3)
        r = quantize_weight_per_channel(w)
        assert r["q"].shape == w.shape
        assert r["scales"].shape == (8,)
        assert r["q"].min().item() >= -127
        assert r["q"].max().item() <= 127

    def test_zero_weight_guard(self):
        w = torch.zeros(2, 3, 3, 3)
        r = quantize_weight_tensor(w)
        assert torch.isfinite(torch.tensor(r["scale"]))
        assert r["q"].abs().sum().item() == 0


# ---------------------------------------------------------------------------
#  Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_quantize_deterministic(self):
        torch.manual_seed(0)
        x = torch.randn(16, 1, 28, 28)
        q1 = quantize_signed(x, 0.01)
        q2 = quantize_signed(x, 0.01)
        assert torch.equal(q1, q2)

    def test_w8a8_forward_deterministic(self):
        torch.manual_seed(0)
        fused = fuse_model_bn(BaselineCNN())
        fused.eval()
        x = torch.randn(8, 1, 28, 28)
        calib = _synthetic_calibration(fused)
        act_scales = build_activation_scales(calib)
        wc = build_weight_config(fused, "per-output-channel", fc_scheme="per-output-channel")
        finalize_weight_config(wc, act_scales)
        out1, _ = w8a8_forward(fused, x, act_scales, wc)
        out2, _ = w8a8_forward(fused, x, act_scales, wc)
        assert torch.equal(out1, out2)


# ---------------------------------------------------------------------------
#  INT32 accumulator
# ---------------------------------------------------------------------------

class TestInt32Accumulator:
    def test_accumulator_exact_integer_math(self):
        # integer activations / weights: the conv accumulate must stay exact
        # and fit in INT32 (no float drift, no Python-int wraparound).
        torch.manual_seed(0)
        qa = torch.randint(-128, 128, (2, 3, 4, 4)).to(torch.int8)
        qw = torch.randint(-127, 128, (5, 3, 2, 2)).to(torch.int8)
        qb = torch.tensor([1, -2, 3, -4, 5], dtype=torch.int32)

        acc = F.conv2d(qa.to(torch.float64), qw.to(torch.float64), stride=1)
        acc = acc + qb.to(torch.float64).view(1, -1, 1, 1)

        assert acc.min().item() >= -(2**31)
        assert acc.max().item() <= 2**31 - 1
        assert torch.equal(acc, acc.round())  # exact integers

        # cross-check against an explicit per-patch summation in torch
        unf = F.unfold(qa.to(torch.float64), kernel_size=2)  # (N, Cin*Kh*Kw, L)
        manual = unf.transpose(1, 2) @ qw.to(torch.float64).reshape(5, -1).t()  # (N, L, 5)
        manual = manual.transpose(1, 2)  # (N, 5, L)
        manual = manual + qb.to(torch.float64).view(1, -1, 1)
        manual = manual.reshape(2, 5, 3, 3)
        assert torch.equal(acc, manual)

    def test_accumulator_saturates_not_wraps(self):
        # float64 (as in the real accumulator) so INT32 bounds are exact
        acc = torch.tensor([2**31 + 100.0, -(2**31) - 50.0, 5.0], dtype=torch.float64)
        info = {"accumulator_max_abs": {}, "accumulator_overflow": {}}
        out = _accumulate_int32(acc, "probe", info)
        assert out[0].item() == 2**31 - 1
        assert out[1].item() == -(2**31)
        assert out[2].item() == 5.0
        assert info["accumulator_overflow"]["probe"] == 2


# ---------------------------------------------------------------------------
#  End-to-end W8A8 smoke test on the fused BaselineCNN
# ---------------------------------------------------------------------------

def _synthetic_calibration(fused_model: nn.Module):
    torch.manual_seed(1)
    images = torch.randn(32, 1, 28, 28)
    labels = torch.zeros(32, dtype=torch.long)
    loader = DataLoader(TensorDataset(images, labels), batch_size=16)
    return calibrate_activations(fused_model, loader, DEVICE)


class TestW8A8Forward:
    def _fused(self):
        torch.manual_seed(0)
        fused = fuse_model_bn(BaselineCNN())
        fused.eval()
        return fused

    def test_shape_finite_and_deterministic(self):
        fused = self._fused()
        calib = _synthetic_calibration(fused)
        act_scales = build_activation_scales(calib)
        wc = build_weight_config(fused, "per-tensor")
        finalize_weight_config(wc, act_scales)

        x = torch.randn(8, 1, 28, 28)
        logits, info = w8a8_forward(fused, x, act_scales, wc)
        assert logits.shape == (8, 10)
        assert torch.isfinite(logits).all()
        for layer, v in info["accumulator_max_abs"].items():
            assert v < 2**31, f"{layer}: accumulator beyond INT32"

    def test_maxpool_passthrough_scale(self):
        fused = self._fused()
        calib = _synthetic_calibration(fused)
        act_scales = build_activation_scales(calib)
        assert act_scales["pool1"] == act_scales["stem_relu"]
        assert act_scales["pool2"] == act_scales["conv2_relu"]

    def test_both_weight_schemes_run(self):
        fused = self._fused()
        calib = _synthetic_calibration(fused)
        act_scales = build_activation_scales(calib)
        for method, fc in (("per-tensor", "per-tensor"), ("per-output-channel", "per-output-channel")):
            wc = build_weight_config(fused, method, fc_scheme=fc)
            finalize_weight_config(wc, act_scales)
            x = torch.randn(4, 1, 28, 28)
            logits, _ = w8a8_forward(fused, x, act_scales, wc)
            assert logits.shape == (4, 10)
            assert torch.isfinite(logits).all()
