"""Tests for the hardware parameter export helpers (``export_baseline_cnn_hardware``).

Hermetic: no MNIST download, no frozen checkpoint.  Covers the format layer
(two's complement, ``.mem`` / ``.mif``, layouts) and proves
:class:`HardwareModel` — the forward rebuilt purely from exported parameters —
reproduces ``Int8Reference`` bit-for-bit on synthetic data.

The end-to-end export / full-10k verification lives in
``model/verify_baseline_cnn_hardware_export.py`` (needs the checkpoint + data).
"""

import torch

from export_baseline_cnn_hardware import (
    HardwareModel,
    conv_weight_address,
    fc_weight_address,
    feature_map_address,
    flatten_conv_weight_oihw,
    flatten_fc_weight_oi,
    parse_twos_complement_hex,
    read_mem,
    read_mif,
    to_twos_complement_hex,
    write_mem,
    write_mif,
)
from onn_model.int8_ptq import INT32_MAX, INT32_MIN, WEIGHT_LAYER_ACCESSORS
from onn_model.int8_reference import (
    Int8Reference,
    TRACE_NODES,
    _resolve_module,
)
from onn_model.models.baseline_cnn import BaselineCNN
from onn_model.quantization import fuse_model_bn


def _fused_model():
    torch.manual_seed(0)
    return fuse_model_bn(BaselineCNN())


def _synth_config(model):
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
# Two's complement
# ---------------------------------------------------------------------------


class TestTwosComplement:
    def test_int8_round_trip(self):
        for v in (-128, -127, -1, 0, 1, 127):
            assert parse_twos_complement_hex(to_twos_complement_hex(v, 8), 8) == v

    def test_int32_round_trip(self):
        for v in (INT32_MIN, INT32_MIN + 1, -1, 0, 1, INT32_MAX - 1, INT32_MAX):
            assert parse_twos_complement_hex(to_twos_complement_hex(v, 32), 32) == v

    def test_negative_hex_patterns(self):
        assert to_twos_complement_hex(-5, 8) == "FB"
        assert to_twos_complement_hex(-1, 32) == "FFFFFFFF"
        assert to_twos_complement_hex(INT32_MAX, 32) == "7FFFFFFF"
        assert to_twos_complement_hex(INT32_MIN, 32) == "80000000"
        assert to_twos_complement_hex(127, 8) == "7F"

    def test_zero_padded_width(self):
        assert to_twos_complement_hex(5, 8) == "05"
        assert to_twos_complement_hex(255, 32) == "000000FF"


# ---------------------------------------------------------------------------
# .mem / .mif
# ---------------------------------------------------------------------------


class TestMemMifFormats:
    def test_mem_round_trip(self, tmp_path):
        vals = [-128, -5, 0, 1, 127]
        p = tmp_path / "w.mem"
        write_mem(p, vals, 8)
        assert read_mem(p, 8) == vals
        # one 2-digit hex value per line, no extra text
        lines = p.read_text(encoding="ascii").splitlines()
        assert lines == ["80", "FB", "00", "01", "7F"]

    def test_mem_int32_round_trip(self, tmp_path):
        vals = [INT32_MIN, -2_000_000_000, -1, 0, 2_000_000_000, INT32_MAX]
        p = tmp_path / "b.mem"
        write_mem(p, vals, 32)
        assert read_mem(p, 32) == vals

    def test_mif_round_trip_and_contiguity(self, tmp_path):
        vals = [-128, -1, 0, 1, 127]
        p = tmp_path / "w.mif"
        write_mif(p, vals, 8, "stem_weight")
        addrs, data = read_mif(p, 8)
        assert data == vals
        assert addrs == list(range(len(vals)))
        header = p.read_text(encoding="ascii").splitlines()
        assert any(ln.strip().upper() == f"WIDTH = 8;" for ln in header)
        assert any(ln.strip().upper() == f"DEPTH = 5;" for ln in header)

    def test_mif_rejects_non_contiguous(self, tmp_path):
        p = tmp_path / "bad.mif"
        p.write_text(
            "WIDTH = 8;\nDEPTH = 3;\nADDRESS_RADIX = DEC;\nDATA_RADIX = HEX;\n"
            "CONTENT\nBEGIN\n0 : 01;\n5 : 02;\n2 : 03;\nEND;\n",
            encoding="ascii",
        )
        import pytest

        with pytest.raises(ValueError):
            read_mif(p, 8)


# ---------------------------------------------------------------------------
# Layouts (OIHW / OI / CHW)
# ---------------------------------------------------------------------------


class TestLayouts:
    def test_conv_weight_oihw_matches_formula(self):
        torch.manual_seed(3)
        w = torch.randint(-3, 4, (4, 2, 3, 3))  # [Cout, Cin, Kh, Kw]
        flat = flatten_conv_weight_oihw(w).tolist()
        cout, cin, kh, kw = w.shape
        for oc in range(cout):
            for ic in range(cin):
                for ky in range(kh):
                    for kx in range(kw):
                        addr = conv_weight_address(oc, ic, ky, kx, cin, kh, kw)
                        assert flat[addr] == w[oc, ic, ky, kx].item()

    def test_fc_weight_oi_matches_formula(self):
        torch.manual_seed(4)
        w = torch.randint(-3, 4, (10, 32))  # [O, F]
        flat = flatten_fc_weight_oi(w).tolist()
        o, f = w.shape
        for oo in range(o):
            for ff in range(f):
                assert flat[fc_weight_address(oo, ff, f)] == w[oo, ff].item()

    def test_chw_matches_formula(self):
        torch.manual_seed(5)
        t = torch.randint(0, 256, (3, 4, 5))
        flat = t.reshape(-1).tolist()
        c, h, w = t.shape
        for cc in range(c):
            for yy in range(h):
                for xx in range(w):
                    assert flat[feature_map_address(cc, yy, xx, h, w)] == t[cc, yy, xx].item()


# ---------------------------------------------------------------------------
# HardwareModel reproduces Int8Reference bit-for-bit (synthetic data)
# ---------------------------------------------------------------------------


class TestHardwareModel:
    def _pair(self):
        model = _fused_model()
        ref = _make_reference(model)
        hw = HardwareModel(
            wq=ref.wq,
            qb=ref.qb,
            requant={l: {"multiplier": ref.requant[l]["multiplier"], "shift": ref.requant[l]["shift"]}
                     for l in ("stem_conv", "conv2", "conv3")},
            s_in=ref.s_in,
        )
        return ref, hw

    def test_trace_bitwise_identical(self):
        ref, hw = self._pair()
        torch.manual_seed(11)
        x = torch.randn(4, 1, 28, 28)
        pred_ref, _, trace_ref = ref.infer(x, return_trace=True)
        pred_hw, trace_hw = hw.infer(x, return_trace=True)
        assert torch.equal(pred_ref, pred_hw)
        for node in TRACE_NODES:
            assert torch.equal(trace_ref[node], trace_hw[node]), f"{node} differs"

    def test_prediction_only_path_matches(self):
        ref, hw = self._pair()
        torch.manual_seed(12)
        x = torch.randn(6, 1, 28, 28)
        pred_ref, _, _ = ref.infer(x, return_trace=False)
        pred_hw = hw.infer(x, return_trace=False)
        assert torch.equal(pred_ref, pred_hw)

    def test_wq_qb_are_int8_int32(self):
        ref, _ = self._pair()
        for layer in ("stem_conv", "conv2", "conv3", "fc"):
            assert ref.wq[layer].dtype == torch.int8, layer
            assert ref.qb[layer].dtype == torch.int32, layer
            assert ref.wq[layer].min().item() >= -127, layer
            assert ref.wq[layer].max().item() <= 127, layer
