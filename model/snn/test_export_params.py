"""Lightweight verification for the fixed RTL parameter export."""

from __future__ import annotations

import json

import torch

from model.snn.export_params import (
    DEFAULT_EXPORT_DIR,
    EXPECTED_CHECKPOINT_SHA256,
    WEIGHT_EXPORTS,
    load_deployment_state,
    read_int8_mem,
    sha256_file,
)
from model.snn.integer_reference import IntegerSNNReference, IntegerWidths


def test_export_round_trip_and_integer_reference_contract() -> None:
    manifest_path = DEFAULT_EXPORT_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    state, digest = load_deployment_state()
    reference = IntegerSNNReference(
        state,
        guard_bits=4,
        widths=IntegerWidths.for_guard_bits(4, readout_current=17),
    )

    assert manifest["seed"] == 17
    assert digest == manifest["checkpoint_sha256"] == EXPECTED_CHECKPOINT_SHA256
    for export_name, (state_key, filename) in WEIGHT_EXPORTS.items():
        record = manifest["weights"][export_name]
        export_path = DEFAULT_EXPORT_DIR / filename
        restored = read_int8_mem(export_path, record["shape"])
        expected = reference.weights[state_key].to(torch.int8)
        assert torch.equal(restored, expected)
        assert restored.numel() == record["element_count"]
        assert int(restored.min().item()) == record["observed_min"]
        assert int(restored.max().item()) == record["observed_max"]
        assert -127 <= record["observed_min"] <= record["observed_max"] <= 127
        assert sha256_file(export_path) == record["file_sha256"]

    metadata = reference.quantization_metadata()
    assert manifest["thresholds"]["lif1"]["integer"] == metadata[
        "threshold_integer"
    ]["conv1.weight"]
    assert manifest["thresholds"]["lif2"]["integer"] == metadata[
        "threshold_integer"
    ]["conv2.weight"]
    assert manifest["thresholds"]["lif1"]["unsigned_bits"] == metadata[
        "threshold_unsigned_bits"
    ]["conv1.weight"]
    assert manifest["thresholds"]["lif2"]["unsigned_bits"] == metadata[
        "threshold_unsigned_bits"
    ]["conv2.weight"]
    assert manifest["temporal_coefficients"] == metadata["temporal_coefficients"]
    assert manifest["hidden_state_guard_bits"] == metadata[
        "hidden_state_guard_bits"
    ]
    exported_widths = {
        "conv1_current": manifest["bit_widths"]["conv1_current"]["bits"],
        "mem1": manifest["bit_widths"]["lif1_membrane"]["bits"],
        "conv2_current": manifest["bit_widths"]["conv2_current"]["bits"],
        "mem2": manifest["bit_widths"]["lif2_membrane"]["bits"],
        "readout_current": manifest["bit_widths"]["readout_current"]["bits"],
        "weighted_logits": manifest["bit_widths"]["weighted_logits"]["bits"],
    }
    assert exported_widths == metadata["widths"]


def test_export_file_set_and_rtl_constants() -> None:
    expected_files = {
        "conv1_weight.mem",
        "conv2_weight.mem",
        "readout_weight.mem",
        "params.svh",
        "manifest.json",
    }
    assert {path.name for path in DEFAULT_EXPORT_DIR.iterdir()} == expected_files
    manifest = json.loads((DEFAULT_EXPORT_DIR / "manifest.json").read_text("utf-8"))
    assert set(manifest["exports"].values()) == expected_files - {"manifest.json"}
    params = (DEFAULT_EXPORT_DIR / "params.svh").read_text(encoding="utf-8")
    assert "LIF1_THRESHOLD = 939" in params
    assert "LIF2_THRESHOLD = 1715" in params
    assert "READOUT_CURRENT_BITS = 17" in params
    assert [
        int(line.rsplit(" ", 1)[-1].rstrip(";"))
        for line in params.splitlines()
        if line.startswith("localparam int unsigned TEMPORAL_COEFF_")
    ] == [5, 4, 3, 2]
