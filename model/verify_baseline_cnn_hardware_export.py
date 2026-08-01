"""Verify the BaselineCNN hardware parameter export against the frozen reference.

The verification is strict: it must prove the exported files are a lossless,
bit-exact rendering of the frozen pure-integer model **and** that they are
self-sufficient (the whole 10,000-image test set can be re-run from the files
alone, with the same 98.47 % accuracy and 100 % prediction agreement).

Checks implemented (mapped to the phase spec):

  1. ``.mem`` and ``.mif`` both restore the original integer tensors;
  2. restored weights / biases are bitwise equal to ``Int8Reference``;
  3. multiplier / shift / constants match item by item;
  4. element counts, bit widths and address continuity are correct;
  5. negative two's-complement round-trips exactly;
  6. every golden-trace node is bit-identical;
  7. the 10 smoke samples' ``fc_acc`` and prediction are bit-identical;
  8. re-running the full 10,000-image test set using *only* the exported
     parameters gives accuracy 0.9847 and 100 % prediction agreement;
  9. exporting twice produces byte-identical controlled files (SHA256 equal),
     and the committed export matches a fresh re-export.

The forward used for check 8 is :class:`HardwareModel`, rebuilt purely from
``manifest.json`` + ``*.mem`` — it never touches the FP32 checkpoint, so the
export is proven self-sufficient rather than re-reading the source weights.

Usage (from the repository root):

    python model/verify_baseline_cnn_hardware_export.py
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from export_baseline_cnn_hardware import (  # noqa: E402
    BIAS_BITS,
    WEIGHT_BITS,
    HardwareModel,
    _hex_digits_for_node,
    _node_format,
    _params_header,
    export_all,
    read_mem,
    read_mif,
    sha256_file,
    to_twos_complement_hex,
)
from onn_model.data import get_mnist_dataset, get_test_loader
from onn_model.int8_reference import (
    ACCUMULATOR_LAYERS,
    Int8Reference,
    TRACE_NODES,
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
DEFAULT_EXPORT_DIR = REPO_ROOT / "fpga" / "baseline_cnn"

EXPECTED_ACCURACY = 0.9847
TEST_BATCH_SIZE = 128

# ---------------------------------------------------------------------------
# Small check bookkeeping
# ---------------------------------------------------------------------------


class CheckReport:
    def __init__(self) -> None:
        self.checks: List[Tuple[str, bool, str]] = []

    def add(self, name: str, ok: bool, detail: str = "") -> bool:
        self.checks.append((name, bool(ok), detail))
        return bool(ok)

    @property
    def passed(self) -> bool:
        return all(ok for _, ok, _ in self.checks)


def _load_manifest(export_dir: Path) -> Dict[str, Any]:
    return json.loads((export_dir / "params" / "manifest.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Build a HardwareModel from the exported files only
# ---------------------------------------------------------------------------


def _hardware_model_from_export(export_dir: Path) -> HardwareModel:
    manifest = _load_manifest(export_dir)
    wq: Dict[str, torch.Tensor] = {}
    qb: Dict[str, torch.Tensor] = {}
    for layer, meta in manifest["weights"].items():
        vals = read_mem(export_dir / meta["file"], WEIGHT_BITS)
        wq[layer] = torch.tensor(vals, dtype=torch.int8).reshape(meta["shape"])
    for layer, meta in manifest["biases"].items():
        vals = read_mem(export_dir / meta["file"], BIAS_BITS)
        qb[layer] = torch.tensor(vals, dtype=torch.int32).reshape(meta["shape"])
    requant: Dict[str, Dict[str, int]] = {}
    for layer in ("stem_conv", "conv2", "conv3"):
        rq = manifest["network"]["layers"][layer]["requant"]
        requant[layer] = {"multiplier": int(rq["multiplier"]), "shift": int(rq["shift"])}
    s_in = float(manifest["network"]["input_scale"])
    return HardwareModel(wq=wq, qb=qb, requant=requant, s_in=s_in)


# ---------------------------------------------------------------------------
# File-level checks (round-trip, bitwise equality, counts, addresses, sign)
# ---------------------------------------------------------------------------


def _check_weight_bias_files(export_dir: Path, ref: Int8Reference, rep: CheckReport) -> None:
    manifest = _load_manifest(export_dir)
    neg_ok = True
    for layer in ("stem_conv", "conv2", "conv3", "fc"):
        w_meta = manifest["weights"][layer]
        b_meta = manifest["biases"][layer]
        w_mem = read_mem(export_dir / w_meta["file"], WEIGHT_BITS)
        w_mif_addrs, w_mif = read_mif(export_dir / w_meta["mif"], WEIGHT_BITS)
        b_mem = read_mem(export_dir / b_meta["file"], BIAS_BITS)
        b_mif_addrs, b_mif = read_mif(export_dir / b_meta["mif"], BIAS_BITS)

        # expected values from the frozen reference
        w_ref = ref.wq[layer].reshape(-1).tolist()
        b_ref = ref.qb[layer].reshape(-1).tolist()

        rep.add(
            f"weights[{layer}].mem_bitwise",
            w_mem == w_ref,
            f"mem({len(w_mem)}) vs ref({len(w_ref)})",
        )
        rep.add(
            f"weights[{layer}].mif_matches_mem",
            w_mif == w_mem,
            "mif values == mem values",
        )
        rep.add(
            f"weights[{layer}].mif_addresses_contiguous",
            w_mif_addrs == list(range(len(w_mem))),
            f"addr[0]={w_mif_addrs[0] if w_mif_addrs else 'n/a'} n={len(w_mif_addrs)}",
        )
        rep.add(
            f"biases[{layer}].mem_bitwise",
            b_mem == b_ref,
            f"mem({len(b_mem)}) vs ref({len(b_ref)})",
        )
        rep.add(
            f"biases[{layer}].mif_matches_mem",
            b_mif == b_mem,
            "mif values == mem values",
        )
        rep.add(
            f"biases[{layer}].mif_addresses_contiguous",
            b_mif_addrs == list(range(len(b_mem))),
            f"addr[0]={b_mif_addrs[0] if b_mif_addrs else 'n/a'} n={len(b_mif_addrs)}",
        )

        # element count / bit width against the manifest record
        rep.add(
            f"weights[{layer}].count_and_width",
            len(w_mem) == w_meta["numel"] == w_ref.__len__() and w_meta["bit_width"] == WEIGHT_BITS,
            f"numel={len(w_mem)} width={w_meta['bit_width']}",
        )
        rep.add(
            f"biases[{layer}].count_and_width",
            len(b_mem) == b_meta["numel"] == b_ref.__len__() and b_meta["bit_width"] == BIAS_BITS,
            f"numel={len(b_mem)} width={b_meta['bit_width']}",
        )

        # negative values must be present (the round-trip is actually exercised)
        n_neg = sum(1 for v in w_mem if v < 0)
        rep.add(
            f"weights[{layer}].has_negative",
            n_neg > 0,
            f"{n_neg} negative values",
        )
        if n_neg == 0:
            neg_ok = False

    rep.add("all_weight_layers_have_negative_values", neg_ok)

    # explicit negative two's-complement round-trip: re-encoding a parsed value
    # must reproduce the exact hex line that was read (lossless incl. negatives).
    rt_ok = True
    for layer in ("stem_conv", "conv2", "conv3", "fc"):
        mem_path = export_dir / manifest["weights"][layer]["file"]
        raw = [ln.split("//", 1)[0].strip() for ln in mem_path.read_text(encoding="ascii").splitlines()]
        raw = [ln for ln in raw if ln]
        for line, val in zip(raw, read_mem(mem_path, WEIGHT_BITS)):
            if to_twos_complement_hex(val, WEIGHT_BITS) != line.upper():
                rt_ok = False
    rep.add("negative_twos_complement_round_trip", rt_ok)


# ---------------------------------------------------------------------------
# Constants / manifest consistency
# ---------------------------------------------------------------------------


def _check_constants(export_dir: Path, ref: Int8Reference, config: Dict[str, Any], rep: CheckReport) -> None:
    manifest = _load_manifest(export_dir)
    ok_gap = manifest["network"]["gap"]["divisor"] == 49
    rep.add("gap_divisor_is_49", ok_gap, str(manifest["network"]["gap"]["divisor"]))

    for layer in ("stem_conv", "conv2", "conv3"):
        rq_manifest = manifest["network"]["layers"][layer]["requant"]
        rq_ref = ref.requant[layer]
        ok = (
            int(rq_manifest["multiplier"]) == rq_ref["multiplier"]
            and int(rq_manifest["shift"]) == rq_ref["shift"]
            and int(rq_manifest["bit_width"]) == 32
            and rq_manifest["signed"] is True
        )
        rep.add(
            f"requant[{layer}]_multiplier_shift",
            ok,
            f"m={rq_manifest['multiplier']} s={rq_manifest['shift']} "
            f"(ref m={rq_ref['multiplier']} s={rq_ref['shift']})",
        )

    # scales (fc input activation = conv3_relu: gap stays in conv3 units)
    fc_in_act = "conv3_relu"
    input_act = {"stem_conv": "input", "conv2": "stem_relu", "conv3": "conv2_relu", "fc": fc_in_act}
    for layer, meta in manifest["network"]["layers"].items():
        ok = abs(float(meta["weight_scale"]) - float(config["weight_scale"][layer])) < 1e-15
        ok = ok and abs(
            float(meta["activation_input_scale"]) - float(config["activation_scale"][input_act[layer]])
        ) < 1e-15
        rep.add(f"scale[{layer}]_matches_config", ok, f"w={meta['weight_scale']}")
    rep.add(
        "input_scale_matches_config",
        abs(float(manifest["network"]["input_scale"]) - float(config["activation_scale"]["input"])) < 1e-15,
    )

    # layouts
    rep.add(
        "layouts_match",
        manifest["layouts"] == {
            "conv_weight": "OIHW",
            "fc_weight": "OI",
            "feature_map": "CHW",
        },
    )

    # the committed .vh is byte-identical to a fresh generation (proves the
    # header carries the current multipliers / shifts and nothing stale)
    vh_text = (export_dir / "params" / "baseline_cnn_params.vh").read_text(encoding="ascii")
    rep.add("vh_reproducible", vh_text == _params_header(ref.requant))
    rep.add("vh_contains_multipliers", all(
        str(ref.requant[l]["multiplier"]) in vh_text and str(ref.requant[l]["shift"]) in vh_text
        for l in ("stem_conv", "conv2", "conv3")
    ))


# ---------------------------------------------------------------------------
# Golden trace
# ---------------------------------------------------------------------------


def _read_trace_node(export_dir: Path, node: str) -> List[int]:
    digits, signed = _node_format(node)
    bits = digits * 4
    return read_mem(
        export_dir / "sim" / "vectors" / "golden_trace" / f"{node}.mem",
        bits,
        unsigned=not signed,
    )


def _check_golden_trace(export_dir: Path, ref: Int8Reference, dataset, rep: CheckReport) -> None:
    tman = json.loads(
        (export_dir / "sim" / "vectors" / "golden_trace" / "trace_manifest.json").read_text(encoding="utf-8")
    )
    idx = int(tman["sample_index"])
    image, label = dataset[idx]
    assert int(label) == 8
    pred, _, trace = ref.infer(image.unsqueeze(0), return_trace=True)

    ok_all = True
    for node in TRACE_NODES:
        t = trace[node]
        files = tman["nodes"][node]
        vals = _read_trace_node(export_dir, node)
        ok = (
            vals == t.reshape(-1).tolist()
            and files["numel"] == t.numel()
            and files["min"] == int(t.min().item())
            and files["max"] == int(t.max().item())
            and files["shape"] == list(t.shape)
            and files["bit_width"] == _hex_digits_for_node(node) * 4
        )
        rep.add(f"golden_trace[{node}]", ok, f"numel={t.numel()} shape={list(t.shape)}")
        ok_all = ok_all and ok

    pred_txt = (export_dir / "sim" / "vectors" / "golden_trace" / "prediction.txt").read_text(encoding="ascii").strip()
    rep.add("golden_trace_prediction", int(pred_txt) == int(pred[0].item()) and int(pred[0].item()) == 8)
    rep.add("golden_trace_all_nodes_bitwise", ok_all)


# ---------------------------------------------------------------------------
# Smoke vectors
# ---------------------------------------------------------------------------


def _check_smoke(export_dir: Path, ref: Int8Reference, dataset, rep: CheckReport) -> None:
    sman = json.loads(
        (export_dir / "sim" / "vectors" / "smoke" / "smoke_manifest.json").read_text(encoding="utf-8")
    )
    assert sman["count"] == 10
    ok_all = True
    for sample in sman["samples"]:
        d = sample["digit"]
        idx = sample["test_index"]
        image, label = dataset[idx]
        pred, _, trace = ref.infer(image.unsqueeze(0), return_trace=True)
        sdir = export_dir / "sim" / "vectors" / "smoke" / sample["dir"]

        fc_vals = read_mem(sdir / "fc_acc.mem", 32)
        iq_vals = read_mem(sdir / "input_q.mem", 8)
        fc_ref = trace["fc_acc"].reshape(-1).tolist()
        iq_ref = trace["input_q"].reshape(-1).tolist()

        ok = (
            int(pred[0].item()) == d
            and int(label) == d
            and sample["prediction"] == d
            and fc_vals == fc_ref
            and iq_vals == iq_ref
            and sample["true_label"] == d
        )
        rep.add(f"smoke[digit{d}]_idx{idx}", ok, f"pred={pred[0].item()} fc={len(fc_vals)}")
        ok_all = ok_all and ok
    rep.add("all_10_smoke_samples_bitwise", ok_all)


# ---------------------------------------------------------------------------
# Full 10,000-image re-run from the exported files only
# ---------------------------------------------------------------------------


@torch.no_grad()
def _run_full_testset(
    ref: Int8Reference,
    hw: HardwareModel,
    test_loader,
) -> Tuple[int, int, int, Dict[int, int]]:
    """Run ref + hw over the whole test set.

    Returns ``(total, hw_correct, agreement, first_correct)`` where
    ``first_correct[d]`` is the smallest index on which the *reference*
    classifies digit ``d`` correctly (used to re-validate smoke selection).
    """
    total = 0
    hw_correct = 0
    agree = 0
    first_correct: Dict[int, int] = {}
    base = 0
    for images, labels in test_loader:
        pred_ref, _, _ = ref.infer(images, return_trace=False)
        pred_hw = hw.infer(images, return_trace=False)
        labels_l = labels.tolist()
        ref_l = pred_ref.tolist()
        hw_l = pred_hw.tolist()
        for i in range(len(labels_l)):
            idx = base + i
            if hw_l[i] == labels_l[i]:
                hw_correct += 1
            if hw_l[i] == ref_l[i]:
                agree += 1
            d = labels_l[i]
            if d not in first_correct and ref_l[i] == labels_l[i]:
                first_correct[d] = idx
        base += len(labels_l)
        total += len(labels_l)
    return total, hw_correct, agree, first_correct


def _check_full_testset(
    export_dir: Path,
    ref: Int8Reference,
    test_loader,
    smoke_manifest: Dict[str, Any],
    rep: CheckReport,
) -> None:
    hw = _hardware_model_from_export(export_dir)
    total, hw_correct, agree, first_correct = _run_full_testset(ref, hw, test_loader)
    acc = hw_correct / total
    agree_rate = agree / total

    rep.add(
        "full_testset_accuracy_98_47",
        acc == EXPECTED_ACCURACY,
        f"accuracy={acc:.6f} ({hw_correct}/{total})",
    )
    rep.add(
        "full_testset_prediction_agreement_100pct",
        agree_rate == 1.0,
        f"agreement={agree_rate:.6f} (disagree={total - agree})",
    )

    # re-validate that each smoke sample is the smallest correct index for its digit
    ok_sel = True
    details = []
    for sample in smoke_manifest["samples"]:
        d = sample["digit"]
        ok = first_correct.get(d) == sample["test_index"]
        ok_sel = ok_sel and ok
        details.append(f"{d}->{first_correct.get(d)}")
    rep.add("smoke_selection_is_smallest_correct_index", ok_sel, " ".join(details))


# ---------------------------------------------------------------------------
# Determinism: export twice, byte-compare controlled files
# ---------------------------------------------------------------------------


def _check_determinism(export_dir: Path, ref: Int8Reference, config, test_loader, rep: CheckReport) -> None:
    tmp_a = Path(tempfile.mkdtemp(prefix="hw_export_a_"))
    tmp_b = Path(tempfile.mkdtemp(prefix="hw_export_b_"))
    try:
        gen_a = export_all(ref, config, test_loader, tmp_a)
        gen_b = export_all(ref, config, test_loader, tmp_b)

        same_tmp = all((tmp_a / f).read_bytes() == (tmp_b / f).read_bytes() for f in gen_a)
        rep.add("repeated_export_byte_identical", same_tmp, f"{len(gen_a)} controlled files")

        same_committed = True
        missing = []
        for f in gen_a:
            p = export_dir / f
            if not p.exists():
                missing.append(f)
                same_committed = False
            elif p.read_bytes() != (tmp_a / f).read_bytes():
                same_committed = False
                missing.append(f)
        rep.add(
            "committed_export_reproducible",
            same_committed,
            f"{len(missing)} mismatched/missing vs a fresh export",
        )

        # checksums.sha256 itself must hash all controlled files
        csum_text = (tmp_a / "params" / "checksums.sha256").read_text(encoding="ascii")
        lines = [ln for ln in csum_text.splitlines() if ln.strip()]
        ok_csum = len(lines) == len(gen_a) - 1  # all generated except itself
        for ln in lines:
            h, _, rel = ln.partition("  ")
            p = tmp_a / rel
            if not p.exists() or sha256_file(p) != h.strip():
                ok_csum = False
        rep.add("checksums_sha256_valid", ok_csum, f"{len(lines)} entries")
    finally:
        shutil.rmtree(tmp_a, ignore_errors=True)
        shutil.rmtree(tmp_b, ignore_errors=True)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the BaselineCNN hardware export")
    parser.add_argument("--export-dir", type=str, default=str(DEFAULT_EXPORT_DIR))
    parser.add_argument("--artifact", type=str, default=str(DEFAULT_ARTIFACT))
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--data-root", type=str, default=str(DEFAULT_DATA_ROOT))
    args = parser.parse_args()

    export_dir = Path(args.export_dir)
    device = torch.device("cpu")

    print("=" * 78)
    print("BaselineCNN hardware export verification")
    print("=" * 78)
    print(f"export-dir : {args.export_dir}")

    model, meta = load_bn_fused_model(args.artifact, BaselineCNN, device)
    model.eval()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    ref = Int8Reference(model, config)
    test_loader = get_test_loader(root=args.data_root, batch_size=TEST_BATCH_SIZE, num_workers=0)
    dataset = get_mnist_dataset(args.data_root, train=False, download=True)

    rep = CheckReport()

    # ---- files / bit widths / addresses / two's complement ----
    _check_weight_bias_files(export_dir, ref, rep)

    # ---- constants / manifest ----
    _check_constants(export_dir, ref, config, rep)

    # ---- golden trace ----
    _check_golden_trace(export_dir, ref, dataset, rep)

    # ---- smoke vectors ----
    smoke_manifest = json.loads(
        (export_dir / "sim" / "vectors" / "smoke" / "smoke_manifest.json").read_text(encoding="utf-8")
    )
    _check_smoke(export_dir, ref, dataset, rep)

    # ---- full 10,000-image re-run from files only ----
    _check_full_testset(export_dir, ref, test_loader, smoke_manifest, rep)

    # ---- determinism ----
    _check_determinism(export_dir, ref, config, test_loader, rep)

    # ---- report ----
    print("\n-- checks ---------------------------------------------------")
    for name, ok, detail in rep.checks:
        flag = "PASS" if ok else "FAIL"
        print(f"{flag}  {name}" + (f"  [{detail}]" if detail else ""))
    print("\n" + "=" * 78)
    if rep.passed:
        print(f"RESULT: all {len(rep.checks)} checks PASSED.")
        return 0
    failed = [n for n, ok, _ in rep.checks if not ok]
    print(f"RESULT: {len(failed)} check(s) FAILED: {failed}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
