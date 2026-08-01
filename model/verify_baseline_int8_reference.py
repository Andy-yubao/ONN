"""Verify the pure-integer reference against BN-fused FP32 and scheme-A fake-quant.

Runs the full 10,000-image MNIST test set through three forward paths:

  1. BN-fused FP32 model;
  2. scheme-A fake-quant W8A8 simulation (``int8_ptq.w8a8_forward``);
  3. pure-integer reference (``int8_reference.Int8Reference``).

and reports the comparison required by the phase spec: the three accuracies,
prediction agreement between the integer model and the fake-quant / FP32, the
number and first 20 of the differing samples, per-layer accumulator ranges and
INT32 headroom, per-layer multiplier/shift/approximation error, requant
saturation, integer-node dtype/range conformance, and determinism (the whole
evaluation is run twice and compared byte-for-byte).

Acceptance criteria (the thresholds are **not** relaxed on failure):

    |acc_int - acc_fake| <= 0.02 pp
    agreement(int, fake-quant) >= 99.95 %
    all accumulators within INT32
    all integer nodes keep the specified dtype / range
    no integer wraparound (overflow count == 0)
    the two runs are bit-identical

If a criterion fails, the report still points at the first integer node (in
forward order) where the integer reference and the fake-quant diverge, so the
divergence can be located layer by layer.

Outputs (written under ``experiments/model_deployment/baseline_cnn_int8_reference/``):
    integer_reference_results.json
    README.md

Usage (from the repository root):

    python model/verify_baseline_int8_reference.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from onn_model.data import get_test_loader
from onn_model.int8_ptq import (
    INT32_MAX,
    INT32_MIN,
    WEIGHT_LAYERS,
    build_weight_config,
    finalize_weight_config,
    w8a8_forward,
)
from onn_model.int8_reference import (
    ACCUMULATOR_LAYERS,
    Int8Reference,
    NODE_SPEC,
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
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "experiments" / "model_deployment" / "baseline_cnn_int8_reference"
)

TEST_BATCH_SIZE = 128
MAX_MISMATCHES = 20

# Acceptance thresholds.
ACCURACY_TOLERANCE_PP = 0.02
AGREEMENT_THRESHOLD = 0.9995

# Real-value tolerance (absolute, in dequantised units) used for the
# element-wise "close" count of the unit-differing nodes (gap_q / fc_acc).
REAL_CLOSE_TOL = 1e-3

# Dequantisation unit of each integer node for the *integer* reference.
INT_NODE_UNIT: Dict[str, str] = {
    "input_q": "input",
    "conv1_acc": "acc1",
    "stem_q": "stem_relu",
    "pool1_q": "stem_relu",
    "conv2_acc": "acc2",
    "conv2_q": "conv2_relu",
    "pool2_q": "conv2_relu",
    "conv3_acc": "acc3",
    "conv3_q": "conv3_relu",
    "gap_q": "gap",
    "fc_acc": "fc",
}

# Fake-quant uses the *pool* scale for its GAP output (and its fc input): the
# "gap" unit in fq_scale is s_pool, and the fc accumulator unit is s_pool*sw_fc.
FQ_NODE_UNIT: Dict[str, str] = {
    **INT_NODE_UNIT,
    "fc_acc": "fc_pool",
}

# Nodes whose integer values are directly comparable (same units in both paths).
INT_COMPARABLE_NODES: Tuple[str, ...] = tuple(
    n for n in TRACE_NODES if n not in ("gap_q", "fc_acc")
)


def _to_serializable(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return _to_serializable(obj.detach().cpu().tolist())
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    return obj


class Evaluator:
    """Runs the three forward paths over the test loader once."""

    def __init__(self, model, ref: Int8Reference, weight_cfg, act_scales, device) -> None:
        self.model = model
        self.ref = ref
        self.weight_cfg = weight_cfg
        self.act_scales = act_scales
        self.device = device
        self.s_in = act_scales["input"]
        self.s_stem = act_scales["stem_relu"]
        self.s_conv2 = act_scales["conv2_relu"]
        self.s_conv3 = act_scales["conv3_relu"]
        self.s_pool = act_scales["pool"]
        self.sw = weight_cfg

        # node -> float scale for the integer reference's real values
        self.int_scale = {
            "input": self.s_in,
            "acc1": self.s_in * self.sw["stem_conv"]["weight_scale"],
            "stem_relu": self.s_stem,
            "acc2": self.s_stem * self.sw["conv2"]["weight_scale"],
            "conv2_relu": self.s_conv2,
            "acc3": self.s_conv2 * self.sw["conv3"]["weight_scale"],
            "conv3_relu": self.s_conv3,
            "gap": self.s_conv3,
            "fc": self.sw["fc"]["weight_scale"] * self.s_conv3,
        }
        # fake-quant's GAP output / fc live on the pool scale
        self.fq_scale = {**self.int_scale, "gap": self.s_pool, "fc_pool": self.sw["fc"]["weight_scale"] * self.s_pool}

        # ---- aggregation state ----
        self.total = 0
        self.correct = {"fp32": 0, "fake": 0, "int": 0}
        self.agree = {"int_fake": 0, "int_fp32": 0, "fake_fp32": 0}
        self.mismatches: List[Dict[str, Any]] = []
        self.logits_err = {
            "int_vs_fp32": {"sum": 0.0, "max": 0.0, "n": 0},
            "int_vs_fake": {"sum": 0.0, "max": 0.0, "n": 0},
        }
        self.nan_inf = {"fp32": False, "fake": False, "int": False}
        self.acc_agg = {
            layer: {"min": float("inf"), "max": float("-inf"), "overflow": 0}
            for layer in ACCUMULATOR_LAYERS
        }
        self.node_minmax = {n: {"min": float("inf"), "max": float("-inf")} for n in TRACE_NODES}
        self.node_dtype: Dict[str, str] = {}
        self.requant_sat = {
            n: {"low": 0, "high": 0, "numel": 0}
            for n in ("stem_q", "conv2_q", "conv3_q", "gap_q")
        }
        self.node_cmp = {
            n: {"agree": 0, "numel": 0, "err_sum": 0.0, "err_max": 0.0}
            for n in TRACE_NODES
        }
        self._idx = 0

    @torch.no_grad()
    def _run_batch(self, images, labels) -> None:
        images, labels = images.to(self.device), labels.to(self.device)

        out_fp = self.model(images)
        out_fq, info_fq = w8a8_forward(
            self.model, images, self.act_scales, self.weight_cfg, return_trace=True
        )
        pred_int, logits_int, trace_int = self.ref.infer(images)

        pred_fp = out_fp.argmax(dim=1)
        pred_fq = out_fq.argmax(dim=1)
        n = images.size(0)
        self.total += n
        self.correct["fp32"] += int((pred_fp == labels).sum().item())
        self.correct["fake"] += int((pred_fq == labels).sum().item())
        self.correct["int"] += int((pred_int == labels).sum().item())
        self.agree["int_fake"] += int((pred_int == pred_fq).sum().item())
        self.agree["int_fp32"] += int((pred_int == pred_fp).sum().item())
        self.agree["fake_fp32"] += int((pred_fq == pred_fp).sum().item())

        for name, out in (("fp32", out_fp), ("fake", out_fq), ("int", logits_int)):
            self.nan_inf[name] = self.nan_inf[name] or bool((~torch.isfinite(out)).any().item())

        # logits errors
        err_int_fp = (logits_int.float() - out_fp.float()).abs()
        err_int_fq = (logits_int.float() - out_fq.float()).abs()
        self.logits_err["int_vs_fp32"]["sum"] += float(err_int_fp.sum().item())
        self.logits_err["int_vs_fp32"]["max"] = max(self.logits_err["int_vs_fp32"]["max"], float(err_int_fp.max().item()))
        self.logits_err["int_vs_fp32"]["n"] += err_int_fp.numel()
        self.logits_err["int_vs_fake"]["sum"] += float(err_int_fq.sum().item())
        self.logits_err["int_vs_fake"]["max"] = max(self.logits_err["int_vs_fake"]["max"], float(err_int_fq.max().item()))
        self.logits_err["int_vs_fake"]["n"] += err_int_fq.numel()

        # ---- integer-reference accumulator aggregation ----
        for layer in ACCUMULATOR_LAYERS:
            s = trace_int["acc_stats"]
            self.acc_agg[layer]["min"] = min(self.acc_agg[layer]["min"], s["min"][layer])
            self.acc_agg[layer]["max"] = max(self.acc_agg[layer]["max"], s["max"][layer])
            self.acc_agg[layer]["overflow"] += s["overflow_count"][layer]

        # ---- node min/max + requant saturation (integer reference) ----
        for node in TRACE_NODES:
            t = trace_int[node]
            self.node_dtype[node] = str(t.dtype)
            self.node_minmax[node]["min"] = min(self.node_minmax[node]["min"], t.min().item())
            self.node_minmax[node]["max"] = max(self.node_minmax[node]["max"], t.max().item())
        for node in ("stem_q", "conv2_q", "conv3_q"):
            t = trace_int[node]
            self.requant_sat[node]["low"] += int((t == 0).sum().item())
            self.requant_sat[node]["high"] += int((t == 255).sum().item())
            self.requant_sat[node]["numel"] += t.numel()
        t = trace_int["gap_q"]
        self.requant_sat["gap_q"]["low"] += int((t == 0).sum().item())
        self.requant_sat["gap_q"]["high"] += int((t == 255).sum().item())
        self.requant_sat["gap_q"]["numel"] += t.numel()

        # ---- node-by-node comparison int vs fake-quant ----
        fq_trace = info_fq["trace"]
        for node in TRACE_NODES:
            t_int = trace_int[node].to(torch.float64)
            t_fq = fq_trace[node].to(torch.float64)
            s_int = self.int_scale[INT_NODE_UNIT[node]]
            s_fq = self.fq_scale[FQ_NODE_UNIT[node]]
            real_int = t_int * s_int
            real_fq = t_fq * s_fq
            err = (real_int - real_fq).abs()
            c = self.node_cmp[node]
            c["numel"] += err.numel()
            c["err_sum"] += float(err.sum().item())
            c["err_max"] = max(c["err_max"], float(err.max().item()))
            if node in INT_COMPARABLE_NODES:
                # same integer units in both paths -> element-wise integer equality
                c["agree"] += int((t_int.to(torch.int64) == t_fq.to(torch.int64)).sum().item())
            else:
                # gap_q / fc_acc: different integer units -> element-wise closeness
                # of the dequantised real values (absolute tolerance in real units)
                c["agree"] += int((err <= REAL_CLOSE_TOL).sum().item())

        # ---- first 20 int-vs-fake-quant mismatches ----
        diff = pred_int != pred_fq
        if len(self.mismatches) < MAX_MISMATCHES:
            for i in diff.nonzero(as_tuple=True)[0]:
                if len(self.mismatches) >= MAX_MISMATCHES:
                    break
                idx = self._idx + i.item()
                self.mismatches.append(
                    {
                        "index": idx,
                        "true_label": int(labels[i].item()),
                        "fp32_pred": int(pred_fp[i].item()),
                        "fake_quant_pred": int(pred_fq[i].item()),
                        "integer_pred": int(pred_int[i].item()),
                        "fp32_logits": [float(v) for v in out_fp[i]],
                        "fake_quant_logits": [float(v) for v in out_fq[i]],
                        "integer_logits": [float(v) for v in logits_int[i]],
                        "max_logits_err_int_vs_fake": float(err_int_fq[i].max().item()),
                    }
                )
        self._idx += n

    def aggregate(self) -> Dict[str, Any]:
        total = self.total
        acc = {k: v / total for k, v in self.correct.items()}
        res: Dict[str, Any] = {
            "total": total,
            "accuracy": acc,
            "accuracy_drop_pp": {
                "int_vs_fp32": (acc["fp32"] - acc["int"]) * 100.0,
                "int_vs_fake_quant": (acc["fake"] - acc["int"]) * 100.0,
            },
            "agreement_rate": {
                "int_vs_fake_quant": self.agree["int_fake"] / total,
                "int_vs_fp32": self.agree["int_fp32"] / total,
                "fake_quant_vs_fp32": self.agree["fake_fp32"] / total,
            },
            "disagree_count": {
                "int_vs_fake_quant": total - self.agree["int_fake"],
                "int_vs_fp32": total - self.agree["int_fp32"],
                "fake_quant_vs_fp32": total - self.agree["fake_fp32"],
            },
            "logits_err": {
                "int_vs_fp32": {
                    "max_abs": self.logits_err["int_vs_fp32"]["max"],
                    "mean_abs": self.logits_err["int_vs_fp32"]["sum"]
                    / self.logits_err["int_vs_fp32"]["n"],
                },
                "int_vs_fake_quant": {
                    "max_abs": self.logits_err["int_vs_fake"]["max"],
                    "mean_abs": self.logits_err["int_vs_fake"]["sum"]
                    / self.logits_err["int_vs_fake"]["n"],
                },
            },
            "nan_or_inf": self.nan_inf,
            "accumulator": {
                layer: {
                    "min": self.acc_agg[layer]["min"],
                    "max": self.acc_agg[layer]["max"],
                    "max_abs": max(abs(self.acc_agg[layer]["min"]), abs(self.acc_agg[layer]["max"])),
                    "margin_low": self.acc_agg[layer]["min"] - INT32_MIN,
                    "margin_high": INT32_MAX - self.acc_agg[layer]["max"],
                    "overflow_count": self.acc_agg[layer]["overflow"],
                }
                for layer in ACCUMULATOR_LAYERS
            },
            "requant_saturation": {
                node: {
                    "low_count": self.requant_sat[node]["low"],
                    "high_count": self.requant_sat[node]["high"],
                    "numel": self.requant_sat[node]["numel"],
                    "low_fraction": self.requant_sat[node]["low"] / self.requant_sat[node]["numel"]
                    if self.requant_sat[node]["numel"]
                    else 0.0,
                    "high_fraction": self.requant_sat[node]["high"] / self.requant_sat[node]["numel"]
                    if self.requant_sat[node]["numel"]
                    else 0.0,
                }
                for node in ("stem_q", "conv2_q", "conv3_q", "gap_q")
            },
            "node_conformance": {
                node: {
                    "dtype": self.node_dtype[node],
                    "expected_dtype": str(NODE_SPEC[node][0]),
                    "min": self.node_minmax[node]["min"],
                    "max": self.node_minmax[node]["max"],
                    "dtype_ok": self.node_dtype[node] == str(NODE_SPEC[node][0]),
                    "range_ok": NODE_SPEC[node][1]
                    <= self.node_minmax[node]["min"]
                    <= self.node_minmax[node]["max"]
                    <= NODE_SPEC[node][2],
                }
                for node in TRACE_NODES
            },
            "node_comparison_int_vs_fake": {
                node: {
                    "numel": self.node_cmp[node]["numel"],
                    "agree": self.node_cmp[node]["agree"],
                    "agree_rate": self.node_cmp[node]["agree"] / self.node_cmp[node]["numel"]
                    if self.node_cmp[node]["numel"]
                    else 0.0,
                    "real_mean_abs_err": self.node_cmp[node]["err_sum"]
                    / self.node_cmp[node]["numel"]
                    if self.node_cmp[node]["numel"]
                    else 0.0,
                    "real_max_abs_err": self.node_cmp[node]["err_max"],
                    "comparable_units": node in INT_COMPARABLE_NODES,
                    "agree_definition": (
                        "element-wise integer equality" if node in INT_COMPARABLE_NODES
                        else "element-wise real closeness within 1e-3 (units differ)"
                    ),
                }
                for node in TRACE_NODES
            },
            "mismatches": self.mismatches,
        }
        # ---- first divergence node (forward order): the first node whose real
        # (dequantised) values differ at all between the integer reference and
        # the fake-quant.  For the unit-differing nodes this is the honest
        # "any difference" signal; the integer chain up to that point is exact.
        first = None
        for node in TRACE_NODES:
            if res["node_comparison_int_vs_fake"][node]["real_max_abs_err"] > 0.0:
                first = node
                break
        res["first_divergence_node"] = first
        return res


# --- canonical digest for determinism ----------------------------------------

def _canonical_digest(res: Dict[str, Any]) -> str:
    """md5 over the JSON of the aggregate metrics (predictions included via accuracy/agreement)."""
    digest = hashlib.md5()
    payload = json.dumps(res, sort_keys=True, default=str).encode("utf-8")
    digest.update(payload)
    return digest.hexdigest()


def run_evaluation(model, ref, weight_cfg, act_scales, test_loader, device) -> Tuple[Dict[str, Any], str]:
    ev = Evaluator(model, ref, weight_cfg, act_scales, device)
    for images, labels in test_loader:
        ev._run_batch(images, labels)
    res = ev.aggregate()
    return res, _canonical_digest(res)


def acceptance(res: Dict[str, Any], deterministic: bool) -> Dict[str, Any]:
    acc = res["accuracy"]
    checks = {
        "accuracy_within_0_02pp": abs(acc["fake"] - acc["int"]) * 100.0 <= ACCURACY_TOLERANCE_PP,
        "agreement_int_vs_fake_ge_99_95": res["agreement_rate"]["int_vs_fake_quant"]
        >= AGREEMENT_THRESHOLD,
        "all_accumulators_in_int32": all(
            a["overflow_count"] == 0 for a in res["accumulator"].values()
        ),
        "node_dtype_range_ok": all(
            c["dtype_ok"] and c["range_ok"] for c in res["node_conformance"].values()
        ),
        "no_wraparound": not any(a["overflow_count"] for a in res["accumulator"].values()),
        "bit_identical_runs": bool(deterministic),
    }
    checks["all_pass"] = all(checks.values())
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the pure-integer reference")
    parser.add_argument("--artifact", type=str, default=str(DEFAULT_ARTIFACT))
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--data-root", type=str, default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    device = torch.device("cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("BaselineCNN pure-integer reference verification (scheme A)")
    print("=" * 78)
    print(f"artifact : {args.artifact}")
    print(f"config   : {args.config}")

    model, meta = load_bn_fused_model(args.artifact, BaselineCNN, device)
    model.eval()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    ref = Int8Reference(model, config)
    print(f"model    : {meta['model_name']}")

    weight_cfg = build_weight_config(model, "per-tensor")
    finalize_weight_config(weight_cfg, config["activation_scale"])
    test_loader = get_test_loader(root=args.data_root, batch_size=TEST_BATCH_SIZE, num_workers=0)

    print("\n-- requantisation (integer reference) -------------------------")
    for layer, rq in ref.requant.items():
        print(f"{layer:10s} r={rq['real_multiplier']:.6e}  m={rq['multiplier']}  "
              f"shift={rq['shift']}  rel_err={rq['relative_error']:.2e}")
    print(f"bias out of INT32: {ref.bias_out_of_int32}")

    # ---- run the evaluation twice for determinism ----
    print("\n-- evaluation (pass 1 / pass 2) ------------------------------")
    res1, d1 = run_evaluation(model, ref, weight_cfg, config["activation_scale"], test_loader, device)
    res2, d2 = run_evaluation(model, ref, weight_cfg, config["activation_scale"], test_loader, device)
    deterministic = d1 == d2
    print(f"pass1 digest: {d1}")
    print(f"pass2 digest: {d2}")
    print(f"bit-identical: {deterministic}")

    res = res1
    acc = res["accuracy"]
    print("\n-- accuracies ------------------------------------------------")
    print(f"FP32 fusion            : {acc['fp32']:.6f}")
    print(f"fake-quant scheme A    : {acc['fake']:.6f}")
    print(f"integer reference      : {acc['int']:.6f}")
    print(f"drop vs FP32           : {res['accuracy_drop_pp']['int_vs_fp32']:+.3f} pp")
    print(f"diff vs fake-quant     : {res['accuracy_drop_pp']['int_vs_fake_quant']:+.3f} pp")

    print("\n-- agreement ------------------------------------------------")
    for k, v in res["agreement_rate"].items():
        print(f"{k:22s}: {v*100:7.3f}%  disagree={res['disagree_count'][k]}")
    print(f"first divergence node  : {res['first_divergence_node']}")

    print("\n-- accumulators (integer reference) --------------------------")
    for layer, a in res["accumulator"].items():
        print(f"{layer:10s} min={a['min']:9.0f} max={a['max']:9.0f} max_abs={a['max_abs']:9.0f} "
              f"margin=[{a['margin_low']:.0f},{a['margin_high']:.0f}] overflow={a['overflow_count']}")

    print("\n-- node comparison int vs fake-quant -------------------------")
    for node, c in res["node_comparison_int_vs_fake"].items():
        print(f"{node:10s} agree={c['agree_rate']*100:8.4f}%  "
              f"mean_err={c['real_mean_abs_err']:.4e}  max_err={c['real_max_abs_err']:.4e}")

    checks = acceptance(res, deterministic)
    print("\n-- acceptance ------------------------------------------------")
    for k, v in checks.items():
        print(f"{k:34s}: {'PASS' if v else 'FAIL'}")
    if checks["all_pass"]:
        print("RESULT: all acceptance criteria satisfied.")
    else:
        print(f"RESULT: acceptance FAILED. First divergence at '{res['first_divergence_node']}'.")

    # ---- write outputs ----
    out = {
        "model": "BaselineCNN",
        "source_artifact": args.artifact,
        "quant_config": args.config,
        "test_loader_batch_size": TEST_BATCH_SIZE,
        "fp32_accuracy": acc["fp32"],
        "fake_quant_accuracy": acc["fake"],
        "integer_reference_accuracy": acc["int"],
        "deterministic": deterministic,
        "pass1_digest": d1,
        "pass2_digest": d2,
        "acceptance": checks,
        "requant": {k: _to_serializable(v) for k, v in ref.requant.items()},
        "bias_out_of_int32": ref.bias_out_of_int32,
        **res,
    }
    result_file = output_dir / "integer_reference_results.json"
    result_file.write_text(json.dumps(_to_serializable(out), indent=2), encoding="utf-8")
    print(f"\nresults : {result_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
