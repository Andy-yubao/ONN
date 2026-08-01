"""INT8 PTQ feasibility evaluation for the BN-fused BaselineCNN.

Software W8A8 simulation of the fused FP32 model
``model/artifacts/baseline_cnn_seed43_bn_fused_fp32.pt``:

    fused FP32 model
      -> calibrate activations (fixed 2048-image training subset)
      -> quantise weights + activations
      -> simulate W8A8 inference (scheme A: per-tensor; scheme B: per-channel)
      -> compare accuracy / agreement / saturation vs the FP32 reference

This phase is a software quantisation study only; it does **not** export
Verilog, ``.mem``, ``.hex`` or claim FPGA bit-accuracy.

Outputs (written under ``experiments/model_deployment/baseline_cnn_int8_ptq/``):
    calibration_stats.json
    ptq_results.json
    candidate_quant_config.json

Usage (from the repository root):

    python model/evaluate_baseline_int8_ptq.py
    python model/evaluate_baseline_int8_ptq.py --data-root model/data
    python model/evaluate_baseline_int8_ptq.py --output-dir experiments/model_deployment/baseline_cnn_int8_ptq
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from onn_model.data import get_test_loader
from onn_model.int8_ptq import (
    ACTIVATION_POSITIONS,
    WEIGHT_LAYERS,
    activation_stats,
    build_activation_scales,
    build_weight_config,
    calibrate_activations,
    evaluate_w8a8,
    finalize_weight_config,
    get_calibration_loader,
    w8a8_forward,
    weight_stats,
)
from onn_model.models.baseline_cnn import BaselineCNN
from onn_model.quantization import load_bn_fused_model

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_ARTIFACT = REPO_ROOT / "model" / "artifacts" / "baseline_cnn_seed43_bn_fused_fp32.pt"
DEFAULT_DATA_ROOT = REPO_ROOT / "model" / "data"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "experiments" / "model_deployment" / "baseline_cnn_int8_ptq"

CALIB_SEED = 12345
CALIB_SAMPLES = 2048
CALIB_BATCH_SIZE = 256
TEST_BATCH_SIZE = 128

# Accuracy-drop acceptance threshold (spec section 8): <= 0.2 percentage point.
ACCEPT_DROP_PP = 0.2


def _to_serializable(obj: Any) -> Any:
    """Convert torch tensors / numpy values to plain JSON types."""
    if isinstance(obj, torch.Tensor):
        return _to_serializable(obj.detach().cpu().tolist())
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    return obj


@torch.no_grad()
def fp32_accuracy(model, loader, device) -> float:
    correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        correct += (model(images).argmax(dim=1) == labels).sum().item()
        total += images.size(0)
    return correct / total if total else 0.0


def run_scheme(
    name: str,
    method: str,
    fc_method: str,
    model,
    test_loader,
    device,
    act_scales,
) -> Dict[str, Any]:
    """Build, finalise and evaluate one scheme; return its full result."""
    weight_cfg = build_weight_config(model, method, fc_scheme=fc_method)
    finalize_weight_config(weight_cfg, act_scales)
    result = evaluate_w8a8(model, test_loader, device, act_scales, weight_cfg)
    return {
        "name": name,
        "weight_method": method,
        "fc_method": fc_method,
        **result,
        "weight_saturation": {
            layer: {
                "count": weight_cfg[layer]["saturated_count"],
                "fraction": weight_cfg[layer]["saturated_fraction"],
            }
            for layer in WEIGHT_LAYERS
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="INT8 PTQ feasibility evaluation")
    parser.add_argument("--artifact", type=str, default=str(DEFAULT_ARTIFACT))
    parser.add_argument("--data-root", type=str, default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--calib-seed", type=int, default=CALIB_SEED)
    parser.add_argument("--calib-samples", type=int, default=CALIB_SAMPLES)
    args = parser.parse_args()

    device = torch.device("cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("BaselineCNN INT8 PTQ feasibility evaluation (W8A8, software)")
    print("=" * 78)
    print(f"artifact   : {args.artifact}")
    print(f"data root  : {args.data_root}")
    print(f"device     : {device}")

    # ----------------------------------------------------------------- load
    fused, meta = load_bn_fused_model(args.artifact, BaselineCNN, device)
    fused.eval()
    print(f"model      : {meta['model_name']}  (source seed {meta.get('source_seed')})")
    print(f"FP32 ref acc (artifact): {meta.get('fused_test_accuracy', '?')}")

    # ----------------------------------------------------- calibration loader
    calib_loader, calib_indices = get_calibration_loader(
        args.data_root,
        num_samples=args.calib_samples,
        seed=args.calib_seed,
        batch_size=CALIB_BATCH_SIZE,
    )
    print(f"\n-- calibration ------------------------------------------------")
    print(f"samples    : {len(calib_indices)}  seed={args.calib_seed}")
    print(f"indices    : {calib_indices[:8]} ... {calib_indices[-3:]}")

    # -------------------------------------------------- activation statistics
    calib = calibrate_activations(fused, calib_loader, device)
    calib_stats = {p: activation_stats(v) for p, v in calib.items()}
    wstats = weight_stats(fused)
    print(f"positions  : {list(calib_stats.keys())}")
    print(f"{'position':12s} {'min':>9s} {'max':>9s} {'max_abs':>9s} {'p99':>9s} "
          f"{'p99.9':>9s} {'p99.99':>9s}")
    for p in ACTIVATION_POSITIONS:
        s = calib_stats[p]
        print(f"{p:12s} {s['min']:9.4f} {s['max']:9.4f} {s['max_abs']:9.4f} "
              f"{s['p99']:9.4f} {s['p99.9']:9.4f} {s['p99.99']:9.4f}")

    # ----------------------------------------------------- test loader
    test_loader = get_test_loader(root=args.data_root, batch_size=TEST_BATCH_SIZE, num_workers=0)

    # ---------------------------------------------------- FP32 baseline
    print("\n-- FP32 baseline ---------------------------------------------")
    acc_fp = fp32_accuracy(fused, test_loader, device)
    print(f"FP32 test accuracy : {acc_fp:.6f}")

    # -------------------------------------------------------- act scales
    act_scales = build_activation_scales(calib)
    act_scales_clip = build_activation_scales(calib, clip_percentile=99.9)
    print("\n-- activation scales (full range) ----------------------------")
    for p, s in act_scales.items():
        clip = act_scales_clip.get(p)
        print(f"{p:12s} scale={s:.6e}" + (f"   p99.9={clip:.6e}" if clip else ""))

    # -------------------------------------------------------- run schemes
    print("\n-- W8A8 schemes ---------------------------------------------")
    schemes: Dict[str, Dict[str, Any]] = {}
    # Scheme A: weights per-tensor (hardware-simple)
    schemes["scheme_a"] = run_scheme(
        "scheme_a", "per-tensor", "per-tensor", fused, test_loader, device, act_scales
    )
    # Scheme B: convs per-output-channel; fc tested both ways
    schemes["scheme_b_fc_tensor"] = run_scheme(
        "scheme_b_fc_tensor", "per-output-channel", "per-tensor",
        fused, test_loader, device, act_scales,
    )
    schemes["scheme_b_fc_channel"] = run_scheme(
        "scheme_b_fc_channel", "per-output-channel", "per-output-channel",
        fused, test_loader, device, act_scales,
    )

    for name, r in schemes.items():
        print(f"{name:20s} acc={r['w8a8_accuracy']:.6f}  drop={r['accuracy_drop_pp']:+.3f} pp  "
              f"agree={r['agreement_rate']*100:6.2f}%  "
              f"disagree={r['disagree']:4d}  max_logits_err={r['max_logits_abs_err']:.3e}  "
              f"nan={r['nan_or_inf']}")

    # --------------------------------------------- p99.9 clipping (if needed)
    best_base = min(schemes.values(), key=lambda r: r["accuracy_drop_pp"])
    if best_base["accuracy_drop_pp"] > ACCEPT_DROP_PP:
        print("\n-- base W8A8 drop > 0.2 pp -> one extra test: B + p99.9 clip --")
        schemes["scheme_b_fc_channel_p99_9"] = run_scheme(
            "scheme_b_fc_channel_p99_9", "per-output-channel", "per-output-channel",
            fused, test_loader, device, act_scales_clip,
        )
        r = schemes["scheme_b_fc_channel_p99_9"]
        print(f"{'p99.9 clip':20s} acc={r['w8a8_accuracy']:.6f}  "
              f"drop={r['accuracy_drop_pp']:+.3f} pp  agree={r['agreement_rate']*100:6.2f}%  "
              f"disagree={r['disagree']:4d}  max_logits_err={r['max_logits_abs_err']:.3e}  "
              f"nan={r['nan_or_inf']}")

    # ----------------------------------------------------- determinism check
    print("\n-- determinism (scheme A, first test batch, run twice) ---------")
    first_images = next(iter(test_loader))[0].to(device)
    wc_a = build_weight_config(fused, "per-tensor")
    finalize_weight_config(wc_a, act_scales)
    out1, _ = w8a8_forward(fused, first_images, act_scales, wc_a)
    out2, _ = w8a8_forward(fused, first_images, act_scales, wc_a)
    det_ok = bool(torch.equal(out1, out2))
    print(f"repeated W8A8 identical : {det_ok}")

    # --------------------------------------------------------- candidate
    def _meets(r: Dict[str, Any]) -> bool:
        return r["accuracy_drop_pp"] <= ACCEPT_DROP_PP and not r["nan_or_inf"]

    candidate = None
    if _meets(schemes["scheme_a"]):
        candidate = "scheme_a"
    elif _meets(schemes["scheme_b_fc_channel"]) or _meets(schemes["scheme_b_fc_tensor"]):
        candidate = min(
            ("scheme_b_fc_tensor", "scheme_b_fc_channel"),
            key=lambda n: schemes[n]["accuracy_drop_pp"],
        )
        if not _meets(schemes[candidate]):
            candidate = None
    elif "scheme_b_fc_channel_p99_9" in schemes and _meets(schemes["scheme_b_fc_channel_p99_9"]):
        candidate = "scheme_b_fc_channel_p99_9"

    print("\n-- recommendation --------------------------------------------")
    if candidate is not None:
        r = schemes[candidate]
        print(f"candidate   : {candidate}  (acc {r['w8a8_accuracy']:.6f}, "
              f"drop {r['accuracy_drop_pp']:+.3f} pp)")
    else:
        print("candidate   : NONE — all INT8 schemes drop > 0.2 pp")
        print("  possible next steps: adjust calibration / mixed precision / INT16 / QAT")

    # ------------------------------------------------------- write outputs
    candidate_meta = schemes[candidate] if candidate is not None else None

    calibration_file = output_dir / "calibration_stats.json"
    calibration_file.write_text(
        json.dumps(
            {
                "calibration": {
                    "source": "MNIST train split",
                    "num_samples": len(calib_indices),
                    "seed": args.calib_seed,
                    "indices": calib_indices,
                },
                "activation": calib_stats,
                "weights": wstats,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\ncalibration stats   : {calibration_file}")

    ptq_file = output_dir / "ptq_results.json"
    ptq_file.write_text(
        json.dumps(
            {
                "model": "BaselineCNN",
                "source_artifact": args.artifact,
                "fp32_accuracy": acc_fp,
                "acceptance_drop_pp": ACCEPT_DROP_PP,
                "deterministic": det_ok,
                "candidate": candidate,
                "schemes": {n: _to_serializable(r) for n, r in schemes.items()},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"ptq results         : {ptq_file}")

    if candidate is not None:
        chosen = schemes[candidate]
        method = chosen["weight_method"]
        fc_method = chosen["fc_method"]
        scales_used = act_scales_clip if candidate.endswith("p99_9") else act_scales
        wc = build_weight_config(fused, method, fc_scheme=fc_method)
        finalize_weight_config(wc, scales_used)
        config = {
            "model": "BaselineCNN",
            "source_artifact": args.artifact,
            "scheme": candidate,
            "calibration": {
                "source": "MNIST train split",
                "num_samples": len(calib_indices),
                "seed": args.calib_seed,
                "indices": calib_indices,
            },
            "weight_quantization": {
                "conv_method": method,
                "fc_method": fc_method,
                "dtype": "sint8",
                "symmetric": True,
                "zero_point": 0,
                "range": [-127, 127],
            },
            "weight_scale": {
                layer: _to_serializable(wc[layer]["weight_scale"]) for layer in WEIGHT_LAYERS
            },
            "activation_scale": {p: scales_used[p] for p in scales_used},
            "input_range": {"dtype": "sint8", "min": -128, "max": 127, "zero_point": 0},
            "output_range": {"dtype": "uint8", "min": 0, "max": 255, "zero_point": 0},
            "accumulator": "int32",
            "rounding": "round-half-away-from-zero",
            "saturation": "clamp-saturate-no-wraparound",
        }
        config_file = output_dir / "candidate_quant_config.json"
        config_file.write_text(json.dumps(config, indent=2), encoding="utf-8")
        print(f"candidate config    : {config_file}")
    else:
        config_file = None

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"FP32 test accuracy        : {acc_fp:.6f}")
    for name in ("scheme_a", "scheme_b_fc_tensor", "scheme_b_fc_channel"):
        r = schemes[name]
        print(f"{name:20s} : acc {r['w8a8_accuracy']:.6f}  drop {r['accuracy_drop_pp']:+.3f} pp")
    if "scheme_b_fc_channel_p99_9" in schemes:
        r = schemes["scheme_b_fc_channel_p99_9"]
        print(f"{'p99.9 clip':20s} : acc {r['w8a8_accuracy']:.6f}  drop {r['accuracy_drop_pp']:+.3f} pp")
    print(f"candidate                 : {candidate if candidate else 'NONE'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
