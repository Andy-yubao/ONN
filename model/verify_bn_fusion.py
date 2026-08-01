"""BN fusion verification for the BaselineCNN delivery candidate.

Loads a trained BaselineCNN checkpoint (default: seed 43), fuses the three
Conv2d + BatchNorm2d pairs into bias-carrying Conv2d layers, and verifies
numerical equivalence:

  * per-layer (hook-based) on a fixed 256-sample batch,
  * end-to-end on the full 10k MNIST test set.

On success the fused FP32 model is saved to ``model/artifacts/``.  No weights
are retrained and no checkpoint is overwritten.

Usage (from the repository root):

    python model/verify_bn_fusion.py
    python model/verify_bn_fusion.py --checkpoint <path/to/best.pt>
    python model/verify_bn_fusion.py --output <path/to/fused.pt>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn

# ``onn_model`` is importable because this script lives in ``model/``.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from onn_model.data import get_test_loader
from onn_model.engine import _get_model
from onn_model.quantization import fuse_model_bn, load_bn_fused_model

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CHECKPOINT = REPO_ROOT / "model" / "runs" / "BaselineCNN" / "20260721_163832" / "best.pt"
DEFAULT_OUTPUT = REPO_ROOT / "model" / "artifacts" / "baseline_cnn_seed43_bn_fused_fp32.pt"

FIXED_SAMPLES = 256
FULL_BATCH_SIZE = 128

# Functional positions compared between the original and fused model.
# The same accessors work on both: ``stem[-1]``/``conv2[-1]``/``conv3[-1]`` are
# the ReLU in both structures (fusion only replaces the first two modules of
# each Sequential, the trailing ReLU keeps its position).
LAYER_TARGETS: Dict[str, str] = {
    "stem_relu": "stem[-1]",
    "pool1": "pool1",
    "conv2_relu": "conv2[-1]",
    "pool2": "pool2",
    "conv3_relu": "conv3[-1]",
    "pool": "pool",
    "logits": "fc",
}

ACCEPT_FIXED_MAX_LOGITS_ERR = 1e-4
ACCEPT_TEST_AGREEMENT = 0.9999
ACCEPT_ACC_DROP = 0.0001  # 0.01 percentage point


def count_batchnorm(model: nn.Module) -> int:
    return sum(1 for m in model.modules() if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)))


def count_module_types(model: nn.Module) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for m in model.modules():
        counts[type(m).__name__] = counts.get(type(m).__name__, 0) + 1
    return counts


def get_layer_module(model: nn.Module, accessor: str) -> nn.Module:
    """Resolve a short accessor string like ``stem[-1]`` on the model."""
    name, _, idx = accessor.partition("[")
    mod = getattr(model, name)
    if idx:
        return mod[int(idx.rstrip("]"))]
    return mod


def collect_layer_outputs(model: nn.Module, x: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Run ``model(x)`` with forward hooks, returning each layer's output."""
    captured: Dict[str, torch.Tensor] = {}
    handles: List[Any] = []

    def _make_hook(key: str):
        def _hook(_mod, _inp, out):
            captured[key] = out.detach().clone()

        return _hook

    for key, accessor in LAYER_TARGETS.items():
        handles.append(get_layer_module(model, accessor).register_forward_hook(_make_hook(key)))
    try:
        model(x)
    finally:
        for h in handles:
            h.remove()
    return captured


def layer_stats(a: torch.Tensor, b: torch.Tensor) -> Dict[str, Any]:
    d = (a - b).float()
    absd = d.abs()
    return {
        "shape_orig": list(a.shape),
        "shape_fused": list(b.shape),
        "shape_match": tuple(a.shape) == tuple(b.shape),
        "max_abs": absd.max().item(),
        "mean_abs": absd.mean().item(),
        "mse": (d * d).mean().item(),
        "nan_orig": bool(torch.isnan(a).any().item()),
        "inf_orig": bool(torch.isinf(a).any().item()),
        "nan_fused": bool(torch.isnan(b).any().item()),
        "inf_fused": bool(torch.isinf(b).any().item()),
    }


@torch.no_grad()
def eval_accuracy(model: nn.Module, loader, device: torch.device) -> float:
    correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        correct += (model(images).argmax(dim=1) == labels).sum().item()
        total += images.size(0)
    return correct / total


@torch.no_grad()
def full_test_compare(
    model_orig: nn.Module,
    model_fused: nn.Module,
    loader,
    device: torch.device,
) -> Dict[str, Any]:
    """Compare logits / predictions of both models over the full test set."""
    total = agree = 0
    max_logits_err = 0.0
    sum_logits_err = 0.0
    count = 0
    mismatches: List[Dict[str, Any]] = []
    idx = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        out_orig = model_orig(images)
        out_fused = model_fused(images)
        pred_orig = out_orig.argmax(dim=1)
        pred_fused = out_fused.argmax(dim=1)

        err = (out_orig - out_fused).abs()
        max_logits_err = max(max_logits_err, err.max().item())
        sum_logits_err += err.sum().item()
        count += err.numel()

        eq = pred_orig == pred_fused
        agree += eq.sum().item()
        total += images.size(0)
        for i in (~eq).nonzero(as_tuple=True)[0]:
            mismatches.append(
                {
                    "index": idx + i.item(),
                    "true_label": labels[i].item(),
                    "orig_pred": pred_orig[i].item(),
                    "fused_pred": pred_fused[i].item(),
                    "orig_logits": [float(v) for v in out_orig[i]],
                    "fused_logits": [float(v) for v in out_fused[i]],
                    "max_err": err[i].max().item(),
                }
            )
        idx += images.size(0)

    return {
        "total": total,
        "agree": agree,
        "disagree": total - agree,
        "agreement_rate": agree / total if total else 0.0,
        "max_logits_abs_err": max_logits_err,
        "mean_logits_abs_err": sum_logits_err / count if count else 0.0,
        "mismatches": mismatches,
    }


def _yes(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify and export BN-fused BaselineCNN")
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    output_path = Path(args.output)
    device = torch.device("cpu")

    print("=" * 78)
    print("BaselineCNN BatchNorm fusion verification")
    print("=" * 78)
    print(f"checkpoint : {ckpt_path}")
    print(f"output     : {output_path}")
    print(f"device     : {device}")

    # ------------------------------------------------------------------ load
    run_dir = ckpt_path.parent
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    model_name = cfg.get("model", "BaselineCNN")
    data_root = cfg.get("data_root", "model/data")
    if not Path(data_root).is_absolute():
        data_root = str(REPO_ROOT / data_root)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    orig = _get_model(model_name)
    strict_ok = True
    try:
        orig.load_state_dict(ckpt["model_state_dict"], strict=True)
    except RuntimeError:
        strict_ok = False
        raise
    orig.eval()

    print(f"model       : {model_name} (ckpt model_name={ckpt.get('model_name')})")
    print(f"seed        : {cfg.get('seed')}  epochs={cfg.get('epochs')}  "
          f"git_commit={json.loads((run_dir / 'environment.json').read_text()).get('git_commit', '?')}")
    print(f"original BN = {count_batchnorm(orig)}")
    print(f"strict load : {_yes(strict_ok)}")

    # ------------------------------------------------------- original not modified
    orig_state_before = {k: v.clone() for k, v in orig.state_dict().items()}

    # ----------------------------------------------------------------- fuse
    fused = fuse_model_bn(orig)
    fused.eval()

    orig_after_ok = all(
        torch.equal(orig_state_before[k], v) for k, v in orig.state_dict().items()
    )

    bn_orig = count_batchnorm(orig)
    bn_fused = count_batchnorm(fused)
    types_orig = count_module_types(orig)
    types_fused = count_module_types(fused)

    print("\n-- module counts -----------------------------------------------")
    print(f"original : BatchNorm2d={bn_orig} Conv2d={types_orig.get('Conv2d', 0)} "
          f"Linear={types_orig.get('Linear', 0)}")
    print(f"fused    : BatchNorm2d={bn_fused} Conv2d={types_fused.get('Conv2d', 0)} "
          f"Linear={types_fused.get('Linear', 0)}")
    print(f"original untouched : {_yes(orig_after_ok)}")

    # ------------------------------------------------- fixed-sample layer compare
    fixed_loader = get_test_loader(root=data_root, batch_size=FIXED_SAMPLES, num_workers=0)
    fixed_x, fixed_y = next(iter(fixed_loader))
    print(f"\n-- fixed-sample validation ({FIXED_SAMPLES} images, no shuffle) ----------")
    assert fixed_x.size(0) == FIXED_SAMPLES

    with torch.no_grad():
        out_orig = collect_layer_outputs(orig, fixed_x.clone())
        out_fused = collect_layer_outputs(fused, fixed_x.clone())

    print(f"{'layer':12s} {'shape':16s} {'max_abs':>10s} {'mean_abs':>10s} {'mse':>10s} "
          f"{'nan/inf':>8s}")
    layer_rows: Dict[str, Dict[str, Any]] = {}
    shapes_match = True
    finite_all = True
    for key in LAYER_TARGETS:
        s = layer_stats(out_orig[key], out_fused[key])
        layer_rows[key] = s
        shapes_match &= s["shape_match"]
        finite_all &= not (s["nan_orig"] or s["inf_orig"] or s["nan_fused"] or s["inf_fused"])
        print(
            f"{key:12s} {str(s['shape_orig']):16s} {s['max_abs']:10.2e} "
            f"{s['mean_abs']:10.2e} {s['mse']:10.2e} "
            f"{'OK' if not (s['nan_orig'] or s['inf_orig'] or s['nan_fused'] or s['inf_fused']) else 'NAN/INF'}"
        )

    fixed_max_logits_err = layer_rows["logits"]["max_abs"]
    fixed_pred_agree = bool(
        (out_orig["logits"].argmax(dim=1) == out_fused["logits"].argmax(dim=1)).all().item()
    )
    print(f"fixed-sample logits max|err|      : {fixed_max_logits_err:.6e}")
    print(f"fixed-sample prediction agreement : {_yes(fixed_pred_agree)}")

    # ------------------------------------------------- full test set
    full_loader = get_test_loader(root=data_root, batch_size=FULL_BATCH_SIZE, num_workers=0)
    print("\n-- full test set (10,000 samples) -------------------------------")
    orig_acc = eval_accuracy(orig, full_loader, device)
    fused_acc = eval_accuracy(fused, full_loader, device)
    compare = full_test_compare(orig, fused, full_loader, device)

    acc_drop = orig_acc - fused_acc
    print(f"original test acc : {orig_acc:.6f}")
    print(f"fused    test acc : {fused_acc:.6f}")
    print(f"acc drop          : {acc_drop:+.6f} ({(acc_drop * 100):+.4f} pp)")
    print(f"prediction agree  : {compare['agree']}/{compare['total']} "
          f"({compare['agreement_rate'] * 100:.4f}%)  disagree={compare['disagree']}")
    print(f"max logits |err|  : {compare['max_logits_abs_err']:.6e}")
    print(f"mean logits |err| : {compare['mean_logits_abs_err']:.6e}")

    # ------------------------------------------------- acceptance criteria
    print("\n-- acceptance criteria ------------------------------------------")
    results: List[Tuple[str, bool, str]] = [
        ("strict=True load", strict_ok, ""),
        ("fused BN count == 0", bn_fused == 0, f"(found {bn_fused})"),
        ("original model untouched", orig_after_ok, ""),
        ("all layer shapes match", shapes_match, ""),
        ("no NaN/Inf in outputs", finite_all, ""),
        (
            f"fixed-sample max logits err < {ACCEPT_FIXED_MAX_LOGITS_ERR}",
            fixed_max_logits_err < ACCEPT_FIXED_MAX_LOGITS_ERR,
            f"({fixed_max_logits_err:.3e})",
        ),
        (
            f"test prediction agreement >= {ACCEPT_TEST_AGREEMENT * 100:.2f}%",
            compare["agreement_rate"] >= ACCEPT_TEST_AGREEMENT,
            f"({compare['agreement_rate'] * 100:.4f}%)",
        ),
        (
            f"fused acc drop <= {ACCEPT_ACC_DROP * 100:.2f} pp",
            acc_drop <= ACCEPT_ACC_DROP,
            f"({acc_drop * 100:+.4f} pp)",
        ),
    ]
    all_pass = True
    for name, ok, extra in results:
        print(f"  [{_yes(ok)}] {name} {extra}")
        all_pass &= ok
    if compare["mismatches"]:
        all_pass = False
        print(f"\n  [{_yes(False)}] prediction mismatches found ({len(compare['mismatches'])})")

    if compare["mismatches"]:
        print("\n-- mismatched samples -------------------------------------------")
        for m in compare["mismatches"]:
            print(f"  idx={m['index']} true={m['true_label']} "
                  f"orig={m['orig_pred']} fused={m['fused_pred']} max_err={m['max_err']:.3e}")
            print(f"    orig logits : {[f'{v:.4f}' for v in m['orig_logits']]}")
            print(f"    fused logits: {[f'{v:.4f}' for v in m['fused_logits']]}")

    # ------------------------------------------------- save fused artifact
    if not all_pass:
        print("\nFUSION VERIFICATION FAILED — artifact NOT saved.")
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    artifact: Dict[str, Any] = {
        "model_name": "BaselineCNN",
        "source_checkpoint": str(ckpt_path),
        "source_seed": int(cfg.get("seed", 0)),
        "bn_fused": True,
        "dtype": "float32",
        "input_shape": [1, 28, 28],  # [C, H, W]
        "state_dict": fused.state_dict(),
        "original_test_accuracy": float(orig_acc),
        "fused_test_accuracy": float(fused_acc),
        "max_abs_error": float(compare["max_logits_abs_err"]),
        "prediction_agreement": float(compare["agreement_rate"]),
    }
    torch.save(artifact, output_path)
    print(f"\nFused FP32 model saved to: {output_path}")

    # ------------------------------------------------- reload round-trip
    fused_loaded, ckpt_loaded = load_bn_fused_model(
        str(output_path), lambda: _get_model("BaselineCNN"), device
    )
    with torch.no_grad():
        out_saved = fused_loaded(fixed_x.clone())
        out_pre = fused(fixed_x.clone())
    roundtrip_ok = bool(torch.equal(out_saved, out_pre))
    shape_ok = list(out_saved.shape) == [FIXED_SAMPLES, 10]
    print(f"reload model_name   : {ckpt_loaded.get('model_name')}")
    print(f"reload BN count     : {count_batchnorm(fused_loaded)}")
    print(f"reload output shape : {list(out_saved.shape)}  ({_yes(shape_ok)})")
    print(f"reload round-trip   : {_yes(roundtrip_ok)}")

    # ------------------------------------------------- final report
    print("\n" + "=" * 78)
    print("BN FUSION VERIFICATION SUMMARY")
    print("=" * 78)
    print(f"original test accuracy : {orig_acc:.6f}")
    print(f"fused    test accuracy : {fused_acc:.6f}")
    print(f"prediction agreement   : {compare['agreement_rate'] * 100:.4f}%")
    print(f"max logits |err|       : {compare['max_logits_abs_err']:.6e}")
    print(f"fused model saved at   : {output_path}")
    print(f"VERDICT                : {'ALL CHECKS PASSED' if all_pass and roundtrip_ok and shape_ok else 'FAILED'}")
    return 0 if (all_pass and roundtrip_ok and shape_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
