"""Evaluate the frozen BaselineCNN under reduced effective input resolutions.

The experiment keeps the BN-fused checkpoint and scheme-A quantisation
configuration frozen.  Only the input preprocessing changes:

    grayscale image -> ToTensor [0, 1] -> area downsample -> nearest upsample
    -> MNIST Normalize -> frozen model

Run from the repository root::

    python model/evaluate_resolution_compatibility.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "onn-matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent))

from onn_model.data import MNIST_MEAN, MNIST_STD, TEST_SIZE, get_mnist_dataset
from onn_model.int8_ptq import build_weight_config, finalize_weight_config, w8a8_forward
from onn_model.int8_reference import (
    Int8Reference,
    WEIGHT_LAYER_ACCESSORS,
    _resolve_module,
    quantize_weight_with_scale,
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
DEFAULT_OUTPUT_DIR = REPO_ROOT / "experiments" / "input_resolution" / "e1_frozen_baseline"

CONDITION_EFFECTIVE_SIZE: Mapping[str, int] = {
    "native_28": 28,
    "down8_up28": 8,
    "down4_up28": 4,
}
INFERENCE_PATHS = ("fp32", "fake_quant", "integer_reference")


class ResolutionCompatibilityTransform:
    """Reduce effective spatial resolution and restore a 28x28 tensor.

    The input must already be a floating-point tensor in ``[0, 1]``.  For a
    reduced condition, area interpolation performs the downsampling and
    nearest-neighbour interpolation restores the model's fixed input shape.
    """

    def __init__(self, condition: str) -> None:
        if condition not in CONDITION_EFFECTIVE_SIZE:
            choices = ", ".join(CONDITION_EFFECTIVE_SIZE)
            raise ValueError(f"unknown condition {condition!r}; expected one of: {choices}")
        self.condition = condition
        self.effective_size = CONDITION_EFFECTIVE_SIZE[condition]

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        if not isinstance(image, torch.Tensor):
            raise TypeError("resolution transform expects a torch.Tensor after ToTensor")
        if image.ndim != 3 or image.shape[-2:] != (28, 28):
            raise ValueError(f"expected [C, 28, 28], got {tuple(image.shape)}")
        if not image.is_floating_point():
            raise TypeError("resolution transform expects a floating-point tensor in [0, 1]")

        if self.effective_size == 28:
            return image.clone()

        batched = image.unsqueeze(0)
        reduced = F.interpolate(
            batched,
            size=(self.effective_size, self.effective_size),
            mode="area",
        )
        restored = F.interpolate(reduced, size=(28, 28), mode="nearest")
        return restored.squeeze(0)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(condition={self.condition!r}, "
            f"effective_size={self.effective_size})"
        )


def build_mnist_resolution_transform(condition: str) -> transforms.Compose:
    """Build the required ToTensor -> resolution -> Normalize pipeline."""

    return transforms.Compose(
        [
            transforms.ToTensor(),
            ResolutionCompatibilityTransform(condition),
            transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
        ]
    )


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_frozen_weight_config(model: torch.nn.Module, config: Mapping[str, Any]) -> Dict[str, Any]:
    """Build fake-quant tensors with the exact frozen scheme-A scales."""

    if config.get("scheme") != "scheme_a":
        raise ValueError(f"expected frozen scheme_a config, got {config.get('scheme')!r}")
    quant = config.get("weight_quantization", {})
    if quant.get("conv_method") != "per-tensor" or quant.get("fc_method") != "per-tensor":
        raise ValueError("scheme A must use per-tensor convolution and FC weight quantisation")

    weight_cfg = build_weight_config(model, "per-tensor")
    frozen_scales = config["weight_scale"]
    for layer, accessor in WEIGHT_LAYER_ACCESSORS.items():
        frozen_scale = float(frozen_scales[layer])
        module = _resolve_module(model, accessor)
        weight_cfg[layer]["weight_scale"] = frozen_scale
        weight_cfg[layer]["q"] = quantize_weight_with_scale(
            module.weight.detach(), frozen_scale
        )
    finalize_weight_config(weight_cfg, config["activation_scale"])
    return weight_cfg


def _confusion_matrix(labels: torch.Tensor, predictions: torch.Tensor) -> torch.Tensor:
    encoded = labels.to(torch.int64) * 10 + predictions.to(torch.int64)
    return torch.bincount(encoded, minlength=100).reshape(10, 10)


def _path_metrics(labels: torch.Tensor, predictions: torch.Tensor) -> Dict[str, Any]:
    matrix = _confusion_matrix(labels, predictions)
    class_totals = matrix.sum(dim=1)
    class_correct = matrix.diag()
    per_class = []
    for digit in range(10):
        total = int(class_totals[digit].item())
        correct = int(class_correct[digit].item())
        per_class.append(
            {
                "class": digit,
                "sample_count": total,
                "correct": correct,
                "accuracy": correct / total if total else None,
            }
        )
    correct = int((labels == predictions).sum().item())
    return {
        "correct": correct,
        "accuracy": correct / labels.numel(),
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
    }


@torch.no_grad()
def evaluate_condition(
    condition: str,
    model: torch.nn.Module,
    reference: Int8Reference,
    weight_cfg: Mapping[str, Any],
    activation_scales: Mapping[str, float],
    data_root: Path,
    batch_size: int,
    num_workers: int,
    download: bool,
) -> tuple[Dict[str, Any], Dict[str, torch.Tensor]]:
    """Evaluate all three frozen inference paths for one input condition."""

    dataset = get_mnist_dataset(
        root=str(data_root), train=False, download=download, add_augment=False
    )
    dataset.transform = build_mnist_resolution_transform(condition)
    if len(dataset) != TEST_SIZE:
        raise RuntimeError(f"expected official {TEST_SIZE}-sample MNIST test set, got {len(dataset)}")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    collected: Dict[str, list[torch.Tensor]] = {
        "labels": [],
        **{path: [] for path in INFERENCE_PATHS},
    }

    for batch_index, (images, labels) in enumerate(loader, start=1):
        fp32_logits = model(images)
        fake_logits, _ = w8a8_forward(
            model, images, dict(activation_scales), dict(weight_cfg), return_trace=False
        )
        integer_predictions, _, _ = reference.infer(images, return_trace=False)

        collected["labels"].append(labels.cpu())
        collected["fp32"].append(fp32_logits.argmax(dim=1).cpu())
        collected["fake_quant"].append(fake_logits.argmax(dim=1).cpu())
        collected["integer_reference"].append(integer_predictions.cpu())

        if batch_index % 20 == 0 or batch_index == len(loader):
            processed = min(batch_index * batch_size, len(dataset))
            print(f"  {condition}: {processed:5d}/{len(dataset)}")

    tensors = {name: torch.cat(parts) for name, parts in collected.items()}
    labels = tensors["labels"]
    total = int(labels.numel())

    def agreement(left: str, right: str) -> float:
        return int((tensors[left] == tensors[right]).sum().item()) / total

    metrics = {
        "sample_count": total,
        "paths": {
            path: _path_metrics(labels, tensors[path]) for path in INFERENCE_PATHS
        },
        "agreement": {
            "integer_vs_fp32": agreement("integer_reference", "fp32"),
            "integer_vs_fake_quant": agreement("integer_reference", "fake_quant"),
            "fake_quant_vs_fp32": agreement("fake_quant", "fp32"),
        },
    }
    return metrics, tensors


def _prediction_change_summary(
    predictions: Mapping[str, Mapping[str, torch.Tensor]],
) -> Dict[str, Any]:
    condition_names = list(CONDITION_EFFECTIVE_SIZE)
    summary: Dict[str, Any] = {}
    for path in INFERENCE_PATHS:
        pairwise: Dict[str, int] = {}
        for left_index, left in enumerate(condition_names):
            for right in condition_names[left_index + 1 :]:
                key = f"{left}_vs_{right}"
                pairwise[key] = int(
                    (predictions[left][path] != predictions[right][path]).sum().item()
                )
        stacked = torch.stack([predictions[name][path] for name in condition_names])
        any_change = (stacked != stacked[0:1]).any(dim=0)
        summary[path] = {
            "pairwise_changed_sample_count": pairwise,
            "changed_in_any_condition": int(any_change.sum().item()),
        }
    return summary


def _add_native_drops(results: Dict[str, Any]) -> None:
    native = results["conditions"]["native_28"]
    for condition_result in results["conditions"].values():
        condition_result["accuracy_drop_vs_native_pp"] = {}
        for path in INFERENCE_PATHS:
            native_accuracy = native["paths"][path]["accuracy"]
            accuracy = condition_result["paths"][path]["accuracy"]
            condition_result["accuracy_drop_vs_native_pp"][path] = round(
                (native_accuracy - accuracy) * 100.0, 10
            )


def _write_csv(results: Mapping[str, Any], path: Path) -> None:
    fields = [
        "condition",
        "effective_size",
        "sample_count",
        "fp32_accuracy",
        "fake_quant_accuracy",
        "integer_reference_accuracy",
        "fp32_drop_vs_native_pp",
        "fake_quant_drop_vs_native_pp",
        "integer_reference_drop_vs_native_pp",
        "integer_vs_fp32_agreement",
        "integer_vs_fake_quant_agreement",
        "fake_quant_vs_fp32_agreement",
        "integer_changed_vs_native_count",
    ]
    native_changes = results["prediction_changes"]["integer_reference"][
        "pairwise_changed_sample_count"
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for condition, metrics in results["conditions"].items():
            change_key = f"native_28_vs_{condition}"
            writer.writerow(
                {
                    "condition": condition,
                    "effective_size": CONDITION_EFFECTIVE_SIZE[condition],
                    "sample_count": metrics["sample_count"],
                    "fp32_accuracy": f"{metrics['paths']['fp32']['accuracy']:.6f}",
                    "fake_quant_accuracy": f"{metrics['paths']['fake_quant']['accuracy']:.6f}",
                    "integer_reference_accuracy": (
                        f"{metrics['paths']['integer_reference']['accuracy']:.6f}"
                    ),
                    "fp32_drop_vs_native_pp": (
                        f"{metrics['accuracy_drop_vs_native_pp']['fp32']:.3f}"
                    ),
                    "fake_quant_drop_vs_native_pp": (
                        f"{metrics['accuracy_drop_vs_native_pp']['fake_quant']:.3f}"
                    ),
                    "integer_reference_drop_vs_native_pp": (
                        f"{metrics['accuracy_drop_vs_native_pp']['integer_reference']:.3f}"
                    ),
                    "integer_vs_fp32_agreement": (
                        f"{metrics['agreement']['integer_vs_fp32']:.6f}"
                    ),
                    "integer_vs_fake_quant_agreement": (
                        f"{metrics['agreement']['integer_vs_fake_quant']:.6f}"
                    ),
                    "fake_quant_vs_fp32_agreement": (
                        f"{metrics['agreement']['fake_quant_vs_fp32']:.6f}"
                    ),
                    "integer_changed_vs_native_count": 0
                    if condition == "native_28"
                    else native_changes[change_key],
                }
            )


def _write_sample_comparison(data_root: Path, path: Path, download: bool) -> None:
    dataset = get_mnist_dataset(
        root=str(data_root), train=False, download=download, add_augment=False
    )
    first_index: Dict[int, int] = {}
    for index, label in enumerate(dataset.targets.tolist()):
        first_index.setdefault(int(label), index)
        if len(first_index) == 10:
            break

    fig, axes = plt.subplots(10, 3, figsize=(6.6, 17.0), constrained_layout=True)
    condition_names = list(CONDITION_EFFECTIVE_SIZE)
    for row, digit in enumerate(range(10)):
        image = transforms.ToTensor()(dataset.data[first_index[digit]].numpy())
        for column, condition in enumerate(condition_names):
            displayed = ResolutionCompatibilityTransform(condition)(image)
            axis = axes[row, column]
            axis.imshow(displayed.squeeze(0).numpy(), cmap="gray", vmin=0.0, vmax=1.0)
            axis.set_xticks([])
            axis.set_yticks([])
            if row == 0:
                axis.set_title(condition)
            if column == 0:
                axis.set_ylabel(f"digit {digit}")
    fig.suptitle("MNIST effective-resolution comparison", fontsize=14)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate frozen BaselineCNN input-resolution compatibility"
    )
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--no-download", action="store_true", help="fail instead of downloading MNIST"
    )
    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.num_workers < 0:
        parser.error("--num-workers must be non-negative")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    artifact = args.artifact.resolve()
    config_path = args.config.resolve()
    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = json.loads(config_path.read_text(encoding="utf-8"))
    model, checkpoint = load_bn_fused_model(str(artifact), BaselineCNN, torch.device("cpu"))
    model.eval()
    reference = Int8Reference(model, config)
    weight_cfg = _build_frozen_weight_config(model, config)

    print("Frozen BaselineCNN resolution compatibility evaluation")
    print(f"artifact : {_relative(artifact)}")
    print(f"config   : {_relative(config_path)}")
    print(f"data     : {_relative(data_root)}")

    conditions: Dict[str, Any] = {}
    predictions: Dict[str, Dict[str, torch.Tensor]] = {}
    canonical_labels: torch.Tensor | None = None
    for condition in CONDITION_EFFECTIVE_SIZE:
        print(f"\n[{condition}]")
        metrics, tensors = evaluate_condition(
            condition=condition,
            model=model,
            reference=reference,
            weight_cfg=weight_cfg,
            activation_scales=config["activation_scale"],
            data_root=data_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            download=not args.no_download,
        )
        if canonical_labels is None:
            canonical_labels = tensors["labels"]
        elif not torch.equal(canonical_labels, tensors["labels"]):
            raise RuntimeError("MNIST labels/order changed between conditions")
        conditions[condition] = metrics
        predictions[condition] = {path: tensors[path] for path in INFERENCE_PATHS}
        accuracy = metrics["paths"]
        print(
            "  accuracy: "
            f"fp32={accuracy['fp32']['accuracy']:.4%}, "
            f"fake={accuracy['fake_quant']['accuracy']:.4%}, "
            f"integer={accuracy['integer_reference']['accuracy']:.4%}"
        )

    results: Dict[str, Any] = {
        "schema_version": 1,
        "experiment": "e1_frozen_baseline_resolution_compatibility",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(["python", "model/evaluate_resolution_compatibility.py", *sys.argv[1:]]),
        "model": {
            "name": "BaselineCNN",
            "checkpoint_model_name": checkpoint.get("model_name"),
            "bn_fused_artifact": _relative(artifact),
            "bn_fused_artifact_sha256": _sha256(artifact),
        },
        "quantization": {
            "scheme": config["scheme"],
            "config": _relative(config_path),
            "config_sha256": _sha256(config_path),
            "weight_scale": config["weight_scale"],
            "activation_scale": config["activation_scale"],
            "rounding": config.get("rounding"),
            "accumulator": config.get("accumulator"),
        },
        "dataset": {
            "name": "MNIST official test split",
            "sample_count": TEST_SIZE,
            "data_root": _relative(data_root),
            "shuffle": False,
        },
        "preprocessing": {
            "order": [
                "grayscale_image",
                "ToTensor_[0,1]",
                "area_downsample",
                "nearest_upsample_to_28x28",
                "Normalize_mean_0.1307_std_0.3081",
            ],
            "conditions": dict(CONDITION_EFFECTIVE_SIZE),
        },
        "conditions": conditions,
        "prediction_changes": _prediction_change_summary(predictions),
    }
    _add_native_drops(results)

    json_path = output_dir / "results.json"
    csv_path = output_dir / "results.csv"
    image_path = output_dir / "sample_comparison.png"
    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_csv(results, csv_path)
    _write_sample_comparison(data_root, image_path, download=not args.no_download)

    print(f"\nresults: {_relative(json_path)}")
    print(f"table  : {_relative(csv_path)}")
    print(f"figure : {_relative(image_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
