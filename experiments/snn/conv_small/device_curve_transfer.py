"""Evaluate frozen SNN checkpoints after training-only device recalibration."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from experiments.common.comparison_protocol import make_loaders
from experiments.common.device_latency_encoder import (
    DeviceLatencyEncoder,
    quantile_boundaries_from_histogram,
)
from experiments.snn.conv_small.model import DeviceIFConvSmall
from experiments.snn.conv_small.train import evaluate, select_device


EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = EXPERIMENT_DIR.parents[2] / "data"
SOURCE_TIME_STEPS = 24
TARGET_TIME_STEPS = 4
TARGET_FIRING_RATIO = 0.30


@dataclass(frozen=True)
class DeviceVariant:
    name: str
    description: str
    g0: float
    alpha: float
    tau: float


ORIGINAL = DeviceVariant("original", "frozen reference device", 0.10, 0.90, 5.0)
VARIANTS = (
    DeviceVariant("faster_tau3", "faster response", 0.10, 0.90, 3.0),
    DeviceVariant("slower_tau8", "slower response", 0.10, 0.90, 8.0),
    DeviceVariant("shifted_gain", "higher baseline and lower gain", 0.16, 0.70, 5.0),
)
CHECKPOINTS = {
    7: EXPERIMENT_DIR / "results/final_three_seed/seed_7/best_model.pt",
    17: EXPERIMENT_DIR / "results/event_regularization/beta_0p5_lambda_0p10_seed17/best_model.pt",
    27: EXPERIMENT_DIR / "results/final_three_seed/seed_27/best_model.pt",
}
REFERENCE_METRICS = {
    7: EXPERIMENT_DIR / "results/final_three_seed/seed_7/metrics.json",
    17: EXPERIMENT_DIR / "results/event_regularization/beta_0p5_lambda_0p10_seed17/metrics.json",
    27: EXPERIMENT_DIR / "results/final_three_seed/seed_27/metrics.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--output",
        type=Path,
        default=EXPERIMENT_DIR / "results/device_curve_transfer/summary.json",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


@torch.inference_mode()
def training_pixels(train_dataset, *, batch_size: int, num_workers: int) -> torch.Tensor:
    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    return torch.cat([images.flatten() for images, _ in loader])


@torch.inference_mode()
def recalibrate_encoder(pixels: torch.Tensor, variant: DeviceVariant) -> tuple[DeviceLatencyEncoder, dict]:
    """Fit threshold and quantile cuts using training pixels only."""
    target_count = round(TARGET_FIRING_RATIO * pixels.numel())
    values, counts = torch.unique(pixels, sorted=True, return_counts=True)
    firing_counts = pixels.numel() - counts.cumsum(0)
    candidate_index = int((firing_counts - target_count).abs().argmin().item())
    if candidate_index + 1 >= values.numel():
        raise RuntimeError("training pixels do not bracket the requested firing ratio")
    intensity_cut = float((values[candidate_index] + values[candidate_index + 1]).item() / 2.0)
    finite_window_gain = 1.0 - math.exp(-SOURCE_TIME_STEPS / variant.tau)
    g_threshold = variant.g0 + variant.alpha * intensity_cut * finite_window_gain

    source_encoder = DeviceLatencyEncoder(
        g0=variant.g0,
        alpha=variant.alpha,
        tau=variant.tau,
        g_threshold=g_threshold,
        time_steps=SOURCE_TIME_STEPS,
    )
    source_times = source_encoder.first_spike_times(pixels)
    firing_mask = source_times >= 0
    latency_histogram = torch.bincount(
        source_times[firing_mask], minlength=SOURCE_TIME_STEPS
    ).to(torch.int64)
    boundaries = quantile_boundaries_from_histogram(latency_histogram, TARGET_TIME_STEPS)
    encoder = DeviceLatencyEncoder(
        g0=variant.g0,
        alpha=variant.alpha,
        tau=variant.tau,
        g_threshold=g_threshold,
        time_steps=SOURCE_TIME_STEPS,
        quantized_time_steps=TARGET_TIME_STEPS,
        time_mapping="quantile",
        quantile_boundaries=boundaries,
    )
    mapped_times = encoder.first_spike_times(pixels)
    mapped_histogram = torch.bincount(
        mapped_times[mapped_times >= 0], minlength=TARGET_TIME_STEPS
    ).to(torch.int64)
    firing_count = int(firing_mask.sum().item())
    sample_count = pixels.numel() // 64
    return encoder, {
        "data_scope": "training split indices [0, 55000) only",
        "training_samples": sample_count,
        "training_pixels": pixels.numel(),
        "target_firing_ratio": TARGET_FIRING_RATIO,
        "actual_firing_ratio": firing_count / pixels.numel(),
        "input_spikes_per_image": firing_count / sample_count,
        "intensity_cut": intensity_cut,
        "g_threshold": g_threshold,
        "quantile_boundaries_inclusive_source_latency": list(boundaries),
        "source_latency_histogram": latency_histogram.tolist(),
        "mapped_training_histogram_per_image": [
            count / sample_count for count in mapped_histogram.tolist()
        ],
    }


@torch.inference_mode()
def check_device_curve(variant: DeviceVariant, encoder: DeviceLatencyEncoder) -> dict:
    """Verify conductance and crossing latency have the required monotonic order."""
    intensities = torch.linspace(0.0, 1.0, 1001)
    conductance = torch.stack(
        [encoder.conductance(intensities, step) for step in range(SOURCE_TIME_STEPS)]
    )
    conductance_monotonic_in_intensity = bool((conductance[:, 1:] >= conductance[:, :-1]).all())
    conductance_monotonic_in_time = bool((conductance[1:] >= conductance[:-1]).all())
    source_encoder = DeviceLatencyEncoder(
        g0=variant.g0,
        alpha=variant.alpha,
        tau=variant.tau,
        g_threshold=encoder.g_threshold,
        time_steps=SOURCE_TIME_STEPS,
    )
    crossing_times = source_encoder.first_spike_times(intensities)
    valid_times = crossing_times[crossing_times >= 0]
    crossing_earlier_at_higher_intensity = bool(
        valid_times.numel() > 0 and (valid_times[1:] <= valid_times[:-1]).all()
    )
    valid = (
        conductance_monotonic_in_intensity
        and conductance_monotonic_in_time
        and crossing_earlier_at_higher_intensity
    )
    if not valid:
        raise RuntimeError(f"device monotonicity check failed for {variant.name}")
    return {
        "valid": valid,
        "conductance_monotonic_in_intensity": conductance_monotonic_in_intensity,
        "conductance_monotonic_in_time": conductance_monotonic_in_time,
        "crossing_earlier_or_equal_at_higher_intensity": crossing_earlier_at_higher_intensity,
        "firing_grid_points": int(valid_times.numel()),
    }


def per_sample_workload(workload: dict, sample_count: int) -> dict:
    return {
        key: value / sample_count
        for key, value in workload.items()
        if key != "input_spike_histogram"
    }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing results: {args.output}")
    device = select_device(args.device)
    train_loader, _, test_loader = make_loaders(args.data_dir, args.batch_size, args.num_workers)
    pixels = training_pixels(
        train_loader.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    missing = [str(path) for path in (*CHECKPOINTS.values(), *REFERENCE_METRICS.values()) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing frozen artifacts: {missing}")

    reference_results = {}
    for seed, path in REFERENCE_METRICS.items():
        metrics = json.loads(path.read_text(encoding="utf-8"))
        reference_results[str(seed)] = {
            "test_accuracy_percent": metrics["final_test_accuracy_percent"],
            "effective_synaptic_additions_per_image": metrics["workload_test_per_sample"][
                "effective_synaptic_additions"
            ],
            "source": str(path.relative_to(EXPERIMENT_DIR)).replace("\\", "/"),
            "reused_existing_final_test": True,
        }

    calibrated_variants = []
    for variant in VARIANTS:
        encoder, calibration = recalibrate_encoder(pixels, variant)
        curve_check = check_device_curve(variant, encoder)
        calibrated_variants.append((variant, encoder, calibration, curve_check))
        print(
            f"checked device={variant.name} threshold={encoder.g_threshold:.9f} "
            f"ratio={calibration['actual_firing_ratio']:.6f} "
            f"boundaries={encoder.quantile_boundaries}",
            flush=True,
        )

    results = {
        "experiment": "frozen SNN device-curve transfer with encoder-only recalibration",
        "weights_retrained_or_finetuned": False,
        "calibration_data": "training split only; validation and test excluded",
        "new_test_inferences": len(VARIANTS) * len(CHECKPOINTS),
        "original_reference": {
            "device": asdict(ORIGINAL),
            "results": reference_results,
        },
        "variants": [],
    }
    original_mean = sum(row["test_accuracy_percent"] for row in reference_results.values()) / 3

    for variant, encoder, calibration, curve_check in calibrated_variants:
        evaluations = {}
        for seed, checkpoint in CHECKPOINTS.items():
            model = DeviceIFConvSmall(
                readout_mode="accumulated_membrane",
                temporal_beta=0.5,
            ).to(device)
            model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
            accuracy, inference_seconds, workload = evaluate(
                model, test_loader, encoder, device, collect_stats=True
            )
            evaluations[str(seed)] = {
                "test_accuracy_percent": accuracy,
                "inference_seconds": inference_seconds,
                "workload_test_per_sample": per_sample_workload(workload, len(test_loader.dataset)),
                "test_input_spike_histogram_total": workload["input_spike_histogram"],
                "checkpoint": str(checkpoint.relative_to(EXPERIMENT_DIR)).replace("\\", "/"),
                "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            }
            print(f"device={variant.name} seed={seed} test={accuracy:.2f}%", flush=True)
        accuracy_mean = sum(row["test_accuracy_percent"] for row in evaluations.values()) / 3
        results["variants"].append(
            {
                "device": asdict(variant),
                "curve_check": curve_check,
                "calibration": calibration,
                "evaluations": evaluations,
                "test_accuracy_mean_percent": accuracy_mean,
                "accuracy_drop_vs_original_mean_pct_points": original_mean - accuracy_mean,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
