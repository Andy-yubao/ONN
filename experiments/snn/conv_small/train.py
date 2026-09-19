"""Train and evaluate the small two-layer IF-SNN on device-coded MNIST."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import subprocess
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision

from experiments.common.device_latency_encoder import (
    DeviceLatencyEncoder,
    quantile_boundaries_from_histogram,
)
from experiments.common.comparison_protocol import PROTOCOL_ID, make_loaders, set_seed
from .model import DeviceIFConvSmall, normalized_temporal_weights


EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = EXPERIMENT_DIR.parents[2] / "data"
FROZEN_T4_QUANTILE_030_THRESHOLD = 0.21026152308606838
FROZEN_T4_QUANTILE_030_BOUNDARIES = (0, 1, 3)
# Frozen once from the existing beta=1 checkpoint on the 55k training split.
# It is the mean L1->Conv2 plus L2->Linear valid-connection count per image.
EVENT_PROXY_NORMALIZATION = 20_876.389236363637


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--time-steps", type=int, choices=(24, 12, 8, 5, 4, 3), default=24,
                        help="Quantize original 24-step TTFS; never truncate the physical window")
    parser.add_argument("--time-mapping", choices=("linear", "quantile"), default="linear")
    parser.add_argument("--target-firing-ratio", choices=("baseline", "0.30"), default="baseline")
    parser.add_argument(
        "--input-preset",
        choices=("calibrate", "t4_quantile_030"),
        default="calibrate",
        help="Use the frozen selected input baseline instead of recalibrating it",
    )
    parser.add_argument("--readout-mode", choices=("final_membrane", "accumulated_membrane"),
                        default="final_membrane")
    parser.add_argument("--temporal-beta", type=float, choices=(1.0, 0.5, 0.25), default=1.0)
    parser.add_argument("--event-lambda", type=float, choices=(0.0, 0.01, 0.03, 0.10), default=0.0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(requested)


@torch.inference_mode()
def calibrate_input_encoder(
    train_dataset,
    *,
    batch_size: int,
    num_workers: int,
    target_firing_ratio: str,
    time_steps: int,
    time_mapping: str,
) -> tuple[DeviceLatencyEncoder, dict]:
    """Freeze threshold and temporal boundaries using only the training split."""
    calibration_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    pixels = torch.cat([images.flatten() for images, _ in calibration_loader])
    g0, alpha, tau, source_steps = 0.10, 0.90, 5.0, 24
    threshold_method = "existing fixed device threshold"
    if target_firing_ratio == "baseline":
        g_threshold = 0.35
    else:
        target_count = round(float(target_firing_ratio) * pixels.numel())
        values, counts = torch.unique(pixels, sorted=True, return_counts=True)
        firing_counts = pixels.numel() - counts.cumsum(0)
        candidate_index = int((firing_counts - target_count).abs().argmin().item())
        if candidate_index + 1 >= values.numel():
            raise RuntimeError("training pixels do not bracket the requested firing ratio")
        intensity_cut = float((values[candidate_index] + values[candidate_index + 1]).item() / 2.0)
        g_threshold = g0 + alpha * intensity_cut * (1.0 - math.exp(-source_steps / tau))
        threshold_method = (
            "direct training-pixel quantile: midpoint between adjacent resized intensities "
            "whose 24-step firing count is closest to 30%; no validation/test data"
        )

    source_encoder = DeviceLatencyEncoder(
        g0=g0,
        alpha=alpha,
        tau=tau,
        g_threshold=g_threshold,
        time_steps=source_steps,
    )
    source_times = source_encoder.first_spike_times(pixels)
    firing_mask = source_times >= 0
    latency_histogram = torch.bincount(
        source_times[firing_mask], minlength=source_steps
    ).to(torch.int64)
    boundaries = None
    if time_mapping == "quantile":
        boundaries = quantile_boundaries_from_histogram(latency_histogram, time_steps)
    encoder = DeviceLatencyEncoder(
        g0=g0,
        alpha=alpha,
        tau=tau,
        g_threshold=g_threshold,
        time_steps=source_steps,
        quantized_time_steps=time_steps,
        time_mapping=time_mapping,
        quantile_boundaries=boundaries,
    )
    mapped_times = encoder.first_spike_times(pixels)
    mapped_histogram = torch.bincount(
        mapped_times[mapped_times >= 0], minlength=time_steps
    ).to(torch.int64)
    sample_count = len(train_dataset)
    firing_count = int(firing_mask.sum().item())
    calibration = {
        "data_scope": "training split indices [0, 55000) only",
        "training_samples": sample_count,
        "training_pixels": pixels.numel(),
        "target_firing_ratio": target_firing_ratio,
        "actual_firing_ratio": firing_count / pixels.numel(),
        "input_spikes_per_image": firing_count / sample_count,
        "g_threshold": g_threshold,
        "threshold_method": threshold_method,
        "time_mapping": time_mapping,
        "quantile_boundaries_inclusive_source_latency": list(boundaries or ()),
        "source_latency_histogram": latency_histogram.tolist(),
        "mapped_training_histogram_total": mapped_histogram.tolist(),
        "mapped_training_histogram_per_image": [
            count / sample_count for count in mapped_histogram.tolist()
        ],
    }
    return encoder, calibration


def frozen_t4_quantile_030_encoder() -> tuple[DeviceLatencyEncoder, dict]:
    """Reuse the selected training-calibrated encoder without recalibration."""
    encoder = DeviceLatencyEncoder(
        g0=0.10,
        alpha=0.90,
        tau=5.0,
        g_threshold=FROZEN_T4_QUANTILE_030_THRESHOLD,
        time_steps=24,
        quantized_time_steps=4,
        time_mapping="quantile",
        quantile_boundaries=FROZEN_T4_QUANTILE_030_BOUNDARIES,
    )
    calibration = {
        "frozen": True,
        "recalibrated_for_this_run": False,
        "source_result": "results/input_encoding_joint/T4_quantile_030/run_config.json",
        "data_scope": "original calibration used training split indices [0, 55000) only",
        "training_samples": 55_000,
        "training_pixels": 3_520_000,
        "target_firing_ratio": "0.30",
        "actual_firing_ratio": 0.300865625,
        "input_spikes_per_image": 19.2554,
        "g_threshold": FROZEN_T4_QUANTILE_030_THRESHOLD,
        "threshold_method": "reused frozen training-set calibration; no recalibration",
        "time_mapping": "quantile",
        "quantile_boundaries_inclusive_source_latency": list(
            FROZEN_T4_QUANTILE_030_BOUNDARIES
        ),
        "mapped_training_histogram_per_image": [
            1.8678545454545454,
            8.4514,
            4.899781818181818,
            4.036363636363636,
        ],
    }
    return encoder, calibration


@torch.inference_mode()
def evaluate(
    model: DeviceIFConvSmall,
    loader: DataLoader,
    encoder: DeviceLatencyEncoder,
    device: torch.device,
    *,
    collect_stats: bool = False,
) -> tuple[float, float, dict[str, int]]:
    model.eval()
    correct = 0
    total = 0
    totals = {
        "effective_synaptic_additions": 0,
        "input_spike_events": 0,
        "layer1_spike_events": 0,
        "layer2_spike_events": 0,
    }
    input_histogram = torch.zeros(encoder.time_steps, dtype=torch.int64)
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        times = encoder.first_spike_times(images)
        result = model(times, encoder.time_steps, collect_stats=collect_stats)
        if collect_stats:
            logits, batch_stats = result
            for key in totals:
                totals[key] += batch_stats[key]
            valid_times = times[times >= 0].detach().cpu()
            input_histogram += torch.bincount(valid_times, minlength=encoder.time_steps)
        else:
            logits = result
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.numel()
    if device.type == "cuda":
        torch.cuda.synchronize()
    totals["input_spike_histogram"] = input_histogram.tolist()
    return 100.0 * correct / total, time.perf_counter() - started, totals


def save_curve(history: list[dict], path: Path) -> None:
    epochs = [row["epoch"] for row in history]
    fig, left = plt.subplots(figsize=(5.8, 3.6), constrained_layout=True)
    right = left.twinx()
    left.plot(epochs, [row["train_loss"] for row in history], "o-", color="tab:blue", label="loss")
    right.plot(
        epochs,
        [row["validation_accuracy"] for row in history],
        "s-",
        color="tab:orange",
        label="validation accuracy",
    )
    left.set(xlabel="Epoch", ylabel="Train loss")
    right.set_ylabel("Validation accuracy (%)")
    left.grid(alpha=0.25)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")
    set_seed(args.seed)
    device = select_device(args.device)
    default_results = EXPERIMENT_DIR / "results" / f"seed_{args.seed}"
    if args.time_steps != 24:
        default_results = EXPERIMENT_DIR / "results" / "time_quantization" / f"T{args.time_steps}_seed{args.seed}"
    results_dir = args.results_dir or default_results
    if results_dir.exists() and any(results_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing results: {results_dir}")
    results_dir.mkdir(parents=True, exist_ok=True)
    train_loader, validation_loader, test_loader = make_loaders(
        args.data_dir, args.batch_size, args.num_workers
    )
    if args.input_preset == "t4_quantile_030":
        encoder, calibration = frozen_t4_quantile_030_encoder()
        input_mapping = "quantile"
        target_firing_ratio = "0.30"
    else:
        encoder, calibration = calibrate_input_encoder(
            train_loader.dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            target_firing_ratio=args.target_firing_ratio,
            time_steps=args.time_steps,
            time_mapping=args.time_mapping,
        )
        input_mapping = args.time_mapping
        target_firing_ratio = args.target_firing_ratio
    model = DeviceIFConvSmall(
        readout_mode=args.readout_mode,
        temporal_beta=args.temporal_beta,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=0.0)
    criterion = nn.CrossEntropyLoss()
    source_paths = (Path(__file__), EXPERIMENT_DIR / "model.py",
                    EXPERIMENT_DIR.parents[1] / "common" / "device_latency_encoder.py",
                    EXPERIMENT_DIR.parents[1] / "common" / "comparison_protocol.py")
    run_config = {
        "protocol_id": PROTOCOL_ID, "seed": args.seed, "epochs": args.epochs,
        "batch_size": args.batch_size, "num_workers": args.num_workers,
        "time_steps": encoder.time_steps, "source_time_steps": encoder.source_time_steps,
        "input_preset": args.input_preset,
        "time_mapping": input_mapping,
        "target_firing_ratio": target_firing_ratio,
        "input_encoding_calibration": calibration,
        "readout_mode": args.readout_mode,
        "temporal_beta": args.temporal_beta,
        "normalized_temporal_weights": normalized_temporal_weights(
            encoder.time_steps, args.temporal_beta
        ).tolist(),
        "event_lambda": args.event_lambda,
        "event_proxy": {
            "formula": "(sum L1 spikes * valid Conv2 spatial fanout * 32 + sum L2 spikes * 10) / batch_size / normalization",
            "normalization": EVENT_PROXY_NORMALIZATION,
            "normalization_source": "frozen beta=1 baseline checkpoint mean on 55k training split only",
            "input_spikes_penalized": False,
        },
        "optimizer": "Adam", "learning_rate_schedule": [args.learning_rate, args.learning_rate * 0.3],
        "weight_decay": 0.0, "gradient_clip_norm": 1.0,
        "total_parameters": model.parameter_count(),
        "readout": ("final membrane: sum of output currents; equal time weights"
                    if args.readout_mode == "final_membrane"
                    else "accumulated membrane: A[T-1]/T; earlier output currents receive larger weights"),
        "source_sha256": {str(p.relative_to(EXPERIMENT_DIR.parents[2])).replace("\\", "/"):
                          hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
    }
    (results_dir / "run_config.json").write_text(json.dumps(run_config, indent=2) + "\n", encoding="utf-8")

    print(
        f"device={device} train={len(train_loader.dataset)} "
        f"validation={len(validation_loader.dataset)} test={len(test_loader.dataset)}"
    )
    print(f"input=[B,1,8,8] T={encoder.time_steps} parameters={model.parameter_count()}")
    print(
        f"mapping={input_mapping} target_firing_ratio={target_firing_ratio} "
        f"actual_train_ratio={calibration['actual_firing_ratio']:.6f} "
        f"g_threshold={encoder.g_threshold:.9f} boundaries={encoder.quantile_boundaries}"
    )
    print(
        f"temporal_beta={args.temporal_beta:g} "
        f"weights={run_config['normalized_temporal_weights']} "
        f"event_lambda={args.event_lambda:g} event_normalization={EVENT_PROXY_NORMALIZATION:.6f}"
    )
    history: list[dict] = []
    best_validation_accuracy = -1.0
    best_epoch = 0
    training_started = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        if epoch == 11:
            for group in optimizer.param_groups:
                group["lr"] = args.learning_rate * 0.3
        epoch_started = time.perf_counter()
        model.train()
        loss_sum = 0.0
        ce_loss_sum = 0.0
        event_regularizer_sum = 0.0
        sample_count = 0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            times = encoder.first_spike_times(images)
            optimizer.zero_grad(set_to_none=True)
            if args.event_lambda > 0:
                logits, raw_event_proxy = model(
                    times,
                    encoder.time_steps,
                    return_event_proxy=True,
                )
                event_regularizer = raw_event_proxy / EVENT_PROXY_NORMALIZATION
            else:
                logits = model(times, encoder.time_steps)
                event_regularizer = logits.new_zeros(())
            ce_loss = criterion(logits, labels)
            loss = ce_loss + args.event_lambda * event_regularizer
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            loss_sum += loss.item() * labels.numel()
            ce_loss_sum += ce_loss.item() * labels.numel()
            event_regularizer_sum += event_regularizer.item() * labels.numel()
            sample_count += labels.numel()

        if device.type == "cuda":
            torch.cuda.synchronize()
        validation_accuracy, _, _ = evaluate(
            model, validation_loader, encoder, device, collect_stats=False
        )
        epoch_elapsed = time.perf_counter() - epoch_started
        average_loss = loss_sum / sample_count
        average_ce_loss = ce_loss_sum / sample_count
        average_event_regularizer = event_regularizer_sum / sample_count
        learning_rate = optimizer.param_groups[0]["lr"]
        row = {
            "epoch": epoch,
            "train_loss": average_loss,
            "train_ce_loss": average_ce_loss,
            "train_normalized_event_regularizer": average_event_regularizer,
            "validation_accuracy": validation_accuracy,
            "learning_rate": learning_rate,
            "elapsed_seconds": epoch_elapsed,
        }
        history.append(row)
        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            best_epoch = epoch
            torch.save(model.state_dict(), results_dir / "best_model.pt")
        print(
            f"epoch={epoch} train_loss={average_loss:.4f} ce={average_ce_loss:.4f} "
            f"R_event={average_event_regularizer:.4f} "
            f"validation_accuracy={validation_accuracy:.2f}% "
            f"lr={learning_rate:.1e} elapsed={epoch_elapsed:.2f}s",
            flush=True,
        )

    total_training_time = time.perf_counter() - training_started
    model.load_state_dict(
        torch.load(results_dir / "best_model.pt", map_location=device, weights_only=True)
    )

    warm_images, _ = next(iter(validation_loader))
    with torch.inference_mode():
        model(encoder.first_spike_times(warm_images.to(device)), encoder.time_steps)
    if device.type == "cuda":
        torch.cuda.synchronize()
    test_accuracy, inference_seconds, workload = evaluate(
        model, test_loader, encoder, device, collect_stats=True
    )
    test_samples = len(test_loader.dataset)
    model_size = (results_dir / "best_model.pt").stat().st_size
    hidden_states_per_sample = model.hidden_state_count() * encoder.time_steps
    readout_state_count = 10 if args.readout_mode == "final_membrane" else 20
    readout_states_per_sample = readout_state_count * encoder.time_steps
    metrics = {
        "protocol_id": PROTOCOL_ID,
        "test_evaluations": 1,
        "checkpoint_selection": "maximum validation accuracy; earliest epoch on ties",
        "preprocessing": "PIL bilinear Resize(8,8), ToTensor; no normalization or augmentation",
        "split_indices": {"train": [0, 55000], "validation": [55000, 60000], "end_exclusive": True},
        "environment": {
            "python": platform.python_version(), "torch": torch.__version__,
            "torchvision": torchvision.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()),
        },
        "optimizer": "Adam",
        "num_workers": args.num_workers,
        "run_config": run_config,
        "test_based_tuning": False,
        "checkpoint_sha256": hashlib.sha256((results_dir / "best_model.pt").read_bytes()).hexdigest(),
        "final_test_accuracy_percent": test_accuracy,
        "best_validation_accuracy_percent": best_validation_accuracy,
        "best_epoch": best_epoch,
        "validation_split": {"train": 55_000, "validation": 5_000},
        "test_set_size": test_samples,
        "target_accuracy_percent": 95.0,
        "target_reached": test_accuracy >= 95.0,
        "epochs_run": args.epochs,
        "total_training_time_seconds": total_training_time,
        "average_time_per_epoch_seconds": total_training_time / args.epochs,
        "full_test_inference_time_seconds": inference_seconds,
        "average_inference_time_per_sample_ms": inference_seconds / test_samples * 1000.0,
        "inference_batch_size": args.batch_size,
        "total_parameters": model.parameter_count(),
        "model_file_size_bytes": model_size,
        "device": str(device),
        "batch_size": args.batch_size,
        "learning_rate_schedule": [args.learning_rate, args.learning_rate * 0.3],
        "weight_decay": 0.0,
        "gradient_clip_norm": 1.0,
        "seed": args.seed,
        "temporal_beta": args.temporal_beta,
        "normalized_temporal_weights": run_config["normalized_temporal_weights"],
        "event_regularization": {
            "lambda": args.event_lambda,
            "normalization": EVENT_PROXY_NORMALIZATION,
            "formula": run_config["event_proxy"]["formula"],
            "final_epoch_train_ce_loss": history[-1]["train_ce_loss"],
            "final_epoch_normalized_event_regularizer": history[-1][
                "train_normalized_event_regularizer"
            ],
            "final_epoch_total_loss": history[-1]["train_loss"],
        },
        "network": ("Conv1x16-IF-subtract-Conv16x32(stride2)-IF-subtract-FC512x10-final-membrane"
                    if args.readout_mode == "final_membrane"
                    else "Conv1x16-IF-subtract-Conv16x32(stride2)-IF-subtract-FC512x10-time-weighted-membrane"),
        "time_steps": encoder.time_steps,
        "threshold": model.threshold,
        "surrogate": "fast_sigmoid_slope5_training_only",
        "dense_equivalent_macs_per_time_step": model.dense_equivalent_macs_per_step(),
        "dense_equivalent_macs_all_time_steps": model.dense_equivalent_macs_per_step()
        * encoder.time_steps,
        "hidden_state_count": model.hidden_state_count(),
        "hidden_state_updates_per_sample": hidden_states_per_sample,
        "readout_state_count": readout_state_count,
        "readout_state_updates_per_sample": readout_states_per_sample,
        "workload_test_total": workload,
        "workload_test_per_sample": {
            key: value / test_samples for key, value in workload.items()
            if key != "input_spike_histogram"
        },
        "input_encoding": {
            "training_calibration": calibration,
            "test_input_spike_histogram_total": workload["input_spike_histogram"],
            "test_input_spike_histogram_per_image": [
                count / test_samples for count in workload["input_spike_histogram"]
            ],
            "test_input_spikes_per_image": workload["input_spike_events"] / test_samples,
            "test_actual_firing_ratio": workload["input_spike_events"] / test_samples / 64,
        },
        "if_activity": {
            "layer1": {"if_count": 1024,
                       "fire_per_if_per_image": workload["layer1_spike_events"] / test_samples / 1024,
                       "neuron_step_firing_rate": workload["layer1_spike_events"] / test_samples / 1024 / encoder.time_steps},
            "layer2": {"if_count": 512,
                       "fire_per_if_per_image": workload["layer2_spike_events"] / test_samples / 512,
                       "neuron_step_firing_rate": workload["layer2_spike_events"] / test_samples / 512 / encoder.time_steps},
        },
        "encoder": {
            "g0": encoder.g0,
            "alpha": encoder.alpha,
            "tau": encoder.tau,
            "g_threshold": encoder.g_threshold,
            "source_time_steps": encoder.source_time_steps,
            "quantized_time_steps": encoder.time_steps,
            "time_mapping": input_mapping,
            "quantile_boundaries_inclusive_source_latency": list(encoder.quantile_boundaries or ()),
        },
        "history": history,
    }
    with (results_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
    with (results_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    save_curve(history, results_dir / "training_curve.png")

    summary = {key: value for key, value in metrics.items() if key not in {"history", "encoder"}}
    print("\nResult summary")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
