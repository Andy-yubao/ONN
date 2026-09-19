"""Train and evaluate the small two-layer IF-SNN on device-coded MNIST."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision

from experiments.common.device_latency_encoder import DeviceLatencyEncoder
from experiments.common.comparison_protocol import PROTOCOL_ID, make_loaders, set_seed
from .model import DeviceIFConvSmall


EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = EXPERIMENT_DIR.parents[2] / "data"


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
    parser.add_argument("--readout-mode", choices=("final_membrane", "accumulated_membrane"),
                        default="final_membrane")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(requested)


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
        else:
            logits = result
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.numel()
    if device.type == "cuda":
        torch.cuda.synchronize()
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
    encoder = DeviceLatencyEncoder(time_steps=24, quantized_time_steps=args.time_steps)
    model = DeviceIFConvSmall(readout_mode=args.readout_mode).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=0.0)
    criterion = nn.CrossEntropyLoss()
    source_paths = (Path(__file__), EXPERIMENT_DIR / "model.py",
                    EXPERIMENT_DIR.parents[1] / "common" / "device_latency_encoder.py",
                    EXPERIMENT_DIR.parents[1] / "common" / "comparison_protocol.py")
    run_config = {
        "protocol_id": PROTOCOL_ID, "seed": args.seed, "epochs": args.epochs,
        "batch_size": args.batch_size, "num_workers": args.num_workers,
        "time_steps": encoder.time_steps, "source_time_steps": encoder.source_time_steps,
        "time_mapping": "floor(old_t * target_T / source_T); -1 remains -1",
        "readout_mode": args.readout_mode,
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
        sample_count = 0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            times = encoder.first_spike_times(images)
            optimizer.zero_grad(set_to_none=True)
            logits = model(times, encoder.time_steps)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            loss_sum += loss.item() * labels.numel()
            sample_count += labels.numel()

        if device.type == "cuda":
            torch.cuda.synchronize()
        validation_accuracy, _, _ = evaluate(
            model, validation_loader, encoder, device, collect_stats=False
        )
        epoch_elapsed = time.perf_counter() - epoch_started
        average_loss = loss_sum / sample_count
        learning_rate = optimizer.param_groups[0]["lr"]
        row = {
            "epoch": epoch,
            "train_loss": average_loss,
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
            f"epoch={epoch} train_loss={average_loss:.4f} "
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
    readout_states_per_sample = 10 * encoder.time_steps
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
        "network": "Conv1x16-IF-subtract-Conv16x32(stride2)-IF-subtract-FC512x10-final-membrane",
        "time_steps": encoder.time_steps,
        "threshold": model.threshold,
        "surrogate": "fast_sigmoid_slope5_training_only",
        "dense_equivalent_macs_per_time_step": model.dense_equivalent_macs_per_step(),
        "dense_equivalent_macs_all_time_steps": model.dense_equivalent_macs_per_step()
        * encoder.time_steps,
        "hidden_state_count": model.hidden_state_count(),
        "hidden_state_updates_per_sample": hidden_states_per_sample,
        "readout_state_count": 10,
        "readout_state_updates_per_sample": readout_states_per_sample,
        "workload_test_total": workload,
        "workload_test_per_sample": {
            key: value / test_samples for key, value in workload.items()
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
            "time_mapping": run_config["time_mapping"],
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
