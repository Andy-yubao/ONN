"""Train the compact 8x8 photoconductive latency-coded Conv-SNN."""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from experiments.common.device_latency_encoder import DeviceLatencyEncoder
from .model import ConvSNN


EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = EXPERIMENT_DIR.parents[2] / "model" / "data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--target-accuracy", type=float, default=95.0)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(requested)


def make_loaders(data_dir: Path, batch_size: int, num_workers: int):
    transform = transforms.Compose(
        [transforms.Resize((8, 8), interpolation=transforms.InterpolationMode.BILINEAR), transforms.ToTensor()]
    )
    train_set = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    common = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    return (
        DataLoader(train_set, shuffle=True, **common),
        DataLoader(test_set, shuffle=False, **common),
    )


@torch.inference_mode()
def evaluate(model, loader, encoder, device) -> tuple[float, float]:
    model.eval()
    correct = 0
    total = 0
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        times = encoder.first_spike_times(images)
        logits = model(times, encoder.time_steps)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.numel()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return 100.0 * correct / total, time.perf_counter() - started


def save_curve(history: list[dict], path: Path) -> None:
    epochs = [row["epoch"] for row in history]
    fig, left = plt.subplots(figsize=(5.8, 3.6), constrained_layout=True)
    right = left.twinx()
    left.plot(epochs, [row["train_loss"] for row in history], "o-", color="tab:blue", label="loss")
    right.plot(epochs, [row["test_accuracy"] for row in history], "s-", color="tab:orange", label="accuracy")
    left.set(xlabel="Epoch", ylabel="Train loss")
    right.set_ylabel("Test accuracy (%)")
    left.grid(alpha=0.25)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = select_device(args.device)
    results_dir = EXPERIMENT_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    train_loader, test_loader = make_loaders(args.data_dir, args.batch_size, args.num_workers)
    encoder = DeviceLatencyEncoder()
    model = ConvSNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    print(f"device={device} train={len(train_loader.dataset)} test={len(test_loader.dataset)}")
    print(f"input=[B,1,8,8] T={encoder.time_steps} parameters={parameter_count}")
    history: list[dict] = []
    best_accuracy = 0.0
    epochs_run = 0
    training_started = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
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
            optimizer.step()
            loss_sum += loss.item() * labels.numel()
            sample_count += labels.numel()

        if device.type == "cuda":
            torch.cuda.synchronize()
        accuracy, _ = evaluate(model, test_loader, encoder, device)
        epoch_elapsed = time.perf_counter() - epoch_started
        average_loss = loss_sum / sample_count
        epochs_run = epoch
        row = {
            "epoch": epoch,
            "train_loss": average_loss,
            "test_accuracy": accuracy,
            "elapsed_seconds": epoch_elapsed,
        }
        history.append(row)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), results_dir / "best_model.pt")
        print(
            f"epoch={epoch} train_loss={average_loss:.4f} "
            f"test_accuracy={accuracy:.2f}% elapsed={epoch_elapsed:.2f}s",
            flush=True,
        )
        if accuracy >= args.target_accuracy:
            print(f"target reached ({accuracy:.2f}% >= {args.target_accuracy:.2f}%); stopping", flush=True)
            break

    total_training_time = time.perf_counter() - training_started
    final_accuracy = history[-1]["test_accuracy"]
    model.load_state_dict(torch.load(results_dir / "best_model.pt", map_location=device, weights_only=True))

    warm_images, _ = next(iter(test_loader))
    with torch.inference_mode():
        model(encoder.first_spike_times(warm_images.to(device)), encoder.time_steps)
    if device.type == "cuda":
        torch.cuda.synchronize()
    best_accuracy_recheck, inference_seconds = evaluate(model, test_loader, encoder, device)
    model_size = (results_dir / "best_model.pt").stat().st_size

    metrics = {
        "final_test_accuracy_percent": final_accuracy,
        "best_test_accuracy_percent": best_accuracy,
        "best_accuracy_recheck_percent": best_accuracy_recheck,
        "target_accuracy_percent": args.target_accuracy,
        "target_reached": best_accuracy >= args.target_accuracy,
        "epochs_run": epochs_run,
        "total_training_time_seconds": total_training_time,
        "average_time_per_epoch_seconds": total_training_time / epochs_run,
        "full_test_inference_time_seconds": inference_seconds,
        "average_inference_time_per_sample_ms": inference_seconds / len(test_loader.dataset) * 1000.0,
        "total_parameters": parameter_count,
        "model_file_size_bytes": model_size,
        "device": str(device),
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "network": "Conv1x32-LIF(beta0.9)-Conv32x64(stride2)-LIF-FC1024x128-LIF-FC128x10-membrane",
        "time_steps": encoder.time_steps,
        "dense_equivalent_macs_per_time_step": model.dense_equivalent_macs_per_step(),
        "dense_equivalent_macs_all_time_steps": model.dense_equivalent_macs_per_step() * encoder.time_steps,
        "encoder": {
            "g0": encoder.g0,
            "alpha": encoder.alpha,
            "tau": encoder.tau,
            "g_threshold": encoder.g_threshold,
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
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
