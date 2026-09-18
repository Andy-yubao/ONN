"""Train and evaluate the minimal 8x8 device-latency IF-SNN experiment."""

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
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from experiments.common.device_latency_encoder import DeviceLatencyEncoder
from .model import MinimalIFSNN


EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = EXPERIMENT_DIR.parents[2] / "model" / "data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
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
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(requested)


def make_loaders(data_dir: Path, batch_size: int, num_workers: int):
    transform = transforms.Compose(
        [transforms.Resize((8, 8), interpolation=transforms.InterpolationMode.BILINEAR), transforms.ToTensor()]
    )
    train_set = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    common = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    train_loader = DataLoader(train_set, shuffle=True, **common)
    test_loader = DataLoader(test_set, shuffle=False, **common)
    return train_loader, test_loader, test_set


def prepare_times(images: torch.Tensor, encoder: DeviceLatencyEncoder) -> torch.Tensor:
    return encoder.first_spike_times(images.flatten(1))


@torch.inference_mode()
def evaluate(model, loader, encoder, device) -> tuple[float, float]:
    model.eval()
    correct = 0
    total = 0
    started = time.perf_counter()
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(prepare_times(images, encoder), encoder.time_steps)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.numel()
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return 100.0 * correct / total, elapsed


def save_sanity_check(test_set, encoder: DeviceLatencyEncoder, results_dir: Path) -> dict:
    image, label = test_set[0]
    pixels = image.squeeze(0)
    times = encoder.first_spike_times(pixels).cpu()

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), constrained_layout=True)
    im0 = axes[0].imshow(pixels, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title(f"8x8 MNIST (label={label})")
    fig.colorbar(im0, ax=axes[0], fraction=0.046)
    masked = np.ma.masked_where(times.numpy() < 0, times.numpy())
    im1 = axes[1].imshow(masked, cmap="viridis_r", vmin=0, vmax=encoder.time_steps - 1)
    axes[1].set_title("First-spike time (-1 = none)")
    for row in range(8):
        for col in range(8):
            axes[1].text(col, row, str(int(times[row, col])), ha="center", va="center", fontsize=6)
    fig.colorbar(im1, ax=axes[1], fraction=0.046)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    figure_path = results_dir / "first_spike_sanity.png"
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)

    firing = times >= 0
    fired_pixels = int(firing.sum())
    # A simple directional check: brighter firing pixels should have earlier times.
    correlation = None
    if fired_pixels >= 2:
        correlation = float(np.corrcoef(pixels[firing].numpy(), times[firing].numpy())[0, 1])
    sanity = {
        "sample_label": int(label),
        "pixels_8x8": pixels.tolist(),
        "first_spike_times_8x8": times.tolist(),
        "fired_pixels": fired_pixels,
        "intensity_time_correlation_firing_pixels": correlation,
        "expected_correlation_sign": "negative",
    }
    with (results_dir / "sanity_check.json").open("w", encoding="utf-8") as handle:
        json.dump(sanity, handle, ensure_ascii=False, indent=2)
    return sanity


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = select_device(args.device)
    results_dir = EXPERIMENT_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    encoder = DeviceLatencyEncoder()
    train_loader, test_loader, test_set = make_loaders(args.data_dir, args.batch_size, args.num_workers)
    model = MinimalIFSNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    print(f"device={device} train={len(train_loader.dataset)} test={len(test_loader.dataset)}")
    print(f"T={encoder.time_steps} parameters={parameter_count}")
    history = []
    best_accuracy = 0.0
    training_started = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        loss_sum = 0.0
        sample_count = 0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(prepare_times(images, encoder), encoder.time_steps)
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
        history.append(
            {"epoch": epoch, "train_loss": average_loss, "test_accuracy": accuracy, "elapsed_seconds": epoch_elapsed}
        )
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), results_dir / "best_model.pt")
        print(
            f"epoch={epoch} train_loss={average_loss:.4f} "
            f"test_accuracy={accuracy:.2f}% elapsed={epoch_elapsed:.2f}s"
        )

    total_training_time = time.perf_counter() - training_started
    final_accuracy = history[-1]["test_accuracy"]
    model.load_state_dict(torch.load(results_dir / "best_model.pt", map_location=device, weights_only=True))

    # One warm-up pass before the deliberately lightweight full-test inference timing.
    warm_images, _ = next(iter(test_loader))
    with torch.inference_mode():
        model(prepare_times(warm_images.to(device), encoder), encoder.time_steps)
    if device.type == "cuda":
        torch.cuda.synchronize()
    best_accuracy_check, inference_seconds = evaluate(model, test_loader, encoder, device)
    sanity = save_sanity_check(test_set, encoder, results_dir)
    model_file_size = (results_dir / "best_model.pt").stat().st_size

    metrics = {
        "final_test_accuracy_percent": final_accuracy,
        "best_test_accuracy_percent": best_accuracy,
        "best_accuracy_recheck_percent": best_accuracy_check,
        "total_training_time_seconds": total_training_time,
        "average_time_per_epoch_seconds": total_training_time / args.epochs,
        "full_test_inference_time_seconds": inference_seconds,
        "average_inference_time_per_sample_ms": inference_seconds / len(test_loader.dataset) * 1000.0,
        "total_parameters": parameter_count,
        "model_file_size_bytes": model_file_size,
        "device": str(device),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "network": "64 -> 128 IF -> 10 non-leaky membrane readout",
        "time_steps": encoder.time_steps,
        "linear_dimensions": [[64, 128], [128, 10]],
        "dense_weight_accumulates_per_time_step": 64 * 128 + 128 * 10,
        "encoder": {
            "g0": encoder.g0,
            "alpha": encoder.alpha,
            "tau": encoder.tau,
            "g_threshold": encoder.g_threshold,
            "minimum_firing_intensity": encoder.minimum_firing_intensity,
        },
        "sanity_check": sanity,
        "history": history,
    }
    with (results_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
    with (results_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    print("\nResult summary")
    print(json.dumps({key: value for key, value in metrics.items() if key not in {"history", "sanity_check"}}, indent=2))
    print(f"sanity_correlation={sanity['intensity_time_correlation_firing_pixels']:.4f} (expected < 0)")


if __name__ == "__main__":
    main()
