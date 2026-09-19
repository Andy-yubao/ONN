"""Frozen-budget matched CNN training with validation-only selection."""

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
import torchvision
from torch import nn

from experiments.common.comparison_protocol import PROTOCOL_ID, make_loaders, set_seed
from .model import MatchedConvSmall

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_DIR = EXPERIMENT_DIR.parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=REPO_DIR / "data")
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    synchronize(device)
    started = time.perf_counter()
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.numel()
    synchronize(device)
    return 100.0 * correct / total, time.perf_counter() - started


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def save_history(path, history):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)


def save_curve(path, history):
    epochs = [row["epoch"] for row in history]
    fig, left = plt.subplots(figsize=(5.8, 3.6), constrained_layout=True)
    right = left.twinx()
    left.plot(epochs, [row["train_loss"] for row in history], "o-", color="tab:blue")
    right.plot(epochs, [row["validation_accuracy"] for row in history], "s-", color="tab:orange")
    left.set(xlabel="Epoch", ylabel="Train loss")
    right.set_ylabel("Validation accuracy (%)")
    left.grid(alpha=0.25)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.learning_rate <= 0 or args.num_workers < 0:
        raise ValueError("Invalid training budget or loader configuration")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    results_dir = args.results_dir or EXPERIMENT_DIR / "results" / f"seed_{args.seed}"
    if results_dir.exists() and any(results_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing results: {results_dir}")
    results_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    train_loader, validation_loader, test_loader = make_loaders(args.data_dir, args.batch_size, args.num_workers)
    model = MatchedConvSmall().to(device)
    config = {
        "protocol_id": PROTOCOL_ID, "seed": args.seed, "epochs_planned": args.epochs,
        "batch_size": args.batch_size, "num_workers": args.num_workers,
        "optimizer": "Adam", "weight_decay": 0.0, "gradient_clip_norm": 1.0,
        "learning_rate_schedule": [args.learning_rate, args.learning_rate * 0.3],
        "learning_rate_drop_epoch": 11,
        "network": "Conv1x16-ReLU-Conv16x32(stride2)-ReLU-FC512x10; bias=True",
        "total_parameters": model.parameter_count(),
        "checkpoint_selection": "maximum validation accuracy; earliest epoch on ties",
        "preprocessing": "PIL bilinear Resize(8,8), ToTensor; no normalization or augmentation",
        "split_indices": {"train": [0, 55000], "validation": [55000, 60000], "end_exclusive": True},
        "validation_split": {"train": 55000, "validation": 5000}, "test_set_size": 10000,
        "device": str(device), "data_dir": str(args.data_dir.resolve()),
        "environment": {
            "python": platform.python_version(), "torch": torch.__version__,
            "torchvision": torchvision.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()),
        },
        "source_sha256": {p: hashlib.sha256((REPO_DIR / p).read_bytes()).hexdigest() for p in (
            "experiments/common/comparison_protocol.py", "experiments/cnn/model.py", "experiments/cnn/train.py")},
    }
    save_json(results_dir / "run_config.json", config)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=0.0)
    criterion = nn.CrossEntropyLoss()
    history = []
    best_accuracy, best_epoch = -1.0, 0
    print(f"seed={args.seed} device={device} train=55000 validation=5000 test=10000 parameters={model.parameter_count()}", flush=True)
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        if epoch == 11:
            for group in optimizer.param_groups:
                group["lr"] = args.learning_rate * 0.3
        epoch_started = time.perf_counter()
        model.train()
        loss_sum, samples = 0.0, 0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            loss_sum += loss.item() * labels.numel()
            samples += labels.numel()
        accuracy, _ = evaluate(model, validation_loader, device)
        row = {"epoch": epoch, "train_loss": loss_sum / samples, "validation_accuracy": accuracy,
               "learning_rate": optimizer.param_groups[0]["lr"], "elapsed_seconds": time.perf_counter() - epoch_started}
        history.append(row)
        if accuracy > best_accuracy:
            best_accuracy, best_epoch = accuracy, epoch
            torch.save(model.state_dict(), results_dir / "best_model.pt")
        save_history(results_dir / "history.csv", history)
        print(f"epoch={epoch} loss={row['train_loss']:.4f} validation={accuracy:.2f}% elapsed={row['elapsed_seconds']:.2f}s", flush=True)
    training_seconds = time.perf_counter() - started
    model.load_state_dict(torch.load(results_dir / "best_model.pt", map_location=device, weights_only=True))
    model.eval()
    warm_images, _ = next(iter(validation_loader))
    with torch.inference_mode():
        model(warm_images.to(device))
    synchronize(device)
    test_accuracy, inference_seconds = evaluate(model, test_loader, device)
    metrics = {**config, "epochs_run": args.epochs, "best_epoch": best_epoch,
               "best_validation_accuracy_percent": best_accuracy, "final_test_accuracy_percent": test_accuracy,
               "test_evaluations": 1, "test_based_tuning": False,
               "total_training_time_seconds": training_seconds,
               "average_time_per_epoch_seconds": training_seconds / args.epochs,
               "full_test_inference_time_seconds": inference_seconds,
               "average_inference_time_per_sample_ms": inference_seconds / len(test_loader.dataset) * 1000,
               "inference_batch_size": args.batch_size,
               "timing_scope": "full loader preprocessing + transfers + forward + accuracy + CUDA sync; validation warmup",
               "model_file_size_bytes": (results_dir / "best_model.pt").stat().st_size,
               "checkpoint_sha256": hashlib.sha256((results_dir / "best_model.pt").read_bytes()).hexdigest(),
               "parameter_payload_fp32_bytes": model.parameter_count() * 4,
               "dense_equivalent_macs_per_sample": model.dense_equivalent_macs_per_sample(),
               "bias_additions_per_sample": 1546, "relu_elements_per_sample": 1536,
               "hidden_feature_elements_sum": 1536, "hidden_feature_fp32_bytes_sum": 6144,
               "target_accuracy_percent": 95.0, "target_reached": test_accuracy >= 95.0,
               "history": history}
    save_json(results_dir / "metrics.json", metrics)
    save_curve(results_dir / "training_curve.png", history)
    print(json.dumps({k: metrics[k] for k in ("seed", "best_epoch", "best_validation_accuracy_percent", "final_test_accuracy_percent", "total_training_time_seconds")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
