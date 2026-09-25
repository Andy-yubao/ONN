"""Train validation-selected 8x8 LIF variants or matched 28x28 SNN/CNN."""

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

import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from experiments.common.comparison_protocol import make_loaders, set_seed
from experiments.common.device_latency_encoder import DeviceLatencyEncoder, quantile_boundaries_from_histogram
from experiments.snn.conv_small.train import frozen_t4_quantile_030_encoder, EVENT_PROXY_NORMALIZATION
from .model import DecayConvSNN, MatchedCNN


REPO = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
NORM_28 = EVENT_PROXY_NORMALIZATION * (49 / 16)
TARGET_FIRING_RATIO_28 = 0.18


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("snn", "cnn"), required=True)
    parser.add_argument("--size", choices=(8, 28), type=int, required=True)
    parser.add_argument("--decay", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=REPO / "data")
    parser.add_argument("--calibration-file", type=Path, default=HERE / "calibration_28.json")
    parser.add_argument("--final-test", action="store_true")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return parser.parse_args()


def loaders_28(data_dir, batch_size):
    transform = transforms.ToTensor()
    full = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    common = dict(batch_size=batch_size, num_workers=0, pin_memory=torch.cuda.is_available())
    return (DataLoader(Subset(full, range(55000)), shuffle=True, **common),
            DataLoader(Subset(full, range(55000, 60000)), shuffle=False, **common),
            DataLoader(test, shuffle=False, **common))


@torch.inference_mode()
def calibrate_28(train_dataset, batch_size):
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    pixel_hist = torch.zeros(256, dtype=torch.long)
    for images, _ in loader:
        pixel_hist += torch.bincount((images * 255).round().long().flatten(), minlength=256)
    all_pixels = int(pixel_hist.sum())
    maximum_firing_ratio = 1 - int(pixel_hist[0]) / all_pixels
    if TARGET_FIRING_RATIO_28 >= maximum_firing_ratio:
        raise ValueError("target firing ratio exceeds the nonzero-pixel ceiling")
    desired = round(TARGET_FIRING_RATIO_28 * all_pixels)
    firing = all_pixels - pixel_hist.cumsum(0)
    cut_index = int((firing - desired).abs().argmin())
    cut = (cut_index + 0.5) / 255.0
    threshold = 0.10 + 0.90 * cut * (1 - math.exp(-24 / 5))
    source = DeviceLatencyEncoder(g_threshold=threshold)
    latency_hist = torch.zeros(24, dtype=torch.long)
    for images, _ in loader:
        times = source.first_spike_times(images)
        valid = times[times >= 0]
        latency_hist += torch.bincount(valid, minlength=24)
    boundaries = quantile_boundaries_from_histogram(latency_hist, 4)
    return {"g_threshold": threshold, "source_time_steps": 24, "time_steps": 4,
            "time_mapping": "quantile", "quantile_boundaries": list(boundaries),
            "training_split": [0, 55000], "training_pixels": all_pixels,
            "target_firing_ratio": TARGET_FIRING_RATIO_28,
            "maximum_nonzero_firing_ratio": maximum_firing_ratio,
            "actual_firing_ratio": int(latency_hist.sum()) / all_pixels,
            "source_latency_histogram": latency_hist.tolist()}


def encoder_from_calibration(calibration):
    return DeviceLatencyEncoder(g_threshold=calibration["g_threshold"],
                                quantized_time_steps=4, time_mapping="quantile",
                                quantile_boundaries=tuple(calibration["quantile_boundaries"]))


def evaluate(model, loader, device, encoder=None, collect_stats=False):
    model.eval()
    correct = total = 0
    counts = {name: 0 for name in ("input_spikes", "layer1_spikes", "layer2_spikes",
                                      "effective_synaptic_additions")}
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if encoder is None:
                logits = model(images)
            else:
                times = encoder.first_spike_times(images)
                if collect_stats:
                    logits, batch_counts = model(times, collect_stats=True)
                    for key, value in batch_counts.items():
                        counts[key] += value
                else:
                    logits = model(times)
            correct += int((logits.argmax(1) == labels).sum().item())
            total += labels.numel()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return 100 * correct / total, time.perf_counter() - start, counts


def save_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    args = arguments()
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("epochs and batch size must be positive")
    if args.size == 8 and args.model != "snn":
        raise ValueError("8x8 run is for LIF/IF comparison only")
    if args.model == "cnn" and args.decay != 1.0:
        raise ValueError("CNN does not have a membrane decay")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.results_dir.exists() and any(args.results_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {args.results_dir}")
    args.results_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    train, validation, test = (make_loaders(args.data_dir, args.batch_size, 0) if args.size == 8
                               else loaders_28(args.data_dir, args.batch_size))
    first_stride = 1 if args.size == 8 else 2
    model = (DecayConvSNN(args.size, first_stride, args.decay) if args.model == "snn"
             else MatchedCNN(args.size, first_stride)).to(device)
    encoder = None
    calibration = None
    normalization = None
    if args.model == "snn":
        if args.size == 8:
            encoder, calibration = frozen_t4_quantile_030_encoder()
            normalization = EVENT_PROXY_NORMALIZATION
        else:
            if args.calibration_file.exists():
                calibration = json.loads(args.calibration_file.read_text(encoding="utf-8"))
            else:
                calibration = calibrate_28(train.dataset, args.batch_size)
                save_json(args.calibration_file, calibration)
            encoder = encoder_from_calibration(calibration)
            normalization = NORM_28
    tracked = [Path(__file__), HERE / "model.py", REPO / "experiments/common/device_latency_encoder.py"]
    config = {"protocol_id": f"mnist{args.size}x{args.size}-sequential-v1", "model": args.model,
              "size": args.size, "seed": args.seed, "membrane_decay": args.decay if args.model == "snn" else None,
              "parameters": model.parameter_count(), "epochs": args.epochs, "batch_size": args.batch_size,
              "data_split": {"train": [0, 55000], "validation": [55000, 60000], "test": "official 10000"},
              "preprocessing": "PIL bilinear resize 8x8 + ToTensor" if args.size == 8 else "ToTensor original 28x28",
              "optimizer": "Adam", "learning_rates": [0.001, 0.0003], "drop_epoch": 11,
              "gradient_clip": 1.0, "weight_decay": 0.0, "time_steps": 4 if encoder else None,
              "readout_beta": 0.5 if encoder else None,
              "event_lambda": 0.10 if encoder else None,
              "event_normalization": normalization,
              "event_normalization_origin": ("frozen 8x8 beta=1 reference" if args.size == 8 else
                  "8x8 reference scaled by hidden spatial area ratio 49/16") if encoder else None,
              "encoder_calibration": calibration, "final_test_requested": args.final_test,
              "environment": {"python": platform.python_version(), "torch": torch.__version__,
                              "torchvision": torchvision.__version__, "cuda": torch.version.cuda,
                              "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                              "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                              "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())},
              "source_sha256": {str(p.relative_to(REPO)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in tracked}}
    save_json(args.results_dir / "run_config.json", config)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    best_accuracy, best_epoch = -1.0, 0
    history = []
    begin = time.perf_counter()
    print(f"model={args.model} size={args.size} decay={args.decay} seed={args.seed} parameters={model.parameter_count()}", flush=True)
    for epoch in range(1, args.epochs + 1):
        if epoch == 11:
            for group in optimizer.param_groups:
                group["lr"] = 0.0003
        epoch_start = time.perf_counter()
        model.train()
        total_loss = total_ce = samples = 0
        for images, labels in train:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            if encoder is not None:
                logits, proxy = model(encoder.first_spike_times(images), return_event_proxy=True)
                ce = criterion(logits, labels)
                loss = ce + 0.10 * proxy / normalization
            else:
                ce = loss = criterion(model(images), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * labels.numel()
            total_ce += ce.item() * labels.numel()
            samples += labels.numel()
        accuracy, _, _ = evaluate(model, validation, device, encoder)
        row = {"epoch": epoch, "train_loss": total_loss / samples, "train_ce_loss": total_ce / samples,
               "validation_accuracy": accuracy, "learning_rate": optimizer.param_groups[0]["lr"],
               "elapsed_seconds": time.perf_counter() - epoch_start}
        history.append(row)
        if accuracy > best_accuracy:
            best_accuracy, best_epoch = accuracy, epoch
            torch.save(model.state_dict(), args.results_dir / "best_model.pt")
        with (args.results_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=row.keys())
            writer.writeheader()
            writer.writerows(history)
        print(f"epoch={epoch} val={accuracy:.2f}% loss={row['train_loss']:.4f} elapsed={row['elapsed_seconds']:.1f}s", flush=True)
    model.load_state_dict(torch.load(args.results_dir / "best_model.pt", map_location=device, weights_only=True))
    validation_accuracy, validation_seconds, validation_counts = evaluate(model, validation, device, encoder,
                                                                          collect_stats=encoder is not None)
    metrics = {"config": config, "best_epoch": best_epoch, "best_validation_accuracy_percent": best_accuracy,
               "reloaded_validation_accuracy_percent": validation_accuracy,
               "validation_seconds": validation_seconds, "validation_workload_total": validation_counts,
               "training_seconds": time.perf_counter() - begin, "dense_macs_per_sample": model.dense_macs(),
               "checkpoint_sha256": hashlib.sha256((args.results_dir / "best_model.pt").read_bytes()).hexdigest(),
               "test_evaluations": 0, "history": history}
    if args.final_test:
        test_accuracy, test_seconds, test_counts = evaluate(model, test, device, encoder,
                                                            collect_stats=encoder is not None)
        metrics.update({"final_test_accuracy_percent": test_accuracy, "test_seconds": test_seconds,
                        "test_workload_total": test_counts, "test_evaluations": 1})
    save_json(args.results_dir / "metrics.json", metrics)
    print(json.dumps({"best_epoch": best_epoch, "best_val": best_accuracy,
                      "test": metrics.get("final_test_accuracy_percent")}), flush=True)


if __name__ == "__main__":
    main()
