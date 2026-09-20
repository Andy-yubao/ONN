"""Run PTQ, calibration, and full-test evaluation for the frozen SNN."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from experiments.common.comparison_protocol import make_loaders, set_seed
from experiments.snn.conv_small.model import DeviceIFConvSmall, normalized_temporal_weights
from experiments.snn.conv_small.train import DEFAULT_DATA_DIR, frozen_t4_quantile_030_encoder
from model.snn.integer_reference import IntegerSNNReference
from model.snn.quantization import dequantize_weight, quantize_weight_symmetric


ROOT = Path(__file__).resolve().parents[2]
CHECKPOINTS = {
    7: ROOT / "experiments/snn/conv_small/results/final_three_seed/seed_7/best_model.pt",
    17: ROOT / "experiments/snn/conv_small/results/event_regularization/beta_0p5_lambda_0p10_seed17/best_model.pt",
    27: ROOT / "experiments/snn/conv_small/results/final_three_seed/seed_27/best_model.pt",
}
EXPECTED_HASHES = {
    7: "44bf212122c0f8fc358b427d9dfd31cbc35dbc456e7048bdf0b8a106bdbf0431",
    17: "ecd4fa940c229bb9c717f33d68db2cdaf45ad6b83b48d597465d5431d8c845e6",
    27: "ffd273a8ee78f1ef85fe4abef74e86b50b3a8ad31b2fd3ad4c9afda00d6b685e",
}


def frozen_model() -> DeviceIFConvSmall:
    return DeviceIFConvSmall(readout_mode="accumulated_membrane", temporal_beta=0.5)


def load_frozen_state(seed: int) -> tuple[dict[str, torch.Tensor], str]:
    path = CHECKPOINTS[seed]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != EXPECTED_HASHES[seed]:
        raise RuntimeError(f"checkpoint hash mismatch for seed {seed}: {digest}")
    return torch.load(path, map_location="cpu", weights_only=True), digest


def make_weight_only_model(state: dict[str, torch.Tensor]) -> tuple[DeviceIFConvSmall, dict]:
    model = frozen_model()
    model.load_state_dict(state)
    metadata = {}
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            quantized, spec = quantize_weight_symmetric(parameter, name)
            parameter.copy_(dequantize_weight(quantized, spec))
            metadata[name] = spec.to_dict()
    return model, metadata


@torch.inference_mode()
def evaluate_float(model, loader, encoder, device: torch.device) -> float:
    model.to(device).eval()
    correct = total = 0
    for images, labels in loader:
        times = encoder.first_spike_times(images.to(device))
        logits = model(times, 4)
        correct += int((logits.argmax(1).cpu() == labels).sum().item())
        total += labels.numel()
    return 100.0 * correct / total


@torch.inference_mode()
def evaluate_integer(reference, loader, encoder) -> float:
    correct = total = 0
    for images, labels in loader:
        logits = reference.forward(encoder.first_spike_times(images))
        correct += int((logits.argmax(1) == labels).sum().item())
        total += labels.numel()
    return 100.0 * correct / total


def _range(values: list[torch.Tensor]) -> dict[str, float]:
    flat = torch.cat([value.detach().flatten().cpu().to(torch.float32) for value in values])
    absolute = flat.abs()
    return {
        "min": float(flat.min().item()),
        "max": float(flat.max().item()),
        "abs_max": float(absolute.max().item()),
        "abs_p99": float(torch.quantile(absolute, 0.99).item()),
        "abs_p999": float(torch.quantile(absolute, 0.999).item()),
    }


@torch.inference_mode()
def collect_fp32_ranges(model, loader, encoder, sample_limit: int) -> dict:
    model.cpu().eval()
    collected: dict[str, list[torch.Tensor]] = {
        "conv1_current": [], "mem1_integrated": [], "mem1_post_reset": [],
        "conv2_current": [], "mem2_integrated": [], "mem2_post_reset": [],
        "readout_current": [], "weighted_logits": [],
    }
    seen = 0
    weights = normalized_temporal_weights(4, 0.5)
    for images, _labels in loader:
        if seen >= sample_limit:
            break
        images = images[: sample_limit - seen]
        times = encoder.first_spike_times(images)
        batch = images.shape[0]
        mem1 = torch.zeros(batch, 16, 8, 8)
        mem2 = torch.zeros(batch, 32, 4, 4)
        weighted = torch.zeros(batch, 10)
        for step in range(4):
            input_spikes = (times == step).to(torch.float32)
            current1 = model.conv1(input_spikes)
            integrated1 = mem1 + current1
            spikes1 = integrated1 > 1.0
            mem1 = integrated1 - spikes1 * 1.0
            current2 = model.conv2(spikes1.to(torch.float32))
            integrated2 = mem2 + current2
            spikes2 = integrated2 > 1.0
            mem2 = integrated2 - spikes2 * 1.0
            current3 = model.readout(spikes2.flatten(1).to(torch.float32))
            weighted = weighted + weights[step] * current3
            for key, value in (
                ("conv1_current", current1), ("mem1_integrated", integrated1),
                ("mem1_post_reset", mem1), ("conv2_current", current2),
                ("mem2_integrated", integrated2), ("mem2_post_reset", mem2),
                ("readout_current", current3), ("weighted_logits", weighted),
            ):
                collected[key].append(value)
        seen += batch
    return {"sample_count": seen, "statistics": {key: _range(value) for key, value in collected.items()}}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results/ptq_summary.json")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--calibration-samples", type=int, default=1024)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(17)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        (args.device if args.device != "auto" else "cpu")
    )
    train_loader, _validation_loader, test_loader = make_loaders(
        args.data_dir, args.batch_size, 0
    )
    calibration_loader = DataLoader(
        train_loader.dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    encoder, encoder_metadata = frozen_t4_quantile_030_encoder()
    results = {
        "model_identity": {
            "status": "frozen SNN research model; deployment candidate, not a production champion",
            "architecture": "Conv1x16-IF-subtract-Conv16x32(stride2)-IF-subtract-FC512x10",
            "input": "encoder latency map [B,1,8,8], values -1 or 0..3",
            "time_steps": 4, "threshold": 1.0, "parameters": 9872,
            "readout": "time-weighted output currents; beta=0.5",
            "fp32_temporal_weights": normalized_temporal_weights(4, 0.5).tolist(),
            "encoder": encoder_metadata,
        },
        "calibration": {"split": "training indices [0,1024), ordered", "sample_count": args.calibration_samples},
        "seeds": {},
    }
    weight_only_metadata = None
    integer_metadata = None
    for seed in (7, 17, 27):
        state, digest = load_frozen_state(seed)
        fp32 = frozen_model()
        fp32.load_state_dict(state)
        weight_only, current_weight_metadata = make_weight_only_model(state)
        # Preserve the original conservative PTQ baseline here.  Compact state
        # selection is performed separately by state_quantization.py.
        reference = IntegerSNNReference(state, guard_bits=8)
        seed_result = {
            "checkpoint_sha256": digest,
            "fp32_accuracy_percent": evaluate_float(fp32, test_loader, encoder, device),
            "weight_only_int8_accuracy_percent": evaluate_float(weight_only, test_loader, encoder, device),
            "integer_reference_accuracy_percent": evaluate_integer(reference, test_loader, encoder),
        }
        results["seeds"][str(seed)] = seed_result
        print(seed, seed_result, flush=True)
        if seed == 17:
            results["calibration"]["fp32_ranges"] = collect_fp32_ranges(
                fp32, calibration_loader, encoder, args.calibration_samples
            )
            weight_only_metadata = current_weight_metadata
            integer_metadata = reference.quantization_metadata()
    results["weight_only_scheme"] = weight_only_metadata
    results["integer_scheme"] = integer_metadata
    for variant, key in (
        ("fp32", "fp32_accuracy_percent"),
        ("weight_only_int8", "weight_only_int8_accuracy_percent"),
        ("integer_reference", "integer_reference_accuracy_percent"),
    ):
        values = [row[key] for row in results["seeds"].values()]
        results.setdefault("aggregate", {})[variant] = {
            "mean_accuracy_percent": sum(values) / len(values),
            "sample_std_pct": float(torch.tensor(values, dtype=torch.float64).std().item()),
        }
    baseline = results["aggregate"]["fp32"]["mean_accuracy_percent"]
    for row in results["aggregate"].values():
        row["delta_vs_fp32_pct"] = row["mean_accuracy_percent"] - baseline
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(results["aggregate"], indent=2), flush=True)


if __name__ == "__main__":
    main()
