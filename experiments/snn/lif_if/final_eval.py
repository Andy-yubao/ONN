"""Evaluate one validation-selected IF/LIF checkpoint on MNIST test exactly once."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from experiments.common.comparison_protocol import make_loaders
from experiments.snn.conv_small.train import frozen_t4_quantile_030_encoder
from .model import DeviceLIFConvSmall
from .train import evaluate, select_device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--decay", required=True, type=float)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--data-dir", type=Path,
                        default=Path(__file__).resolve().parents[3] / "data")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    device = select_device(args.device)
    model = DeviceLIFConvSmall(
        readout_mode="accumulated_membrane", temporal_beta=0.5, decay=args.decay
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    encoder, calibration = frozen_t4_quantile_030_encoder()
    _, _, test_loader = make_loaders(args.data_dir, args.batch_size, 0)
    accuracy, seconds, workload = evaluate(model, test_loader, encoder, device, collect_stats=True)
    samples = len(test_loader.dataset)
    result = {
        "seed": args.seed,
        "decay": args.decay,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "checkpoint_selection": "maximum validation accuracy; earliest epoch on ties",
        "test_evaluations": 1,
        "test_accuracy_percent": accuracy,
        "test_samples": samples,
        "test_seconds": seconds,
        "parameters": model.parameter_count(),
        "encoder_calibration": calibration,
        "workload_test_per_sample": {
            key: value / samples for key, value in workload.items()
            if key != "input_spike_histogram"
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
