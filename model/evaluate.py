#!/usr/bin/env python
"""Evaluate a trained checkpoint on the MNIST test set.

Usage:
    python model/evaluate.py \\
        --config model/configs/tiny_resnet.json \\
        --checkpoint model/runs/TinyResNet/20250101_120000/best.pt
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from onn_model.engine import evaluate_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained MNIST model checkpoint")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to JSON config matching the checkpoint",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint .pt file (best.pt or last.pt)",
    )
    parser.add_argument("--device", type=str, default=None, help='Override device')
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: config file not found: {config_path}")
        sys.exit(1)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"Error: checkpoint not found: {ckpt_path}")
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    if args.device:
        config["device"] = args.device

    print(f"Evaluating {ckpt_path} ...")
    results = evaluate_checkpoint(ckpt_path, config)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
