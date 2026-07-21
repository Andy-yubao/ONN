#!/usr/bin/env python
"""Train a model on MNIST.

Usage:
    python model/train.py --config model/configs/baseline_cnn.json
    python model/train.py --config model/configs/tiny_resnet.json
    python model/train.py --config model/configs/baseline_cnn.json --smoke-test
    python model/train.py --config model/configs/tiny_resnet.json --epochs 2 --device cpu
"""

import argparse
import json
import sys
from pathlib import Path

# Ensure the model package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from onn_model.engine import run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an MNIST baseline model")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to JSON config file (e.g. model/configs/baseline_cnn.json)",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--device", type=str, default=None, help='Override device ("cpu"/"cuda"/"auto")')
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--smoke-test", action="store_true", help="Run 2 batches per epoch only")
    parser.add_argument("--run-dir", type=str, default=None, help="Custom output directory")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint (.pt)")
    parser.add_argument("--eval-only", action="store_true", help="Only evaluate a checkpoint")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: config file not found: {config_path}")
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # CLI overrides
    overrides = {}
    if args.epochs is not None:
        overrides["epochs"] = args.epochs
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.device is not None:
        overrides["device"] = args.device
    if args.lr is not None:
        overrides["learning_rate"] = args.lr

    run_dir = Path(args.run_dir) if args.run_dir else None
    resume_ckpt = Path(args.resume) if args.resume else None

    print(f"Model:  {config.get('model', '?')}")
    print(f"Device: {config.get('device', 'auto')}")
    if args.smoke_test:
        print("*** SMOKE TEST MODE ***")

    run_experiment(
        config,
        run_dir=run_dir,
        smoke_test=args.smoke_test,
        resume_checkpoint=resume_ckpt,
        eval_only=args.eval_only,
        override_kwargs=overrides if overrides else None,
    )


if __name__ == "__main__":
    main()
