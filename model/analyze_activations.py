#!/usr/bin/env python
"""Analyze channel activations of a trained model on the validation set.

Usage:
    python model/analyze_activations.py \\
        --config model/configs/tiny_resnet.json \\
        --checkpoint model/runs/TinyResNet/20250101_120000/best.pt
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze channel activations of a trained model")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to training config JSON",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to best.pt / last.pt checkpoint",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output", type=str, default=None, help="Save activity JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    device_str = args.device or config.get("device", "auto")
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    # Load model
    from onn_model.engine import _get_model
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_name = ckpt.get("model_name", config.get("model", "BaselineCNN"))
    model = _get_model(model_name).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded {model_name} from {args.checkpoint}")

    # Validation loader
    from onn_model.data import get_train_val_loaders
    _, val_loader, _ = get_train_val_loaders(
        root=config.get("data_root", "model/data"),
        batch_size=config.get("batch_size", 128),
        seed=config.get("seed", 42),
    )

    # Analysis
    from onn_model.activity import analyze_activations, summarize_activity
    results = analyze_activations(model, val_loader, device)

    summary = summarize_activity(results)
    print(summary)

    # Save
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nActivity data saved to {out_path}")

    # Also save summary text
    if args.output:
        summary_path = out_path.with_suffix(".md")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(summary)
        print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
