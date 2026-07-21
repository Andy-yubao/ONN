#!/usr/bin/env python
"""Profile model — parameter counts, MACs, intermediate shapes.

Usage:
    python model/profile_model.py \\
        --model BaselineCNN \\
        --batch-size 1
    python model/profile_model.py \\
        --model TinyResNet \\
        --output model/profiles/tiny_resnet_profile.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile an MNIST baseline model")
    parser.add_argument(
        "--model",
        type=str,
        default="BaselineCNN",
        choices=["BaselineCNN", "TinyResNet"],
        help="Model to profile",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for profiling")
    parser.add_argument("--output", type=str, default=None, help="Save profile JSON to path")
    parser.add_argument("--markdown", action="store_true", help="Also print Markdown summary")
    return parser.parse_args()


def print_markdown(result: dict) -> None:
    """Print a human-readable Markdown summary."""
    print("\n## Model Profile Summary")
    print(f"\n- **Model**: {result['model_name']}")
    print(f"- **Total params**: {result['total_params']:,}")
    print(f"- **Trainable params**: {result['trainable_params']:,}")
    print(f"- **Total MACs**: {result['total_macs']:,}")
    print(f"- **Max intermediate elements**: {result['max_intermediate_elements']:,}")
    print(f"- **Input shape**: {result['input_shape']}")
    print("\n### Per-Layer Details")
    print("| Layer | Input Shape | Output Shape | Params | MACs |")
    print("|-------|-------------|--------------|-------:|-----:|")
    for d in result["layer_details"]:
        in_s = "×".join(str(s) for s in d["input_shape"])
        out_s = "×".join(str(s) for s in d["output_shape"])
        print(f"| {d['layer']} | {in_s} | {out_s} | {d['params']:,} | {d['macs']:,} |")


def main() -> None:
    args = parse_args()

    # Build model
    if args.model == "BaselineCNN":
        from onn_model.models.baseline_cnn import BaselineCNN
        model = BaselineCNN()
    else:
        from onn_model.models.tiny_resnet import TinyResNet
        model = TinyResNet()

    model.eval()

    # Profiling
    from onn_model.profiling import profile_model

    input_shape = (args.batch_size, 1, 28, 28)
    result = profile_model(model, input_shape)

    # Print
    print(json.dumps(result, indent=2, ensure_ascii=False))

    if args.markdown:
        print_markdown(result)

    # Save
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\nProfile saved to {out_path}")


if __name__ == "__main__":
    main()
