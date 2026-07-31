#!/usr/bin/env python
"""Profile model — parameter counts, MACs, FPGA proxy estimates.

Usage:
    python model/profile_model.py --model BaselineCNN
    python model/profile_model.py --model MicroCNNSmall --batch-size 1 --markdown
    python model/profile_model.py --model TinyResNet --output model/profiles/tiny_resnet_profile.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.resolve()))

import torch


MODEL_CHOICES = [
    "BaselineCNN",
    "TinyResNet",
    "MicroCNNSmall",
    "MicroCNNExtraSmall",
    "DepthwiseMicroCNN",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile an MNIST model")
    parser.add_argument(
        "--model",
        type=str,
        default="BaselineCNN",
        choices=MODEL_CHOICES,
        help="Model to profile",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for profiling")
    parser.add_argument("--output", type=str, default=None, help="Save profile JSON to path")
    parser.add_argument("--markdown", action="store_true", help="Also print Markdown summary")
    return parser.parse_args()


def _build_model(model_name: str):
    if model_name == "BaselineCNN":
        from onn_model.models.baseline_cnn import BaselineCNN
        return BaselineCNN()
    elif model_name == "TinyResNet":
        from onn_model.models.tiny_resnet import TinyResNet
        return TinyResNet()
    elif model_name in ("MicroCNNSmall",):
        from onn_model.models.compact_cnn import MicroCNNSmall
        return MicroCNNSmall()
    elif model_name in ("MicroCNNExtraSmall",):
        from onn_model.models.compact_cnn import MicroCNNExtraSmall
        return MicroCNNExtraSmall()
    elif model_name in ("DepthwiseMicroCNN",):
        from onn_model.models.compact_cnn import DepthwiseMicroCNN
        return DepthwiseMicroCNN()
    else:
        raise ValueError(f"Unknown model: {model_name}")


def print_markdown(result: dict) -> None:
    """Print a human-readable Markdown summary."""
    fpga = result.get("fpga_proxy", {})

    print("\n## Model Profile Summary")
    print(f"\n- **Model**: {result['model_name']}")
    print(f"- **Total params**: {result['total_params']:,}")
    print(f"- **Trainable params**: {result['trainable_params']:,}")
    print(f"- **Total MACs**: {result['total_macs']:,}")
    print(f"- **Max intermediate elements**: {result['max_intermediate_elements']:,}")
    print(f"- **Input shape**: {result['input_shape']}")

    print("\n### FPGA Resource Proxy")
    print(f"- **Conv params**: {fpga.get('conv_params', 0):,}")
    print(f"- **Linear params**: {fpga.get('linear_params', 0):,}")
    print(f"- **W8 weight bytes**: {fpga.get('total_weight_bytes_int8', 0):,}")
    print(f"- **W16 weight bytes**: {fpga.get('total_weight_bytes_int16', 0):,}")
    print(f"- **FP32 weight bytes**: {fpga.get('total_weight_bytes_fp32', 0):,}")
    print(f"- **Peak activation elements**: {fpga.get('peak_activation_elements', 0):,}")
    print(f"- **A8 peak activation bytes**: {fpga.get('peak_activation_bytes_int8', 0):,}")
    print(f"- **A16 peak activation bytes**: {fpga.get('peak_activation_bytes_int16', 0):,}")
    print(f"- **A8 ping-pong estimate**: {fpga.get('estimated_ping_pong_activation_bytes_int8', 0):,}")

    print("\n### Per-Layer Details")
    print("| Layer | Name | Input Shape | Output Shape | Params | MACs |")
    print("|-------|------|-------------|--------------|-------:|-----:|")
    for d in result["layer_details"]:
        in_s = "×".join(str(s) for s in d["input_shape"])
        out_s = "×".join(str(s) for s in d["output_shape"])
        print(f"| {d['layer']} | {d['name']} | {in_s} | {out_s} | {d['params']:,} | {d['macs']:,} |")

    print()

    # EP4CE10 reference
    print("### EP4CE10F17C8 Reference")
    print("| Resource | Budget |")
    print("|----------|-------:|")
    print("| On-chip RAM | ~414 Kbit (~51.75 KiB) |")
    print("| Logic elements | ~10K LE |")
    print("| Hardware multipliers | Limited |")
    print()
    print("**Caveats:**")
    print("- These are software-level conservative proxies, not post-synthesis values.")
    print("- True streaming implementations may require only line buffers.")
    print("- Residual networks need additional shortcut cache.")
    print("- M9K fragmentation and port configuration not yet accounted for.")
    print("- Cannot replace Quartus synthesis.")


def main() -> None:
    args = parse_args()

    model = _build_model(args.model)
    model.eval()

    from onn_model.profiling import profile_model

    input_shape = (args.batch_size, 1, 28, 28)
    result = profile_model(model, input_shape)

    # Ensure model_name is set
    result["model_name"] = type(model).__name__

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
