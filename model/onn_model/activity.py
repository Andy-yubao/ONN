"""Channel activation analysis for subsequent biologically-plausible learning rules.

Computes mean absolute activation and non-zero ratio per channel on the
validation set.  This analysis does **not** modify model weights or
inference output.
"""

from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader


@torch.no_grad()
def analyze_activations(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    target_layers: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compute channel activity statistics on the validation set.

    For each targeted layer (default: all Conv2d), computes:

    * ``mean_abs_activation``   — :math:`A_c = \\frac{1}{NHW}\\sum_{n,h,w} |y_{n,c,h,w}|`
    * ``nonzero_ratio``         — fraction of elements where ``y > 0``
    * ``zero_ratio``            — fraction of elements where ``y == 0``
    * ``sample_count``          — number of samples seen

    Parameters
    ----------
    model : nn.Module
    loader : DataLoader
    device : torch.device
    target_layers : list of str or None
        Module names to hook.  If ``None``, hooks all ``Conv2d`` children.

    Returns
    -------
    dict
        ``{"layer_name": {"channels": [stats_per_channel], "shape": ...}, ...}``
    """
    model.eval()
    results: Dict[str, Any] = {}

    # Collect activations per layer
    activation_data: Dict[str, list] = {}
    handles = []

    def _make_hook(name: str):
        def _hook(_module, _inp, out):
            # out: (N, C, H, W)
            y = out.detach()
            N, C, H, W = y.shape
            flat = y.view(N, C, -1)  # (N, C, H*W)
            abs_mean = flat.abs().mean(dim=(0, 2))  # (C,)
            nonzero = (flat > 0).float().mean(dim=(0, 2))  # (C,)
            zero = (flat == 0).float().mean(dim=(0, 2))  # (C,)

            if name not in activation_data:
                activation_data[name] = {
                    "abs_mean_sum": torch.zeros(C, device=y.device),
                    "nonzero_sum": torch.zeros(C, device=y.device),
                    "zero_sum": torch.zeros(C, device=y.device),
                    "count": 0,
                    "shape": (N, C, H, W),
                }
            d = activation_data[name]
            d["abs_mean_sum"] += abs_mean
            d["nonzero_sum"] += nonzero
            d["zero_sum"] += zero
            d["count"] += 1
            d["shape"] = (N, C, H, W)

        return _hook

    # Identify target modules
    for name, module in model.named_modules():
        if target_layers is not None:
            if name in target_layers:
                handles.append(module.register_forward_hook(_make_hook(name)))
        else:
            # Hook all Conv2d by default
            if isinstance(module, nn.Conv2d):
                handles.append(module.register_forward_hook(_make_hook(name)))

    # Run inference
    for images, _ in loader:
        images = images.to(device)
        model(images)

    # Remove hooks
    for h in handles:
        h.remove()

    # Aggregate
    for layer_name, data in activation_data.items():
        cnt = data["count"]
        channels = []
        abs_mean = data["abs_mean_sum"] / cnt
        nonzero = data["nonzero_sum"] / cnt
        zero = data["zero_sum"] / cnt

        for c in range(abs_mean.size(0)):
            channels.append({
                "channel": c,
                "mean_abs_activation": round(abs_mean[c].item(), 6),
                "nonzero_ratio": round(nonzero[c].item(), 6),
                "zero_ratio": round(zero[c].item(), 6),
            })

        results[layer_name] = {
            "channels": channels,
            "shape": list(data["shape"]),
            "sample_count": cnt,
            "channels_mean_abs": round(abs_mean.mean().item(), 6),
            "channels_std_abs": round(abs_mean.std().item(), 6),
        }

    return results


def summarize_activity(results: Dict[str, Any]) -> str:
    """Create a human-readable summary of activation analysis results."""
    lines = ["# Channel Activity Analysis Summary\n"]
    for layer_name, data in results.items():
        lines.append(f"\n## Layer: {layer_name}  (shape {data['shape']})")
        lines.append(f"  Samples: {data['sample_count']}")
        lines.append(f"  Mean abs activation: {data['channels_mean_abs']:.6f} ± {data['channels_std_abs']:.6f}")

        chs = data["channels"]
        sorted_by_act = sorted(chs, key=lambda c: c["mean_abs_activation"], reverse=True)
        lines.append(f"  Most active channels: {[c['channel'] for c in sorted_by_act[:5]]}")
        lines.append(f"  Least active channels: {[c['channel'] for c in sorted_by_act[-5:]]}")

        low_count = sum(1 for c in chs if c["mean_abs_activation"] < 0.01)
        near_zero = sum(1 for c in chs if c["zero_ratio"] > 0.9)
        lines.append(f"  Channels with mean_abs < 0.01: {low_count}/{len(chs)}")
        lines.append(f"  Channels with zero_ratio > 0.9: {near_zero}/{len(chs)}")

    return "\n".join(lines)
