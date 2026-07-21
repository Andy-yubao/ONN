"""Channel activation analysis for subsequent biologically-plausible learning rules.

Computes mean absolute activation and active-ratio per channel on the
validation set.  This analysis does **not** modify model weights, BN
running statistics, or inference output.

**What is measured**
By default the analysis hooks the *post-activation* output of each
functional stage (e.g. the final ReLU of a Conv-BN-ReLU block, or the
output ReLU of a BasicBlock).  This is the signal actually passed to
the next layer, *not* the raw convolution output.

**Corrected terminology (M1-B.1 audit)**
- ``active_ratio``    — fraction of elements where ``|value| > epsilon``  (was ``nonzero_ratio``)
- ``near_zero_ratio`` — fraction of elements where ``|value| <= epsilon`` (was ``zero_ratio``)
- ``mean_abs_activation`` — unchanged, but now computed over all elements
- ``sample_count`` — number of **images** processed (not batches)
- ``batch_count`` — number of **batches** processed
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader


DEFAULT_EPSILON = 1e-6


@torch.no_grad()
def analyze_activations(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    epsilon: float = DEFAULT_EPSILON,
    debug_raw_conv: bool = False,
) -> Dict[str, Any]:
    """Compute channel activity statistics on the validation set.

    By default hooks are registered on the modules returned by
    ``model.get_activity_targets()`` (post-activation outputs of each
    functional stage).  When ``debug_raw_conv=True`` raw ``Conv2d``
    outputs are used instead.

    For each targeted layer the per-channel statistics are:

    * ``mean_abs_activation`` — :math:`\\frac{1}{NHW}\\sum_{n,h,w} |y_{n,c,h,w}|`
    * ``active_ratio`` — fraction of elements where :math:`|y| > \\epsilon`
    * ``near_zero_ratio`` — fraction of elements where :math:`|y| \\leq \\epsilon`

    Accumulation is element-weighted, **not** batch-mean-averaged, so the
    last (potentially partial) batch receives the correct weight.

    Parameters
    ----------
    model : nn.Module
    loader : DataLoader
    device : torch.device
    epsilon : float
        Threshold below which an element is considered near-zero.
    debug_raw_conv : bool
        If True, hook raw ``Conv2d`` outputs instead of post-activation targets.

    Returns
    -------
    dict
        ``{"layer_name": {"channels": [per-channel dicts], "shape": ...,  …}, …}``
    """
    # --- preserve model state ------------------------------------------------
    was_training = model.training
    model.eval()

    results: Dict[str, Any] = {}
    activation_data: Dict[str, dict] = {}
    handles: list = []

    try:
        # --- identify target modules -----------------------------------------
        if debug_raw_conv:
            targets: Dict[str, nn.Module] = {
                name: mod
                for name, mod in model.named_modules()
                if isinstance(mod, nn.Conv2d)
            }
        else:
            if hasattr(model, "get_activity_targets"):
                targets = model.get_activity_targets()
            else:
                # Fallback: hook all ReLU modules (parameterless, safe)
                targets = {
                    name: mod
                    for name, mod in model.named_modules()
                    if isinstance(mod, nn.ReLU)
                }

        # --- register hooks --------------------------------------------------
        def _make_hook(
            name: str,
        ):
            def _hook(_module, _inp, out):
                y = out.detach()
                N, C, H, W = y.shape
                flat = y.view(N, C, -1)  # (N, C, H*W)
                element_count = N * H * W  # total spatial positions in this batch

                # Per-channel accumulators (element-weighted)
                sum_abs_ch = flat.abs().sum(dim=(0, 2))  # (C,)
                active_ch = (flat.abs() > epsilon).float().sum(dim=(0, 2))  # (C,)

                if name not in activation_data:
                    activation_data[name] = {
                        "sum_abs": torch.zeros(C, device=y.device),
                        "active_sum": torch.zeros(C, device=y.device),
                        "sample_count": 0,
                        "batch_count": 0,
                        "element_count_per_channel": 0,
                        "shape": (N, C, H, W),
                    }
                d = activation_data[name]
                d["sum_abs"] += sum_abs_ch
                d["active_sum"] += active_ch
                d["sample_count"] += N
                d["batch_count"] += 1
                d["element_count_per_channel"] += element_count
                d["shape"] = (N, C, H, W)

            return _hook

        for name, module in targets.items():
            handles.append(module.register_forward_hook(_make_hook(name)))

        # --- run inference ---------------------------------------------------
        for images, _ in loader:
            images = images.to(device)
            model(images)

    finally:
        # --- hooks MUST be removed even on exception -------------------------
        for h in handles:
            h.remove()

        # --- restore original train/eval state -------------------------------
        if was_training:
            model.train()

    # --- aggregate -----------------------------------------------------------
    for layer_name, data in activation_data.items():
        channels: List[Dict[str, Any]] = []
        total_elements = data["element_count_per_channel"]
        n_channels = data["sum_abs"].size(0)

        for c in range(n_channels):
            mean_abs = (data["sum_abs"][c] / total_elements).item()
            active_ratio = (data["active_sum"][c] / total_elements).item()
            entry: Dict[str, Any] = {
                "channel": c,
                "mean_abs_activation": round(mean_abs, 8),
                "active_ratio": round(active_ratio, 8),
                "near_zero_ratio": round(1.0 - active_ratio, 8),
            }
            channels.append(entry)

        ch_mean_abs = data["sum_abs"] / total_elements
        results[layer_name] = {
            "target": layer_name,
            "shape": list(data["shape"]),
            "epsilon": epsilon,
            "sample_count": data["sample_count"],
            "batch_count": data["batch_count"],
            "element_count_per_channel": total_elements,
            "channels": channels,
            "channels_mean_abs": round(ch_mean_abs.mean().item(), 8),
            "channels_std_abs": round(ch_mean_abs.std().item(), 8),
        }

    return results


def summarize_activity(results: Dict[str, Any]) -> str:
    """Create a human-readable summary of activation analysis results."""
    lines = [
        "# Channel Activity Analysis Summary",
        "",
        f"Statistics are based on post-activation outputs",
        f"(epsilon = {_get_epsilon(results):.0e}).",
        "",
    ]
    for layer_name, data in results.items():
        chs = data["channels"]
        sorted_by_act = sorted(chs, key=lambda c: c["mean_abs_activation"], reverse=True)

        lines.append(f"## Layer: {layer_name}")
        lines.append(f"  Shape: {data['shape']}")
        lines.append(f"  Samples: {data['sample_count']}  |  Batches: {data['batch_count']}")
        lines.append(
            f"  Mean abs: {data['channels_mean_abs']:.6f} ± {data['channels_std_abs']:.6f}"
        )
        lines.append(
            f"  Most active channels:    {[c['channel'] for c in sorted_by_act[:5]]}"
        )
        lines.append(
            f"  Least active channels:   {[c['channel'] for c in sorted_by_act[-5:]]}"
        )

        low_active = sum(1 for c in chs if c["mean_abs_activation"] < 0.01)
        near_zero_ch = sum(1 for c in chs if c["near_zero_ratio"] > 0.9)
        lines.append(f"  Channels mean_abs < 0.01: {low_active}/{len(chs)}")
        lines.append(f"  Channels near_zero_ratio > 0.9: {near_zero_ch}/{len(chs)}")

        lines.append("")

    # Overall caveat
    lines.append("")
    lines.append("## Limitations")
    lines.append("")
    lines.append(
        "- Mean absolute values are **not directly comparable across layers** "
        "because they depend on the preceding BatchNorm scale and the layer's "
        "own weight distribution."
    )
    lines.append(
        "- A low active-ratio channel may still be useful for discriminating "
        "a small number of classes.  Do not prune based solely on activity."
    )
    lines.append(
        "- These statistics describe the validation-set behaviour of the "
        "*trained* model.  Activity patterns can change during training."
    )

    return "\n".join(lines)


def _get_epsilon(results: Dict[str, Any]) -> float:
    """Extract epsilon from first layer that has it."""
    for data in results.values():
        if isinstance(data, dict) and "epsilon" in data:
            return data["epsilon"]
    return DEFAULT_EPSILON
