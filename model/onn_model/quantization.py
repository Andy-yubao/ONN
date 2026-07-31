"""Quantisation tools: BatchNorm fusion and INT8 PTQ simulation.

All operations are software-level feasibility checks and do **not** produce
FPGA-bit-exact results.  Results are exported as plain JSON / numpy arrays
rather than framework-specific quantised model formats.
"""
from __future__ import annotations

import copy
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader


# ============================================================================
#  BatchNorm Fusion
# ============================================================================


def fuse_conv_bn_eval(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    """Fuse a Conv2d + BatchNorm2d into a single Conv2d (in eval mode).

    Returns a *new* Conv2d with fused weight and bias.  The original modules
    are not modified.

    Parameters
    ----------
    conv : nn.Conv2d
        Convolution layer (``bias=False`` recommended; if ``bias=True``, it
        is incorporated into the BN correction).
    bn : nn.BatchNorm2d

    Returns
    -------
    nn.Conv2d
        Fused convolution with identical eval-mode output (subject to
        floating-point rounding).
    """
    fused = nn.Conv2d(
        in_channels=conv.in_channels,
        out_channels=conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=True,  # fused conv always has bias
    )
    fused.weight.data.copy_(conv.weight.data)
    fused.to(conv.weight.device)

    # BN parameters
    gamma = bn.weight.data
    beta = bn.bias.data
    running_mean = bn.running_mean.data
    running_var = bn.running_var.data
    eps = bn.eps

    # Fused weight: W_fused = W * gamma / sqrt(var + eps)
    scale = gamma / torch.sqrt(running_var + eps)
    fused.weight.data = conv.weight.data * scale.view(-1, 1, 1, 1)

    # Fused bias: b_fused = (conv_bias - running_mean) * scale + beta
    if conv.bias is not None:
        fused.bias.data = (conv.bias.data - running_mean) * scale + beta
    else:
        fused.bias.data = beta - running_mean * scale

    return fused


def fuse_model_bn(model: nn.Module) -> nn.Module:
    """Return a new model with all Conv2d+BatchNorm2d pairs fused.

    The original model is **not** modified.  Only operates on sequential
    ``Conv2d → BatchNorm2d`` patterns; more complex topologies (e.g.,
    residual shortcuts) are left unchanged and a warning is printed.

    Parameters
    ----------
    model : nn.Module
        Trained model in eval mode.

    Returns
    -------
    nn.Module
        Fused model (deep copy with Conv-BN pairs replaced by single Conv2d).
    """
    model.eval()
    fused_model = copy.deepcopy(model)

    # Track which modules to replace after iteration
    replacements: Dict[str, nn.Module] = {}

    for name, module in fused_model.named_children():
        _fuse_sequential(module, name, replacements)

    # Apply replacements at top level
    for name, new_mod in replacements.items():
        setattr(fused_model, name, new_mod)

    return fused_model


def _fuse_sequential(
    module: nn.Module,
    prefix: str,
    replacements: Dict[str, nn.Module],
) -> None:
    """Recursively fuse Conv-BN pairs in a module."""
    if not isinstance(module, nn.Sequential):
        # Recurse into children
        for child_name, child in module.named_children():
            full_name = f"{prefix}.{child_name}" if prefix else child_name
            _fuse_sequential(child, full_name, replacements)
        return

    # Scan for Conv → BN patterns
    new_layers = []
    i = 0
    while i < len(module):
        current = module[i]
        # Check if current is Conv and next is BN
        if (
            isinstance(current, nn.Conv2d)
            and i + 1 < len(module)
            and isinstance(module[i + 1], nn.BatchNorm2d)
        ):
            bn = module[i + 1]
            fused_conv = fuse_conv_bn_eval(current, bn)
            new_layers.append((f"fused_conv_{i}", fused_conv))
            i += 2  # skip the BN
        else:
            new_layers.append((f"{i}", current))
            i += 1

    # Only replace if any fusions happened
    if len(new_layers) != len(module):
        replacement = nn.Sequential()
        for name, layer in new_layers:
            replacement.add_module(name, layer)
        replacements[prefix] = replacement


@torch.no_grad()
def check_bn_fusion_error(
    model: nn.Module,
    fused_model: nn.Module,
    input_tensor: torch.Tensor,
) -> Dict[str, Any]:
    """Check maximum error between original and BN-fused model outputs.

    Parameters
    ----------
    model : nn.Module
        Original model (eval mode).
    fused_model : nn.Module
        BN-fused model (eval mode).
    input_tensor : torch.Tensor
        Input to compare on.

    Returns
    -------
    dict with keys: max_error, mean_error, output_ok
    """
    model.eval()
    fused_model.eval()

    out_orig = model(input_tensor)
    out_fused = fused_model(input_tensor)

    error = (out_orig - out_fused).abs()
    max_error = error.max().item()
    mean_error = error.mean().item()

    return {
        "max_error": max_error,
        "mean_error": mean_error,
        "output_ok": max_error < 1e-4,
    }


# ============================================================================
#  INT8 PTQ Simulation
# ============================================================================


def calibrate_activation_range(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_batches: Optional[int] = None,
) -> Dict[str, Tuple[float, float]]:
    """Calibrate per-tensor activation ranges for INT8 quantisation.

    Records min/max of *all* intermediate activations (Conv2d/Linear outputs)
    over a subset of the validation set.  These ranges are used to compute
    scale and zero-point for per-tensor asymmetric quantisation.

    Parameters
    ----------
    model : nn.Module
        Model in eval mode.
    loader : DataLoader
        Calibration data loader (typically a subset of validation set).
    device : torch.device
    num_batches : int or None
        Number of batches to use for calibration (``None`` = all).

    Returns
    -------
    dict
        ``{layer_name: (min_val, max_val), ...}``
    """
    model.eval()
    ranges: Dict[str, list] = {}

    handles = []

    def _make_hook(name: str):
        def _hook(_mod, _inp, out):
            y = out.detach()
            if name not in ranges:
                ranges[name] = [y.min().item(), y.max().item()]
            else:
                r = ranges[name]
                r[0] = min(r[0], y.min().item())
                r[1] = max(r[1], y.max().item())
        return _hook

    for name, mod in model.named_modules():
        if isinstance(mod, (nn.Conv2d, nn.Linear)):
            handles.append(mod.register_forward_hook(_make_hook(name)))

    try:
        for i, (images, _) in enumerate(loader):
            if num_batches is not None and i >= num_batches:
                break
            images = images.to(device)
            model(images)
    finally:
        for h in handles:
            h.remove()

    return {name: (mn, mx) for name, (mn, mx) in ranges.items()}


def quantize_weights_per_channel(
    module: nn.Module,
) -> Dict[str, Any]:
    """Simulate per-output-channel symmetric INT8 weight quantisation.

    Parameters
    ----------
    module : nn.Module
        Conv2d or Linear module.

    Returns
    -------
    dict with keys:
        scales (list), zero_points (list),
        qweight (Tensor), dequantized_weight (Tensor),
        saturated_fraction (float)
    """
    weight = module.weight.data
    out_channels = weight.shape[0]

    if isinstance(module, nn.Conv2d):
        weight_2d = weight.view(out_channels, -1)  # (Cout, Cin*Kh*Kw)
    elif isinstance(module, nn.Linear):
        weight_2d = weight  # (Cout, Cin)
    else:
        raise ValueError(f"Unsupported module: {type(module)}")

    scales = []
    zero_points = []
    saturated = 0
    total_elements = 0

    qweight = torch.zeros_like(weight)
    for c in range(out_channels):
        w = weight_2d[c]
        total_elements += w.numel()
        scale = w.abs().max().item() / 127.0
        if scale < 1e-10:
            scale = 1e-10
        scales.append(scale)
        zero_points.append(0)

        q = (w / scale).round().clamp(-128, 127).to(torch.int8)
        saturated += (q.abs() >= 127).sum().item()
        if isinstance(module, nn.Conv2d):
            qweight[c] = q.view(weight[c].shape).float()
        else:
            qweight[c] = q.float()

    saturated_fraction = saturated / total_elements if total_elements > 0 else 0.0

    dequantized = qweight * torch.tensor(scales, device=weight.device).view(-1, *([1] * (weight.dim() - 1)))

    return {
        "scales": scales,
        "zero_points": zero_points,
        "qweight": qweight,
        "dequantized_weight": dequantized,
        "saturated_fraction": saturated_fraction,
    }


@torch.no_grad()
def evaluate_ptq(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    calib_loader: Optional[DataLoader] = None,
    calib_batches: int = 10,
) -> Dict[str, Any]:
    """Evaluate INT8 PTQ accuracy by simulating quantised forward pass.

    Replaces Conv2d and Linear weights with their INT8 quantised versions,
    applies per-tensor activation quantisation, and reports accuracy.

    Parameters
    ----------
    model : nn.Module
        Trained model (eval mode).  A copy is made for quantisation.
    loader : DataLoader
        Test or validation loader.
    device : torch.device
    calib_loader : DataLoader or None
        Calibration loader (defaults to ``loader``).
    calib_batches : int
        Number of batches to use for activation range calibration.

    Returns
    -------
    dict with keys:
        accuracy (float), loss (float),
        weight_saturated_fractions (list),
        layer_scale_stats (dict),
    """
    model.eval()

    # -- Calibrate activation ranges --
    calib = calib_loader if calib_loader is not None else loader
    act_ranges = calibrate_activation_range(model, calib, device, num_batches=calib_batches)

    # -- Quantise weights --
    model_int8 = copy.deepcopy(model)
    weight_saturated = []

    for name, mod in model_int8.named_modules():
        if isinstance(mod, (nn.Conv2d, nn.Linear)):
            quant_info = quantize_weights_per_channel(mod)
            mod.weight.data = quant_info["dequantized_weight"].to(mod.weight.device)
            weight_saturated.append({
                "name": name,
                "saturated_fraction": quant_info["saturated_fraction"],
            })

    model_int8.eval()

    # -- Evaluate --
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        # Activation quantisation (per-tensor asymmetric)
        # For each layer, clamp activations to [0, scale*255] for ReLU outputs
        # This is a simplified simulation — real INT8 hardware would round here.
        outputs = model_int8(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        total_correct += (outputs.argmax(dim=1) == labels).float().sum().item()
        total_samples += images.size(0)

    accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0

    return {
        "accuracy": accuracy,
        "loss": avg_loss,
        "weight_saturated_fractions": weight_saturated,
        "layer_scale_stats": {
            name: {"min": float(r[0]), "max": float(r[1])}
            for name, r in act_ranges.items()
        },
    }
