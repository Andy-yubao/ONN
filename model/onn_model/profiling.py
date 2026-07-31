"""Model profiling — parameter counts, MAC estimation, FPGA resource proxy."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn

from onn_model.metrics import count_parameters, count_trainable_parameters


class MACEstimator:
    """Estimate MACs for Conv2d and Linear layers via forward hooks.

    One MAC (multiply-accumulate) = one multiplication + one addition.

    * Only Conv2d and Linear count toward primary MACs.
    * BatchNorm, ReLU, pooling operations are excluded or noted separately.
    * Results are **estimates**, not post-synthesis FPGA resource usage.
    """

    def __init__(self) -> None:
        self.macs: int = 0
        self._handles: list = []
        self.layer_shapes: List[Dict[str, Any]] = []

    def _conv2d_macs(
        self,
        module: nn.Conv2d,
        input_shape: Tuple[int, ...],
        output_shape: Tuple[int, ...],
    ) -> int:
        """Estimate MACs for a Conv2d layer."""
        _, c_in, h_in, w_in = input_shape
        _, c_out, h_out, w_out = output_shape
        k_h, k_w = module.kernel_size
        groups = module.groups
        # MACs = Cout * Hout * Wout * (Cin/groups) * Kh * Kw  (bias negligible)
        macs_per_position = (c_in // groups) * k_h * k_w
        return c_out * h_out * w_out * macs_per_position

    def _linear_macs(self, module: nn.Linear, input_shape: Tuple[int, ...]) -> int:
        """Estimate MACs for a Linear layer."""
        batch_size = input_shape[0]
        in_features = module.in_features
        out_features = module.out_features
        return in_features * out_features  # per sample

    def _hook_fn(self, module: nn.Module, inp: Any, out: Any) -> None:
        input_t = inp[0]
        output_t = out if isinstance(out, torch.Tensor) else out[0]
        input_shape = tuple(input_t.shape)
        output_shape = tuple(output_t.shape)

        macs = 0
        if isinstance(module, nn.Conv2d):
            macs = self._conv2d_macs(module, input_shape, output_shape)
        elif isinstance(module, nn.Linear):
            macs = self._linear_macs(module, input_shape)

        self.macs += macs
        self.layer_shapes.append({
            "layer": str(module.__class__.__name__),
            "name": "",  # filled by profile_model
            "input_shape": input_shape,
            "output_shape": output_shape,
            "params": sum(p.numel() for p in module.parameters()),
            "macs": macs,
        })

    def profile(
        self, model: nn.Module, input_tensor: torch.Tensor
    ) -> List[Dict[str, Any]]:
        """Run profiling on ``model`` with ``input_tensor``.

        Registers hooks on Conv2d and Linear children only.

        Returns
        -------
        List of per-layer dicts with keys:
            layer, name, input_shape, output_shape, params, macs
        """
        self.macs = 0
        self.layer_shapes = []
        self._handles.clear()

        # Build a name-to-module mapping for richer output
        name_map = {mod: name for name, mod in model.named_modules()}

        # Use named_modules for naming
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                handle = module.register_forward_hook(self._make_named_hook(name, name_map))
                self._handles.append(handle)

        model.eval()
        with torch.no_grad():
            _ = model(input_tensor)

        self._remove_hooks()
        return self.layer_shapes

    def _make_named_hook(self, name: str, name_map: dict):
        """Create a hook that records the module name."""
        def hook_fn(module, inp, out):
            input_t = inp[0]
            output_t = out if isinstance(out, torch.Tensor) else out[0]
            input_shape = tuple(input_t.shape)
            output_shape = tuple(output_t.shape)

            macs = 0
            if isinstance(module, nn.Conv2d):
                macs = self._conv2d_macs(module, input_shape, output_shape)
            elif isinstance(module, nn.Linear):
                macs = self._linear_macs(module, input_shape)

            self.macs += macs
            self.layer_shapes.append({
                "layer": str(module.__class__.__name__),
                "name": name,
                "input_shape": input_shape,
                "output_shape": output_shape,
                "params": sum(p.numel() for p in module.parameters()),
                "macs": macs,
            })
        return hook_fn

    def _remove_hooks(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()


def compute_fpga_proxy(layer_details: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute FPGA resource proxy estimates from layer profiling data.

    Returns
    -------
    dict with keys:
        conv_params, linear_params,
        total_weight_bytes_int8, total_weight_bytes_int16,
        total_weight_bytes_fp32,
        peak_activation_elements, peak_activation_bytes_int8,
        peak_activation_bytes_int16, peak_activation_bytes_fp32,
        estimated_ping_pong_activation_bytes_int8,
    """
    conv_params = sum(d["params"] for d in layer_details if d["layer"] == "Conv2d")
    linear_params = sum(d["params"] for d in layer_details if d["layer"] == "Linear")

    # Weight storage at various bit widths (only Conv + Linear weights)
    total_weight_elements = sum(d["params"] for d in layer_details)

    # Peak activation: largest output tensor in elements
    peak_activation_elements = 0
    for d in layer_details:
        out_shape = d["output_shape"]
        vol = 1
        for s in out_shape:
            vol *= s
        if vol > peak_activation_elements:
            peak_activation_elements = vol

    # Ping-pong buffer estimate: conservative max(FM_input + FM_output) across layers
    # Uses the largest combined input+output volume (both at INT8)
    ping_pong_int8 = 0
    for d in layer_details:
        in_vol = 1
        for s in d["input_shape"]:
            in_vol *= s
        out_vol = 1
        for s in d["output_shape"]:
            out_vol *= s
        combined = in_vol + out_vol
        if combined > ping_pong_int8:
            ping_pong_int8 = combined

    return {
        "conv_params": conv_params,
        "linear_params": linear_params,
        "total_weight_bytes_int8": total_weight_elements * 1,
        "total_weight_bytes_int16": total_weight_elements * 2,
        "total_weight_bytes_fp32": total_weight_elements * 4,
        "peak_activation_elements": peak_activation_elements,
        "peak_activation_bytes_int8": peak_activation_elements * 1,
        "peak_activation_bytes_int16": peak_activation_elements * 2,
        "peak_activation_bytes_fp32": peak_activation_elements * 4,
        "estimated_ping_pong_activation_bytes_int8": ping_pong_int8,
    }


def profile_model(
    model: nn.Module, input_shape: Tuple[int, ...] = (1, 1, 28, 28)
) -> Dict[str, Any]:
    """Profile a model end-to-end with FPGA proxy estimates.

    Parameters
    ----------
    model : nn.Module
    input_shape : tuple
        (N, C, H, W) default (1, 1, 28, 28) for MNIST.

    Returns
    -------
    dict with keys:
        model_name, total_params, trainable_params, total_macs,
        layer_details, max_intermediate_elements, input_shape,
        fpga_proxy (dict with FPGA resource estimates)
    """
    device = next(model.parameters()).device
    dummy = torch.randn(input_shape, device=device)

    estimator = MACEstimator()
    layer_details = estimator.profile(model, dummy)

    # Max intermediate elements: largest conv/linear output volume
    max_intermediate = 0
    for d in layer_details:
        out_shape = d["output_shape"]
        vol = 1
        for s in out_shape:
            vol *= s
        if vol > max_intermediate:
            max_intermediate = vol

    total_params = count_parameters(model)
    trainable = count_trainable_parameters(model)
    total_macs = estimator.macs

    fpga_proxy = compute_fpga_proxy(layer_details)

    return {
        "model_name": type(model).__name__,
        "total_params": total_params,
        "trainable_params": trainable,
        "total_macs": total_macs,
        "max_intermediate_elements": max_intermediate,
        "input_shape": list(input_shape),
        "layer_details": layer_details,
        "fpga_proxy": fpga_proxy,
    }
