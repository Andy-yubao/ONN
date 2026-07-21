"""Metrics and statistics utilities."""

from typing import Optional

import torch


def accuracy(output: torch.Tensor, target: torch.Tensor, topk: int = 1) -> float:
    """Compute top-k accuracy.

    Parameters
    ----------
    output : Tensor
        Model output logits, shape ``(N, num_classes)``.
    target : Tensor
        Ground-truth class indices, shape ``(N,)``.
    topk : int
        ``k`` for top-k accuracy (default 1).

    Returns
    -------
    float
        Accuracy in range [0, 1].
    """
    with torch.no_grad():
        batch_size = target.size(0)
        if batch_size == 0:
            return 0.0
        _, pred = output.topk(topk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        return correct.reshape(-1).float().sum().item() / batch_size


def finite_loss(loss: torch.Tensor) -> bool:
    """Check that loss is finite (not NaN or Inf)."""
    return bool(torch.isfinite(loss).all())


def count_parameters(model: torch.nn.Module) -> int:
    """Return total number of parameters."""
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model: torch.nn.Module) -> int:
    """Return number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
