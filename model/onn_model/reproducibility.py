"""Reproducibility utilities: seeding, environment capture."""

import random
import os
import sys
import subprocess
from typing import Any

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility across all RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Disable cuDNN determinism for performance; results are still seed-reproducible
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def get_git_commit() -> str:
    """Return short SHA of current HEAD, or 'unknown'."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except FileNotFoundError:
        return "unknown"


def capture_environment() -> dict[str, Any]:
    """Capture runtime environment metadata into a dict."""
    return {
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "torch_cuda_available": torch.cuda.is_available(),
        "torch_cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "torch_cudnn_version": torch.backends.cudnn.version()
        if torch.backends.cudnn.is_available()
        else None,
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "git_commit": get_git_commit(),
        "seed_set": None,  # filled at runtime
    }
