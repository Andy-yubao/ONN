"""Reproducibility utilities: seeding, deterministic mode, RNG state capture."""

import os
import random
import subprocess
import sys
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """Set random seed for all RNGs and optionally enable deterministic mode.

    Parameters
    ----------
    seed : int
        Base seed for Python ``random``, ``numpy``, and PyTorch CPU/CUDA RNGs.
    deterministic : bool
        If ``True``, also enables:

        - ``torch.backends.cudnn.deterministic = True``
        - ``torch.backends.cudnn.benchmark = False``
        - ``torch.use_deterministic_algorithms(True, warn_only=True)``

        This guarantees bitwise reproducibility for the same seed and hardware,
        at the cost of reduced performance (especially for cuDNN convolution
        selection).  Use for strict comparison experiments.

        If ``False``, cuDNN auto-tuner is enabled for best performance.
        Results are **not** guaranteed to be bitwise identical across runs
        even with the same seed, due to non-deterministic GPU algorithms.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except AttributeError:
            pass  # older PyTorch versions
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def seed_worker(worker_id: int) -> None:
    """Initialise a DataLoader worker's RNG with a seed based on the main seed.

    Use with ``torch.utils.data.DataLoader(..., worker_init_fn=seed_worker,
    generator=worker_generator)`` where ``worker_generator`` is a
    ``torch.Generator`` created with the experiment seed.

    See https://pytorch.org/docs/stable/notes/randomness.html#dataloader
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_worker_generator(seed: int) -> torch.Generator:
    """Return a ``torch.Generator`` for DataLoader worker seeding."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def capture_rng_state() -> Dict[str, Any]:
    """Capture current RNG states for saving in checkpoints.

    Returns
    -------
    dict
        With keys: ``python_random_state``, ``numpy_random_state``,
        ``torch_cpu_rng_state``, ``torch_cuda_rng_state`` (if available).
    """
    state: Dict[str, Any] = {
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_cpu_rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda_rng_state"] = torch.cuda.get_rng_state_all()
    return state


def load_rng_state(state: Dict[str, Any]) -> None:
    """Restore RNG states from a dict produced by ``capture_rng_state()``."""
    if "python_random_state" in state:
        random.setstate(state["python_random_state"])
    if "numpy_random_state" in state:
        np.random.set_state(state["numpy_random_state"])
    if "torch_cpu_rng_state" in state:
        torch.set_rng_state(state["torch_cpu_rng_state"])
    if "torch_cuda_rng_state" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda_rng_state"])


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


def capture_environment(deterministic: bool = False) -> dict[str, Any]:
    """Capture runtime environment metadata into a dict."""
    return {
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "torch_cuda_available": torch.cuda.is_available(),
        "torch_cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "torch_cudnn_version": (
            torch.backends.cudnn.version()
            if torch.backends.cudnn.is_available()
            else None
        ),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "git_commit": get_git_commit(),
        "seed_set": None,  # filled at runtime
        "deterministic": deterministic,
    }
