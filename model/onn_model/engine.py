"""Training engine — single-epoch loops, checkpoint I/O, experiment runner."""

import csv
import json
import os
import shutil
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from onn_model.metrics import accuracy, finite_loss, count_parameters, count_trainable_parameters
from onn_model.reproducibility import (
    set_seed,
    capture_environment,
    capture_rng_state,
    load_rng_state,
    seed_worker,
    make_worker_generator,
)


@dataclass
class ExperimentState:
    """In-memory experiment state that gets dumped to disk at the end."""

    config: Dict[str, Any] = field(default_factory=dict)
    environment: Dict[str, Any] = field(default_factory=dict)
    history: Dict[str, list] = field(default_factory=lambda: {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    })
    best_val_acc: float = 0.0
    best_val_loss: float = float("inf")
    best_epoch: int = -1


@dataclass
class DataBundle:
    """Container for pre-created data loaders (used by tests to inject synthetic data).

    When passed to :func:`run_experiment`, the experiment uses these loaders
    instead of creating MNIST-based loaders.  This allows tests to run without
    downloading MNIST.
    """

    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: Optional[DataLoader] = None


def _resolve_device(device_str: str) -> torch.device:
    """Resolve ``auto`` to CUDA or CPU."""
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def _get_model(model_name: str, num_classes: int = 10) -> nn.Module:
    """Factory: return a model instance by name."""
    if model_name == "BaselineCNN":
        from onn_model.models.baseline_cnn import BaselineCNN

        return BaselineCNN(num_classes=num_classes)
    elif model_name == "TinyResNet":
        from onn_model.models.tiny_resnet import TinyResNet

        return TinyResNet(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def _get_optimizer(model: nn.Module, config: Dict[str, Any]) -> optim.Optimizer:
    name = config.get("optimizer", "AdamW")
    lr = config.get("learning_rate", 0.001)
    wd = config.get("weight_decay", 0.0001)
    if name == "AdamW":
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif name == "Adam":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif name == "SGD":
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    else:
        raise ValueError(f"Unsupported optimizer: {name}")


def _get_scheduler(
    optimizer: optim.Optimizer, config: Dict[str, Any], steps: int
) -> Optional[object]:
    """Return a scheduler object with a ``step()`` method, or ``None``."""
    name = config.get("scheduler", "CosineAnnealingLR")
    epochs = config.get("epochs", 15)
    if name == "CosineAnnealingLR":
        return lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif name == "StepLR":
        step_size = config.get("scheduler_step", 10)
        gamma = config.get("scheduler_gamma", 0.1)
        return lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif name is None or name == "":
        return None
    else:
        raise ValueError(f"Unsupported scheduler: {name}")


def _save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Optional[object],
    scaler: Optional[torch.cuda.amp.GradScaler],
    epoch: int,
    state: ExperimentState,
    is_best: bool,
) -> None:
    """Save a full training checkpoint including all resumable state."""
    checkpoint: Dict[str, Any] = {
        "model_name": type(model).__name__,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "config": state.config,
        "epoch": epoch,
        "best_val_acc": state.best_val_acc,
        "best_val_loss": state.best_val_loss,
        "best_epoch": state.best_epoch,
        "history": {k: v[:] for k, v in state.history.items()},
        "rng_state": capture_rng_state(),
    }
    if scaler is not None:
        checkpoint["scaler_state_dict"] = scaler.state_dict()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)
    if is_best:
        best_path = path.parent / "best.pt"
        shutil.copy2(path, best_path)


def _save_run_artifacts(run_dir: Path, state: ExperimentState, config: Dict[str, Any]) -> None:
    """Save config, environment, history, and metrics to disk."""
    run_dir.mkdir(parents=True, exist_ok=True)

    # Config snapshot
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    # Environment snapshot
    with open(run_dir / "environment.json", "w", encoding="utf-8") as f:
        json.dump(state.environment, f, indent=2, ensure_ascii=False)

    # Training history as CSV
    csv_path = run_dir / "history.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
        for i in range(len(state.history["epoch"])):
            writer.writerow([
                state.history["epoch"][i],
                state.history["train_loss"][i],
                state.history["train_acc"][i],
                state.history["val_loss"][i],
                state.history["val_acc"][i],
            ])

    # Summary metrics
    total_params = count_parameters(_get_model(config.get("model", "BaselineCNN")))
    metrics_dict = {
        "best_val_acc": state.best_val_acc,
        "best_val_loss": state.best_val_loss,
        "best_epoch": state.best_epoch,
        "total_params": total_params,
        "best_checkpoint": str(run_dir / "best.pt"),
    }
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=2, ensure_ascii=False)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    max_batches: Optional[int] = None,
) -> Tuple[float, float]:
    """Run one training epoch.

    Parameters
    ----------
    max_batches : int or None
        If set, only process this many batches per epoch (for smoke tests).

    Returns
    -------
    (avg_loss, avg_acc)
    """
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch_index, (images, labels) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break

        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        if scaler is not None and device.type == "cuda":
            with torch.amp.autocast("cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        total_correct += accuracy(outputs, labels) * images.size(0)
        total_samples += images.size(0)

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    avg_acc = total_correct / total_samples if total_samples > 0 else 0.0
    return avg_loss, avg_acc


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> Tuple[float, float]:
    """Run one validation epoch.

    Parameters
    ----------
    max_batches : int or None
        If set, only process this many batches (for smoke tests).

    Returns
    -------
    (avg_loss, avg_acc)
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch_index, (images, labels) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break

        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        total_correct += accuracy(outputs, labels) * images.size(0)
        total_samples += images.size(0)

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    avg_acc = total_correct / total_samples if total_samples > 0 else 0.0
    return avg_loss, avg_acc


def run_experiment(
    config: Dict[str, Any],
    run_dir: Optional[Path] = None,
    smoke_test: bool = False,
    resume_checkpoint: Optional[Path] = None,
    eval_only: bool = False,
    override_kwargs: Optional[Dict[str, Any]] = None,
    data_bundle: Optional[DataBundle] = None,
) -> ExperimentState:
    """Run a full training experiment.

    Parameters
    ----------
    config : dict
        Experiment configuration.
    run_dir : Path or None
        Output directory (auto-generated if ``None``).
    smoke_test : bool
        If ``True``, run at most 2 training batches + 2 validation batches
        per epoch, for at most 2 epochs.
    resume_checkpoint : Path or None
        Path to a ``.pt`` checkpoint to resume from.  The checkpoint must
        have been saved by ``_save_checkpoint`` (full state).
    eval_only : bool
        If ``True``, only evaluate the model at ``resume_checkpoint``.
    override_kwargs : dict or None
        Per-key overrides applied to ``config`` before running.
    data_bundle : DataBundle or None
        Pre-created loaders for injecting synthetic data (tests).  When
        provided, MNIST-based loaders are **not** created.

    Returns
    -------
    ExperimentState
    """
    if override_kwargs:
        config = {**config, **override_kwargs}

    # Seeding (must happen before any DataLoader creation)
    seed = config.get("seed", 42)
    deterministic = config.get("deterministic", False)
    set_seed(seed, deterministic=deterministic)

    # Device
    device = _resolve_device(config.get("device", "auto"))

    # Data
    if data_bundle is not None:
        train_loader = data_bundle.train_loader
        val_loader = data_bundle.val_loader
        test_loader = data_bundle.test_loader
    else:
        from onn_model.data import get_train_val_loaders, get_test_loader

        data_root = config.get("data_root", "model/data")
        try:
            train_loader, val_loader, _ = get_train_val_loaders(
                root=data_root,
                batch_size=config.get("batch_size", 128),
                seed=seed,
                num_workers=config.get("num_workers", 0),
            )
            test_loader = get_test_loader(
                root=data_root,
                batch_size=config.get("batch_size", 128),
                num_workers=config.get("num_workers", 0),
            )
        except Exception as e:
            raise RuntimeError(f"Failed to create data loaders: {e}")

    # Smoke-test limits
    max_train_batches: Optional[int] = None
    max_val_batches: Optional[int] = None
    if smoke_test:
        max_train_batches = 2
        max_val_batches = 2

    # Model
    model = _get_model(config.get("model", "BaselineCNN")).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = _get_optimizer(model, config)
    scheduler = _get_scheduler(optimizer, config, len(train_loader))
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    # State
    state = ExperimentState(config=config)

    # Checkpoint resume — must happen before run_dir creation so we can
    # reuse the original run directory
    start_epoch = 0
    resume_run_dir: Optional[Path] = None
    if resume_checkpoint and resume_checkpoint.exists():
        ckpt = torch.load(resume_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        # Restore scheduler
        if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
            try:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            except Exception:
                pass  # benign if scheduler type changed

        # Restore scaler
        if scaler is not None and ckpt.get("scaler_state_dict") is not None:
            try:
                scaler.load_state_dict(ckpt["scaler_state_dict"])
            except Exception:
                pass

        # Restore training state
        start_epoch = ckpt.get("epoch", -1) + 1
        state.best_val_acc = ckpt.get("best_val_acc", 0.0)
        state.best_val_loss = ckpt.get("best_val_loss", float("inf"))
        state.best_epoch = ckpt.get("best_epoch", -1)
        state.history = ckpt.get(
            "history",
            {"epoch": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []},
        )

        # Restore RNG state
        if "rng_state" in ckpt:
            load_rng_state(ckpt["rng_state"])

        # Reuse the checkpoint's run directory by default
        resume_run_dir = resume_checkpoint.parent
        print(
            f"Resumed from epoch {ckpt.get('epoch', '?')} at {resume_checkpoint}  "
            f"(best_val_acc={state.best_val_acc:.4f})"
        )

    # Environment capture (after seeding so deterministic flag is set)
    state.environment = capture_environment(deterministic=deterministic)
    state.environment["seed_set"] = seed

    # Run directory: reuse resume dir by default unless user specified --run-dir
    if run_dir is None:
        if resume_run_dir is not None:
            run_dir = resume_run_dir
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = config.get("model", "model")
            run_dir = Path("model/runs") / model_name / ts
    run_dir.mkdir(parents=True, exist_ok=True)

    # Eval-only mode
    if eval_only:
        print("Eval-only mode: computing test metrics...")
        test_loss, test_acc = validate_one_epoch(model, test_loader, criterion, device)
        print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
        return state

    # Training loop
    epochs = config.get("epochs", 15)
    if smoke_test:
        epochs = min(epochs, 2)

    print(f"Training for {epochs} epoch(s), starting from epoch {start_epoch + 1}")
    if max_train_batches is not None:
        print(f"  max_train_batches={max_train_batches}  max_val_batches={max_val_batches}")

    for epoch in range(start_epoch, epochs):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scaler,
            max_batches=max_train_batches,
        )
        val_loss, val_acc = validate_one_epoch(
            model,
            val_loader,
            criterion,
            device,
            max_batches=max_val_batches,
        )

        if scheduler is not None:
            scheduler.step()

        # Record
        state.history["epoch"].append(epoch + 1)
        state.history["train_loss"].append(train_loss)
        state.history["train_acc"].append(train_acc)
        state.history["val_loss"].append(val_loss)
        state.history["val_acc"].append(val_acc)

        # Track best
        improved = (val_acc > state.best_val_acc) or (
            val_acc == state.best_val_acc and val_loss < state.best_val_loss
        )
        if improved:
            state.best_val_acc = val_acc
            state.best_val_loss = val_loss
            state.best_epoch = epoch + 1

        print(
            f"Epoch {epoch+1:2d}/{epochs}  "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}  "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}  "
            f"{'*' if improved else ' '}"
        )

        # Save checkpoints (full state including scheduler, scaler, history, RNG)
        last_path = run_dir / "last.pt"
        _save_checkpoint(
            last_path,
            model,
            optimizer,
            scheduler,
            scaler,
            epoch,
            state,
            is_best=improved,
        )

    # Finalize
    _save_run_artifacts(run_dir, state, config)
    print(f"\nResults saved to {run_dir}")
    print(f"Best Val Acc: {state.best_val_acc:.4f} (epoch {state.best_epoch})")

    return state


@torch.no_grad()
def evaluate_checkpoint(
    checkpoint_path: Path,
    config: Dict[str, Any],
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Load a checkpoint and evaluate on the MNIST test set.

    Parameters
    ----------
    checkpoint_path : Path
        Path to ``best.pt`` or ``last.pt``.
    config : dict
        Config matching the checkpoint.
    device : torch.device or None

    Returns
    -------
    dict with keys: test_loss, test_acc, model_name, checkpoint
    """
    if device is None:
        device = _resolve_device(config.get("device", "auto"))

    from onn_model.data import get_test_loader

    data_root = config.get("data_root", "model/data")
    test_loader = get_test_loader(
        root=data_root,
        batch_size=config.get("batch_size", 128),
        num_workers=config.get("num_workers", 0),
    )

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_name = ckpt.get("model_name", config.get("model", "BaselineCNN"))
    model = _get_model(model_name).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = validate_one_epoch(model, test_loader, criterion, device)

    results = {
        "test_loss": test_loss,
        "test_acc": test_acc,
        "model_name": model_name,
        "checkpoint": str(checkpoint_path),
    }
    print(f"Test  Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
    return results
