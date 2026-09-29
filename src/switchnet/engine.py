from __future__ import annotations

import csv
import random
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import numpy as np
from torch import nn

from .analysis import (
    estimate_flops,
    gradient_conflict_analysis,
    save_gradient_csv,
    specialization_analysis,
)
from .config import ExperimentConfig
from .data import DataBundle
from .logging_utils import CSVLogger, TensorBoardLogger
from .losses import specialization_margin_loss
from .metrics import ClassificationAccumulator
from .models import ForwardOutput
from .utils import count_parameters, environment_metadata, save_json


def _numpy_rng_state() -> dict[str, Any]:
    name, keys, position, has_gaussian, cached_gaussian = np.random.get_state()
    return {
        "name": name,
        "keys": keys.tolist(),
        "position": position,
        "has_gaussian": has_gaussian,
        "cached_gaussian": cached_gaussian,
    }


def _restore_numpy_rng_state(state: dict[str, Any]) -> None:
    np.random.set_state(
        (
            state["name"],
            np.asarray(state["keys"], dtype=np.uint32),
            int(state["position"]),
            int(state["has_gaussian"]),
            float(state["cached_gaussian"]),
        )
    )


def _criterion(
    output: ForwardOutput, targets: torch.Tensor, model: nn.Module, cfg: ExperimentConfig
) -> tuple[torch.Tensor, dict[str, float]]:
    classification = F.cross_entropy(output.logits, targets)
    spec = output.logits.new_zeros(())
    if getattr(model, "use_specialization_loss", False):
        spec = specialization_margin_loss(output.scores, targets, cfg.model.spec_margin)
    total = classification + cfg.model.spec_weight * spec
    return total, {"classification_loss": float(classification.detach()), "specialization_loss": float(spec.detach())}


def _run_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    cfg: ExperimentConfig,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None,
    phase: str = "evaluation",
) -> tuple[dict[str, Any], float]:
    training = optimizer is not None
    model.train(training)
    accumulator = ClassificationAccumulator(cfg.data.num_classes)
    classification_sum = 0.0
    specialization_sum = 0.0
    started = time.perf_counter()
    for batch_index, (inputs, targets) in enumerate(loader, start=1):
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        amp_enabled = scaler is not None and scaler.is_enabled()
        with torch.set_grad_enabled(training), torch.autocast(
            device_type=device.type, enabled=amp_enabled
        ):
            # Labels control gradients only in training. Evaluation is label-free.
            output: ForwardOutput = model(inputs, targets if training else None)
            loss, components = _criterion(output, targets, model, cfg)
        if not torch.isfinite(loss).item():
            raise FloatingPointError(
                f"Non-finite {phase} loss at batch {batch_index}: "
                f"classification={components['classification_loss']}, "
                f"specialization={components['specialization_loss']}. "
                "Stop this run and inspect the last finite checkpoint."
            )
        if training:
            assert optimizer is not None
            if scaler is not None:
                scaler.scale(loss).backward()
                if cfg.train.grad_clip is not None:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if cfg.train.grad_clip is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
                optimizer.step()
        accumulator.update(output.logits, targets, float(loss.detach()))
        classification_sum += components["classification_loss"] * len(targets)
        specialization_sum += components["specialization_loss"] * len(targets)
    metrics = accumulator.compute()
    metrics["classification_loss"] = classification_sum / max(1, accumulator.examples)
    metrics["specialization_loss"] = specialization_sum / max(1, accumulator.examples)
    return metrics, time.perf_counter() - started


def _optimizer(model: nn.Module, cfg: ExperimentConfig) -> torch.optim.Optimizer:
    name = cfg.train.optimizer.lower()
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=cfg.train.lr,
            weight_decay=cfg.train.weight_decay,
            momentum=0.9,
            nesterov=True,
        )
    raise ValueError(f"Unknown optimizer: {cfg.train.optimizer}")


def _scheduler(
    optimizer: torch.optim.Optimizer, cfg: ExperimentConfig
) -> torch.optim.lr_scheduler.LRScheduler | None:
    name = cfg.train.scheduler.lower()
    if name in {"none", "off"}:
        return None
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, cfg.train.epochs))
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, cfg.train.epochs // 3), gamma=0.1)
    raise ValueError(f"Unknown scheduler: {cfg.train.scheduler}")


def _save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    epoch: int,
    best_metric: float,
    best_model_state: dict[str, torch.Tensor] | None,
    data: DataBundle,
    training_seconds: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "training_semantics_version": 2,
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "best_metric": best_metric,
            "best_model": best_model_state,
            "training_seconds": training_seconds,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "python_rng_state": random.getstate(),
            "numpy_rng_state": _numpy_rng_state(),
            "loader_generator_states": {
                name: loader.generator.get_state()
                for name, loader in (("train", data.train), ("val", data.val), ("test", data.test))
                if loader.generator is not None
            },
        },
        path,
    )


def _load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    device: torch.device,
    data: DataBundle,
) -> tuple[int, float, dict[str, torch.Tensor], float]:
    # weights_only avoids executing arbitrary pickle payloads from untrusted checkpoints.
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if getattr(model, "isolate_private_gradients", False) and checkpoint.get(
        "training_semantics_version", 1
    ) < 2:
        raise ValueError(
            "This checkpoint used the old positive-only evidence gradient rule. "
            "Start a new run for the corrected model."
        )
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if checkpoint.get("torch_rng_state") is not None:
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
    if torch.cuda.is_available() and checkpoint.get("cuda_rng_state") is not None:
        torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint["cuda_rng_state"]])
    if checkpoint.get("python_rng_state") is not None:
        random.setstate(checkpoint["python_rng_state"])
    if checkpoint.get("numpy_rng_state") is not None:
        _restore_numpy_rng_state(checkpoint["numpy_rng_state"])
    loader_states = checkpoint.get("loader_generator_states", {})
    for name, loader in (("train", data.train), ("val", data.val), ("test", data.test)):
        if loader.generator is not None and name in loader_states:
            loader.generator.set_state(loader_states[name].cpu())
    best_model = checkpoint.get("best_model") or {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }
    return (
        int(checkpoint["epoch"]) + 1,
        float(checkpoint.get("best_metric", -float("inf"))),
        best_model,
        float(checkpoint.get("training_seconds", 0.0)),
    )


def train_experiment(
    model: nn.Module,
    data: DataBundle,
    cfg: ExperimentConfig,
    run_dir: str | Path,
    device: torch.device,
    resume: str | Path | None = None,
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    model = model.to(device)
    optimizer = _optimizer(model, cfg)
    scheduler = _scheduler(optimizer, cfg)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.train.amp and device.type == "cuda")
    csv_logger = CSVLogger(run_dir / "metrics.csv")
    tb_logger = TensorBoardLogger(run_dir / "tensorboard", cfg.logging.tensorboard)
    start_epoch, best_metric = 1, -float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    total_training_seconds = 0.0
    if resume is not None:
        start_epoch, best_metric, best_state, total_training_seconds = _load_checkpoint(
            resume, model, optimizer, scheduler, device, data
        )
    epochs_without_improvement = 0

    for epoch in range(start_epoch, cfg.train.epochs + 1):
        train_metrics, seconds = _run_epoch(
            model, data.train, device, cfg, optimizer, scaler, f"training epoch {epoch}"
        )
        val_metrics, _ = _run_epoch(model, data.val, device, cfg, None, None, f"validation epoch {epoch}")
        total_training_seconds += seconds
        lr = optimizer.param_groups[0]["lr"]
        row = {
            "epoch": epoch,
            "lr": lr,
            "train_loss": train_metrics["loss"],
            "train_classification_loss": train_metrics["classification_loss"],
            "train_specialization_loss": train_metrics["specialization_loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_loss": val_metrics["loss"],
            "val_classification_loss": val_metrics["classification_loss"],
            "val_specialization_loss": val_metrics["specialization_loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "epoch_seconds": seconds,
        }
        csv_logger.log(row)
        tb_logger.log(epoch, {key.replace("_", "/", 1): value for key, value in row.items()})
        print(
            f"epoch={epoch:03d} train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f}"
        )
        if float(val_metrics["macro_f1"]) > best_metric:
            best_metric = float(val_metrics["macro_f1"])
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
            is_best = True
        else:
            epochs_without_improvement += 1
            is_best = False
        # Store scheduler state after stepping so resume starts with the exact next-epoch LR.
        if scheduler is not None:
            scheduler.step()
        if is_best:
            _save_checkpoint(
                run_dir / "checkpoints" / "best.pt",
                model,
                optimizer,
                scheduler,
                epoch,
                best_metric,
                best_state,
                data,
                total_training_seconds,
            )
        if cfg.train.checkpoint_every > 0 and epoch % cfg.train.checkpoint_every == 0:
            _save_checkpoint(
                run_dir / "checkpoints" / f"epoch_{epoch:04d}.pt",
                model,
                optimizer,
                scheduler,
                epoch,
                best_metric,
                best_state,
                data,
                total_training_seconds,
            )
        if (
            cfg.train.early_stopping_patience > 0
            and epochs_without_improvement >= cfg.train.early_stopping_patience
        ):
            break
    tb_logger.close()
    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics, test_seconds = _run_epoch(model, data.test, device, cfg, None, None)
    gradient_result = None
    specialization_result = None
    if cfg.analysis.enabled:
        gradient_result = gradient_conflict_analysis(
            model,
            data.train,
            data.num_classes,
            device,
            cfg.analysis.gradient_batches,
        )
        save_gradient_csv(gradient_result, run_dir / "analysis")
        if cfg.analysis.specialization:
            specialization_result = specialization_analysis(model, data.test, data.num_classes, device)
            if specialization_result is not None:
                save_json(run_dir / "analysis" / "specialization.json", specialization_result)
    flops = (
        estimate_flops(model, data.input_shape, device)
        if cfg.analysis.enabled and cfg.analysis.flops
        else None
    )
    parameters = count_parameters(model)
    summary: dict[str, Any] = {
        "model": cfg.model.name,
        "dataset": cfg.data.name,
        "seed": cfg.train.seed,
        "parameters": parameters,
        "flops_per_example": flops,
        "training_seconds": total_training_seconds,
        "test_seconds": test_seconds,
        "best_val_macro_f1": best_metric,
        "test": test_metrics,
        "class_counts": data.class_counts,
        "device": str(device),
        "environment": environment_metadata(),
        "parameter_match": getattr(model, "match_metadata", None),
        "config": cfg.to_dict(),
        "gradient_analysis": gradient_result,
        "specialization": specialization_result,
    }
    save_json(run_dir / "summary.json", summary)
    return summary


def write_combined_summary(path: str | Path, summaries: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "model",
        "dataset",
        "seed",
        "parameters",
        "flops_per_example",
        "training_seconds",
        "best_val_macro_f1",
        "test_accuracy",
        "test_macro_f1",
        "test_balanced_accuracy",
        "mean_score_selectivity",
        "mean_one_vs_rest_auroc",
        "calibration_error",
        "shared_conflict_rate",
        "private_conflict_rate",
    ]
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    if path.exists():
        with path.open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                rows[(row["model"], row["seed"])] = row
    for summary in summaries:
        spec = summary.get("specialization") or {}
        gradient = summary.get("gradient_analysis") or {}
        row = {
            "model": summary["model"],
            "dataset": summary["dataset"],
            "seed": summary["seed"],
            "parameters": summary["parameters"],
            "flops_per_example": summary["flops_per_example"],
            "training_seconds": summary["training_seconds"],
            "best_val_macro_f1": summary["best_val_macro_f1"],
            "test_accuracy": summary["test"]["accuracy"],
            "test_macro_f1": summary["test"]["macro_f1"],
            "test_balanced_accuracy": summary["test"]["balanced_accuracy"],
            "mean_score_selectivity": spec.get("mean_score_selectivity"),
            "mean_one_vs_rest_auroc": spec.get("mean_one_vs_rest_auroc"),
            "calibration_error": spec.get("expected_calibration_error_15_bins"),
            "shared_conflict_rate": gradient.get("shared", {}).get("conflict_rate"),
            "private_conflict_rate": gradient.get("private", {}).get("conflict_rate"),
        }
        rows[(str(row["model"]), str(row["seed"]))] = row
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows.values())
