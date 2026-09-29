from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .models import ForwardOutput
from .utils import save_json


def _flat_grad(loss: torch.Tensor, parameters: list[nn.Parameter]) -> torch.Tensor:
    grads = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    pieces = [torch.zeros_like(p).flatten() if g is None else g.flatten() for p, g in zip(parameters, grads)]
    if not pieces:
        return torch.empty(0, device=loss.device)
    return torch.cat(pieces)


def _cosine_matrix(vectors: list[torch.Tensor | None]) -> np.ndarray:
    size = len(vectors)
    matrix = np.full((size, size), np.nan, dtype=np.float64)
    for i, left in enumerate(vectors):
        if left is None or left.numel() == 0 or left.norm() == 0:
            continue
        for j, right in enumerate(vectors):
            if right is None or right.numel() == 0 or right.norm() == 0:
                continue
            matrix[i, j] = float(F.cosine_similarity(left, right, dim=0))
    return matrix


def _json_matrix(matrix: np.ndarray) -> list[list[float | None]]:
    return [
        [float(value) if np.isfinite(value) else None for value in row]
        for row in matrix
    ]


def _binary_auroc(scores: np.ndarray, positives: np.ndarray) -> float | None:
    positive_count = int(positives.sum())
    negative_count = len(positives) - positive_count
    if positive_count == 0 or negative_count == 0:
        return None
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    sorted_positive = positives[order].astype(np.int64)
    threshold_ends = np.r_[np.flatnonzero(np.diff(sorted_scores)), len(scores) - 1]
    true_positive = np.cumsum(sorted_positive)[threshold_ends]
    false_positive = threshold_ends + 1 - true_positive
    tpr = np.r_[0.0, true_positive / positive_count]
    fpr = np.r_[0.0, false_positive / negative_count]
    integrate = getattr(np, "trapezoid", None)
    if integrate is None:  # NumPy < 2.0
        integrate = np.trapz
    return float(integrate(tpr, fpr))


def _expected_calibration_error(logits: torch.Tensor, targets: torch.Tensor, bins: int = 15) -> float:
    probabilities = logits.softmax(dim=1)
    confidence, predictions = probabilities.max(dim=1)
    correct = predictions.eq(targets)
    error = confidence.new_zeros(())
    boundaries = torch.linspace(0, 1, bins + 1, device=confidence.device)
    for index in range(bins):
        lower, upper = boundaries[index], boundaries[index + 1]
        selected = (confidence > lower) & (confidence <= upper)
        if selected.any():
            error += selected.float().mean() * (
                correct[selected].float().mean() - confidence[selected].mean()
            ).abs()
    return float(error)


def gradient_conflict_analysis(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    num_classes: int,
    device: torch.device,
    max_batches: int = 1,
) -> dict[str, Any]:
    """Class-wise cosine/conflict matrices in shared and private parameter spaces."""
    if not hasattr(model, "parameter_spaces"):
        raise TypeError("Model must implement parameter_spaces()")
    spaces = model.parameter_spaces()  # type: ignore[attr-defined]
    accumulated: dict[str, list[list[torch.Tensor]]] = {
        name: [[] for _ in range(num_classes)] for name in ("shared", "private")
    }
    was_training = model.training
    model.eval()
    for batch_idx, (inputs, targets) in enumerate(loader):
        if batch_idx >= max_batches:
            break
        inputs, targets = inputs.to(device), targets.to(device)
        for cls in targets.unique().tolist():
            # One representative gradient per class keeps CIFAR-100 analysis tractable.
            if accumulated["shared"][cls]:
                continue
            mask = targets == cls
            model.zero_grad(set_to_none=True)
            output: ForwardOutput = model(inputs[mask], targets[mask])
            loss = F.cross_entropy(output.logits, targets[mask])
            for name in ("shared", "private"):
                params = [p for p in spaces[name] if p.requires_grad]
                accumulated[name][cls].append(_flat_grad(loss, params).detach().cpu())
        if all(accumulated["shared"]):
            break
    model.train(was_training)

    result: dict[str, Any] = {}
    for name, class_lists in accumulated.items():
        vectors = [torch.stack(items).mean(0) if items else None for items in class_lists]
        cosine = _cosine_matrix(vectors)
        off_diag = ~np.eye(num_classes, dtype=bool)
        valid = off_diag & np.isfinite(cosine)
        result[name] = {
            "cosine_matrix": _json_matrix(cosine),
            "conflict_matrix": np.where(np.isfinite(cosine), cosine < 0, False).astype(int).tolist(),
            "mean_off_diagonal_cosine": float(cosine[valid].mean()) if valid.any() else None,
            "conflict_rate": float((cosine[valid] < 0).mean()) if valid.any() else None,
            "classes_observed": [idx for idx, vector in enumerate(vectors) if vector is not None],
        }
    return result


@torch.no_grad()
def specialization_analysis(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    num_classes: int,
    device: torch.device,
) -> dict[str, Any] | None:
    if not hasattr(model, "experts"):
        return None
    score_sums = torch.zeros(num_classes, num_classes, dtype=torch.float64)
    activation_sums = torch.zeros(num_classes, num_classes, dtype=torch.float64)
    counts = torch.zeros(num_classes, dtype=torch.long)
    routing_correct = 0
    gate_routing_correct = 0
    gate_total = 0
    total = 0
    all_scores: list[torch.Tensor] = []
    all_logits: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    model.eval()
    for inputs, targets in loader:
        inputs = inputs.to(device)
        output: ForwardOutput = model(inputs, None)
        scores = output.scores.detach().cpu().double()
        activations_raw = output.expert_activations
        assert activations_raw is not None
        activations = activations_raw.detach().cpu().double().norm(dim=2)
        targets_cpu = targets.cpu()
        all_scores.append(scores)
        all_logits.append(output.logits.detach().cpu())
        all_targets.append(targets_cpu)
        routing_correct += int((scores.argmax(dim=1) == targets_cpu).sum())
        if output.gate_weights is not None:
            gate_routing_correct += int(
                (output.gate_weights.detach().cpu().argmax(dim=1) == targets_cpu).sum()
            )
            gate_total += len(targets)
        total += len(targets)
        for cls in targets_cpu.unique().tolist():
            mask = targets_cpu == cls
            score_sums[cls] += scores[mask].sum(dim=0)
            activation_sums[cls] += activations[mask].sum(dim=0)
            counts[cls] += int(mask.sum())
    mean_scores = score_sums / counts.clamp_min(1).unsqueeze(1)
    mean_activations = activation_sums / counts.clamp_min(1).unsqueeze(1)
    selectivity = []
    activation_selectivity = []
    for expert in range(num_classes):
        other = torch.arange(num_classes) != expert
        selectivity.append(float(mean_scores[expert, expert] - mean_scores[other, expert].mean()))
        activation_selectivity.append(
            float(mean_activations[expert, expert] - mean_activations[other, expert].mean())
        )
    scores_array = torch.cat(all_scores).numpy()
    targets_array = torch.cat(all_targets).numpy()
    positive_score_mean = []
    negative_score_mean = []
    one_vs_rest_auroc = []
    for expert in range(num_classes):
        positive = targets_array == expert
        negative = ~positive
        positive_score_mean.append(
            float(scores_array[positive, expert].mean()) if positive.any() else None
        )
        negative_score_mean.append(
            float(scores_array[negative, expert].mean()) if negative.any() else None
        )
        one_vs_rest_auroc.append(_binary_auroc(scores_array[:, expert], positive))
    logits_cpu = torch.cat(all_logits)
    targets_cpu = torch.cat(all_targets)
    target_scores = logits_cpu.gather(1, targets_cpu[:, None]).squeeze(1)
    non_target_scores = logits_cpu.masked_fill(
        F.one_hot(targets_cpu, num_classes).bool(), -torch.inf
    ).max(dim=1).values
    valid_aurocs = [value for value in one_vs_rest_auroc if value is not None]
    return {
        "mean_score_by_true_class_and_expert": mean_scores.tolist(),
        "mean_activation_by_true_class_and_expert": mean_activations.tolist(),
        "score_selectivity_per_expert": selectivity,
        "mean_score_selectivity": float(np.mean(selectivity)),
        "activation_selectivity_per_expert": activation_selectivity,
        "mean_activation_selectivity": float(np.mean(activation_selectivity)),
        "expert_routing_accuracy": routing_correct / max(1, total),
        "gate_routing_accuracy": gate_routing_correct / gate_total if gate_total else None,
        "positive_score_mean_per_expert": positive_score_mean,
        "negative_score_mean_per_expert": negative_score_mean,
        "one_vs_rest_auroc_per_expert": one_vs_rest_auroc,
        "mean_one_vs_rest_auroc": float(np.mean(valid_aurocs)) if valid_aurocs else None,
        "mean_target_vs_best_other_logit_gap": float(
            (target_scores - non_target_scores).mean()
        ),
        "expected_calibration_error_15_bins": _expected_calibration_error(
            logits_cpu, targets_cpu
        ),
        "activation_definition": "L2 norm of the pre-LayerNorm expert adapter residual",
    }


def estimate_flops(model: nn.Module, input_shape: tuple[int, ...], device: torch.device) -> int | None:
    try:
        from thop import profile
    except ImportError:
        return None
    was_training = model.training
    model.eval()
    sample = torch.zeros((1, *input_shape), device=device)
    flops, _ = profile(model, inputs=(sample,), verbose=False)
    model.train(was_training)
    return int(flops)


def save_gradient_csv(result: dict[str, Any], directory: str | Path) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for space, values in result.items():
        matrix = values["cosine_matrix"]
        with (directory / f"gradient_cosine_{space}.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(["class"] + list(range(len(matrix))))
            for idx, row in enumerate(matrix):
                writer.writerow([idx] + row)
    save_json(directory / "gradient_analysis.json", result)
