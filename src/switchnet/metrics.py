from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ClassificationAccumulator:
    num_classes: int

    def __post_init__(self) -> None:
        self.confusion = torch.zeros(self.num_classes, self.num_classes, dtype=torch.long)
        self.loss_sum = 0.0
        self.examples = 0

    def update(self, logits: torch.Tensor, targets: torch.Tensor, loss: float) -> None:
        predictions = logits.argmax(dim=1).detach().cpu()
        truth = targets.detach().cpu()
        indices = truth * self.num_classes + predictions
        self.confusion += torch.bincount(
            indices, minlength=self.num_classes * self.num_classes
        ).reshape(self.num_classes, self.num_classes)
        self.loss_sum += float(loss) * len(targets)
        self.examples += len(targets)

    def compute(self) -> dict[str, float | list[float]]:
        cm = self.confusion.float()
        tp = cm.diag()
        support = cm.sum(dim=1)
        predicted = cm.sum(dim=0)
        recall = tp / support.clamp_min(1)
        precision = tp / predicted.clamp_min(1)
        f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
        active = support > 0
        return {
            "loss": self.loss_sum / max(1, self.examples),
            "accuracy": float(tp.sum() / cm.sum().clamp_min(1)),
            "macro_f1": float(f1[active].mean()) if active.any() else 0.0,
            "balanced_accuracy": float(recall[active].mean()) if active.any() else 0.0,
            "per_class_recall": recall.tolist(),
            "per_class_f1": f1.tolist(),
            "confusion_matrix": self.confusion.tolist(),
        }

