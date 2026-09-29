from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


class CSVLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = [
            "epoch",
            "lr",
            "train_loss",
            "train_classification_loss",
            "train_specialization_loss",
            "train_accuracy",
            "train_macro_f1",
            "val_loss",
            "val_classification_loss",
            "val_specialization_loss",
            "val_accuracy",
            "val_macro_f1",
            "epoch_seconds",
        ]
        if self.path.exists():
            with self.path.open("r", newline="", encoding="utf-8") as handle:
                existing_fields = csv.DictReader(handle).fieldnames
            if existing_fields != self.fieldnames:
                raise ValueError(
                    f"Metrics schema differs in {self.path}; use a new run name for this version."
                )
        else:
            with self.path.open("w", newline="", encoding="utf-8") as handle:
                csv.DictWriter(handle, fieldnames=self.fieldnames).writeheader()

    def log(self, row: dict[str, Any]) -> None:
        with self.path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
            writer.writerow({key: row.get(key) for key in self.fieldnames})


class TensorBoardLogger:
    def __init__(self, directory: str | Path, enabled: bool) -> None:
        self.writer = None
        if enabled:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(str(directory))
            except ImportError:
                print("TensorBoard is not installed; continuing with CSV logging only.")

    def log(self, epoch: int, metrics: dict[str, Any]) -> None:
        if self.writer is None:
            return
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(key, value, epoch)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
