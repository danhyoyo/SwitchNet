from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset

from .config import DataConfig
from .utils import worker_seed_fn


@dataclass
class DataBundle:
    train: DataLoader
    val: DataLoader
    test: DataLoader
    input_shape: tuple[int, ...]
    num_classes: int
    class_counts: list[int]


class TargetSubset(Subset):
    """Pickle-safe subset carrying local targets for long-tail resampling."""

    def __init__(self, dataset: Dataset, indices: list[int], targets: np.ndarray) -> None:
        super().__init__(dataset, indices)
        self.targets = targets[indices].tolist()


def _class_counts(total: int, num_classes: int, imbalance_ratio: float) -> list[int]:
    if imbalance_ratio < 1.0:
        raise ValueError("imbalance_ratio must be >= 1 (largest/smallest class)")
    if num_classes == 1:
        return [total]
    weights = np.array(
        [imbalance_ratio ** (-i / (num_classes - 1)) for i in range(num_classes)],
        dtype=np.float64,
    )
    exact = total * weights / weights.sum()
    counts = np.floor(exact).astype(int)
    counts = np.maximum(counts, 1)
    while counts.sum() < total:
        counts[int(np.argmax(exact - counts))] += 1
    while counts.sum() > total:
        candidates = np.where(counts > 1)[0]
        counts[candidates[int(np.argmin(exact[candidates] - counts[candidates]))]] -= 1
    return counts.tolist()


def make_gaussian_dataset(
    size: int,
    num_classes: int,
    input_dim: int,
    separation: float,
    noise_std: float,
    imbalance_ratio: float,
    seed: int,
) -> tuple[TensorDataset, list[int]]:
    """Gaussian classes with means on orthogonal/circular directions.

    ``separation / noise_std`` controls overlap. Test/validation should normally be
    balanced so macro metrics are not coupled to the training prior.
    """
    if num_classes < 2:
        raise ValueError("num_classes must be at least 2")
    if size < num_classes:
        raise ValueError("dataset size must be at least num_classes")
    rng = np.random.default_rng(seed)
    counts = _class_counts(size, num_classes, imbalance_ratio)
    means = np.zeros((num_classes, input_dim), dtype=np.float32)
    if input_dim >= num_classes:
        means[:, :num_classes] = np.eye(num_classes, dtype=np.float32) * separation
        means -= means.mean(axis=0, keepdims=True)
    elif input_dim == 1:
        # A circular projection aliases classes in one dimension; evenly spaced means do not.
        means[:, 0] = np.linspace(-separation, separation, num_classes, dtype=np.float32)
    else:
        angles = np.linspace(0, 2 * math.pi, num_classes, endpoint=False)
        means[:, 0] = np.cos(angles) * separation
        if input_dim > 1:
            means[:, 1] = np.sin(angles) * separation
    xs, ys = [], []
    for cls, count in enumerate(counts):
        xs.append(rng.normal(means[cls], noise_std, size=(count, input_dim)).astype(np.float32))
        ys.append(np.full(count, cls, dtype=np.int64))
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    order = rng.permutation(len(y))
    return TensorDataset(torch.from_numpy(x[order]), torch.from_numpy(y[order])), counts


def _targets(dataset: Dataset) -> np.ndarray:
    if hasattr(dataset, "targets"):
        return np.asarray(dataset.targets)
    raise TypeError("Dataset must expose a targets attribute for long-tail subsampling")


def make_long_tail_subset(
    dataset: Dataset, num_classes: int, imbalance_ratio: float, seed: int
) -> tuple[Subset, list[int]]:
    targets = _targets(dataset)
    max_count = min(int((targets == c).sum()) for c in range(num_classes))
    desired = [
        max(1, int(max_count * imbalance_ratio ** (-c / max(1, num_classes - 1))))
        for c in range(num_classes)
    ]
    rng = np.random.default_rng(seed)
    indices: list[int] = []
    for cls, count in enumerate(desired):
        candidates = np.flatnonzero(targets == cls)
        indices.extend(rng.choice(candidates, size=min(count, len(candidates)), replace=False).tolist())
    rng.shuffle(indices)
    return Subset(dataset, indices), desired


def _loader(dataset: Dataset, cfg: DataConfig, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=cfg.num_workers > 0,
        worker_init_fn=worker_seed_fn,
        generator=generator,
    )


def build_dataloaders(cfg: DataConfig, seed: int) -> DataBundle:
    name = cfg.name.lower()
    if name == "synthetic":
        train, counts = make_gaussian_dataset(
            cfg.train_size,
            cfg.num_classes,
            cfg.input_dim,
            cfg.class_separation,
            cfg.noise_std,
            cfg.imbalance_ratio,
            seed,
        )
        val, _ = make_gaussian_dataset(
            cfg.val_size,
            cfg.num_classes,
            cfg.input_dim,
            cfg.class_separation,
            cfg.noise_std,
            1.0,
            seed + 1,
        )
        test, _ = make_gaussian_dataset(
            cfg.test_size,
            cfg.num_classes,
            cfg.input_dim,
            cfg.class_separation,
            cfg.noise_std,
            1.0,
            seed + 2,
        )
        shape = (cfg.input_dim,)
        num_classes = cfg.num_classes
    elif name in {"cifar10", "cifar100"}:
        from torchvision import datasets, transforms

        dataset_cls = datasets.CIFAR10 if name == "cifar10" else datasets.CIFAR100
        num_classes = 10 if name == "cifar10" else 100
        if cfg.num_classes != num_classes:
            cfg.num_classes = num_classes
        if name == "cifar10":
            mean = (0.4914, 0.4822, 0.4465)
            std = (0.2470, 0.2435, 0.2616)
        else:
            mean = (0.5071, 0.4867, 0.4408)
            std = (0.2675, 0.2565, 0.2761)
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        eval_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        full_train_aug = dataset_cls(cfg.root, train=True, transform=train_transform, download=cfg.download)
        full_train_eval = dataset_cls(cfg.root, train=True, transform=eval_transform, download=cfg.download)
        test = dataset_cls(cfg.root, train=False, transform=eval_transform, download=cfg.download)
        targets = _targets(full_train_aug)
        rng = np.random.default_rng(seed)
        train_indices, val_indices = [], []
        for cls in range(num_classes):
            cls_indices = np.flatnonzero(targets == cls)
            rng.shuffle(cls_indices)
            n_val = max(1, int(0.1 * len(cls_indices)))
            val_indices.extend(cls_indices[:n_val].tolist())
            train_indices.extend(cls_indices[n_val:].tolist())
        train_base: Dataset = Subset(full_train_aug, train_indices)
        if cfg.imbalance_ratio > 1.0:
            # Local targets are materialized for deterministic second-stage sampling.
            wrapped = TargetSubset(full_train_aug, train_indices, targets)
            lt_local, counts = make_long_tail_subset(wrapped, num_classes, cfg.imbalance_ratio, seed)
            train_base = lt_local
        else:
            counts = [int((targets[train_indices] == c).sum()) for c in range(num_classes)]
        train = train_base
        val = Subset(full_train_eval, val_indices)
        shape = (3, 32, 32)
    else:
        raise ValueError(f"Unknown dataset {cfg.name!r}")

    return DataBundle(
        train=_loader(train, cfg, True, seed),
        val=_loader(val, cfg, False, seed + 1),
        test=_loader(test, cfg, False, seed + 2),
        input_shape=shape,
        num_classes=num_classes,
        class_counts=counts,
    )
