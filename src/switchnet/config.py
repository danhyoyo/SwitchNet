from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    name: str = "synthetic"
    root: str = "./data"
    num_classes: int = 3
    input_dim: int = 16
    train_size: int = 1500
    val_size: int = 300
    test_size: int = 600
    class_separation: float = 2.0
    noise_std: float = 1.0
    imbalance_ratio: float = 1.0
    batch_size: int = 128
    num_workers: int = 0
    download: bool = False


@dataclass
class ModelConfig:
    name: str = "switchnet_v1"
    feature_dim: int = 64
    encoder_width: int = 128
    expert_bottleneck: int = 16
    dropout: float = 0.0
    spec_margin: float = 1.0
    spec_weight: float = 0.2
    parameter_match_tolerance: float = 0.03
    gate_parameter_match: bool = True
    gate_max_hidden: int = 256


@dataclass
class TrainConfig:
    epochs: int = 30
    lr: float = 1.0e-3
    weight_decay: float = 1.0e-4
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    grad_clip: float | None = 5.0
    device: str = "auto"
    amp: bool = True
    seed: int = 42
    deterministic: bool = True
    checkpoint_every: int = 5
    early_stopping_patience: int = 0


@dataclass
class AnalysisConfig:
    enabled: bool = True
    gradient_batches: int = 1
    specialization: bool = True
    flops: bool = True


@dataclass
class LoggingConfig:
    output_dir: str = "runs"
    tensorboard: bool = True


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _merge_dataclass(instance: Any, values: dict[str, Any]) -> Any:
    known = set(instance.__dataclass_fields__)
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"Unknown config keys for {type(instance).__name__}: {sorted(unknown)}")
    for key, value in values.items():
        setattr(instance, key, value)
    return instance


def load_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    allowed = {"data", "model", "train", "analysis", "logging"}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"Unknown top-level config keys: {sorted(unknown)}")
    cfg = ExperimentConfig()
    for section in allowed:
        if section in raw:
            _merge_dataclass(getattr(cfg, section), raw[section] or {})
    return validate_config(cfg)


def apply_overrides(cfg: ExperimentConfig, overrides: list[str]) -> ExperimentConfig:
    """Apply CLI overrides such as ``train.epochs=2`` with YAML value parsing."""
    for item in overrides:
        if "=" not in item or "." not in item.split("=", 1)[0]:
            raise ValueError(f"Invalid override {item!r}; expected section.key=value")
        dotted, raw_value = item.split("=", 1)
        section, key = dotted.split(".", 1)
        if not hasattr(cfg, section) or key not in getattr(cfg, section).__dataclass_fields__:
            raise ValueError(f"Unknown override key: {dotted}")
        setattr(getattr(cfg, section), key, yaml.safe_load(raw_value))
    return validate_config(cfg)


def validate_config(cfg: ExperimentConfig) -> ExperimentConfig:
    """Fail early on invalid settings instead of failing deep in training."""
    if cfg.data.num_classes < 2:
        raise ValueError("data.num_classes must be at least 2")
    if cfg.data.input_dim < 1:
        raise ValueError("data.input_dim must be positive")
    for name in ("train_size", "val_size", "test_size", "batch_size"):
        if getattr(cfg.data, name) <= 0:
            raise ValueError(f"data.{name} must be positive")
    if cfg.data.noise_std <= 0:
        raise ValueError("data.noise_std must be positive")
    if cfg.data.imbalance_ratio < 1:
        raise ValueError("data.imbalance_ratio must be >= 1")
    if cfg.model.feature_dim < 1 or cfg.model.encoder_width < 1:
        raise ValueError("model dimensions must be positive")
    if cfg.model.expert_bottleneck < 1 or cfg.model.gate_max_hidden < 1:
        raise ValueError("expert_bottleneck and gate_max_hidden must be positive")
    if not 0 <= cfg.model.dropout < 1:
        raise ValueError("model.dropout must be in [0, 1)")
    if cfg.model.spec_margin < 0 or cfg.model.spec_weight < 0:
        raise ValueError("specialization margin and weight must be non-negative")
    if cfg.model.parameter_match_tolerance < 0:
        raise ValueError("model.parameter_match_tolerance must be non-negative")
    if cfg.train.epochs < 1 or cfg.train.lr <= 0 or cfg.train.weight_decay < 0:
        raise ValueError("invalid training epochs/lr/weight_decay")
    if cfg.train.checkpoint_every < 0 or cfg.train.early_stopping_patience < 0:
        raise ValueError("checkpoint and early-stopping intervals must be non-negative")
    if cfg.analysis.gradient_batches < 1:
        raise ValueError("analysis.gradient_batches must be positive")
    return cfg
