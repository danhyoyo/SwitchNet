from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import nn

from .config import ModelConfig
from .utils import count_parameters

MODEL_NAMES = (
    "dense",
    "multi_branch",
    "switchnet_v1",
    "switchnet_v2",
    "switchnet_gate",
)


class MLPEncoder(nn.Module):
    def __init__(self, input_dim: int, width: int, feature_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, width),
            nn.LayerNorm(width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.flatten(1))


class CIFAREncoder(nn.Module):
    """Small shared CNN; experts remain lightweight even for CIFAR-100."""

    def __init__(self, width: int, feature_dim: int, dropout: float) -> None:
        super().__init__()
        c3 = max(64, width)
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, c3, 3, padding=1, bias=False),
            nn.BatchNorm2d(c3),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.projection = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(c3, feature_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(self.features(x))


def build_encoder(input_shape: tuple[int, ...], cfg: ModelConfig) -> nn.Module:
    if len(input_shape) == 1:
        return MLPEncoder(input_shape[0], cfg.encoder_width, cfg.feature_dim, cfg.dropout)
    if input_shape == (3, 32, 32):
        return CIFAREncoder(cfg.encoder_width, cfg.feature_dim, cfg.dropout)
    raise ValueError(f"Unsupported input shape: {input_shape}")


class EvidenceExpert(nn.Module):
    """Residual bottleneck adapter plus a scalar class-evidence readout."""

    def __init__(self, feature_dim: int, bottleneck: int, dropout: float) -> None:
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(feature_dim, bottleneck),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck, feature_dim),
        )
        self.norm = nn.LayerNorm(feature_dim)
        self.evidence = nn.Linear(feature_dim, 1)

    def forward(
        self, z: torch.Tensor, private_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if private_mask is not None:
            return self.controlled_forward(z, private_mask)
        return self.forward_with_activation(z)

    def forward_with_activation(
        self, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = self.adapter(z)
        h = self.norm(z + residual)
        return h, self.evidence(h).squeeze(-1), residual

    def controlled_forward(
        self, z: torch.Tensor, private_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Mask per-example parameter gradients while retaining all input gradients.

        The adapter and its normalization are class-private. The evidence readout
        must see both positive and negative examples to calibrate its class score.
        Both paths still differentiate with respect to ``z``. Reusing one dropout
        mask makes the two paths numerically identical.
        """
        down = self.adapter[0]
        dropout = self.adapter[2]
        up = self.adapter[3]
        assert isinstance(down, nn.Linear)
        assert isinstance(dropout, nn.Dropout)
        assert isinstance(up, nn.Linear)

        dropout_scale = None
        if self.training and dropout.p > 0:
            shape = (z.shape[0], down.out_features)
            dropout_scale = z.new_empty(shape).bernoulli_(1.0 - dropout.p).div_(1.0 - dropout.p)

        def path(detached: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            def maybe_detach(value: torch.Tensor | None) -> torch.Tensor | None:
                return value.detach() if detached and value is not None else value

            hidden = F.linear(z, maybe_detach(down.weight), maybe_detach(down.bias))
            hidden = F.gelu(hidden)
            if dropout_scale is not None:
                hidden = hidden * dropout_scale
            residual = F.linear(hidden, maybe_detach(up.weight), maybe_detach(up.bias))
            normalized = F.layer_norm(
                z + residual,
                self.norm.normalized_shape,
                maybe_detach(self.norm.weight),
                maybe_detach(self.norm.bias),
                self.norm.eps,
            )
            score = F.linear(
                normalized,
                self.evidence.weight,
                self.evidence.bias,
            ).squeeze(-1)
            return normalized, score, residual

        trainable = path(detached=False)
        frozen = path(detached=True)
        mask = private_mask.to(dtype=z.dtype)

        def select(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
            expanded = mask
            while expanded.ndim < left.ndim:
                expanded = expanded.unsqueeze(-1)
            return left * expanded + right * (1.0 - expanded)

        return tuple(select(left, right) for left, right in zip(trainable, frozen))  # type: ignore[return-value]


@dataclass
class ForwardOutput:
    logits: torch.Tensor
    scores: torch.Tensor
    expert_features: torch.Tensor | None = None
    expert_activations: torch.Tensor | None = None
    gate_weights: torch.Tensor | None = None


class DenseBaseline(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        feature_dim: int,
        hidden_width: int,
        num_classes: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_width, num_classes),
        )
        self.hidden_width = hidden_width
        self.match_metadata: dict[str, float | int] = {}

    def forward(self, x: torch.Tensor, targets: torch.Tensor | None = None) -> ForwardOutput:
        del targets
        logits = self.classifier(self.encoder(x))
        return ForwardOutput(logits=logits, scores=logits)

    def parameter_spaces(self) -> dict[str, list[nn.Parameter]]:
        return {
            "shared": list(self.encoder.parameters()),
            "private": list(self.classifier.parameters()),
            "integration": [],
        }


class MultiExpertModel(nn.Module):
    """Common implementation for multi-branch and SwitchNet variants.

    Every expert is evaluated for every sample. Isolated variants use trainable and
    detached-parameter paths with identical values. This zeros non-target adapter
    gradients while preserving gradients to the shared encoder and evidence heads.
    """

    def __init__(
        self,
        encoder: nn.Module,
        feature_dim: int,
        bottleneck: int,
        num_classes: int,
        dropout: float,
        isolate_private_gradients: bool,
        use_gate: bool,
        use_specialization_loss: bool,
        gate_hidden: int | None = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.experts = nn.ModuleList(
            [EvidenceExpert(feature_dim, bottleneck, dropout) for _ in range(num_classes)]
        )
        self.num_classes = num_classes
        self.isolate_private_gradients = isolate_private_gradients
        self.use_gate = use_gate
        self.use_specialization_loss = use_specialization_loss
        self.match_metadata: dict[str, float | int | bool] = {}
        if use_gate:
            gate_input_dim = num_classes * feature_dim + num_classes
            gate_hidden = gate_hidden or max(16, min(256, gate_input_dim // 2))
            self.gate = nn.Sequential(
                nn.Linear(gate_input_dim, gate_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(gate_hidden, num_classes),
            )
            self.integration_head = nn.Linear(feature_dim, num_classes)
        else:
            self.gate = None
            self.integration_head = None

    def forward(self, x: torch.Tensor, targets: torch.Tensor | None = None) -> ForwardOutput:
        z = self.encoder(x)
        features, scores, activations = [], [], []
        for expert_idx, expert in enumerate(self.experts):
            private_mask = (
                targets == expert_idx
                if self.isolate_private_gradients and targets is not None
                else None
            )
            h, s, activation = expert(z, private_mask)
            features.append(h)
            scores.append(s)
            activations.append(activation)
        feature_tensor = torch.stack(features, dim=1)  # [B, C, D]
        score_tensor = torch.stack(scores, dim=1)  # [B, C]
        activation_tensor = torch.stack(activations, dim=1)
        if self.use_gate:
            assert self.gate is not None and self.integration_head is not None
            gate_input = torch.cat([feature_tensor.flatten(1), score_tensor], dim=1)
            weights = torch.softmax(self.gate(gate_input), dim=1)
            fused = (feature_tensor * weights.unsqueeze(-1)).sum(dim=1)
            # Residual evidence keeps direct all-expert comparison in the prediction.
            logits = self.integration_head(fused) + score_tensor
        else:
            weights = None
            logits = score_tensor
        return ForwardOutput(
            logits=logits,
            scores=score_tensor,
            expert_features=feature_tensor,
            expert_activations=activation_tensor,
            gate_weights=weights,
        )

    def parameter_spaces(self) -> dict[str, list[nn.Parameter]]:
        integration: list[nn.Parameter] = [
            p for expert in self.experts for p in expert.evidence.parameters()
        ]
        if self.gate is not None:
            integration.extend(self.gate.parameters())
        if self.integration_head is not None:
            integration.extend(self.integration_head.parameters())
        return {
            "shared": list(self.encoder.parameters()),
            "private": [
                p
                for expert in self.experts
                for module in (expert.adapter, expert.norm)
                for p in module.parameters()
            ],
            "integration": integration,
        }


def _dense_classifier_parameter_count(feature_dim: int, width: int, num_classes: int) -> int:
    return feature_dim * width + width + width * num_classes + num_classes


def _expert_parameter_count(feature_dim: int, bottleneck: int, num_classes: int) -> int:
    per_expert = 2 * feature_dim * bottleneck + bottleneck + 4 * feature_dim + 1
    return num_classes * per_expert


def _gate_non_encoder_parameter_count(
    feature_dim: int, bottleneck: int, num_classes: int, gate_hidden: int
) -> int:
    gate_input = num_classes * feature_dim + num_classes
    gate = gate_input * gate_hidden + gate_hidden + gate_hidden * num_classes + num_classes
    integration = feature_dim * num_classes + num_classes
    return _expert_parameter_count(feature_dim, bottleneck, num_classes) + gate + integration


def _best_gate_dimensions(
    target: int,
    feature_dim: int,
    max_bottleneck: int,
    num_classes: int,
    max_hidden: int,
    tolerance: float,
) -> tuple[int, int, int]:
    candidates = [
        (
            abs(actual - target) / max(1, target),
            abs(_expert_parameter_count(feature_dim, bottleneck, num_classes) / target - 0.6),
            -hidden,
            bottleneck,
            hidden,
            actual,
        )
        for bottleneck in range(1, max_bottleneck + 1)
        for hidden in range(1, max_hidden + 1)
        for actual in [
            _gate_non_encoder_parameter_count(feature_dim, bottleneck, num_classes, hidden)
        ]
    ]
    feasible = [candidate for candidate in candidates if candidate[0] <= tolerance]
    if feasible:
        # Reserve roughly 40% of the non-encoder budget for a meaningful gate instead
        # of winning the count match with a degenerate one-unit gate.
        _, _, _, bottleneck, hidden, actual = min(
            feasible, key=lambda item: (item[1], item[0], item[2])
        )
    else:
        _, _, _, bottleneck, hidden, actual = min(candidates)
    return bottleneck, hidden, actual


def _best_dense_width(target_private_params: int, feature_dim: int, num_classes: int) -> int:
    estimate = max(1, round((target_private_params - num_classes) / (feature_dim + num_classes + 1)))
    candidates = range(max(1, estimate - 8), estimate + 9)
    return min(
        candidates,
        key=lambda width: abs(
            _dense_classifier_parameter_count(feature_dim, width, num_classes) - target_private_params
        ),
    )


def _multi_expert(
    name: str,
    cfg: ModelConfig,
    input_shape: tuple[int, ...],
    num_classes: int,
) -> MultiExpertModel:
    variants = {
        "multi_branch": (False, False, False),
        "switchnet_v1": (True, False, False),
        "switchnet_v2": (True, False, True),
        "switchnet_gate": (True, True, True),
    }
    isolate, gate, spec = variants[name]
    bottleneck = cfg.expert_bottleneck
    gate_hidden = None
    match_metadata: dict[str, float | int | bool] = {}
    if gate and cfg.gate_parameter_match:
        target = _expert_parameter_count(
            cfg.feature_dim, cfg.expert_bottleneck, num_classes
        )
        bottleneck, gate_hidden, actual = _best_gate_dimensions(
            target,
            cfg.feature_dim,
            cfg.expert_bottleneck,
            num_classes,
            cfg.gate_max_hidden,
            cfg.parameter_match_tolerance,
        )
        match_metadata = {
            "parameter_matched": True,
            "target_non_encoder_parameters": target,
            "actual_non_encoder_parameters": actual,
            "effective_expert_bottleneck": bottleneck,
            "effective_gate_hidden": gate_hidden,
            "expert_budget_fraction": _expert_parameter_count(
                cfg.feature_dim, bottleneck, num_classes
            )
            / target,
            "relative_error": abs(actual - target) / max(1, target),
        }
    model = MultiExpertModel(
        build_encoder(input_shape, cfg),
        cfg.feature_dim,
        bottleneck,
        num_classes,
        cfg.dropout,
        isolate,
        gate,
        spec,
        gate_hidden,
    )
    model.match_metadata = match_metadata
    return model


def build_model(
    name: str,
    cfg: ModelConfig,
    input_shape: tuple[int, ...],
    num_classes: int,
) -> nn.Module:
    name = name.lower()
    if name not in MODEL_NAMES:
        raise ValueError(f"Unknown model {name!r}; choose from {MODEL_NAMES}")
    if name != "dense":
        return _multi_expert(name, cfg, input_shape, num_classes)

    # Match all V1 non-encoder parameters, including the evidence readouts.
    reference = _multi_expert("switchnet_v1", cfg, input_shape, num_classes)
    target_non_encoder = sum(p.numel() for expert in reference.experts for p in expert.parameters())
    width = _best_dense_width(target_non_encoder, cfg.feature_dim, num_classes)
    model = DenseBaseline(
        build_encoder(input_shape, cfg), cfg.feature_dim, width, num_classes, cfg.dropout
    )
    actual_non_encoder = sum(p.numel() for p in model.classifier.parameters())
    target_total = count_parameters(reference)
    actual_total = count_parameters(model)
    model.match_metadata = {
        "target_non_encoder_parameters": target_non_encoder,
        "actual_non_encoder_parameters": actual_non_encoder,
        "target_total_parameters": target_total,
        "actual_total_parameters": actual_total,
        "relative_error": abs(actual_total - target_total) / max(1, target_total),
    }
    return model


def named_trainable(parameters: Iterable[nn.Parameter]) -> list[nn.Parameter]:
    return [p for p in parameters if p.requires_grad]
