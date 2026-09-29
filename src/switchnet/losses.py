from __future__ import annotations

import torch
import torch.nn.functional as F


def specialization_margin_loss(scores: torch.Tensor, targets: torch.Tensor, margin: float) -> torch.Tensor:
    """Require the target expert evidence to exceed every non-target expert."""
    target_scores = scores.gather(1, targets[:, None])
    margins = margin - target_scores + scores
    margins = margins.masked_fill(F.one_hot(targets, scores.shape[1]).bool(), 0.0)
    return torch.clamp_min(margins, 0.0).sum(dim=1).mean() / max(1, scores.shape[1] - 1)

