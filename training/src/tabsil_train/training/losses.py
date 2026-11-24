"""Prior-fitting losses: classification cross-entropy, regression pinball loss over the quantile
outputs."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from tabsil.models.quantile_distribution import pinball_loss


def prior_loss(logits: torch.Tensor, y_eval: torch.Tensor, task: str) -> torch.Tensor:
    """Mean loss over the eval rows. Classification: cross-entropy, regression: the pinball loss
    averaged over the quantile levels (`quantile_distribution`)."""
    logits = logits.float()
    if task == "classification":
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y_eval.reshape(-1).long())
    return pinball_loss(logits, y_eval.float()).mean()
