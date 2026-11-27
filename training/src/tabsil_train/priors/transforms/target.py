"""Regression target scaling with the train rows (`TrainRowTargetScale`).

The graph SCM standardizes a regression target over all rows of the table (after clipping it at 4
sds; `tabicl.prior._graph_scm`, `transforms.dependence.regression_target`). Inference standardizes
it with the train rows only (`tabsil.inference`), since the test targets are unknown there. This
transform makes the training tables match inference: the target is standardized again with the
mean and sd of the train rows.

Adapted from NVIDIA structured-data-models (Apache-2.0) `train/kumo/tabular/loader.py`
(`PriorDataset.__iter__`: the target minus the context mean, over the context sd with
`correction=0`, an sd below 1e-8 replaced by 1; `PriorDataset._sample_dataset`: a regression
table whose context sd is at most `min_regression_train_std`, 0.1 in every Kumo-Tabular regression
stage, is drawn again). See NOTICE.
"""

from __future__ import annotations

import torch

from tabsil_train.priors.pipeline import Group, Rng

# Kumo-Tabular's `--min-regression-train-std` (all regression stages) and its sd floor
MIN_TRAIN_STD = 0.1
STD_FLOOR = 1e-8


class TrainRowTargetScale:
    """Standardize each dataset's regression target with its train rows (mean and sd with ddof 0,
    as at inference). The group is dropped (drawn again) if a dataset's train-row sd, in the units
    of the incoming target (all rows standardized), is at most `min_train_std`: the scaled target
    would reach tens of sds. It draws no random numbers."""

    def __init__(self, min_train_std: float = MIN_TRAIN_STD) -> None:
        if min_train_std < 0:
            raise ValueError(f"min_train_std must be at least 0: {min_train_std}")
        self.min_train_std = min_train_std

    def __call__(self, group: Group, rng: Rng) -> Group | None:
        y = group.y.double()  # (rows, datasets)
        train = y[: group.n_train]
        mean = train.mean(0)
        std = train.std(0, correction=0)
        if (std <= self.min_train_std).any():
            return None
        std = torch.where(std < STD_FLOOR, torch.ones_like(std), std)
        return group.with_tensors(group.x, ((y - mean) / std).to(group.y.dtype))


def train_row_scale(y: torch.Tensor, n_train: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The train-row mean and sd (ddof 0, floor as `TrainRowTargetScale`) of batched targets
    y (B, R): two (B, 1) tensors. Used to validate a model trained with the scaling on tables drawn
    without it (`tabsil_train.training.loop._validate`)."""
    train = y[:, :n_train].double()
    mean = train.mean(1, keepdim=True)
    std = train.std(1, correction=0, keepdim=True)
    std = torch.where(std < STD_FLOOR, torch.ones_like(std), std)
    return mean.to(y.dtype), std.to(y.dtype)
