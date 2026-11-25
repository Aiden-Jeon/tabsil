"""Feature value transforms: coarse values (`Coarsen`) and measurement noise (`MeasurementNoise`).

Both are options of the graph SCM prior (off by default and not in a registered prior yet). They
change only x, so the regression borders do not depend on them.
"""

from __future__ import annotations

import numpy as np
import torch

from tabsil_train.priors.pipeline import Group, Rng
from tabsil_train.priors.transforms.categorical import log_uniform_int


def coarsen(values: np.ndarray, r: np.random.Generator) -> np.ndarray:
    """30-100% of the columns reduced to 2-8 ordered levels (`Coarsen`)."""
    n, f = values.shape
    out = values.copy()
    count = max(1, round(min(r.uniform(0.3, 1.3), 1.0) * f))
    grid = r.random() < 0.5  # one kind per table
    for column in r.permutation(f)[:count]:
        v = values[:, column]
        if not np.isfinite(v).all() or n < 2:
            continue
        bins = log_uniform_int(r, 2, 8)
        coarse = None
        if grid:
            # round to a step, then censor to `bins` consecutive integers (+0.0: one zero)
            step = np.exp(r.uniform(np.log(0.25), np.log(2.0)))
            low = -(bins // 2)
            coarse = np.clip(np.round(v / step), low, low + bins - 1) + 0.0
            if (coarse == coarse[0]).all():
                coarse = None  # censoring left one value: use quantile bins instead
        if coarse is None:
            # equal-frequency bins by rank (ties in random order), each replaced by its mean
            rank = np.empty(n, dtype=np.int64)
            rank[np.lexsort((r.random(n), v))] = np.arange(n)
            index = rank * bins // n
            means = np.bincount(index, weights=v, minlength=bins) / np.maximum(
                np.bincount(index, minlength=bins), 1
            )
            coarse = means[index]
        out[:, column] = coarse
    return out


class Coarsen:
    """With probability `dataset_prob` per dataset, reduce 30-100% of the columns to 2-8 ordered
    levels: rounded to a regular grid (step LogUniform(0.25, 2) of the standardized values,
    censored to the level count) or equal-frequency bins replaced by their mean, one kind per
    table. Like rounded measurements, Likert scales or binned ages.

    Adapted from NVIDIA structured-data-models (Apache-2.0)
    `sdm/prior/tabular/transform/coarsen.py` `Coarsen` (Kumo-Tabular, 10% of tables in stage 2
    and 3); see NOTICE. sdm correlates the level counts and steps within a table; here they are
    drawn per column.
    """

    def __init__(self, dataset_prob: float) -> None:
        if not 0.0 <= dataset_prob <= 1.0:
            raise ValueError(f"dataset_prob must be between 0 and 1: {dataset_prob}")
        self.dataset_prob = dataset_prob

    def __call__(self, group: Group, rng: Rng) -> Group:
        r = rng.numpy
        xs = []
        for d in range(group.x.shape[1]):
            x = group.x[:, d]
            if r.random() < self.dataset_prob:
                out = coarsen(x.double().cpu().numpy(), r)
                x = torch.from_numpy(out).to(dtype=x.dtype, device=x.device)
            xs.append(x)
        return group.with_tensors(torch.stack(xs, 1), group.y)


class MeasurementNoise:
    """With probability `dataset_prob` per dataset, every column with probability
    `column_prob` becomes a noisy measurement of itself: sqrt(w)·x + sqrt(1-w)·std(x)·ε,
    signal share w ~ U(0, 1), ε ~ N(0, 1).

    Kumo-Tabular draws about 10% of its numeric columns from a distribution around the SCM signal
    (sdm `DistributionalMixtureConverter`, `dist_num_conv_prob=0.1`, signal share U(0, 1)) inside
    its SCM. The TabICL SCM has no such converter, so this adds the noise to the finished column
    instead (children in the graph still see the clean value, as with a measured variable).
    """

    def __init__(self, dataset_prob: float, column_prob: float = 0.1) -> None:
        for name, p in (("dataset_prob", dataset_prob), ("column_prob", column_prob)):
            if not 0.0 <= p <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1: {p}")
        self.dataset_prob = dataset_prob
        self.column_prob = column_prob

    def __call__(self, group: Group, rng: Rng) -> Group:
        r = rng.numpy
        xs = []
        for d in range(group.x.shape[1]):
            x = group.x[:, d]
            if r.random() < self.dataset_prob:
                values = x.double().cpu().numpy().copy()
                n, f = values.shape
                columns = np.flatnonzero(r.random(f) < self.column_prob)
                for column in columns:
                    v = values[:, column]
                    w = r.uniform(0.0, 1.0)
                    std = np.nanstd(v) if np.isfinite(v).any() else 0.0
                    values[:, column] = np.sqrt(w) * v + np.sqrt(1 - w) * std * (
                        r.standard_normal(n)
                    )
                x = torch.from_numpy(values).to(dtype=x.dtype, device=x.device)
            xs.append(x)
        return group.with_tensors(torch.stack(xs, 1), group.y)
