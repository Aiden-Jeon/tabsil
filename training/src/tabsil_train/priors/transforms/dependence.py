"""Transforms that change how columns depend on each other and on the target: correlated column
blocks (`CorrelateColumns`) and a target recomputed from a few columns (`Relabel`).

Both are options of the graph SCM prior, off by default. They run on the finished SCM table, before
`CoverClasses`, so the class order of the rows still follows the final target.

- `CorrelateColumns` changes only x.
- `Relabel` changes y. The regression borders are therefore drawn with it (`graph_scm.make_borders`
  with `relabel_prob`).

Adapted from NVIDIA structured-data-models (Apache-2.0) `sdm/prior/tabular/transform/correlate.py`
`CorrelateColumns` and `relabel.py` `Relabel` (Kumo-Tabular: 30% and 10% of tables); see NOTICE.
The differences are listed in each class.
"""

from __future__ import annotations

import numpy as np
import torch

from tabsil_train.priors.pipeline import Group, Rng
from tabsil_train.priors.transforms.categorical import log_uniform_int, power_law_weights

# A column with at most this many distinct values is treated like a categorical column: blending
# keeps its values and their counts. This is the TabICL SCM's largest categorical size with
# `use_corrected_cat_sizes` (`sample_categorical_sizes`, `max_cat_size=100`). The TabICL SCM does
# not say which of its columns are categorical, so discretized numeric columns with few values are
# also kept discrete here (sdm knows the column types and blends those linearly).
DISCRETE_LEVELS = 100
# TabICL's regression target processing (`tabicl.prior._graph_scm`: `outlier_removing`, then
# `standard_scaling`)
OUTLIER_SDS = 4.0


def _standardize(v: np.ndarray) -> np.ndarray:
    std = v.std()
    return (v - v.mean()) / (std if std > 1e-12 else 1.0)


def blend_discrete(column: np.ndarray, latent: np.ndarray, rho: float, r: np.random.Generator):
    """A discrete column moved toward `latent` without changing its values or their counts (sdm
    `_discretize_correlated_column`): the levels get a random order, the level scores are mixed
    with the latent (weights rho and sqrt(1 - rho²)), and the rows sorted by the mixed score take
    the column's values in level order."""
    values, inverse, counts = np.unique(column, return_inverse=True, return_counts=True)
    if len(values) < 2:
        return column
    order = r.permutation(len(values))
    rank = np.empty(len(values), dtype=np.int64)
    rank[order] = np.arange(len(values))
    score = _standardize(rank[inverse].astype(np.float64))
    mixed = rho * latent + np.sqrt(max(1.0 - rho * rho, 0.0)) * score
    out = np.empty_like(column)
    out[np.argsort(mixed, kind="stable")] = np.repeat(values[order], counts[order])
    return out


def correlate(values: np.ndarray, r: np.random.Generator) -> np.ndarray:
    """2 to all columns, split into 1-4 blocks; each block of two or more columns is moved toward
    its own standard normal factor with strength rho ~ U(0.2, 0.7) (`CorrelateColumns`)."""
    n, f = values.shape
    out = values.copy()
    if f < 2:
        return out
    k = int(r.integers(2, f + 1))
    columns = r.permutation(f)[:k]
    n_blocks = int(r.integers(1, 5))
    block_of = r.integers(0, n_blocks, size=k)
    for b in range(n_blocks):
        block = columns[block_of == b]
        if len(block) < 2:
            continue
        rho = r.uniform(0.2, 0.7)
        latent = r.standard_normal(n)
        for column in block:
            v = out[:, column]
            if len(np.unique(v)) <= DISCRETE_LEVELS:
                out[:, column] = blend_discrete(v, latent, rho, r)
            else:
                # sdm blends without rescaling and standardizes all columns afterwards
                # (`ScaleFeatures`); the TabICL columns are standardized already, so only the
                # blended ones are standardized again
                out[:, column] = _standardize((1 - rho) * v + rho * latent)
    return out


class CorrelateColumns:
    """With probability `dataset_prob` per dataset, blend random feature blocks toward shared
    Gaussian factors (`correlate`). Real tables have groups of related measurements (lab panels,
    survey items, sensor arrays) that the SCM graph alone rarely makes.

    Adapted from sdm `CorrelateColumns` (Kumo-Tabular: 30% of tables, in every stage). sdm
    preserves categorical columns by their declared type; here a column with at most
    `DISCRETE_LEVELS` distinct values is preserved (`DISCRETE_LEVELS`).
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
            if r.random() < self.dataset_prob and torch.isfinite(x).all():
                out = correlate(x.double().cpu().numpy(), r)
                x = torch.from_numpy(out).to(dtype=x.dtype, device=x.device)
            xs.append(x)
        return group.with_tensors(torch.stack(xs, 1), group.y)


# Activations of the random score network (a subset of sdm's `fixed_activation` pool)
_ACTIVATIONS = (
    np.tanh,
    lambda v: np.maximum(v, 0.0),
    lambda v: np.where(v > 0, v, 0.01 * v),
    np.sin,
    np.abs,
    np.square,
    lambda v: 1.0 / (1.0 + np.exp(-v)),
    lambda v: np.exp(-v * v),
    np.sign,
)


def _linear(inputs: np.ndarray, r: np.random.Generator, width: int = 1) -> np.ndarray:
    w = r.standard_normal((inputs.shape[1], width)) / np.sqrt(inputs.shape[1])
    return inputs @ w + r.normal(0.0, 0.1, size=width)


def _network(inputs: np.ndarray, r: np.random.Generator) -> np.ndarray:
    h = inputs
    for _ in range(int(r.integers(1, 3))):
        activation = _ACTIVATIONS[int(r.integers(len(_ACTIVATIONS)))]
        h = activation(_linear(h, r, log_uniform_int(r, 1, 32)))
        h = np.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)
    return _linear(h, r)


def relabel_score(inputs: np.ndarray, r: np.random.Generator) -> np.ndarray:
    """A random score of the selected columns, standardized: linear (half the time), a 1-2 layer
    network with a random activation (30%), or the product of two such functions (20%), as in
    sdm's relabel function mixture. Equal inputs get equal scores."""
    kind = r.choice(3, p=(0.5, 0.3, 0.2))
    if kind == 0:
        score = _linear(inputs, r)
    elif kind == 1:
        score = _network(inputs, r)
    else:
        parts = [_linear(inputs, r) if r.random() < 0.5 else _network(inputs, r) for _ in "ab"]
        score = parts[0] * parts[1]
    return _standardize(np.nan_to_num(score[:, 0]))


def score_classes(score: np.ndarray, n_classes: int, r: np.random.Generator) -> np.ndarray | None:
    """Cut a score into `n_classes` classes with random shares, or None when the score has fewer
    distinct values than classes (sdm `Relabel._classify`).

    - Shares are a softmax of max(log(w + 1e-4), -floor), w power-law weights and floor ~
      LogUniform(0.5, 6), so one class can dominate.
    - The cuts sit between distinct score values, so every class keeps at least one row.
    - Half the time the cut is exact. Otherwise each row's rank is jittered by a normal with sd
      n / (2a(C - 1)), a ~ LogUniform(20, 200), so that labels flip near the cuts. If that empties
      a class, the exact cut is kept.
    - Class numbers are shuffled.
    """
    n = len(score)
    ranked = np.sort(score)
    distinct = np.unique(ranked)
    if len(distinct) < n_classes:
        return None
    floor = np.exp(r.uniform(np.log(0.5), np.log(6.0)))
    logits = np.maximum(np.log(r.permutation(power_law_weights(r, n_classes)) + 1e-4), -floor)
    share = np.exp(logits - logits.max())
    share = np.cumsum(share / share.sum())[:-1]
    # cut k falls after the distinct value holding the share quantile, pushed apart so that every
    # class keeps at least one distinct value
    position = np.searchsorted(distinct, ranked[(share * (n - 1)).astype(np.int64)], side="right")
    offset = np.arange(n_classes - 1)
    position = np.maximum.accumulate(position - offset) + offset
    position = np.clip(position, 1 + offset, len(distinct) - n_classes + 1 + offset)
    thresholds = 0.5 * (distinct[position - 1] + distinct[position])
    labels = (score[:, None] > thresholds[None, :]).sum(1)
    if r.random() >= 0.5:
        a = np.exp(r.uniform(np.log(20.0), np.log(200.0)))
        rank = 0.5 * (np.searchsorted(ranked, score) + np.searchsorted(ranked, score, "right") - 1)
        jitter = r.standard_normal(n) * (0.5 * n / (a * (n_classes - 1)))
        jittered = ranked[np.clip(np.round(rank + jitter).astype(np.int64), 0, n - 1)]
        noisy = (jittered[:, None] > thresholds[None, :]).sum(1)
        # the jitter can empty a small class; then the exact cut is kept (sdm keeps the jittered
        # labels, so a class can go missing there)
        if len(np.unique(noisy)) == n_classes:
            labels = noisy
    return r.permutation(n_classes)[labels]


class Relabel:
    """With probability `dataset_prob` per dataset, replace the target by a random function of 1-4
    feature columns (`relabel_score`). Real targets often depend on a handful of columns through a
    simple rule, while the SCM target usually depends on many nodes.

    - Classification keeps the group's class count (`score_classes`); a dataset whose score has
      too few distinct values keeps its target.
    - Regression gets the score processed as TabICL processes its targets: values beyond 4 sds
      are clipped, then the target is standardized.

    Adapted from sdm `Relabel` (Kumo-Tabular: 10% of tables in stages 2 and 3). Differences: the
    score functions are a smaller numpy version of sdm's function mixture, and the columns are
    picked uniformly (sdm prefers numerical columns; the TabICL SCM does not mark the categorical
    ones).
    """

    def __init__(self, dataset_prob: float, *, classification: bool) -> None:
        if not 0.0 <= dataset_prob <= 1.0:
            raise ValueError(f"dataset_prob must be between 0 and 1: {dataset_prob}")
        self.dataset_prob = dataset_prob
        self.classification = classification

    def __call__(self, group: Group, rng: Rng) -> Group:
        r = rng.numpy
        ys = []
        for d in range(group.y.shape[1]):
            x, y = group.x[:, d], group.y[:, d]
            if r.random() < self.dataset_prob and torch.isfinite(x).all():
                new = self._relabel(x.double().cpu().numpy(), group.hp, r)
                if new is not None:
                    y = torch.from_numpy(new).to(dtype=y.dtype, device=y.device)
            ys.append(y)
        return group.with_tensors(group.x, torch.stack(ys, 1))

    def _relabel(self, values: np.ndarray, hp: dict, r: np.random.Generator) -> np.ndarray | None:
        f = values.shape[1]
        if f == 0:
            return None
        k = int(r.integers(1, min(4, f) + 1))
        score = relabel_score(values[:, r.permutation(f)[:k]], r)
        if self.classification:
            return score_classes(score, int(hp["num_classes"]), r)
        return regression_target(score)


def regression_target(score: np.ndarray) -> np.ndarray:
    """TabICL's regression target processing in numpy (`tabicl.prior._reg2cls`):
    `outlier_removing` finds the inliers within `OUTLIER_SDS` sds, clips at `OUTLIER_SDS` sds of
    the inliers' mean and sd, then `standard_scaling` standardizes (sd with ddof 1, at least
    1e-6)."""
    ddof = 1 if len(score) > 1 else 0
    mean, std = score.mean(), max(score.std(ddof=ddof), 1e-6)
    inliers = score[np.abs(score - mean) <= OUTLIER_SDS * std]
    if len(inliers) > ddof:
        mean, std = inliers.mean(), max(inliers.std(ddof=ddof), 1e-6)
    clipped = np.clip(score, mean - OUTLIER_SDS * std, mean + OUTLIER_SDS * std)
    return (clipped - clipped.mean()) / max(clipped.std(ddof=ddof), 1e-6)
