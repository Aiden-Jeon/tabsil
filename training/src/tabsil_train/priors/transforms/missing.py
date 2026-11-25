"""Missing values.

- `Missing` (TabPFN v2 paper, "Post-processing"): in some datasets, erase each cell with the same
  probability (MCAR). The v1 prior uses only this.
- `StructuredMissing`: missing values that depend on values (MAR/MNAR) and come in column groups,
  adapted from NVIDIA structured-data-models (Apache-2.0)
  `sdm/prior/tabular/transform/missing/{structured,near_empty,latent}.py` (Kumo-Tabular); see
  NOTICE. Seldon's prior gives half of its tables MCAR/MAR/MNAR missing values, often jointly over
  column groups.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from tabsil_train.priors.pipeline import Group, Rng


class Missing:
    """With probability `dataset_prob` per dataset, set a fraction r ~ U(0, max_rate) of x to NaN.

    The v2 paper does not give these two values; they are our choice in the graph SCM prior
    (`missing_dataset_prob`, `max_missing_rate`).
    """

    def __init__(self, dataset_prob: float, max_rate: float) -> None:
        self.dataset_prob = dataset_prob
        self.max_rate = max_rate

    def __call__(self, group: Group, rng: Rng) -> Group:
        r = rng.numpy
        xs = []
        for d in range(group.x.shape[1]):
            x = group.x[:, d]
            if r.random() < self.dataset_prob:
                rate = r.uniform(0.0, self.max_rate)
                mask = torch.from_numpy(r.random(tuple(x.shape)) < rate).to(x.device)
                x = x.masked_fill(mask, float("nan"))
            xs.append(x)
        return group.with_tensors(torch.stack(xs, 1), group.y)


# ---------------------------------------------------------------------------
# Structured missing values. Each mechanism takes the cells (rows, features) as float64 with NaN
# for missing values and sets more of them to NaN in place
# ---------------------------------------------------------------------------


def _count(r: np.random.Generator, total: int, low: float, high: float) -> int:
    """An integer in [ceil(total·low), floor(total·high)] (at least 1; sdm `_sample_count`)."""
    lower = max(1, math.ceil(total * low))
    upper = max(lower, min(total, math.floor(total * high)))
    return int(r.integers(lower, upper + 1))


def _log_uniform(r: np.random.Generator, low: float, high: float, size=None):
    return np.exp(r.uniform(np.log(low), np.log(high), size))


def mask_rows(values: np.ndarray, r: np.random.Generator) -> None:
    """The same 5-20% of rows missing in 50-90% of the columns (sdm `MaskRows`)."""
    n, f = values.shape
    rows = r.permutation(n)[: _count(r, n, 0.05, 0.2)]
    columns = r.permutation(f)[: _count(r, f, 0.5, 0.9)]
    values[np.ix_(rows, columns)] = np.nan


def mask_block(values: np.ndarray, r: np.random.Generator) -> None:
    """The same 10-30% of rows missing in 2-5 columns (sdm `MaskBlock`)."""
    n, f = values.shape
    if f < 2:
        return
    row_count = _count(r, n, 0.1, 0.3)
    column_count = int(r.integers(2, min(5, f) + 1))
    rows = r.permutation(n)[:row_count]
    columns = r.permutation(f)[:column_count]
    values[np.ix_(rows, columns)] = np.nan


def mask_by_target(
    values: np.ndarray, y: np.ndarray, classification: bool, r: np.random.Generator
) -> None:
    """1-3 columns missing in the rows of one class, or of one 10-30% tail of a regression target
    (MNAR through the target; sdm `MaskByTarget`)."""
    n, f = values.shape
    columns = r.permutation(f)[: int(r.integers(1, min(3, f) + 1))]
    if classification:
        classes = np.unique(y)
        rows = np.flatnonzero(y == classes[r.integers(len(classes))])
    else:
        tail = _count(r, n, 0.1, 0.3)
        order = np.argsort(y, kind="stable")
        rows = order[-tail:] if r.random() < 0.5 else order[:tail]
    values[np.ix_(rows, columns)] = np.nan


def mask_censoring(values: np.ndarray, r: np.random.Generator) -> None:
    """1-6 columns lose the rows in one 5-25% tail of a column: their own (self-masking MNAR) or,
    with probability 1/2, another column's (MAR) (sdm `MaskCensoring`)."""
    n, f = values.shape
    for destination in r.permutation(f)[: int(r.integers(1, min(6, f) + 1))]:
        source = destination
        if f > 1 and r.random() < 0.5:
            others = np.delete(np.arange(f), destination)
            source = others[r.integers(len(others))]
        finite = np.flatnonzero(np.isfinite(values[:, source]))
        if len(finite) == 0:
            continue
        count = _count(r, len(finite), 0.05, 0.25)
        order = finite[np.argsort(values[finite, source], kind="stable")]
        rows = order[-count:] if r.random() < 0.5 else order[:count]
        values[rows, destination] = np.nan


def mask_structured(
    values: np.ndarray, y: np.ndarray, classification: bool, r: np.random.Generator
) -> None:
    """One of rows / block / by-target / nothing (weights 1, 1, 1, 4), then censoring if nothing
    changed or with probability 1/2 (sdm `MaskStructured`). sdm's fourth choice, `MaskCategorical`
    (random rows of categorical columns), is left out: the graph SCM table does not mark its
    categorical columns."""
    observed = np.isfinite(values).sum()
    choice = r.choice(4, p=np.array([1, 1, 1, 4]) / 7)
    if choice == 0:
        mask_rows(values, r)
    elif choice == 1:
        mask_block(values, r)
    elif choice == 2:
        mask_by_target(values, y, classification, r)
    if np.isfinite(values).sum() == observed or r.random() < 0.5:
        mask_censoring(values, r)


def mask_near_empty(values: np.ndarray, r: np.random.Generator, min_observed: int = 4) -> None:
    """50-80% of the columns, split into groups of power-law sizes, are observed only in 1-10% of
    the rows (the same rows within a group, groups overlapping as little as possible; sdm
    `MaskNearEmpty`). A column that would keep fewer than two distinct values swaps one observed
    row for a row with another value. sdm correlates the group rates within a table; here they are
    drawn independently from LogUniform(0.01, 0.1)."""
    n, f = values.shape
    column_count = _count(r, f, 0.5, 0.8)
    columns = r.permutation(f)[:column_count]
    group_count = max(1, _count(r, column_count, 0.06, 0.16))
    exponent = r.uniform(0.9, 1.6)
    weights = np.arange(1, group_count + 1, dtype=float) ** -exponent
    weights = weights[r.permutation(group_count)]
    family = r.choice(group_count, size=column_count, p=weights / weights.sum())
    family[:group_count] = np.arange(group_count)  # every group gets a column
    rate = _log_uniform(r, 0.01, 0.1, group_count)
    observed_count = np.clip(np.ceil(rate * n).astype(int), min(min_observed, n), n)
    sizes = np.bincount(family, minlength=group_count)
    load = np.zeros(n)
    for g in np.argsort(-sizes, kind="stable"):
        observed_rows = np.argsort(load + r.random(n), kind="stable")[: observed_count[g]]
        load[observed_rows] += sizes[g]
        observed = np.zeros(n, dtype=bool)
        observed[observed_rows] = True
        for column in columns[family == g]:
            column_values = values[:, column]
            seen = column_values[observed]
            distinct = np.unique(seen[np.isfinite(seen)])
            keep = observed
            if len(distinct) < 2:
                candidates = np.isfinite(column_values) & ~observed
                if len(distinct) == 1:
                    candidates &= column_values != distinct[0]
                candidates = np.flatnonzero(candidates)
                if len(candidates):
                    keep = observed.copy()
                    keep[observed_rows[r.integers(len(observed_rows))]] = False
                    keep[candidates[r.integers(len(candidates))]] = True
            values[~keep, column] = np.nan


def mask_latent(values: np.ndarray, r: np.random.Generator) -> None:
    """An unobserved row factor drives the missing rate of column blocks (MAR; sdm `MaskLatent`).

    The factor marks a 10-50% share of rows by a noisy random projection of 1-3 well-observed proxy
    columns; 20-60% of the other columns, split into blocks, are missing at a rate of U(0.3, 0.9)
    in those rows and U(0, 0.1) elsewhere (one rate pair per block; sdm correlates them within a
    table)."""
    n, f = values.shape
    if f < 2:
        return
    proxy_count = int(r.integers(1, min(3, f - 1) + 1))
    candidates = np.flatnonzero(np.isfinite(values).mean(0) >= 0.5)
    if len(candidates) < proxy_count:
        candidates = np.arange(f)
    proxies = candidates[r.permutation(len(candidates))[:proxy_count]]
    remaining = np.setdiff1d(np.arange(f), proxies)
    column_count = _count(r, len(remaining), 0.2, 0.6)
    columns = remaining[r.permutation(len(remaining))[:column_count]]
    score = np.nan_to_num(values[:, proxies], nan=0.0) @ r.standard_normal(proxy_count)
    score = (score - score.mean()) / max(score.std(ddof=1) if n > 1 else 0.0, 1e-6)
    score = score + _log_uniform(r, 0.25, 1.0) * r.standard_normal(n)
    factor = score > np.quantile(score, 1.0 - r.uniform(0.1, 0.5))
    block_count = int(r.integers(1, max(1, len(columns) // 4) + 1))
    block = r.integers(block_count, size=len(columns))
    off, on = r.uniform(0.0, 0.1, block_count), r.uniform(0.3, 0.9, block_count)
    rate = np.where(factor[None, :], on[:, None], off[:, None])  # (blocks, rows)
    missing = r.random((block_count, n)) < rate
    sub = values[:, columns]
    sub[missing[block].T] = np.nan
    values[:, columns] = sub


class StructuredMissing:
    """Value-dependent missing values in column groups, per dataset (sdm Kumo-Tabular prior):

    1. with probability `near_empty_prob` `mask_near_empty`, otherwise with probability
       `structured_prob` `mask_structured` (rows / block / by-target, then MAR/MNAR censoring)
    2. independently, with probability `latent_prob`, `mask_latent`

    Kumo-Tabular uses 0.05, 0.2 and 0.03 in its stage 2 and 3 (none in stage 1). A probability of
    0 draws no random number for that step. `classification` says whether y holds classes (for the
    by-target mechanism).
    """

    def __init__(
        self,
        *,
        structured_prob: float = 0.0,
        near_empty_prob: float = 0.0,
        latent_prob: float = 0.0,
        classification: bool,
    ) -> None:
        for name, p in (
            ("structured_prob", structured_prob),
            ("near_empty_prob", near_empty_prob),
            ("latent_prob", latent_prob),
        ):
            if not 0.0 <= p <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1: {p}")
        self.structured_prob = structured_prob
        self.near_empty_prob = near_empty_prob
        self.latent_prob = latent_prob
        self.classification = classification

    def __call__(self, group: Group, rng: Rng) -> Group:
        r = rng.numpy
        xs = []
        for d in range(group.x.shape[1]):
            x = group.x[:, d]
            values = x.double().cpu().numpy().copy()
            y = group.y[:, d].double().cpu().numpy()
            changed = True
            if self.near_empty_prob and r.random() < self.near_empty_prob:
                mask_near_empty(values, r)
            elif self.structured_prob and r.random() < self.structured_prob:
                mask_structured(values, y, self.classification, r)
            else:
                changed = False
            if self.latent_prob and r.random() < self.latent_prob:
                mask_latent(values, r)
                changed = True
            if changed and values.shape[1]:
                mask = torch.from_numpy(np.isnan(values)).to(x.device)
                x = x.masked_fill(mask, float("nan"))
            xs.append(x)
        return group.with_tensors(torch.stack(xs, 1), group.y)
