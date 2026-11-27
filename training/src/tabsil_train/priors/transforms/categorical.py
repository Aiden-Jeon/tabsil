"""Categorical columns with many categories (`HighCardinality`).

The TabICL graph SCM makes categorical columns with at most 9 categories (`tabicl` `PriorConfig`
`use_corrected_cat_sizes=False`, which TabICLv2 trained with). Real tables also have columns with
tens to thousands of categories (TabArena-Lite: Amazon_employee_access 7,518, kddcup09_appetency
15,415). `HighCardinality` recodes a few columns of some tables into such columns.

The recoded column looks like the one inference gives the model (`tabsil.inference.preprocessing`):
a column with more than `MAX_ONE_HOT_CATEGORIES` categories is not one-hot encoded; its ordinal
codes, fitted on the train rows, go in as numbers (the model standardizes them with the train rows),
and a category that is not among the train rows becomes NaN. So the codes here are 0..K-1 over the
categories of the train rows, in an order unrelated to the source values (ordinal codes follow the
sorted category names), and eval-row categories missing from the train rows are NaN.

How the categories are cut is adapted from NVIDIA structured-data-models (Apache-2.0)
`sdm/prior/tabular/transform/recode.py` `RecodeHighCardinality`; see NOTICE.
"""

from __future__ import annotations

import numpy as np
import torch

from tabsil_train.priors.pipeline import Group, Rng

# Inference one-hot encodes categorical columns with at most this many categories
# (`tabsil.inference.preprocessing.MAX_ONE_HOT_CATEGORIES`; a test checks they agree). Recoded
# columns get more categories than this, so inference gives them to the model as codes, as here
MAX_ONE_HOT_CATEGORIES = 20


def log_uniform_int(rng: np.random.Generator, low: int, high: int) -> int:
    """An integer in [low, high], log-uniform."""
    if high <= low:
        return low
    value = int(np.exp(rng.uniform(np.log(low), np.log(high + 1))))
    return min(max(value, low), high)


def power_law_weights(rng: np.random.Generator, n: int) -> np.ndarray:
    """n positive weights summing to 1, sorted from largest to smallest: w_i ∝ i^-q exp(σ z_i),
    q ~ LogUniform(0.1 / log(n + 1), 6), σ ~ LogUniform(1e-4, 10) (sdm `random_weight`). Large q
    gives a few dominant categories and a long tail of rare ones."""
    q = np.exp(rng.uniform(np.log(0.1 / np.log(n + 1)), np.log(6.0)))
    sigma = np.exp(rng.uniform(np.log(1e-4), np.log(10.0)))
    log_w = sigma * rng.standard_normal(n) - q * np.log(np.arange(1, n + 1))
    w = np.exp(log_w - log_w.max())
    return np.sort(w / w.sum())[::-1]


def level_codes(values: np.ndarray, n_levels: int, rng: np.random.Generator) -> np.ndarray:
    """Cut a column into `n_levels` categories by value rank, with power-law category sizes (every
    category gets at least one row), and number the categories in random order.

    When there are at least as many categories as distinct values, a change of value also starts
    a new category, so the codes refine the values instead of merging distinct ones (sdm
    `RecodeHighCardinality`). Rows with equal values are ordered at random, so a category with
    many equal rows can be split into several codes."""
    n = len(values)
    order = np.lexsort((rng.random(n), values))  # by value, ties in random order
    sorted_values = values[order]
    weights = power_law_weights(rng, n_levels)
    edges = np.cumsum((1.0 + (n - n_levels) * weights) / n)
    position = (np.arange(n) + 0.5) / n
    level = np.minimum(np.searchsorted(edges, position, side="right"), n_levels - 1)
    same = sorted_values[1:] == sorted_values[:-1]
    refine = n_levels >= n - int(same.sum())
    new = (level[1:] != level[:-1]) | (refine & ~same)
    sorted_code = np.concatenate([[0], np.cumsum(new)])
    shuffled = rng.permutation(int(sorted_code[-1]) + 1)[sorted_code]
    codes = np.empty(n, dtype=np.int64)
    codes[order] = shuffled
    return codes


def train_row_codes(codes: np.ndarray, n_train: int) -> np.ndarray:
    """The codes as inference encodes them: the categories of the first `n_train` rows numbered
    0..K-1 (keeping their relative order), other categories NaN."""
    categories = np.unique(codes[:n_train])
    index = np.searchsorted(categories, codes)
    known = (index < len(categories)) & (
        categories[np.minimum(index, len(categories) - 1)] == codes
    )
    return np.where(known, index, np.nan)


class HighCardinality:
    """With probability `dataset_prob` per dataset, recode `num_columns` (inclusive range) random
    columns as categorical columns with K ~ LogUniformInt(`min_levels`, rows) categories.

    sdm `RecodeHighCardinality` (Kumo-Tabular, 5% of tables) recodes 1-3 columns with at least
    201 categories, since its SCM already makes columns with up to 200. The TabICL SCM stops at 9,
    so the default lower end here is just above the inference one-hot limit (21), covering both
    ranges. Any column can be the source (numeric or categorical); its values decide the
    categories, so the recoded column keeps their information behind a random code order. Tables
    with fewer rows than `min_levels` are left as they are.
    """

    def __init__(
        self,
        dataset_prob: float,
        *,
        num_columns: tuple[int, int] = (1, 3),
        min_levels: int = MAX_ONE_HOT_CATEGORIES + 1,
    ) -> None:
        if not 0.0 <= dataset_prob <= 1.0:
            raise ValueError(f"dataset_prob must be between 0 and 1: {dataset_prob}")
        if not 1 <= num_columns[0] <= num_columns[1] or min_levels < 2:
            raise ValueError(f"bad num_columns {num_columns} or min_levels {min_levels}")
        self.dataset_prob = dataset_prob
        self.num_columns = num_columns
        self.min_levels = min_levels

    def __call__(self, group: Group, rng: Rng) -> Group:
        r = rng.numpy
        xs = []
        for d in range(group.x.shape[1]):
            x = group.x[:, d]
            if r.random() < self.dataset_prob:
                x = self._recode(x, group.n_train, r)
            xs.append(x)
        return group.with_tensors(torch.stack(xs, 1), group.y)

    def _recode(self, x: torch.Tensor, n_train: int, r: np.random.Generator) -> torch.Tensor:
        values = x.double().cpu().numpy()
        n_rows, n_features = values.shape
        most = min(self.num_columns[1], n_features)
        if n_rows < self.min_levels or most < self.num_columns[0]:
            return x
        count = int(r.integers(self.num_columns[0], most + 1))
        out = values.copy()
        for column in r.permutation(n_features)[:count]:
            source = values[:, column]
            if not np.isfinite(source).all():
                continue
            n_levels = log_uniform_int(r, self.min_levels, n_rows)
            out[:, column] = train_row_codes(level_codes(source, n_levels, r), n_train)
        return torch.from_numpy(out).to(dtype=x.dtype, device=x.device)
