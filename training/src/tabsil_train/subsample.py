"""Subsampling of the training rows (the context) at inference, for experiments on whether the
model uses long contexts on large tables.

`SubsampledEstimator` wraps an inference estimator: if fit receives more than `max_rows` training
rows, it passes on `max_rows` rows drawn at random. All ensemble members fit their preprocessing
and context on the selected rows, so they share one subsample. Classification draws class-
stratified rows and keeps at least one row per class (`subsample_indices`).
"""

from __future__ import annotations

from typing import Any

import numpy as np


def subsample_indices(y: Any, n: int, *, stratify: bool, seed: int) -> np.ndarray:
    """Positions (ascending) of the `n` training rows to keep; all rows if there are at most `n`.

    With `stratify`, each class gets close to `n · class fraction` rows: the floored quota with at
    least one row per class, then adjusted one row at a time, adding to the classes with the
    largest fractional part when short and removing from the classes most above their quota when
    over. With more classes than `n`, one row per class can exceed `n`. The same `y`, `n` and
    `seed` give the same result.
    """
    if n < 1:
        raise ValueError(f"n must be at least 1: {n}")
    y = np.asarray(y)
    total = len(y)
    if total <= n:
        return np.arange(total)
    rng = np.random.default_rng(seed)
    if not stratify:
        return np.sort(rng.choice(total, n, replace=False))
    _, inverse = np.unique(y, return_inverse=True)
    inverse = inverse.reshape(-1)
    counts = np.bincount(inverse)
    quota = n * counts / total
    take = np.minimum(np.maximum(np.floor(quota).astype(int), 1), counts)
    while take.sum() < n:  # short after flooring: largest fraction first (classes with rows left)
        room = np.flatnonzero(take < counts)
        take[room[np.argmax((quota - take)[room])]] += 1
    while take.sum() > n:  # over from the 1-row minimum: most above quota first (keep 1 row)
        spare = np.flatnonzero(take > 1)
        if not len(spare):
            break
        take[spare[np.argmax((take - quota)[spare])]] -= 1
    # per class, draw `take` of its row positions (original order) at random
    by_class = np.split(np.argsort(inverse, kind="stable"), np.cumsum(counts)[:-1])
    parts = [rng.choice(rows, k, replace=False) for rows, k in zip(by_class, take, strict=True)]
    return np.sort(np.concatenate(parts))


ROW_ALIGNED_FIT_KWARGS = ("groups",)  # row-aligned fit kwargs: reduced to the same rows


def _take(value: Any, ix: np.ndarray) -> Any:
    return value.iloc[ix] if hasattr(value, "iloc") else np.asarray(value)[ix]


class SubsampledEstimator:
    """Wrap an estimator and reduce the training rows of fit to `max_rows` (`subsample_indices`).

    Row-aligned fit kwargs (`ROW_ALIGNED_FIT_KWARGS`, currently `groups`) are reduced to the same
    rows. `train_rows_` records (original, selected) row counts. All other attributes (`predict`,
    `predict_proba`, `classes_`, `after_predict`, ...) are those of the wrapped estimator.
    """

    def __init__(self, estimator, max_rows: int, *, stratify: bool, seed: int = 0) -> None:
        if max_rows < 1:
            raise ValueError(f"max_rows must be at least 1: {max_rows}")
        self.estimator = estimator
        self.max_rows = max_rows
        self.stratify = stratify
        self.seed = seed

    def __getattr__(self, name: str):
        # during copy/pickle (before `estimator` is set), do not look up the wrapped estimator
        if name.startswith("__") or name == "estimator":
            raise AttributeError(name)
        return getattr(self.estimator, name)

    def fit(self, X, y, **fit_kwargs) -> SubsampledEstimator:
        total = len(y)
        ix = subsample_indices(y, self.max_rows, stratify=self.stratify, seed=self.seed)
        self.train_rows_ = (total, len(ix))
        if len(ix) < total:
            X, y = _take(X, ix), _take(y, ix)
            fit_kwargs = {
                k: _take(v, ix) if k in ROW_ALIGNED_FIT_KWARGS and v is not None else v
                for k, v in fit_kwargs.items()
            }
        self.estimator.fit(X, y, **fit_kwargs)
        return self
