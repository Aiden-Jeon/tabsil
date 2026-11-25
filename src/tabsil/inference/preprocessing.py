"""Preprocessing of one ensemble member.

A member fits its preprocessing on the training rows (`MemberPreprocessor.fit`) and applies the
same transform to the training and test rows. TabSil has one member kind, `"standard"`: the column
values are kept as they are (the model standardizes them with the training rows), with these steps:

- numeric values more than `outlier_sigma` (default `OUTLIER_SIGMA`) standard deviations from the
  column mean are clipped to that bound
- (`Member.one_hot`) categorical columns with at most `MAX_ONE_HOT_CATEGORIES` categories become
  one-hot blocks (`one_hot_sizes`)
- a sample identifier column of standard normal values is appended and the column order is
  shuffled. Training rows draw it from the member seed; a test row's value is a hash of that row's
  values and the member seed (`row_identifier`), so its prediction does not depend on the other
  test rows, their order or the batch it is predicted in (inductive predictions)

How the members differ (seeds, class permutations, the regression target transform) is in
`tabsil.inference.ensemble`.

A table with more features than the member cap gives each member a subset of its original columns:
`feature_subset` draws one per member from the member seed, `balanced_feature_subsets` spreads the
columns evenly over the members.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

OUTLIER_SIGMA = 12.0
MAX_ONE_HOT_CATEGORIES = 20  # benchmark socmob (17) is expanded; colleges (thousands) stays codes
MEMBER_KINDS = ("standard",)
FEATURE_SUBSAMPLINGS = ("random", "balanced")


def to_numeric(X: Any, categorical: list[int] | None) -> tuple[np.ndarray, list[int], Any]:
    """Encode categorical and string columns as ordinal codes. Missing values stay NaN, and so do
    categories not seen in these rows (when the encoder is applied to other rows)."""
    df = pd.DataFrame(X).reset_index(drop=True)
    cat = set(categorical or [])
    cat |= {i for i, dtype in enumerate(df.dtypes) if not pd.api.types.is_numeric_dtype(dtype)}
    cat_ix = sorted(cat)
    encoder = None
    if cat_ix:
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan)
        encoder.fit(df.iloc[:, cat_ix].astype(str).where(df.iloc[:, cat_ix].notna()))
    return apply_encoder(df, cat_ix, encoder), cat_ix, encoder


def apply_encoder(df: pd.DataFrame, cat_ix: list[int], encoder: Any) -> np.ndarray:
    df = pd.DataFrame(df).reset_index(drop=True)
    out = np.empty(df.shape, dtype=np.float64)
    num_ix = [i for i in range(df.shape[1]) if i not in cat_ix]
    if num_ix:
        out[:, num_ix] = df.iloc[:, num_ix].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if cat_ix:
        cats = df.iloc[:, cat_ix]
        out[:, cat_ix] = encoder.transform(cats.astype(str).where(cats.notna()))
    return out


def feature_subset(n_features: int, max_features: int | None, seed: int) -> np.ndarray | None:
    """The sorted original column indices one member uses; None (all columns) when there are at
    most `max_features` columns (or `max_features` is None).

    Drawn per member, like the "random" method (`_subsample_features_random`) of the official TabPFN
    inference code. `seed` comes from the member (`Member.seed`), so a member uses the same columns
    in every fit. No randomness for tables within the cap."""
    if max_features is None or n_features <= max_features:
        return None
    rng = np.random.default_rng(seed)
    return np.sort(rng.permutation(n_features)[:max_features])


def balanced_feature_subsets(
    n_features: int, max_features: int | None, n_members: int, rng: np.random.Generator
) -> list[np.ndarray | None]:
    """Draw each member's original column indices so the columns are spread evenly (the
    "balanced" method of the official TabPFN-3.5 inference code).

    Members take `max_features` columns in turn from one shuffled column order; when it runs out,
    it is reshuffled and refilled (without the columns that member already has). So every column is
    used about equally often. Tables within the cap get None (all columns) for every member and use
    no randomness.

    Adapted from TabPFN (Apache-2.0) `src/tabpfn/preprocessing/ensemble.py`
    `_subsample_features_balanced` and `_draw_balanced_from_pool`; see NOTICE."""
    if max_features is None or n_features <= max_features:
        return [None] * n_members
    order = rng.permutation(n_features)
    pool: list[int] = []
    out: list[np.ndarray | None] = []
    for _ in range(n_members):
        slots: list[int] = []
        while len(slots) < max_features:
            if not pool:
                taken = set(slots)
                pool = [i for i in range(n_features) if i not in taken]
                rng.shuffle(pool)
            take = min(max_features - len(slots), len(pool))
            slots += pool[:take]
            pool = pool[take:]
        out.append(np.sort(order[np.array(slots)]))
    return out


def select_features(
    X: np.ndarray, cat_ix: list[int], features: np.ndarray | None
) -> tuple[np.ndarray, list[int]]:
    """A column subset of X (`features`, all if None) and the categorical column indices in it."""
    if features is None:
        return X, cat_ix
    cat = set(cat_ix)
    return X[:, features], [j for j, i in enumerate(features) if int(i) in cat]


def _one_hot_width(n: int) -> int:
    return 1 if n == 2 else n  # 2 categories become a single 0/1 column


def _splitmix64(x: np.ndarray) -> np.ndarray:
    """SplitMix64 finalizer: a bijective mix of 64-bit integers (wraps around on overflow)."""
    x = np.asarray(x, dtype=np.uint64)
    with np.errstate(over="ignore"):
        x = x + np.uint64(0x9E3779B97F4A7C15)
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return x ^ (x >> np.uint64(31))


def _unit_interval(h: np.ndarray) -> np.ndarray:
    # the top 53 bits as a float in (0, 1): never 0 or 1, so log() and the normal tail stay finite
    return ((h >> np.uint64(11)).astype(np.float64) + 0.5) * 2.0**-53


def row_identifier(X: np.ndarray, seed: int) -> np.ndarray:
    """A sample identifier per row, (n, 1) standard normal values: each is a hash of that row's
    values and `seed` only, so it does not depend on the other rows or the row's position (equal
    rows get equal values). The hash reads the float64 bits of each cell (NaN and -0.0 are made
    canonical first), so it does not change with the pandas or NumPy version."""
    X = np.array(X, dtype=np.float64, ndmin=2)
    X[np.isnan(X)] = np.nan  # one NaN bit pattern
    X += 0.0  # -0.0 becomes 0.0
    bits = np.ascontiguousarray(X).view(np.uint64)
    h = np.full(X.shape[0], _splitmix64(np.uint64(seed)), dtype=np.uint64)
    for j in range(bits.shape[1]):
        h = _splitmix64(h ^ bits[:, j])
    # Box-Muller on two uniforms from the row hash
    u1, u2 = _unit_interval(h), _unit_interval(_splitmix64(h))
    return (np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2))[:, None]


def one_hot_sizes(n_categories: dict[int, int], n_features: int, max_width: int) -> dict[int, int]:
    """The columns to one-hot encode and their category counts.

    Columns with at most `MAX_ONE_HOT_CATEGORIES` categories are expanded. If the column count after
    expansion exceeds `max_width` (the feature count the model saw in training), columns revert to
    codes starting from the one with the most categories (columns with at most 2 categories, which
    do not add columns, stay expanded).
    """
    sizes = {i: n for i, n in n_categories.items() if n <= MAX_ONE_HOT_CATEGORIES}
    width = n_features + sum(_one_hot_width(n) - 1 for n in sizes.values())
    wide = [i for i in sizes if _one_hot_width(sizes[i]) > 1]
    for i in sorted(wide, key=lambda i: -sizes[i]):
        if width <= max_width:
            break
        width -= _one_hot_width(sizes.pop(i)) - 1
    return sizes


@dataclass
class Member:
    class_permutation: np.ndarray | None
    target_power: bool
    seed: int
    one_hot: bool = True
    outlier_sigma: float = OUTLIER_SIGMA  # clip beyond this many std devs from the mean


class MemberPreprocessor:
    """Preprocessing of one member. Fit on the training rows only."""

    def __init__(self, member: Member, cat_ix: list[int], max_one_hot_width: int) -> None:
        self.member = member
        self.cat_ix = cat_ix
        self.max_one_hot_width = max_one_hot_width

    def fit(self, X: np.ndarray) -> MemberPreprocessor:
        rng = np.random.default_rng(self.member.seed)
        sigma = self.member.outlier_sigma
        mean, std = np.nanmean(X, axis=0), np.nanstd(X, axis=0)
        self.lower, self.upper = mean - sigma * std, mean + sigma * std
        # categorical columns are not clipped (clipping codes would merge rare categories)
        self.lower[self.cat_ix], self.upper[self.cat_ix] = -np.inf, np.inf
        X = self._clip(X)
        # categorical codes run from 0 to (number of categories in the training rows - 1)
        n_categories = {
            i: int(np.nanmax(X[:, i])) + 1 if np.isfinite(X[:, i]).any() else 1 for i in self.cat_ix
        }
        self.one_hot_sizes = (
            one_hot_sizes(n_categories, X.shape[1], self.max_one_hot_width)
            if self.member.one_hot
            else {}
        )
        # Numeric columns with no value in the training rows are missing in every row
        self.empty_ix = [
            i for i in range(X.shape[1]) if i not in self.cat_ix and not np.isfinite(X[:, i]).any()
        ]
        # column order: shuffle the transformed columns and the one sample identifier together
        self.order = rng.permutation(self._base(X).shape[1] + 1)
        return self

    def _clip(self, X: np.ndarray) -> np.ndarray:
        return np.clip(X, self.lower, self.upper)

    def _base(self, X: np.ndarray) -> np.ndarray:
        out = X.copy()
        out[:, self.empty_ix] = np.nan
        if not self.one_hot_sizes:
            return out
        # drop the code columns and append the one-hot columns (the leading columns keep the
        # original column order, so their positions stay the same)
        keep = [j for j in range(out.shape[1]) if j not in self.one_hot_sizes]
        blocks = [self._one_hot(X[:, i], n) for i, n in self.one_hot_sizes.items()]
        return np.concatenate([out[:, keep], *blocks], axis=1)

    @staticmethod
    def _one_hot(codes: np.ndarray, n: int) -> np.ndarray:
        # missing and out-of-range codes become NaN: treated as missing, as in the code column
        known = np.isfinite(codes) & (codes >= 0) & (codes < n)
        block = np.full((len(codes), n), np.nan)
        block[known] = 0.0
        block[np.flatnonzero(known), codes[known].astype(int)] = 1.0
        return block[:, 1:] if n == 2 else block

    def transform(self, X: np.ndarray, identifier: np.ndarray) -> np.ndarray:
        """The member's input columns for rows `X`; `identifier` is the (n, 1) sample identifier
        column (training rows: drawn from the member seed; test rows: `row_identifier`)."""
        base = self._base(self._clip(X))
        return np.concatenate([base, identifier], axis=1)[:, self.order]
