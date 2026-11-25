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
  values and the member seed (`row_hashes`, `row_identifier`), so its prediction does not depend
  on the other test rows, their order or the batch it is predicted in (inductive predictions)

How the members differ (seeds, class permutations, the regression target transform) is in
`tabsil.inference.ensemble`.

Options (inference experiments; the defaults are TabSil's):

- `Member.kind` (`MEMBER_KINDS`): a transform of the numeric columns fitted on the training rows,
  after the outlier clip. `"standard"` keeps the values; the others first standardize the column
  and clip it to ±100, then `"power"` applies a Yeo-Johnson power transform (standardized),
  `"robust"` centers on the median, divides by the interquartile range and squashes smoothly into
  ±3 (x / sqrt(1 + (x / 3)^2)), and `"quantile"` maps the column to standard normal quantiles of
  its training-row ranks (rank-Gauss). The choices follow the round-robin numeric processor of
  Kumo-Tabular (NVIDIA structured-data-models `sdm/models/kumo/tabular/recipe.py`)
- `Member.flip_sign`: each numeric column is negated with probability 1/2 (Kumo's `FlipSign`)
- `Member.categories` (`CATEGORY_ENCODINGS`), for categorical columns that stay codes and have more
  than `MAX_ONE_HOT_CATEGORIES` categories: `"codes"` keeps the codes, `"counts"` adds a column
  with log(1 + the number of training rows with that category) (Kumo's `AddCategoryCounts`; a
  missing or unseen category is missing)

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
MEMBER_KINDS = ("standard", "power", "robust", "quantile")
CATEGORY_ENCODINGS = ("codes", "counts")
ROBUST_BOUND = 3.0  # the "robust" kind squashes values into ±this
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
    # the top 52 bits as a float in (0, 1): (k + 0.5) * 2^-52 is exact in float64 for k < 2^52,
    # so the value is never 0 or 1 and log() stays finite
    return ((h >> np.uint64(12)).astype(np.float64) + 0.5) * 2.0**-52


def row_hashes(X: np.ndarray) -> np.ndarray:
    """A 64-bit hash per row of its float64 cell values (NaN and -0.0 made canonical first), so
    it depends on that row only. It reads the bits of the values, not pandas' hashing."""
    X = np.asarray(X, dtype=np.float64)
    h = np.zeros(X.shape[0], dtype=np.uint64)
    for j in range(X.shape[1]):
        col = X[:, j] + 0.0  # a contiguous copy of the column; -0.0 becomes 0.0
        col[np.isnan(col)] = np.nan  # one NaN bit pattern
        h = _splitmix64(h ^ col.view(np.uint64))
    return h


def row_identifier(hashes: np.ndarray, seed: int) -> np.ndarray:
    """A sample identifier per row, (n, 1) standard normal values from the row hashes
    (`row_hashes`) and `seed` only: it does not depend on the other rows or the row's position,
    and equal rows get equal values."""
    h = _splitmix64(np.asarray(hashes, dtype=np.uint64) ^ _splitmix64(np.uint64(seed)))
    # Box-Muller on two uniforms from the hash
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
    kind: str = "standard"  # numeric column transform (`MEMBER_KINDS`, module docstring)
    flip_sign: bool = False  # negate each numeric column with probability 1/2
    categories: str = "codes"  # many-category code columns (`CATEGORY_ENCODINGS`)


class _NumericTransform:
    """A member kind's transform of the numeric columns (module docstring), fitted on the training
    rows. Missing values stay missing."""

    def __init__(self, kind: str, X: np.ndarray, seed: int) -> None:
        from sklearn.preprocessing import PowerTransformer, QuantileTransformer

        if kind not in MEMBER_KINDS:
            raise ValueError(f"the member kind is one of {MEMBER_KINDS}: {kind}")
        self.kind = kind
        self.mean = np.nan_to_num(np.nanmean(X, axis=0)) if X.shape[0] else np.zeros(X.shape[1])
        std = np.nan_to_num(np.nanstd(X, axis=0)) if X.shape[0] else np.ones(X.shape[1])
        self.std = np.where(std > 1e-12, std, 1.0)
        z = self._standardize(X)
        self.fitted = None
        if kind == "power":
            self.fitted = PowerTransformer(method="yeo-johnson", standardize=True).fit(z)
        elif kind == "quantile":
            n = int(np.isfinite(z).sum(0).max(initial=1))
            self.fitted = QuantileTransformer(
                n_quantiles=max(2, min(1000, n)),
                output_distribution="normal",
                subsample=10**9,
                random_state=seed % 2**32,
            ).fit(z)
        elif kind == "robust":
            lo, self.median, hi = np.nanpercentile(z, [25, 50, 75], axis=0)
            iqr = np.nan_to_num(hi - lo)
            self.median = np.nan_to_num(self.median)
            self.iqr = np.where(iqr > 1e-12, iqr, 1.0)

    def _standardize(self, X: np.ndarray) -> np.ndarray:
        return np.clip((X - self.mean) / self.std, -100.0, 100.0)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        if self.kind == "standard" or X.shape[1] == 0:
            return X
        z = self._standardize(X)
        if self.kind == "robust":
            r = (z - self.median) / self.iqr
            return r / np.sqrt(1.0 + np.square(r / ROBUST_BOUND))
        with np.errstate(all="ignore"):
            out = np.asarray(self.fitted.transform(z), dtype=np.float64)
        # a value the transform cannot map (overflow of the power transform) keeps its z value
        bad = ~np.isfinite(out) & np.isfinite(z)
        out[bad] = z[bad]
        return out


class MemberPreprocessor:
    """Preprocessing of one member. Fit on the training rows only."""

    def __init__(self, member: Member, cat_ix: list[int], max_one_hot_width: int) -> None:
        self.member = member
        self.cat_ix = cat_ix
        self.max_one_hot_width = max_one_hot_width

    def fit(self, X: np.ndarray) -> MemberPreprocessor:
        rng = np.random.default_rng(self.member.seed)
        member = self.member
        if member.categories not in CATEGORY_ENCODINGS:
            raise ValueError(f"categories is one of {CATEGORY_ENCODINGS}: {member.categories}")
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
        self._fit_options(X, n_categories)
        # column order: shuffle the transformed columns and the one sample identifier together
        self.order = rng.permutation(self._features(X).shape[1] + 1)
        return self

    def _fit_options(self, X: np.ndarray, n_categories: dict[int, int]) -> None:
        # the member options of the module docstring, from their own random streams (so that the
        # defaults draw exactly what they drew before the options existed)
        member = self.member
        cat = set(self.cat_ix)
        self.numeric_ix = [i for i in range(X.shape[1]) if i not in cat and i not in self.empty_ix]
        self.numeric = None
        if member.kind != "standard" and self.numeric_ix:
            self.numeric = _NumericTransform(member.kind, X[:, self.numeric_ix], member.seed + 5)
        self.signs = None
        if member.flip_sign and self.numeric_ix:
            flips = np.random.default_rng(member.seed + 6).random(len(self.numeric_ix)) < 0.5
            self.signs = np.where(flips, -1.0, 1.0)
        # many-category columns that stay codes
        many = [
            i
            for i in self.cat_ix
            if i not in self.one_hot_sizes and n_categories[i] > MAX_ONE_HOT_CATEGORIES
        ]
        self.count_maps = {}
        if member.categories == "counts":
            for i in many:
                codes = X[:, i][np.isfinite(X[:, i])].astype(int)
                counts = np.bincount(codes, minlength=n_categories[i]).astype(np.float64)
                self.count_maps[i] = np.log1p(counts)

    @staticmethod
    def _lookup(codes: np.ndarray, table: np.ndarray) -> np.ndarray:
        # table[code] for codes 0..len(table)-1; missing and other codes are missing
        known = np.isfinite(codes) & (codes >= 0) & (codes < len(table))
        out = np.full(len(codes), np.nan)
        out[known] = table[codes[known].astype(int)]
        return out

    def _features(self, X: np.ndarray) -> np.ndarray:
        """The member's columns for clipped rows X, before the identifier and the shuffle: the
        numeric transform and sign flips on X, then `_base`, then the count columns."""
        counts = [self._lookup(X[:, i], t) for i, t in self.count_maps.items()]
        if self.numeric is not None or self.signs is not None:
            X = X.copy()
            if self.numeric is not None:
                X[:, self.numeric_ix] = self.numeric(X[:, self.numeric_ix])
            if self.signs is not None:
                X[:, self.numeric_ix] *= self.signs
        base = self._base(X)
        return np.concatenate([base, np.stack(counts, axis=1)], axis=1) if counts else base

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
        base = self._features(self._clip(X))
        return np.concatenate([base, identifier], axis=1)[:, self.order]
