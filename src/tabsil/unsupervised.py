"""Unsupervised TabSil: per-column conditional distributions, row log densities and outlier scores.

The part everything here stands on is the conditional distribution of one column given other
columns (`TabSilUnsupervised.conditional`), computed by the regular TabSil inference classes with
the column as the target and the other columns as features:

- numeric column: `TabSilRegressor.predict_distribution` (bucket probabilities on the shared
  borders, in the column's original units); the density at a value is the probability of its
  bucket divided by the bucket width (`NumericConditional`)
- categorical column: `TabSilClassifier.predict_proba` (`CategoricalConditional`)

A row's log density is the chain rule over a column order, log p(x) = Σ_i log p(x_{π_i} | x_{π_<i}),
averaged over several random orders as the log of the arithmetic mean of the densities
(log-sum-exp). The outlier score is the negative log density (higher = more anomalous).

    from tabsil.unsupervised import TabSilUnsupervised

    model = TabSilUnsupervised().fit(x_train)  # the default Hub weights (`tabsil.recipe`)
    scores = model.outlier_score(x_test)       # (n,), higher = more anomalous
    log_p = model.log_density(x_test)          # (n,), = -scores

Imputation and synthetic data can later be built on the same part (`conditional`: the distribution
of a missing column given the observed ones, or sampled column by column along an order); they
are not implemented here.

The method follows the official TabPFN extension (`TabPFNUnsupervisedModel.outliers` of
PriorLabs/tabpfn-extensions a049fd7, `src/tabpfn_extensions/unsupervised/unsupervised.py`,
Apache-2.0): random column orders (10 by default), the chain rule per order, a standard normal noise
column as the only feature of the first column in an order, the log-sum-exp mean over orders, and
the categorical rule of `infer_categorical_features` (at most 40 distinct values and more than 10
rows per value on average). Where this module differs:

- category values: the official casts a categorical column to `int` to get class indices, which
  merges the values of a min-max scaled column into 0 and 1; here the distinct context values are
  the classes (exact match). A value not in the context, a class index the official sends to a
  fixed probability of 0.1, gets `PROBABILITY_FLOOR` here
- density lookup: the official takes the bar distribution NLL of its regressor; here the bucket of
  a value is found on the borders of `predict_distribution` (a value outside the borders gets
  `PROBABILITY_FLOOR` over the outermost bucket width)
- one evaluation per (column, conditioning set): a conditional density does not depend on the
  order of the conditioning columns, so orders sharing a prefix set reuse it, and the
  conditioning columns enter in index order. The first and the last position of all orders have
  only one pair per column (the marginal, and the column given all others), so with d columns at
  most 2d of those evaluations are left instead of 2 x `n_permutations`
- the column orders come from `random_state` (numpy) rather than Python's global `random`, and are
  all orders when there are no more than `n_permutations` of them
- categorical conditioning columns are passed to the inference classes as categorical
  (`categorical_features_indices`, one-hot encoded by TabSil's preprocessing); TabPFN infers them
- missing values: a missing target value is integrated out (contributes 0 to the log density) and
  context rows missing the target are left out of that column's context; missing conditioning
  values go to the model as NaN. (The official drops context rows missing the target too.)

Cost: every (column, conditioning set) is one fit and one prediction of the test rows with the
whole context, so up to `n_permutations` x columns predictions of `n_estimators` members each.
`n_estimators` (fewer members) is the main lever; the inference package is used as is.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import permutations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

# Probability given to an outcome the model gives no mass: a category not in the context, a numeric
# value outside the borders. Same as the clamp of `EnsembleRegressor.predict_distribution`
PROBABILITY_FLOOR = 1e-12
# The categorical rule of tabpfn-extensions `infer_categorical_features`
MAX_CATEGORIES = 40
MIN_ROWS_PER_CATEGORY = 10
N_PERMUTATIONS = 10


def infer_categorical(
    X: np.ndarray,
    max_categories: int = MAX_CATEGORIES,
    min_rows_per_category: int = MIN_ROWS_PER_CATEGORY,
) -> list[int]:
    """Columns treated as categorical: at most `max_categories` distinct values (missing values
    not counted) and more than `min_rows_per_category` rows per value on average (the rule of
    tabpfn-extensions `infer_categorical_features`)."""
    out = []
    for j in range(X.shape[1]):
        col = X[:, j]
        n_unique = len(np.unique(col[~np.isnan(col)]))
        if 0 < n_unique <= max_categories and X.shape[0] / n_unique > min_rows_per_category:
            out.append(j)
    return out


def column_orders(n_columns: int, n_permutations: int, random_state: int) -> list[tuple[int, ...]]:
    """`n_permutations` distinct random column orders, or all orders when there are not more than
    that."""
    if math.factorial(n_columns) <= n_permutations:
        return list(permutations(range(n_columns)))
    rng = np.random.default_rng(random_state)
    out: list[tuple[int, ...]] = []
    while len(out) < n_permutations:
        order = tuple(int(i) for i in rng.permutation(n_columns))
        if order not in out:
            out.append(order)
    return out


@dataclass
class NumericConditional:
    """The predictive distribution of a numeric column for each row: bucket log probabilities
    `log_probs` (n, K) on `borders` (K + 1,), in the column's original units (float64)."""

    log_probs: np.ndarray
    borders: np.ndarray

    def log_prob(self, values: np.ndarray) -> np.ndarray:
        """Log density at `values` (n,): the log probability of the bucket holding the value
        minus the log of its width. Buckets are [b_k, b_{k+1}) (the last one closed), so a value
        on a border of zero-width buckets falls in the next bucket of positive width. A value
        outside the borders gets `PROBABILITY_FLOOR` over the width of the nearest outer bucket.
        A missing value gives 0 (integrated out)."""
        values = np.asarray(values, dtype=np.float64)
        b = self.borders
        n_buckets = len(b) - 1
        idx = np.searchsorted(b, values, side="right") - 1
        idx = np.where(values == b[-1], n_buckets - 1, idx)  # the last bucket includes its end
        inside = (idx >= 0) & (idx < n_buckets)
        idx = idx.clip(0, n_buckets - 1)
        width = np.maximum(b[idx + 1] - b[idx], np.finfo(np.float64).tiny)
        log_p = self.log_probs[np.arange(len(values)), idx]
        out = np.where(inside, log_p, math.log(PROBABILITY_FLOOR)) - np.log(width)
        return np.where(np.isnan(values), 0.0, out)


@dataclass
class CategoricalConditional:
    """The predictive distribution of a categorical column for each row: probabilities `proba`
    (n, C) of the context's distinct values `classes` (C,)."""

    proba: np.ndarray
    classes: np.ndarray

    def log_prob(self, values: np.ndarray) -> np.ndarray:
        """Log probability of `values` (n,). A value not among `classes` gets
        `PROBABILITY_FLOOR`; a missing value gives 0 (integrated out)."""
        values = np.asarray(values, dtype=np.float64)
        idx = np.searchsorted(self.classes, values).clip(0, len(self.classes) - 1)
        known = self.classes[idx] == values
        p = np.where(known, self.proba[np.arange(len(values)), idx], 0.0)
        out = np.log(np.maximum(p, PROBABILITY_FLOOR))
        return np.where(np.isnan(values), 0.0, out)


Conditional = NumericConditional | CategoricalConditional


def _numeric_array(X: Any) -> np.ndarray:
    if isinstance(X, pd.DataFrame):
        bad = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
        if bad:
            raise ValueError(f"numeric columns only (encode categories as numbers): {bad}")
        X = X.to_numpy()
    if torch.is_tensor(X):
        X = X.detach().cpu().numpy()
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-dimensional (rows, columns): shape {X.shape}")
    return X


class TabSilUnsupervised:
    """Column-wise conditional distributions of a table with TabSil, and from them the row log
    density and outlier score (module docstring).

    `model` is a Hub repo id or local folder serving both tasks (None: the default Hub weights,
    `tabsil.recipe.HUB_DEFAULT`) or a model together with its regression `borders`.
    `categorical_features` lists the categorical columns (None: `infer_categorical` on the fit
    rows). `n_estimators` and the other keyword arguments go to every `TabSilClassifier` and
    `TabSilRegressor` (`device`, `max_features_per_member`, ...).

    Fitting only stores the rows (the context); every density call fits and predicts with them.
    """

    def __init__(
        self,
        model: nn.Module | str | Path | None = None,
        borders: torch.Tensor | None = None,
        *,
        n_permutations: int = N_PERMUTATIONS,
        categorical_features: list[int] | None = None,
        n_estimators: int | None = None,
        random_state: int = 0,
        **estimator_kwargs: Any,
    ) -> None:
        from tabsil.hub import model_or_pretrained
        from tabsil.recipe import HUB_DEFAULT

        if isinstance(model, nn.Module):
            if borders is None:
                raise ValueError("pass the regression borders together with a model")
        else:
            if borders is not None:
                raise ValueError("borders are only given with a model (downloads include them)")
            model, borders = model_or_pretrained(model, HUB_DEFAULT, "regression")
            if "classification" not in model.tasks:
                raise ValueError("the model must serve classification as well as regression")
        if n_permutations < 1:
            raise ValueError(f"n_permutations must be >= 1: {n_permutations}")
        self.model = model
        self.borders = borders
        self.n_permutations = n_permutations
        self.categorical_features = categorical_features
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.estimator_kwargs = estimator_kwargs

    # --- fit -------------------------------------------------------------------------------

    def fit(self, X: Any, y: Any = None) -> TabSilUnsupervised:
        """Store the context rows `X` (numeric, NaN for missing) and find the categorical
        columns and their values. `y` is ignored."""
        X = _numeric_array(X)
        self.X_ = X
        self.n_features_in_ = X.shape[1]
        if self.categorical_features is None:
            self.categorical_ = infer_categorical(X)
        else:
            self.categorical_ = sorted(int(j) for j in self.categorical_features)
            if any(j < 0 or j >= X.shape[1] for j in self.categorical_):
                raise ValueError(f"categorical_features out of range: {self.categorical_features}")
        self.classes_ = {j: np.unique(X[~np.isnan(X[:, j]), j]) for j in self.categorical_}
        return self

    def _check_fitted(self, X: Any) -> np.ndarray:
        if not hasattr(self, "X_"):
            raise RuntimeError("call fit first")
        X = _numeric_array(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"{X.shape[1]} columns, fitted with {self.n_features_in_}")
        return X

    def _uses_classifier(self, column: int) -> bool:
        # as the official: categorical, and no more classes than the classifier takes
        from tabsil.inference.ensemble import max_classes_of

        return column in self.classes_ and len(self.classes_[column]) <= max_classes_of(self.model)

    def _estimator(self, classifier: bool, categorical: list[int]):
        from tabsil.recipe import TabSilClassifier, TabSilRegressor

        kwargs = {
            "n_estimators": self.n_estimators,
            "random_state": self.random_state,
            "categorical_features_indices": categorical or None,
            **self.estimator_kwargs,
        }
        if classifier:
            return TabSilClassifier(self.model, **kwargs)
        return TabSilRegressor(self.model, self.borders, **kwargs)

    # --- the conditional distribution ------------------------------------------------------

    def conditional(self, X: Any, target: int, given: list[int] | tuple[int, ...]) -> Conditional:
        """The distribution of column `target` for each row of `X` given its columns `given`,
        learned in context from the fit rows (that observe `target`). With no `given` columns
        the only feature is a standard normal noise column (as the official), so this is the
        column's marginal."""
        X = self._check_fitted(X)
        given = sorted(int(j) for j in given)
        if target in given:
            raise ValueError(f"target {target} is among the given columns {given}")
        context = self.X_[~np.isnan(self.X_[:, target])]
        if len(context) == 0:
            raise ValueError(f"column {target} has no observed value in the fit rows")
        y = context[:, target]
        classifier = self._uses_classifier(target)
        if classifier and len(self.classes_[target]) == 1:  # a constant column: nothing to fit
            return CategoricalConditional(np.ones((len(X), 1)), self.classes_[target])
        if given:
            x_fit, x_pred = context[:, given], X[:, given]
            categorical = [k for k, j in enumerate(given) if j in self.classes_]
        else:
            rng = np.random.default_rng([self.random_state, target])
            x_fit = rng.standard_normal((len(context), 1))
            x_pred = np.random.default_rng([self.random_state, target, 1]).standard_normal(
                (len(X), 1)
            )
            categorical = []
        est = self._estimator(classifier, categorical)
        if classifier:
            codes = np.searchsorted(self.classes_[target], y)
            est.fit(x_fit, codes)
            proba = np.zeros((len(X), len(self.classes_[target])))
            proba[:, est.classes_.astype(int)] = est.predict_proba(x_pred)
            return CategoricalConditional(proba, self.classes_[target])
        est.fit(x_fit, y)
        log_probs, borders = est.predict_distribution(x_pred)
        return NumericConditional(log_probs.double().numpy(), borders.double().numpy())

    # --- densities -------------------------------------------------------------------------

    def column_orders(self) -> list[tuple[int, ...]]:
        """The column orders the chain rule is averaged over (`column_orders`)."""
        return column_orders(self.n_features_in_, self.n_permutations, self.random_state)

    def log_density_per_order(self, X: Any) -> tuple[np.ndarray, list[tuple[int, ...]]]:
        """(chain-rule log densities (orders, n), the orders). Each (column, conditioning set) is
        evaluated once and shared by the orders that contain it."""
        X = self._check_fitted(X)
        orders = self.column_orders()
        cache: dict[tuple[int, tuple[int, ...]], np.ndarray] = {}
        out = np.zeros((len(orders), len(X)))
        for k, order in enumerate(orders):
            for i, target in enumerate(order):
                key = (target, tuple(sorted(order[:i])))
                if key not in cache:
                    cache[key] = self.conditional(X, target, key[1]).log_prob(X[:, target])
                out[k] += cache[key]
        return out, orders

    def log_density(self, X: Any) -> np.ndarray:
        """Log density of each row of `X` (n,): the log of the mean over the column orders of the
        chain-rule densities. Numeric columns contribute densities in their original units,
        categorical columns probabilities."""
        per_order, orders = self.log_density_per_order(X)
        m = per_order.max(0)
        return m + np.log(np.exp(per_order - m).mean(0))

    def outlier_score(self, X: Any) -> np.ndarray:
        """Outlier score of each row of `X` (n,): the negative log density, higher = more
        anomalous."""
        return -self.log_density(X)
