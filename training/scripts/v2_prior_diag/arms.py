"""Inference arms of the v2 plan for priors and tasks, tier 1
(`training/docs/v2-prior-task-plan.md`, items 1.2 and 1.3): wrappers around the released inference
classes that the diagnostics and the benchmark runners of this folder share. Nothing in `tabsil`
changes; an arm that is kept would be moved there.

Forecast arms (`ArmForecaster`, names in `FORECAST_ARMS`):

- `default`: the weights' inference settings (0.1.1: 8 members over the numeric cycle, half of them
  with the target power transform)
- `no-target-power`: the same members without the target power transform, whose members keep their
  distribution inside the context's target range (`_clipped_inverse` of
  `tabsil.inference.ensemble`)
- `standard`: 8 members of the `standard` feature kind (the `robust` and `quantile` kinds saturate
  a running index beyond the context rows), half of them with the target power transform
- `plain8`: both changes
- `plain`: D1's earlier arm, 2 `standard` members (members 0 and 1 of the default cycle, which
  have no target power transform)
- any of them with `+trend`: a straight line is fitted by least squares to the context target on
  the row index (after the forecaster's filling, on the rows it keeps), the model forecasts the
  residual, and the line's continuation is added to the point forecast and to every quantile

Outlier scores (each `scorer(X_train, X_test, seed)` returns {arm name: scores, higher = more
anomalous}):

- `density_scorer`: the released score, the negative chain-rule log density with the whole context
- `small_context_scorer`: the log density of each test row under `n_contexts` random subsets of
  `rows` context rows (drawn without replacement), each with one random column order and 2
  members. Three readings of the same evaluations: `ctx<rows>` is the mean over the subsets of the
  negative log density (the arm of the plan), `ctx<rows>-median` their median, and
  `ctx<rows>-mix` the negative log of the mean density. `rows=None` uses the whole context for
  every column order (`ctxfull`), the control of the settings
- `reference_scorer`: the probability of the class "reference row" from `TabSilClassifier` fitted
  on the context rows (class 0) and as many reference rows (class 1). `uniform` draws them in the
  bounding box of the context, widened by `PAD` of its range on each side (categorical columns:
  uniform over the context's values); `marginals` permutes every column of the context on its
  own, which keeps the columns' distributions and removes their dependence. Mean over `draws`
  reference sets
- `refined_scorer`: the density score after the context rows with the highest score among
  themselves (`drop` of them) are left out
"""

from __future__ import annotations

import contextlib
import dataclasses
from collections.abc import Callable

import numpy as np
import pandas as pd

from tabsil import TabSilClassifier, TabSilForecaster, TabSilUnsupervised
from tabsil.inference.ensemble import EnsembleRegressor
from tabsil.tasks.unsupervised import infer_categorical

# arm -> (TabSilForecaster keyword arguments, whether members may use the target power transform)
FORECAST_ARMS: dict[str, tuple[dict, bool]] = {
    "default": ({}, True),
    "no-target-power": ({}, False),
    "standard": ({"member_kinds": ("standard",)}, True),
    "plain8": ({"member_kinds": ("standard",)}, False),
    "plain": ({"n_estimators": 2, "member_kinds": ("standard",)}, True),
}
TREND = "+trend"
# A residual whose range is below this share of the target's size is an exact line
EXACT_LINE = 1e-9
PAD = 0.1

Scorer = Callable[[np.ndarray, np.ndarray, int], dict[str, np.ndarray]]


@contextlib.contextmanager
def no_target_power():
    """Within the block, the members of every `EnsembleRegressor.fit` have no target power
    transform (`Member.target_power` False); seeds and feature kinds stay as they are."""
    original = EnsembleRegressor._default_members

    def members(self, rng):
        return [dataclasses.replace(m, target_power=False) for m in original(self, rng)]

    EnsembleRegressor._default_members = members
    try:
        yield
    finally:
        EnsembleRegressor._default_members = original


def split_arm(arm: str) -> tuple[str, bool]:
    """(base arm, whether the trend is fitted outside the model)."""
    base = arm.removesuffix(TREND)
    if base not in FORECAST_ARMS:
        known = [*FORECAST_ARMS, *(a + TREND for a in FORECAST_ARMS)]
        raise ValueError(f"unknown forecast arm {arm!r}: one of {known}")
    return base, arm.endswith(TREND)


def fit_line(y: np.ndarray) -> tuple[float, float]:
    """(intercept, slope) of the least-squares line through y on the index 0, 1, ..., n - 1."""
    n = len(y)
    if n < 2:
        return float(y[0]), 0.0
    slope, intercept = np.polyfit(np.arange(n, dtype=float), y, 1)
    return float(intercept), float(slope)


class ArmForecaster(TabSilForecaster):
    """`TabSilForecaster` with the recipe of one forecast arm (module docstring). The other
    keyword arguments are `TabSilForecaster`'s."""

    def __init__(self, model, borders, arm: str = "default", **kwargs) -> None:
        base, self.trend = split_arm(arm)
        options, self.target_power = FORECAST_ARMS[base]
        super().__init__(model, borders, **{**options, **kwargs})
        self.arm = arm

    def _predict_series(self, past: pd.DataFrame, steps: pd.DataFrame, levels: list[float]):
        with contextlib.nullcontext() if self.target_power else no_target_power():
            if not self.trend:
                return super()._predict_series(past, steps, levels)
            return self._predict_detrended(past, steps, levels)

    def _predict_detrended(self, past: pd.DataFrame, steps: pd.DataFrame, levels: list[float]):
        # the forecaster's own context handling, done here first so that the line is fitted to
        # the rows and the filled values the model gets: fill within the series, then keep the
        # last `max_context_length` rows
        target = self.target_column
        fill = [target, *(c for c in self.covariates_ if c in past.columns)]
        past = past.copy()
        past[fill] = past[fill].ffill().bfill()
        past[target] = past[target].fillna(0.0)
        past = past.iloc[-self.max_context_length :].reset_index(drop=True)
        y = past[target].to_numpy(float)
        n = len(y)
        intercept, slope = fit_line(y)
        residual = y - (intercept + slope * np.arange(n))
        if np.ptp(residual) <= EXACT_LINE * max(1.0, float(np.abs(y).max())):
            residual = np.zeros(n)  # the forecaster returns a constant target as it is
        past[target] = residual
        point, quantiles = super()._predict_series(past, steps, levels)
        line = intercept + slope * (n + np.arange(len(steps)))
        return point + line, quantiles + line[:, None]


# ---------------------------------------------------------------------------
# Outlier scores
# ---------------------------------------------------------------------------
def density_scorer(
    model, borders, device: str, n_permutations: int = 10, n_estimators: int | None = None
) -> Scorer:
    def score(X_train: np.ndarray, X_test: np.ndarray, seed: int) -> dict[str, np.ndarray]:
        est = TabSilUnsupervised(
            model,
            borders,
            n_permutations=n_permutations,
            n_estimators=n_estimators,
            random_state=seed,
            device=device,
        )
        return {"density": est.fit(X_train).outlier_score(X_test)}

    return score


def context_log_densities(
    model,
    borders,
    device: str,
    X_train: np.ndarray,
    X_test: np.ndarray,
    seed: int,
    rows: int | None,
    n_contexts: int,
    n_estimators: int,
) -> np.ndarray:
    """(n_contexts, test rows) log densities: context b is `rows` random rows of `X_train` (all
    of them if `rows` is None or not below their count) with its own column order and member
    seeds. The categorical columns are those of the whole context for every subset."""
    categorical = infer_categorical(X_train)
    rng = np.random.default_rng([seed, rows or 0])
    subset = rows is not None and rows < len(X_train)
    out = []
    for b in range(n_contexts):
        context = (
            X_train[np.sort(rng.choice(len(X_train), rows, replace=False))] if subset else X_train
        )
        est = TabSilUnsupervised(
            model,
            borders,
            n_permutations=1,
            categorical_features=categorical,
            n_estimators=n_estimators,
            random_state=seed + 1000 * (b + 1),
            device=device,
        )
        out.append(est.fit(context).log_density(X_test))
    return np.stack(out)


def aggregate_contexts(name: str, log_densities: np.ndarray) -> dict[str, np.ndarray]:
    """The three readings of per-context log densities (contexts, rows) (module docstring)."""
    top = log_densities.max(0)
    mixture = top + np.log(np.exp(log_densities - top).mean(0))
    return {
        name: -log_densities.mean(0),
        f"{name}-median": -np.median(log_densities, axis=0),
        f"{name}-mix": -mixture,
    }


def small_context_scorer(
    model, borders, device: str, rows: int | None, n_contexts: int = 32, n_estimators: int = 2
) -> Scorer:
    name = f"ctx{rows}" if rows else "ctxfull"

    def score(X_train: np.ndarray, X_test: np.ndarray, seed: int) -> dict[str, np.ndarray]:
        logs = context_log_densities(
            model, borders, device, X_train, X_test, seed, rows, n_contexts, n_estimators
        )
        return aggregate_contexts(name, logs)

    return score


def reference_rows(X: np.ndarray, kind: str, rng: np.random.Generator) -> np.ndarray:
    """As many reference rows as X has rows (module docstring, `reference_scorer`)."""
    n, d = X.shape
    if kind == "marginals":
        return np.column_stack([rng.permutation(X[:, j]) for j in range(d)])
    if kind != "uniform":
        raise ValueError(f"unknown reference kind {kind!r}: uniform or marginals")
    lo, hi = np.nanmin(X, 0), np.nanmax(X, 0)
    span = hi - lo
    out = rng.uniform(lo - PAD * span, hi + PAD * span, size=(n, d))
    for j in infer_categorical(X):
        values = np.unique(X[~np.isnan(X[:, j]), j])
        out[:, j] = rng.choice(values, n)
    return out


def reference_scorer(
    model, device: str, kind: str, draws: int = 3, n_estimators: int | None = None
) -> Scorer:
    def score(X_train: np.ndarray, X_test: np.ndarray, seed: int) -> dict[str, np.ndarray]:
        categorical = infer_categorical(X_train) or None
        labels = np.r_[np.zeros(len(X_train), int), np.ones(len(X_train), int)]
        total = np.zeros(len(X_test))
        for k in range(draws):
            rng = np.random.default_rng([seed, k, 7])
            rows = np.vstack([X_train, reference_rows(X_train, kind, rng)])
            clf = TabSilClassifier(
                model,
                n_estimators=n_estimators,
                categorical_features_indices=categorical,
                random_state=seed + 1000 * k,
                device=device,
            ).fit(rows, labels)
            total += clf.predict_proba(X_test)[:, list(clf.classes_).index(1)]
        return {f"ref-{kind}": total / draws}

    return score


def refined_scorer(
    model, borders, device: str, drop: float = 0.1, n_permutations: int = 10
) -> Scorer:
    def score(X_train: np.ndarray, X_test: np.ndarray, seed: int) -> dict[str, np.ndarray]:
        est = TabSilUnsupervised(
            model, borders, n_permutations=n_permutations, random_state=seed, device=device
        )
        own = est.fit(X_train).outlier_score(X_train)
        keep = np.sort(np.argsort(own, kind="stable")[: len(X_train) - round(drop * len(X_train))])
        return {"refine": est.fit(X_train[keep]).outlier_score(X_test)}

    return score


def outlier_scorers(model, borders, device: str, n_permutations: int = 10) -> dict[str, Scorer]:
    """Scorer name -> scorer. A scorer's arms are named after it (`ctx64` gives `ctx64`,
    `ctx64-median` and `ctx64-mix`). `n_permutations` is the column order count of the whole
    context scores (`density`, `refine`)."""
    return {
        "density": density_scorer(model, borders, device, n_permutations),
        "ctx64": small_context_scorer(model, borders, device, 64),
        "ctx256": small_context_scorer(model, borders, device, 256),
        "ctxfull": small_context_scorer(model, borders, device, None, n_contexts=8),
        "ref-uniform": reference_scorer(model, device, "uniform"),
        "ref-marginals": reference_scorer(model, device, "marginals"),
        "refine": refined_scorer(model, borders, device, n_permutations=n_permutations),
    }
