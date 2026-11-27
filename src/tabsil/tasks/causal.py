"""Treatment effect estimation with TabSil as the base learner (meta-learners, zero-shot).

`TabSilCATE` estimates the conditional average treatment effect τ(x) = E[Y(1) − Y(0) | X = x] of a
binary treatment on a continuous outcome from observational data (X, T, Y), under the usual
assumptions of these estimators (no unmeasured confounding, overlap). The outcome and propensity
models are the regular inference classes: `TabSilRegressor.predict` (the predictive mean) for the
outcome and `TabSilClassifier.predict_proba` for the propensity score e(x) = P(T = 1 | X = x). No
weights are trained; each base model is one in-context fit.

    from tabsil.causal import TabSilCATE

    est = TabSilCATE(device="cuda").fit(X, treatment, y)  # the S-learner
    tau = est.effect(X_test)    # (n,) CATE per row
    ate = est.ate()             # mean CATE over the training rows

Learners (`learner`):

- `"s"` (S-learner): one outcome model μ(x, t) on the features plus the treatment as the last
  column; τ(x) = μ(x, 1) − μ(x, 0)
- `"t"` (T-learner): μ₀ on the control rows and μ₁ on the treated rows; τ(x) = μ₁(x) − μ₀(x)
- `"x"` (X-learner, Künzel et al. 2019, PNAS 116(10)): the T-learner's μ₀, μ₁; imputed effects
  D₁ = Y − μ₀(X) on the treated and D₀ = μ₁(X) − Y on the control rows; τ₁, τ₀ regress D₁, D₀ on X
  within each group; τ(x) = e(x) τ₀(x) + (1 − e(x)) τ₁(x)
- `"dr"` (DR-learner, Kennedy 2023, EJS 17(2)): μ₀, μ₁ and e are cross-fitted over `n_folds`
  folds stratified by treatment (each row's nuisances come from the folds without it), the doubly
  robust pseudo-outcome φ = μ₁ − μ₀ + T (Y − μ₁) / e − (1 − T) (Y − μ₀) / (1 − e) is computed with
  e clipped to [`propensity_clip`, 1 − `propensity_clip`], and τ regresses φ on X. The mean of φ
  is the AIPW estimate of the ATE (`pseudo_outcomes_`)

`ate(X)` is the mean of `effect(X)` (default: over the training rows). `effect` of the training
rows themselves predicts rows that are in the base models' context (as any fitted model predicts
its own training rows); only the DR-learner's nuisances are out of fold.

Where this differs from the reference implementations (EconML 0.17.1 `SLearner`, `TLearner`,
`XLearner`, `DRLearner`):

- base learners: TabSil in-context fits with the model's inference defaults (8 members), not
  fitted estimators; `regressor` / `classifier` take any scikit-learn estimator instead (cloned for
  every fit), so the same meta-learner can run with another base learner
- S-learner: the treatment is one 0/1 column appended to the features, an ordinary numeric column
  for TabSil (EconML appends two columns, 1 − T and T). With TabSil, tables whose features plus
  the treatment exceed the per-member column cap (768) are an error, since members drawing column
  subsets could leave the treatment out
- X-learner: the propensity model is TabSil classification fitted on all rows (EconML's default is
  a logistic regression)
- DR-learner: `n_folds` = 5 stratified folds (EconML's default `cv` is 2), and the propensity is
  clipped at 0.01 (EconML's `min_propensity` is 1e-6); the final model is TabSil regression on φ
  (EconML's default `model_final` is a linear regression, so its default CATE is linear in x)
- a base model whose target is constant (a group whose outcomes are all equal, for example)
  predicts that constant without running the model (z-normalizing a constant target is undefined)
- a non-finite prediction of any base model raises `ValueError` instead of propagating

Binary treatments (0/1 or booleans) and continuous outcomes only. Features are numeric (encode
categorical columns beforehand); NaN is passed on to the base learners (TabSil accepts it).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

LEARNERS = ("s", "t", "x", "dr")
# The S-learner: the best of the four on IHDP and second on ACIC 2016 (best average rank,
# docs/task-results.md "Treatment effects"), and the cheapest (one outcome fit)
DEFAULT_LEARNER = "s"
DEFAULT_N_FOLDS = 5
DEFAULT_PROPENSITY_CLIP = 0.01


class _Constant:
    """Base model of a constant target: predicts that value."""

    def __init__(self, value: float) -> None:
        self.value = float(value)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(len(X), self.value)


def _features(X: Any) -> np.ndarray:
    try:
        X = np.asarray(X, dtype=float)
    except (TypeError, ValueError) as e:
        raise ValueError(f"features must be numeric (encode categorical columns): {e}") from None
    if X.ndim != 2:
        raise ValueError(f"X must be 2-dimensional (rows, features): shape {X.shape}")
    if np.isinf(X).any():
        raise ValueError("X has infinite values")
    return X


def _treatment(treatment: Any, n: int) -> np.ndarray:
    t = np.asarray(treatment)
    if t.ndim != 1 or len(t) != n:
        raise ValueError(f"treatment must be one value per row ({n}): shape {t.shape}")
    if t.dtype == bool:
        t = t.astype(int)
    try:
        values = t.astype(float)
    except (TypeError, ValueError):
        raise ValueError("treatment must be binary: 0/1 or booleans") from None
    if not np.isin(values, (0.0, 1.0)).all():
        raise ValueError(f"treatment must be binary 0/1: values {np.unique(values)[:5]}")
    return values.astype(int)


def _outcome(y: Any, n: int) -> np.ndarray:
    try:
        y = np.asarray(y, dtype=float)
    except (TypeError, ValueError):
        raise ValueError("the outcome must be numeric") from None
    if y.ndim != 1 or len(y) != n:
        raise ValueError(f"the outcome must be one value per row ({n}): shape {y.shape}")
    if not np.isfinite(y).all():
        raise ValueError("the outcome has missing or infinite values")
    return y


def _finite(values: np.ndarray, what: str) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        bad = int((~np.isfinite(values)).sum())
        raise ValueError(f"{what}: {bad} non-finite predictions")
    return values


class TabSilCATE:
    """Conditional average treatment effects with TabSil base learners (module docstring for the
    learners and where they differ from EconML).

    `model` and `borders` are given as for `tabsil.recipe.TabSilRegressor` (None: the default Hub
    repo at its default tag, `tabsil.recipe.HUB_DEFAULT`; a Hub repo id or a local folder; or a
    model with its `borders`). The model must serve regression (outcomes) and, for the X- and
    DR-learners, classification (propensity scores); the weights are loaded once and shared by
    every base model. `n_estimators`,
    `device` and the other keyword arguments go to each `TabSilRegressor` / `TabSilClassifier`.
    `regressor` / `classifier` replace TabSil by a scikit-learn estimator (cloned per fit); when
    both replace it, or the learner needs no classifier, no TabSil weights are loaded.

        est = TabSilCATE(learner="dr").fit(X, treatment, y)
        est.effect(X_test), est.ate()
    """

    def __init__(
        self,
        model: nn.Module | str | Path | None = None,
        borders: torch.Tensor | None = None,
        *,
        learner: str = DEFAULT_LEARNER,
        n_folds: int = DEFAULT_N_FOLDS,
        propensity_clip: float = DEFAULT_PROPENSITY_CLIP,
        regressor: Any = None,
        classifier: Any = None,
        n_estimators: int | None = None,
        device: str = "cpu",
        random_state: int = 0,
        **estimator_kwargs: Any,
    ) -> None:
        from tabsil.tasks.regression import check_borders

        if learner not in LEARNERS:
            raise ValueError(f"learner is one of {LEARNERS}: {learner!r}")
        if n_folds < 2:
            raise ValueError(f"n_folds must be >= 2: {n_folds}")
        if not 0 < propensity_clip < 0.5:
            raise ValueError(f"propensity_clip must be in (0, 0.5): {propensity_clip}")
        # the TabSil tasks this learner uses: outcomes (unless `regressor`), propensity (X and DR,
        # unless `classifier`)
        needs_regression = regressor is None
        needs_classification = classifier is None and learner in ("x", "dr")
        check_borders(model, borders, required=needs_regression)
        if not isinstance(model, nn.Module) and (needs_regression or needs_classification):
            from tabsil.hub import model_or_pretrained
            from tabsil.recipe import HUB_DEFAULT

            task = "regression" if needs_regression else "classification"
            model, borders = model_or_pretrained(model, HUB_DEFAULT, task)
        tasks = getattr(model, "tasks", None)
        if needs_classification and tasks is not None and "classification" not in tasks:
            raise ValueError(
                f"the {learner}-learner's propensity model needs a model serving classification"
                " (or pass `classifier`)"
            )
        self.model = model
        self.borders = borders
        self.learner = learner
        self.n_folds = int(n_folds)
        self.propensity_clip = float(propensity_clip)
        self.regressor = regressor
        self.classifier = classifier
        self.n_estimators = n_estimators
        self.device = device
        self.random_state = random_state
        self.estimator_kwargs = estimator_kwargs

    # --- base models ---------------------------------------------------------------------------

    def _seed(self) -> int:
        # a different seed for every base model of a fit, the same sequence for the same state
        self._fits += 1
        return self.random_state + self._fits

    def _fit_regressor(self, X: np.ndarray, y: np.ndarray):
        if np.ptp(y) == 0:
            return _Constant(y[0])
        if self.regressor is not None:
            from sklearn.base import clone

            return clone(self.regressor).fit(X, y)
        from tabsil.tasks.regression import TabSilRegressor

        return TabSilRegressor(
            self.model,
            self.borders,
            n_estimators=self.n_estimators,
            device=self.device,
            random_state=self._seed(),
            **self.estimator_kwargs,
        ).fit(X, y)

    def _fit_propensity(self, X: np.ndarray, t: np.ndarray):
        if self.classifier is not None:
            from sklearn.base import clone

            return clone(self.classifier).fit(X, t)
        from tabsil.tasks.classification import TabSilClassifier

        return TabSilClassifier(
            self.model,
            n_estimators=self.n_estimators,
            device=self.device,
            random_state=self._seed(),
            **self.estimator_kwargs,
        ).fit(X, t)

    @staticmethod
    def _propensity(model, X: np.ndarray) -> np.ndarray:
        proba = _finite(model.predict_proba(X), "propensity model")
        classes = list(np.asarray(model.classes_).astype(int))
        return proba[:, classes.index(1)]

    # --- fit -----------------------------------------------------------------------------------

    def fit(self, X: Any, treatment: Any, y: Any) -> TabSilCATE:
        """Fit the learner's base models on the rows (X, treatment, y): features (n, d), a binary
        treatment (n,) and a continuous outcome (n,). Each treatment group needs at least two
        rows (the DR-learner: at least `n_folds`)."""
        X = _features(X)
        t = _treatment(treatment, len(X))
        y = _outcome(y, len(X))
        need = self.n_folds if self.learner == "dr" else 2
        counts = {"control": int((t == 0).sum()), "treated": int((t == 1).sum())}
        if min(counts.values()) < need:
            raise ValueError(f"each treatment group needs at least {need} rows: {counts}")
        self._fits = 0
        self.n_features_in_ = X.shape[1]
        self.X_ = X
        getattr(self, f"_fit_{self.learner}")(X, t, y)
        return self

    def _fit_s(self, X, t, y):
        if self.regressor is None:
            # beyond the per-member column cap, members draw column subsets, and a member without
            # the treatment column would give every row an effect of exactly 0
            from tabsil.recipe import default_inference

            cap = self.estimator_kwargs.get(
                "max_features_per_member",
                default_inference(self.model, "regression")["max_features_per_member"],
            )
            if cap is not None and X.shape[1] + 1 > cap:
                raise ValueError(
                    f"the S-learner adds the treatment to {X.shape[1]} features, more than the"
                    f" {cap} columns per member, so some members would not see it: use another"
                    " learner or max_features_per_member=None"
                )
        self.mu_ = self._fit_regressor(np.column_stack([X, t]), y)

    def _fit_t(self, X, t, y):
        self.mu0_ = self._fit_regressor(X[t == 0], y[t == 0])
        self.mu1_ = self._fit_regressor(X[t == 1], y[t == 1])

    def _fit_x(self, X, t, y):
        self._fit_t(X, t, y)
        X0, X1 = X[t == 0], X[t == 1]
        d1 = y[t == 1] - _finite(self.mu0_.predict(X1), "control outcome model")
        d0 = _finite(self.mu1_.predict(X0), "treated outcome model") - y[t == 0]
        self.tau0_ = self._fit_regressor(X0, d0)
        self.tau1_ = self._fit_regressor(X1, d1)
        self.propensity_ = self._fit_propensity(X, t)

    def _fit_dr(self, X, t, y):
        from sklearn.model_selection import StratifiedKFold

        mu0, mu1, e = np.empty(len(X)), np.empty(len(X)), np.empty(len(X))
        folds = StratifiedKFold(self.n_folds, shuffle=True, random_state=self.random_state)
        for train, held in folds.split(X, t):
            Xf, tf, yf = X[train], t[train], y[train]
            m0 = self._fit_regressor(Xf[tf == 0], yf[tf == 0])
            m1 = self._fit_regressor(Xf[tf == 1], yf[tf == 1])
            mu0[held] = _finite(m0.predict(X[held]), "control outcome model")
            mu1[held] = _finite(m1.predict(X[held]), "treated outcome model")
            e[held] = self._propensity(self._fit_propensity(Xf, tf), X[held])
        e = np.clip(e, self.propensity_clip, 1 - self.propensity_clip)
        phi = mu1 - mu0 + t * (y - mu1) / e - (1 - t) * (y - mu0) / (1 - e)
        self.pseudo_outcomes_ = phi
        self.tau_ = self._fit_regressor(X, phi)

    # --- effects -------------------------------------------------------------------------------

    def effect(self, X: Any = None) -> np.ndarray:
        """The CATE τ(x) of each row of X (default: the training rows), shape (n,)."""
        if not hasattr(self, "X_"):
            raise RuntimeError("call fit before effect")
        X = self.X_ if X is None else _features(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, fitted with {self.n_features_in_}")
        if self.learner == "s":
            n = len(X)
            both = self.mu_.predict(np.vstack([np.column_stack([X, np.ones(n)]),
                                               np.column_stack([X, np.zeros(n)])]))  # fmt: skip
            both = _finite(both, "outcome model")
            return both[:n] - both[n:]
        if self.learner == "t":
            mu1 = _finite(self.mu1_.predict(X), "treated outcome model")
            return mu1 - _finite(self.mu0_.predict(X), "control outcome model")
        if self.learner == "x":
            e = self._propensity(self.propensity_, X)
            tau0 = _finite(self.tau0_.predict(X), "control effect model")
            tau1 = _finite(self.tau1_.predict(X), "treated effect model")
            return e * tau0 + (1 - e) * tau1
        return _finite(self.tau_.predict(X), "effect model")

    def ate(self, X: Any = None) -> float:
        """The average treatment effect: the mean of `effect(X)` (default: over the training
        rows)."""
        return float(np.mean(self.effect(X)))
