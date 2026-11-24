"""Wrap an sklearn-style estimator as a `tabarena` system (`ExternalSystemModel`).

This module imports `tabarena`, so it is loaded only when `tabsil.benchmarks.arena` needs it.
System configs may be saved as yaml, so estimator factories live in this module's registry and
only their names are passed (`debug_mode=True` runs in the same process, so the registry stays
visible).

If the factory has `fit_groups = True`, fit also receives `groups` (a group id per row), built
from the task's split info (group/time columns in `validation_metadata`;
`tabsil.benchmarks.arena.split_groups`). If it has `bag_pool`
(`tabsil.benchmarks.bag_pool.BagPool`), the bagging bags are fit and predicted concurrently by
the pool's GPU workers instead of this process. Bag splitting and prediction averaging are the
same.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from tabarena.benchmark.exec_models import ExternalSystemModel

from tabsil.benchmarks.arena import split_groups

_FACTORIES: dict[str, Callable[[str, list[int]], Any]] = {}


def register_factory(make_estimator: Callable[[str, list[int]], Any]) -> str:
    # same function, same name: workers that call `arena.run` per dataset (`--shards`) run with
    # the same config (factory0) as `--shards 1`
    for key, registered in _FACTORIES.items():
        if registered is make_estimator:
            return key
    key = f"factory{len(_FACTORIES)}"
    _FACTORIES[key] = make_estimator
    return key


def experiments(suite, method: str, key: str, bagging: int) -> list:
    from tabarena.benchmark.experiment import (
        BeyondArenaExperimentBundle,
        TabArenaV0pt1ExperimentBundle,
    )
    from tabarena.utils.config_utils import SystemConfigGenerator

    generator = SystemConfigGenerator(
        model_cls=EstimatorSystem,
        name=method,
        manual_configs=[{"factory": key, "bagging": bagging}],
    )
    bundle = (
        BeyondArenaExperimentBundle
        if suite.context == "BeyondArenaContext"
        else TabArenaV0pt1ExperimentBundle
    )
    return bundle(models=[(generator, 0)], system_experiments=True).build_experiments()


def prepare_frame(X: pd.DataFrame) -> tuple[pd.DataFrame, list[int]]:
    """Convert datetime columns to seconds and return non-numeric columns (categorical, text,
    non-bool object) as categorical indices. Column order is unchanged."""
    X = X.copy()
    categorical = []
    for i, col in enumerate(X.columns):
        s = X[col]
        if pd.api.types.is_datetime64_any_dtype(s):
            s = s.dt.tz_localize(None) if getattr(s.dt, "tz", None) is not None else s
            # seconds since 1970-01-01 regardless of unit (datetime64[ns]/[us]/[s]); NaT -> NaN
            X[col] = (s - pd.Timestamp(0)) / pd.Timedelta(seconds=1)
        elif pd.api.types.is_bool_dtype(s):
            X[col] = s.astype(float)
        elif not pd.api.types.is_numeric_dtype(s):
            categorical.append(i)
    return X, categorical


class EstimatorSystem(ExternalSystemModel):
    """Fit one estimator from a registry factory (k of them if `bagging` >= 2) on the train
    split."""

    warmup_torch_device = True

    def __init__(self, *, factory: str, bagging: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.factory = factory
        self.bagging = bagging
        self.estimators_: list = []

    def _fit_system(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        problem_type: str,
        random_state,
        validation_metadata=None,
        **_,
    ):
        make = _FACTORIES[self.factory]
        X = X.reset_index(drop=True)
        groups = (
            split_groups(X, validation_metadata) if getattr(make, "fit_groups", False) else None
        )
        self.grouped_ = groups is not None
        X, categorical = prepare_frame(X)
        y = y.reset_index(drop=True)
        self.problem_type_ = problem_type
        self.categorical_ = categorical
        if self.bagging and self.bagging > 1:
            parts = self._bags(X, y, problem_type, random_state)
        else:
            parts = [np.arange(len(X))]
        self.pool_, self.handles_, self.pool_classes_ = None, [], []
        pool = getattr(make, "bag_pool", None)
        if pool is not None and len(parts) > 1:  # GPU workers fit the bags concurrently
            self.estimators_ = []
            self.handles_, self.pool_classes_ = pool.fit(
                problem_type,
                categorical,
                [
                    (X.iloc[ix], y.iloc[ix].to_numpy(), None if groups is None else groups[ix])
                    for ix in parts
                ],
            )
            self.pool_ = pool
            return self
        self.estimators_ = [
            make(problem_type, categorical).fit(
                X.iloc[ix],
                y.iloc[ix].to_numpy(),
                **({} if groups is None else {"groups": groups[ix]}),
            )
            for ix in parts
        ]
        return self

    def post_fit(self, X: pd.DataFrame, y: pd.Series, X_test: pd.DataFrame):
        # one test row for use after prediction (post_predict)
        self.test_row_ = X_test.iloc[:1]

    def post_predict(self):
        """Hook outside train/inference timing. If an estimator has `after_predict(X_row)`, call it
        with one test row (preprocessed as in `_predict`), e.g. to time single-row prediction."""
        row = getattr(self, "test_row_", None)
        hooks = [getattr(e, "after_predict", None) for e in self.estimators_]
        if row is None or not any(hooks):
            return
        Xp, _ = prepare_frame(row)
        Xp = Xp.reset_index(drop=True)
        for hook in hooks:
            if hook is not None:
                hook(Xp)

    def _bags(self, X, y, problem_type: str, random_state) -> list[np.ndarray]:
        seed = 0 if random_state is None else int(random_state)
        splitter = KFold(self.bagging, shuffle=True, random_state=seed)
        if problem_type != "regression" and y.value_counts().min() >= self.bagging:
            splitter = StratifiedKFold(self.bagging, shuffle=True, random_state=seed)
        return [train_ix for train_ix, _ in splitter.split(X, y)]

    def _predict(self, X: pd.DataFrame) -> pd.Series:
        Xp, _ = prepare_frame(X)
        Xp = Xp.reset_index(drop=True)
        if getattr(self, "pool_", None) is not None:
            preds = self._pool_predict(Xp, proba=False)
        else:
            preds = [e.predict(Xp) for e in self.estimators_]
        pred = np.mean(preds, axis=0)
        return pd.Series(pred, index=X.index)

    def _predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        Xp, _ = prepare_frame(X)
        Xp = Xp.reset_index(drop=True)
        # bags may see different classes: align columns to the union of all bags' classes, with
        # 0 for missing ones. Column order must be the natural label order (np.unique, same as
        # the estimator's LabelEncoder), because TabArena reads the second column for binary and
        # columns by position for multiclass
        if getattr(self, "pool_", None) is not None:
            bag_classes = [np.asarray(c) for c in self.pool_classes_]
            probas = self._pool_predict(Xp, proba=True)
        else:
            bag_classes = [np.asarray(e.classes_) for e in self.estimators_]
            probas = (e.predict_proba(Xp) for e in self.estimators_)
        classes = list(np.unique(np.concatenate(bag_classes)))
        total = pd.DataFrame(0.0, index=range(len(Xp)), columns=classes)
        for bag, p in zip(bag_classes, probas, strict=True):
            proba = pd.DataFrame(np.asarray(p), columns=list(bag))
            total = total.add(proba.reindex(columns=classes, fill_value=0.0))
        total = total / len(bag_classes)
        total.index = X.index
        return total

    def _pool_predict(self, X: pd.DataFrame, proba: bool) -> list[np.ndarray]:
        # if prediction fails, TabArena does not call cleanup (`cleanup_on_failure` defaults to
        # False): drop the bags left on the workers here (in-process estimators go away with the
        # system)
        try:
            return self.pool_.predict(self.handles_, X, proba=proba)
        except Exception:
            self.cleanup()
            raise

    def cleanup(self):
        self.estimators_ = []
        if getattr(self, "pool_", None) is not None:
            self.pool_.drop(self.handles_)
            self.pool_, self.handles_, self.pool_classes_ = None, [], []
