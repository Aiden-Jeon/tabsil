"""Wrap an sklearn-style estimator as a `tabarena` system (`ExternalSystemModel`).

This module imports `tabarena`, so it is loaded only when `tabsil_train.benchmarks.arena` needs it.
System configs may be saved as yaml, so estimator factories live in this module's registry and
only their names are passed (`debug_mode=True` runs in the same process, so the registry stays
visible).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from tabarena.benchmark.exec_models import ExternalSystemModel

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


def experiments(suite, method: str, key: str) -> list:
    from tabarena.benchmark.experiment import (
        BeyondArenaExperimentBundle,
        TabArenaV0pt1ExperimentBundle,
    )
    from tabarena.utils.config_utils import SystemConfigGenerator

    generator = SystemConfigGenerator(
        model_cls=EstimatorSystem,
        name=method,
        manual_configs=[{"factory": key}],
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
    """Fit one estimator from a registry factory on the train split."""

    warmup_torch_device = True

    def __init__(self, *, factory: str, **kwargs):
        super().__init__(**kwargs)
        self.factory = factory
        self.estimators_: list = []

    def _fit_system(self, X: pd.DataFrame, y: pd.Series, *, problem_type: str, **_):
        make = _FACTORIES[self.factory]
        X, categorical = prepare_frame(X.reset_index(drop=True))
        self.problem_type_ = problem_type
        self.categorical_ = categorical
        estimator = make(problem_type, categorical).fit(X, y.reset_index(drop=True).to_numpy())
        self.estimators_ = [estimator]
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

    def _predict(self, X: pd.DataFrame) -> pd.Series:
        Xp, _ = prepare_frame(X)
        (estimator,) = self.estimators_
        pred = estimator.predict(Xp.reset_index(drop=True))
        return pd.Series(pred, index=X.index)

    def _predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        Xp, _ = prepare_frame(X)
        (estimator,) = self.estimators_
        classes = np.asarray(estimator.classes_)
        proba = np.asarray(estimator.predict_proba(Xp.reset_index(drop=True)), dtype=float)
        # columns in the natural label order (np.unique, same as the estimator's LabelEncoder),
        # because TabArena reads the second column for binary and columns by position for
        # multiclass. Reordered by position: a list of boolean labels (jm1,
        # online_shoppers_intention) used as column keys would be read as a row mask
        order = np.argsort(classes, kind="stable")
        return pd.DataFrame(proba[:, order], columns=list(classes[order]), index=X.index)

    def cleanup(self):
        self.estimators_ = []
