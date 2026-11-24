"""Subsampling training rows at inference (`tabsil.inference.subsample`,
`--subsample-train-rows`)."""

from __future__ import annotations

import json
import types

import numpy as np
import pandas as pd
import pytest
import torch
from helpers import load_script, tiny_model

from tabsil.inference.ensemble import EnsembleClassifier, EnsembleRegressor
from tabsil.inference.subsample import SubsampledEstimator, subsample_indices


def _eval_script():
    return load_script("scripts/eval.py", "eval_subsample")


def _tiny_model():
    # one multitask model: classification and regression (999 quantiles)
    return tiny_model()


class _Recorder:
    """Estimator that records the arguments passed to fit."""

    classes_ = np.array([0, 1])

    def fit(self, X, y, **kwargs):
        self.X, self.y, self.kwargs = X, y, kwargs
        return self

    def predict(self, X):
        return np.zeros(len(X))


def test_regression_subsample_size_and_order():
    ix = subsample_indices(np.arange(1000.0), 100, stratify=False, seed=0)
    assert len(ix) == 100 and len(np.unique(ix)) == 100
    assert np.all(np.diff(ix) > 0) and ix.min() >= 0 and ix.max() < 1000
    # At most n rows: keep all, in order
    small = np.arange(50.0)
    assert np.array_equal(subsample_indices(small, 50, stratify=False, seed=0), np.arange(50))
    assert np.array_equal(subsample_indices(small, 80, stratify=True, seed=0), np.arange(50))
    with pytest.raises(ValueError):
        subsample_indices(np.arange(5.0), 0, stratify=False, seed=0)


def test_stratified_subsample_keeps_class_ratios_and_rare_classes():
    # Classes a 900, b 99, c 1 row -> 100 rows: about a 90, b 9; c at least 1 row
    y = np.array(["a"] * 900 + ["b"] * 99 + ["c"])
    ix = subsample_indices(y, 100, stratify=True, seed=0)
    assert len(ix) == 100 and len(np.unique(ix)) == 100
    picked = pd.Series(y[ix]).value_counts()
    assert picked["c"] == 1 and abs(picked["a"] - 90) <= 1 and abs(picked["b"] - 9.9) <= 1
    # Exactly n whether quotas sum below n (floor) or above it (min 1 row)
    y = np.repeat(np.arange(7), [400, 300, 200, 50, 30, 15, 5])
    for n in (7, 10, 13, 333):
        ix = subsample_indices(y, n, stratify=True, seed=1)
        assert len(ix) == n and set(y[ix]) == set(range(7))
    # More classes than n: 1 row per class
    ix = subsample_indices(np.repeat(np.arange(5), 3), 3, stratify=True, seed=0)
    assert sorted(np.repeat(np.arange(5), 3)[ix]) == [0, 1, 2, 3, 4]


def test_subsample_is_reproducible_by_seed():
    y = np.random.default_rng(0).choice(3, 5000)
    for stratify in (True, False):
        a = subsample_indices(y, 500, stratify=stratify, seed=3)
        assert np.array_equal(a, subsample_indices(y, 500, stratify=stratify, seed=3))
        assert not np.array_equal(a, subsample_indices(y, 500, stratify=stratify, seed=4))


def test_wrapper_subsamples_rows_and_row_aligned_fit_args():
    X = pd.DataFrame({"a": np.arange(300.0), "b": np.arange(300.0) * 2})
    y = np.repeat([0, 1], 150)
    groups = np.arange(300) // 10
    inner = _Recorder()
    est = SubsampledEstimator(inner, 40, stratify=True, seed=0).fit(X, y, groups=groups, other=3)
    assert est.train_rows_ == (300, 40)
    rows = inner.X["a"].to_numpy().astype(int)
    assert len(rows) == 40 and np.array_equal(inner.y, y[rows])
    assert np.array_equal(inner.kwargs["groups"], groups[rows]) and inner.kwargs["other"] == 3
    # Non-row arguments are not cut even if their length equals the training row count
    weights = np.arange(300.0)
    inner = _Recorder()
    SubsampledEstimator(inner, 40, stratify=True, seed=0).fit(X, y, per_feature=weights)
    assert inner.kwargs["per_feature"] is weights
    assert (inner.y == 0).sum() == 20  # stratified
    # Other attributes come from the wrapped estimator
    assert np.array_equal(est.classes_, [0, 1]) and len(est.predict(X)) == 300
    # At most max_rows rows: passed through as is (same object)
    inner = _Recorder()
    SubsampledEstimator(inner, 300, stratify=True).fit(X, y, groups=groups)
    assert inner.X is X and inner.y is y and inner.kwargs["groups"] is groups


def test_ensemble_on_subsample_equals_fit_on_the_chosen_rows():
    g = np.random.default_rng(0)
    X = g.normal(size=(200, 4))
    y = (X[:, 0] + 0.3 * g.normal(size=200) > 0).astype(int)
    model = _tiny_model()
    kw = dict(n_estimators=2, device="cpu", random_state=0)
    wrapped = SubsampledEstimator(EnsembleClassifier(model, **kw), 60, stratify=True, seed=5)
    wrapped.fit(X, y)
    ix = subsample_indices(y, 60, stratify=True, seed=5)
    direct = EnsembleClassifier(model, **kw).fit(X[ix], y[ix])
    assert np.array_equal(wrapped.predict_proba(X[:50]), direct.predict_proba(X[:50]))


def test_eval_flag_off_is_the_old_estimator(monkeypatch, tmp_path):
    script = _eval_script()
    model = _tiny_model()
    monkeypatch.setattr(script, "load_model", lambda *a, **k: (model, None))
    paths = {"classification": "unused"}
    factories = {"classification": types.SimpleNamespace(model=model, n_estimators=8)}
    task = types.SimpleNamespace(categorical_indices=[])
    g = np.random.default_rng(1)
    X, y = g.normal(size=(120, 3)), g.choice(3, 120)

    off = script.parse_args(["--classifier-path", "unused", "--out-dir", str(tmp_path)])
    assert off.subsample_train_rows is None
    plain = script.make_estimator_factory("classification", off)(task)
    assert type(plain) is EnsembleClassifier
    printed = json.loads(script.arena_fingerprint(off, paths, factories))
    assert "subsample_train_rows" not in printed  # default keeps the old fingerprint

    on = script.parse_args(
        ["--classifier-path", "unused", "--out-dir", str(tmp_path), "--subsample-train-rows", "50"]
    )
    wrapped = script.make_estimator_factory("classification", on)(task)
    assert isinstance(wrapped, SubsampledEstimator) and wrapped.stratify
    printed = json.loads(script.arena_fingerprint(on, paths, factories))
    assert printed["subsample_train_rows"] == 50
    # At most N training rows: bit-identical to no subsampling
    big = script.parse_args(
        ["--classifier-path", "unused", "--out-dir", str(tmp_path), "--subsample-train-rows", "500"]
    )
    same = script.make_estimator_factory("classification", big)(task).fit(X, y)
    plain.fit(X, y)
    assert np.array_equal(same.predict_proba(X[:30]), plain.predict_proba(X[:30]))
    with pytest.raises(SystemExit):
        script.parse_args(["--subsample-train-rows", "0"])


def test_eval_regression_subsample_is_not_stratified(monkeypatch, tmp_path):
    script = _eval_script()
    model = _tiny_model()
    borders = torch.linspace(-3, 3, 17)
    monkeypatch.setattr(script, "load_model", lambda *a, **k: (model, borders))
    args = script.parse_args(
        ["--regressor-path", "unused", "--out-dir", str(tmp_path), "--subsample-train-rows", "40"]
    )
    est = script.make_estimator_factory("regression", args)(
        types.SimpleNamespace(categorical_indices=[])
    )
    assert isinstance(est, SubsampledEstimator) and not est.stratify
    assert isinstance(est.estimator, EnsembleRegressor)
    X = np.random.default_rng(2).normal(size=(90, 3))
    est.fit(X, X[:, 0])
    assert est.train_rows_ == (90, 40) and est.predict(X[:5]).shape == (5,)
