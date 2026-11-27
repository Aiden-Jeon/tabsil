"""The eval script's `--subsample-train-rows` (`tabsil_train.benchmarks.tabular.subsample`; the
subsampling itself: test_subsample.py)."""

from __future__ import annotations

import json
import types

import numpy as np
import pytest
import torch
from train_helpers import load_script, tiny_model

from tabsil.inference.ensemble import EnsembleClassifier, EnsembleRegressor
from tabsil_train.benchmarks.tabular.subsample import SubsampledEstimator


def _eval_script():
    return load_script("scripts/eval.py", "eval_subsample")


def _tiny_model():
    # one multitask model: classification and regression (999 quantiles)
    return tiny_model()


def test_eval_flag_off_is_the_old_estimator(monkeypatch, tmp_path):
    script = _eval_script()
    model = _tiny_model()
    monkeypatch.setattr(script, "load_model", lambda *a, **k: (model, None))
    paths = {"classification": "unused"}
    task = types.SimpleNamespace(categorical_indices=[])
    g = np.random.default_rng(1)
    X, y = g.normal(size=(120, 3)), g.choice(3, 120)

    off = script.parse_args(["--classifier-path", "unused", "--out-dir", str(tmp_path)])
    assert off.subsample_train_rows is None
    factories = {"classification": script.make_estimator_factory("classification", off)}
    plain = factories["classification"](task)
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
