"""Test-time compute in the evaluation: the arena split groups and run times, and the eval script's
`--test-time` estimators (the methods themselves: tests/test_test_time_compute.py)."""

import json
import types

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.datasets import load_breast_cancer
from train_helpers import load_script, tiny_model

from tabsil.inference.diag_scale import DiagScale
from tabsil.inference.ensemble import EnsembleClassifier
from tabsil.inference.ensemble_selection import (
    GreedyEnsembleClassifier,
)
from tabsil_train.benchmarks import arena

BORDERS = torch.linspace(-3, 3, 17)


def _tiny_model():
    return tiny_model()


def _classification(n=120):
    x, y = load_breast_cancer(return_X_y=True)
    return x[:n, :8], y[:n]


def test_split_groups_from_task_split_metadata():
    X = pd.DataFrame(
        {
            "patient": ["a", "b", "a", "c", None],
            "site": [1, 1, 2, 2, 2],
            "when": pd.to_datetime(["2020-01-01", "2020-01-01", "2021-06-01", None, "2022-01-01"]),
        }
    )
    meta = types.SimpleNamespace
    g = arena.split_groups(X, meta(group_on="patient", time_on=None))
    assert g[0] == g[2] and len(set(g)) == 4  # missing values form one group
    g = arena.split_groups(X, meta(group_on=["patient", "site"], time_on=None))
    assert len(set(g)) == 5
    t = arena.split_groups(X, meta(group_on=None, time_on="when"), n_intervals=2)
    assert t[0] == t[1] and t[3] == -1  # equal times share a bin, missing times are -1
    assert t[0] <= t[2] <= t[4] and len(set(t[[0, 2, 4]])) == 2  # contiguous bins in time order
    assert arena.split_groups(X, meta(group_on=None, time_on=None)) is None
    assert arena.split_groups(X, None) is None
    assert arena.split_groups(X, meta(group_on="missing", time_on=None)) is None


def test_run_times_average_the_splits_that_ran():
    ours = pd.DataFrame(
        {
            "imputed": [False, False, True],
            "time_train_s": [1.0, 3.0, 100.0],
            "time_infer_s": [0.5] * 3,
        }
    )
    t = arena.run_times(ours)
    assert t["time_train_s_mean"] == 2.0 and t["time_train_s_total"] == 4.0
    assert t["time_infer_s_mean"] == 0.5
    assert arena.run_times(ours.drop(columns=list(arena.TIME_COLUMNS))) == {}


def _eval_script():
    return load_script("scripts/eval.py", "eval_test_time")


def test_eval_script_builds_test_time_estimators_and_records_fits(monkeypatch, tmp_path):
    script = _eval_script()
    monkeypatch.setattr(script, "load_model", lambda path, task, cache: (_tiny_model(), None))
    x, y = _classification(60)
    for test_time, kind in [
        ("none", EnsembleClassifier),
        ("greedy", GreedyEnsembleClassifier),
        ("diagscale", DiagScale),
    ]:
        args = script.parse_args(
            ["--classifier-path", "clf.pt", "--test-time", test_time, "--pool-size", "4"]
            + ["--n-estimators", "2", "--out-dir", str(tmp_path / test_time), "--device", "cpu"]
        )
        make = script.make_estimator_factory("classification", args)
        estimator = make(types.SimpleNamespace(categorical_indices=[]))
        assert isinstance(estimator, kind)
        if test_time == "none":
            assert not (tmp_path / test_time / "test_time.jsonl").exists()
            assert script.test_time_params(args) == {"test_time": "none"}
            continue
        if test_time == "diagscale":
            estimator.kwargs["epochs"] = 1
        estimator.fit(x, y)
        record = json.loads((tmp_path / test_time / "test_time.jsonl").read_text())
        assert record["rows"] == 60 and record["features"] == 8
        assert ("selection" in record) == (test_time == "greedy")
        assert ("adaptation" in record) == (test_time == "diagscale")
    # regression takes --test-time none only (the eval script stops before any estimator is built)
    monkeypatch.setattr(script, "load_model", lambda path, task, cache: (_tiny_model(), BORDERS))
    args = script.parse_args(
        ["--regressor-path", "reg.pt", "--test-time", "greedy", "--out-dir", str(tmp_path)]
    )
    with pytest.raises(SystemExit, match="test-time none"):
        script.make_estimator_factory("regression", args)
    params = script.test_time_params(script.parse_args(["--test-time", "diagscale+greedy"]))
    assert params == {
        "test_time": "diagscale+greedy",
        "selection": "greedy",
        "pool_size": 24,
        "min_improvement": 0.01,  # default
        "oof_split": "task",
    }
    # the paper setup reproduces the original settings (and the same arena result cache)
    paper = ["--test-time", "greedy", "--min-improvement", "0", "--oof-split", "random"]
    params = script.test_time_params(script.parse_args(paper))
    assert params == {"test_time": "greedy", "selection": "greedy", "pool_size": 24}
    args = script.parse_args(["--test-time", "none"])
    assert args.min_improvement == 0.01 and args.oof_split == "task"  # unused without greedy
    guarded = ["--test-time", "greedy", "--min-improvement", "0.01", "--min-temperature", "0.5"]
    args = script.parse_args(
        [*guarded, "--oof-split", "task", "--classifier-path", "clf.pt", "--n-estimators", "2"]
    )
    assert script.test_time_params(args) == {
        "test_time": "greedy",
        "selection": "greedy",
        "pool_size": 24,
        "min_improvement": 0.01,
        "min_temperature": 0.5,
        "oof_split": "task",
    }
    args.out_dir, args.device = str(tmp_path / "guarded"), "cpu"
    make = script.make_estimator_factory("classification", args)
    assert make.fit_groups
    estimator = make(types.SimpleNamespace(categorical_indices=[]))
    assert estimator.min_improvement == 0.01 and min(estimator.temperatures) == 0.5
    estimator.pool_size = 4
    estimator.fit(x, y, groups=np.arange(len(y)) % 5)  # record_fits forwards groups too
    record = json.loads((tmp_path / "guarded" / "test_time.jsonl").read_text())
    assert record["selection"]["grouped"] and record["oof_split"] == "task"
    args.min_temperature = 10.0
    with pytest.raises(SystemExit):
        script.make_estimator_factory("classification", args)
    with pytest.raises(SystemExit):
        script.main(["--test-time", "none", "--min-improvement", "0.01"])


def test_arena_factory_passes_groups_only_for_task_out_of_fold(monkeypatch, tmp_path):
    # the arena system gets run_arena's classification/regression dispatch (fit_groups lives there)
    script = _eval_script()
    monkeypatch.setattr(script, "load_model", lambda path, task, cache: (_tiny_model(), None))
    monkeypatch.setattr(script, "arena_suite", lambda name: None)
    monkeypatch.setattr(script, "arena_limits", lambda models, args, caps=None: {})
    monkeypatch.setattr(script, "print_arena_summary", lambda summary: None)
    seen = {}

    def fake_run(suite, make, **kwargs):
        seen["make"] = make
        return {}

    monkeypatch.setattr(arena, "run", fake_run)
    for oof_split, expected in [("task", True), ("random", False)]:
        args = script.parse_args(
            ["--suite", "beyondarena-lite", "--classifier-path", "clf.pt", "--test-time", "greedy"]
            + ["--oof-split", oof_split, "--out-dir", str(tmp_path), "--device", "cpu"]
        )
        script.run_arena(args, tmp_path, use_mlflow=False)
        assert seen["make"].fit_groups is expected


def test_split_groups_reads_string_and_ordered_category_times():
    meta = types.SimpleNamespace(group_on=None, time_on="when")
    strings = pd.DataFrame({"when": ["2020-01-01", "2021-01-01", "2022-01-01", "2023-01-01"]})
    assert list(arena.split_groups(strings, meta, n_intervals=2)) == [0, 0, 1, 1]
    quarters = pd.Categorical(["2020Q2", "2020Q1", "2020Q4", "2020Q3"], ordered=True)
    quarters = quarters.reorder_categories(["2020Q1", "2020Q2", "2020Q3", "2020Q4"])
    t = arena.split_groups(pd.DataFrame({"when": quarters}), meta, n_intervals=2)
    assert list(t) == [0, 0, 1, 1]  # in category order
    # random folds when two bins are impossible (constant or unparsable values)
    assert arena.split_groups(pd.DataFrame({"when": [5, 5, 5]}), meta) is None
    assert arena.split_groups(pd.DataFrame({"when": ["a", "b", "c"]}), meta) is None
