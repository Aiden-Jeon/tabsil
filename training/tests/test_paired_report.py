"""Tests for the v2 evaluation code: the paired report, the development suite, TabArena's error on
AMLB folds, the 3-split TabArena suite and the weights in the arena cache key."""

import argparse
import types

import numpy as np
import pandas as pd
import pytest
import torch
import train_helpers as helpers
from sklearn.metrics import log_loss, roc_auc_score

from tabsil_train.benchmarks import amlb, arena, paired, tabarena


def _splits(errors: dict, kind: dict, splits: int = 1, imputed=()) -> pd.DataFrame:
    rows = []
    for d, e in errors.items():
        for s in range(splits):
            value = e[s] if isinstance(e, (list, tuple)) else e
            rows.append(
                {
                    "dataset": d,
                    "fold": s,
                    "metric_error": value,
                    "problem_type": kind[d],
                    "imputed": (d, s) in imputed,
                }
            )
    return pd.DataFrame(rows)


KIND = {"a": "binary", "b": "binary", "c": "multiclass", "r": "regression"}


def test_g_is_the_geometric_mean_of_split_averaged_ratios():
    a = paired.splits_frame(_splits({"a": [0.1, 0.3], "b": 0.2, "c": 1.0, "r": 4.0}, KIND, 2), "a")
    b = paired.splits_frame(_splits({"a": [0.1, 0.1], "b": 0.2, "c": 0.5, "r": 8.0}, KIND, 2), "b")
    report = paired.compare({"s": (a, b)}, rounds=200)
    ratios = report["ratios"].set_index("dataset")["ratio"]
    # splits are averaged first: a's dataset "a" is 0.2 against 0.1
    assert ratios["a"] == pytest.approx(0.5) and ratios["b"] == pytest.approx(1.0)
    s = report["suites"]["s"]
    assert s["all"]["G"] == pytest.approx((0.5 * 1.0 * 0.5 * 2.0) ** 0.25)
    assert (s["all"]["better"], s["all"]["worse"], s["all"]["equal"]) == (2, 1, 1)
    assert s["all"]["splits"] == 8
    assert s["classification"]["datasets"] == 3 and s["regression"]["G"] == pytest.approx(2.0)
    assert s["all"]["ci_low"] <= s["all"]["G"] <= s["all"]["ci_high"]
    assert 0 < s["all"]["wilcoxon_p"] <= 1
    assert "pooled" not in report


def test_bootstrap_is_reproducible_and_pooling_names_suites():
    rng = np.random.default_rng(1)
    names = [f"d{i}" for i in range(20)]
    kind = dict.fromkeys(names, "regression")
    a = paired.splits_frame(
        _splits(dict(zip(names, rng.uniform(1, 2, 20), strict=True)), kind), "a"
    )
    b = a.copy()
    b["error"] = b["error"] * rng.uniform(0.9, 1.0, 20)
    one = paired.compare({"x": (a, b), "y": (a, a)}, rounds=500, seed=3)
    two = paired.compare({"x": (a, b), "y": (a, a)}, rounds=500, seed=3)
    assert one["suites"]["x"]["all"] == two["suites"]["x"]["all"]
    assert one["suites"]["x"]["all"]["G"] < 1 and one["suites"]["x"]["all"]["ci_high"] < 1
    # identical runs: G 1, every dataset equal, p 1
    same = one["suites"]["y"]["all"]
    assert same["G"] == 1.0 and same["equal"] == 20 and same["wilcoxon_p"] == 1.0
    assert one["pooled"]["all"]["datasets"] == 40
    assert set(one["ratios"]["suite"]) == {"x", "y"}
    text = paired.format_report(one, {"x": {"all": (1700.0, 1740.0)}})
    assert "### pooled" in text and "1700 → 1740" in text


def test_imputed_missing_and_mismatched_splits_are_errors():
    frame = _splits({"a": 0.1, "b": 0.2}, KIND, imputed={("b", 0)})
    with pytest.raises(ValueError, match="imputed splits \\(b split 0\\)"):
        paired.splits_frame(frame, "run")
    assert len(paired.splits_frame(frame, "run", allow_imputed=True)) == 2
    with pytest.raises(ValueError, match="no error"):
        paired.splits_frame(_splits({"a": np.nan}, KIND), "run")
    with pytest.raises(ValueError, match="metric_error"):
        paired.splits_frame(pd.DataFrame({"dataset": ["a"], "fold": [0], "score": [0.9]}), "run")
    a = paired.splits_frame(_splits({"a": 0.1, "b": 0.2}, KIND), "a")
    b = paired.splits_frame(_splits({"a": 0.1}, KIND), "b")
    with pytest.raises(ValueError, match="different splits"):
        paired.dataset_ratios(a, b)


def test_mean_of_runs_is_the_split_mean_of_their_errors():
    kind = {"a": "binary", "r": "regression"}
    one = paired.splits_frame(_splits({"a": [0.1, 0.3], "r": 4.0}, kind, 2), "one")
    two = paired.splits_frame(_splits({"a": [0.3, 0.5], "r": 2.0}, kind, 2), "two")
    mean = paired.mean_splits([one, two.iloc[::-1]])  # the row order does not matter
    assert list(mean.columns) == paired.COLUMNS
    errors = mean.set_index(["dataset", "split"])["error"]
    assert errors[("a", 0)] == pytest.approx(0.2) and errors[("a", 1)] == pytest.approx(0.4)
    assert errors[("r", 1)] == pytest.approx(3.0)
    arm = paired.splits_frame(_splits({"a": [0.15, 0.15], "r": 3.0}, kind, 2), "arm")
    ratios = paired.compare({"s": (mean, arm)}, rounds=50)["ratios"].set_index("dataset")["ratio"]
    assert ratios["a"] == pytest.approx(0.5) and ratios["r"] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="different splits"):
        paired.mean_splits([one, two[two["dataset"] == "a"]])
    multiclass = {**kind, "a": "multiclass"}
    other_kind = paired.splits_frame(_splits({"a": 0.1, "r": 1.0}, multiclass, 2), "other")
    with pytest.raises(ValueError, match="problem type"):
        paired.mean_splits([one, other_kind])


def test_report_script_averages_runs_joined_with_plus(tmp_path, capsys):
    script = helpers.load_script("scripts/paired_report.py", "paired_report_script")
    kind = {"a": "binary", "r": "regression"}
    paths = {}
    for run, (a, r) in {"c1": (0.1, 2.0), "c2": (0.3, 4.0), "arm": (0.1, 3.0)}.items():
        d = tmp_path / run / "tabarena-3split"
        d.mkdir(parents=True)
        _splits({"a": a, "r": r}, kind, 3).to_csv(d / "results_per_split.csv")
        (d / "summary.json").write_text('{"elo": 1500}')
        paths[run] = str(d / "results_per_split.csv")
    out = tmp_path / "out"
    script.main(["--a", f"{paths['c1']}+{paths['c2']}", "--b", paths["arm"], "--out-dir", str(out)])
    ratios = pd.read_csv(out / "ratios.csv").set_index("dataset")["ratio"]
    assert ratios["a"] == pytest.approx(0.5) and ratios["r"] == pytest.approx(1.0)
    # the mean of two runs has no Elo of its own
    assert "1500 →" not in capsys.readouterr().out


def test_zero_errors_are_floored():
    a = paired.splits_frame(_splits({"a": 0.0, "b": 0.0}, KIND), "a")
    b = paired.splits_frame(_splits({"a": 0.0, "b": 1e-5}, KIND), "b")
    ratios = paired.dataset_ratios(a, b).set_index("dataset")["ratio"]
    assert ratios["a"] == 1.0 and ratios["b"] == pytest.approx(10.0)


def test_report_script_reads_local_files_and_pools(tmp_path, capsys):
    script = helpers.load_script("scripts/paired_report.py", "paired_report_script")
    for run, scale in (("a", 1.0), ("b", 0.9)):
        d = tmp_path / run / "tabarena-3split"
        d.mkdir(parents=True)
        _splits({"a": 0.2 * scale, "r": 3.0}, KIND, 3).to_csv(d / "results_per_split.csv")
        (d / "summary.json").write_text('{"elo": 1500, "breakdown": {"regression": {"elo": 1400}}}')
        rows = _splits({"x": 0.5 * scale}, {"x": "multiclass"}, 10)
        rows.drop(columns=["imputed"]).to_csv(tmp_path / run / "eval_rows.csv")
    out = tmp_path / "out"
    script.main(
        [
            *("--a", str(tmp_path / "a/tabarena-3split/results_per_split.csv")),
            *("--b", str(tmp_path / "b/tabarena-3split/results_per_split.csv")),
            *("--a", str(tmp_path / "a/eval_rows.csv"), "--b", str(tmp_path / "b/eval_rows.csv")),
            *("--out-dir", str(out)),
        ]
    )
    text = capsys.readouterr().out
    assert "### tabarena-3split" in text and "### amlb" in text and "### pooled" in text
    assert "1500 → 1500" in text
    ratios = pd.read_csv(out / "ratios.csv")
    assert len(ratios) == 3 and (out / "report.json").exists()
    with pytest.raises(SystemExit):
        script.parse_args(["--a", "x", "--a", "y", "--b", "z"])


# development suite


def test_dev_suite_is_amlb_without_the_tabarena_datasets():
    assert len(amlb.DEV_SUITE["classification"]) == 23
    assert len(amlb.DEV_SUITE["regression"]) == 25
    every = set(amlb.task_names("classification")) | set(amlb.task_names("regression"))
    assert set(amlb.TABARENA_OVERLAP) <= every
    dev = set(amlb.DEV_SUITE["classification"]) | set(amlb.DEV_SUITE["regression"])
    assert dev.isdisjoint(amlb.TABARENA_OVERLAP) and len(dev | set(amlb.TABARENA_OVERLAP)) == 57
    names = {**amlb.task_names("classification"), **amlb.task_names("regression")}
    assert names[359964] == "dna" and amlb.TABARENA_OVERLAP[359964][0] == "splice"


def test_eval_script_selects_the_dev_suite():
    script = helpers.load_script("scripts/eval.py", "eval_script_dev")
    tasks = script.selected_tasks(argparse.Namespace(suite="dev", tasks=None))
    assert len(tasks) == 48 and ("classification", 359955) not in tasks
    assert ("regression", 361236) in tasks  # auction_verification stays
    with pytest.raises(SystemExit, match="359955"):
        script.selected_tasks(argparse.Namespace(suite="dev", tasks="359955"))
    assert script.parse_args(["--suite", "dev"]).suite == "dev"


def test_metric_error_is_tabarenas():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 50)
    p = rng.uniform(size=50)
    proba = np.c_[1 - p, p]
    classes = np.array([0, 1])
    assert amlb.metric_error("binary", y, proba, classes) == pytest.approx(1 - roc_auc_score(y, p))
    y3 = rng.integers(0, 3, 60)
    proba3 = rng.dirichlet(np.ones(3), 60)
    labels = np.array(["a", "b", "c"])
    assert amlb.metric_error("multiclass", labels[y3], proba3, labels) == pytest.approx(
        log_loss(y3, proba3, labels=[0, 1, 2])
    )
    # a test row of a class missing from training is dropped, as for ROC AUC
    with_unseen = np.r_[labels[y3], ["d"]]
    assert amlb.metric_error(
        "multiclass", with_unseen, np.r_[proba3, [[1 / 3] * 3]], labels
    ) == pytest.approx(log_loss(y3, proba3, labels=[0, 1, 2]))
    t = rng.normal(size=20)
    assert amlb.metric_error("regression", t, t + 2.0) == pytest.approx(2.0)
    assert amlb.problem_type("classification", 2) == "binary"
    assert amlb.problem_type("classification", 5) == "multiclass"
    assert amlb.problem_type("regression", 0) == "regression"


def test_evaluate_task_rows_carry_the_error(monkeypatch):
    X = pd.DataFrame({"x": np.arange(40, dtype=float)})
    y = pd.Series(np.arange(40) % 3)

    class Split:
        def get_train_test_split_indices(self, fold, repeat, sample):
            test = np.arange(fold, 40, 4)
            return np.setdiff1d(np.arange(40), test), test

    task = amlb.OpenMLTask(1, "t", X, y, [], "classification", Split())

    class Prior:
        def fit(self, X, y):
            self.classes_, counts = np.unique(y, return_counts=True)
            self.p_ = counts / counts.sum()
            return self

        def predict_proba(self, X):
            return np.tile(self.p_, (len(X), 1))

    rows = amlb.evaluate_task(task, lambda t: Prior(), n_folds=2)
    assert [r["problem_type"] for r in rows] == ["multiclass", "multiclass"]
    assert all(r["metric_error"] == pytest.approx(np.log(3), rel=0.05) for r in rows)


# TabArena with 3 splits


def test_three_split_suite_keeps_splits_0_to_2_of_every_dataset():
    s = tabarena.SUITE_3SPLIT
    assert s.name == "tabarena-3split" and s.subset == () and s.folds() == [0, 1, 2]
    assert "splits_all" in s.public_csv and s.breakdown["regression"][0] == ("regression",)
    assert tabarena.SUITE.folds() is None
    grid = pd.DataFrame(
        {
            "dataset": ["d"] * 6 + ["e"] * 3,
            "split": [0, 1, 2, 3, 4, 5, 0, 1, 2],
            "problem_type": ["binary"] * 9,
            "max_train_rows": [10] * 9,
            "n_features": [2] * 9,
            "n_classes": [2] * 9,
        }
    )
    context = types.SimpleNamespace(
        task_metadata_collection=types.SimpleNamespace(task_grid=lambda: grid),
        subset_predicates={},
    )
    kept = arena.task_grid(context, s, arena.ModelLimits())
    assert len(kept) == 6 and set(kept["split"]) == {0, 1, 2}


def test_eval_script_accepts_the_three_split_suite():
    script = helpers.load_script("scripts/eval.py", "eval_script_3split")
    assert script.arena_suite("tabarena-3split") is tabarena.SUITE_3SPLIT
    assert script.parse_args(["--suite", "tabarena-3split"]).suite == "tabarena-3split"


# the weights in the arena cache key


def test_weights_digest_follows_the_weights_not_the_path():
    script = helpers.load_script("scripts/eval.py", "eval_script_digest")
    model = helpers.tiny_model()
    same = helpers.tiny_model()
    same.load_state_dict(model.state_dict())
    borders = torch.linspace(-1, 1, 5)
    assert script.weights_digest(model, borders) == script.weights_digest(same, borders.clone())
    with torch.no_grad():
        next(same.parameters()).view(-1)[0] += 1e-3
    assert script.weights_digest(model, borders) != script.weights_digest(same, borders)
    assert script.weights_digest(model, borders) != script.weights_digest(model, borders * 2)


def test_arena_fingerprint_has_weights_and_applied_inference(monkeypatch, tmp_path):
    script = helpers.load_script("scripts/eval.py", "eval_script_fp")
    models = [helpers.tiny_model(), helpers.tiny_model()]
    with torch.no_grad():
        next(models[1].parameters()).view(-1)[0] += 1.0
    hub = "hf://aiden-jeon/tabsil"
    argv = ["--suite", "tabarena-lite", "--classifier-path", hub, "--out-dir", str(tmp_path)]
    paths = {"classification": hub, "regression": None}
    prints = []
    for model in models:
        monkeypatch.setattr(script, "load_model", lambda *a, m=model, **k: (m, None))
        args = script.parse_args(argv)
        _, factories = script.arena_estimator_factory(args, paths)
        prints.append(script.arena_fingerprint(args, paths, factories))
    # same path, other weights: another cache
    assert prints[0] != prints[1]
    applied = factories["classification"].inference
    assert applied["n_estimators"] == 8 and applied["categories"] == "counts"
    assert script.applied_inference_params(factories)["inference_classification"].startswith("{")
