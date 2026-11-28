"""Train / inference time in the evaluation, defined as on the TabArena leaderboard (seconds per 1K
rows, averaged over each dataset's splits, median over datasets): the arena summary, comparison
table, console and MLflow output, and the AMLB fit / prediction records. No data is downloaded."""

import argparse
import json
import time
import types

import numpy as np
import pandas as pd
import pytest
from train_helpers import load_script

from tabsil_train.benchmarks.tabular import amlb, arena

SIZES = pd.DataFrame(
    {"num_instances_train": [1000.0, 2000.0, 500.0], "num_instances_test": [100.0, 400.0, 50.0]},
    index=pd.Index(["a", "b", "c"], name="dataset"),
)


def _split_results(method="new", imputed=False, **times):
    # one row per (dataset, fold); `times` overrides time_train_s / time_infer_s
    frame = pd.DataFrame(
        {
            "dataset": ["a", "a", "b", "c"],
            "fold": [0, 1, 0, 0],
            "method": method,
            "metric_error": 0.1,
            "time_train_s": [1.0, 3.0, 2.0, 10.0],
            "time_infer_s": [0.1, 0.3, 0.4, 0.5],
            "imputed": imputed,
        }
    )
    for col, values in times.items():
        frame[col] = values
    return frame


def test_per_1k_medians_normalize_by_dataset_rows_then_take_the_median_over_datasets():
    t = arena.per_1k_medians(_split_results(), SIZES)
    # train per 1K: a = mean(1, 3) / 1000 rows = 2, b = 2 / 2000 = 1, c = 10 / 500 = 20 -> median 2
    # infer per 1K: a = mean(1, 3), b = 1, c = 10 -> median 2
    assert t == {"time_train_s_per_1K_median": 2.0, "time_infer_s_per_1K_median": 2.0}
    # a dataset without a row count does not count
    t = arena.per_1k_medians(_split_results(), SIZES.drop(index="c"))
    assert t["time_train_s_per_1K_median"] == pytest.approx(1.5)
    assert arena.per_1k_medians(_split_results().iloc[:0], SIZES) == {}
    no_time = _split_results().drop(columns=list(arena.TIME_COLUMNS))
    assert arena.per_1k_medians(no_time, SIZES) == {}


def test_per_1k_medians_match_tabarena_aggregation():
    # same numbers as bencheval's aggregation (per (method, dataset) mean, median over datasets)
    bencheval = pytest.importorskip("bencheval.evaluator")
    results = _split_results()
    for col, size in arena.PER_1K_ROWS.items():
        results[f"{col}_per_1K"] = results[col] * 1000 / results["dataset"].map(SIZES[size])
    evaluator = bencheval.BenchmarkEvaluator(
        method_col="method",
        task_col="dataset",
        seed_column="fold",
        columns_to_agg_extra=["time_train_s_per_1K", "time_infer_s_per_1K"],
    )
    agg = evaluator.aggregate(
        results[["method", "dataset", "fold", "time_train_s_per_1K", "time_infer_s_per_1K"]]
    )
    ours = arena.per_1k_medians(_split_results(), SIZES)
    for col in arena.TIME_COLUMNS:
        expected = agg.loc["new", f"median_{col}_per_1K"]
        assert ours[f"{col}_per_1K_median"] == pytest.approx(expected)


def test_run_times_keep_raw_seconds_and_add_leaderboard_time():
    ours = _split_results(imputed=[False, False, False, True])
    board = pd.Series({"median_time_train_s_per_1K": 4.5, "median_time_infer_s_per_1K": 0.25})
    t = arena.run_times(ours, SIZES, board)
    # raw seconds of the splits that ran (unchanged keys)
    assert t["time_train_s_mean"] == 2.0 and t["time_train_s_total"] == 6.0
    assert t["time_infer_s_total"] == pytest.approx(0.8)
    # per 1K over the splits that ran (c is imputed): median(2, 1) = 1.5
    assert t["time_train_s_per_1K_median"] == pytest.approx(1.5)
    # the leaderboard's own value (all splits)
    assert t["time_train_s_per_1K_median_leaderboard"] == 4.5
    assert t["time_infer_s_per_1K_median_leaderboard"] == 0.25
    # without sizes or a leaderboard row: only the raw seconds, as before
    assert set(arena.run_times(ours)) == {
        "time_train_s_mean",
        "time_train_s_total",
        "time_infer_s_mean",
        "time_infer_s_total",
    }
    nan_board = pd.Series({"median_time_train_s_per_1K": np.nan})
    assert "time_train_s_per_1K_median_leaderboard" not in arena.run_times(ours, None, nan_board)


def test_baseline_times_cover_the_same_splits_as_ours():
    ours = _split_results(imputed=[False, True, False, True])  # ran a/0 and b/0
    xgb = _split_results(
        method="TA-XGB (default)", time_train_s=[4.0, 100.0, 8.0, 100.0], time_infer_s=0.2
    )
    per_split = pd.concat([ours, xgb, _split_results(method="Other (default)")])
    leaderboard = pd.DataFrame(
        {
            "method": ["new", "TA-XGB (default)", "Other (default)"],
            "median_time_train_s_per_1K": [1.0, 7.0, 3.0],
            "median_time_infer_s_per_1K": [0.1, 0.2, 0.3],
        }
    )
    out = arena.baseline_times(leaderboard, per_split, ours, SIZES)
    assert set(out) == {"XGBoost (default)"}  # other baselines have no results
    xgb_times = out["XGBoost (default)"]
    # on our run splits: a = 4 / 1000 rows -> 4, b = 8 / 2000 rows -> 4
    assert xgb_times["time_train_s_per_1K_median"] == pytest.approx(4.0)
    assert xgb_times["time_train_s_per_1K_median_leaderboard"] == 7.0
    assert xgb_times["time_infer_s_per_1K_median_leaderboard"] == 0.2
    assert xgb_times["run_splits"] == 2


def _context(frame: pd.DataFrame):
    collection = types.SimpleNamespace(to_dataframe=lambda: frame)
    return types.SimpleNamespace(task_metadata_collection=collection)


def test_dataset_sizes_average_all_splits_of_each_dataset():
    # task metadata: one row per split
    frame = pd.DataFrame(
        {
            "tabarena_task_name": ["a", "a", "b"],
            "num_instances_train": [900, 1100, 50],
            "num_instances_test": [100, 300, 10],
        }
    )
    sizes = arena.dataset_sizes(_context(frame))
    assert sizes.loc["a"].to_dict() == {"num_instances_train": 1000.0, "num_instances_test": 200.0}
    assert sizes.loc["b", "num_instances_train"] == 50.0
    empty = arena.dataset_sizes(_context(pd.DataFrame()))
    assert empty.empty and set(arena.PER_1K_ROWS.values()) <= set(empty.columns)


def test_comparison_puts_public_and_recomputed_times_side_by_side():
    train, infer = "Median Train Time (s/1K) [⬇️]", "Median Predict Time (s/1K) [⬇️]"
    website = pd.DataFrame(
        {
            "Model": ["[XGBoost (default)](u)", "new"],
            "Elo [⬆️]": [1180.0, 1300.0],
            train: [1.2, 0.8],
            infer: [0.04, 0.12],
        }
    )
    public = pd.DataFrame(
        {"Model": ["[XGBoost (default)](u)"], "Elo [⬆️]": [1181.0], train: [1.25], infer: [0.045]}
    )
    public["name"] = public["Model"].map(arena.display_name)
    out = arena.compare_with_public(website, public, "new").set_index("name")
    assert out.loc["new", "train_s_per_1K"] == 0.8 and out.loc["new", "infer_s_per_1K"] == 0.12
    assert np.isnan(out.loc["new", "train_s_per_1K_public"])
    assert out.loc["XGBoost (default)", "train_s_per_1K"] == 1.2
    assert out.loc["XGBoost (default)", "train_s_per_1K_public"] == 1.25
    assert out.loc["XGBoost (default)", "infer_s_per_1K_public"] == 0.045
    # without a public table the public time columns exist and are empty
    out = arena.compare_with_public(website, None, "new")
    assert out["train_s_per_1K_public"].isna().all()


def _summary():
    time = {
        "time_train_s_mean": 12.0,
        "time_infer_s_mean": 1.5,
        "time_train_s_per_1K_median": 0.8,
        "time_infer_s_per_1K_median": 0.11,
        "time_train_s_per_1K_median_leaderboard": 0.9,
        "time_infer_s_per_1K_median_leaderboard": 0.125,
    }
    scores = {
        "elo": 1300.0,
        "elo_ci_plus": 10.0,
        "elo_ci_minus": 9.0,
        "score": 0.5,
        "improvability": 5.0,
        "rank": 3.0,
        "imputed_pct": 10.0,
        "imputed_pct_leaderboard": 10.0,
        "position": 3,
        "n_methods": 40,
        "winrate": {},
        "time": time,
    }
    xgb = {
        "time_train_s_per_1K_median": 1.1,
        "time_infer_s_per_1K_median": 0.04,
        "time_train_s_per_1K_median_leaderboard": 1.2,
        "time_infer_s_per_1K_median_leaderboard": 0.045,
        "run_splits": 46,
    }
    return {
        "suite": "tabarena-lite",
        "run_id": "abc",
        **scores,
        "time_baselines": {"XGBoost (default)": xgb},
        "breakdown": {"binary": dict(scores)},
    }


def test_console_summary_prints_per_1k_times_next_to_baselines(capsys):
    script = load_script("scripts/eval.py", "eval_times_console")
    script.print_arena_summary(_summary())
    lines = capsys.readouterr().out.splitlines()
    expected = "time train 12.0s infer 1.5s s/1K train/infer run 0.8/0.11 leaderboard 0.9/0.125"
    assert expected in lines[0]
    assert "XGBoost (default) run 1.1/0.04 leaderboard 1.2/0.045" in lines[1]
    assert lines[2].lstrip().startswith("binary") and "leaderboard 0.9/0.125" in lines[2]
    # small times keep their digits (in-context fits are fast)
    assert script.per_1k_text({"time_train_s_per_1K_median": 0.0042}).startswith("0.0042/")
    assert script.per_1k_text({"time_train_s_per_1K_median": 1613.3}).startswith("1613/")


def test_mlflow_metrics_include_per_1k_times(monkeypatch, tmp_path):
    script = load_script("scripts/eval.py", "eval_times_mlflow")
    logged = {}
    fake = types.SimpleNamespace(
        log_params=lambda params: None,
        log_metrics=logged.update,
        log_artifact=lambda path, where=None: None,
    )
    monkeypatch.setattr(script, "mlflow", fake)
    args = argparse.Namespace(
        suite="tabarena-lite",
        method_name="tabsil",
        seed=0,
        max_features_per_member=None,
        subsample_train_rows=None,
        classifier_path="clf.pt",
        regressor_path=None,
        out_dir=str(tmp_path),
    )
    script.log_arena_mlflow(args, _summary(), tmp_path, {})
    assert logged["tabarena_lite_time_train_s_per_1K_median_leaderboard"] == 0.9
    assert logged["tabarena_lite_time_infer_s_per_1K_median"] == 0.11
    assert logged["tabarena_lite_time_train_s_mean"] == 12.0  # raw seconds stay
    assert logged["tabarena_lite_binary_time_infer_s_per_1K_median_leaderboard"] == 0.125


class _Folds:
    def get_train_test_split_indices(self, fold, repeat, sample):
        ix = np.arange(40)
        return ix[ix % 4 != fold], ix[ix % 4 == fold]  # 30 train, 10 test rows


class _Mean:
    def fit(self, X, y):
        self.mean = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.mean)


def _task():
    g = np.random.default_rng(0)
    X = pd.DataFrame({"a": g.normal(size=40), "b": g.normal(size=40)})
    return amlb.OpenMLTask(7, "toy", X, pd.Series(g.normal(size=40)), [], "regression", _Folds())


def test_amlb_rows_record_fit_and_prediction_time():
    rows = amlb.evaluate_task(_task(), lambda t: _Mean(), n_folds=2)
    assert len(rows) == 2
    for row in rows:
        assert set(amlb.TIME_KEYS) <= set(row)
        assert (row["train_rows"], row["test_rows"], row["fit_rows"]) == (30, 10, 30)
        assert row["fit_seconds"] >= 0 and row["predict_seconds"] >= 0


def test_amlb_fit_rows_follow_subsampling():
    from tabsil_train.benchmarks.tabular.subsample import SubsampledEstimator

    make = lambda t: SubsampledEstimator(_Mean(), 12, stratify=False)  # noqa: E731
    rows = amlb.evaluate_task(_task(), make, n_folds=1)
    assert (rows[0]["train_rows"], rows[0]["fit_rows"]) == (30, 12)


class _SlowDistribution(_Mean):
    def predict_distribution(self, X):
        time.sleep(0.2)
        edges = np.linspace(-3, 3, 11)
        return np.log(np.full((len(X), 10), 0.1)), edges


def test_amlb_prediction_time_leaves_out_a_separate_distribution_call():
    # without `predict_full`, `uncertainty` adds a `predict_distribution` call: not timed, so the
    # prediction time is the same with or without `uncertainty`
    rows = amlb.evaluate_task(_task(), lambda t: _SlowDistribution(), n_folds=1, uncertainty=True)
    assert "crps" in rows[0] and rows[0]["predict_seconds"] < 0.2


def test_amlb_time_per_1k_and_medians():
    row = {"train_rows": 2000, "test_rows": 500, "fit_seconds": 4.0, "predict_seconds": 1.0}
    assert amlb.per_1k(row) == {"fit_seconds_per_1K": 2.0, "predict_seconds_per_1K": 2.0}
    rows = [
        {"task_id": 1, **row},
        {"task_id": 1, **row, "fit_seconds": 8.0},  # task 1: mean(4, 8) / 2000 rows -> 3
        {"task_id": 2, **row, "fit_seconds": 20.0},  # 10
        {"task_id": 3, **row, "fit_seconds": 2.0},  # 1
    ]
    t = amlb.time_medians(rows)
    assert t == {"time_train_s_per_1K_median": 3.0, "time_infer_s_per_1K_median": 2.0}
    # as TabArena: divided by the dataset's mean fold size, not each fold's
    uneven = [
        {"task_id": 1, **row, "train_rows": 1000, "fit_seconds": 1.0},
        {"task_id": 1, **row, "train_rows": 3000, "fit_seconds": 3.0},
    ]
    assert amlb.time_medians(uneven)["time_train_s_per_1K_median"] == pytest.approx(1.0)
    assert amlb.time_medians([]) == {}


def test_amlb_fit_records_follow_the_arena_record_format(tmp_path):
    script = load_script("scripts/eval.py", "eval_times_amlb")
    args = script.parse_args(["--suite", "regression", "--out-dir", str(tmp_path)])
    task = _task()
    rows = amlb.evaluate_task(task, lambda t: _Mean(), n_folds=2)
    script.write_amlb_fits(args, "regression", task, rows)
    records = [json.loads(line) for line in script.fits_path(args).read_text().splitlines()]
    assert len(records) == 2
    first = records[0]
    # the arena fit record's keys (`RecordedEstimator`) plus the prediction time
    assert {k: first[k] for k in ("suite", "task", "dataset", "rows", "features")} == {
        "suite": "regression",
        "task": "regression",
        "dataset": "toy",
        "rows": 30,
        "features": 2,
    }
    assert (first["task_id"], first["fold"], first["test_rows"]) == (7, 0, 10)
    assert first["train_rows"] == 30
    assert first["fit_seconds"] == rows[0]["fit_seconds"]
    assert first["predict_seconds"] == rows[0]["predict_seconds"]
    assert first["predict_seconds_per_1K"] == pytest.approx(rows[0]["predict_seconds"] * 100)
    # the uncertainty table does not average the time columns as metrics
    per_dataset = script.summarize_uncertainty([{**r, "suite": "regression"} for r in rows])
    assert not set(amlb.TIME_KEYS) & set(per_dataset.columns)
