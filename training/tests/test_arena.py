"""Rule tests for the TabArena/BeyondArena harness (`tabsil_train.benchmarks.arena`). No data or
baseline results are downloaded. System tests wrapping the estimator run only with the `arena` extra
(tabarena)."""

import argparse
import json
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import train_helpers as helpers

from tabsil_train.benchmarks import arena, beyondarena, tabarena


def _row(**kw):
    base = {"problem_type": "binary", "max_train_rows": 1000, "n_features": 10, "n_classes": 2}
    return pd.Series({**base, **kw})


def test_limits_mark_splits_too_big_for_the_model():
    lim = arena.ModelLimits(max_train_rows=10_000, max_features=500, max_classes=10)
    assert lim.skip_reason(_row()) is None
    assert lim.skip_reason(_row(max_train_rows=10_001)) == "train rows 10001 > 10000"
    assert lim.skip_reason(_row(n_features=501)) == "features 501 > 500"
    assert lim.skip_reason(_row(problem_type="multiclass", n_classes=11)) == "classes 11 > 10"
    # Regression has n_classes -1, so the class limit does not apply
    assert lim.skip_reason(_row(problem_type="regression", n_classes=-1)) is None
    assert arena.ModelLimits().skip_reason(_row(max_train_rows=10**7)) is None


def test_limits_per_problem_type_skip_types_without_a_model():
    limits = {"regression": arena.ModelLimits(max_train_rows=1000)}
    assert arena.skip_reason(limits, _row()) == "problem_type binary"
    assert arena.skip_reason(limits, _row(problem_type="regression", n_classes=-1)) is None
    reason = arena.skip_reason(limits, _row(problem_type="regression", max_train_rows=1001))
    assert reason == "train rows 1001 > 1000"


def test_display_name_strips_link_and_imputed_tag():
    assert arena.display_name("[TabPFNv2 (default)](https://x) [35.29% IMPUTED]") == (
        "TabPFNv2 (default)"
    )
    assert arena.display_name("TabFM+") == "TabFM+"


def _per_split():
    rows = []
    for d in ("a", "b", "c"):
        rows.append({"dataset": d, "fold": 0, "method": "RF (default)", "metric_error": 0.5})
        rows.append({"dataset": d, "fold": 0, "method": "XGB (default)", "metric_error": 0.4})
    rows.append({"dataset": "a", "fold": 0, "method": "new", "metric_error": 0.3})
    rows.append({"dataset": "b", "fold": 0, "method": "new", "metric_error": 0.4})
    return pd.DataFrame(rows)


def test_missing_splits_are_filled_with_the_fillna_method():
    ours = arena.with_imputed(_per_split(), "new", "RF (default)")
    assert list(ours["dataset"]) == ["a", "b", "c"]
    assert list(ours["imputed"]) == [False, False, True]
    assert ours.loc[2, "metric_error"] == 0.5  # c is the RF result
    assert (ours["method"] == "new").all()


def test_winrate_counts_ties_as_half_and_reports_run_only_splits():
    per_split = _per_split()
    ours = arena.with_imputed(per_split, "new", "RF (default)")
    w = arena.winrates(per_split, ours)["XGBoost (default)"]
    # a wins, b ties, c (imputed RF 0.5 vs 0.4) loses
    assert w["all"] == pytest.approx((1 + 0.5 + 0) / 3)
    assert w["run"] == pytest.approx(0.75)
    assert (w["splits"], w["run_splits"]) == (3, 2)
    assert "TabPFNv2 (default)" not in arena.winrates(per_split, ours)  # method with no results
    # A "TA-" prefix and different case (as in BeyondArena results) still match the same method
    renamed = per_split.replace({"method": {"XGB (default)": "TA-xgb (default)"}})
    assert arena.winrates(renamed, ours)["XGBoost (default)"] == w


def test_skip_reasons_are_counted_by_kind():
    reasons = pd.Series([None, "train rows 11640 > 10000", "train rows 20000 > 10000", "failed"])
    assert arena._reason_counts(reasons) == {"train rows > 10000": 2, "run": 1, "failed": 1}


def test_comparison_puts_public_and_recomputed_scores_side_by_side():
    website = pd.DataFrame(
        {
            "Model": ["[XGBoost (default)](u)", "new"],
            "Elo [⬆️]": [1180.0, 1300.0],
            "Score [⬆️]": [0.05, 0.2],
        }
    )
    public = pd.DataFrame({"Model": ["[XGBoost (default)](u)", "[Other (default)](u)"]})
    public["Elo [⬆️]"] = [1181.0, 1500.0]
    public["Score [⬆️]"] = [0.05, 0.6]
    public["name"] = public["Model"].map(arena.display_name)
    out = arena.compare_with_public(website, public, "new")
    assert list(out["name"]) == ["new", "XGBoost (default)", "Other (default)"]
    assert out.loc[1, "elo_public"] == 1181.0
    assert bool(out.loc[0, "is_new"]) and np.isnan(out.loc[0, "elo_public"])
    assert np.isnan(out.loc[2, "elo"])  # method not in this run's pool


def test_comparison_without_public_table_or_new_method():
    website = pd.DataFrame({"Model": ["[XGBoost (default)](u)"], "Elo [⬆️]": [1180.0]})
    out = arena.compare_with_public(website, None, None)
    assert list(out["name"]) == ["XGBoost (default)"]
    assert np.isnan(out.loc[0, "elo_public"]) and not out["is_new"].any()


def test_public_table_failure_does_not_raise(monkeypatch):
    def fail(path):
        raise OSError("HTTP 503")

    monkeypatch.setattr(arena, "load_public_leaderboard", fail)
    assert arena.try_public_leaderboard("data/x.csv") is None


def test_run_id_changes_with_the_estimator_fingerprint():
    a = arena._run_id("clf=step_100k", "tabsil")
    assert a == arena._run_id("clf=step_100k", "tabsil")
    assert a != arena._run_id("clf=step_200k", "tabsil")
    assert a != arena._run_id("clf=step_100k", "tabsil-v2")
    # pinned: a different value means existing arena caches (out_dir/runs/<id>) are no longer found
    assert a == "f88b11cf80e2"


def test_default_arena_fingerprint_and_run_id_are_pinned(monkeypatch, tmp_path):
    # the default settings' fingerprint and run id: if either changes, existing arena caches
    # (out_dir/runs/<id>) are no longer found and their splits run again
    script = _eval_script()
    model = helpers.tiny_model()
    monkeypatch.setattr(script, "load_model", lambda *a, **k: (model, None))
    # the weights' digest follows the tiny model's random init, so it is fixed here (its own
    # test: test_weights_digest_follows_the_weights_not_the_path)
    digest = "0" * 40
    monkeypatch.setattr(script, "weights_digest", lambda model, borders=None: digest)
    hub = "hf://aiden-jeon/tabsil"
    argv = ["--suite", "tabarena-lite", "--classifier-path", hub, "--regressor-path", hub]
    args = script.parse_args([*argv, "--out-dir", str(tmp_path)])
    paths = {"classification": hub, "regression": hub}
    _, factories = script.arena_estimator_factory(args, paths)
    fingerprint = script.arena_fingerprint(args, paths, factories)
    value = json.loads(fingerprint)
    kinds = ["standard", "power", "robust", "quantile"]
    common = {
        "categories": "counts",
        "feature_subsampling": "balanced",
        "max_features_per_member": 768,
        "member_kinds": kinds,
        "n_estimators": 8,
    }
    assert value == {
        "inference": {
            "classification": {**common, "softmax_temperature": 1.0},
            "regression": common,
        },
        "max_features_per_member": 768,
        "n_estimators": {"classification": 8, "regression": 8},
        "paths": {"classification": hub, "regression": hub},
        "weights": {"classification": digest, "regression": digest},
    }
    # the id follows the fingerprint's text (key order, separators), not only its content
    assert arena._run_id(fingerprint, "tabsil") == "70635c9c5fac"


def test_summary_and_cache_folder_record_method_and_fingerprint(monkeypatch, tmp_path):
    # summary.json names the cache folder's run id; the folder's fingerprint.txt holds the method
    # and the fingerprint only
    pytest.importorskip("tabarena")
    grid = pd.DataFrame({"dataset": ["d"], "split": [0], "fold": [0], "repeat": [0]})
    grid["skip_reason"] = None
    ours = pd.DataFrame({"dataset": ["d"], "fold": [0], "imputed": [True]})
    context = types.SimpleNamespace(
        fillna_method="RF",
        compare=None,
        leaderboard_to_website_format=lambda leaderboard: leaderboard,
    )
    monkeypatch.setattr(arena, "make_context", lambda suite: context)
    monkeypatch.setattr(arena, "task_grid", lambda context, suite, limits: grid.copy())
    monkeypatch.setattr(arena, "model_pool", lambda context, method: None)
    monkeypatch.setattr(arena, "_quiet", lambda *a, **k: (pd.DataFrame(), pd.DataFrame()))
    monkeypatch.setattr(arena, "_new_method_name", lambda leaderboard, method: None)
    monkeypatch.setattr(arena, "with_imputed", lambda per_split, name, fillna: ours)
    monkeypatch.setattr(arena, "try_public_leaderboard", lambda path: None)
    monkeypatch.setattr(arena, "compare_with_public", lambda *a: pd.DataFrame())
    monkeypatch.setattr(arena, "winrates", lambda per_split, ours: {})
    sizes = pd.DataFrame(columns=list(arena.PER_1K_ROWS.values()), dtype=float)
    monkeypatch.setattr(arena, "dataset_sizes", lambda context: sizes)
    monkeypatch.setattr(arena, "baseline_times", lambda *a: {})
    summary = arena.run(
        tabarena.SUITE,
        None,
        limits=arena.ModelLimits(),
        out_dir=tmp_path,
        fingerprint="fp",
        results=[],
    )
    assert "bagging" not in summary and summary["run_id"] == arena._run_id("fp", "tabsil")
    # the split ran (no skip reason) but has no result: a failure, named in the summary
    assert summary["failed_splits"] == 1 and summary["failed"] == ["d split 0"]
    text = (tmp_path / "runs" / summary["run_id"] / "fingerprint.txt").read_text()
    assert text == "tabsil\nfp\n"


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


def test_method_without_results_has_no_name():
    board = pd.DataFrame({"method": ["RF (default)", "tabsil_c1"]})
    assert arena._new_method_name(board, "tabsil") == "tabsil_c1"
    assert arena._new_method_name(board.iloc[:1], "tabsil") is None


def test_suites_point_at_the_official_splits_and_public_tables():
    assert tabarena.SUITE.subset == ("lite",)
    assert "splits_lite" in tabarena.SUITE.public_csv
    assert "imputation_yes" in tabarena.SUITE.public_csv
    assert tabarena.SUITE.breakdown["regression"][0] == ("lite", "regression")
    core = beyondarena.suite()
    assert core.subset == ("core2k",) and core.name == "beyondarena"
    assert core.breakdown["split_temporal"][0] == ("core2k", "temporal")
    assert "split_temporal/size_all" in core.breakdown["split_temporal"][1]
    lite = beyondarena.suite("lite")
    assert lite.subset == ("core2k", "lite")
    assert lite.breakdown["size_tiny"][0] == ("core2k", "lite", "tiny")


def _system():
    pytest.importorskip("tabarena")
    from tabsil_train.benchmarks import _arena_system

    return _arena_system


class _Constant:
    """Classifier predicting the train-row class frequencies (each bag sees different classes)."""

    def fit(self, X, y):
        self.classes_, counts = np.unique(y, return_counts=True)
        self.p_ = counts / counts.sum()
        self.n_fit_ = len(X)
        return self

    def predict_proba(self, X):
        return np.tile(self.p_, (len(X), 1))


def test_prepare_frame_turns_dates_into_numbers_and_strings_into_categoricals():
    system = _system()
    X = pd.DataFrame(
        {
            "num": [1.0, 2.0],
            "when": pd.to_datetime(["2020-01-01", None]),
            "text": ["a b", "c"],
            "flag": [True, False],
        }
    )
    out, categorical = system.prepare_frame(X)
    assert categorical == [2]
    assert out["when"].iloc[0] == pytest.approx(1577836800.0) and np.isnan(out["when"].iloc[1])
    # datetime64[us]/[s] read from parquet give the same seconds
    for unit in ("us", "s"):
        coarse, _ = system.prepare_frame(X.assign(when=X["when"].astype(f"datetime64[{unit}]")))
        assert coarse["when"].iloc[0] == pytest.approx(1577836800.0)
    assert out["flag"].tolist() == [1.0, 0.0]
    assert X["when"].dtype.kind == "M"  # original frame unchanged


def test_probability_columns_follow_the_natural_label_order():
    # Integer labels 2 and 10: in string order ("10" < "2") TabArena reads column 2 as class 10
    system = _system()
    key = system.register_factory(lambda problem_type, categorical: _Constant())
    model = object.__new__(system.EstimatorSystem)
    model.factory = key
    X = pd.DataFrame({"x": np.arange(10.0)})
    y = pd.Series([2] * 8 + [10] * 2)
    model._fit_system(X, y, problem_type="binary", random_state=0)
    proba = model._predict_proba(X)
    assert list(proba.columns) == [2, 10]
    np.testing.assert_allclose(proba.iloc[:, 1], 0.2)
    labels = pd.Series(list(range(12)) * 2)  # multiclass 0..11
    model._fit_system(
        pd.DataFrame({"x": np.arange(24.0)}), labels, problem_type="multiclass", random_state=0
    )
    assert list(model._predict_proba(X).columns) == list(range(12))
    # boolean labels (jm1, online_shoppers_intention): False before True, picked by position
    flags = pd.Series([True] * 8 + [False] * 2)
    model._fit_system(X, flags, problem_type="binary", random_state=0)
    proba = model._predict_proba(X)
    assert list(proba.columns) == [False, True] and list(proba.index) == list(X.index)
    np.testing.assert_allclose(proba[True], 0.8)


def _eval_script():
    return helpers.load_script("scripts/eval.py", "eval_script_arena")


def test_eval_stops_when_a_split_failed():
    # A split that raised is imputed by TabArena; the run must fail instead of reporting scores
    script = _eval_script()
    script.check_failed_splits({"failed": []})
    script.check_failed_splits({})
    message = "2 splits failed and were imputed: jm1 split 0, x split 1"
    with pytest.raises(SystemExit, match=message):
        script.check_failed_splits({"failed": ["jm1 split 0", "x split 1"]})


def test_eval_script_limits_follow_the_supported_size():
    from tabsil.inference.ensemble import max_classes_of
    from tabsil.recipe import SUPPORTED_SIZE

    script = _eval_script()
    clf = reg = helpers.tiny_model()  # one multitask model serves both tasks
    rows, features = SUPPORTED_SIZE["max_train_rows"], SUPPORTED_SIZE["max_features"]
    args = argparse.Namespace(max_train_rows=None, max_features=None)
    limits = script.arena_limits({"classification": (clf, None), "regression": (reg, "b")}, args)
    assert limits["binary"] == arena.ModelLimits(rows, features, max_classes_of(clf))
    assert limits["multiclass"] is limits["binary"]
    assert limits["regression"] == arena.ModelLimits(rows, features)
    only = script.arena_limits(
        {"regression": (reg, "b")}, argparse.Namespace(max_train_rows=2048, max_features=None)
    )
    # without a classification checkpoint the classification problems are left out
    assert only == {"regression": arena.ModelLimits(2048, features)}


def test_eval_script_accepts_arena_suites():
    script = _eval_script()
    args = script.parse_args(["--suite", "beyondarena-lite"])
    assert args.suite == "beyondarena-lite"
    # bagging and the test-time methods were removed
    for gone in (["--bagging", "8"], ["--bag-gpus", "2"], ["--test-time", "greedy"]):
        with pytest.raises(SystemExit):
            script.parse_args(["--suite", "beyondarena-lite", *gone])
    assert script.arena_suite("beyondarena").subset == ("core2k",)
    assert script.arena_suite("tabarena-lite").name == "tabarena-lite"


def test_post_predict_calls_after_predict_with_one_test_row():
    system = _system()
    seen = []

    class _Hooked(_Constant):
        def after_predict(self, row):
            seen.append(row)

    key = system.register_factory(lambda problem_type, categorical: _Hooked())
    model = object.__new__(system.EstimatorSystem)
    model.factory = key
    X = pd.DataFrame({"x": np.arange(10.0), "when": pd.date_range("2020-01-01", periods=10)})
    y = pd.Series([0, 1] * 5)
    model._fit_system(X, y, problem_type="binary", random_state=0)
    model.post_fit(X, y, X.iloc[3:])
    model._predict_proba(X)
    assert seen == []  # not called during prediction (the timed part)
    model.post_predict()
    assert len(seen) == 1 and len(seen[0]) == 1
    assert seen[0]["x"].iloc[0] == 3.0 and seen[0]["when"].dtype.kind == "f"  # same preprocessing


def test_dataset_queue_puts_expensive_datasets_first_and_skips_unrun_splits():
    # Dataset cost = total cells of the splits to run. Most expensive first (same grid, same order)
    rows = []
    for name, n_rows, n_features, splits in [
        ("big", 100_000, 50, 1),
        ("mid_a", 10_000, 100, 3),
        ("mid_b", 20_000, 50, 3),
        ("small", 1_000, 10, 30),
        ("tiny", 100, 5, 10),
    ]:
        rows += [
            {
                "dataset": name,
                "max_train_rows": n_rows,
                "n_features": n_features,
                "skip_reason": None,
            }
        ] * splits
    rows.append(
        {"dataset": "skipped", "max_train_rows": 10**7, "n_features": 10, "skip_reason": "x"}
    )
    grid = pd.DataFrame(rows)
    queue = arena.dataset_queue(grid)
    assert queue == arena.dataset_queue(grid.sample(frac=1, random_state=0))
    # mid_a and mid_b tie at 3 million cells: by name
    assert queue == [("big", 1), ("mid_a", 3), ("mid_b", 3), ("small", 30), ("tiny", 10)]
    # --subsample-train-rows: count train rows capped at that value (big is 10k × 50 = 500k cells)
    queue = arena.dataset_queue(grid, subsample_rows=10_000)
    assert queue == [("mid_a", 3), ("mid_b", 3), ("big", 1), ("small", 30), ("tiny", 10)]


def test_shard_worker_command_replaces_dataset_and_output_flags(tmp_path):
    script = _eval_script()
    argv = [
        "--suite", "beyondarena", "--classifier-path", "c.pt", "--shards", "4",
        "--datasets=a,b", "--out-dir", "out", "--mlflow", "--method-name", "m",
        "--heartbeat-minutes", "1",
    ]  # fmt: skip
    worker_dir = tmp_path / "shards" / "shard1"
    cmd = script.shard_worker_command(argv, worker_dir)
    assert cmd == [
        "--suite", "beyondarena", "--classifier-path", "c.pt", "--method-name", "m",
        "--out-dir", str(worker_dir), "--shard-worker",
    ]  # fmt: skip
    args = script.parse_args(cmd)
    assert args.shard_worker and args.shards == 1 and args.datasets is None
    names = [("big/one", 2), ("big_one", 1), ("Big_one", 1)]
    jobs = script.shard_jobs(names, tmp_path, "beyondarena", "abc")
    # out-dir is keyed by dataset, not worker (on rerun any worker reuses the cache)
    dataset_dir = Path(jobs[0]["out_dir"])
    assert dataset_dir.parent.parent == tmp_path / "shards" / "datasets"
    assert dataset_dir.name == "beyondarena" and dataset_dir.parent.name.startswith("big_one-")
    assert jobs[0] == {
        "dataset": "big/one",
        "splits": 2,
        "out_dir": str(dataset_dir),
        "results": str(dataset_dir / "results.pkl"),
        "started": str(dataset_dir / "started"),
        "cache": str(dataset_dir / "runs" / "abc"),
    }
    # Names that collide after sanitizing, or differ only in case, still get different folders
    assert len({Path(j["out_dir"]).parent.name.lower() for j in jobs}) == 3
    with pytest.raises(SystemExit):
        script.main(["--suite", "classification", "--shards", "2"])


# Fake worker: takes a job on stdin, leaves a received marker, sleeps `sleep` seconds, then writes
# a result file.
# "bad" dies after the marker, "early" dies before it, "last_words" dies after writing the result
FAKE_WORKER = """
import json, pickle, sys, time
from pathlib import Path
for line in iter(sys.stdin.readline, ""):
    job = json.loads(line)
    if job["dataset"] == "early" and sys.argv[1] == "0":
        sys.exit(4)
    Path(job["out_dir"]).mkdir(parents=True, exist_ok=True)
    Path(job["started"]).touch()
    if job["dataset"] == "bad":
        sys.exit(3)
    start = time.time()
    time.sleep(job["sleep"])
    row = {"dataset": job["dataset"], "worker": sys.argv[1], "start": start, "end": time.time()}
    with open(job["results"], "wb") as f:
        pickle.dump([row], f)
    if job["dataset"] == "last_words":
        sys.exit(5)
"""


def _fake_jobs(tmp_path, spec, host_bytes=None):
    jobs = []
    for name, splits, sleep in spec:
        out = tmp_path / name
        jobs.append(
            {
                "dataset": name,
                "splits": splits,
                "out_dir": str(out),
                "results": str(out / "results.pkl"),
                "started": str(out / "started"),
                "cache": str(out / "runs" / "id"),
                "sleep": sleep,
            }
        )
        if (host_bytes or {}).get(name):
            jobs[-1]["host_bytes"] = host_bytes[name]
    return jobs


def _fake_rows(jobs):
    import pickle

    rows = {}
    for job in jobs:
        if Path(job["results"]).exists():
            with open(job["results"], "rb") as f:
                (row,) = pickle.load(f)
            rows[row["dataset"]] = row
    return rows


def _fake_spawn(i):
    import subprocess
    import sys

    return subprocess.Popen(
        [sys.executable, "-c", FAKE_WORKER, str(i)], stdin=subprocess.PIPE, text=True
    )


def test_run_queue_gives_the_next_dataset_to_the_worker_that_is_free(tmp_path, capsys):
    import pickle

    script = _eval_script()
    # While worker 0 holds the expensive one (1 s), worker 1 runs all the rest (half each if static)
    spec = [("big", 4, 1.0)] + [(f"small{k}", 1, 0.01) for k in range(5)]
    jobs = _fake_jobs(tmp_path, spec)
    (tmp_path / "small0").mkdir()
    (tmp_path / "small0" / "results.pkl").write_bytes(b"stale")  # stale results get removed
    missing, failed = script.run_queue(jobs, _fake_spawn, 2, heartbeat_s=0.2, poll_s=0.01)
    assert missing == [] and failed == {}
    workers = {}
    for job in jobs:
        with open(job["results"], "rb") as f:
            (row,) = pickle.load(f)
        workers[row["dataset"]] = row["worker"]
    assert workers["big"] == "0"
    assert all(workers[f"small{k}"] == "1" for k in range(5))
    out = capsys.readouterr().out
    assert "[heartbeat" in out and "splits 9/9 (datasets 6/6, queue 0)" in out
    assert "worker 0: big splits 0/4" in out  # per-worker dataset and elapsed time while working


def test_run_queue_reports_a_dead_worker_and_its_dataset(tmp_path):
    script = _eval_script()
    jobs = _fake_jobs(tmp_path, [("bad", 2, 0.0), ("a", 1, 0.2), ("b", 1, 0.01), ("c", 1, 0.01)])
    missing, failed = script.run_queue(jobs, _fake_spawn, 2, heartbeat_s=60, poll_s=0.01)
    # A dead worker's dataset is reported as missing (not silently imputed). The rest finish
    assert missing == ["bad"]
    assert failed == {0: "exit 3 while running bad"}
    assert all(Path(j["results"]).exists() for j in jobs[1:])


@pytest.mark.parametrize("first", ["early", "last_words"])
def test_run_queue_gives_unstarted_work_of_a_dead_worker_to_another(tmp_path, first):
    import pickle

    script = _eval_script()
    # worker 0 dies before taking a job (early) or after writing its result (last_words). Worker 1
    # takes the remaining jobs; jobs that never started are not dropped as missing
    jobs = _fake_jobs(tmp_path, [(first, 1, 0.0), ("a", 1, 0.3), ("b", 1, 0.0), ("c", 1, 0.0)])
    missing, failed = script.run_queue(jobs, _fake_spawn, 2, heartbeat_s=60, poll_s=0.01)
    assert missing == []
    assert list(failed) == [0] and "while running" not in failed[0]
    for job in jobs:
        with open(job["results"], "rb") as f:
            assert pickle.load(f)[0]["dataset"] == job["dataset"]


# --- CPU cell-state budget (`--host-memory-budget-gb`, large tables on interleaved models) -------


def test_run_queue_holds_big_tables_within_the_host_budget(tmp_path, capsys):
    script = _eval_script()
    # big_a and big_b need 60 each (budget 100): big_b starts after big_a ends. Idle workers
    # meanwhile take small tables (even those queued behind big_b). mid (30) fits the budget, but
    # since big_b waits ahead of it, mid starts after big_b does (most expensive first)
    spec = [("big_a", 1, 0.6), ("big_b", 1, 0.3), ("mid", 1, 0.3)]
    spec += [(f"small{k}", 1, 0.01) for k in range(4)]
    host = {"big_a": 60, "big_b": 60, "mid": 30}
    jobs = _fake_jobs(tmp_path, spec, host)
    missing, failed = script.run_queue(
        jobs, _fake_spawn, 3, heartbeat_s=0.2, poll_s=0.01, host_budget=100
    )
    assert missing == [] and failed == {}
    rows = _fake_rows(jobs)
    assert rows["big_b"]["start"] >= rows["big_a"]["end"]
    # mid could run with big_a (60 + 30 ≤ 100) but starts after big_b (= after big_a ends)
    assert rows["mid"]["start"] >= rows["big_a"]["end"]
    assert rows["mid"]["start"] < rows["big_b"]["end"]  # runs alongside big_b (60 + 30 ≤ 100)
    assert all(rows[f"small{k}"]["end"] < rows["big_a"]["end"] for k in range(4))
    out = capsys.readouterr().out
    assert "big_b waits for host memory (cell state 0 GB, in use 0 / budget 0 GB)" in out
    assert "host cell state" in out  # heartbeat


def test_run_queue_without_budget_keeps_the_queue_order(tmp_path):
    script = _eval_script()
    # Without a budget (None, the default for non-interleaved models or unknown limits) jobs run
    # concurrently as before
    jobs = _fake_jobs(tmp_path, [("big_a", 1, 0.4), ("big_b", 1, 0.4)], {"big_a": 60, "big_b": 60})
    missing, failed = script.run_queue(jobs, _fake_spawn, 2, heartbeat_s=60, poll_s=0.01)
    assert missing == [] and failed == {}
    rows = _fake_rows(jobs)
    assert rows["big_b"]["start"] < rows["big_a"]["end"]


def test_run_queue_runs_a_table_bigger_than_the_budget_alone(tmp_path):
    script = _eval_script()
    # A table above the budget still runs when no other large table is running (no deadlock)
    spec = [("huge", 1, 0.3), ("big", 1, 0.2), ("small", 1, 0.01)]
    jobs = _fake_jobs(tmp_path, spec, {"huge": 500, "big": 60})
    missing, failed = script.run_queue(
        jobs, _fake_spawn, 2, heartbeat_s=60, poll_s=0.01, host_budget=100
    )
    assert missing == [] and failed == {}
    rows = _fake_rows(jobs)
    assert rows["big"]["start"] >= rows["huge"]["end"]
    assert rows["small"]["end"] < rows["huge"]["end"]


def test_run_queue_frees_the_budget_of_a_dead_worker(tmp_path):
    script = _eval_script()
    # When the worker holding "bad" (a large table) dies, its share goes to the next large table
    spec = [("bad", 1, 0.0), ("big", 1, 0.01), ("small", 1, 0.01)]
    jobs = _fake_jobs(tmp_path, spec, {"bad": 80, "big": 80})
    missing, failed = script.run_queue(
        jobs, _fake_spawn, 2, heartbeat_s=60, poll_s=0.01, host_budget=100
    )
    assert missing == ["bad"] and failed == {0: "exit 3 while running bad"}
    assert set(_fake_rows(jobs)) == {"big", "small"}


def test_host_memory_budget(tmp_path):
    script = _eval_script()
    v2, v1 = tmp_path / "memory.max", tmp_path / "memory.limit_in_bytes"
    meminfo = tmp_path / "meminfo"
    files = (str(v2), str(v1))

    def budget(args):
        return script.host_memory_budget(args, files, str(meminfo))

    args = argparse.Namespace(host_memory_budget_gb=None)
    assert budget(args) is None  # neither cgroup nor meminfo: off
    meminfo.write_text("MemTotal:       2097060284 kB\nMemFree:  1 kB\n")
    assert budget(args) == 2097060284 * 1024 // 2  # cgroup unreadable: 50% of RAM
    v1.write_text("9223372036854771712\n")  # v1 unlimited: physical memory
    assert budget(args) == 2097060284 * 1024 // 2
    v1.write_text("1000000000000\n")
    assert budget(args) == 500_000_000_000
    v2.write_text("max\n")  # v2 first: unlimited → physical memory
    assert budget(args) == 2097060284 * 1024 // 2
    v2.write_text("2000000000000\n")
    assert budget(args) == 1_000_000_000_000
    args.host_memory_budget_gb = 700
    assert budget(args) == 700_000_000_000
    args.host_memory_budget_gb = 0
    assert budget(args) is None
    assert script.parse_args(["--host-memory-budget-gb", "800"]).host_memory_budget_gb == 800
    assert script.parse_args([]).host_memory_budget_gb is None


def test_host_state_estimate_only_for_factories_with_a_model():
    import types

    from tabsil.models.column_row_transformer import (
        ColumnRowTransformer,
        ColumnRowTransformerConfig,
    )

    script = _eval_script()
    interleaved = ColumnRowTransformer(
        ColumnRowTransformerConfig.tiny(column_blocks=2, row_blocks=2)
    )
    e = interleaved.config.emsize

    def factory(model):
        return types.SimpleNamespace(model=model)

    # no factory with a model: the queue does not budget host memory (None)
    plain = {"classification": factory(None), "regression": factory(None)}
    assert script.host_state_estimator(plain, {"classification": 768}) is None
    factories = {"classification": factory(interleaved), "regression": factory(None)}
    caps = {"classification": 768, "regression": 768}
    estimate = script.host_state_estimator(factories, caps)
    interleaved.interleaved_state_device_bytes = 1 << 30
    # member columns = min(features, 768) + 1 id
    assert estimate("binary", 1_250_000, 988) == 1_250_000 * 769 * e * 4
    assert estimate("multiclass", 1_250_000, 225) == 1_250_000 * 226 * e * 4
    assert estimate("regression", 1_250_000, 988) == 0  # that factory has no model
    assert estimate("binary", 1000, 10) == 0  # stays on the device
    # One-hot expansion (only with categorical columns) can grow columns up to the one-hot cap
    factories["classification"].max_one_hot_width = 160
    estimate = script.host_state_estimator(factories, caps)
    for categorical, columns in ((True, 160), (False, 43)):
        assert estimate("binary", 1_250_000, 43, categorical) == 1_250_000 * (columns + 1) * e * 4
    assert estimate("binary", 1_250_000, 988) == 1_250_000 * 769 * e * 4

    grid = pd.DataFrame(
        [
            {"dataset": "big", "problem_type": "binary", "max_train_rows": 10**6,
             "n_features": 100, "skip_reason": None},
            {"dataset": "big", "problem_type": "binary", "max_train_rows": 10**6,
             "n_features": 100, "skip_reason": None},
            {"dataset": "guess", "problem_type": "binary", "max_train_rows": 10**6,
             "n_features": 100, "skip_reason": None},
            {"dataset": "skipped", "problem_type": "binary", "max_train_rows": 10**6,
             "n_features": 100, "skip_reason": "x"},
            {"dataset": "tiny", "problem_type": "binary", "max_train_rows": 100,
             "n_features": 5, "skip_reason": None},
        ]
    )  # fmt: skip
    got = script.host_state_bytes(grid, {"big": 1_100_000}, estimate)
    # Unknown rows default to train rows × 1.5; skipped and zero-cost datasets are left out. The
    # offload keeps only the row chunks with train rows, so both count 10^6 train rows (rounded up
    # to whole chunks), less than the whole state
    assert got == {
        "big": estimate("binary", 1_100_000, 100, True, 10**6),
        "guess": estimate("binary", 1_500_000, 100, True, 10**6),
    }
    assert got["big"] == got["guess"] < estimate("binary", 1_100_000, 100)
    jobs = script.shard_jobs([("big", 1), ("tiny", 1)], Path("/tmp/x"), "beyondarena", "id", got)
    assert jobs[0]["host_bytes"] == got["big"] and "host_bytes" not in jobs[1]


def test_run_queue_does_not_let_smaller_big_tables_pass_a_waiting_one(tmp_path):
    script = _eval_script()
    # huge (90, budget 100) waits while a (30) runs. b and c (30 each) behind it fit the budget but
    # wait until huge starts (so huge is not left at the tail). small runs meanwhile
    spec = [("a", 1, 0.3), ("huge", 1, 0.3), ("b", 1, 0.05), ("c", 1, 0.05), ("small", 1, 0.01)]
    jobs = _fake_jobs(tmp_path, spec, {"a": 30, "huge": 90, "b": 30, "c": 30})
    missing, failed = script.run_queue(
        jobs, _fake_spawn, 3, heartbeat_s=60, poll_s=0.01, host_budget=100
    )
    assert missing == [] and failed == {}
    rows = _fake_rows(jobs)
    assert rows["huge"]["start"] >= rows["a"]["end"]
    assert rows["b"]["start"] >= rows["huge"]["end"] and rows["c"]["start"] >= rows["huge"]["end"]
    assert rows["small"]["end"] < rows["a"]["end"]


@pytest.mark.parametrize("one_hot", [False, True])
def test_host_state_columns_bound_the_member_input_width(one_hot):
    # The estimated member column count (eval.py) is not below what the real preprocessing
    # (`MemberPreprocessor`) produces. Fails here if preprocessing starts adding columns
    import types

    from tabsil.inference.preprocessing import Member, MemberPreprocessor

    script = _eval_script()
    rng = np.random.default_rng(0)
    n, n_num, cats = 200, 6, [3, 5, 2, 12]  # 6 numeric + 4 categorical (3, 5, 2, 12 levels)
    X = np.column_stack(
        [rng.normal(size=(n, n_num))] + [rng.integers(0, k, size=n) for k in cats]
    ).astype(float)
    cat_ix = list(range(n_num, n_num + len(cats)))
    width = 16
    member = Member(class_permutation=None, target_power=False, seed=0, one_hot=one_hot)
    prep = MemberPreprocessor(member, cat_ix, width).fit(X)
    actual = prep.transform(X, np.random.default_rng(1).standard_normal((len(X), 1))).shape[1]

    model = types.SimpleNamespace(
        inference_host_state_bytes=lambda rows, columns: columns,  # returns the column count as is
    )
    factory = types.SimpleNamespace(model=model, max_one_hot_width=width)
    estimate = script.host_state_estimator({"classification": factory}, {"classification": None})
    assert estimate("binary", 1, X.shape[1], True) >= actual
    if not one_hot:
        assert estimate("binary", 1, X.shape[1], False) == actual  # original columns + identifier


def test_split_rows_takes_the_largest_train_plus_test():
    import types

    frame = pd.DataFrame(
        {
            "tabarena_task_name": ["a", "a", "b"],
            "num_instances_train": [100, 80, 1_000_000],
            "num_instances_test": [10, 50, 250_000],
        }
    )
    collection = types.SimpleNamespace(to_dataframe=lambda: frame)
    context = types.SimpleNamespace(task_metadata_collection=collection)
    assert arena.split_rows(context) == {"a": 130, "b": 1_250_000}
    assert arena.split_rows(context, subsample_rows=90) == {"a": 130, "b": 250_090}
    empty = types.SimpleNamespace(to_dataframe=pd.DataFrame)
    assert arena.split_rows(types.SimpleNamespace(task_metadata_collection=empty)) == {}


def test_register_factory_gives_the_same_key_to_the_same_function():
    pytest.importorskip("tabarena")
    from tabsil_train.benchmarks import _arena_system as system

    def make(problem_type, categorical):
        return None

    key = system.register_factory(make)
    assert system.register_factory(make) == key
    assert system.register_factory(lambda p, c: None) != key
