"""ADBench suite: split protocol, run bookkeeping (resume, failures, rows reused only with the
same weights and settings) and the Elo against the copied fm-agent-bench results. No download:
tables are synthetic or monkeypatched."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from tabsil_train.benchmarks import adbench as ab


def _table(n: int, d: int = 4, rate: float = 0.1, seed: int = 0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < rate).astype(int)
    X = rng.standard_normal((n, d)) + 3 * y[:, None]
    return X, y


def test_split_protocol():
    X, y = _table(1500)
    X_tr, X_te, y_tr, y_te = ab.adbench_split(X, y, 1)
    assert len(X_te) == 450 and len(X_tr) == 1050
    assert np.isclose(X_tr.min(0), 0).all() and np.isclose(X_tr.max(0), 1).all()
    assert abs(y_tr.mean() - y_te.mean()) < 0.01  # stratified
    again = ab.adbench_split(X, y, 1)
    np.testing.assert_array_equal(again[0], X_tr)
    assert not np.array_equal(ab.adbench_split(X, y, 2)[0], X_tr)
    big = ab.adbench_split(*_table(12_000), 1)
    assert len(big[0]) + len(big[1]) == ab.MAX_ROWS
    with pytest.raises(ValueError, match="1000"):
        ab.adbench_split(*_table(999), 1)


def test_run_resumes_and_reports_failures(tmp_path, monkeypatch):
    monkeypatch.setattr(ab, "load", lambda dataset, root=None: _table(1500))
    calls = []

    def ok(X_train, X_test, seed):
        calls.append(len(X_train))
        return np.abs(X_test).sum(1)

    def broken(X_train, X_test, seed):
        return np.full(len(X_test), np.nan)

    scorers = {"iforest": ok, "tabsil_unsup": broken}
    methods = ["iforest", "iforest+clean", "tabsil_unsup"]
    rows, failures = ab.run(scorers, ["yeast"], ["s1"], methods, tmp_path, log=lambda s: None)
    assert sorted(rows.method) == ["iforest", "iforest+clean"]
    assert rows.auroc.gt(0.9).all()
    assert calls[1] < calls[0]  # +clean: only the rows labeled normal
    assert [f["method"] for f in failures] == ["tabsil_unsup"]
    assert (tmp_path / "failures.json").exists()
    rows, failures = ab.run(scorers, ["yeast"], ["s1"], methods, tmp_path, log=lambda s: None)
    assert len(calls) == 2 and len(rows) == 2 and len(failures) == 1  # finished rows skipped


def test_check_methods():
    assert ab.check_methods(["tabsil_unsup+clean", "iforest"])
    with pytest.raises(ValueError, match="unknown"):
        ab.check_methods(["ecod"])


def test_reference_elo_reproduces_fm_agent_bench():
    ref = ab.reference_results()
    assert set(ref.dataset) == set(ab.DATASETS)
    assert not ref.duplicated(["dataset", "split", "method"]).any()
    ref = ref[~ref.method.str.endswith(ab.CLEAN)]
    long = ref.assign(metric_error=1 - ref.auroc)[["method", "dataset", "split", "metric_error"]]
    ours = ab.elo(ab.fill(long, list(dict.fromkeys(long.method)))).set_index("method")
    old = pd.read_csv(ab.REFERENCE_DIR / "anomaly_elo.csv").query("pool == 'fm'")
    old = old.set_index("method")
    for col in ("elo", "elo+", "elo-", "imputed"):
        np.testing.assert_allclose(ours.loc[old.index, col], old[col], atol=0.06)


def _eval_script():
    import train_helpers as helpers

    return helpers.load_script("scripts/eval.py", "eval_script_adbench")


def _args(out, model, *more):
    return ["--suite", "adbench", "--regressor-path", str(model), "--datasets", "yeast",
            "--adbench-splits", "s1", "--n-estimators", "1", "--n-permutations", "1",
            "--device", "cpu", "--out-dir", str(out), *more]  # fmt: skip


@pytest.fixture
def suite(tmp_path, monkeypatch):
    """`eval.py` on a synthetic table: (script, output folder, results.csv, IsolationForest
    calls)."""
    monkeypatch.setattr(ab, "load", lambda dataset, root=None: _table(1500))
    monkeypatch.delenv("MLFLOW_RUN_ID", raising=False)
    calls, iforest = [], ab.iforest_scores
    monkeypatch.setattr(ab, "iforest_scores", lambda *a: calls.append(1) or iforest(*a))
    out = tmp_path / "eval"
    return _eval_script(), out, out / "adbench" / "results.csv", calls


def test_eval_suite_adbench_reuses_rows_of_the_same_weights_and_settings(suite, tmp_path):
    from train_helpers import tiny_checkpoint

    from tabsil_train.hub import load_model, weights_digest

    script, out, results, iforest_calls = suite
    model = tiny_checkpoint(tmp_path / "a.pt")
    script.main(_args(out, model))
    rows = pd.read_csv(results)
    assert sorted(rows.method) == ["iforest", "tabsil_unsup", "tabsil_unsup+clean"]
    ours = rows[rows.method != "iforest"]
    digest = weights_digest(*load_model(model, "regression", tmp_path))
    settings = f"weights={digest} n_estimators=1 seed=0 n_permutations=1 "
    assert ours.opts.str.startswith(settings).all()
    assert ours.opts.nunique() == 1  # the +clean row has its method's settings
    first = results.read_text()
    summary = out / "adbench" / "summary.json"
    assert "mean_auroc" in json.loads(summary.read_text())
    assert (out / "adbench" / "summary.csv").exists()

    # the same weights at another path: the rerun resumes, nothing is computed again
    script.main(_args(out, tiny_checkpoint(tmp_path / "copy.pt")))
    assert results.read_text() == first and len(iforest_calls) == 1

    # other weights stop the run, also when they replace the file at the first path
    for path in (tmp_path / "b.pt", tmp_path / "a.pt"):
        with pytest.raises(ValueError, match="other settings"):
            script.main(_args(out, tiny_checkpoint(path, seed=1)))
    # and so do other settings of the same weights
    for flags in (["--seed", "1"], ["--n-permutations", "2"], ["--n-estimators", "2"]):
        with pytest.raises(ValueError, match="other settings"):
            script.main(_args(out, tmp_path / "copy.pt", *flags))

    # the IsolationForest row does not depend on the model: it is reused with other weights
    script.main(_args(out, tmp_path / "b.pt", "--adbench-methods", "iforest"))
    assert results.read_text() == first and len(iforest_calls) == 1
    # and the report is of that run's method only: the TabSil rows in the file are the first
    # model's, so neither their numbers nor the first run's summary.csv are in it
    assert "mean_auroc" not in json.loads(summary.read_text())
    assert not (out / "adbench" / "summary.csv").exists()
    assert (out / "adbench" / "table_d4.csv").exists()


def test_eval_suite_adbench_shards_check_the_settings_before_the_workers(
    suite, tmp_path, monkeypatch
):
    import mlflow
    from train_helpers import in_process_workers, tiny_checkpoint

    script, out, results, iforest_calls = suite
    local = tiny_checkpoint(tmp_path / "a.pt")
    # the parent gets a runs:/ URI and hands its download, a local path, to the workers
    monkeypatch.setattr(
        mlflow.artifacts, "download_artifacts", lambda artifact_uri, dst_path: local
    )
    uri = "runs:/abc123/checkpoints/latest.pt"
    workers = in_process_workers(script, monkeypatch)
    script.main(_args(out, uri, "--shards", "2"))
    assert len(workers) == 2 and all(local in argv and uri not in argv for argv in workers)
    rows = pd.read_csv(results)
    assert sorted(rows.method) == ["iforest", "tabsil_unsup", "tabsil_unsup+clean"]
    assert rows[rows.method != "iforest"].opts.nunique() == 1
    first = results.read_text()

    # the workers' rows carry the settings the parent expects: reruns find every job finished
    script.main(_args(out, uri, "--shards", "2"))
    script.main(_args(out, local))
    assert len(workers) == 2 and results.read_text() == first and len(iforest_calls) == 1

    # other weights: the parent stops before it filters the finished jobs and starts workers
    with pytest.raises(ValueError, match="other settings"):
        script.main(_args(out, tiny_checkpoint(tmp_path / "b.pt", seed=1), "--shards", "2"))
    assert len(workers) == 2 and results.read_text() == first


def test_eval_suite_adbench_without_a_model(suite):
    # only IsolationForest: no model is given, and the --shards parent has no job to start
    script, out, results, _ = suite
    args = ["--suite", "adbench", "--datasets", "yeast", "--adbench-splits", "s1",
            "--shards", "2", "--device", "cpu", "--out-dir", str(out)]  # fmt: skip
    script.main([*args, "--adbench-methods", "iforest"])
    assert list(pd.read_csv(results).method) == ["iforest"]
    assert (out / "adbench" / "summary.json").exists()
    with pytest.raises(SystemExit, match="adbench needs --regressor-path"):
        script.main(args)  # the default methods have TabSil's


def test_eval_suite_adbench_shards_write_the_header_of_an_empty_file(suite, tmp_path, monkeypatch):
    from train_helpers import in_process_workers, tiny_checkpoint

    script, out, results, _ = suite
    results.parent.mkdir(parents=True)
    results.touch()  # a run cut short before its first row
    in_process_workers(script, monkeypatch)
    model = tiny_checkpoint(tmp_path / "a.pt")
    script.main(_args(out, model, "--shards", "2", "--adbench-methods", "tabsil_unsup"))
    rows = pd.read_csv(results)
    assert list(rows.columns) == ab.RESULT_COLUMNS and list(rows.method) == ["tabsil_unsup"]


def test_elo_pool_and_summary_with_our_method():
    ref = ab.reference_results()
    iforest = ref[ref.method == "iforest"]
    ours = iforest.assign(method=ab.OURS, auroc=iforest.auroc + 0.01, aupr=iforest.aupr)
    table = ab.elo(ab.elo_pool(ours)).set_index("method")
    assert table.loc["iforest", "elo"] == 1000
    assert table.loc[ab.OURS, "elo"] > 1000 and table.loc[ab.OURS, "imputed"] == 0
    summary = ab.summary_table(ours).set_index("dataset")
    np.testing.assert_allclose(
        summary[f"{ab.OURS}-{ab.OFFICIAL}"], summary[ab.OURS] - summary[ab.OFFICIAL]
    )
