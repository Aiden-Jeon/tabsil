"""ADBench suite: split protocol, run bookkeeping (resume, failures) and the Elo against the
copied fm-agent-bench results. No download: tables are synthetic or monkeypatched."""

from __future__ import annotations

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
