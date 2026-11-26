"""Imputation suite: ARFF parsing and the sha256 check, split and mask protocol, metrics, the CPU
baselines, run bookkeeping (resume, failures) and the report. No download: tables are synthetic
or monkeypatched."""

from __future__ import annotations

import importlib.util
import json

import numpy as np
import pandas as pd
import pytest

from tabsil_train.benchmarks import impute as bi

ARFF = """@relation tiny
@attribute x numeric
@attribute kind {b, a, c}
@attribute y real
@data
1.5,a,2
2.5,c,?
3.5,b,4
?,a,5
"""


def _table(n: int = 300, seed: int = 0):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal(n)
    b = 2 * a + 0.1 * rng.standard_normal(n)
    c = (a > 0).astype(float)  # categorical, follows a
    d = rng.integers(0, 3, n).astype(float)  # categorical noise
    return np.column_stack([a, b, c, d]), [2, 3], ["a", "b", "c", "d"]


def test_parse_arff_codes_and_drops_rows_with_missing_values():
    X, categorical, names = bi.parse_arff(ARFF)
    assert names == ["x", "kind", "y"] and categorical == [1]
    np.testing.assert_array_equal(X, [[1.5, 1.0, 2.0], [3.5, 0.0, 4.0]])  # codes: declared order
    with pytest.raises(ValueError, match="unsupported"):
        bi.parse_arff(ARFF.replace("@attribute y real", "@attribute y string"))


def test_arff_file_is_checked_against_its_sha256(tmp_path):
    (tmp_path / "cmc.arff").write_text(ARFF)
    with pytest.raises(ValueError, match="sha256"):
        bi.arff_path("cmc", tmp_path)
    assert set(bi.SHA256) == {spec[0] for spec in bi.TABLES.values()}


def test_split_and_mask_protocol():
    X, _, _ = _table(3000)
    rows, mask = bi.make_split("adult", X, 0.3, 0)
    assert rows.shape == (bi.MAX_ROWS, 4) and mask.shape == rows.shape
    assert abs(mask.mean() - 0.3) < 0.01
    again_rows, again_mask = bi.make_split("adult", X, 0.3, 0)
    np.testing.assert_array_equal(again_rows, rows)
    np.testing.assert_array_equal(again_mask, mask)
    other_rows, other_mask = bi.make_split("adult", X, 0.3, 1)
    assert not np.array_equal(other_mask, mask) and not np.array_equal(other_rows, rows)
    small, small_mask = bi.make_split("cmc", X[:500], 0.5, 0)
    np.testing.assert_array_equal(small, X[:500])  # all rows, in order
    assert not np.array_equal(bi.make_split("yeast", X[:500], 0.5, 0)[1], small_mask)
    with pytest.raises(ValueError, match="fewer than 2"):
        bi.make_split("cmc", X[:3], 0.9, 0)


def test_score_and_nearest_class():
    truth = np.array([[0.0, 1.0], [2.0, 0.0], [4.0, 2.0], [6.0, 1.0]])
    mask = np.array([[True, True], [False, True], [True, False], [False, False]])
    guess = truth.copy()
    guess[0] = [1.0, 0.6]  # numeric off by 1; 0.6 -> class 1 (nearest of 0, 1, 2): right
    guess[2, 0] = 4.0
    guess[1, 1] = 1.9  # -> 2: wrong
    s = bi.score(truth, guess, mask, categorical=[1])
    std = truth[:, 0].std()
    assert s["nrmse"] == pytest.approx(np.sqrt(0.5) / std)
    assert s["accuracy"] == pytest.approx(0.5)
    assert (s["masked_numeric"], s["masked_categorical"]) == (2, 2)
    assert json.loads(s["columns"]) == {"0": pytest.approx(s["nrmse"], abs=1e-5), "1": 0.5}
    np.testing.assert_array_equal(
        bi.nearest_class(np.array([-3.0, 0.5, 1.2, 9.0]), np.array([0.0, 1.0, 2.0])),
        [0.0, 0.0, 1.0, 2.0],
    )
    np.testing.assert_array_equal(bi.nearest_class(np.array([3.0]), np.array([5.0])), [5.0])


@pytest.mark.parametrize("method", sorted(bi.CPU_IMPUTERS))
def test_cpu_baselines_fill_every_cell(method):
    X, categorical, _ = _table()
    _, mask = bi.make_split("cmc", X, 0.3, 0)
    X_masked = np.where(mask, np.nan, X)
    out = bi.CPU_IMPUTERS[method](X_masked.copy(), categorical, 0)
    assert out.shape == X.shape and np.isfinite(out).all()
    np.testing.assert_allclose(out[~mask], X[~mask])
    for j in categorical:
        assert np.isin(out[:, j], np.unique(X[:, j])).all()
    s = bi.score(X, out, mask, categorical)
    if method == "mean":
        assert s["nrmse"] == pytest.approx(1.0, abs=0.1)
    else:  # the others use b ~ 2a (KNN's distances also see the noise column)
        assert s["nrmse"] < 0.9


def test_missforest_stops_early_on_numeric_only_tables(monkeypatch):
    X, _, _ = _table()
    X = X[:, :2]
    _, mask = bi.make_split("cmc", X, 0.2, 0)
    rounds = []
    from sklearn.ensemble import RandomForestRegressor

    fit = RandomForestRegressor.fit

    def counting_fit(self, *args, **kwargs):
        rounds.append(1)
        return fit(self, *args, **kwargs)

    monkeypatch.setattr(RandomForestRegressor, "fit", counting_fit)
    out = bi.impute_missforest(np.where(mask, np.nan, X), [], 0, n_trees=10)
    assert np.isfinite(out).all()
    assert len(rounds) < 2 * 10  # two columns per round: stopped before 10 rounds


def test_extension_categorical_rule():
    X, categorical, _ = _table()
    X = np.column_stack([X, np.tile(np.arange(5.0), 60)])  # 5 values, 60 rows each
    X[0, 4] = np.nan  # counted as one more value, as np.unique does
    assert bi.extension_categorical(X, [2]) == [2, 3, 4]
    assert bi.extension_categorical(X[:50], []) == [2, 3]  # 50 / 6 < 10 rows per value for 4


def _fake_load(name, root=None):
    return _table()


def test_run_resumes_and_reports_failures(tmp_path, monkeypatch):
    monkeypatch.setattr(bi, "load", _fake_load)
    calls = []

    def ok(X, categorical, seed):
        calls.append(seed)
        return bi.impute_mean(X, categorical, seed)

    def ours(X, categorical, seed):
        filled = bi.impute_mean(X, categorical, seed)
        return {"tabsil": filled, "tabsil_median": filled}

    def broken(X, categorical, seed):
        raise RuntimeError("boom")

    def leaves_nan(X, categorical, seed):
        return X

    imputers = {"mean": ok, "tabsil": ours, "knn": broken, "mice": leaves_nan}
    args = (["cmc", "yeast"], [0.1, 0.5], [0, 1], ["mean", "tabsil", "knn", "mice"], tmp_path)
    rows, failures = bi.run(imputers, *args, log=lambda _: None)
    assert len(rows) == 2 * 2 * 2 * 3  # mean, tabsil, tabsil_median
    assert len(failures) == 2 * 2 * 2 * 2
    assert {f["method"] for f in failures} == {"knn", "mice"}
    assert "non-finite" in next(f["error"] for f in failures if f["method"] == "mice")
    assert (tmp_path / "failures.json").exists()
    assert len(calls) == 8
    rows, _ = bi.run(imputers, *args, log=lambda _: None)  # resume: nothing reruns
    assert len(calls) == 8 and len(rows) == 24
    mean = rows[rows.method == "mean"]
    assert mean.nrmse.between(0.9, 1.1).all() and mean.accuracy.between(0.2, 0.8).all()


def test_report(tmp_path, monkeypatch):
    monkeypatch.setattr(bi, "load", _fake_load)
    imputers = {m: bi.CPU_IMPUTERS[m] for m in ("mean", "mice")}
    datasets, rates, seeds = ["cmc", "yeast"], [0.1, 0.3], [0, 1]
    rows, failures = bi.run(imputers, datasets, rates, seeds, ["mean", "mice"], tmp_path,
                            log=lambda _: None)  # fmt: skip
    assert not failures
    out = bi.report(rows, tmp_path, datasets, rates, seeds)
    assert out["missing_units"] == {"mean": 0.0, "mice": 0.0}
    summary = pd.read_csv(tmp_path / "summary.csv")
    nrmse = summary[summary.metric == "nrmse"].set_index(["rate", "method"])
    assert nrmse.loc[(0.1, "mice"), "mean"] < nrmse.loc[(0.1, "mean"), "mean"]
    assert nrmse.loc[(0.1, "mice"), "rank"] == 1.0 and nrmse.loc[(0.1, "mean"), "rank"] == 2.0
    assert (nrmse.tables == 2).all()
    assert out["nrmse"]["0.1"]["mice"] == pytest.approx(nrmse.loc[(0.1, "mice"), "mean"], abs=1e-4)
    partial = rows[~((rows.method == "mice") & (rows.dataset == "yeast"))]
    assert bi.report(partial, tmp_path, datasets, rates, seeds)["missing_units"]["mice"] == 0.5
    assert (tmp_path / "per_table_nrmse.csv").exists()


def test_tabsil_imputer_with_a_tiny_model():
    import torch
    from train_helpers import tiny_model

    X, categorical, _ = _table(80)
    _, mask = bi.make_split("cmc", X, 0.2, 0)
    X_masked = np.where(mask, np.nan, X)
    impute = bi.tabsil_imputer(tiny_model(), torch.linspace(-4, 4, 33), 2, "cpu")
    out = impute(X_masked, categorical, 0)
    assert set(out) == {"tabsil", "tabsil_median"}
    for X_imp in out.values():
        assert np.isfinite(X_imp).all()
        np.testing.assert_array_equal(X_imp[~mask], X[~mask])
    inferred = bi.tabsil_imputer(
        tiny_model(), torch.linspace(-4, 4, 33), 2, "cpu", extension_kinds=True
    )
    assert set(inferred(X_masked, categorical, 0)) == {"tabsil_inferred"}


@pytest.mark.skipif(
    importlib.util.find_spec("tabpfn_extensions") is None, reason="needs the impute extra"
)
def test_official_model_counts_classes_over_observed_values(monkeypatch):
    import torch
    from tabpfn_extensions.unsupervised import unsupervised

    # TabPFN v2's class limit without loading weights
    monkeypatch.setattr(unsupervised, "get_max_num_classes", lambda clf: 10)
    model = bi.official_model_class()(tabpfn_clf=object(), tabpfn_reg=object())
    model.categorical_features = [0]
    y = torch.tensor([0.0, 1.0, 2.0] + [float("nan")] * 20)  # 3 classes, 20 missing
    assert model.use_classifier_(0, y)
    assert model.use_classifier_(0, y.numpy())
    assert not model.use_classifier_(1, y)  # not categorical
    assert not model.use_classifier_(0, torch.arange(30.0))  # more classes than TabPFN v2 takes
    # the unpatched extension counts every NaN as a class
    assert not unsupervised.TabPFNUnsupervisedModel.use_classifier_(model, 0, y)


def _eval_script():
    import train_helpers as helpers

    return helpers.load_script("scripts/eval.py", "eval_script_impute")


def test_eval_suite_impute(tmp_path, monkeypatch):
    import json

    script = _eval_script()
    monkeypatch.setattr(bi, "load", _fake_load)
    monkeypatch.delenv("MLFLOW_RUN_ID", raising=False)
    grid = ["--suite", "impute", "--datasets", "cmc,yeast", "--impute-rates", "0.3",
            "--impute-seeds", "0,1", "--out-dir", str(tmp_path)]  # fmt: skip
    script.main([*grid, "--impute-methods", "mean,mice"])
    summary = json.loads((tmp_path / "impute" / "summary.json").read_text())
    assert summary["failures"] == 0
    assert summary["missing_units"] == {"mean": 0.0, "mice": 0.0}
    assert set(summary["nrmse"]["0.3"]) == {"mean", "mice"}
    with pytest.raises(SystemExit, match="regressor-path"):
        script.main([*grid, "--impute-methods", "tabsil"])
    with pytest.raises(SystemExit, match="unknown methods"):
        script.main([*grid, "--impute-methods", "hyperimpute"])
    with pytest.raises(SystemExit, match="not in the impute suite"):
        script.main([*grid, "--datasets", "iris"])
    monkeypatch.setitem(bi.CPU_IMPUTERS, "knn", lambda X, c, s: 1 / 0)  # a method that raises
    with pytest.raises(SystemExit, match="impute: 4 failed"):  # 2 tables x 2 seeds
        script.main([*grid, "--impute-methods", "mean,knn"])
