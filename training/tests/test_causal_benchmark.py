"""Treatment effect benchmark suite (`tabsil_train.benchmarks.causal`) on small synthetic files in
the benchmarks' formats (no download)."""

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from tabsil_train.benchmarks import causal

N_ACIC = 200


def _pin(root, monkeypatch):
    """Point the cache at `root` and pin the hashes of the files written there."""
    monkeypatch.setenv("TABSIL_CAUSAL_DIR", str(root))
    pins = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir()}
    monkeypatch.setattr(causal, "SHA256", pins)


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    for name, n in (("ihdp_npci_1-100.train.npz", 30), ("ihdp_npci_1-100.test.npz", 8)):
        mu0 = rng.normal(size=(n, 100))
        np.savez(
            tmp_path / name,
            x=rng.normal(size=(n, 25, 100)),
            t=(rng.uniform(size=(n, 100)) < 0.4).astype(float),
            yf=rng.normal(size=(n, 100)),
            ycf=rng.normal(size=(n, 100)),
            mu0=mu0,
            mu1=mu0 + 4,
            ate=np.int64(4),
            yadd=np.int64(0),
            ymul=np.int64(1),
        )
    x = pd.DataFrame(rng.normal(size=(N_ACIC, 58)), columns=[f"c{i}" for i in range(58)])
    for j, values in ((1, "ABC"), (20, "AB"), (23, "ABCDE")):  # x_2, x_21, x_24
        x[f"c{j}"] = rng.choice(list(values), size=N_ACIC)
    x.to_csv(tmp_path / "x.csv", index=False)
    for i in range(1, 11):
        mu0 = rng.normal(size=N_ACIC)
        pd.DataFrame(
            {
                "z": (rng.uniform(size=N_ACIC) < 0.3).astype(int),
                "y0": mu0 + rng.normal(size=N_ACIC),
                "y1": mu0 + i + rng.normal(size=N_ACIC),
                "mu0": mu0,
                "mu1": mu0 + i,
            }
        ).to_csv(tmp_path / f"zymu_{i}.csv", index=False)
    _pin(tmp_path, monkeypatch)
    return tmp_path


def test_ihdp_split_uses_the_realization_column(data_dir):
    s = causal.ihdp_split(3)
    with np.load(data_dir / "ihdp_npci_1-100.train.npz") as d:
        np.testing.assert_array_equal(s.X_train, d["x"][..., 2])
        np.testing.assert_array_equal(s.y_train, d["yf"][:, 2])
        np.testing.assert_allclose(s.tau_train, d["mu1"][:, 2] - d["mu0"][:, 2])
    assert s.X_train.shape == (30, 25) and s.X_test.shape == (8, 25)
    assert set(np.unique(s.t_train)) <= {0, 1}
    with pytest.raises(ValueError, match="1..100"):
        causal.ihdp_split(0)


def test_acic_split_follows_causalpfn(data_dir):
    s = causal.acic_split(4)
    cut = int(N_ACIC * 0.9)
    assert len(s.X_train) == cut and len(s.X_test) == N_ACIC - cut
    idx = np.random.default_rng(42 + 3).permutation(N_ACIC)  # simulation 4 -> seed 45
    sim = pd.read_csv(data_dir / "zymu_4.csv")
    t = sim.z.to_numpy()
    np.testing.assert_array_equal(s.t_train, t[idx[:cut]])
    np.testing.assert_allclose(s.y_train, np.where(t == 1, sim.y1, sim.y0)[idx[:cut]])
    np.testing.assert_allclose(s.tau_test, (sim.mu1 - sim.mu0).to_numpy()[idx[cut:]])
    # categorical x_2, x_21, x_24 as sorted category codes, appended after the other 55 columns
    x = pd.read_csv(data_dir / "x.csv")
    X = causal.acic_covariates()
    assert X.shape == (N_ACIC, 58)
    np.testing.assert_array_equal(X[:, -3], pd.Categorical(x["c1"]).codes)
    np.testing.assert_array_equal(X[:, -1], pd.Categorical(x["c23"]).codes)
    np.testing.assert_allclose(X[:, 0], x["c0"])
    np.testing.assert_allclose(X[:, 1], x["c2"])  # x_2 moved to the end
    np.testing.assert_allclose(s.X_train, X[idx[:cut]])


def test_a_changed_file_is_rejected(data_dir, monkeypatch):
    monkeypatch.setitem(causal.SHA256, "x.csv", "0" * 64)
    with pytest.raises(ValueError, match="sha256"):
        causal.acic_covariates()


def test_metrics():
    s = causal.Split(
        np.zeros((3, 1)), np.array([0, 1, 1]), np.zeros(3), np.array([1.0, 2.0, 3.0]),
        np.zeros((2, 1)), np.array([0.0, 4.0]),
    )  # fmt: skip
    m = causal.metrics(s, np.array([1.0, 2.0, 6.0]), np.array([1.0, 1.0]))
    assert m["pehe_in"] == pytest.approx(np.sqrt(3.0))
    assert m["ate_err_in"] == pytest.approx(1.0)
    assert m["pehe_out"] == pytest.approx(np.sqrt(5.0))
    assert m["ate_err_out"] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="non-finite"):
        causal.metrics(s, np.array([1.0, np.nan, 2.0]), np.zeros(2))
    with pytest.raises(ValueError, match="shape"):
        causal.metrics(s, np.zeros(2), np.zeros(2))


def test_parse_realizations():
    assert causal.parse_realizations(None, "acic2016") == list(range(1, 11))
    assert causal.parse_realizations("1-3,7,2", "ihdp") == [1, 2, 3, 7]
    with pytest.raises(ValueError, match="1..10"):
        causal.parse_realizations("9-11", "acic2016")
    with pytest.raises(ValueError, match="unknown methods"):
        causal.check_methods(["tabsil_t", "bart"])


def _oracle(split, seed):
    return split.tau_train + 0.1, split.tau_test - 0.2


def test_run_records_resumes_and_fails_without_imputing(data_dir, tmp_path):
    calls = []

    def broken(split, seed):
        calls.append(seed)
        if seed == 2:
            raise RuntimeError("boom")
        if seed == 3:
            return split.tau_train * np.nan, split.tau_test
        return _oracle(split, seed)

    plan = {"ihdp": [1, 2, 3], "acic2016": [5]}
    out = tmp_path / "out"
    estimators = {"tabsil_t": _oracle, "econml_t": broken}
    rows, failures = causal.run(estimators, plan, ["tabsil_t", "econml_t"], out, log=lambda s: 0)
    assert len(rows) == 6  # 4 cells of the oracle, 2 of broken (ihdp 1, acic 5)
    assert {(f["realization"], f["method"]) for f in failures} == {(2, "econml_t"), (3, "econml_t")}
    assert json.loads((out / "failures.json").read_text())[0]["error"].startswith("RuntimeError")
    ok = rows[rows.method == "tabsil_t"]
    np.testing.assert_allclose(ok.pehe_in, 0.1)
    np.testing.assert_allclose(ok.ate_err_out, 0.2)
    assert set(rows.columns) == set(causal.RESULT_COLUMNS)

    summary = causal.report(rows, out, plan, ["tabsil_t", "econml_t"])
    assert summary["missing_pct"] == pytest.approx(100 * 2 / 8)
    assert summary["methods"]["ihdp/tabsil_t"]["realizations"] == 3
    table = pd.read_csv(out / "summary.csv")
    assert {"pehe_out", "pehe_out_se"} <= set(table.columns)

    # a rerun computes only the cells without a row
    calls.clear()
    rows, failures = causal.run(estimators, plan, ["tabsil_t", "econml_t"], out, log=lambda s: 0)
    assert sorted(calls) == [2, 3] and len(rows) == 6 and len(failures) == 2


def test_tabsil_estimator_runs_on_a_tiny_model(data_dir):
    import torch

    from tabsil.models import randomize_zero_weights
    from tabsil.models.column_row_transformer import (
        ColumnRowTransformer,
        ColumnRowTransformerConfig,
    )

    torch.manual_seed(0)
    model = randomize_zero_weights(ColumnRowTransformer(ColumnRowTransformerConfig.tiny())).eval()
    est = causal.tabsil_estimator(model, torch.linspace(-4, 4, 33), "x", 2, "cpu")
    split = causal.ihdp_split(1)
    effect_in, effect_out = est(split, 1)
    assert effect_in.shape == (30,) and effect_out.shape == (8,)
    assert np.isfinite(causal.metrics(split, effect_in, effect_out)["pehe_out"])


@pytest.mark.parametrize("method", causal.ECONML)
def test_econml_baselines(data_dir, method):
    pytest.importorskip("econml")
    split = causal.ihdp_split(1)
    effect_in, effect_out = causal.econml_estimator(method)(split, 0)
    assert effect_in.shape == (30,) and effect_out.shape == (8,)
    assert np.isfinite(effect_in).all() and np.isfinite(effect_out).all()


def test_reference_numbers_are_well_formed():
    ref = causal.reference()
    assert {"source", "protocol", "benchmark", "method"} <= set(ref.columns)
    assert set(ref.benchmark) <= set(causal.BENCHMARKS)
    values = ref[[c for c in ref.columns if c in causal.METRICS]]
    assert values.notna().any(axis=1).all()  # every row has a number


def _eval_script():
    import train_helpers as helpers

    return helpers.load_script("scripts/eval.py", "eval_script_causal")


def _checkpoint(path):
    import torch

    from tabsil.models import randomize_zero_weights, save_checkpoint
    from tabsil.models.column_row_transformer import (
        ColumnRowTransformer,
        ColumnRowTransformerConfig,
    )

    torch.manual_seed(0)
    model = randomize_zero_weights(ColumnRowTransformer(ColumnRowTransformerConfig.tiny())).eval()
    save_checkpoint(path, model, torch.linspace(-4, 4, 33))
    return str(path)


def test_eval_suite_causal_end_to_end(data_dir, tmp_path):
    script = _eval_script()
    out = tmp_path / "eval"
    args = ["--suite", "causal", "--regressor-path", _checkpoint(tmp_path / "m.pt"),
            "--datasets", "ihdp,acic2016", "--causal-realizations", "1-2",
            "--causal-methods", "tabsil_t,tabsil_s", "--n-estimators", "2", "--device", "cpu",
            "--out-dir", str(out)]  # fmt: skip
    script.main(args)
    summary = json.loads((out / "causal" / "summary.json").read_text())
    assert summary["planned_cells"] == 8 and summary["results"] == 8
    assert summary["missing_pct"] == 0 and summary["failures"] == 0
    assert set(summary["methods"]) == {f"{b}/{m}" for b in ("ihdp", "acic2016")
                                       for m in ("tabsil_t", "tabsil_s")}  # fmt: skip
    rows = pd.read_csv(out / "causal" / "results.csv")
    assert (rows.opts == "n_estimators=2").all()


def test_eval_suite_causal_argument_errors(data_dir, tmp_path):
    script = _eval_script()
    base = ["--suite", "causal", "--out-dir", str(tmp_path)]
    with pytest.raises(SystemExit, match="benchmarks"):
        script.main([*base, "--datasets", "jobs"])
    with pytest.raises(SystemExit, match="1..100"):
        script.main([*base, "--datasets", "ihdp", "--causal-realizations", "101"])
    with pytest.raises(SystemExit, match="unknown methods"):
        script.main([*base, "--causal-methods", "bart"])
    with pytest.raises(SystemExit, match="regressor-path"):
        script.main([*base, "--causal-methods", "tabsil_t"])
