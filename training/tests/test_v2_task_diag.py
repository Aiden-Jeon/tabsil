"""The tier-1 diagnostics of the v2 plan for priors and tasks (`scripts/v2_prior_diag`): the
forecast and outlier arms, the masks and reference values of D4, the tables of D5 and of the
rare-class check, run on a tiny randomized model or without a model. No benchmark data is read."""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
import train_helpers as helpers

DIAG_DIR = Path(__file__).resolve().parents[1] / "scripts" / "v2_prior_diag"
MODULES = (
    "common",
    "arms",
    "d1_forecast",
    "d2_rare_class",
    "d3_contamination",
    "d4_missing",
    "d5_overlap",
    "adbench_arms",
    "run_all",
)
BORDERS = torch.linspace(-4, 4, 33)


@pytest.fixture(scope="module")
def diag(tmp_path_factory):
    """The scripts of the folder as modules (they import each other by name), writing to a
    temporary folder."""
    out = tmp_path_factory.mktemp("diag")
    saved = {k: os.environ.get(k) for k in ("DIAG_OUT", "DIAG_DEVICE")}
    os.environ.update(DIAG_OUT=str(out), DIAG_DEVICE="cpu")
    sys.path.insert(0, str(DIAG_DIR))
    try:
        yield SimpleNamespace(out=out, **{m: importlib.import_module(m) for m in MODULES})
    finally:
        sys.path.remove(str(DIAG_DIR))
        for name in MODULES:
            sys.modules.pop(name, None)
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _series(y: np.ndarray) -> pd.DataFrame:
    ts = pd.date_range("2024-01-01", periods=len(y), freq="D")
    return pd.DataFrame({"timestamp": ts, "target": y})


# --- forecast arms ----------------------------------------------------------------------------


def test_forecast_arms_set_the_recipe(diag):
    arms = diag.arms
    model = helpers.tiny_model()
    plain8 = arms.ArmForecaster(model, BORDERS, "plain8+trend", n_estimators=4)
    assert plain8.regressor_kwargs == {"member_kinds": ("standard",)}
    assert plain8.trend and not plain8.target_power and plain8.n_estimators == 4
    default = arms.ArmForecaster(model, BORDERS, "default")
    assert default.regressor_kwargs == {} and default.target_power and not default.trend
    assert arms.ArmForecaster(model, BORDERS, "plain").n_estimators == 2
    with pytest.raises(ValueError, match="unknown forecast arm"):
        arms.ArmForecaster(model, BORDERS, "plain9")


def test_no_target_power_only_within_the_block(diag):
    from tabsil.inference.ensemble import EnsembleRegressor
    from tabsil.tasks.regression import TabSilRegressor

    rng = np.random.default_rng(0)
    x, y = rng.normal(size=(40, 3)), rng.lognormal(size=40)
    original = EnsembleRegressor._default_members

    def power_members() -> list[bool]:
        reg = TabSilRegressor(helpers.tiny_model(), BORDERS, n_estimators=4).fit(x, y)
        return [m is not None for m in reg.border_maps_]

    assert power_members() == [False, False, True, True]
    with diag.arms.no_target_power():
        assert power_members() == [False] * 4
    assert EnsembleRegressor._default_members is original
    with pytest.raises(RuntimeError), diag.arms.no_target_power():
        raise RuntimeError
    assert EnsembleRegressor._default_members is original
    assert power_members() == [False, False, True, True]


def test_trend_arm_continues_an_exact_line(diag):
    t = np.arange(50.0)
    fc = diag.arms.ArmForecaster(helpers.tiny_model(), BORDERS, "default+trend", n_estimators=2)
    pred = fc.fit(_series(3.0 + 0.5 * t)).predict(6)
    expected = 3.0 + 0.5 * (50 + np.arange(6))
    np.testing.assert_allclose(pred["predictions"], expected, atol=1e-8)
    np.testing.assert_allclose(pred["0.1"], expected, atol=1e-8)
    np.testing.assert_allclose(pred["0.9"], expected, atol=1e-8)


def test_trend_arm_forecasts_the_residual_and_adds_the_line(diag):
    rng = np.random.default_rng(1)
    t = np.arange(80.0)
    y = np.sin(2 * np.pi * t / 7) + 0.3 * rng.normal(size=80)
    make = lambda: diag.arms.ArmForecaster(  # noqa: E731
        helpers.tiny_model(), BORDERS, "plain8+trend", n_estimators=2
    )
    base = make().fit(_series(y)).predict(5)
    shifted = make().fit(_series(y + 10.0 + 2.0 * t)).predict(5)
    line = 10.0 + 2.0 * (80 + np.arange(5))
    for column in ("predictions", "0.1", "0.5", "0.9"):
        np.testing.assert_allclose(shifted[column], base[column] + line, atol=1e-3)
    # the residual of the first series has no line left, so it is what the model forecasts
    intercept, slope = diag.arms.fit_line(y)
    plain = diag.arms.ArmForecaster(helpers.tiny_model(), BORDERS, "plain8", n_estimators=2)
    residual = plain.fit(_series(y - intercept - slope * t)).predict(5)
    continued = intercept + slope * (80 + np.arange(5))
    np.testing.assert_allclose(base["predictions"], residual["predictions"] + continued, atol=1e-6)


def test_trend_arm_fits_the_rows_the_forecaster_keeps(diag):
    t = np.arange(60.0)
    y = np.where(t < 30, 100.0, 1.0 + 0.25 * (t - 30))  # the first 30 rows are cut
    y[40] = np.nan  # filled forward before the fit
    fc = diag.arms.ArmForecaster(
        helpers.tiny_model(), BORDERS, "default+trend", n_estimators=2, max_context_length=30
    )
    pred = fc.fit(_series(y)).predict(3)
    kept = pd.Series(y).ffill().to_numpy()[30:]
    intercept, slope = diag.arms.fit_line(kept)
    assert abs(slope - 0.25) < 0.01
    assert np.all(np.abs(pred["predictions"] - (intercept + slope * (30 + np.arange(3)))) < 2.0)


# --- outlier arms -----------------------------------------------------------------------------


def _outlier_table(seed: int = 0):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=120)
    X = np.column_stack([a, a + 0.1 * rng.normal(size=120), rng.integers(0, 3, 120) * 0.5])
    return X[:90], X[90:]


def test_aggregate_contexts_readings(diag):
    logs = np.array([[0.0, -2.0, -1.0], [-2.0, -2.0, -1.0], [-4.0, -2.0, -7.0]])
    out = diag.arms.aggregate_contexts("ctx8", logs)
    assert list(out) == ["ctx8", "ctx8-median", "ctx8-mix"]
    np.testing.assert_allclose(out["ctx8"], [2.0, 2.0, 3.0])
    np.testing.assert_allclose(out["ctx8-median"], [2.0, 2.0, 1.0])
    np.testing.assert_allclose(out["ctx8-mix"], -np.log(np.exp(logs).mean(0)))


def test_small_context_scorer_draws_subsets_of_the_context(diag, monkeypatch):
    arms = diag.arms
    X_train, X_test = _outlier_table()
    seen = []
    fit = arms.TabSilUnsupervised.fit

    def record(self, X, y=None):
        seen.append((np.asarray(X).copy(), self.categorical_features, self.random_state))
        return fit(self, X, y)

    monkeypatch.setattr(arms.TabSilUnsupervised, "fit", record)
    scorer = arms.small_context_scorer(helpers.tiny_model(), BORDERS, "cpu", 20, n_contexts=3)
    out = scorer(X_train, X_test, seed=1)
    assert list(out) == ["ctx20", "ctx20-median", "ctx20-mix"]
    assert all(s.shape == (30,) and np.isfinite(s).all() for s in out.values())
    assert len(seen) == 3 and len({ctx.tobytes() for ctx, _, _ in seen}) == 3
    rows = {r.tobytes() for r in X_train}
    for context, categorical, _ in seen:
        assert context.shape == (20, 3) and all(r.tobytes() in rows for r in context)
        assert categorical == [2]  # the kinds of the whole context, not of the subset
    assert len({state for _, _, state in seen}) == 3
    again = scorer(X_train, X_test, seed=1)
    np.testing.assert_allclose(again["ctx20"], out["ctx20"])
    # more rows than the context has: the whole context every time
    seen.clear()
    whole = arms.small_context_scorer(helpers.tiny_model(), BORDERS, "cpu", None, n_contexts=2)
    assert list(whole(X_train, X_test, seed=1)) == ["ctxfull", "ctxfull-median", "ctxfull-mix"]
    assert all(context.shape == X_train.shape for context, _, _ in seen)


def test_reference_rows(diag):
    X_train, _ = _outlier_table()
    rng = np.random.default_rng(0)
    marginals = diag.arms.reference_rows(X_train, "marginals", rng)
    assert marginals.shape == X_train.shape and not np.array_equal(marginals, X_train)
    for j in range(3):
        np.testing.assert_array_equal(np.sort(marginals[:, j]), np.sort(X_train[:, j]))
    uniform = diag.arms.reference_rows(X_train, "uniform", rng)
    lo, hi = X_train.min(0), X_train.max(0)
    pad = diag.arms.PAD * (hi - lo)
    assert (uniform >= lo - pad).all() and (uniform <= hi + pad).all()
    assert (uniform[:, 0] > hi[0]).any()  # the padding is used
    assert set(uniform[:, 2]) <= set(X_train[:, 2])  # a categorical column keeps its values
    with pytest.raises(ValueError, match="unknown reference kind"):
        diag.arms.reference_rows(X_train, "normal", rng)


def test_reference_and_refined_scorers(diag, monkeypatch):
    arms = diag.arms
    X_train, X_test = _outlier_table()
    model = helpers.tiny_model()
    for kind in ("uniform", "marginals"):
        out = arms.reference_scorer(model, "cpu", kind, draws=2, n_estimators=2)(X_train, X_test, 1)
        scores = out[f"ref-{kind}"]
        assert scores.shape == (30,) and ((scores >= 0) & (scores <= 1)).all()
    contexts = []
    fit = arms.TabSilUnsupervised.fit

    def record(self, X, y=None):
        contexts.append(len(X))
        return fit(self, X, y)

    monkeypatch.setattr(arms.TabSilUnsupervised, "fit", record)
    refined = arms.refined_scorer(model, BORDERS, "cpu", drop=0.1, n_permutations=2)
    out = refined(X_train, X_test, 1)
    assert contexts == [90, 81]
    assert out["refine"].shape == (30,) and np.isfinite(out["refine"]).all()
    assert set(arms.outlier_scorers(model, BORDERS, "cpu")) == {
        "density", "ctx64", "ctx256", "ctxfull", "ref-uniform", "ref-marginals", "refine",
    }  # fmt: skip


# --- D1 and D3 with arms ----------------------------------------------------------------------


def test_d1_runs_the_arms_of_the_environment(diag, monkeypatch, capsys):
    d1 = diag.d1_forecast
    monkeypatch.setattr(d1, "load_model", lambda: (helpers.tiny_model(), BORDERS))
    monkeypatch.setattr(d1, "ARMS", ["plain", "plain+trend"])
    monkeypatch.setattr(d1, "SEEDS", range(1))
    monkeypatch.setattr(d1, "L", 70)
    monkeypatch.setattr(d1, "H", 14)
    d1.main()
    plain = pd.read_csv(diag.out / "d1_forecast_plain.csv")
    trend = pd.read_csv(diag.out / "d1_forecast_plain+trend.csv")
    assert len(plain) == len(trend) == 10 and set(trend.arm) == {"plain+trend"}
    # the series and the baselines do not depend on the arm
    np.testing.assert_allclose(plain.mase_ols, trend.mase_ols)
    assert "arm plain+trend" in capsys.readouterr().out


def test_d3_scores_every_arm_on_the_same_tables(diag, monkeypatch):
    d3 = diag.d3_contamination
    ctx, test, labels = d3.tables("cluster", 0.1, 0)
    assert ctx.shape == (600, 6) and test.shape == (330, 6) and labels.sum() == 30
    again = d3.tables("cluster", 0.1, 0)
    np.testing.assert_array_equal(again[0], ctx)
    monkeypatch.setattr(d3, "load_model", lambda: (helpers.tiny_model(), BORDERS))
    monkeypatch.setattr(d3, "N_CTX", 80)
    monkeypatch.setattr(d3, "N_IN", 30)
    monkeypatch.setattr(d3, "N_OUT", 6)
    monkeypatch.setattr(d3, "KINDS", ["cluster"])
    monkeypatch.setattr(d3, "FRACTIONS", (0.1,))
    monkeypatch.setattr(d3, "SEEDS", range(1))
    monkeypatch.setattr(d3, "N_PERMUTATIONS", 1)
    monkeypatch.setattr(d3, "ARMS", ["density", "ref-marginals"])
    d3.main()
    wide = pd.read_csv(diag.out / "d3_contamination.csv")
    arms = pd.read_csv(diag.out / "d3_contamination_arms.csv")
    assert list(arms.arm) == ["density", "ref-marginals"]
    assert wide.auroc_tabsil.iloc[0] == pytest.approx(arms.auroc.iloc[0])
    assert wide.inlier_logdens.iloc[0] == pytest.approx(-arms.inlier_score.iloc[0])
    monkeypatch.setattr(d3, "ARMS", ["nothing"])
    with pytest.raises(SystemExit, match="unknown D3_ARMS"):
        d3.main()


# --- D2, rare class ---------------------------------------------------------------------------


def test_rare_class_labels_and_pseudo_count(diag):
    d2 = diag.d2_rare_class
    y = d2.labels(50, 3, 2, np.random.default_rng(0))
    assert np.bincount(y).tolist() == [2, 24, 24]
    y = d2.labels(200, 10, 3, np.random.default_rng(0))
    assert np.bincount(y)[0] == 3 and len(np.bincount(y)) == 10 and np.bincount(y)[1:].min() >= 21
    for a in (0.0, 0.5, 1.0):
        p = (3 + a) / (200 + 10 * a)
        assert d2.pseudo_count(p, 200, 10, 3) == pytest.approx(a)


# --- D4 ---------------------------------------------------------------------------------------


def test_d4_masks_have_the_rate_and_the_structure(diag, monkeypatch):
    d4 = diag.d4_missing
    monkeypatch.setattr(d4, "N_ROWS", 200_000)
    X = d4.gaussian_table(0)
    assert np.corrcoef(X[:, 0], X[:, 1])[0, 1] == pytest.approx(d4.RHO, abs=0.01)
    masks = {m: d4.make_mask(m, X, [(0, 1)], np.random.default_rng(1)) for m in d4.MECHANISMS}
    for name, mask in masks.items():
        assert mask[:, 0].mean() == pytest.approx(d4.RATE, abs=0.005), name
        assert not mask[:, 2:].any()
    for name in ("mcar", "by_other", "self"):
        assert not masks[name][:, 1].any()
    # by another column: the rows to impute have a larger driver, not (beyond the correlation) a
    # larger target at a given driver
    by_other = masks["by_other"][:, 0]
    assert X[by_other, 1].mean() - X[~by_other, 1].mean() > 1.0
    # self-masking: the missing values are larger (section 5: kappa lambda and its counterpart)
    self_mask = masks["self"][:, 0]
    alpha, kappa = d4.offset(d4.B_SELF) / np.sqrt(2), 1 / np.sqrt(2)
    from scipy.stats import norm

    assert X[self_mask, 0].mean() == pytest.approx(kappa * norm.pdf(alpha) / d4.RATE, abs=0.01)
    # the driver's mask: only where the target is observed, or in every row
    obs, every = masks["other_obs"], masks["other_all"]
    assert not (obs[:, 0] & obs[:, 1]).any()
    assert obs[~obs[:, 0], 1].mean() == pytest.approx(d4.RATE, abs=0.005)
    assert every[every[:, 0], 1].mean() == pytest.approx(d4.RATE, abs=0.01)
    assert X[every[:, 1], 0].mean() > 0.9  # hidden where the target is large
    with pytest.raises(ValueError, match="unknown mechanism"):
        d4.make_mask("mnar", X, [(0, 1)], np.random.default_rng(1))


def test_d4_reference_values_follow_the_derivation(diag, monkeypatch):
    """The values of `v2-task-formulations.md` section 5 for a fit on the complete cases."""
    d4 = diag.d4_missing
    monkeypatch.setattr(d4, "N_ROWS", 400_000)
    X = d4.gaussian_table(0)
    ref = {}
    for mechanism in d4.MECHANISMS:
        mask = d4.make_mask(mechanism, X, [(0, 1)], np.random.default_rng(2))
        ref[mechanism] = d4.gaussian_references(X, mask, mechanism)
    best = np.sqrt(1 - d4.RHO**2)
    # completely at random and by another column: the same conditional, no bias
    for mechanism in ("mcar", "by_other"):
        assert abs(ref[mechanism]["ols_bias"]) < 0.02
        assert ref[mechanism]["ols_rmse"] == pytest.approx(best, abs=0.01)
    # the counter-example: the rows that show the driver are a tilted sample
    assert ref["other_obs"]["ols_bias"] == pytest.approx(-0.30, abs=0.03)
    assert abs(ref["other_obs"]["full_bias"]) < 0.01
    assert ref["other_obs"]["driver_hidden"] == 0.0
    # the same mask in every row: the tilt holds for the rows to impute too
    assert abs(ref["other_all"]["ols_bias_shown"]) < 0.03
    assert abs(ref["other_all"]["ols_bias_hidden"]) < 0.02
    assert ref["other_all"]["full_bias_hidden"] < -0.9
    # self-masking: the closed form of the complete cases' bias
    assert ref["self"]["closed_bias"] == pytest.approx(ref["self"]["ols_bias"], abs=0.05)
    assert ref["self"]["closed_bias"] < -0.8
    assert np.isnan(ref["mcar"]["closed_bias"])


def test_d4_column_pairs(diag):
    d4 = diag.d4_missing
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=300), rng.normal(size=300)
    X = np.column_stack(
        [a, b, rng.integers(0, 3, 300), a + 0.1 * rng.normal(size=300), np.ones(300),
         -b + 0.5 * rng.normal(size=300), rng.normal(size=300)]
    )  # fmt: skip
    assert d4.column_pairs(X, categorical=[2]) == [(0, 3), (1, 5)]
    assert d4.column_pairs(X[:, [0, 3]], categorical=[]) == [(0, 1)]
    assert d4.column_pairs(X[:, [0, 2, 4]], categorical=[1]) == []
    scores = d4.normal_scores(X[:, [0, 2]])
    assert np.argsort(scores[:, 0]).tolist() == np.argsort(X[:, 0]).tolist()
    assert len(np.unique(scores[:, 1])) == 3 and np.isfinite(scores).all()


def test_d4_parts_run(diag, monkeypatch):
    d4 = diag.d4_missing
    from tabsil_train.benchmarks.unsupervised import impute

    monkeypatch.setattr(d4, "N_ROWS", 120)
    monkeypatch.setattr(d4, "SEEDS", (0,))
    rng = np.random.default_rng(0)
    a = rng.normal(size=150)
    table = np.column_stack(
        [a, a + 0.3 * rng.normal(size=150), rng.integers(0, 2, 150), rng.normal(size=150)]
    )
    monkeypatch.setattr(impute, "load", lambda name, root=None: (table, [2], list("abcd")))
    tabsil = impute.tabsil_imputer(helpers.tiny_model(), BORDERS, 2, "cpu")
    gaussian = d4.run_gaussian(tabsil)
    assert list(gaussian.mechanism) == list(d4.MECHANISMS)
    assert np.isfinite(gaussian[["model_bias", "model_rmse", "ols_rmse", "full_rmse"]]).all().all()
    suite = d4.run_suite({"tabsil": tabsil, "mean": impute.impute_mean}, ["yeast"])
    assert len(suite) == len(d4.MECHANISMS) * 2 and set(suite.targets) == {1}
    by_method = suite.pivot(index="mechanism", columns="method", values="nrmse")
    # the mean of the observed values misses self-masked cells by more than random ones
    assert by_method.loc["self", "mean"] > by_method.loc["mcar", "mean"]
    assert suite[suite.method == "mean"].set_index("mechanism").bias["self"] < -0.5


# --- D5 ---------------------------------------------------------------------------------------


def test_d5_tables(diag):
    d5 = diag.d5_overlap
    wide, narrow = d5.make_table(3, 0.5), d5.make_table(3, 0.02)
    for key in ("X", "X_test", "tau", "raw"):
        np.testing.assert_array_equal(wide[key], narrow[key])  # one realization, two clips
    assert wide["X"].shape == (d5.N_TRAIN, d5.N_FEATURES) and wide["tau"].shape == (d5.N_TEST,)
    assert wide["tau"].mean() == pytest.approx(1.0, abs=0.1)
    assert wide["tau"].std() == pytest.approx(0.5, abs=0.1)
    # randomized at clip 0.5; at 0.02 the treatment follows the features
    u = wide["X"][:, :2]
    fit = lambda t: np.linalg.lstsq(np.c_[np.ones(len(u)), u], t, rcond=None)[0][1:]  # noqa: E731
    assert np.abs(fit(wide["t"])).max() < 0.06
    assert np.abs(fit(narrow["t"])).max() > 0.15
    assert 0.3 < narrow["t"].mean() < 0.7
    assert d5.pehe(np.array([1.0, 3.0]), np.array([1.0, 1.0])) == pytest.approx(np.sqrt(2.0))


# --- runners ----------------------------------------------------------------------------------


def test_adbench_arms_runner(diag, monkeypatch, capsys):
    runner = diag.adbench_arms
    rng = np.random.default_rng(0)
    y = (rng.random(1200) < 0.1).astype(int)
    X = rng.normal(size=(1200, 3)) + 3 * y[:, None]
    monkeypatch.setattr(runner.adbench, "load", lambda dataset, root=None: (X, y))
    monkeypatch.setattr(runner, "load_model", lambda: (helpers.tiny_model(), BORDERS))
    monkeypatch.setenv("ADBENCH_ARMS", "iforest,ref-marginals")
    monkeypatch.setenv("ADBENCH_DATASETS", "cardio")
    runner.main()
    rows = pd.read_csv(diag.out / "adbench_arms.csv")
    assert list(rows.arm) == ["iforest", "ref-marginals"] * 3
    assert list(rows.split) == ["s1", "s1", "s2", "s2", "s3", "s3"]
    assert rows[rows.arm == "iforest"].auroc.min() > 0.95
    assert "mean over 3 cells" in capsys.readouterr().out
    monkeypatch.setenv("ADBENCH_ARMS", "lof")
    with pytest.raises(SystemExit, match="unknown ADBENCH_ARMS"):
        runner.main()


def test_run_all_defaults_and_names(diag):
    run_all = diag.run_all
    assert run_all.DEFAULT == ("d1_forecast", "d2_calibration", "d3_contamination")
    assert set(run_all.DEFAULT) < set(run_all.DIAGNOSTICS)
    for name in run_all.DIAGNOSTICS:
        assert (DIAG_DIR / f"{name}.py").is_file(), name
    with pytest.raises(SystemExit, match="unknown diagnostics"):
        run_all.main(["d9_nothing"])
