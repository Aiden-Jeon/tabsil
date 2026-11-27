"""TabSilUnsupervised.impute: missing values from the per-column conditional distributions (tiny
randomized model on the CPU; `test_recovers_obvious_patterns_with_v1_weights` uses the v1 weights
when they are in the local Hugging Face cache)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from helpers import tiny_model

from tabsil.tasks.regression import TabSilRegressor
from tabsil.tasks.unsupervised import (
    CategoricalConditional,
    NumericConditional,
    TabSilUnsupervised,
    nearest_class,
)

BORDERS = torch.linspace(-4, 4, 33)


def _model(**kwargs) -> TabSilUnsupervised:
    return TabSilUnsupervised(tiny_model(), BORDERS, n_estimators=2, **kwargs)


def _table(n: int = 60, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.standard_normal(n)
    b = 2 * a + 1 + 0.1 * rng.standard_normal(n)
    c = (a > 0).astype(float) * 0.5 + 1.0  # categorical: 1.0 or 1.5, follows a
    return np.column_stack([a, b, c])


def _masked(x: np.ndarray, rate: float = 0.2, seed: int = 1) -> np.ndarray:
    out = x.copy()
    out[np.random.default_rng(seed).random(x.shape) < rate] = np.nan
    return out


def test_shapes_determinism_and_observed_cells_kept():
    x = _masked(_table())
    model = _model(categorical_features=[2]).fit(x)
    out = model.impute(x)
    assert out.shape == x.shape and out.dtype == np.float64
    assert np.isfinite(out).all()
    observed = ~np.isnan(x)
    np.testing.assert_array_equal(out[observed], x[observed])
    np.testing.assert_array_equal(_model(categorical_features=[2]).fit(x).impute(x), out)
    # numeric cells within the fit rows' range, categorical cells among the classes
    assert np.isin(out[:, 2], [1.0, 1.5]).all()
    for j in (0, 1):
        assert np.nanmin(x[:, j]) <= out[:, j].min() and out[:, j].max() <= np.nanmax(x[:, j])
    median = model.impute(x, strategy="median")
    np.testing.assert_array_equal(median[observed], x[observed])
    np.testing.assert_array_equal(median[:, 2], out[:, 2])  # categorical: the mode either way


def test_multiple_imputation():
    x = _masked(_table())
    model = _model(categorical_features=[2]).fit(x)
    draws = model.impute(x, n_samples=5)
    assert draws.shape == (5, *x.shape) and np.isfinite(draws).all()
    observed = ~np.isnan(x)
    for d in draws:
        np.testing.assert_array_equal(d[observed], x[observed])
    assert np.isin(draws[..., 2], [1.0, 1.5]).all()
    missing_a = np.isnan(x[:, 0])
    assert len(np.unique(draws[:, missing_a, 0])) > 1  # draws differ
    again = _model(categorical_features=[2]).fit(x).impute(x, n_samples=5)
    np.testing.assert_array_equal(again, draws)
    with pytest.raises(ValueError, match="n_samples"):
        model.impute(x, n_samples=0)
    with pytest.raises(ValueError, match="strategy"):
        model.impute(x, strategy="mode")


def test_one_evaluation_per_missing_column_given_all_other_columns(monkeypatch):
    x = _masked(_table())
    x[:, 1] = _table()[:, 1]  # column 1 observed everywhere: not evaluated
    model = _model(categorical_features=[2]).fit(x)
    calls = []
    real = TabSilUnsupervised.conditional

    def spy(self, X, target, given):
        calls.append((X.copy(), target, list(given)))
        return real(self, X, target, given)

    monkeypatch.setattr(TabSilUnsupervised, "conditional", spy)
    model.impute(x)
    assert [(t, g) for _, t, g in calls] == [(0, [1, 2]), (2, [0, 1])]
    for X_rows, target, _ in calls:
        rows = np.isnan(x[:, target])
        # the rows missing the target, with their other missing cells still missing
        np.testing.assert_array_equal(X_rows, x[rows])


def test_no_missing_value_is_returned_as_is(monkeypatch):
    x = _table()
    model = _model().fit(x)
    monkeypatch.setattr(TabSilUnsupervised, "conditional", lambda *a: pytest.fail("evaluated"))
    out = model.impute(x)
    np.testing.assert_array_equal(out, x)
    assert out is not x
    assert model.impute(x, n_samples=2).shape == (2, *x.shape)


def test_columns_missing_everywhere():
    x = _table()
    fit_rows = x[:40].copy()
    fit_rows[:, 0] = np.nan  # no observed value of column 0 to learn from
    model = _model(categorical_features=[2]).fit(fit_rows)
    test = x[40:].copy()
    test[3, 0] = np.nan
    with pytest.raises(ValueError, match="no observed value"):
        model.impute(test)
    test = x[40:].copy()
    test[:, 1] = np.nan  # missing in every row of X, observed in the fit rows: imputed
    out = _model(categorical_features=[2]).fit(x[:40]).impute(test)
    assert np.isfinite(out).all()
    np.testing.assert_array_equal(out[:, [0, 2]], test[:, [0, 2]])
    # a row with every cell missing still gets values (from the missing pattern alone)
    test[0] = np.nan
    assert np.isfinite(_model(categorical_features=[2]).fit(x[:40]).impute(test)).all()


def test_point_values_from_the_distributions():
    model = _model(categorical_features=[2]).fit(_table())
    x = np.array([[np.nan, 1.0, np.nan], [0.5, np.nan, np.nan]])
    borders = np.array([-1.0, 0.0, 1.0, 2.0])
    num0 = NumericConditional(np.log(np.array([[0.1, 0.2, 0.7]])), borders, mean=np.array([0.25]))
    num1 = NumericConditional(np.log(np.array([[0.5, 0.25, 0.25]])), borders)  # no mean
    cat = CategoricalConditional(np.array([[0.3, 0.7], [0.6, 0.4]]), np.array([1.0, 1.5]))
    conds = {0: (np.array([0]), num0), 1: (np.array([1]), num1), 2: (np.array([0, 1]), cat)}
    median = model.impute_from(x, conds, "median")
    # median of [0.1, 0.2, 0.7] on [-1, 0, 1, 2]: 0.5 - 0.3 = 0.2 of the last bucket's 0.7
    np.testing.assert_allclose(median[0, 0], 1.0 + 0.2 / 0.7)
    np.testing.assert_allclose(median[1, 1], 0.0)
    np.testing.assert_array_equal(median[:, 2], [1.5, 1.0])  # categorical: the mode
    with pytest.raises(ValueError, match="no mean"):  # "mean" without a mean: no silent median
        model.impute_from(x, conds, "mean")
    conds[1] = (np.array([1]), NumericConditional(num1.log_probs, borders, mean=np.array([0.4])))
    mean = model.impute_from(x, conds, "mean")
    np.testing.assert_allclose(mean, [[0.25, 1.0, 1.5], [0.5, 0.4, 1.0]])
    with pytest.raises(ValueError, match="conditionals"):
        model.impute_from(x, {0: conds[0]}, "mean")


def test_distribution_helpers():
    borders = np.array([0.0, 1.0, 1.0, 3.0])  # a zero-width bucket at 1
    cond = NumericConditional(np.log(np.array([[0.5, 1e-30, 0.5]] * 2)), borders)
    np.testing.assert_allclose(cond.quantile(0.5), [1.0, 1.0])
    np.testing.assert_allclose(cond.quantile(0.75), [2.0, 2.0])
    draws = cond.sample(np.random.default_rng(0), 4000)
    assert draws.shape == (2, 4000)
    assert ((draws >= 0) & (draws <= 3)).all()
    assert abs((draws < 1).mean() - 0.5) < 0.03  # half the mass below 1, uniform within
    assert abs((draws[draws > 1] < 2).mean() - 0.5) < 0.03
    cat = CategoricalConditional(np.array([[0.2, 0.8], [0.5, 0.5]]), np.array([0.0, 2.0]))
    np.testing.assert_array_equal(cat.mode(), [2.0, 0.0])  # a tie: the smaller class
    picks = cat.sample(np.random.default_rng(0), 4000)
    assert set(np.unique(picks)) == {0.0, 2.0}
    assert abs((picks[0] == 2.0).mean() - 0.8) < 0.03


def test_class_mode_takes_the_most_probable_class_not_the_mean():
    # half the probability near class 0, half near class 40: the mean (20) is a class with
    # almost no probability; the mode is 0 or 40
    borders = np.linspace(-0.5, 40.5, 42)  # one bucket per integer 0..40
    probs = np.full((1, 41), 1e-12)
    probs[0, 0], probs[0, 40] = 0.5, 0.5 + 1e-6
    cond = NumericConditional(np.log(probs), borders, mean=np.array([20.0]))
    classes = np.arange(41.0)
    np.testing.assert_array_equal(cond.class_mode(classes), [40.0])
    # buckets go to the nearest class: classes 0 and 40 only -> buckets 0..20 to 0, 21..40 to 40
    np.testing.assert_array_equal(cond.class_mode(np.array([0.0, 40.0])), [40.0])
    np.testing.assert_array_equal(nearest_class(np.array([19.0, 21.0]), np.array([0.0, 40.0])),
                                  [0.0, 40.0])  # fmt: skip


def test_impute_fills_column_by_column_like_impute_from():
    x = _masked(_table())
    model = _model(categorical_features=[2]).fit(x)
    np.testing.assert_array_equal(
        model.impute(x), model.impute_from(x, model.missing_conditionals(x), "mean")
    )
    np.testing.assert_array_equal(
        model.impute(x, "median", n_samples=2),
        model.impute_from(x, model.missing_conditionals(x), "median", n_samples=2),
    )


def test_regressed_categorical_column_gets_a_class(monkeypatch):
    # more classes than the classifier takes -> the regressor; its values are classes
    rng = np.random.default_rng(0)
    x = np.column_stack([rng.standard_normal(60), rng.integers(0, 4, 60).astype(float) * 2])
    x_missing = x.copy()
    x_missing[:10, 1] = np.nan
    monkeypatch.setattr("tabsil.inference.ensemble.max_classes_of", lambda model: 2)
    model = _model(categorical_features=[1]).fit(x_missing)
    assert not model._uses_classifier(1)
    out = model.impute(x_missing, n_samples=3)
    assert np.isin(out[..., 1], [0.0, 2.0, 4.0, 6.0]).all()
    point = model.impute(x_missing)
    assert np.isin(point[:, 1], [0.0, 2.0, 4.0, 6.0]).all()


def test_numeric_conditional_matches_the_regressor():
    # the conditional's distribution is predict_distribution's, its mean predict's (#19's
    # log densities and outlier scores are unchanged)
    x = _table()
    model = _model(categorical_features=[2]).fit(x[:40])
    cond = model.conditional(x[40:], 0, [1, 2])
    reg = TabSilRegressor(
        tiny_model(), BORDERS, n_estimators=2, random_state=0, categorical_features_indices=[1]
    ).fit(x[:40, 1:], x[:40, 0])
    log_probs, borders = reg.predict_distribution(x[40:, 1:])
    np.testing.assert_array_equal(cond.log_probs, log_probs.double().numpy())
    np.testing.assert_array_equal(cond.borders, borders.double().numpy())
    np.testing.assert_allclose(cond.mean, reg.predict(x[40:, 1:]), rtol=1e-6)


def _cached_v1():
    try:
        from huggingface_hub import try_to_load_from_cache

        from tabsil.recipe import HUB_REPO, HUB_REVISION

        path = try_to_load_from_cache(HUB_REPO, "model.safetensors", revision=HUB_REVISION)
    except Exception:  # noqa: BLE001 - no cache, no test
        return None
    return path if isinstance(path, str) else None


@pytest.mark.skipif(_cached_v1() is None, reason="v1 weights not in the local Hugging Face cache")
def test_recovers_obvious_patterns_with_v1_weights():
    rng = np.random.default_rng(0)
    n = 200
    a = rng.standard_normal(n)
    b = 2 * a + 1 + 0.05 * rng.standard_normal(n)
    c = np.where(a > 0, 3.0, 7.0)  # categorical: the sign of a
    x = np.column_stack([a, b, c])
    x_missing = x.copy()
    x_missing[rng.random(n) < 0.2, 1] = np.nan
    x_missing[rng.random(n) < 0.2, 2] = np.nan
    out = TabSilUnsupervised(n_estimators=1, device="cpu").fit(x_missing).impute(x_missing)
    miss_b, miss_c = np.isnan(x_missing[:, 1]), np.isnan(x_missing[:, 2])
    rmse = np.sqrt(np.mean((out[miss_b, 1] - b[miss_b]) ** 2))
    assert rmse < 0.2 * b.std()
    assert (out[miss_c, 2] == c[miss_c]).mean() >= 0.9
