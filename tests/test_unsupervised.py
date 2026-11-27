"""TabSilUnsupervised: conditional distributions, chain-rule log density, outlier score (tiny
randomized model on the CPU)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from helpers import tiny_model

from tabsil.tasks.unsupervised import (
    PROBABILITY_FLOOR,
    CategoricalConditional,
    NumericConditional,
    TabSilUnsupervised,
    column_orders,
    infer_categorical,
)

BORDERS = torch.linspace(-4, 4, 33)


def _model(**kwargs) -> TabSilUnsupervised:
    return TabSilUnsupervised(tiny_model(), BORDERS, n_estimators=2, **kwargs)


def _table(n: int = 60, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.standard_normal(n)
    b = a + 0.1 * rng.standard_normal(n)
    c = rng.integers(0, 3, n).astype(float) * 0.5  # categorical: 3 values, 20 rows each
    return np.column_stack([a, b, c])


def test_infer_categorical_rule():
    x = _table()
    assert infer_categorical(x) == [2]
    # 3 values but only 20 rows: 20 / 3 <= 10 rows per value -> numeric
    assert infer_categorical(x[:20]) == []


def test_column_orders_distinct_and_all_when_few():
    orders = column_orders(4, 10, 0)
    assert len(orders) == len(set(orders)) == 10
    assert all(sorted(o) == [0, 1, 2, 3] for o in orders)
    assert column_orders(4, 10, 0) == orders
    assert len(column_orders(3, 10, 0)) == 6  # 3! <= 10: every order


def test_shapes_and_determinism():
    x = _table()
    model = _model(n_permutations=3).fit(x[:40])
    assert model.categorical_ == [2]
    log_p = model.log_density(x[40:])
    assert log_p.shape == (20,) and np.isfinite(log_p).all()
    np.testing.assert_array_equal(model.outlier_score(x[40:]), -log_p)
    again = _model(n_permutations=3).fit(x[:40]).log_density(x[40:])
    np.testing.assert_array_equal(again, log_p)


def test_log_density_is_log_mean_over_orders_of_chain_rule():
    x = _table()
    model = _model(n_permutations=2).fit(x[:40])
    per_order, orders = model.log_density_per_order(x[40:])
    assert len(orders) == 2
    for k, order in enumerate(orders):
        chain = sum(
            model.conditional(x[40:], j, order[:i]).log_prob(x[40:, j]) for i, j in enumerate(order)
        )
        np.testing.assert_allclose(per_order[k], chain)
    expected = np.log(np.exp(per_order).mean(0))
    np.testing.assert_allclose(model.log_density(x[40:]), expected)


def test_conditional_kinds():
    x = _table()
    model = _model().fit(x[:40])
    cat = model.conditional(x[40:], 2, [0, 1])
    assert isinstance(cat, CategoricalConditional)
    np.testing.assert_array_equal(cat.classes, [0.0, 0.5, 1.0])
    np.testing.assert_allclose(cat.proba.sum(1), 1.0, atol=1e-5)
    num = model.conditional(x[40:], 0, [2])
    assert isinstance(num, NumericConditional)
    assert num.log_probs.shape == (20, len(BORDERS) - 1)
    # the borders are in the column's original units: z borders x std + mean of the context
    y = x[:40, 0]
    np.testing.assert_allclose(
        num.borders, BORDERS.double().numpy() * y.std() + y.mean(), atol=1e-5
    )
    # order of the conditioning columns does not matter (evaluated in index order)
    np.testing.assert_array_equal(
        model.conditional(x[40:], 0, [2, 1]).log_prob(x[40:, 0]),
        model.conditional(x[40:], 0, [1, 2]).log_prob(x[40:, 0]),
    )
    with pytest.raises(ValueError, match="among the given"):
        model.conditional(x[40:], 0, [0])


def test_numeric_density_is_bucket_probability_over_width():
    borders = np.array([0.0, 1.0, 1.0, 3.0])  # a zero-width bucket at 1
    log_probs = np.log(np.array([[0.5, 0.0 + 1e-30, 0.5]] * 5))
    cond = NumericConditional(log_probs, borders)
    out = cond.log_prob(np.array([0.5, 1.0, 2.0, 3.0, 4.0]))
    np.testing.assert_allclose(out[:4], np.log([0.5, 0.25, 0.25, 0.25]))
    assert out[4] == pytest.approx(np.log(PROBABILITY_FLOOR / 2.0))  # beyond the borders
    assert cond.log_prob(np.array([np.nan] * 5))[0] == 0.0  # missing: integrated out


def test_categorical_unseen_value_gets_floor():
    cond = CategoricalConditional(np.array([[0.2, 0.8]] * 3), np.array([0.0, 1.0]))
    out = cond.log_prob(np.array([1.0, 0.5, np.nan]))
    np.testing.assert_allclose(out, [np.log(0.8), np.log(PROBABILITY_FLOOR), 0.0])


def test_obvious_outliers_score_highest():
    rng = np.random.default_rng(1)
    n = 80
    a = rng.standard_normal(n)
    inliers = np.column_stack([a, a + 0.1 * rng.standard_normal(n)])
    model = _model(n_permutations=2).fit(inliers[:60])
    test = np.vstack([inliers[60:], [[8.0, 8.0], [-8.0, 8.0]]])
    scores = model.outlier_score(test)
    assert scores[-2:].min() > scores[:-2].max()


def test_unseen_category_is_an_outlier():
    x = _table(n=80)
    model = _model(n_permutations=2).fit(x[:60])
    test = x[60:].copy()
    test[0, 2] = 0.25  # not a value of the categorical column
    scores = model.outlier_score(test)
    assert scores.argmax() == 0


def test_missing_values():
    x = _table()
    x_fit = x[:40].copy()
    x_fit[[0, 5], 1] = np.nan
    model = _model(n_permutations=2).fit(x_fit)
    test = x[40:].copy()
    test[0, 0] = np.nan
    log_p = model.log_density(test)
    assert np.isfinite(log_p).all()


def test_constant_column_and_checks():
    x = _table()
    x[:, 2] = 1.0
    model = _model(n_permutations=2).fit(x[:40])
    cond = model.conditional(x[40:], 2, [0])
    np.testing.assert_array_equal(cond.log_prob(x[40:, 2]), 0.0)
    with pytest.raises(ValueError, match="columns"):
        model.log_density(x[40:, :2])
    with pytest.raises(RuntimeError, match="fit"):
        _model().log_density(x)
    with pytest.raises(ValueError, match="borders"):
        TabSilUnsupervised(tiny_model())
