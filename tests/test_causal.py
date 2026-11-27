"""Treatment effect estimation (`tabsil.tasks.causal`).

Shapes, determinism and the TabSil plumbing run on a tiny randomized model (it predicts nothing
useful). Whether a learner recovers known effects is checked with scikit-learn base learners
passed as `regressor` / `classifier`, which exercises the same meta-learner code."""

import numpy as np
import pytest
import torch
from helpers import tiny_model
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression

from tabsil.tasks.causal import LEARNERS, TabSilCATE

BORDERS = torch.linspace(-4, 4, 33)


def _data(n=400, d=4, effect="constant", seed=0, confounded=True):
    """Rows with a known CATE: outcome = x0 + 0.5 x1 + T tau(x) + noise; with `confounded` the
    treatment probability depends on x0 (overlap kept within [0.2, 0.8])."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    p = 0.2 + 0.6 / (1 + np.exp(-X[:, 0])) if confounded else np.full(n, 0.5)
    t = (rng.uniform(size=n) < p).astype(int)
    tau = np.full(n, 2.0) if effect == "constant" else 1.0 + 2.0 * X[:, 1]
    y = X[:, 0] + 0.5 * X[:, 1] + t * tau + 0.3 * rng.normal(size=n)
    return X, t, y, tau


def _tiny(**kwargs) -> TabSilCATE:
    return TabSilCATE(tiny_model(), BORDERS, n_estimators=2, **kwargs)


def _linear(**kwargs) -> TabSilCATE:
    return TabSilCATE(regressor=LinearRegression(), classifier=LogisticRegression(), **kwargs)


@pytest.mark.parametrize("learner", LEARNERS)
def test_tabsil_shapes_and_finite(learner):
    X, t, y, _ = _data(n=60)
    est = _tiny(learner=learner, n_folds=3).fit(X, t, y)
    tau = est.effect(X[:7])
    assert tau.shape == (7,) and np.isfinite(tau).all()
    assert est.effect().shape == (60,)  # the training rows
    assert est.ate() == pytest.approx(est.effect().mean())
    assert isinstance(est.ate(X[:7]), float)


@pytest.mark.parametrize("learner", LEARNERS)
def test_tabsil_same_seed_same_effect(learner):
    X, t, y, _ = _data(n=50, seed=1)
    a = _tiny(learner=learner, n_folds=2, random_state=4).fit(X, t, y).effect(X[:5])
    b = _tiny(learner=learner, n_folds=2, random_state=4).fit(X, t, y).effect(X[:5])
    np.testing.assert_array_equal(a, b)


def test_tabsil_base_models_get_different_seeds():
    X, t, y, _ = _data(n=50)
    est = _tiny(learner="x").fit(X, t, y)
    seeds = [est.mu0_.random_state, est.mu1_.random_state, est.tau0_.random_state,
             est.tau1_.random_state, est.propensity_.random_state]  # fmt: skip
    assert len(set(seeds)) == len(seeds)


def test_tabsil_s_learner_uses_the_treatment_column():
    X, t, y, _ = _data(n=50)
    est = _tiny(learner="s").fit(X, t, y)
    # the outcome model takes the features plus the treatment column, and flipping it matters
    assert est.mu_.predict(np.column_stack([X[:3], np.ones(3)])).shape == (3,)
    assert not np.allclose(est.effect(X[:5]), 0)


@pytest.mark.parametrize("learner", LEARNERS)
def test_constant_effect_is_recovered(learner):
    X, t, y, tau = _data(n=2000, effect="constant")
    est = _linear(learner=learner).fit(X, t, y)
    assert est.ate() == pytest.approx(2.0, abs=0.05)
    assert np.sqrt(np.mean((est.effect(X) - tau) ** 2)) < 0.1


@pytest.mark.parametrize("learner", ["t", "x", "dr"])
def test_heterogeneous_effect_is_recovered(learner):
    X, t, y, tau = _data(n=2000, effect="heterogeneous", seed=2)
    est = _linear(learner=learner).fit(X, t, y)
    X_new, _, _, tau_new = _data(n=500, effect="heterogeneous", seed=3)
    pehe = np.sqrt(np.mean((est.effect(X_new) - tau_new) ** 2))
    assert pehe < 0.1, pehe  # tau ranges over about +-4 around 1
    assert est.ate(X_new) == pytest.approx(tau_new.mean(), abs=0.05)


def test_heterogeneous_effect_with_a_nonlinear_base_learner():
    X, t, y, tau = _data(n=2000, effect="heterogeneous", seed=5)
    est = TabSilCATE(
        learner="x",
        regressor=GradientBoostingRegressor(random_state=0),
        classifier=LogisticRegression(),
    ).fit(X, t, y)
    X_new, _, _, tau_new = _data(n=500, effect="heterogeneous", seed=6)
    assert np.sqrt(np.mean((est.effect(X_new) - tau_new) ** 2)) < 0.5


def test_s_learner_with_a_linear_model_finds_only_a_constant_effect():
    # S-learner with an additive model: tau(x) = coefficient of T, the same for every row
    X, t, y, _ = _data(n=1000, effect="heterogeneous")
    tau = _linear(learner="s").fit(X, t, y).effect(X)
    assert np.ptp(tau) < 1e-8


def test_naive_difference_is_biased_but_learners_are_not():
    X, t, y, _ = _data(n=4000, effect="constant", seed=7)
    naive = y[t == 1].mean() - y[t == 0].mean()
    assert abs(naive - 2.0) > 0.2  # confounding through x0
    assert abs(_linear(learner="dr").fit(X, t, y).ate() - 2.0) < 0.05


def test_dr_pseudo_outcomes_average_to_the_aipw_estimate():
    X, t, y, _ = _data(n=1500, seed=8)
    est = _linear(learner="dr", n_folds=3).fit(X, t, y)
    assert est.pseudo_outcomes_.shape == (1500,)
    assert est.pseudo_outcomes_.mean() == pytest.approx(2.0, abs=0.06)
    # linear final model: the mean of its fitted values is the mean of the pseudo-outcomes
    assert est.ate() == pytest.approx(est.pseudo_outcomes_.mean())


def test_propensity_clip_bounds_the_weights():
    class Extreme(LogisticRegression):
        def predict_proba(self, X):
            p = np.where(X[:, 0] > 0, 1.0, 0.0)
            return np.column_stack([1 - p, p])

    X, t, y, _ = _data(n=300)
    est = TabSilCATE(learner="dr", regressor=LinearRegression(), classifier=Extreme(),
                     propensity_clip=0.1).fit(X, t, y)  # fmt: skip
    assert np.isfinite(est.pseudo_outcomes_).all()
    # residual weights are at most 1 / 0.1
    assert np.abs(est.pseudo_outcomes_).max() < 10 * np.abs(y).max() + 20


def test_constant_outcome_group_is_handled_without_the_model():
    X, t, y, _ = _data(n=60)
    y = np.where(t == 0, 3.0, y)  # every control outcome is 3
    est = _tiny(learner="t").fit(X, t, y)
    np.testing.assert_allclose(est.mu0_.predict(X[:4]), 3.0)
    assert np.isfinite(est.effect(X[:4])).all()


def test_boolean_treatment_and_lists():
    X, t, y, _ = _data(n=200)
    a = _linear(learner="t").fit(X, t.astype(bool), y).effect(X[:3])
    b = _linear(learner="t").fit(X.tolist(), list(t), list(y)).effect(X[:3])
    np.testing.assert_allclose(a, b)


def test_non_finite_prediction_raises():
    class Broken(LinearRegression):
        def predict(self, X):
            return np.full(len(X), np.nan)

    X, t, y, _ = _data(n=100)
    with pytest.raises(ValueError, match="non-finite"):
        TabSilCATE(learner="t", regressor=Broken()).fit(X, t, y).effect(X)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda X, t, y: (X, t * 2, y), "binary"),
        (lambda X, t, y: (X, np.where(t == 1, "a", "b"), y), "binary"),
        (lambda X, t, y: (X, t[:-1], y), "one value per row"),
        (lambda X, t, y: (X, t, y[:-1]), "one value per row"),
        (lambda X, t, y: (X, t, np.where(t == 1, np.nan, y)), "missing or infinite"),
        (lambda X, t, y: (X[:, 0], t, y), "2-dimensional"),
        (lambda X, t, y: (np.where(X > 1, np.inf, X), t, y), "infinite"),
        (lambda X, t, y: (X, np.zeros_like(t), y), "at least 2 rows"),
        (lambda X, t, y: (X.astype(str).astype(object) + "x", t, y), "numeric"),
    ],
)
def test_input_errors(change, message):
    X, t, y, _ = _data(n=50)
    with pytest.raises(ValueError, match=message):
        _linear(learner="t").fit(*change(X, t, y))


def test_dr_needs_n_folds_rows_per_group():
    X, t, y, _ = _data(n=50)
    t = np.zeros(50, dtype=int)
    t[:3] = 1
    with pytest.raises(ValueError, match="at least 5 rows"):
        _linear(learner="dr").fit(X, t, y)


def test_constructor_errors_and_unfitted():
    with pytest.raises(ValueError, match="learner"):
        _linear(learner="r")
    with pytest.raises(ValueError, match="n_folds"):
        _linear(n_folds=1)
    with pytest.raises(ValueError, match="propensity_clip"):
        _linear(propensity_clip=0.5)
    with pytest.raises(ValueError, match="borders"):
        TabSilCATE(tiny_model())
    with pytest.raises(ValueError, match="borders"):
        TabSilCATE("some/repo", BORDERS)
    with pytest.raises(RuntimeError, match="fit"):
        _linear().effect(np.zeros((2, 3)))
    X, t, y, _ = _data(n=50)
    with pytest.raises(ValueError, match="features"):
        _linear(learner="t").fit(X, t, y).effect(X[:, :2])


def test_no_weights_are_loaded_with_scikit_learn_base_learners():
    est = TabSilCATE(learner="t", regressor=LinearRegression())  # no classifier needed
    assert est.model is None


def test_s_learner_rejects_tables_beyond_the_member_column_cap():
    X, t, y, _ = _data(n=40, d=6)
    with pytest.raises(ValueError, match="columns per member"):
        _tiny(learner="s", max_features_per_member=6).fit(X, t, y)
    # the other learners do not add a column; the S-learner within the cap is fine
    _tiny(learner="t", max_features_per_member=6).fit(X, t, y)
    _tiny(learner="s", max_features_per_member=7).fit(X, t, y)


def test_x_and_dr_need_a_model_serving_classification():
    class RegressionOnly(torch.nn.Module):
        tasks = ("regression",)

    with pytest.raises(ValueError, match="classification"):
        TabSilCATE(RegressionOnly(), BORDERS, learner="x")
    TabSilCATE(RegressionOnly(), BORDERS, learner="t")  # no propensity model
    TabSilCATE(RegressionOnly(), BORDERS, learner="dr", classifier=LogisticRegression())
