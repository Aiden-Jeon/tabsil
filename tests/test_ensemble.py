"""Preprocessing member ensemble (`EnsembleClassifier`·`EnsembleRegressor`) and TabSil inference
class tests, run on a tiny randomized `ColumnRowTransformer`."""

import numpy as np
import pandas as pd
import pytest
import torch
from helpers import tiny_model
from sklearn.datasets import load_breast_cancer, load_diabetes

from tabsil.inference.ensemble import EnsembleClassifier, _probs_on_borders
from tabsil.inference.preprocessing import (
    Member,
    MemberPreprocessor,
    balanced_feature_subsets,
    feature_subset,
    select_features,
)
from tabsil.models import load_checkpoint, save_checkpoint
from tabsil.recipe import TabSilClassifier, TabSilRegressor


def _tiny_model():
    # one multitask model: classification with 3 label embeddings, regression with 999 quantiles
    return tiny_model()


def test_train_row_order_and_test_row_independence():
    model = _tiny_model()
    x = torch.randn(1, 30, 5)
    y = torch.randint(0, 3, (1, 20)).float()
    with torch.no_grad():
        base = model(x, y, "classification")
        perm = torch.randperm(20)
        shuffled = model(torch.cat([x[:, perm], x[:, 20:]], dim=1), y[:, perm], "classification")
        one_by_one = torch.cat(
            [
                model(torch.cat([x[:, :20], x[:, j : j + 1]], dim=1), y, "classification")
                for j in range(20, 30)
            ],
            dim=1,
        )
    assert torch.allclose(base, shuffled, atol=1e-5)
    assert torch.allclose(base, one_by_one, atol=1e-5)


def test_test_rows_do_not_change_when_other_test_rows_change():
    model = _tiny_model()
    x = torch.randn(1, 25, 4)
    y = torch.randint(0, 2, (1, 20)).float()
    other = x.clone()
    other[:, 21:] = torch.randn(1, 4, 4)  # only the first test row is unchanged
    with torch.no_grad():
        task = "classification"
        assert torch.allclose(model(x, y, task)[:, 0], model(other, y, task)[:, 0], atol=1e-5)


def test_missing_values_constant_columns_and_odd_feature_counts():
    model = _tiny_model()
    x = torch.randn(2, 25, 7)
    x[x > 1.5] = float("nan")
    x[:, :, 3] = 5.0  # constant column
    y = torch.randint(0, 2, (2, 15)).float()
    with torch.no_grad():
        out = model(x, y, "classification")
    assert out.shape == (2, 10, 3)
    assert torch.isfinite(out).all()


def test_column_empty_in_training_rows_is_left_missing():
    # A BeyondArena heart_disease_hungary split: a numeric column with no value in the training rows
    g = np.random.default_rng(0)
    x = g.normal(size=(40, 4))
    x[:30, 2] = np.nan  # empty in training rows (first 30), present in test rows
    y_class, y_value = (x[:, 0] > 0).astype(int), x[:, 0] + 0.1 * g.normal(size=40)
    clf = TabSilClassifier(_tiny_model()).fit(x[:30], y_class[:30])
    assert np.isfinite(clf.predict_proba(x[30:])).all()
    reg = TabSilRegressor(_tiny_model(), torch.linspace(-3, 3, 11))
    assert np.isfinite(reg.fit(x[:30], y_value[:30]).predict(x[30:])).all()


def test_checkpoint_roundtrip(tmp_path):
    model = _tiny_model()
    borders = torch.linspace(-3, 3, 9)
    save_checkpoint(tmp_path / "m.pt", model, borders)
    loaded, loaded_borders = load_checkpoint(tmp_path / "m.pt")
    x, y = torch.randn(1, 12, 3), torch.randn(1, 8)  # regression output: real-valued targets
    with torch.no_grad():
        assert torch.allclose(model(x, y, "regression"), loaded(x, y, "regression"))
    assert torch.equal(borders, loaded_borders)


def test_classifier_runs_and_handles_categorical_dataframe():
    x, y = load_breast_cancer(return_X_y=True)
    clf = TabSilClassifier(_tiny_model()).fit(x[:80], y[:80])
    proba = clf.predict_proba(x[80:100])
    assert proba.shape == (20, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)

    g = np.random.default_rng(0)
    df = pd.DataFrame({"a": g.normal(size=60), "color": g.choice(["red", "blue", None], 60)})
    labels = g.choice(["cat", "dog", "fish"], 60)
    clf = TabSilClassifier(_tiny_model()).fit(df[:40], labels[:40])
    assert set(clf.predict(df[40:])) <= {"cat", "dog", "fish"}


def test_regressor_runs():
    x, y = load_diabetes(return_X_y=True)
    borders = torch.linspace(-4, 4, 33)
    reg = TabSilRegressor(_tiny_model(), borders).fit(x[:80], y[:80])
    pred = reg.predict(x[80:100])
    assert pred.shape == (20,)
    assert np.isfinite(pred).all()


def test_regressor_tail_bucket_uses_training_target_range():
    from tabsil.inference.ensemble import _bucket_values

    # borders with a wide outermost bucket: the half-normal tail mean lands near z 30
    borders = torch.tensor([-30.0, -4.0, -1.0, 1.0, 4.0, 30.0])
    values = _bucket_values(borders, np.array([-1.0, 0.0, 6.0]))
    assert values[0] == -4.0  # inner border when training targets lie inside it
    assert values[-1] == 6.0  # training target max when it exceeds the inner border
    assert np.allclose(values[1:-1], [-2.5, 0.0, 2.5])
    assert _bucket_values(borders, np.array([0.0, 60.0]))[-1] == 30.0  # clipped inside outer border


def test_clipped_inverse_stays_in_training_range_on_both_sides():
    from sklearn.preprocessing import PowerTransformer

    from tabsil.inference.ensemble import _clipped_inverse

    rng = np.random.default_rng(0)
    # λ > 2: mass at the left edge + a right clump (the inverse ends on the left)
    # λ < 0: long right tail (the inverse ends on the right)
    for z in [
        np.concatenate([np.full(50, -4.5), rng.normal(0.5, 0.3, 500)]),
        rng.lognormal(0, 1.5, 2000),
    ]:
        z = (z - z.mean()) / z.std()
        transform = PowerTransformer(method="yeo-johnson", standardize=True)
        u = transform.fit_transform(z.reshape(-1, 1)).ravel()
        assert transform.lambdas_[0] > 2 or transform.lambdas_[0] < 0
        g = _clipped_inverse(transform, u)
        back = g(np.linspace(-30, 30, 601))
        assert np.isfinite(back).all() and (np.diff(back) >= 0).all()
        assert back.min() >= z.min() - 1e-6 and back.max() <= z.max() + 1e-6
        assert np.allclose(g(u), z, atol=1e-6)  # the exact inverse within the training range


def test_clipped_inverse_sets_a_non_finite_end_to_the_training_end():
    from tabsil.inference.ensemble import _clipped_inverse

    u_train = np.linspace(-1.0, 2.0, 7)
    z_train = 3 * u_train

    class EdgeNonFinite:  # inverse undefined at the training ends (Yeo-Johnson λ far from 1)
        def inverse_transform(self, u):
            return np.where(u >= 2.0, np.nan, np.where(u <= -1.0, -np.inf, 3.1 * u))

    g = _clipped_inverse(EdgeNonFinite(), u_train, z_train)
    # non-finite -> the training end on that side; finite values kept within [z_min, z_max]
    np.testing.assert_array_equal(g(np.array([-5.0, 0.0, 1.0, 1.99, 9.0])), [-3, 0, 3.1, 6, 6])
    assert np.isnan(_clipped_inverse(EdgeNonFinite(), u_train)(np.array([9.0]))).all()  # without z


def _power_member_end_is_nan(reg: TabSilRegressor, y: np.ndarray) -> bool:
    """Whether the plain Yeo-Johnson inverse of the power members is NaN at the training maximum
    (the case the tests below are about; it depends on the library versions)."""
    from sklearn.preprocessing import PowerTransformer

    power = [i for i, m in enumerate(reg.border_maps_) if m is not None]
    z = (y - reg.y_mean_) / reg.y_std_
    transform = PowerTransformer(method="yeo-johnson", standardize=True).fit(z.reshape(-1, 1))
    with np.errstate(all="ignore"):
        return bool(np.isnan(transform.inverse_transform([[max(reg.y_trains_[power[0]])]])).all())


def _check_finite_distribution(y: np.ndarray, seed: int) -> None:
    x = np.random.default_rng(seed).normal(size=(len(y), 2))
    # TabSil's borders end at about ±22.6, beyond the power members' training range
    borders = torch.cat(
        [torch.tensor([-22.6]), torch.linspace(-4.3, 4.2, 31), torch.tensor([22.6])]
    )
    reg = TabSilRegressor(_tiny_model(), borders, n_estimators=4).fit(x, y)
    assert _power_member_end_is_nan(reg, y)
    assert all(torch.isfinite(m).all() for m in reg.border_maps_ if m is not None)
    log_probs, out_borders = reg.predict_distribution(x[:10])
    assert torch.isfinite(log_probs).all() and torch.isfinite(out_borders).all()
    assert np.isfinite(reg.predict(x[:10])).all()


def test_zero_inflated_target_gives_a_finite_distribution():
    # 95% zeros and a few spikes (a redset_1H series): Yeo-Johnson λ ≈ -20, whose inverse at the
    # training maximum is NaN up to rounding. It made the power members' border maps, and so the
    # whole predict_distribution mixture, NaN
    rng = np.random.default_rng(4)
    y = np.zeros(1440)
    spikes = rng.choice(1440, 75, replace=False)
    y[spikes] = np.round(rng.lognormal(1.5, 1.0, 75))
    y[spikes[0]] = 143
    _check_finite_distribution(y, 4)


def test_right_tailed_target_near_zero_gives_a_finite_distribution():
    # most values just above 0 and a few far right (ADBench PageBlocks column 6): λ ≈ -25, the
    # same NaN at the training maximum
    rng = np.random.default_rng(1)
    y = np.concatenate([rng.exponential(0.0002, 1500), rng.uniform(0.05, 1, 5)])
    _check_finite_distribution(y, 1)


def test_regressor_bucket_values_stay_in_training_target_range():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(300, 3))
    y = rng.lognormal(0, 1.5, 300)  # right-skewed target: power member λ < 0
    inner = torch.linspace(-4.7, 4.4, 31)
    borders = torch.cat([torch.tensor([-29.0]), inner, torch.tensor([32.0])])
    reg = TabSilRegressor(_tiny_model(), borders, n_estimators=4).fit(x, y)
    z = (y - reg.y_mean_) / reg.y_std_
    lo, hi = min(float(borders[1]), z.min()), max(float(borders[-2]), z.max())
    for values in reg.bucket_values_:
        assert lo - 1e-5 <= float(values.min()) and float(values.max()) <= hi + 1e-5
    assert np.isfinite(reg.predict(x[:20])).all()
    log_probs, _ = reg.predict_distribution(x[:20])
    assert torch.allclose(log_probs.exp().sum(-1), torch.ones(20), atol=1e-4)


def test_probs_on_same_borders_are_unchanged():
    borders = torch.linspace(-2, 2, 11)
    probs = torch.softmax(torch.randn(3, 10), dim=-1)
    assert torch.allclose(_probs_on_borders(probs, borders, borders), probs, atol=1e-6)


def test_nearly_constant_training_column_does_not_blow_up():
    model = _tiny_model()
    x = torch.randn(1, 30, 3)
    x[:, :20, 1] = 1.0 + 1e-7 * torch.randn(20)  # nearly constant in training rows
    x[:, 20:, 1] = 50.0  # very different in validation rows
    y = torch.randint(0, 2, (1, 20)).float()
    out = model(x, y, "classification")
    out.sum().backward()
    assert torch.isfinite(out).all()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_outlier_clip_leaves_categorical_codes_alone():
    from tabsil.inference.preprocessing import Member, MemberPreprocessor

    x = np.zeros((1000, 2))
    x[:3, 0] = 50.0  # rare category code 50
    x[:, 1] = np.random.default_rng(0).normal(size=1000)
    member = Member(class_permutation=None, target_power=False, seed=0, one_hot=False)
    prep = MemberPreprocessor(member, cat_ix=[0], max_one_hot_width=160).fit(x)
    assert prep.upper[0] == np.inf and np.isfinite(prep.upper[1])
    codes = prep._base(prep._clip(x))[:, 0]
    assert np.array_equal(codes, x[:, 0])  # the rare code 50 is not clipped into another category


def test_one_hot_expands_small_categorical_columns_only():
    from tabsil.inference.preprocessing import Member, MemberPreprocessor

    rng = np.random.default_rng(0)
    x = np.stack(
        [
            rng.integers(0, 3, 200),
            rng.integers(0, 2, 200),
            rng.integers(0, 30, 200),
            rng.normal(size=200),
        ],
        axis=1,
    ).astype(float)
    x[0, 0] = np.nan
    member = Member(class_permutation=None, target_power=False, seed=0, one_hot=True)
    prep = MemberPreprocessor(member, cat_ix=[0, 1, 2], max_one_hot_width=160).fit(x)
    assert prep.one_hot_sizes == {0: 3, 1: 2}  # the 30-category column stays ordinal
    base = prep._base(prep._clip(x))
    assert base.shape[1] == 2 + 3 + 1  # 1 ordinal + 1 numeric, 3 categories -> 3 columns, 2 -> 1
    block = base[:, 2:5]
    assert np.isnan(block[0]).all()  # missing is NaN
    assert np.array_equal(block[1:].argmax(1), x[1:, 0]) and (block[1:].sum(1) == 1).all()
    assert np.array_equal(base[:, 5], x[:, 1])
    assert np.array_equal(base[:, 1], x[:, 3])  # standard members keep the numeric values
    out = prep.transform(x, np.random.default_rng(1).standard_normal((len(x), 1)))
    assert out.shape == (200, base.shape[1] + 1)  # sample identifier column


MAX_ONE_HOT_WIDTH = 160  # the fallback one-hot width cap (`inference.ensemble.MAX_FEATURES`)
MODEL_MAX_FEATURES = 100  # the default `max_features` of the model config (TabSil's stage 1)


def test_one_hot_width_is_capped_by_reverting_the_widest_columns():
    from tabsil.inference.preprocessing import one_hot_sizes

    # 8 columns of 20 categories + 10 binary + 2 numeric: expands to 2 + 160 + 10 = 172
    n_categories = {i: 20 for i in range(8)} | {i: 2 for i in range(8, 18)}
    sizes = one_hot_sizes(n_categories, n_features=20, max_width=MAX_ONE_HOT_WIDTH)
    assert sum(1 if n == 2 else n for n in sizes.values()) + 2 <= MAX_ONE_HOT_WIDTH
    assert len([n for n in sizes.values() if n == 20]) == 7 and 0 not in sizes  # only one reverts
    assert all(i in sizes for i in range(8, 18))  # binary columns stay one-hot
    # when the original columns already exceed the cap, no widening expansion happens
    assert one_hot_sizes({0: 5, 1: 2}, MAX_ONE_HOT_WIDTH + 1, MAX_ONE_HOT_WIDTH) == {1: 2}


def test_one_hot_out_of_range_codes_are_missing():
    from tabsil.inference.preprocessing import MemberPreprocessor

    block = MemberPreprocessor._one_hot(np.array([0.0, 2.0, 3.0, np.nan, -1.0]), 3)
    assert np.array_equal(block[:2], [[1, 0, 0], [0, 0, 1]])
    assert np.isnan(block[2:]).all()  # out-of-range and missing are missing, as in ordinal columns


def test_estimators_run_with_categorical_columns():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"a": rng.choice(list("xyz"), 80), "b": rng.normal(size=80)})
    y = (df["b"] > 0).astype(int).to_numpy()
    clf = TabSilClassifier(_tiny_model()).fit(df[:60], y[:60])
    assert clf.predict_proba(df[60:]).shape == (20, 2)
    reg = TabSilRegressor(_tiny_model(), torch.linspace(-4, 4, 33))
    assert reg.fit(df[:60], df["b"][:60]).predict(df[60:]).shape == (20,)


def test_predict_is_repeatable_and_regressor_keeps_borders_on_cpu():
    x, y = load_breast_cancer(return_X_y=True)
    clf = TabSilClassifier(_tiny_model()).fit(x[:60], y[:60])
    assert np.array_equal(clf.predict_proba(x[60:70]), clf.predict_proba(x[60:70]))
    reg = TabSilRegressor(_tiny_model(), torch.linspace(-4, 4, 33).double())
    assert reg.borders.device.type == "cpu" and reg.borders.dtype == torch.float32


def test_one_hot_width_follows_the_feature_count_the_model_was_trained_on():
    # a model trained on long tables (max_features 500) may expand past 100 columns
    from helpers import smoke_model

    model = smoke_model(max_features=500).eval()
    assert model.config.max_features == 500
    assert TabSilClassifier(model).max_one_hot_width == 500
    assert TabSilClassifier(_tiny_model()).max_one_hot_width == MODEL_MAX_FEATURES
    assert TabSilClassifier(model, max_one_hot_width=30).max_one_hot_width == 30


def test_max_one_hot_width_falls_back_to_the_default_without_config_max_features():
    # a model that only follows the registry contract must run without max_features in its config
    from types import SimpleNamespace

    from tabsil.inference.ensemble import max_features_of

    class NoMaxFeatures(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(n_out=10)

    model = NoMaxFeatures()
    assert max_features_of(_tiny_model()) == MODEL_MAX_FEATURES
    assert max_features_of(model) == MAX_ONE_HOT_WIDTH  # MAX_FEATURES(160)
    assert TabSilClassifier(model).max_one_hot_width == MAX_ONE_HOT_WIDTH


def test_balanced_feature_subsets_use_every_column_evenly():
    rng = np.random.default_rng(0)
    subsets = balanced_feature_subsets(20, 6, 8, rng)
    counts = np.bincount(np.concatenate(subsets), minlength=20)
    assert all(len(set(s)) == 6 and (np.diff(s) > 0).all() for s in subsets)
    assert counts.max() - counts.min() <= 1 and counts.sum() == 48
    assert balanced_feature_subsets(5, 6, 3, rng) == [None, None, None]
    x = rng.normal(size=(120, 20))
    y = (x[:, 0] > 0).astype(int)
    clf = TabSilClassifier(
        _tiny_model(), n_estimators=4, max_features_per_member=6, feature_subsampling="balanced"
    ).fit(x[:90], y[:90])
    assert all(len(s) == 6 for s in clf.feature_subsets_)
    assert clf.predict_proba(x[90:]).shape == (30, 2)


def test_tabsil_inference_defaults_come_from_the_model():
    # The weights decide the inference settings (`tabsil.recipe.default_inference`)
    from tabsil.recipe import default_inference

    X, y = load_breast_cancer(return_X_y=True)
    model = _tiny_model()
    expected = {
        "n_estimators": 8,
        "member_kinds": ("standard",),
        "max_features_per_member": 768,
        "feature_subsampling": "balanced",
    }
    # 1) without saved settings: TabSil's defaults, for both tasks
    assert default_inference(model, "classification") == expected
    assert default_inference(_tiny_model(), "regression") == expected
    clf = TabSilClassifier(model).fit(X[:60], y[:60])
    assert len(clf.preprocessors_) == 8 and clf.member_kinds == ("standard",)
    assert clf.max_features_per_member == 768 and clf.feature_subsampling == "balanced"
    with pytest.raises(ValueError):
        TabSilClassifier(_tiny_model(), member_kinds=("rank",))
    # 2) settings saved with the weights (the Hub config.json `inference`) come first
    model.inference_config = {
        "member_kinds": ["standard"],
        "n_estimators": 3,
        "max_features_per_member": None,
        "feature_subsampling": "random",
    }
    clf = TabSilClassifier(model)
    assert clf.member_kinds == ("standard",) and clf.n_estimators == 3
    assert clf.max_features_per_member is None and clf.feature_subsampling == "random"
    # 3) arguments come before both (experiments)
    clf = TabSilClassifier(model, member_kinds="tabsil-v1", n_estimators=2)
    assert clf.member_kinds == ("standard",) and clf.n_estimators == 2
    clf = TabSilClassifier(_tiny_model(), max_features_per_member=4)
    assert clf.max_features_per_member == 4


def test_tabsil_regressor_needs_borders_with_a_model():
    with pytest.raises(ValueError, match="borders"):
        TabSilRegressor(_tiny_model())


def test_member_kinds_parse_names_and_lists():
    from tabsil.recipe import resolve_member_kinds

    assert resolve_member_kinds("tabsil-v1") == ("standard",)
    assert resolve_member_kinds("standard") == ("standard",)
    assert resolve_member_kinds(["standard", "standard"]) == ("standard", "standard")
    assert resolve_member_kinds("numeric-cycle") == ("standard", "power", "robust", "quantile")
    assert resolve_member_kinds("standard,power") == ("standard", "power")
    for bad in ("", "v9", "rank", "standard,yeo"):
        with pytest.raises(ValueError, match="member_kinds"):
            resolve_member_kinds(bad)


def test_regressor_halves_the_target_power():
    X, y = load_diabetes(return_X_y=True)
    reg = TabSilRegressor(_tiny_model(), torch.linspace(-3, 3, 21), n_estimators=6)
    reg.fit(X[:60], y[:60])
    powers = [p.member.target_power for p in reg.preprocessors_]
    assert powers == [False, False, True, True, False, False]


def test_outlier_sigma_of_a_member_sets_the_clip():
    x = np.concatenate([np.zeros((99, 1)), [[100.0]]])

    def clipped(sigma):
        m = Member(None, False, seed=0, outlier_sigma=sigma)
        return MemberPreprocessor(m, [], 10).fit(x)._clip(x).max()

    assert clipped(12.0) == 100.0  # default: within 12 std (~119), so unchanged
    assert np.isclose(clipped(3.0), x.mean() + 3 * x.std())


def test_members_take_feature_subsets_above_the_cap():
    # None (all) within the cap; above it, cap columns chosen by member seed (sorted, unique)
    assert feature_subset(8, None, 0) is None and feature_subset(8, 8, 0) is None
    cols = feature_subset(30, 6, 5)
    assert len(cols) == 6 and np.array_equal(cols, np.unique(cols))
    assert np.array_equal(cols, feature_subset(30, 6, 5))
    assert not np.array_equal(cols, feature_subset(30, 6, 6))
    x, cat = select_features(np.arange(12.0).reshape(2, 6), [1, 4], np.array([0, 4, 5]))
    assert x.tolist() == [[0, 4, 5], [6, 10, 11]] and cat == [1]

    X, y = load_breast_cancer(return_X_y=True)
    X, y = X[:120, :8], y[:120]
    model = _tiny_model()
    full = EnsembleClassifier(model, n_estimators=3).fit(X[:90], y[:90])
    # a cap >= the feature count behaves as before
    same = EnsembleClassifier(model, n_estimators=3, max_features_per_member=8).fit(X[:90], y[:90])
    assert np.array_equal(same.predict_proba(X[90:]), full.predict_proba(X[90:]))
    sub = EnsembleClassifier(model, n_estimators=3, max_features_per_member=3).fit(X[:90], y[:90])
    proba = sub.predict_proba(X[90:])
    assert proba.shape == (30, 2) and np.allclose(proba.sum(1), 1)
    assert all(len(c) == 3 for c in sub.feature_subsets_)
    assert len({tuple(c) for c in sub.feature_subsets_}) > 1  # differs per member
    assert not np.allclose(proba, full.predict_proba(X[90:]))


def test_row_identifier_depends_on_the_row_and_seed_only():
    from tabsil.inference.preprocessing import _unit_interval, row_hashes, row_identifier

    def ident(x, seed):
        return row_identifier(row_hashes(x), seed)

    rng = np.random.default_rng(0)
    x = rng.normal(size=(50, 4))
    x[3, 1] = np.nan
    full = ident(x, 7)
    assert full.shape == (50, 1) and np.isfinite(full).all()
    # the same row gets the same value alone, in another order and next to a copy of itself
    for i in (0, 3, 49):
        assert ident(x[i : i + 1], 7)[0, 0] == full[i, 0]
    assert np.array_equal(ident(x[::-1], 7), full[::-1])
    assert np.array_equal(ident(np.vstack([x[:1], x]), 7)[1:], full)
    # other seeds and other values give other values; NaN payloads and -0.0 do not matter
    assert not np.isclose(ident(x, 8), full).any()
    other = x.copy()
    other[0, 0] += 1e-12
    assert ident(other, 7)[0, 0] != full[0, 0]
    signed = np.array([[0.0, np.nan]]), np.array([[-0.0, -np.nan]])
    assert ident(signed[0], 1)[0, 0] == ident(signed[1], 1)[0, 0]
    # the uniforms stay strictly inside (0, 1) at both ends of the hash range
    ends = _unit_interval(np.array([0, 2**64 - 1], dtype=np.uint64))
    assert 0 < ends[0] and ends[1] < 1
    # standard normal over many rows
    many = ident(rng.normal(size=(20000, 3)), 0)[:, 0]
    assert abs(many.mean()) < 0.03 and abs(many.std() - 1) < 0.03


def _inductive_tables():
    rng = np.random.default_rng(0)
    numeric = rng.normal(size=(80, 5))
    mixed = pd.DataFrame(
        {
            "a": pd.Categorical(rng.choice(list("xyz"), 80)),
            "b": rng.normal(size=80),
            "c": rng.choice(["p", "q", None], 80),
            "d": rng.normal(size=80),
        }
    )
    mixed.loc[rng.random(80) < 0.2, "b"] = np.nan
    y = rng.normal(size=80)
    return [(numeric, y), (mixed, y)]


def _predict_ways(predict, X_test):
    """Each test row's predictions: in the full test set, alone, reversed and next to a copy of
    row 0. TabArena accepts inductive predictions only (`audit_transduction`)."""
    X_test = pd.DataFrame(X_test).reset_index(drop=True)
    n = len(X_test)
    full = predict(X_test)
    alone = np.concatenate([predict(X_test.iloc[[i]]) for i in range(10)])
    reversed_ = predict(X_test.iloc[::-1])[::-1]
    twin = predict(pd.concat([X_test.iloc[[0]], X_test]))[1:]
    assert len(full) == len(reversed_) == len(twin) == n
    return full, {"alone": alone, "reversed": reversed_, "twin": twin}


@pytest.mark.parametrize("train_context", [True, False], ids=["context", "joint"])
@pytest.mark.parametrize("chunked", [False, True])
@pytest.mark.parametrize("table", [0, 1], ids=["numeric", "nan_and_categorical"])
def test_test_row_predictions_do_not_depend_on_the_other_test_rows(table, chunked, train_context):
    X, y = _inductive_tables()[table]
    X = pd.DataFrame(X)
    model = _tiny_model()
    if chunked:
        model.inference_chunk_cells = 64  # force the row-chunked forward on this small table
    kw = {"n_estimators": 3, "train_context": train_context}
    clf = TabSilClassifier(model, **kw).fit(X[:60], (y[:60] > 0).astype(int))
    reg = TabSilRegressor(model, torch.linspace(-4, 4, 33), **kw).fit(X[:60], y[:60])
    assert (clf.contexts_ is not None) == train_context
    # The model inputs of a test row are the same bits alone, in another order or next to a copy
    for est in (clf, reg):
        inputs = {
            name: est._test_inputs(rows)
            for name, rows in {
                "full": X[60:],
                "alone": X[60:61],
                "reversed": X[60:].iloc[::-1],
                "twin": pd.concat([X[60:61], X[60:]]),
            }.items()
        }
        for i in range(len(est.preprocessors_)):
            full, alone, rev, twin = (get(i) for get in inputs.values())
            for same in (alone[0], full[0]), (twin[1:], full), (rev[::-1], full):
                np.testing.assert_array_equal(*same)  # NaN cells (missing) compare equal
    # So predictions differ only by float32 rounding, which depends on the batch shape (up to
    # about 2e-6 on this tiny random model, more with the row-chunked forward); a
    # position-dependent identifier changed them by 1e-3 to 4e-2
    for predict in (clf.predict_proba, reg.predict):
        full, ways = _predict_ways(predict, X[60:])
        for name, pred in ways.items():
            np.testing.assert_allclose(pred, full[: len(pred)], atol=1e-5, rtol=0, err_msg=name)


def _context_tables():
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(90, 7)), columns=[f"x{i}" for i in range(7)])
    X["cat"] = pd.Categorical(rng.choice(list("pqrs"), 90))  # one-hot in members that pick it
    X.loc[rng.random(90) < 0.1, "x1"] = np.nan
    y = X["x0"].to_numpy() + 0.3 * rng.normal(size=90)
    return X, y


@pytest.mark.parametrize("chunked", [False, True])
@pytest.mark.parametrize(
    "setting", ["all_columns", "column_subsets", "one_per_forward", "many_classes"]
)
def test_train_context_predictions_equal_the_joint_forward(setting, chunked):
    # Fit-time train contexts (`train_context=True`, the default) give the predictions of running
    # the training and test rows together (`train_context=False`) within float32 rounding
    X, y = _context_tables()
    kw = {"n_estimators": 4}
    if setting == "column_subsets":  # members with and without the one-hot column: several runs
        kw.update(max_features_per_member=4, feature_subsampling="random")
    model = _tiny_model() if setting != "many_classes" else tiny_model(n_out=10)
    if chunked:
        model.inference_chunk_cells = 64
    labels = (y > 0).astype(int) if setting != "many_classes" else np.digitize(y, [-1, 0, 1]) * 4
    if setting == "many_classes":  # 13 classes over 10 target embeddings (grouped)
        labels = np.arange(90) % 13
    borders = torch.linspace(-4, 4, 33)
    estimators = {}
    for on in (True, False):
        clf = EnsembleClassifier(model, train_context=on, **kw).fit(X[:60], labels[:60])
        reg = TabSilRegressor(model, borders, train_context=on, **kw).fit(X[:60], y[:60])
        if setting == "one_per_forward":
            clf.predict_batch_bytes = reg.predict_batch_bytes = 1
        estimators[on] = clf, reg
    clf, reg = estimators[True]
    assert clf.contexts_ is not None and reg.contexts_ is not None
    assert estimators[False][0].contexts_ is None
    if setting == "column_subsets":
        assert len({p.order.size for p in clf.preprocessors_}) > 1 and len(clf.contexts_) > 1
    else:
        assert [(a, b) for a, b, _ in clf.contexts_] == [(0, 4)]  # one batch of 4 members
    (clf_joint, reg_joint), X_test = estimators[False], X[60:]
    np.testing.assert_allclose(
        clf.predict_proba(X_test), clf_joint.predict_proba(X_test), atol=1e-5, rtol=0
    )
    np.testing.assert_allclose(reg.predict(X_test), reg_joint.predict(X_test), atol=1e-5, rtol=0)
    log_p, b = reg.predict_distribution(X_test)
    log_p_joint, b_joint = reg_joint.predict_distribution(X_test)
    assert torch.equal(b, b_joint)
    torch.testing.assert_close(log_p.exp(), log_p_joint.exp(), atol=1e-5, rtol=0)


def test_train_context_falls_back_to_the_joint_forward_above_its_budget():
    X, y = _context_tables()
    clf = EnsembleClassifier(_tiny_model(), n_estimators=2)
    clf.train_context_device_bytes = 0  # no room: predict runs the training rows each time
    clf.fit(X[:60], (y[:60] > 0).astype(int))
    assert clf.contexts_ is None and clf.X_trains_ is not None
    full = EnsembleClassifier(_tiny_model(), n_estimators=2).fit(X[:60], (y[:60] > 0).astype(int))
    assert full.X_trains_ is None  # the contexts replace the transformed training rows
    np.testing.assert_allclose(
        clf.predict_proba(X[60:]), full.predict_proba(X[60:]), atol=1e-5, rtol=0
    )


def test_train_context_bytes_counts_the_context():
    model = _tiny_model()
    with torch.inference_mode():
        x, y = torch.randn(2, 30, 5), torch.randint(0, 3, (2, 30)).float()
        ctx = model.train_context(x, y, "classification")
    estimate = model.train_context_bytes(30, 5, classes=3, batch=2, attention_itemsize=4)
    assert estimate == ctx.nbytes
    with torch.inference_mode():
        reg = model.train_context(x, torch.randn(2, 30), "regression")
    assert model.train_context_bytes(30, 5, classes=None, batch=2, attention_itemsize=4) == (
        reg.nbytes
    )
    # no tensor keeps a larger storage alive (e.g. every head behind the first head's keys; with
    # one table the first head is a contiguous view of them)
    with torch.inference_mode():
        one = model.train_context(x[:1], y[:1], "classification")
    for t in ctx.tensors() + one.tensors():
        assert t.untyped_storage().nbytes() == t.numel() * t.element_size()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_train_context_predictions_equal_the_joint_forward_on_cuda():
    # Under bfloat16 autocast the training rows go through other matrix shapes with and without
    # the train context, so predictions differ by bf16 rounding (as when the test rows are
    # predicted in another batch)
    X, y = _context_tables()
    model = _tiny_model()
    labels = (y > 0).astype(int)
    kw = {"n_estimators": 4, "device": "cuda"}
    on = EnsembleClassifier(model, **kw).fit(X[:60], labels[:60])
    off = EnsembleClassifier(model, train_context=False, **kw).fit(X[:60], labels[:60])
    assert on.contexts_ is not None and on.contexts_[0][2].icl_kv[0][0].dtype == torch.bfloat16
    got, want = on.predict_proba(X[60:]), off.predict_proba(X[60:])
    print("cuda context vs joint max |diff|", np.abs(got - want).max())
    np.testing.assert_allclose(got, want, atol=5e-3, rtol=0)


def _option_table():
    rng = np.random.default_rng(2)
    X = np.column_stack(
        [
            rng.lognormal(size=300),  # skewed numeric
            rng.normal(size=300),
            rng.integers(0, 40, 300),  # 40 categories: stays codes
            rng.integers(0, 3, 300),  # 3 categories: one-hot
        ]
    ).astype(float)
    X[rng.random(X.shape) < 0.1] = np.nan
    return X


@pytest.mark.parametrize("kind", ["power", "robust", "quantile"])
def test_numeric_member_kinds_are_monotone_and_keep_missing_values(kind):
    X = _option_table()
    base = MemberPreprocessor(Member(None, False, seed=0), [2, 3], 100).fit(X)
    prep = MemberPreprocessor(Member(None, False, seed=0, kind=kind), [2, 3], 100).fit(X)
    want, got = base._features(base._clip(X)), prep._features(prep._clip(X))
    assert got.shape == want.shape
    assert np.array_equal(np.isnan(got), np.isnan(want))
    np.testing.assert_array_equal(got[:, 2:], want[:, 2:])  # code and one-hot columns unchanged
    for j in (0, 1):  # numeric columns: an increasing map of the original values
        ok = np.isfinite(want[:, j])
        order = np.argsort(want[ok, j], kind="stable")
        assert (np.diff(got[ok, j][order]) >= -1e-12).all()
        assert not np.allclose(got[ok, j], want[ok, j])
    if kind == "robust":
        assert np.nanmax(np.abs(got[:, :2])) < 3.0
    if kind == "quantile":  # rank-Gauss: about standard normal
        assert abs(np.nanmean(got[:, 0])) < 0.1 and abs(np.nanstd(got[:, 0]) - 1) < 0.15


def test_flip_sign_negates_about_half_of_the_numeric_columns():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 40))
    base = MemberPreprocessor(Member(None, False, seed=3), [], 100).fit(X)
    prep = MemberPreprocessor(Member(None, False, seed=3, flip_sign=True), [], 100).fit(X)
    got, want = prep._features(X), base._features(X)
    flipped = np.all(got == -want, axis=0)
    assert np.all(flipped | np.all(got == want, axis=0)) and 10 < flipped.sum() < 30


def test_many_category_codes_are_shuffled_or_counted():
    X = _option_table()
    base = MemberPreprocessor(Member(None, False, seed=1), [2, 3], 100).fit(X)
    want = base._features(X)
    shuffled = MemberPreprocessor(Member(None, False, seed=1, categories="shuffled"), [2, 3], 100)
    got = shuffled.fit(X)._features(X)
    codes, mapped = want[:, 2], got[:, 2]  # the 40-category code column (after the numerics)
    ok = np.isfinite(codes)
    assert np.array_equal(np.isnan(mapped), ~ok)
    pairs = set(zip(codes[ok], mapped[ok], strict=True))
    assert len(pairs) == len(set(codes[ok])) == len({m for _, m in pairs})  # a bijection
    assert any(a != b for a, b in pairs)
    counted = MemberPreprocessor(Member(None, False, seed=1, categories="counts"), [2, 3], 100)
    got = counted.fit(X)._features(X)
    assert got.shape[1] == want.shape[1] + 1
    np.testing.assert_array_equal(got[:, :-1], want)
    n = np.bincount(codes[ok].astype(int), minlength=40)
    np.testing.assert_allclose(got[ok, -1], np.log1p(n[codes[ok].astype(int)]))
    assert np.isnan(got[~ok, -1]).all()
    unseen = X[:1].copy()
    unseen[0, 2] = 45.0  # a code beyond the training categories is missing
    assert np.isnan(counted._features(unseen)[0, -1]) and np.isnan(shuffled._features(unseen)[0, 2])


def test_member_options_reach_the_members_and_predict():
    X, y = _context_tables()
    kw = {"n_estimators": 4, "member_kinds": "numeric-cycle", "flip_sign": True}
    clf = TabSilClassifier(_tiny_model(), categories="counts", **kw)
    clf.fit(X[:60], (y[:60] > 0).astype(int))
    members = [p.member for p in clf.preprocessors_]
    assert [m.kind for m in members] == ["standard", "power", "robust", "quantile"]
    assert all(m.flip_sign and m.categories == "counts" for m in members)
    assert np.allclose(clf.predict_proba(X[60:]).sum(1), 1)
    reg = TabSilRegressor(_tiny_model(), torch.linspace(-4, 4, 33), categories="shuffled", **kw)
    assert np.isfinite(reg.fit(X[:60], y[:60]).predict(X[60:])).all()
    with pytest.raises(ValueError, match="categories"):
        EnsembleClassifier(_tiny_model(), n_estimators=1, categories="target")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cudnn_attention_only_for_large_attention():
    # cuDNN attention builds a plan per new input shape (~0.2 s), so small forwards leave it out
    clf = EnsembleClassifier(_tiny_model(), n_estimators=1, device="cuda")
    with clf._forward_mode(clf.cudnn_attention_pairs - 1):
        assert not torch.backends.cuda.cudnn_sdp_enabled()
        assert torch.backends.cuda.flash_sdp_enabled()
    enabled = torch.backends.cuda.cudnn_sdp_enabled()
    with clf._forward_mode(clf.cudnn_attention_pairs):
        assert torch.backends.cuda.cudnn_sdp_enabled() == enabled
