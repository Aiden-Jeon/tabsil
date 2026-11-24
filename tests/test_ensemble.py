"""Preprocessing member ensemble (`EnsembleClassifier`·`EnsembleRegressor`) and TabSil inference
class tests, run on a tiny randomized `ColumnRowTransformer`."""

import numpy as np
import pandas as pd
import pytest
import torch
from helpers import tiny_model
from sklearn.datasets import load_breast_cancer, load_diabetes

from tabsil.inference.ensemble import _probs_on_borders
from tabsil.inference.preprocessing import balanced_feature_subsets
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
    out = prep.transform(x, np.random.default_rng(1))
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
    from dataclasses import replace

    from helpers import tabsil_smoke

    wide = replace(tabsil_smoke(), max_features=500, max_cells=64 * 500)
    assert wide.model_config().max_features == 500
    model = wide.build_model().eval()
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
        TabSilClassifier(_tiny_model(), member_kinds=("quantile",))
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
    for bad in ("", "v9", "quantile", "standard,power"):
        with pytest.raises(ValueError, match="member_kinds"):
            resolve_member_kinds(bad)


def test_regressor_halves_the_target_power():
    X, y = load_diabetes(return_X_y=True)
    reg = TabSilRegressor(_tiny_model(), torch.linspace(-3, 3, 21), n_estimators=6)
    reg.fit(X[:60], y[:60])
    powers = [p.member.target_power for p in reg.preprocessors_]
    assert powers == [False, False, True, True, False, False]
