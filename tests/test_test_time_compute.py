"""Spending more compute at inference: greedy member-pool selection and DiagScale. Both take
classification only (the eval script and the arena split groups: training/tests)."""

import copy

import numpy as np
import pytest
import torch
from helpers import tiny_model
from sklearn.datasets import load_breast_cancer

from tabsil.inference.diag_scale import DiagScale, add_diag_scale, fit_diag_scale
from tabsil.inference.ensemble import EnsembleClassifier, EnsembleRegressor
from tabsil.inference.ensemble_selection import (
    GreedyEnsembleClassifier,
    _log_loss,
    greedy_weights,
)
from tabsil.inference.preprocessing import Member, MemberPreprocessor

BORDERS = torch.linspace(-3, 3, 17)


def _tiny_model():
    return tiny_model()


def _classification(n=120):
    x, y = load_breast_cancer(return_X_y=True)
    return x[:n, :8], y[:n]


def test_greedy_weights_pick_with_replacement_and_break_ties_to_the_first():
    y = np.array([0.0, 1.0])
    preds = np.array([[0.0, 0.0], [0.0, 2.0], [0.0, 2.0]])  # the mean of 0 and 1 is the answer

    def rmse(p):
        return np.sqrt(np.mean((p - y) ** 2, axis=-1))

    w = greedy_weights(preds, rmse, n_rounds=4)
    assert np.allclose(w, [0.5, 0.5, 0.0])  # 1 and 2 are equal, so only the lower index
    assert greedy_weights(np.zeros((3, 2)), rmse, n_rounds=2)[0] == 1.0  # all equal -> member 0


def test_outlier_sigma_of_a_member_sets_the_clip():
    x = np.concatenate([np.zeros((99, 1)), [[100.0]]])

    def clipped(sigma):
        m = Member(None, False, seed=0, outlier_sigma=sigma)
        return MemberPreprocessor(m, [], 10).fit(x)._clip(x).max()

    assert clipped(12.0) == 100.0  # default: within 12 std (~119), so unchanged
    assert np.isclose(clipped(3.0), x.mean() + 3 * x.std())


def test_pool_starts_with_the_default_members():
    x, y = _classification()
    greedy = GreedyEnsembleClassifier(_tiny_model(), n_estimators=4, pool_size=7, n_folds=2)
    greedy.fit(x, y)
    default = EnsembleClassifier(_tiny_model(), n_estimators=4).fit(x, y)
    rng = np.random.default_rng(0)
    pool = greedy._pool(rng, greedy._default_members(rng, 2), 2)
    assert len(pool) == 7
    for a, b in zip(pool[:4], [p.member for p in default.preprocessors_], strict=True):
        assert (a.seed, a.one_hot) == (b.seed, b.one_hot)
        assert np.array_equal(a.class_permutation, b.class_permutation)


def test_pool_members_vary_one_hot_and_outlier_sigma():
    rng = np.random.default_rng(0)
    pool = GreedyEnsembleClassifier(_tiny_model(), n_estimators=2, pool_size=20)._pool(rng, [], 2)
    assert {m.one_hot for m in pool} == {True, False}
    assert len({m.outlier_sigma for m in pool}) > 1


def test_choosing_only_candidate_zero_is_the_default_ensemble(monkeypatch):
    x, y = _classification()
    monkeypatch.setattr(
        GreedyEnsembleClassifier, "_select", lambda self, oof, loss: np.eye(len(oof))[0]
    )
    greedy = GreedyEnsembleClassifier(_tiny_model(), n_estimators=4, pool_size=6, n_folds=2)
    default = EnsembleClassifier(_tiny_model(), n_estimators=4)
    out, ref = (e.fit(x[:90], y[:90]).predict_proba(x[90:]) for e in (greedy, default))
    assert np.allclose(out, ref, atol=1e-5)
    assert greedy.selection_["used_members"] == 4  # only the default members are refit


def test_greedy_classifier_outputs_and_out_of_fold_loss():
    x, y = _classification()
    clf = GreedyEnsembleClassifier(_tiny_model(), n_estimators=4, pool_size=8)
    proba = clf.fit(x[:90], y[:90]).predict_proba(x[90:])
    assert proba.shape == (30, 2) and np.allclose(proba.sum(axis=1), 1.0)
    s = clf.selection_
    assert s["n_folds"] == 8 and s["candidates"] == 1 + 8 * 9
    assert s["oof_loss_selected"] <= s["oof_loss_default"] + 1e-12  # member 0 is a candidate too
    assert s["used_members"] == len(clf.preprocessors_) <= 8
    total = sum(sum(w.values()) for w in clf.member_weights_)
    assert np.isclose(total, 1.0)


def test_greedy_classifier_handles_a_class_with_one_row():
    x, y = _classification(60)
    y = y.copy()
    y[0] = 2  # a class with one training row: some fold lacks it
    clf = GreedyEnsembleClassifier(_tiny_model(), n_estimators=2, pool_size=3).fit(x, y)
    assert clf.selection_["n_folds"] == 2
    assert np.allclose(clf.predict_proba(x[:5]).sum(axis=1), 1.0)


def test_selection_modes():
    x, y = _classification()
    kw = {"n_estimators": 2, "pool_size": 4, "n_folds": 3}
    best = GreedyEnsembleClassifier(_tiny_model(), selection="best", **kw).fit(x, y)
    assert len(best.member_weights_) == 1 or best.selection_["weight_default"] == 1.0
    # min_improvement 0: the uniform mix is kept even when it is not 1% better than the default
    uniform = GreedyEnsembleClassifier(
        _tiny_model(), selection="uniform", min_improvement=0.0, **kw
    ).fit(x, y)
    assert uniform.selection_["used_members"] == 4
    assert all(len(w) == 9 for w in uniform.member_weights_)
    with pytest.raises(ValueError):
        GreedyEnsembleClassifier(_tiny_model(), selection="mean", **kw)
    with pytest.raises(ValueError):
        GreedyEnsembleClassifier(_tiny_model(), n_estimators=4, pool_size=2)


def test_min_improvement_falls_back_to_the_default_ensemble(monkeypatch):
    x, y = _classification()
    kw = {"n_estimators": 4, "pool_size": 6, "n_folds": 2}
    # an unreachable ratio: always the default ensemble's prediction
    strict = GreedyEnsembleClassifier(_tiny_model(), min_improvement=0.999, **kw)
    out = strict.fit(x[:90], y[:90]).predict_proba(x[90:])
    ref = EnsembleClassifier(_tiny_model(), n_estimators=4).fit(x[:90], y[:90])
    assert np.allclose(out, ref.predict_proba(x[90:]), atol=1e-5)
    s = strict.selection_
    assert s["reverted"] and s["weight_default"] == 1.0 and s["used_members"] == 4
    assert s["oof_loss_selected"] == s["oof_loss_default"]
    # without the guard it does not fall back (same pool and folds)
    free = GreedyEnsembleClassifier(_tiny_model(), min_improvement=0.0, **kw).fit(x[:90], y[:90])
    assert not free.selection_["reverted"]


def test_guard_needs_an_absolute_drop_in_log_loss_when_the_default_is_near_zero():
    clf = GreedyEnsembleClassifier(_tiny_model(), n_estimators=2, min_improvement=0.01)
    # default (member 0) log loss 1e-4, chosen candidate 0: 100% relative but below 1e-3 nats
    oof = np.array([[np.exp(-1e-4)] * 4, [1.0] * 4])
    weights, reverted = clf._guard(np.array([0.0, 1.0]), oof, _log_loss)
    assert reverted and np.array_equal(weights, [1.0, 0.0])
    oof = np.array([[np.exp(-0.1)] * 4, [np.exp(-0.05)] * 4])  # 0.1 -> 0.05: passes both
    assert not clf._guard(np.array([0.0, 1.0]), oof, _log_loss)[1]


def test_min_temperature_drops_sharp_temperatures_from_the_pool():
    x, y = _classification()
    clf = GreedyEnsembleClassifier(
        _tiny_model(), n_estimators=2, pool_size=3, n_folds=2, min_temperature=0.5
    ).fit(x, y)
    assert 0.25 not in clf.temperatures and len(clf.temperatures) == 8
    assert clf.selection_["candidates"] == 1 + 3 * 8
    assert all(t >= 0.5 or t == clf.softmax_temperature for w in clf.member_weights_ for t in w)
    with pytest.raises(ValueError):
        GreedyEnsembleClassifier(_tiny_model(), n_estimators=2, min_temperature=10.0)
    with pytest.raises(ValueError):
        GreedyEnsembleClassifier(_tiny_model(), n_estimators=2, min_improvement=1.0)


def test_groups_keep_each_group_in_one_out_of_fold_fold(monkeypatch):
    x, y = _classification()
    groups = np.repeat(np.arange(12), 10)  # 12 groups of 10 rows
    cls = GreedyEnsembleClassifier
    seen = []
    original = cls._out_of_fold

    def capture(self, X, y, members, splits):
        seen.append(splits)
        return original(self, X, y, members, splits)

    monkeypatch.setattr(cls, "_out_of_fold", capture)
    args = (_tiny_model(),)
    est = cls(*args, n_estimators=2, pool_size=3, n_folds=4).fit(
        x, y, groups=[f"g{g}" for g in groups]
    )
    assert est.selection_["grouped"] and est.selection_["n_folds"] == 4
    for train_ix, test_ix in seen[0]:
        assert not set(groups[train_ix]) & set(groups[test_ix])
    # fewer groups than folds -> one fold per group; a single group -> random folds
    est = cls(*args, n_estimators=2, pool_size=3, n_folds=8).fit(x, y, groups=groups % 3)
    assert est.selection_["n_folds"] == 3
    est = cls(*args, n_estimators=2, pool_size=3, n_folds=4).fit(x, y, groups=np.zeros(len(y)))
    assert not est.selection_["grouped"]
    with pytest.raises(ValueError):
        cls(*args, n_estimators=2, pool_size=3).fit(x, y, groups=groups[:5])


def test_diag_scale_starts_as_the_frozen_model_and_only_scales_queries():
    model = _tiny_model()
    adapted = copy.deepcopy(model)
    deltas = add_diag_scale(adapted)
    # column block (summarize, broadcast) + 1 row block at E 16, 2 ICL layers at n_cls × E = 32
    assert [d.numel() for d in deltas] == [16, 16, 16, 32, 32]
    x, y = torch.randn(1, 30, 5), torch.randint(0, 2, (1, 20)).float()
    task = "classification"
    with torch.no_grad():
        assert torch.equal(model(x, y, task), adapted(x, y, task))
        deltas[1].fill_(0.5)
        assert not torch.allclose(model(x, y, task), adapted(x, y, task))
    loss = adapted(x, y, task).square().mean()
    loss.backward()
    assert deltas[0].grad is not None
    assert set(model.state_dict()) == set(adapted.state_dict()) - {
        "column_blocks.0.summarize.diag_scale",
        "column_blocks.0.broadcast.diag_scale",
        "row_blocks.0.attention.diag_scale",
        *(f"icl_layers.{i}.attention.diag_scale" for i in range(2)),
    }
    with pytest.raises(ValueError):
        add_diag_scale(adapted)


def test_diag_scale_without_epochs_keeps_the_frozen_predictions():
    x, y = _classification()
    model = _tiny_model()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    clf = DiagScale(EnsembleClassifier(model, n_estimators=2), epochs=0).fit(x[:90], y[:90])
    ref = EnsembleClassifier(model, n_estimators=2).fit(x[:90], y[:90])
    assert np.allclose(clf.predict_proba(x[90:]), ref.predict_proba(x[90:]), atol=1e-6)
    assert clf.adaptation_["best_epoch"] == -1 and clf.adaptation_["delta_norm"] == 0.0
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())  # original model


def test_diag_scale_trains_and_picks_an_epoch_on_validation():
    x, y = _classification(160)
    estimator = EnsembleClassifier(_tiny_model(), n_estimators=2)
    model, info = fit_diag_scale(estimator, x, y, epochs=4, patience=2)
    assert info["val_rows"] == 40 and info["n_params"] == 112  # 16 × 3 + 32 × 2
    assert len(info["val_curve"]) == info["epochs_run"] + 1 and info["train_loss"]
    assert info["val_loss_best"] == info["val_curve"][info["best_epoch"] + 1]
    if info["best_epoch"] == -1:
        assert info["delta_norm"] == 0.0
    else:
        assert info["val_loss_best"] < info["val_loss_before"]


def test_diag_scale_wraps_greedy_selection():
    x, y = _classification()
    greedy = GreedyEnsembleClassifier(_tiny_model(), n_estimators=2, pool_size=3, n_folds=2)
    clf = DiagScale(greedy, epochs=2).fit(x[:90], y[:90])
    assert np.allclose(clf.predict_proba(x[90:]).sum(axis=1), 1.0)
    assert list(clf.classes_) == [0, 1] and "used_members" in clf.selection_
    assert clf.estimator_.model is not greedy.model  # the trained copy
    with pytest.raises(AttributeError):
        DiagScale(greedy).classes_  # noqa: B018 (absent before fit)


def test_test_time_methods_take_classification_only():
    # the regression output is 999 quantiles, which only the basic EnsembleRegressor reads
    reg = EnsembleRegressor(_tiny_model(), BORDERS, n_estimators=2)
    with pytest.raises(ValueError, match="classification only"):
        DiagScale(reg)
    with pytest.raises(ValueError, match="classification only"):
        fit_diag_scale(reg, np.zeros((10, 2)), np.zeros(10))


def test_group_folds_never_leave_an_empty_fold_or_fail_on_small_classes(monkeypatch):
    from tabsil.inference import ensemble_selection

    x, y = _classification(120)
    groups = np.repeat(np.arange(8), 15)
    kw = {"n_estimators": 2, "pool_size": 3, "n_folds": 8}
    clf = GreedyEnsembleClassifier(_tiny_model(), **kw)
    # if stratification packs groups so that a fold has no validation rows, that fold is dropped
    real = ensemble_selection.StratifiedGroupKFold

    class WithEmptyFold(real):
        def split(self, X, y=None, groups=None):
            yield from super().split(X, y, groups)
            yield np.arange(len(X)), np.array([], dtype=int)

    monkeypatch.setattr(ensemble_selection, "StratifiedGroupKFold", WithEmptyFold)
    splits = clf._group_splits(y, groups)
    assert len(splits) == 8 and all(len(tr) and len(te) for tr, te in splits)
    assert clf.fit(x, y, groups=groups).selection_["n_folds"] == 8
    monkeypatch.setattr(ensemble_selection, "StratifiedGroupKFold", real)
    # one group per row, class counts [6, 5] < 8 folds: GroupKFold when stratified split fails
    x, y = x[:11], np.array([0] * 6 + [1] * 5)
    small = GreedyEnsembleClassifier(_tiny_model(), **kw).fit(x, y, groups=np.arange(11))
    assert small.selection_["grouped"] and small.selection_["n_folds"] == 8


def test_members_take_feature_subsets_above_the_cap():
    from tabsil.inference.preprocessing import feature_subset, select_features

    # None (all) within the cap; above it, cap columns chosen by member seed (sorted, unique)
    assert feature_subset(8, None, 0) is None and feature_subset(8, 8, 0) is None
    cols = feature_subset(30, 6, 5)
    assert len(cols) == 6 and np.array_equal(cols, np.unique(cols))
    assert np.array_equal(cols, feature_subset(30, 6, 5))
    assert not np.array_equal(cols, feature_subset(30, 6, 6))
    x, cat = select_features(np.arange(12.0).reshape(2, 6), [1, 4], np.array([0, 4, 5]))
    assert x.tolist() == [[0, 4, 5], [6, 10, 11]] and cat == [1]

    X, y = _classification()
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
    # greedy: a member uses the same columns in folds and the final fit (set by member seed)
    greedy = GreedyEnsembleClassifier(
        model, n_estimators=2, pool_size=5, n_folds=2, max_features_per_member=3
    ).fit(X[:90], y[:90])
    for prep, cols in zip(greedy.preprocessors_, greedy.feature_subsets_, strict=True):
        assert np.array_equal(cols, feature_subset(8, 3, prep.member.seed + 4))
    assert greedy.predict_proba(X[90:]).shape == (30, 2)
    # DiagScale's training episodes use the same cap
    clf = DiagScale(EnsembleClassifier(model, n_estimators=2, max_features_per_member=3), epochs=1)
    assert clf.fit(X[:90], y[:90]).predict_proba(X[90:]).shape == (30, 2)
