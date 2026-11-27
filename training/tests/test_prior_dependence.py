"""Tests of the graph SCM prior options that change column dependence and the target
(`CorrelateColumns`, `Relabel`, the TabICL `PriorConfig` corrections) and of the
`graph_scm_realism` prior that turns them on."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from tabsil_train.priors import graph_scm as P
from tabsil_train.priors import prior_module
from tabsil_train.priors.graph_scm.variants import HICARD_MISSING as V2
from tabsil_train.priors.graph_scm.variants import REALISM as R
from tabsil_train.priors.pipeline import Group, Rng
from tabsil_train.priors.transforms import (
    Coarsen,
    CorrelateColumns,
    HighCardinality,
    Relabel,
    StructuredMissing,
)
from tabsil_train.priors.transforms import dependence as D
from tabsil_train.training.config import TrainConfig

SIZES = dict(min_rows=60, max_rows=120, log_uniform_rows=True, max_cells=120 * 12, max_features=12)


def _same(a, b):
    return torch.equal(a.x.nan_to_num(-999.0), b.x.nan_to_num(-999.0)) and torch.equal(a.y, b.y)


def _config(prior_type="graph_scm"):
    return replace(TrainConfig.smoke(), prior_type=prior_type, filter_unpredictable=False)


# --- off by default ---------------------------------------------------------------------------


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_target_options_off_give_the_v1_prior(task):
    off = {"relabel_prob": 0.0, "corrected_num_converters": False, "corrected_cat_sizes": False}
    off |= {"correlate_prob": 0.0}
    # TabICL draws from the global random state, which building a prior reseeds: sample each
    # prior completely before building the next
    want = P.GraphSCMPrior(task, seed=4, **SIZES).sample_batch(2)
    prior = P.GraphSCMPrior(task, seed=4, **SIZES, **off)
    assert prior.dependence == () and prior.realism == ()
    assert _same(prior.sample_batch(2), want)
    config = prior.pipeline.generator._config
    assert not config.use_corrected_num_converters and not config.use_corrected_cat_sizes


def test_corrections_reach_the_tabicl_config_and_change_the_data():
    base = P.GraphSCMPrior("regression", seed=2, **SIZES).sample_batch(3)
    for flag in ("corrected_num_converters", "corrected_cat_sizes"):
        prior = P.GraphSCMPrior("regression", seed=2, **SIZES, **{flag: True})
        config = prior.pipeline.generator._config
        assert getattr(config, "use_" + flag) is True
        assert not _same(prior.sample_batch(3), base)


def test_corrected_cat_sizes_give_more_categories():
    # The share of few-valued columns with more than 9 values: TabICL stops at 9 without the fix
    def wide_share(corrected):
        prior = P.GraphSCMPrior(
            "classification",
            seed=0,
            min_rows=400,
            max_rows=400,
            max_cells=400 * 20,
            max_features=20,
            binary_prob=0.5,
            corrected_cat_sizes=corrected,
        )
        counts = []
        for _ in range(6):
            x = prior.sample_batch(4).x
            for d in range(x.shape[0]):
                for c in range(x.shape[2]):
                    k = len(torch.unique(x[d, :, c]))
                    if k <= 100:
                        counts.append(k)
        return np.mean(np.array(counts) > 9)

    assert wide_share(True) > wide_share(False) + 0.05


# --- correlated column blocks -----------------------------------------------------------------


def test_blend_discrete_keeps_values_and_counts_and_follows_the_latent():
    r = np.random.default_rng(0)
    column = r.integers(0, 5, size=2000).astype(float)
    latent = r.standard_normal(2000)
    out = D.blend_discrete(column, latent, 0.7, r)
    assert sorted(out) == sorted(column)
    # each level's rows sit in one band of the latent: the latent explains much of the level
    means = [latent[out == v].mean() for v in np.unique(out)]
    within = np.mean([latent[out == v].var() for v in np.unique(out)])
    assert np.var(means) > 0.2 and within < 0.8


def test_correlate_raises_correlation_within_blocks():
    r = np.random.default_rng(1)
    values = r.standard_normal((3000, 8))
    before = np.abs(np.corrcoef(values.T)[np.triu_indices(8, 1)]).max()
    after = []
    for seed in range(5):
        out = D.correlate(values, np.random.default_rng(seed))
        after.append(np.abs(np.corrcoef(out.T)[np.triu_indices(8, 1)]).max())
        changed = ~np.isclose(out, values).all(0)
        assert np.allclose(out[:, changed].mean(0), 0, atol=1e-8)
        assert np.allclose(out[:, changed].std(0), 1)
    assert before < 0.1 and min(after) > 0.15


def test_correlate_columns_leaves_other_datasets_and_missing_tables_alone():
    x = torch.randn(200, 2, 5, dtype=torch.float64)
    x[0, 1, 0] = float("nan")
    group = Group(x, torch.zeros(200, 2), 150, {})
    out = CorrelateColumns(1.0)(group, Rng.from_seed(0))
    assert not torch.equal(out.x[:, 0], x[:, 0])
    assert torch.equal(out.x[:, 1].nan_to_num(9.0), x[:, 1].nan_to_num(9.0))
    assert torch.equal(
        CorrelateColumns(0.0)(group, Rng.from_seed(0)).x.nan_to_num(9.0), x.nan_to_num(9.0)
    )


# --- relabeled targets ------------------------------------------------------------------------


def test_score_classes_cuts_into_every_class():
    r = np.random.default_rng(3)
    for n_classes in (2, 5, 10):
        for _ in range(20):
            score = r.standard_normal(300)
            labels = D.score_classes(score, n_classes, r)
            assert set(labels.tolist()) == set(range(n_classes))
    # ties: a score with few distinct values keeps whole runs in one class
    score = np.repeat(np.arange(6.0), 50)
    labels = D.score_classes(score, 3, r)
    for v in range(6):
        assert len(set(labels[score == v].tolist())) <= 2  # the jittered cut can split a run
    assert D.score_classes(np.repeat([0.0, 1.0], 100), 3, r) is None


def test_relabel_score_gives_equal_scores_to_equal_rows():
    r = np.random.default_rng(4)
    inputs = np.repeat(r.standard_normal((40, 3)), 5, axis=0)
    for _ in range(20):
        score = D.relabel_score(inputs, r)
        assert np.allclose(score.reshape(40, 5), score.reshape(40, 5)[:, :1])


def test_regression_target_matches_tabicl():
    from tabicl.prior._reg2cls import outlier_removing, standard_scaling

    r = np.random.default_rng(5)
    score = np.concatenate([r.standard_normal(500), [40.0, -25.0]])
    t = torch.from_numpy(score)[:, None]
    want = standard_scaling(outlier_removing(t, threshold=4.0))[:, 0].numpy()
    np.testing.assert_allclose(D.regression_target(score), want, atol=1e-10)


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_relabel_in_the_prior(task):
    prior = P.GraphSCMPrior(task, seed=6, binary_prob=0.5, **SIZES, relabel_prob=1.0)
    assert [type(t) for t in prior.dependence] == [Relabel]
    for _ in range(4):
        batch = prior.sample_batch(3)
        for d in range(3):
            y = batch.y[d]
            if task == "classification":
                train, test = set(y[: batch.n_train].tolist()), set(y[batch.n_train :].tolist())
                assert test <= train  # CoverClasses runs after Relabel
                assert min(train) >= 0 and max(train) < P.MAX_CLASSES
            else:
                assert abs(float(y.mean())) < 1e-4 and abs(float(y.std()) - 1) < 1e-3


# --- the assembled prior ----------------------------------------------------------------------


def test_realism_prior_is_registered_with_its_options():
    config = _config("graph_scm_realism")
    assert prior_module(config.prior_type) is R and config.n_out == P.MAX_CLASSES
    assert {k: R.OPTIONS[k] for k in V2.OPTIONS} == V2.OPTIONS
    for task in ("classification", "regression"):
        prior = config.make_prior(task, seed=0)
        assert [type(t) for t in prior.dependence] == [CorrelateColumns, Relabel]
        assert [type(t) for t in prior.realism] == [Coarsen, HighCardinality, StructuredMissing]
        generator = prior.pipeline.generator._config
        assert generator.use_corrected_num_converters and generator.use_corrected_cat_sizes
        assert prior.sample_batch(4).x.shape[:2] == (4, 64)
    # the options change y, so the borders are drawn with them
    small = replace(config, border_datasets=4)
    v1 = replace(small, prior_type="graph_scm")
    borders = R.make_borders(small, 20, seed=1)
    assert borders.shape == (21,) and (borders.diff() > 0).all()
    assert not torch.equal(borders, P.make_borders(v1, 20, seed=1))
    with pytest.raises(TypeError, match="unknown options"):
        P.make_borders(v1, 20, seed=1, coarsen_prob=0.1)


def test_realism_prior_is_reproducible_from_the_seed():
    a = P.GraphSCMPrior("classification", seed=3, **SIZES, **R.OPTIONS).sample_batch(3)
    b = P.GraphSCMPrior("classification", seed=3, **SIZES, **R.OPTIONS).sample_batch(3)
    assert _same(a, b)
