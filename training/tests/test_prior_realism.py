"""Tests of the graph SCM prior options that make tables more like real ones (high-cardinality
categorical columns, structured missing values, coarse and noisy values) and of the
`graph_scm_hicard_missing` prior that turns some of them on."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from tabsil_train.priors import graph_scm as P
from tabsil_train.priors import graph_scm_hicard_missing as V2
from tabsil_train.priors import prior_module
from tabsil_train.priors.graph_scm.generator import GraphSCM
from tabsil_train.priors.hyperparameters import ClassCount
from tabsil_train.priors.pipeline import Group, Pipeline, Rng
from tabsil_train.priors.transforms import (
    Coarsen,
    CoverClasses,
    DropInvalid,
    HighCardinality,
    MeasurementNoise,
    Missing,
    StructuredMissing,
)
from tabsil_train.priors.transforms import categorical as C
from tabsil_train.priors.transforms import missing as M
from tabsil_train.training.config import TrainConfig

SIZES = dict(min_rows=60, max_rows=120, log_uniform_rows=True, max_cells=120 * 12, max_features=12)
ALL_OFF = {name: 0.0 for name in P.REALISM_FIELDS}


def _same(a, b):
    return torch.equal(a.x.nan_to_num(-999.0), b.x.nan_to_num(-999.0)) and torch.equal(a.y, b.y)


def _gaussian_group(n_rows=400, n_train=300, n_features=6, n_datasets=1, seed=0, classes=None):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n_rows, n_datasets, n_features, generator=g)
    y = x.sum(-1)
    if classes:
        y = torch.bucketize(y, torch.linspace(-2, 2, classes - 1)).float()
    return Group(x, y, n_train, {})


# --- the v1 settings are unchanged ------------------------------------------------------------


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_options_off_give_the_v1_prior(task):
    # The v1 prior assembled from its parts, as before the options existed
    def v1_batches(seed):
        rng = Rng.from_seed(seed)
        generator = GraphSCM(task, seed=seed)
        clf = task == "classification"
        pipe = Pipeline(
            generator,
            rng=rng,
            hyperparameters=ClassCount(P.MAX_CLASSES, 0.5) if clf else None,
            transforms=[DropInvalid(), *([CoverClasses()] if clf else []), Missing(0.3, 0.3)],
            max_attempts=P.MAX_ATTEMPTS,
        )
        out = []
        for _ in range(3):
            n_train, n_eval, n_features = P.sample_split_size(
                rng.numpy,
                **SIZES,
                min_train_fraction=0.3,
                max_train_fraction=0.9,
                min_train_rows=P.MIN_TRAIN_ROWS[task],
            )
            out.append(pipe.sample_batch(2, n_train + n_eval, n_train, n_features))
        return out

    # TabICL draws from the global random state, which building a prior reseeds: sample each
    # prior completely before building the next
    for options in ({}, ALL_OFF):
        expected = v1_batches(5)
        prior = P.GraphSCMPrior(task, seed=5, binary_prob=0.5, **SIZES, **options)
        assert prior.realism == ()
        for want in expected:
            assert _same(prior.sample_batch(2), want)


def test_make_prior_rejects_unknown_options_and_sets_the_given_ones():
    config = replace(TrainConfig.smoke(), filter_unpredictable=False)
    with pytest.raises(TypeError, match="unknown options"):
        P.make_prior(config, "regression", seed=0, missing_rate=0.5)
    assert P.make_prior(config, "regression", seed=0).realism == ()
    prior = P.make_prior(config, "regression", seed=0, coarsen_prob=0.5)
    assert [type(t) for t in prior.realism] == [Coarsen]


# --- the assembled prior ----------------------------------------------------------------------


def test_hicard_missing_prior_is_registered_with_its_options():
    config = replace(TrainConfig.smoke(), prior_type="graph_scm_hicard_missing")
    config = replace(config, filter_unpredictable=False)
    assert prior_module(config.prior_type) is V2 and config.n_out == P.MAX_CLASSES
    for task in ("classification", "regression"):
        prior = config.make_prior(task, seed=0)
        kinds = [type(t) for t in prior.realism]
        assert kinds == [HighCardinality, StructuredMissing]
        assert prior.realism[0].dataset_prob == V2.OPTIONS["high_cardinality_prob"]
        missing = prior.realism[1]
        assert (missing.structured_prob, missing.near_empty_prob, missing.latent_prob) == (
            0.2,
            0.05,
            0.03,
        )
        assert missing.classification is (task == "classification")
        batch = prior.sample_batch(4)
        assert batch.x.shape[:2] == (4, 64)
    # the options change only x, so the regression borders are the graph SCM prior's
    small = replace(config, border_datasets=4)
    v1 = replace(small, prior_type="graph_scm")
    assert torch.equal(V2.make_borders(small, 20, seed=1), P.make_borders(v1, 20, seed=1))


def test_options_are_reproducible_from_the_seed():
    options = dict(V2.OPTIONS, high_cardinality_prob=1.0, structured_missing_prob=1.0)
    options |= dict(coarsen_prob=0.5, noise_prob=0.5)
    a = P.GraphSCMPrior("classification", seed=3, **SIZES, **options).sample_batch(3)
    b = P.GraphSCMPrior("classification", seed=3, **SIZES, **options).sample_batch(3)
    assert _same(a, b)
    assert torch.isnan(a.x).any()


# --- high-cardinality categorical columns -----------------------------------------------------


def test_one_hot_limit_matches_inference():
    from tabsil.inference.preprocessing import MAX_ONE_HOT_CATEGORIES

    assert C.MAX_ONE_HOT_CATEGORIES == MAX_ONE_HOT_CATEGORIES


def test_level_codes_cut_by_value_and_refine_few_distinct_values():
    r = np.random.default_rng(0)
    values = r.standard_normal(500)
    codes = C.level_codes(values, 40, r)
    assert len(np.unique(codes)) == 40  # every category gets a row
    # a category is a range of values (the codes are in random order, the ranges do not overlap)
    ranges = sorted((values[codes == c].min(), values[codes == c].max()) for c in range(40))
    assert all(hi < lo for (_, hi), (lo, _) in zip(ranges, ranges[1:], strict=False))
    # more categories than distinct values: every category has one value (it refines them)
    few = r.integers(0, 5, 500).astype(float)
    codes = C.level_codes(few, 60, r)
    assert all(len(np.unique(few[codes == c])) == 1 for c in np.unique(codes))


def test_train_row_codes_number_train_categories_and_hide_unseen_ones():
    codes = np.array([7, 3, 7, 9, 3, 5, 11, 9])
    out = C.train_row_codes(codes, n_train=4)
    # train categories 3, 7, 9 -> 0, 1, 2; 5 and 11 appear only in eval rows -> NaN
    np.testing.assert_array_equal(out[:4], [1, 0, 1, 2])
    assert out[4] == 0 and np.isnan(out[5]) and np.isnan(out[6]) and out[7] == 2


def test_high_cardinality_columns_look_like_inference_codes():
    n_rows, n_train = 1000, 700
    levels, correlations = [], []
    transform = HighCardinality(1.0)
    rng = Rng.from_seed(0)
    for seed in range(60):
        group = _gaussian_group(n_rows, n_train, n_features=8, seed=seed)
        out = transform(group, rng)
        x0, x1 = group.x[:, 0].numpy(), out.x[:, 0].numpy()
        changed = np.flatnonzero(~np.isclose(x0, x1, equal_nan=True).all(0))
        assert 1 <= len(changed) <= 3
        assert torch.equal(out.y, group.y)
        for c in changed:
            train, rest = x1[:n_train, c], x1[n_train:, c]
            k = len(np.unique(train))
            # train rows: codes 0..K-1, no missing values; eval rows: a known code or NaN
            np.testing.assert_array_equal(np.unique(train), np.arange(k))
            known = rest[np.isfinite(rest)]
            assert set(np.unique(known)) <= set(range(k))
            levels.append(k)
            correlations.append(np.corrcoef(train, x0[:n_train, c])[0, 1])
    levels = np.array(levels)
    # categories spread log-uniformly from just above the one-hot limit to the row count
    assert (levels > C.MAX_ONE_HOT_CATEGORIES).mean() > 0.8
    assert levels.min() < 60 and levels.max() > 300
    assert 0.1 < (levels < 100).mean() < 0.7
    # the code order is unrelated to the source values
    assert abs(np.mean(correlations)) < 0.1


def test_high_cardinality_leaves_short_tables_and_other_datasets_alone():
    rng = Rng.from_seed(0)
    short = _gaussian_group(n_rows=15, n_train=10)
    assert torch.equal(HighCardinality(1.0)(short, rng).x, short.x)
    group = _gaussian_group(n_datasets=3)
    assert torch.equal(HighCardinality(0.0)(group, rng).x, group.x)


# --- structured missing values ----------------------------------------------------------------


def test_structured_missing_with_zero_probabilities_draws_nothing():
    group = _gaussian_group(n_datasets=2)
    rng = Rng.from_seed(0)
    state = rng.numpy.bit_generator.state
    out = StructuredMissing(classification=False)(group, rng)
    assert torch.equal(out.x, group.x) and rng.numpy.bit_generator.state == state


def test_missing_by_target_follows_one_class():
    group = _gaussian_group(n_features=5, classes=4)
    values = group.x[:, 0].double().numpy().copy()
    y = group.y[:, 0].numpy()
    M.mask_by_target(values, y, True, np.random.default_rng(1))
    rows = np.isnan(values).any(1)
    assert len(np.unique(y[rows])) == 1 and rows.sum() == (y == y[rows][0]).sum()


def test_censoring_masks_a_value_tail_of_a_column():
    # every masked column lost one tail of its own values (MNAR) or of another column's (MAR).
    # Columns are censored in turn, so a source's tail is taken over the rows it still has: the
    # masked rows lie beyond every row that is observed in both columns at the end
    own = other = 0
    for seed in range(30):
        r = np.random.default_rng(seed)
        original = r.standard_normal((300, 6))
        values = original.copy()
        M.mask_censoring(values, r)
        for c in np.flatnonzero(np.isnan(values).any(0)):
            rows = np.isnan(values[:, c])

            def is_tail(source, rows=rows, original=original, values=values):
                masked = original[rows, source]
                kept = original[~rows & np.isfinite(values[:, source]), source]
                return masked.max() < kept.min() or masked.min() > kept.max()

            assert any(is_tail(s) for s in range(6))
            own += is_tail(c)
            other += not is_tail(c)
    assert own > 10 and other > 10  # both MNAR and MAR occur


def test_latent_missing_depends_on_the_observed_columns():
    # The missing indicator of a latent-masked column is predictable from the other columns,
    # unlike MCAR at the same rate
    latent, mcar = [], []
    for seed in range(40):
        r = np.random.default_rng(seed)
        base = r.standard_normal((500, 8))
        values = base.copy()
        M.mask_latent(values, r)
        for c in np.flatnonzero(np.isnan(values).any(0)):
            miss = np.isnan(values[:, c]).astype(float)
            others = [j for j in range(8) if j != c and not np.isnan(values[:, j]).any()]
            if not others or miss.std() == 0:
                continue
            fit = np.linalg.lstsq(base[:, others], miss - miss.mean(), rcond=None)[0]
            latent.append(np.corrcoef(base[:, others] @ fit, miss)[0, 1])
            noise = (r.random(500) < miss.mean()).astype(float)
            fit = np.linalg.lstsq(base[:, others], noise - noise.mean(), rcond=None)[0]
            mcar.append(np.corrcoef(base[:, others] @ fit, noise)[0, 1])
    assert len(latent) > 20
    assert np.mean(latent) > np.mean(mcar) + 0.15


def test_near_empty_columns_keep_few_shared_rows():
    for seed in range(10):
        r = np.random.default_rng(seed)
        values = r.standard_normal((1000, 20))
        M.mask_near_empty(values, r)
        masked = np.isnan(values).any(0)
        assert 10 <= masked.sum() <= 16  # 50-80% of the columns
        observed = (~np.isnan(values[:, masked])).mean(0)
        assert (observed <= 0.1 + 2e-3).all() and (observed >= 4 / 1000).all()
        # columns come in groups observed in the same rows (up to one swapped row)
        patterns = {tuple(np.flatnonzero(~np.isnan(values[:, c]))) for c in np.flatnonzero(masked)}
        assert len(patterns) < masked.sum()


def test_structured_missing_rates_in_the_prior():
    # With every table chosen, every table gets missing values (and the targets none)
    prior = P.GraphSCMPrior(
        "regression",
        seed=0,
        missing_dataset_prob=0.0,
        structured_missing_prob=1.0,
        **SIZES,
    )
    batch = prior.sample_batch(16)
    assert torch.isnan(batch.x).flatten(1).any(1).all()
    assert not torch.isnan(batch.y).any()


# --- coarse and noisy values ------------------------------------------------------------------


def test_coarsen_reduces_columns_to_few_levels():
    rng = Rng.from_seed(0)
    fractions = []
    for seed in range(20):
        group = _gaussian_group(n_features=10, seed=seed)
        out = Coarsen(1.0)(group, rng).x[:, 0]
        levels = np.array([len(torch.unique(out[:, c])) for c in range(10)])
        coarse = levels <= 8
        assert coarse.any()
        fractions.append(coarse.mean())
        # levels keep the value order (a coarse column is a monotone function of the original)
        for c in np.flatnonzero(coarse):
            a, b = group.x[:, 0, c], out[:, c]
            order = torch.argsort(a)
            assert (torch.diff(b[order]) >= 0).all()
    assert 0.3 <= np.mean(fractions) <= 1.0


def test_measurement_noise_keeps_scale_and_lowers_correlation():
    rng = Rng.from_seed(0)
    noisy_corr = []
    for seed in range(30):
        group = _gaussian_group(n_rows=2000, n_features=20, seed=seed)
        out = MeasurementNoise(1.0)(group, rng).x[:, 0]
        for c in range(20):
            a, b = group.x[:, 0, c], out[:, c]
            if not torch.equal(a, b):
                noisy_corr.append(np.corrcoef(a, b)[0, 1])
                assert 0.8 < b.std() / a.std() < 1.2
    assert 20 < len(noisy_corr) < 120  # about 10% of 600 columns
    assert np.mean(noisy_corr) < 0.9
