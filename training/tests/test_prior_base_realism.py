"""Tests of the `graph_scm_base_realism` prior (Kumo-Tabular's every-stage prior parts at its
stage-1 values) and of the parts it adds: the lower bound of the random missing rate, the too-easy
table filter, and the discrete-level threshold of the correlated column blocks."""

from dataclasses import replace

import numpy as np
import pytest
import torch
from train_helpers import tabsil_smoke

from tabsil_train.priors import PRIOR_TYPES, prior_module
from tabsil_train.priors import graph_scm as P
from tabsil_train.priors.graph_scm import generator as G
from tabsil_train.priors.graph_scm.variants import BASE_REALISM as B
from tabsil_train.priors.pipeline import Group, Rng
from tabsil_train.priors.transforms import CorrelateColumns, Missing, StructuredMissing
from tabsil_train.priors.transforms import dependence as D

SIZES = dict(min_rows=60, max_rows=120, log_uniform_rows=True, max_cells=120 * 12, max_features=12)


def _config(**changes):
    return replace(tabsil_smoke(), prior_type="graph_scm_base_realism", **{**SIZES, **changes})


def test_the_prior_is_registered_with_kumo_stage_1_values():
    assert "graph_scm_base_realism" in PRIOR_TYPES
    config = _config()
    assert prior_module(config.prior_type) is B and config.n_out == P.MAX_CLASSES
    for task in ("classification", "regression"):
        prior = config.make_prior(task, seed=0)
        # correlated blocks in 30% of the tables, keeping columns of up to 200 values
        (correlate,) = prior.dependence
        assert isinstance(correlate, CorrelateColumns)
        assert (correlate.dataset_prob, correlate.discrete_levels) == (0.3, 200)
        # structured missing values in 20% of the tables; no near-empty or latent ones
        (structured,) = prior.realism
        assert isinstance(structured, StructuredMissing)
        assert (structured.structured_prob, structured.near_empty_prob, structured.latent_prob) == (
            0.2,
            0.0,
            0.0,
        )
        # random missing cells in 1% of the tables at U(0.01, 0.08), whatever the config says
        assert config.missing_dataset_prob == 0.3
        assert (prior.missing_dataset_prob, prior.min_missing_rate, prior.max_missing_rate) == (
            0.01,
            0.01,
            0.08,
        )
        generator = prior.pipeline.generator
        assert generator._config.use_corrected_num_converters
        assert generator._config.use_corrected_cat_sizes
        assert generator.reject_trivial_prob == 1.0 and generator.filters
        assert generator.probe_config().remove_trivial_datasets
        assert generator.filter_unpredictable is config.filter_unpredictable
        batch = prior.sample_batch(3)
        assert batch.x.shape[0] == 3 and torch.isfinite(batch.y).all()


def test_stage_2_and_3_parts_are_left_out():
    for name in ("high_cardinality_prob", "near_empty_missing_prob", "latent_missing_prob"):
        assert B.OPTIONS.get(name, 0.0) == 0.0
    for name in ("relabel_prob", "coarsen_prob", "noise_prob"):
        assert B.OPTIONS.get(name, 0.0) == 0.0


def test_borders_are_drawn_with_the_options_that_change_y():
    assert B.TARGET_OPTIONS == {
        "corrected_num_converters": True,
        "corrected_cat_sizes": True,
        "reject_trivial_prob": 1.0,
    }
    config = _config(border_datasets=4, filter_unpredictable=False)
    borders = B.make_borders(config, 20, seed=1)
    assert borders.shape == (21,) and (borders.diff() > 0).all()
    assert not torch.equal(borders, P.make_borders(replace(config, prior_type="graph_scm"), 20, 1))


# --- random missing cells with a lower bound on the rate --------------------------------------


def test_missing_rate_has_a_lower_bound():
    x = torch.randn(2000, 30, 5)
    group = Missing(1.0, 0.08, min_rate=0.01)(
        Group(x, torch.zeros(2000, 30), 1000, {}), Rng.from_seed(0)
    )
    rates = group.x.isnan().float().mean((0, 2))
    assert rates.min() > 0.005 and rates.max() < 0.09
    with pytest.raises(ValueError):
        Missing(0.5, 0.1, min_rate=0.2)


def test_missing_without_a_lower_bound_draws_as_before():
    # the default lower bound 0 draws the same rates as U(0, max_rate)
    x = torch.randn(50, 6, 4)
    group = Group(x, torch.zeros(50, 6), 25, {})
    a = Missing(0.5, 0.3)(group, Rng.from_seed(3)).x
    r = Rng.from_seed(3).numpy
    expected = []
    for d in range(6):
        column = x[:, d]
        if r.random() < 0.5:
            rate = r.uniform(0.0, 0.3)
            column = column.masked_fill(
                torch.from_numpy(r.random(tuple(column.shape)) < rate), float("nan")
            )
        expected.append(column)
    assert torch.equal(a.nan_to_num(-9.0), torch.stack(expected, 1).nan_to_num(-9.0))


# --- the too-easy table filter ----------------------------------------------------------------


class _Fake:
    """A generator stand-in for `filtered`: records the probe settings of every attempt."""

    def __init__(self, generator, results):
        self.inner, self.results, self.seen = generator, list(results), []
        self.filter_unpredictable = generator.filter_unpredictable
        self.filters = generator.filters
        self.probe_config = generator.probe_config
        self._config = generator._config

    def _should_filter(self, x, y, config, *, is_classif):
        self.seen.append((config.remove_trivial_datasets, config.filter_unpredictable_datasets))
        return self.results.pop(0)


def _make():
    return torch.randn(20, 3), torch.randn(20)


@pytest.mark.parametrize(
    ("unpredictable", "p", "expected"),
    [
        (False, 0.0, []),  # v1 without the filter: no probe
        (True, 0.0, [(False, True)] * 3),  # v1 (stage 1 recipe): unpredictable only
        (False, 1.0, [(True, False)] * 3),  # too-easy only
        (True, 1.0, [(True, True)] * 3),  # both (graph_scm_base_realism with the v1 recipe)
    ],
)
def test_filtered_probes_with_the_asked_checks(unpredictable, p, expected):
    generator = G.GraphSCM(
        "regression", seed=0, filter_unpredictable=unpredictable, reject_trivial_prob=p
    )
    fake = _Fake(generator, [True, True, False])
    G.filtered(_make, fake, regression=True)
    assert fake.seen == expected


def test_reject_trivial_prob_below_one_checks_some_attempts():
    generator = G.GraphSCM("regression", seed=0, reject_trivial_prob=0.5)
    checks = [generator.probe_config().remove_trivial_datasets for _ in range(400)]
    assert 150 < sum(checks) < 250
    with pytest.raises(ValueError):
        G.GraphSCM("regression", seed=0, reject_trivial_prob=1.5)


def test_the_probe_rejects_a_table_the_trees_solve():
    generator = G.GraphSCM("regression", seed=0, reject_trivial_prob=1.0)
    rng = np.random.default_rng(0)
    x = torch.from_numpy(rng.integers(0, 4, size=(300, 2))).float()
    trivial = x[:, 0] * 2.0 - 1.0  # a step function of one column: the trees fit it exactly
    noisy = trivial + torch.from_numpy(rng.normal(size=300)).float()
    probe = generator.probe_config()
    assert generator._should_filter(x, trivial, probe, is_classif=False)
    assert not generator._should_filter(x, noisy, probe, is_classif=False)
    # without the option the same table is kept
    assert not generator._should_filter(x, trivial, generator._config, is_classif=False)


# --- correlated blocks keep columns of up to discrete_levels values --------------------------


def test_correlated_blocks_keep_columns_up_to_the_discrete_levels():
    rng = np.random.default_rng(0)
    n = 1000
    levels = rng.integers(0, 150, size=n).astype(float)  # a column of 150 values
    values = np.stack([levels, rng.normal(size=n), rng.normal(size=n)], 1)

    def kept(discrete_levels):
        for seed in range(20):  # a draw that blends the first column
            out = D.correlate(values, np.random.default_rng(seed), discrete_levels)
            if not np.array_equal(out[:, 0], values[:, 0]):
                return np.array_equal(np.sort(out[:, 0]), np.sort(values[:, 0]))
        raise AssertionError("no draw blended the first column")

    assert kept(200)  # its values and their counts stay
    assert not kept(D.DISCRETE_LEVELS)  # 100: blended linearly
    with pytest.raises(ValueError):
        CorrelateColumns(0.3, discrete_levels=0)


def test_the_run_param_names_the_values_the_prior_fixes():
    import json

    from tabsil_train.training.loop import prior_options

    assert prior_options("graph_scm") == "{}"
    fixed = json.loads(prior_options("graph_scm_base_realism"))
    assert fixed == B.OPTIONS
    assert (fixed["missing_dataset_prob"], fixed["max_missing_rate"]) == (0.01, 0.08)
