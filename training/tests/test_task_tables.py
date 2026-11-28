"""Task-shaped tables (`priors.transforms.tasks`): a regression table whose rows are ordered by a
column before the cut (`SortedSplit`), and a classification table turned into data rows with
hidden anomalies against reference rows (`OutlierTable`), alone and inside the graph SCM prior."""

import numpy as np
import pytest
import torch

from tabsil_train.priors import graph_scm as P
from tabsil_train.priors import prior_module
from tabsil_train.priors.pipeline import Group, Rng
from tabsil_train.priors.transforms import OutlierTable, SortedSplit
from tabsil_train.priors.transforms import tasks as T

SIZES = dict(min_rows=80, max_rows=160, log_uniform_rows=True, max_cells=160 * 12, max_features=12)


def _regression_table(n: int = 200, seed: int = 0):
    r = np.random.default_rng(seed)
    a = r.normal(size=n)
    x = np.column_stack([a, r.integers(0, 4, n).astype(float), r.normal(size=n)])
    return x, 2.0 * a + 0.3 * r.normal(size=n)


def _class_table(n: int = 300, classes: int = 4, seed: int = 0):
    r = np.random.default_rng(seed)
    y = r.integers(0, classes, n)
    x = r.normal(size=(n, 5)) + 4.0 * y[:, None]
    x[:, 4] = r.integers(0, 3, n)  # a discrete column
    return x, y.astype(float)


# --- SortedSplit ------------------------------------------------------------------------------


def test_sorted_split_orders_the_rows_by_a_column_with_enough_values():
    x, y = _regression_table()
    for seed in range(8):
        out_x, out_y = T.sorted_split(x, y, 150, np.random.default_rng(seed), index_prob=0.0)
        # the discrete column never orders the rows; one of the two others does
        ordered = [
            j
            for j in (0, 2)
            if out_x[:150, j].max() <= out_x[150:, j].min()
            or out_x[:150, j].min() >= out_x[150:, j].max()
        ]
        assert len(ordered) == 1
        # the same rows, moved together with their targets
        assert sorted(map(tuple, out_x)) == sorted(map(tuple, x))
        assert out_y[:150].mean() == pytest.approx(0.0, abs=1e-9)
        assert out_y[:150].std() == pytest.approx(1.0)
        rows = {tuple(row): target for row, target in zip(x, y, strict=True)}
        back = np.array([rows[tuple(row)] for row in out_x])
        scale = np.polyfit(back, out_y, 1)
        np.testing.assert_allclose(scale[0] * back + scale[1], out_y, atol=1e-9)
    directions = {
        bool(T.sorted_split(x, y, 150, np.random.default_rng(s), 0.0)[0][0, 0] < 0)
        for s in range(20)
    }
    assert directions == {True, False}  # ascending and descending both occur


def test_sorted_split_puts_the_eval_targets_beyond_the_train_targets():
    x, y = _regression_table()
    x = x[:, :1]  # only the column the target follows
    out_x, out_y = T.sorted_split(x, y, 150, np.random.default_rng(1), index_prob=0.0)
    assert abs(out_y[150:].mean()) > 2.0  # in train-row standard deviations
    assert np.abs(out_y).max() <= T.MAX_ABS_TARGET
    steep = T.sorted_split(x, 1000.0 * (x[:, 0] > 1.0), 150, np.random.default_rng(1), 0.0)
    assert steep is None or np.abs(steep[1]).max() <= T.MAX_ABS_TARGET


def test_sorted_split_by_the_column_the_target_follows():
    x, y = _regression_table()
    np.testing.assert_allclose(T.rank_correlation(x[:, [0, 2]], y)[1], 0.0, atol=0.2)
    assert T.rank_correlation(x[:, [0, 2]], y)[0] > 0.9
    assert T.rank_correlation(x, np.zeros(200)).tolist() == [0.0, 0.0, 0.0]
    assert T.ordering_columns(x).tolist() == [0, 2]
    for seed in range(6):
        out_x, out_y = T.sorted_split(
            x, y, 150, np.random.default_rng(seed), index_prob=0.0, target_prob=1.0
        )
        # always column 0, which the target follows: the eval targets leave the train range
        assert out_x[:150, 0].max() <= out_x[150:, 0].min() or (
            out_x[:150, 0].min() >= out_x[150:, 0].max()
        )
        beyond = (np.abs(out_y[150:]) > np.abs(out_y[:150]).max()).mean()
        assert abs(out_y[150:].mean()) > 2.0 and beyond > 0.05


def test_sorted_split_index_and_tables_it_leaves():
    x, y = _regression_table()
    out_x, _ = T.sorted_split(x, y, 150, np.random.default_rng(0), index_prob=1.0)
    index = [j for j in range(3) if np.array_equal(out_x[:, j], np.arange(200.0))]
    assert len(index) == 1 and index[0] in (0, 2)
    # a train-row target without spread: drawn again
    assert T.sorted_split(x, np.zeros(200), 150, np.random.default_rng(0), 0.0) is None
    # no column to order by: the table as it is
    discrete = np.column_stack([x[:, 1], x[:, 1] + 1])
    same_x, same_y = T.sorted_split(discrete, y, 150, np.random.default_rng(0), 0.0)
    assert same_x is discrete and same_y is y


def test_sorted_split_transform_and_its_probability():
    torch.manual_seed(0)
    group = Group(torch.randn(100, 4, 3), torch.randn(100, 4), 70, {})
    rng = Rng.from_seed(0)
    assert SortedSplit(0.0)(group, rng).x.equal(group.x)
    out = SortedSplit(1.0, index_prob=0.0, target_prob=0.5)(group, Rng.from_seed(0))
    assert out.x.shape == group.x.shape and out.x.dtype == group.x.dtype
    for d in range(4):
        train = out.y[:70, d].double()
        assert train.mean().abs() < 1e-5 and (train.std(correction=0) - 1).abs() < 1e-5
        assert any(
            out.x[:70, d, j].max() <= out.x[70:, d, j].min()
            or out.x[:70, d, j].min() >= out.x[70:, d, j].max()
            for j in range(3)
        )
    flat = Group(group.x, torch.zeros(100, 4), 70, {})
    assert SortedSplit(1.0)(flat, Rng.from_seed(0)) is None
    with pytest.raises(ValueError, match="between 0 and 1"):
        SortedSplit(1.5)


# --- OutlierTable -----------------------------------------------------------------------------


@pytest.mark.parametrize("kind", T.ANOMALY_KINDS)
def test_outlier_table_layout(kind):
    x, y = _class_table()
    n_train = 200
    for seed in range(6):
        out = T.outlier_table(x, y, n_train, np.random.default_rng(seed), kinds=(kind,))
        assert out is not None
        out_x, out_y = out
        assert out_x.shape == x.shape and set(np.unique(out_y)) == {0.0, 1.0}
        assert np.isfinite(out_x).all()
        context_y, query_y = out_y[:n_train], out_y[n_train:]
        # reference rows are 20% to 50% of the context (more only when inliers run short)
        assert 0.2 * n_train - 1 <= context_y.sum() <= (0.5 if kind != "mode" else 0.9) * n_train
        assert query_y.sum() == 50  # half of the 100 queries are anomalies
        # a discrete column keeps its values in every kind of row
        assert set(np.unique(out_x[:, 4])) <= {0.0, 1.0, 2.0}


def test_outlier_table_mode_takes_whole_classes():
    x, y = _class_table()
    rows = {tuple(row): label for row, label in zip(x, y, strict=True)}
    for seed in range(6):
        out_x, out_y = T.outlier_table(
            x, y, 200, np.random.default_rng(seed), kinds=("mode",), reference_kinds=("uniform",)
        )
        query_x, query_y = out_x[200:], out_y[200:]
        inlier_classes = {rows[tuple(row)] for row in query_x[query_y == 0]}
        anomaly_classes = {rows[tuple(row)] for row in query_x[query_y == 1]}
        assert inlier_classes and anomaly_classes and not inlier_classes & anomaly_classes
        # hidden anomalies of the context: data rows (label 0) of the anomalous classes
        data = out_x[:200][out_y[:200] == 0]
        hidden = [rows[tuple(row)] in anomaly_classes for row in data]
        assert sum(hidden) <= T.MAX_CONTAMINATION * len(data) + 1
    contaminated = sum(
        any(
            rows.get(tuple(row)) in {rows[tuple(q)] for q in o[0][200:][o[1][200:] == 1]}
            for row in o[0][:200][o[1][:200] == 0]
        )
        for o in (
            T.outlier_table(x, y, 200, np.random.default_rng(s), ("mode",), ("uniform",))
            for s in range(10)
        )
    )
    assert contaminated >= 5  # most contexts hold hidden anomalies


def test_outlier_table_anomalies_and_references():
    x, y = _class_table()
    r = np.random.default_rng(0)
    lo, hi = x.min(0), x.max(0)
    pad = T.PAD * (hi - lo)
    uniform = T.uniform_rows(x, 400, r)
    assert (uniform >= lo - pad).all() and (uniform <= hi + pad).all()
    assert (uniform[:, 0] > hi[0]).any() and set(uniform[:, 4]) <= {0.0, 1.0, 2.0}
    marginal = T.marginal_rows(x, 400, r)
    assert all(set(marginal[:, j]) <= set(x[:, j]) for j in range(5))
    assert abs(np.corrcoef(marginal[:, 0], marginal[:, 1])[0, 1]) < 0.2  # x's is about 0.95
    local = T.local_rows(x, 300, r)
    moved = np.array([[v not in set(x[:, j]) for j, v in enumerate(row)] for row in local])
    assert moved[:, :4].sum(1).max() <= 1 and moved.any()  # at most a third of 5 columns
    center, width = T.cluster_center(x, r)
    z = np.abs(center - x.mean(0)) / x.std(0)
    assert (z[:4] >= 1.5 - 1e-9).all() and (z[:4] <= 3.0 + 1e-9).all()
    assert width[4] == 0.0 and center[4] in (0.0, 1.0, 2.0)
    dependency = T.dependency_rows(x, 300, r)
    assert all(set(dependency[:, j]) <= set(x[:, j]) for j in range(5))
    assert not all(tuple(row) in set(map(tuple, x)) for row in dependency)


def test_outlier_table_falls_back_and_leaves_small_tables():
    x, y = _class_table()
    one_class = np.zeros(len(y))
    out_x, out_y = T.outlier_table(x, one_class, 200, np.random.default_rng(0), kinds=("mode",))
    assert set(np.unique(out_y)) == {0.0, 1.0}  # built as a `local` table
    assert T.nominal_classes(one_class, np.random.default_rng(0)) is None
    nominal = T.nominal_classes(y, np.random.default_rng(0))
    assert 1 <= len(nominal) <= 3
    assert T.outlier_table(x[:12], y[:12], 6, np.random.default_rng(0)) is None
    small = Group(torch.randn(12, 2, 3), torch.randint(0, 2, (12, 2)).float(), 6, {})
    out = OutlierTable(1.0)(small, Rng.from_seed(0))
    assert out.x.equal(small.x) and out.y.equal(small.y)
    with pytest.raises(ValueError, match="anomaly kinds"):
        OutlierTable(1.0, kinds=("spike",))
    with pytest.raises(ValueError, match="reference kinds"):
        OutlierTable(1.0, reference_kinds=("normal",))


# --- in the graph SCM prior -------------------------------------------------------------------


def test_the_options_are_off_by_default_and_reach_the_prior():
    plain = P.GraphSCMPrior("regression", seed=3, **SIZES).sample_batch(2)
    again = P.GraphSCMPrior("regression", seed=3, sorted_split_prob=0.0, **SIZES).sample_batch(2)
    assert torch.equal(plain.y, again.y)
    assert P.GraphSCMPrior("regression", seed=3, **SIZES).task_tables == ()
    regression = P.GraphSCMPrior("regression", seed=3, sorted_split_prob=1.0, **SIZES)
    assert [type(t) for t in regression.task_tables] == [SortedSplit]
    classification = P.GraphSCMPrior("classification", seed=3, outlier_table_prob=1.0, **SIZES)
    assert [type(t) for t in classification.task_tables] == [OutlierTable]
    # each option belongs to one task
    assert P.GraphSCMPrior("regression", seed=3, outlier_table_prob=1.0, **SIZES).task_tables == ()
    assert (
        P.GraphSCMPrior("classification", seed=3, sorted_split_prob=1.0, **SIZES).task_tables == ()
    )


def test_prior_batches_of_task_tables():
    regression = P.GraphSCMPrior(
        "regression", seed=1, sorted_split_prob=1.0, missing_dataset_prob=0.0, **SIZES
    )
    sorted_tables = 0
    for _ in range(4):
        batch = regression.sample_batch(2)
        assert batch.y.abs().max() <= T.MAX_ABS_TARGET
        for x, y in zip(batch.x, batch.y, strict=True):
            train, rest = x[: batch.n_train], x[batch.n_train :]
            # strictly beyond (a constant column is not an ordering column)
            ordered = (train.max(0).values < rest.min(0).values) | (
                train.min(0).values > rest.max(0).values
            )
            if ordered.any():
                # a table with a column to order by: the target is in train-row units
                sorted_tables += 1
                mean = y[: batch.n_train].double().mean()
                assert mean.abs() < 1e-4
    assert sorted_tables >= 4  # a table whose columns are all discrete stays as it is
    classification = P.GraphSCMPrior(
        "classification",
        seed=1,
        outlier_table_prob=1.0,
        high_cardinality_prob=0.5,
        structured_missing_prob=0.5,
        **SIZES,
    )
    for _ in range(3):
        batch = classification.sample_batch(2)
        assert set(batch.y.unique().tolist()) == {0.0, 1.0}
        assert (batch.y[:, : batch.n_train].sum(1) > 0).all()  # reference rows in every context
        # half of the queries are anomalies (fewer when a `mode` table's classes are small)
        share = batch.y[:, batch.n_train :].mean(1)
        assert ((share > 0) & (share <= 0.5)).all()


def test_registered_task_priors():
    probe = prior_module("graph_scm_task_probe")
    assert probe.OPTIONS["sorted_split_prob"] == 1.0 and probe.OPTIONS["outlier_table_prob"] == 1.0
    base = prior_module("graph_scm_hicard_missing").OPTIONS
    assert {k: probe.OPTIONS[k] for k in base} == base
    assert probe.TARGET_OPTIONS == {}  # the borders are the base prior's


def test_training_runs_on_the_probe_prior(tmp_path):
    from dataclasses import replace

    from train_helpers import tabsil_smoke

    from tabsil_train.training.loop import prior_options, train

    config = replace(
        tabsil_smoke(),
        prior_type="graph_scm_task_probe",
        filter_unpredictable=False,
        border_datasets=6,
        batch_size=4,
        micro_batch=4,
        min_rows=64,
        max_rows=64,
        max_features=4,
        max_cells=256,
        steps=4,
        warmup_steps=1,
        log_every=1,
        val_every=4,
        checkpoint_every=4,
    )
    assert '"sorted_split_prob": 1.0' in prior_options(config.prior_type)
    result = train(config, tmp_path, mlflow_log=False)
    assert torch.isfinite(torch.tensor(result["history"]["train_loss"])).all()
