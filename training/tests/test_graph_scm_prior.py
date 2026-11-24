"""Tests of the training data: the graph SCM prior (TabICL prior) and its table size rule."""

from dataclasses import replace

import pytest
import torch

from tabsil_train.priors import graph_scm as P
from tabsil_train.training.config import TrainConfig

# One table size: 80 rows, 50 of them train rows (0.625 is exact in binary), at most 5 features
FIXED = dict(min_rows=80, max_rows=80, min_train_fraction=0.625, max_train_fraction=0.625)
FIXED |= dict(max_features=5, max_cells=80 * 5)


def test_classification_batch_shapes_and_label_split():
    prior = P.GraphSCMPrior("classification", seed=0, **FIXED)
    batch = prior.sample_batch(3)
    assert batch.x.shape[:2] == (3, 80) and 1 <= batch.x.shape[2] <= 5
    assert batch.y.shape == (3, 80)
    assert batch.n_train == 50
    for y in batch.y:
        train, rest = set(y[:50].tolist()), set(y[50:].tolist())
        assert rest <= train and len(train) >= 2  # masked classes all seen in training
        assert max(train) < P.MAX_CLASSES


def test_missing_values_are_added_only_when_requested():
    clean = P.GraphSCMPrior("regression", seed=1, missing_dataset_prob=0.0, **FIXED)
    assert not torch.isnan(clean.sample_batch(2).x).any()
    noisy = P.GraphSCMPrior(
        "regression", seed=1, missing_dataset_prob=1.0, max_missing_rate=0.5, **FIXED
    )
    assert torch.isnan(noisy.sample_batch(4).x).any()


def test_regression_targets_are_standardized():
    sizes = {**FIXED, "min_rows": 250, "max_rows": 250, "max_cells": 250 * 5}
    prior = P.GraphSCMPrior("regression", seed=2, **sizes)
    y = prior.sample_batch(4).y
    assert torch.allclose(y.mean(dim=1), torch.zeros(4), atol=1e-3)
    # z-scored, so the std is at most 1; a dataset with a near-constant target can stay far below
    std = y.std(dim=1, correction=0)
    assert (std <= 1 + 1e-2).all() and (std > 0.99).any()


def test_regression_borders_are_increasing():
    borders = P.regression_borders(n_buckets=100, n_datasets=20, seed=0, **FIXED)
    assert borders.shape == (101,)
    assert (borders[1:] > borders[:-1]).all()


def test_classification_with_very_few_training_rows_does_not_hang():
    # the smallest train share still leaves two train rows for classification
    sizes = {**FIXED, "min_rows": 4, "max_rows": 4, "min_train_fraction": 0.1}
    sizes |= {"max_train_fraction": 0.1, "max_cells": 4 * 5}
    prior = P.GraphSCMPrior("classification", seed=3, missing_dataset_prob=0.0, **sizes)
    for _ in range(3):
        batch = prior.sample_batch(4)
        assert batch.n_train == 2
        for y in batch.y:
            train = set(y[: batch.n_train].tolist())
            assert len(train) >= 2 and set(y[batch.n_train :].tolist()) <= train


def test_borders_use_the_training_table_sizes(monkeypatch):
    # Training prior and regression borders come from the same dataset size distribution (long
    # tables too)
    config = replace(
        TrainConfig.smoke(),
        min_rows=40,
        max_rows=60,
        log_uniform_rows=True,
        max_cells=3000,
        max_features=20,
        border_datasets=3,
        filter_unpredictable=False,
    )
    seen = []
    real = P.GraphSCMPrior

    def spy(*args, **kwargs):
        seen.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(P, "GraphSCMPrior", spy)
    P.make_borders(config, n_buckets=10, seed=0)
    P.make_prior(config, "regression", seed=0)
    borders_sizes, prior_sizes = ({k: kw[k] for k in P.SIZE_FIELDS} for kw in seen)
    assert borders_sizes == prior_sizes == {k: getattr(config, k) for k in P.SIZE_FIELDS}
    assert seen[0]["filter_unpredictable"] is seen[1]["filter_unpredictable"] is False


def test_border_prior_with_default_sizes_is_the_stage_1_prior():
    # Passing default size settings gives the same borders as a prior built without them
    explicit = P.regression_borders(50, 4, seed=3, **P._sizes(TrainConfig()))
    assert torch.equal(explicit, P.regression_borders(50, 4, seed=3))


def test_prior_rejects_sizes_it_cannot_draw():
    with pytest.raises(ValueError, match="max_cells"):
        P.GraphSCMPrior("regression", min_rows=60_000, max_rows=60_000, max_cells=50_000)
    with pytest.raises(ValueError, match="min_rows"):
        P.GraphSCMPrior("regression", min_rows=4096, max_rows=2048, max_cells=400_000)
    # classification needs two train rows and one eval row
    with pytest.raises(ValueError, match="min_rows"):
        P.GraphSCMPrior("classification", min_rows=2, max_rows=8, max_cells=100)
    P.GraphSCMPrior("regression", min_rows=2, max_rows=8, max_cells=100)
    # the training config checks the sizes for both tasks
    with pytest.raises(ValueError, match="min_rows"):
        replace(TrainConfig.smoke(), min_rows=2, max_rows=8)


def test_make_prior_takes_the_task():
    config = replace(TrainConfig.smoke(), filter_unpredictable=False)
    for task in ("classification", "regression"):
        assert config.make_prior(task, seed=0).task == task
    with pytest.raises(ValueError, match="task"):
        config.make_prior("multitask", seed=0)
