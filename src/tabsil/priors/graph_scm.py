"""Graph SCM prior (prior axis, `prior_type="graph_scm"`): the TabICLv2 graph SCM generator plus a
table size rule and missing values. It chains the parts (`tabsil.priors.generators.graph_scm`,
`transforms`) with `Pipeline`, and is the prior used by the TabSil recipe (`tabsil.recipe`).

1. Table size: each batch draws the rows (train + eval) and the train-row share, then the feature
   count, with the TabICLv2 training-script rule (`table_size.sample_split_size`). All datasets
   of a batch share one size so that the batch can be computed together on the GPU.
2. Per dataset: class count (`ClassCount`, classification; 2 with probability `binary_prob`) ->
   graph SCM (`filter_unpredictable`: redraw unpredictable graphs and datasets) -> drop datasets
   that produced NaN (`DropInvalid`) -> reorder rows so that every eval class also appears among
   the train rows (`CoverClasses`, classification). A dropped dataset is redrawn from the class
   count on (at most `MAX_ATTEMPTS` times).
3. Missing values (TabPFN v2 paper, "Post-processing"; `Missing`): in some datasets every cell is
   erased with the same probability (MCAR). The paper gives no rates; `missing_dataset_prob` and
   `max_missing_rate` are our choice.

    prior = GraphSCMPrior(task="classification", seed=0)
    batch = prior.sample_batch(8)  # x (8, rows, features), y (8, rows), n_train

TabICL uses the global numpy and torch random state, and with the same seed it still produces
different datasets if the Python string hash seed (`PYTHONHASHSEED`) differs. Fix both for exactly
reproducible data.
"""

from __future__ import annotations

from typing import Literal

import torch

from tabsil.priors.base import PriorBatch
from tabsil.priors.generators.graph_scm import GraphSCM
from tabsil.priors.hyperparameters import ClassCount
from tabsil.priors.pipeline import Pipeline, Rng
from tabsil.priors.table_size import check_split_sizes, sample_split_size
from tabsil.priors.transforms import CoverClasses, DropInvalid, Missing

# Most classes of a classification dataset (TabPFN v2 paper); the classifier's output count
# (`TrainConfig.n_out`)
MAX_CLASSES = 10
MAX_ATTEMPTS = 100  # most redraws for one dataset (so that a bad setting raises instead of hanging)
# Fewest train rows of a table: classification needs two so that two classes can appear
MIN_TRAIN_ROWS = {"classification": 2, "regression": 1}


class GraphSCMPrior:
    def __init__(
        self,
        task: Literal["classification", "regression"],
        seed: int = 0,
        *,
        missing_dataset_prob: float = 0.3,
        max_missing_rate: float = 0.3,
        max_features: int = 100,
        log_uniform_rows: bool = False,
        max_cells: int = 1_024 * 100,
        min_rows: int = 1024,
        max_rows: int = 1024,
        min_train_fraction: float = 0.3,
        max_train_fraction: float = 0.9,
        binary_prob: float = 0.0,
        filter_unpredictable: bool = False,
    ) -> None:
        self._task = task
        self.min_train_rows = MIN_TRAIN_ROWS[task]
        check_split_sizes(
            min_rows=min_rows,
            max_rows=max_rows,
            min_train_fraction=min_train_fraction,
            max_train_fraction=max_train_fraction,
            max_features=max_features,
            max_cells=max_cells,
            min_train_rows=self.min_train_rows,
        )
        # Size settings are read for every batch; the task and missing-value settings go into the
        # parts, so they are read-only properties
        self.max_features = max_features
        self.log_uniform_rows = log_uniform_rows
        self.max_cells = max_cells
        self.min_rows = min_rows
        self.max_rows = max_rows
        self.min_train_fraction = min_train_fraction
        self.max_train_fraction = max_train_fraction
        self.rng = Rng.from_seed(seed)
        # This also seeds TabICL's global random state
        generator = GraphSCM(task, seed=seed, filter_unpredictable=filter_unpredictable)
        classification = task == "classification"
        hyperparameters = ClassCount(MAX_CLASSES, binary_prob) if classification else None
        # Everything before the missing values (regression borders and `_one_dataset` stop here)
        self._accept = [DropInvalid(), CoverClasses()] if classification else [DropInvalid()]
        self._missing = Missing(missing_dataset_prob, max_missing_rate)
        self.pipeline = Pipeline(
            generator,
            rng=self.rng,
            hyperparameters=hyperparameters,
            transforms=[*self._accept, self._missing],
            max_attempts=MAX_ATTEMPTS,
        )

    @property
    def task(self) -> str:
        return self._task

    @property
    def missing_dataset_prob(self) -> float:
        return self._missing.dataset_prob

    @property
    def max_missing_rate(self) -> float:
        return self._missing.max_rate

    def sample_shape(self) -> tuple[int, int, int]:
        """(train rows, features, eval rows) of the next table (`table_size.sample_split_size`)."""
        n_train, n_eval, n_features = sample_split_size(
            self.rng.numpy,
            min_rows=self.min_rows,
            max_rows=self.max_rows,
            min_train_fraction=self.min_train_fraction,
            max_train_fraction=self.max_train_fraction,
            max_features=self.max_features,
            log_uniform_rows=self.log_uniform_rows,
            max_cells=self.max_cells,
            min_train_rows=self.min_train_rows,
        )
        return n_train, n_features, n_eval

    def _one_dataset(
        self, n_train: int, n_features: int, n_eval: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One dataset before missing values are added: x (rows, features), y (rows)."""
        group = self.pipeline.sample_group(
            n_train + n_eval, n_train, n_features, transforms=self._accept
        )
        return group.x[:, 0], group.y[:, 0]

    def sample_batch(self, batch_size: int) -> PriorBatch:
        """A batch of `batch_size` datasets of one size (`sample_shape`)."""
        n_train, n_features, n_eval = self.sample_shape()
        return self.pipeline.sample_batch(batch_size, n_train + n_eval, n_train, n_features)


# Dataset size fields and their defaults (the same as the size arguments of `GraphSCMPrior`).
# Add new fields here only
SIZE_DEFAULTS = {
    "max_features": 100,
    "log_uniform_rows": False,
    "max_cells": 1_024 * 100,
    "min_rows": 1024,
    "max_rows": 1024,
    "min_train_fraction": 0.3,
    "max_train_fraction": 0.9,
}
SIZE_FIELDS = tuple(SIZE_DEFAULTS)
# Prior options that change the training data distribution
PRIOR_FIELDS = ("binary_prob", "filter_unpredictable")
# Those that also change the regression target distribution (the regression borders follow the
# training prior in these; binary_prob only affects classification)
BORDER_PRIOR_FIELDS = ("filter_unpredictable",)


def _sizes(config) -> dict:
    # Only the dataset size fields of the training config (so that the prior and the regression
    # borders use the same size distribution)
    return {name: getattr(config, name) for name in SIZE_FIELDS}


def check_config(config) -> None:
    """Whether this prior can produce the dataset sizes of the training config, for both tasks
    (called when a `TrainConfig` is created)."""
    sizes = {k: v for k, v in _sizes(config).items() if k != "log_uniform_rows"}
    for min_train_rows in MIN_TRAIN_ROWS.values():
        check_split_sizes(**sizes, min_train_rows=min_train_rows)


def make_prior(config, task: str, seed: int) -> GraphSCMPrior:
    """The `task` prior with the dataset sizes of the training config
    (`tabsil.training.config.TrainConfig`)."""
    return GraphSCMPrior(
        task,
        seed=seed,
        missing_dataset_prob=config.missing_dataset_prob,
        **_sizes(config),
        **{name: getattr(config, name) for name in PRIOR_FIELDS},
    )


def make_borders(config, n_buckets: int, seed: int, n_jobs: int = 1) -> torch.Tensor:
    """Regression borders (`regression_borders`) drawn with the dataset sizes of the training
    config."""
    return regression_borders(
        n_buckets,
        config.border_datasets,
        seed,
        n_jobs=n_jobs,
        **{name: getattr(config, name) for name in BORDER_PRIOR_FIELDS},
        **_sizes(config),
    )


def _border_targets(args: tuple[int, int, dict, dict]) -> torch.Tensor:
    # The unique target values of n regression datasets (no missing values added)
    n_datasets, seed, sizes, options = args
    prior = GraphSCMPrior(
        "regression",
        seed=seed,
        missing_dataset_prob=0.0,
        **options,
        **sizes,
    )
    values = []
    for _ in range(n_datasets):
        _, y = prior._one_dataset(*prior.sample_shape())
        values.append(torch.unique(y.float()))
    return torch.cat(values)


def regression_borders(
    n_buckets: int = 5000,
    n_datasets: int = 2000,
    seed: int = 0,
    *,
    n_jobs: int = 1,
    **sizes_and_options,
) -> torch.Tensor:
    """Borders of the regression output buckets (TabPFN v2 paper, Methods "Regression details").

    Many datasets are drawn from the prior, their z-normalized targets are pooled, and the
    1/n_buckets quantiles of that distribution become the borders; half-normal tails cover both
    ends (`FullSupportBarDistribution`). Only the unique values of each dataset are pooled (our
    choice), so that datasets with many repeated values (discretized targets, values clipped at
    4σ) do not pull quantiles together into near-zero-width buckets.
    Dataset size arguments are keywords named as in `SIZE_FIELDS`, with `SIZE_DEFAULTS` for
    missing ones; they mean the same as in `GraphSCMPrior`, and in training `make_borders` passes
    the training prior's values. `filter_unpredictable` (`BORDER_PRIOR_FIELDS`, default from
    `GraphSCMPrior`) should also match the training prior (filtered datasets' targets never
    appear in training).
    """
    options = {k: v for k, v in sizes_and_options.items() if k in BORDER_PRIOR_FIELDS}
    sizes = {k: v for k, v in sizes_and_options.items() if k not in BORDER_PRIOR_FIELDS}
    unknown = set(sizes) - set(SIZE_DEFAULTS)
    if unknown:
        raise TypeError(f"regression_borders got unknown size/prior arguments: {sorted(unknown)}")
    # Split over n_jobs processes (a different seed per process)
    counts = [n_datasets // n_jobs + (i < n_datasets % n_jobs) for i in range(n_jobs)]
    table = {**SIZE_DEFAULTS, **sizes}
    jobs = [(n, seed + i, table, options) for i, n in enumerate(counts)]
    if n_jobs == 1:
        targets = _border_targets(jobs[0])
    else:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(n_jobs) as pool:
            targets = torch.cat(list(pool.map(_border_targets, jobs)))
    targets = targets[torch.randperm(len(targets))[: 2**24]]  # input size limit of torch.quantile
    borders = torch.quantile(targets, torch.linspace(0, 1, n_buckets + 1))
    # Repeated values would give zero-width buckets, so spread them slightly (same border count)
    for i in range(1, len(borders)):
        borders[i] = torch.maximum(borders[i], borders[i - 1] + 1e-6)
    return borders
