"""Graph SCM prior (prior axis, `prior_type="graph_scm"`): the TabICLv2 graph SCM generator plus a
table size rule and missing values. It chains the parts (`tabsil_train.priors.generators.graph_scm`,
`transforms`) with `Pipeline`, and is the prior used by the TabSil recipe (`tabsil_train.recipe`).

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
   `max_missing_rate` are our choice (the rate is U(`min_missing_rate`, `max_missing_rate`), with
   a lower bound of 0 by default).

Options that make the tables more like real ones, all off by default (then no random number is
drawn for them, so the datasets are exactly those of the v1 prior). They change only x, so the
regression borders do not depend on them. In the order applied, between steps 2 and 3:

- `noise_prob` (`transforms.MeasurementNoise`): noisy measurements of about 10% of the columns
- `coarsen_prob` (`transforms.Coarsen`): columns reduced to 2-8 ordered levels
- `high_cardinality_prob` (`transforms.HighCardinality`): 1-3 columns recoded as categorical
  columns with 21 to (rows) categories, coded as inference codes them (train-row categories,
  unseen ones NaN)
- `structured_missing_prob`, `near_empty_missing_prob`, `latent_missing_prob`
  (`transforms.StructuredMissing`): value-dependent (MAR/MNAR) missing values in column groups,
  before the MCAR step 3

Options that change how the columns depend on each other and on the target, also off by default.
They run between DropInvalid and CoverClasses in step 2. Some also change y, so the regression
borders are drawn with them (`make_borders(..., **options)`, `TARGET_FIELDS`):

- `correlate_prob` (`transforms.CorrelateColumns`): random column blocks blended toward shared
  factors (x only). `correlate_discrete_levels` (default 100) is the most distinct values of a
  column that blending keeps as they are
- `relabel_prob` (`transforms.Relabel`): the target recomputed from 1-4 columns
- `corrected_num_converters`, `corrected_cat_sizes` (`generators.graph_scm.GraphSCM`): the two
  TabICL `PriorConfig` corrections (Kumaraswamy-warped dataset columns, up to 200 categories)
- `reject_trivial_prob` (`generators.graph_scm.GraphSCM`): with this probability a dataset that an
  ExtraTrees probe predicts almost perfectly is drawn again

Regression target scaling, off by default: with `target_train_rows`
(`transforms.TrainRowTargetScale`, a training config field like `filter_unpredictable`), the
target is standardized again with the train rows after the SCM's clip and standardization over all
rows, as inference standardizes it, and a table whose train-row sd is at most 0.1 is drawn again
(Kumo-Tabular's loader). It runs last in step 2, so the regression borders are drawn with it
(`BORDER_PRIOR_FIELDS`).

    prior = GraphSCMPrior(task="classification", seed=0)
    batch = prior.sample_batch(8)  # x (8, rows, features), y (8, rows), n_train

TabICL uses the global numpy and torch random state, and with the same seed it still produces
different datasets if the Python string hash seed (`PYTHONHASHSEED`) differs. Fix both for exactly
reproducible data.
"""

from __future__ import annotations

from typing import Literal

import torch

from tabsil_train.priors.base import PriorBatch
from tabsil_train.priors.graph_scm.generator import GraphSCM
from tabsil_train.priors.pipeline import Pipeline, Rng
from tabsil_train.priors.table_size import check_split_sizes, sample_split_size
from tabsil_train.priors.transforms import (
    Coarsen,
    CorrelateColumns,
    CoverClasses,
    DropInvalid,
    HighCardinality,
    MeasurementNoise,
    Missing,
    Relabel,
    StructuredMissing,
    TrainRowTargetScale,
)
from tabsil_train.priors.transforms.dependence import DISCRETE_LEVELS

# Most classes of a classification dataset (TabPFN v2 paper); the classifier's output count
# (`TrainConfig.n_out`)
MAX_CLASSES = 10
MAX_ATTEMPTS = 100  # most redraws for one dataset (so that a bad setting raises instead of hanging)
# Fewest train rows of a table: classification needs two so that two classes can appear
MIN_TRAIN_ROWS = {"classification": 2, "regression": 1}


class ClassCount:
    """Class count `num_classes` ~ U{2, ..., min(max_classes, train rows)}.

    Every class must appear among the train rows, so the class count never exceeds the train row
    count. With `binary_prob` > 0, 2 is picked first with that probability, otherwise the count is
    drawn from the uniform distribution above (TabICLv2 `PriorDataset` uses 0.5, giving binary
    tasks 0.5 + 0.5/9 ≈ 56% of the time). With 0, no extra random number is drawn, so the datasets
    are the same as without the option.
    """

    def __init__(self, max_classes: int, binary_prob: float = 0.0) -> None:
        if not 0.0 <= binary_prob <= 1.0:
            raise ValueError(f"binary_prob must be between 0 and 1: {binary_prob}")
        self.max_classes = max_classes
        self.binary_prob = binary_prob

    def __call__(self, rng: Rng, n_train: int, n_features: int) -> dict:
        if self.binary_prob and rng.numpy.random() < self.binary_prob:
            return {"num_classes": 2}
        max_classes = min(self.max_classes, n_train)
        return {"num_classes": int(rng.numpy.integers(2, max_classes + 1))}


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
        noise_prob: float = 0.0,
        coarsen_prob: float = 0.0,
        high_cardinality_prob: float = 0.0,
        structured_missing_prob: float = 0.0,
        near_empty_missing_prob: float = 0.0,
        latent_missing_prob: float = 0.0,
        correlate_prob: float = 0.0,
        relabel_prob: float = 0.0,
        corrected_num_converters: bool = False,
        corrected_cat_sizes: bool = False,
        target_train_rows: bool = False,
        min_missing_rate: float = 0.0,
        correlate_discrete_levels: int = DISCRETE_LEVELS,
        reject_trivial_prob: float = 0.0,
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
        generator = GraphSCM(
            task,
            seed=seed,
            filter_unpredictable=filter_unpredictable,
            corrected_num_converters=corrected_num_converters,
            corrected_cat_sizes=corrected_cat_sizes,
            reject_trivial_prob=reject_trivial_prob,
        )
        classification = task == "classification"
        hyperparameters = ClassCount(MAX_CLASSES, binary_prob) if classification else None
        # Column dependence options (they run before CoverClasses, since Relabel changes y)
        dependence = []
        if correlate_prob:
            dependence.append(CorrelateColumns(correlate_prob, correlate_discrete_levels))
        if relabel_prob:
            dependence.append(Relabel(relabel_prob, classification=classification))
        self.dependence = tuple(dependence)
        # Everything before the missing values (regression borders and `_one_dataset` stop here).
        # The train-row target scaling comes last, after every part that changes y
        last = [CoverClasses()] if classification else []
        if target_train_rows and not classification:
            last.append(TrainRowTargetScale())
        self._accept = [DropInvalid(), *dependence, *last]
        self._missing = Missing(missing_dataset_prob, max_missing_rate, min_missing_rate)
        # The options: a part is only added when on, so the v1 settings draw the same numbers
        realism = []
        if noise_prob:
            realism.append(MeasurementNoise(noise_prob))
        if coarsen_prob:
            realism.append(Coarsen(coarsen_prob))
        if high_cardinality_prob:
            realism.append(HighCardinality(high_cardinality_prob))
        if structured_missing_prob or near_empty_missing_prob or latent_missing_prob:
            realism.append(
                StructuredMissing(
                    structured_prob=structured_missing_prob,
                    near_empty_prob=near_empty_missing_prob,
                    latent_prob=latent_missing_prob,
                    classification=classification,
                )
            )
        self.realism = tuple(realism)
        self.pipeline = Pipeline(
            generator,
            rng=self.rng,
            hyperparameters=hyperparameters,
            transforms=[*self._accept, *realism, self._missing],
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

    @property
    def min_missing_rate(self) -> float:
        return self._missing.min_rate

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
PRIOR_FIELDS = ("binary_prob", "filter_unpredictable", "target_train_rows")
# Options that change only x, off by default (module docstring). They are not training config
# fields: a prior that turns them on is its own module with fixed values (`tabsil_train.priors`)
REALISM_FIELDS = (
    "noise_prob",
    "coarsen_prob",
    "high_cardinality_prob",
    "structured_missing_prob",
    "near_empty_missing_prob",
    "latent_missing_prob",
    "correlate_prob",
    "correlate_discrete_levels",
)
# The MCAR missing values (step 3; x only). missing_dataset_prob is a training config field; a prior
# module may fix it and the rates instead (then the config value is not used)
MISSING_FIELDS = ("missing_dataset_prob", "max_missing_rate", "min_missing_rate")
# Options that also change the target, off by default. Like REALISM_FIELDS they are fixed by a
# prior module, which passes them to `make_borders` too
TARGET_FIELDS = (
    "relabel_prob",
    "corrected_num_converters",
    "corrected_cat_sizes",
    "reject_trivial_prob",
)
# Those that also change the regression target distribution (the regression borders follow the
# training prior in these; binary_prob only affects classification)
BORDER_PRIOR_FIELDS = ("filter_unpredictable", "target_train_rows")


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


def make_prior(config, task: str, seed: int, **options) -> GraphSCMPrior:
    """The `task` prior with the dataset sizes of the training config
    (`tabsil_train.training.config.TrainConfig`). `options` are the `GraphSCMPrior` options a
    prior assembled on this one fixes (`REALISM_FIELDS`, `MISSING_FIELDS`, `TARGET_FIELDS`)."""
    unknown = set(options) - set(REALISM_FIELDS) - set(MISSING_FIELDS) - set(TARGET_FIELDS)
    if unknown:
        raise TypeError(f"make_prior got unknown options: {sorted(unknown)}")
    return GraphSCMPrior(
        task,
        seed=seed,
        **{"missing_dataset_prob": config.missing_dataset_prob, **options},
        **_sizes(config),
        **{name: getattr(config, name) for name in PRIOR_FIELDS},
    )


def make_borders(config, n_buckets: int, seed: int, n_jobs: int = 1, **options) -> torch.Tensor:
    """Regression borders (`regression_borders`) drawn with the dataset sizes of the training
    config. `options` are the `TARGET_FIELDS` a prior assembled on this one fixes (the options
    that change y)."""
    unknown = set(options) - set(TARGET_FIELDS)
    if unknown:
        raise TypeError(f"make_borders got unknown options: {sorted(unknown)}")
    return regression_borders(
        n_buckets,
        config.border_datasets,
        seed,
        n_jobs=n_jobs,
        **{name: getattr(config, name) for name in BORDER_PRIOR_FIELDS},
        **options,
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
    appear in training), and so should the `TARGET_FIELDS` options.
    """
    prior_fields = (*BORDER_PRIOR_FIELDS, *TARGET_FIELDS)
    options = {k: v for k, v in sizes_and_options.items() if k in prior_fields}
    sizes = {k: v for k, v in sizes_and_options.items() if k not in prior_fields}
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
