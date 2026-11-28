"""Graph SCM prior variants (prior axis): the `graph_scm` prior (`tabsil_train.priors.graph_scm`,
the v1 training data) with some of its options turned on, at fixed values. Each has its own
`prior_type` name (`tabsil_train.priors.PRIOR_TYPES`):

- `HICARD_MISSING` (`graph_scm_hicard_missing`): high-cardinality categorical columns and
  structured missing values
- `REALISM` (`graph_scm_realism`): `HICARD_MISSING` plus the realism parts of Kumo-Tabular's prior
  that TabSil's lacks
- `BASE_REALISM` (`graph_scm_base_realism`): the realism parts that Kumo-Tabular's prior has in
  every stage
- `TASK_PROBE` (`graph_scm_task_probe`): `HICARD_MISSING` with every table task-shaped, for a
  short fine-tuning that asks whether such tables move a task metric

    TrainConfig(prior_type="graph_scm_hicard_missing")

A variant is a `Variant`: the options it fixes (`OPTIONS`) and what the training loop asks of a
registered prior (`make_prior`, `make_borders`, `check_config`, `MAX_CLASSES`), as the `graph_scm`
module provides them for its own name. The comment above each variant says why it has its values.
An experiment that changes the options or their values gets a new `Variant` here and its own name
in `PRIOR_TYPES`.
"""

from __future__ import annotations

import torch

from tabsil_train.priors import graph_scm
from tabsil_train.priors.graph_scm import GraphSCMPrior


class Variant:
    """The `graph_scm` prior with `options` at fixed values. `tabsil_train.priors.prior_module`
    gives this object for the variant's name; it is used like the `graph_scm` module."""

    MAX_CLASSES = graph_scm.MAX_CLASSES
    check_config = staticmethod(graph_scm.check_config)

    def __init__(self, options: dict) -> None:
        # The `GraphSCMPrior` options the variant fixes (logged as the `prior_options` run param)
        self.OPTIONS = options
        # The options that change y (the regression borders are drawn with them)
        self.TARGET_OPTIONS = {k: v for k, v in options.items() if k in graph_scm.TARGET_FIELDS}

    def make_prior(self, config, task: str, seed: int) -> GraphSCMPrior:
        """The `task` prior with the dataset sizes of the training config and `OPTIONS`."""
        return graph_scm.make_prior(config, task, seed, **self.OPTIONS)

    def make_borders(self, config, n_buckets: int, seed: int, n_jobs: int = 1) -> torch.Tensor:
        """Regression borders drawn from this prior's targets (`TARGET_OPTIONS`)."""
        return graph_scm.make_borders(config, n_buckets, seed, n_jobs, **self.TARGET_OPTIONS)


# ---------------------------------------------------------------------------
# graph_scm_hicard_missing
# ---------------------------------------------------------------------------
# Graph SCM prior with high-cardinality categorical columns and structured missing values
# (prior axis, `prior_type="graph_scm_hicard_missing"`).
#
# The `graph_scm` prior (`tabsil_train.priors.graph_scm`, the v1 training data) with two of its
# options turned on, at fixed values (`OPTIONS`):
#
# - `high_cardinality_prob` 0.1 (`transforms.HighCardinality`): 1-3 columns of 10% of the tables
#   become categorical columns with LogUniformInt(21, rows) categories, coded as inference codes
#   them. Kumo-Tabular recodes 5% of its tables (at least 201 categories), but its SCM already
#   makes columns with up to 200 categories, where the TabICL SCM stops at 9; 10% covers both
#   ranges
# - `structured_missing_prob` 0.2, `near_empty_missing_prob` 0.05, `latent_missing_prob` 0.03
#   (`transforms.StructuredMissing`): Kumo-Tabular's stage 2/3 values. With the v1 MCAR values
#   (30% of tables), about half of the tables have missing values, as in Seldon's prior
#
# Everything else (table sizes, classes, filters, MCAR) is the graph SCM prior's, and so are the
# regression borders: the options change only x.
HICARD_MISSING = Variant(
    {
        "high_cardinality_prob": 0.1,
        "structured_missing_prob": 0.2,
        "near_empty_missing_prob": 0.05,
        "latent_missing_prob": 0.03,
    }
)

# ---------------------------------------------------------------------------
# graph_scm_realism
# ---------------------------------------------------------------------------
# Graph SCM prior with the realism parts of Kumo-Tabular's prior that TabSil's lacks
# (prior axis, `prior_type="graph_scm_realism"`).
#
# It is `graph_scm_hicard_missing` (`HICARD_MISSING`, unchanged, so that the run that tests this
# prior can use the #14 stage 2 run as its control) plus, at Kumo-Tabular's rates
# (NVIDIA structured-data-models `train/kumo/tabular/prior.py` `make_prior`):
#
# - `corrected_num_converters`: dataset columns (and the regression target) can be
#   Kumaraswamy-warped, so skewed and bounded marginals appear. sdm warps the dataset value in
#   every stage
# - `corrected_cat_sizes`: the SCM's categorical columns get up to 200 categories instead of 9, as
#   sdm draws them (`max_cardinality=LogUniformInt(2, 200)`) in every stage
# - `correlate_prob` 0.3 (`transforms.CorrelateColumns`): sdm `correlated_block_prob`, every stage
# - `relabel_prob` 0.1 (`transforms.Relabel`): sdm `relabel_prob`, stages 2 and 3
# - `coarsen_prob` 0.1 (`transforms.Coarsen`): sdm `coarsen_prob`, stages 2 and 3
#
# Not included: sdm's count/distributional columns (`DistributionalMixtureConverter`; TabSil's
# nearest part is `noise_prob`), the trivial-table filter, the regression target tail warp (stage 3
# only), and the stage 3 row replay. The first two options and Relabel change y, so the
# regression borders are drawn with them (`make_borders`).
REALISM = Variant(
    {
        **HICARD_MISSING.OPTIONS,
        "corrected_num_converters": True,
        "corrected_cat_sizes": True,
        "correlate_prob": 0.3,
        "relabel_prob": 0.1,
        "coarsen_prob": 0.1,
    }
)

# ---------------------------------------------------------------------------
# graph_scm_base_realism
# ---------------------------------------------------------------------------
# Graph SCM prior with the realism parts that Kumo-Tabular's prior has in every stage
# (prior axis, `prior_type="graph_scm_base_realism"`).
#
# The `graph_scm` prior (the v1 training data) with the parts that Kumo-Tabular turns on in all
# three of its stages, at its stage-1 values (`OPTIONS`; NVIDIA structured-data-models
# `train/kumo/tabular/prior.py` `make_prior`, `loader.py` `PriorDataset`, and the stage-1 commands
# of `train/kumo/tabular/README.md`):
#
# - `corrected_num_converters`: dataset columns (and the regression target) can be
#   Kumaraswamy-warped (sdm `DistributionalMixtureConverter` warps the dataset value)
# - `corrected_cat_sizes`: categorical columns get up to 200 categories instead of 9 (sdm
#   `max_cardinality=LogUniformInt(2, 200)`)
# - `correlate_prob` 0.3 (`transforms.CorrelateColumns`): sdm `correlated_block_prob`. Columns with
#   at most 200 distinct values keep their values (`correlate_discrete_levels` 200), so that every
#   categorical column the SCM makes is kept categorical, as sdm keeps its categorical columns
# - `structured_missing_prob` 0.2 (`transforms.StructuredMissing`, MAR/MNAR in column groups): sdm
#   `structured_missingness_prob`. Kumo's near-empty and latent-factor missing values are stage 2
#   and 3 parts and stay off
# - random missing cells (`transforms.Missing`): 1% of the tables, each cell missing at a rate of
#   U(0.01, 0.08) (sdm `mcar_missingness_prob` 0.01, `MaskRandom(Uniform(0.01, 0.08))`). This
#   replaces the v1 values (30% of the tables at U(0, 0.3)), so the training config's
#   `missing_dataset_prob` is not used
# - `reject_trivial_prob` 1 (`generator.GraphSCM`): a table that an ExtraTrees probe predicts with
#   an OOB RMSE of at most 0.05 times the mean's is drawn again (sdm `should_drop` with the
#   loader's `keep_trivial_prob` 0 in stage 1; stages 2 and 3 keep half of them)
#
# Left out, as stage 2 and 3 parts of Kumo's prior: high-cardinality codes, near-empty columns,
# latent-factor missing values, relabeled targets, coarse values (`graph_scm_realism` has them).
# Not ported: noisy-measurement columns, per-table function preferences, argmax classes,
# heavy-tailed targets, row replay. Everything else is the graph SCM prior's: table sizes, the
# class count (`binary_prob`; Kumo draws 2-10 classes uniformly), `filter_unpredictable` (Kumo's
# `drop_unlearnable`, on in every stage).
#
# The column warp, the category sizes and the too-easy filter change y, so the regression borders
# are drawn with them (`TARGET_OPTIONS`).
BASE_REALISM = Variant(
    {
        "corrected_num_converters": True,
        "corrected_cat_sizes": True,
        "correlate_prob": 0.3,
        "correlate_discrete_levels": 200,
        "structured_missing_prob": 0.2,
        "missing_dataset_prob": 0.01,
        "min_missing_rate": 0.01,
        "max_missing_rate": 0.08,
        "reject_trivial_prob": 1.0,
    }
)

# ---------------------------------------------------------------------------
# graph_scm_task_probe
# ---------------------------------------------------------------------------
# Graph SCM prior in which every table is task-shaped (prior axis,
# `prior_type="graph_scm_task_probe"`): the probe of `training/docs/v2-prior-task-plan.md` (tier
# 2), a short fine-tuning of weights trained on `graph_scm_hicard_missing`.
#
# It is `graph_scm_hicard_missing` (`HICARD_MISSING`, the prior of the weights the probe starts
# from) with both task options at 1 (`transforms.tasks`):
#
# - `sorted_split_prob` 1: every regression table has its rows ordered by one column before the
#   cut and its target standardized with the train rows (forecasting)
# - `outlier_table_prob` 1: every classification table is data rows with hidden anomalies
#   labeled 0 and reference rows labeled 1, with the true label asked for held-out data rows
#   (outlier detection)
#
# With every table of a task type task-shaped the model sees the most such tables a run of its
# length can give: the best case for a prior that mixes them in at a small share. It is not a
# prior to train a model for release on: no ordinary table is left.
#
# The regression borders are the base prior's (the options are not `TARGET_OPTIONS`): the probe
# takes its borders from the weights it starts from.
TASK_PROBE = Variant(
    {
        **HICARD_MISSING.OPTIONS,
        "sorted_split_prob": 1.0,
        "outlier_table_prob": 1.0,
    }
)

__all__ = ["BASE_REALISM", "HICARD_MISSING", "REALISM", "TASK_PROBE", "Variant"]
