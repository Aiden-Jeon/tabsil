"""Graph SCM prior with the realism parts that Kumo-Tabular's prior has in every stage
(prior axis, `prior_type="graph_scm_base_realism"`).

The `graph_scm` prior (the v1 training data) with the parts that Kumo-Tabular turns on in all three
of its stages, at its stage-1 values (`OPTIONS`; NVIDIA structured-data-models
`train/kumo/tabular/prior.py` `make_prior`, `loader.py` `PriorDataset`, and the stage-1 commands of
`train/kumo/tabular/README.md`):

- `corrected_num_converters`: dataset columns (and the regression target) can be Kumaraswamy-warped
  (sdm `DistributionalMixtureConverter` warps the dataset value)
- `corrected_cat_sizes`: categorical columns get up to 200 categories instead of 9 (sdm
  `max_cardinality=LogUniformInt(2, 200)`)
- `correlate_prob` 0.3 (`transforms.CorrelateColumns`): sdm `correlated_block_prob`. Columns with
  at most 200 distinct values keep their values (`correlate_discrete_levels` 200), so that every
  categorical column the SCM makes is kept categorical, as sdm keeps its categorical columns
- `structured_missing_prob` 0.2 (`transforms.StructuredMissing`, MAR/MNAR in column groups): sdm
  `structured_missingness_prob`. Kumo's near-empty and latent-factor missing values are stage 2 and
  3 parts and stay off
- random missing cells (`transforms.Missing`): 1% of the tables, each cell missing at a rate of
  U(0.01, 0.08) (sdm `mcar_missingness_prob` 0.01, `MaskRandom(Uniform(0.01, 0.08))`). This
  replaces the v1 values (30% of the tables at U(0, 0.3)), so the training config's
  `missing_dataset_prob` is not used
- `reject_trivial_prob` 1 (`generators.graph_scm.GraphSCM`): a table that an ExtraTrees probe
  predicts with an OOB RMSE of at most 0.05 times the mean's is drawn again (sdm `should_drop`
  with the loader's `keep_trivial_prob` 0 in stage 1; stages 2 and 3 keep half of them)

Left out, as stage 2 and 3 parts of Kumo's prior: high-cardinality codes, near-empty columns,
latent-factor missing values, relabeled targets, coarse values (`graph_scm_realism` has them).
Not ported: noisy-measurement columns, per-table function preferences, argmax classes,
heavy-tailed targets, row replay. Everything else is the graph SCM prior's: table sizes, the class
count (`binary_prob`; Kumo draws 2-10 classes uniformly), `filter_unpredictable` (Kumo's
`drop_unlearnable`, on in every stage).

The column warp, the category sizes and the too-easy filter change y, so the regression borders are
drawn with them (`TARGET_OPTIONS`).

    TrainConfig(prior_type="graph_scm_base_realism")
"""

from __future__ import annotations

import torch

from tabsil_train.priors import graph_scm
from tabsil_train.priors.graph_scm import GraphSCMPrior, check_config

MAX_CLASSES = graph_scm.MAX_CLASSES
OPTIONS = {
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
# The options that change y (the regression borders are drawn with them)
TARGET_OPTIONS = {k: v for k, v in OPTIONS.items() if k in graph_scm.TARGET_FIELDS}


def make_prior(config, task: str, seed: int) -> GraphSCMPrior:
    """The `task` prior with the dataset sizes of the training config and `OPTIONS`."""
    return graph_scm.make_prior(config, task, seed, **OPTIONS)


def make_borders(config, n_buckets: int, seed: int, n_jobs: int = 1) -> torch.Tensor:
    """Regression borders drawn from this prior's targets (`TARGET_OPTIONS`)."""
    return graph_scm.make_borders(config, n_buckets, seed, n_jobs, **TARGET_OPTIONS)


__all__ = ["MAX_CLASSES", "OPTIONS", "TARGET_OPTIONS", "check_config", "make_borders", "make_prior"]
