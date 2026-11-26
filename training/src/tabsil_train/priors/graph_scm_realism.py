"""Graph SCM prior with the realism parts of Kumo-Tabular's prior that TabSil's lacks
(prior axis, `prior_type="graph_scm_realism"`).

It is `graph_scm_hicard_missing` (`HICARD_MISSING`, unchanged, so that the run that tests this
prior can use the #14 stage 2 run as its control) plus, at Kumo-Tabular's rates
(NVIDIA structured-data-models `train/kumo/tabular/prior.py` `make_prior`):

- `corrected_num_converters`: dataset columns (and the regression target) can be Kumaraswamy-warped,
  so skewed and bounded marginals appear. sdm warps the dataset value in every stage
- `corrected_cat_sizes`: the SCM's categorical columns get up to 200 categories instead of 9, as
  sdm draws them (`max_cardinality=LogUniformInt(2, 200)`) in every stage
- `correlate_prob` 0.3 (`transforms.CorrelateColumns`): sdm `correlated_block_prob`, every stage
- `relabel_prob` 0.1 (`transforms.Relabel`): sdm `relabel_prob`, stages 2 and 3
- `coarsen_prob` 0.1 (`transforms.Coarsen`): sdm `coarsen_prob`, stages 2 and 3

Not included: sdm's count/distributional columns (`DistributionalMixtureConverter`; TabSil's nearest
part is `noise_prob`), the trivial-table filter, the regression target tail warp (stage 3 only), and
the stage 3 row replay. The first three options and Relabel change y, so the regression borders are
drawn with them (`make_borders`).

    TrainConfig(prior_type="graph_scm_realism")
"""

from __future__ import annotations

import torch

from tabsil_train.priors import graph_scm
from tabsil_train.priors.graph_scm import GraphSCMPrior, check_config
from tabsil_train.priors.graph_scm_hicard_missing import OPTIONS as HICARD_MISSING

MAX_CLASSES = graph_scm.MAX_CLASSES
OPTIONS = {
    **HICARD_MISSING,
    "corrected_num_converters": True,
    "corrected_cat_sizes": True,
    "correlate_prob": 0.3,
    "relabel_prob": 0.1,
    "coarsen_prob": 0.1,
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
