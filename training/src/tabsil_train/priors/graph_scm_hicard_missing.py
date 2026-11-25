"""Graph SCM prior with high-cardinality categorical columns and structured missing values
(prior axis, `prior_type="graph_scm_hicard_missing"`).

The `graph_scm` prior (`tabsil_train.priors.graph_scm`, the v1 training data) with two of its
options turned on, at fixed values (`OPTIONS`):

- `high_cardinality_prob` 0.1 (`transforms.HighCardinality`): 1-3 columns of 10% of the tables
  become categorical columns with LogUniformInt(21, rows) categories, coded as inference codes
  them. Kumo-Tabular recodes 5% of its tables (at least 201 categories), but its SCM already makes
  columns with up to 200 categories, where the TabICL SCM stops at 9; 10% covers both ranges
- `structured_missing_prob` 0.2, `near_empty_missing_prob` 0.05, `latent_missing_prob` 0.03
  (`transforms.StructuredMissing`): Kumo-Tabular's stage 2/3 values. With the v1 MCAR values
  (30% of tables), about half of the tables have missing values, as in Seldon's prior

Everything else (table sizes, classes, filters, MCAR) is the graph SCM prior's, and so are the
regression borders: the options change only x.

    TrainConfig(prior_type="graph_scm_hicard_missing")
"""

from __future__ import annotations

from tabsil_train.priors import graph_scm
from tabsil_train.priors.graph_scm import GraphSCMPrior, check_config, make_borders

MAX_CLASSES = graph_scm.MAX_CLASSES
OPTIONS = {
    "high_cardinality_prob": 0.1,
    "structured_missing_prob": 0.2,
    "near_empty_missing_prob": 0.05,
    "latent_missing_prob": 0.03,
}


def make_prior(config, task: str, seed: int) -> GraphSCMPrior:
    """The `task` prior with the dataset sizes of the training config and `OPTIONS`."""
    return graph_scm.make_prior(config, task, seed, **OPTIONS)


__all__ = ["MAX_CLASSES", "OPTIONS", "check_config", "make_borders", "make_prior"]
