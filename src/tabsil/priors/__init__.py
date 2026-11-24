"""Training data (the prior axis). Modules are named after what they produce; model-specific
settings live in `tabsil.recipe`.

A tabular prior is built by chaining per-stage parts with `pipeline.Pipeline`. A new prior usually
adds one generator or a few transforms and reuses the other parts.

- `base`: `PriorBatch`, the batch handed to the training loop, and `IGNORE_INDEX`, the target the
  loss ignores
- `pipeline`: the hyperparameters -> generator -> transforms -> `PriorBatch` assembler
- `table_size`: the table size rule that draws the train rows and feature count of each batch
- `hyperparameters`: class count
- `generators`: the continuous table of one group (graph SCM)
- `transforms`: group post-processing (categorical columns, scaling, classes, padding, missing
  values, filters)

Assembled priors:

- `graph_scm`: graph SCM generator + table size rule + missing values (TabSil's training prior)

The training loop (`tabsil.training.loop`) picks its prior from `PRIOR_TYPES` by name
(`TrainConfig.prior_type`). A prior module provides:

- `make_prior(config, task, seed)`: the `task` prior, an object whose `sample_batch(batch_size)`
  gives a `PriorBatch`. The table size settings are read from `config` (`TrainConfig`)
- `make_borders(config, n_buckets, seed, n_jobs)`: the regression bar distribution borders, drawn
  with the same table size settings as the training prior
- `check_config(config)`: whether the prior can produce the table size settings (both tasks,
  ValueError otherwise)
- `MAX_CLASSES`: the most classes of a classification dataset (the classifier's output count)

Only fully assembled priors are registered by name (one combination = one module in this folder
= one name). An experiment that changes parts or numbers gets a new assembly module and its own
name, and the training config only names it, so the `prior_type` run parameter alone tells which
data a model was trained on. Like the other axes, names are generic nouns for what the prior is.
"""

from __future__ import annotations

import importlib
from types import ModuleType

from tabsil.priors.base import PriorBatch

# name -> module (imported on use; graph_scm needs the tabicl package)
PRIOR_TYPES: dict[str, str] = {
    "graph_scm": "tabsil.priors.graph_scm",
}
# old name -> current name, accepted when reading older training runs whose params still use it
PRIOR_TYPE_ALIASES: dict[str, str] = {
    "tabicl": "graph_scm",
}
# Training runs that did not record prior_type all used the graph_scm prior
DEFAULT_PRIOR_TYPE = "graph_scm"


def canonical_prior_type(prior_type: str) -> str:
    """The current name for an old name. Unknown names are returned as is (`prior_module` raises
    the ValueError)."""
    return PRIOR_TYPE_ALIASES.get(prior_type, prior_type)


def prior_module(prior_type: str) -> ModuleType:
    name = canonical_prior_type(prior_type)
    if name not in PRIOR_TYPES:
        raise ValueError(f"prior_type must be one of {tuple(PRIOR_TYPES)}: {prior_type}")
    return importlib.import_module(PRIOR_TYPES[name])


__all__ = [
    "DEFAULT_PRIOR_TYPE",
    "PRIOR_TYPES",
    "PRIOR_TYPE_ALIASES",
    "PriorBatch",
    "canonical_prior_type",
    "prior_module",
]
