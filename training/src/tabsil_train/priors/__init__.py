"""Training data (the prior axis). Modules are named after what they produce; model-specific
settings live in `tabsil_train.recipe`.

A tabular prior is built by chaining per-stage parts with `pipeline.Pipeline`. A new prior usually
adds one generator or a few transforms and reuses the other parts.

Parts that priors share:

- `base`: `PriorBatch`, the batch handed to the training loop, and `IGNORE_INDEX`, the target the
  loss ignores
- `pipeline`: the hyperparameters -> generator -> transforms -> `PriorBatch` assembler
- `table_size`: the table size rule that draws the train rows and feature count of each batch
- `transforms`: group post-processing (classes, missing values, high-cardinality categorical
  columns, coarse and noisy values, filters)

The graph SCM, in one folder:

- `graph_scm`: the assembled prior (generator + class count + table size rule + missing values)
  and the options a variant fixes
- `graph_scm.generator`: its generator, the continuous table of one group from the TabICLv2 graph
  SCM
- `graph_scm.variants`: the prior with some of its options at fixed values, each under its own
  name

Registered priors (`PRIOR_TYPES`; the three after `graph_scm` are its variants):

- `graph_scm`: graph SCM generator + table size rule + missing values (TabSil's training prior)
- `graph_scm_hicard_missing`: `graph_scm` with high-cardinality categorical columns and structured
  (MAR/MNAR, column-group) missing values (`graph_scm` options at fixed values)
- `graph_scm_realism`: `graph_scm_hicard_missing` plus Kumo-Tabular's other realism parts
  (Kumaraswamy-warped columns, up to 200 categories, correlated column blocks, relabeled targets,
  coarse values)
- `graph_scm_base_realism`: `graph_scm` with the realism parts Kumo-Tabular uses in every stage, at
  its stage-1 values (Kumaraswamy-warped columns, up to 200 categories, correlated column blocks,
  structured missing values, random missing cells in 1% of the tables, too-easy tables drawn
  again)
- `real_tables`: subsamples of the OpenML tables of a pinned corpus with their own targets, through
  the preprocessing of one inference member (`real_tables`); it continues a model and takes the
  regression borders of its start checkpoint

The training loop (`tabsil_train.training.loop`) picks its prior from `PRIOR_TYPES` by name
(`TrainConfig.prior_type`). What a name is registered to (`prior_module`: the `graph_scm` or
`real_tables` module, or a `Variant` of `graph_scm.variants`) provides:

- `make_prior(config, task, seed)`: the `task` prior, an object whose `sample_batch(batch_size)`
  gives a `PriorBatch`. The table size settings are read from `config` (`TrainConfig`)
- `make_borders(config, n_buckets, seed, n_jobs)`: the regression bar distribution borders, drawn
  with the same table size settings as the training prior (`real_tables` has none: a run on it
  starts from a checkpoint and keeps its borders)
- `check_config(config)`: whether the prior can produce the table size settings (both tasks,
  ValueError otherwise)
- `MAX_CLASSES`: the most classes of a classification dataset (the classifier's output count)
- `OPTIONS` (variants and `real_tables`): the options it fixes (for `real_tables` the corpus and
  its digest), recorded as the `prior_options` run parameter

Only fully assembled priors are registered by name (one combination = one name). An experiment
that changes parts or numbers gets its own name (for `graph_scm` options, a new `Variant` in
`graph_scm.variants`), and the training config only names it, so the `prior_type` run parameter
alone tells which data a model was trained on. Like the other axes, names are generic nouns for
what the prior is.
"""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import TYPE_CHECKING

from tabsil_train.priors.base import PriorBatch

if TYPE_CHECKING:
    from tabsil_train.priors.graph_scm.variants import Variant

# name -> where the prior is: a module, or `module:name` for an object of a module (imported on
# use; graph_scm needs the tabicl package)
PRIOR_TYPES: dict[str, str] = {
    "graph_scm": "tabsil_train.priors.graph_scm",
    "graph_scm_hicard_missing": "tabsil_train.priors.graph_scm.variants:HICARD_MISSING",
    "graph_scm_realism": "tabsil_train.priors.graph_scm.variants:REALISM",
    "graph_scm_base_realism": "tabsil_train.priors.graph_scm.variants:BASE_REALISM",
    "real_tables": "tabsil_train.priors.real_tables",
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


def prior_module(prior_type: str) -> ModuleType | Variant:
    """What `prior_type` is registered to: the `graph_scm` module, or the `Variant` of a name
    registered as `module:name`. Both provide what the module docstring lists."""
    name = canonical_prior_type(prior_type)
    if name not in PRIOR_TYPES:
        raise ValueError(f"prior_type must be one of {tuple(PRIOR_TYPES)}: {prior_type}")
    module_name, _, attribute = PRIOR_TYPES[name].partition(":")
    module = importlib.import_module(module_name)
    return getattr(module, attribute) if attribute else module


__all__ = [
    "DEFAULT_PRIOR_TYPE",
    "PRIOR_TYPES",
    "PRIOR_TYPE_ALIASES",
    "PriorBatch",
    "canonical_prior_type",
    "prior_module",
]
