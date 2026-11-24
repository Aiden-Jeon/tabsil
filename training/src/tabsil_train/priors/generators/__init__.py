"""Generators: the continuous table (x, y) of one group (datasets sharing their hyperparameters).

A new prior usually adds one file here. A generator is a callable shaped like
`tabsil_train.priors.pipeline.Generator`: `(n_rows, n_features, n_datasets, hp, rng) -> (x, y)`,
with x of shape (rows, datasets, features) and y of shape (rows, datasets).

- `graph_scm.GraphSCM`: the TabICLv2 graph SCM (TabSil's training prior; needs the `tabicl`
  package)

Modules are imported on use (nothing is imported here, so that `tabicl` is not loaded through
`graph_scm`).
"""
