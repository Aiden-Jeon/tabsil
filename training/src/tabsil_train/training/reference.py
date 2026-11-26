"""The reference validation set (`TrainConfig.reference_val_tables`): tables that do not depend on
the run's seed or on the arm's prior, for a validation loss that can be compared across runs.

The ordinary validation batches (`val_*` metrics) are drawn from the run's own prior with a seed
derived from the run seed, so two runs with different priors or seeds are validated on different
tables. The reference set is drawn from one fixed prior with fixed seeds:

- the `graph_scm` prior with the v1 recipe's prior settings (`REFERENCE_PRIOR`), whatever
  `prior_type`, `binary_prob`, `filter_unpredictable` and `missing_dataset_prob` the run uses
- the run's table size settings (`min_rows`, `max_rows`, train share, features, cells), so that
  the tables have the size the run trains on. Runs with the same table sizes (every run of one
  curriculum stage) get the same tables
- seeds from `REFERENCE_SEED`, per task and per chunk of `REFERENCE_CHUNK` batches. The chunks are
  drawn in parallel processes, and the set does not depend on how many there are. A larger set
  extends a smaller one (its first chunks are the same). The processes run with the Python hash
  seed 0 (the graph SCM prior depends on it), whatever the training process has

Its loss is logged as `ref_loss_classification`, `ref_acc_classification`, `ref_loss_regression`
and their mean `ref_loss` (with `_avg` for the LAWA average). The set also depends on the tabicl
version (and possibly on the platform's floating point), so the run gets a `reference_val_digest`
tag: two runs whose digests match were validated on the same tables.

What the loss compares: the loss of one task on the same tables, between runs of any seed, prior
or optimizer. It is a second readout for arms that share a prior (the plan's phase 1B and phase
2). Between runs on different priors it also measures how far the reference prior is from each
run's own training data, so it is not a measure of accuracy on real tables.

With the train-row target scaling (`TrainConfig.target_train_rows`, the plan's YCTX arm) the set is
still drawn without it. A run with the scaling gets each reference regression target standardized
with its train rows, as in its training, and its loss is mapped back to the set's units (each
table's loss times its train-row sd, `loop._validate`). Then:

- the classification loss and accuracy compare runs with and without the scaling directly (the
  option changes only regression tables)
- the regression loss is in the same units for both, on the same tables, each model given the
  targets in the form it was trained on. It compares how well the two fit the reference prior's
  targets in those units. It does not compare them in the train-row units in which inference
  works: there each table would count by 1 / its train-row sd, so tables with a narrow train-row
  spread would count more. The set also keeps the tables with a train-row sd of at most 0.1 that
  the scaled run never trained on
- the `val_*` loss of a run with the scaling is in train-row units on its own prior, so it does
  not compare with that of a run without it
"""

from __future__ import annotations

import hashlib
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

import torch

from tabsil_train.priors import PriorBatch

# The prior of the reference set: graph_scm with the prior settings of the v1 recipe
# (`tabsil_train.recipe.TRAINING`, the `TrainConfig` defaults)
REFERENCE_PRIOR = {
    "prior_type": "graph_scm",
    "binary_prob": 0.5,
    "filter_unpredictable": True,
    "missing_dataset_prob": 0.3,
    "target_train_rows": False,
}
REFERENCE_SEED = 30_000_000  # disjoint from the training (seed tuples), val and border seeds
REFERENCE_BATCH = 8  # tables per batch (one table size per batch), independent of val_batch
REFERENCE_CHUNK = 4  # batches drawn by one process from one seed
REFERENCE_HASH_SEED = "0"  # the Python hash seed of the processes that draw the set
TASK_SEEDS = {"classification": 0, "regression": 1_000_000}


def reference_config(config):
    """`config` with the reference prior settings (`REFERENCE_PRIOR`); the table sizes are kept."""
    return replace(config, **REFERENCE_PRIOR)


def _chunk(args: tuple) -> list[PriorBatch]:
    config, task, seed, n_batches = args
    torch.set_num_threads(1)
    prior = config.make_prior(task, seed)
    return [prior.sample_batch(REFERENCE_BATCH) for _ in range(n_batches)]


def reference_sets(config, tasks, n_jobs: int = 1) -> dict[str, list[PriorBatch]]:
    """The `config.reference_val_tables` reference tables of each task, in batches of
    `REFERENCE_BATCH`.

    The chunks are drawn in up to `n_jobs` new processes (spawned) with the Python hash seed
    `REFERENCE_HASH_SEED`, so the tables do not depend on the hash seed of the training process,
    and drawing them leaves this process's random state (tabicl reseeds the global one)
    untouched."""
    n_batches = config.reference_val_tables // REFERENCE_BATCH
    ref = reference_config(config)
    jobs = [
        (ref, task, REFERENCE_SEED + TASK_SEEDS[task] + i, min(REFERENCE_CHUNK, n_batches - start))
        for task in tasks
        for i, start in enumerate(range(0, n_batches, REFERENCE_CHUNK))
    ]
    # a spawned process reads its hash seed from the environment when it starts
    saved = os.environ.get("PYTHONHASHSEED")
    os.environ["PYTHONHASHSEED"] = REFERENCE_HASH_SEED
    try:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max(1, min(n_jobs, len(jobs))), mp_context=context) as pool:
            chunks = list(pool.map(_chunk, jobs))
    finally:
        if saved is None:
            del os.environ["PYTHONHASHSEED"]
        else:
            os.environ["PYTHONHASHSEED"] = saved
    out: dict[str, list[PriorBatch]] = {task: [] for task in tasks}
    for (_, task, _, _), chunk in zip(jobs, chunks, strict=True):
        out[task].extend(chunk)
    return out


def reference_batches(config, task: str, n_jobs: int = 1) -> list[PriorBatch]:
    """The reference tables of one task (`reference_sets`)."""
    return reference_sets(config, (task,), n_jobs)[task]


def reference_jobs() -> int:
    """Processes that draw the reference set (as for the regression borders)."""
    return max(1, min(32, (os.cpu_count() or 1) // 2))


def reference_digest(sets: dict[str, list[PriorBatch]]) -> str:
    """A short hash of the reference tables (the `reference_val_digest` run tag)."""
    h = hashlib.sha256()
    for task in sorted(sets):
        h.update(task.encode())
        for batch in sets[task]:
            h.update(str(batch.n_train).encode())
            for t in (batch.x, batch.y):
                h.update(t.float().nan_to_num(nan=-12345.0).contiguous().numpy().tobytes())
    return h.hexdigest()[:16]


__all__ = [
    "REFERENCE_BATCH",
    "REFERENCE_CHUNK",
    "REFERENCE_PRIOR",
    "REFERENCE_SEED",
    "reference_batches",
    "reference_config",
    "reference_digest",
    "reference_jobs",
    "reference_sets",
]
