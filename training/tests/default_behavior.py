"""Fingerprints of the default training data and training behavior, for
`test_default_behavior.py`.

The v2 recipe options (reference validation set, train-row target scaling, optimizer groups, the
`graph_scm_base_realism` prior) are all off by default, and the v1 control arm and the stage-2 runs
started before them must train exactly as before. This module records what the default settings
produce: batches and regression borders of the three priors that existed before the options, the
validation batches, and the losses and weights of a short smoke training run. It uses only APIs
that existed at commit 01973a5, so the stored values (`data/default_behavior.json`) were written
by running it on that commit:

    mkdir /tmp/tabsil-base && git archive 01973a5 src training/src | tar -x -C /tmp/tabsil-base
    PYTHONHASHSEED=0 PYTHONPATH=/tmp/tabsil-base/src:/tmp/tabsil-base/training/src \
        uv run --package tabsil-train python training/tests/default_behavior.py \
        training/tests/data/default_behavior.json

The graph SCM prior depends on the Python hash seed, so it must run with PYTHONHASHSEED=0 (the test
starts it in a subprocess).
"""

from __future__ import annotations

import json
import sys
import tempfile
from dataclasses import replace

import torch

# The priors registered before the v2 recipe options
PRIORS = ("graph_scm", "graph_scm_hicard_missing", "graph_scm_realism")
TASKS = ("classification", "regression")
SIZES = dict(min_rows=60, max_rows=120, log_uniform_rows=True, max_features=12, max_cells=120 * 12)


def tensor_print(t: torch.Tensor) -> list:
    """Shape, NaN count, and the sum, absolute sum and squared sum of the finite values."""
    t = t.detach().double().cpu()
    finite = t[torch.isfinite(t)]
    return [
        list(t.shape),
        int(torch.isnan(t).sum()),
        float(finite.sum()),
        float(finite.abs().sum()),
        float(finite.square().sum()),
    ]


def batch_print(batch) -> dict:
    return {"x": tensor_print(batch.x), "y": tensor_print(batch.y), "n_train": int(batch.n_train)}


def fingerprints() -> dict:
    from tabsil_train.priors import prior_module
    from tabsil_train.training.config import TrainConfig
    from tabsil_train.training.loop import _make_val_batches, train

    torch.set_num_threads(1)
    out: dict = {"priors": {}, "borders": {}, "val": {}}
    for name in PRIORS:
        config = replace(TrainConfig.smoke(), prior_type=name, border_datasets=12, **SIZES)
        for task in TASKS:
            prior = config.make_prior(task, seed=3)
            out["priors"][f"{name}/{task}"] = [batch_print(prior.sample_batch(3)) for _ in range(2)]
        borders = prior_module(name).make_borders(config, 40, seed=1, n_jobs=1)
        out["borders"][name] = tensor_print(borders)
    config = replace(TrainConfig.smoke(), **SIZES)
    for task in TASKS:
        out["val"][task] = [batch_print(b) for b in _make_val_batches(config, task, n_batches=2)]
    # A short smoke run with the default settings (Muon on every parameter, collapse guard on)
    run = replace(
        TrainConfig.smoke(),
        steps=6,
        warmup_steps=2,
        log_every=2,
        val_every=3,
        checkpoint_every=6,
        keep_every=None,
    )
    with tempfile.TemporaryDirectory() as tmp:
        result = train(run, tmp, mlflow_log=False)
        state = torch.load(f"{tmp}/latest.pt", map_location="cpu", weights_only=True)
    out["train"] = {
        "history": result["history"],
        "weights": {k: tensor_print(v) for k, v in sorted(state["model"].items())},
        "borders": tensor_print(state["borders"]),
        "optimizer_type": state["optimizer_type"],
        "optimizer_groups": [
            {k: v for k, v in g.items() if k != "params"} | {"n_params": len(g["params"])}
            for g in state["optimizer"]["param_groups"]
        ],
    }
    return out


if __name__ == "__main__":
    # the training loop prints its progress, so the result goes to the file given as argument
    with open(sys.argv[1], "w") as f:
        json.dump(fingerprints(), f, indent=1, sort_keys=True)
        f.write("\n")
