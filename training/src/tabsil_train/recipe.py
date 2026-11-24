"""TabSil training recipe: the TabICLv2 3-stage curriculum (the model structure and the inference
defaults are in `tabsil.recipe`; `NAME` and `STRUCTURE` are re-exported here).

Training (`curriculum`): the TabICLv2 3-stage curriculum (arXiv 2602.11139 and its public training
scripts) with Muon on all parameters, batch 256 on 32 GPUs, 1% warmup and cosine decay per stage:

    stage  rows (train share)        steps  lr
    1      1,024 (30-90%)            250K   8e-4
    2      400-10,240 (80%)           20K   1e-4   starts from the stage 1 weights
    3      400-60,000 (80%)            5K   2e-5   starts from the stage 2 weights

Tables have at most 100 features (fewer for long tables, by the TabICLv2 table size rule). The
prior is the `graph_scm` prior (the TabICLv2 graph SCM from the `tabicl` package,
`tabsil_train.priors.graph_scm`).

The training parts listed above (Muon on every parameter, the TabICLv2 table size rule) are fixed
in the code (`tabsil_train.training`, `tabsil_train.priors`); this module holds the numbers.

    config = curriculum(1)  # stage 2 and 3 start from the previous stage's weights (--init-run-id)
"""

from __future__ import annotations

from tabsil.recipe import MODEL_TYPE, NAME, STRUCTURE
from tabsil_train.training.config import TrainConfig

__all__ = [
    "ACTIVATION_CHECKPOINTING_ROWS",
    "CURRICULUM",
    "CURRICULUM_WARMUP",
    "MODEL_TYPE",
    "NAME",
    "STRUCTURE",
    "TRAINING",
    "curriculum",
]

# ---------------------------------------------------------------------------
# Training: TabICLv2 3-stage curriculum
# ---------------------------------------------------------------------------
# The stages follow the public TabICLv2 training scripts (soda-inria/tabicl
# `scripts/train_v2_clf_stage{1,2,3}.sh`, arXiv 2602.11139 Section 4.1 and Appendix B.1): each stage
# takes only the weights of the previous one (`init_from`) and starts a new optimizer with a 1%
# warmup followed by cosine decay. Values that differ from the scripts:
# - batch 256 (32 GPUs) for one multitask model, with step counts chosen so that each task sees the
#   same number of datasets as the scripts' batch 64: 250K x 256 / 2 = 500K x 64 (stage 2 and 3
#   likewise 20K and 5K)
# - lr 8e-4 / 1e-4 / 2e-5 (the stage 1 script value, the later stages at the same ratio)
# - micro batch 4 / 1 / 1: the wide column/row stages do not fit 8 tables of 1,024 rows x 100
#   features on one H100 80 GB
# - activation checkpointing for tables with more than 2,048 rows in stage 2 and 3 (the largest
#   stage 2 table, 10,240 rows, does not fit one H100 without it)
# - rewind lr factor 0.5 when a collapse is rolled back
TRAINING = {
    "prior_type": "graph_scm",
    "batch_size": 256,
    "muon_momentum": 0.9,
    "muon_weight_decay": 0.01,
    "max_features": 100,
    "binary_prob": 0.5,
    "filter_unpredictable": True,
    "rewind_lr_factor": 0.5,
    "phases": (),
}
ACTIVATION_CHECKPOINTING_ROWS = 2_048
CURRICULUM = {
    1: {
        "steps": 250_000,
        "lr": 8e-4,
        "grad_clip": 10.0,
        "min_rows": 1_024,
        "max_rows": 1_024,
        "min_train_fraction": 0.3,
        "max_train_fraction": 0.9,
        "log_uniform_rows": False,
        "max_cells": 1_024 * 100,
        "micro_batch": 4,
    },
    2: {
        "steps": 20_000,
        "lr": 1e-4,
        "grad_clip": 10.0,
        "min_rows": 400,
        "max_rows": 10_240,
        "min_train_fraction": 0.79,
        "max_train_fraction": 0.81,
        "log_uniform_rows": True,
        "max_cells": 10_240 * 100,
        "micro_batch": 1,
        "checkpoint_every": 1_000,
        "keep_every": 5_000,
        "activation_checkpointing_rows": ACTIVATION_CHECKPOINTING_ROWS,
    },
    3: {
        "steps": 5_000,
        "lr": 2e-5,
        "grad_clip": 1.0,
        "min_rows": 400,
        "max_rows": 60_000,
        "min_train_fraction": 0.79,
        "max_train_fraction": 0.81,
        "log_uniform_rows": True,
        "max_cells": 60_000 * 100,
        "micro_batch": 1,
        "val_every": 1_000,
        "checkpoint_every": 1_000,
        "keep_every": 1_000,
        "activation_checkpointing_rows": ACTIVATION_CHECKPOINTING_ROWS,
    },
}
# Warmup is 1% of the stage length (the scripts' `--warmup_proportion 0.01`)
CURRICULUM_WARMUP = 0.01


def curriculum(stage: int, steps: int | None = None) -> TrainConfig:
    """The training config of curriculum `stage` (1, 2 or 3). Stage 2 and 3 start from the previous
    stage's checkpoint weights (the training script enforces it). With `steps`, the stage trains for
    that many steps and the warmup (1%) and cosine schedule follow it."""
    if stage not in CURRICULUM:
        raise ValueError(f"{NAME} curriculum stage is one of {tuple(CURRICULUM)}: {stage}")
    values = {**STRUCTURE, **TRAINING, **CURRICULUM[stage]}
    if steps is not None:
        if steps < 1:
            raise ValueError(f"stage {stage} needs at least one step: {steps}")
        values["steps"] = steps
    values["warmup_steps"] = int(CURRICULUM_WARMUP * values["steps"])
    return TrainConfig(**values)
