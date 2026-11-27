"""`tabsil_train.recipe`: the curriculum values and the structure they train (the inference
defaults: tests/test_recipe.py)."""

from dataclasses import fields

import pytest

from tabsil_train import recipe
from tabsil_train.training.config import TrainConfig

# Every TrainConfig field of curriculum stages 1-3, written out. Fields that are the same in
# every stage:
EXPECTED_COMMON = {
    "prior_type": "graph_scm",
    "model_type": "column_row_transformer",
    "emsize": 256,
    "nlayers": 24,
    "nhead": 16,
    "features_per_group": 3,
    "n_inducing": 256,
    "n_cls": 4,
    "column_blocks": 6,
    "row_blocks": 6,
    "column_heads": 4,
    "row_heads": 4,
    "max_features": 100,
    "binary_prob": 0.5,
    "filter_unpredictable": True,
    "target_train_rows": False,
    "missing_dataset_prob": 0.3,
    "border_datasets": 20_000,
    "batch_size": 256,
    "stop_step": None,
    "deadline": None,
    "seed": 0,
    "num_workers": 8,
    "prefetch_factor": 4,
    "gc_freeze": True,
    "log_every": 100,
    "val_batch": 8,
    "reference_val_tables": 0,
    "collapse_threshold": 0.5,
    "collapse_patience": 3,
    "snapshot_every": 2000,
    "max_rewinds": 10,
    "rewind_lr_factor": 0.5,
    "rewind_min_lr": 2.5e-05,
    "compile": False,
    "tf32": False,
    "weight_average": 0,
    "weight_average_every": 500,
    "muon_weight_decay": 0.01,
    "muon_momentum": 0.9,
    "optimizer": "muon",
    "muon_distributed": "off",
    "phases": (),
    "grad_cosine": False,
    "rank_timing": False,
}
# Fields that differ between the stages
EXPECTED_STAGES = {
    1: {
        "log_uniform_rows": False,
        "max_cells": 102_400,
        "min_rows": 1024,
        "max_rows": 1024,
        "min_train_fraction": 0.3,
        "max_train_fraction": 0.9,
        "micro_batch": 4,
        "steps": 250_000,
        "warmup_steps": 2500,
        "lr": 0.0008,
        "grad_clip": 10.0,
        "val_every": 5000,
        "checkpoint_every": 5000,
        "keep_every": 25_000,
        "activation_checkpointing_rows": None,
    },
    2: {
        "log_uniform_rows": True,
        "max_cells": 1_024_000,
        "min_rows": 400,
        "max_rows": 10_240,
        "min_train_fraction": 0.79,
        "max_train_fraction": 0.81,
        "micro_batch": 1,
        "steps": 20_000,
        "warmup_steps": 200,
        "lr": 0.0001,
        "grad_clip": 10.0,
        "val_every": 5000,
        "checkpoint_every": 1000,
        "keep_every": 5000,
        "activation_checkpointing_rows": 2048,
    },
    3: {
        "log_uniform_rows": True,
        "max_cells": 6_000_000,
        "min_rows": 400,
        "max_rows": 60_000,
        "min_train_fraction": 0.79,
        "max_train_fraction": 0.81,
        "micro_batch": 1,
        "steps": 5000,
        "warmup_steps": 50,
        "lr": 2e-05,
        "grad_clip": 1.0,
        "val_every": 1000,
        "checkpoint_every": 1000,
        "keep_every": 1000,
        "activation_checkpointing_rows": 2048,
    },
}


@pytest.mark.parametrize("stage", [1, 2, 3])
def test_curriculum_sets_every_field(stage):
    config = recipe.curriculum(stage)
    expected = {**EXPECTED_COMMON, **EXPECTED_STAGES[stage]}
    names = {f.name for f in fields(TrainConfig)}
    assert set(expected) == names  # the expected values cover every field
    got = {name: getattr(config, name) for name in names}
    assert {k: v for k, v in got.items() if v != expected[k]} == {}


def test_curriculum_steps_set_the_warmup():
    config = recipe.curriculum(2, steps=1_000)
    assert (config.steps, config.warmup_steps) == (1_000, 10)
    with pytest.raises(ValueError, match="stage"):
        recipe.curriculum(4)
    with pytest.raises(ValueError, match="step"):
        recipe.curriculum(1, steps=0)


def test_curriculum_has_the_structure():
    for stage in recipe.CURRICULUM:
        config = recipe.curriculum(stage)
        assert {k: getattr(config, k) for k in recipe.STRUCTURE} == recipe.STRUCTURE


def test_default_config_is_stage_1():
    assert TrainConfig() == recipe.curriculum(1)


def test_smoke_and_medium_sizes():
    smoke, stage1 = TrainConfig.smoke(), recipe.curriculum(1)
    assert (smoke.emsize, smoke.nlayers, smoke.nhead, smoke.n_inducing) == (32, 2, 2, 8)
    assert smoke.column_blocks == smoke.row_blocks == 2
    # the rest of the structure and the training setup are stage 1's
    for name in ("features_per_group", "n_cls", "column_heads", "row_heads", "binary_prob"):
        assert getattr(smoke, name) == getattr(stage1, name)
    assert (smoke.muon_weight_decay, smoke.muon_momentum) == (0.01, 0.9)
    smoke.build_model()  # builds at smoke size
    medium = TrainConfig.medium()
    assert {k: getattr(medium, k) for k in recipe.STRUCTURE} == recipe.STRUCTURE
    assert (medium.steps, medium.warmup_steps) == (20_000, 1_000)
