"""Small configs and models shared by the training tests."""

from __future__ import annotations

import importlib.util
from dataclasses import replace
from pathlib import Path

import torch

from tabsil.models import randomize_zero_weights
from tabsil.models.column_row_transformer import ColumnRowTransformer, ColumnRowTransformerConfig
from tabsil_train.training.config import TrainConfig

# training/ (the tabsil-train package folder: scripts/, hub/)
TRAINING_DIR = Path(__file__).resolve().parents[1]


def tabsil_smoke(**changes) -> TrainConfig:
    """The smoke-size TabSil training config (what `scripts/train.py --config-name smoke` uses)."""
    return replace(TrainConfig.smoke(), **changes)


def tiny_model(seed: int = 0, **changes) -> ColumnRowTransformer:
    """A tiny randomized `ColumnRowTransformer` (`ColumnRowTransformerConfig.tiny`) in eval mode."""
    torch.manual_seed(seed)
    model = ColumnRowTransformer(ColumnRowTransformerConfig.tiny(**changes))
    return randomize_zero_weights(model).eval()


def load_script(relative_path: str, module_name: str):
    """Import a script of the tabsil-train package (e.g. `scripts/train.py`, relative to
    `training/`) as a module."""
    path = TRAINING_DIR / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
