"""Small models shared by the tests."""

from __future__ import annotations

import torch

from tabsil.models import randomize_zero_weights
from tabsil.models.column_row_transformer import ColumnRowTransformer, ColumnRowTransformerConfig


def tiny_model(seed: int = 0, **changes) -> ColumnRowTransformer:
    """A tiny randomized `ColumnRowTransformer` (`ColumnRowTransformerConfig.tiny`) in eval mode."""
    torch.manual_seed(seed)
    model = ColumnRowTransformer(ColumnRowTransformerConfig.tiny(**changes))
    return randomize_zero_weights(model).eval()


def smoke_model_config(**changes) -> ColumnRowTransformerConfig:
    """The model config of the smoke-size TabSil training config (tabsil-train
    `TrainConfig.smoke().model_config()`, checked there): width 32, 2 row ICL layers, 2 heads,
    8 inducing points, 2 column/row stage pairs, at most 8 features, class cap 10."""
    values = {
        "emsize": 32,
        "nlayers": 2,
        "nhead": 2,
        "n_inducing": 8,
        "column_blocks": 2,
        "row_blocks": 2,
        "max_features": 8,
        "n_out": 10,
    }
    return ColumnRowTransformerConfig(**(values | changes))


def smoke_model(**changes) -> ColumnRowTransformer:
    """A freshly initialized smoke-size model (what `TrainConfig.smoke().build_model()` builds),
    in train mode like a built model. Seed the torch RNG beforehand for fixed weights."""
    return ColumnRowTransformer(smoke_model_config(**changes))
