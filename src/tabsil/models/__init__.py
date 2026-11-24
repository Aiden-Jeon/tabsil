"""Models (model axis) and the model registry.

- `column_row_transformer`: column (inducing points) -> row (CLS) -> row ICL, the TabSil backbone
- `bar_distribution`: predictive distribution that treats regression as bucket classification
- `quantile_distribution`: regression quantile head (TabSil's regression output)

TabSil's structure values are in `tabsil.recipe`. `randomize_zero_weights` (for following the
forward path with an untrained model in notebooks and tests) is also here.

The shared training loop, checkpoint and Hub code build models through the registry below rather
than calling the model class directly. A registry model class provides:

- class attributes `model_type` (its name in `MODEL_TYPES` below) and `config_class` (a
  dataclass with an `n_out` field and `to_dict()`)
- `__init__(config)`, `.config`
- `forward(x, y_train, task)`: x (B, R, F) holds the train rows followed by the test rows,
  y_train (B, n_train); `task` ("classification" or "regression") picks the output -> test-row
  outputs (B, R - n_train, outputs of that task)
- `num_parameters()`
- `decoder_weight()`: the output layer weight (training metric `decoder_norm`), or None
- (optional) `repeated_blocks()`: the repeated blocks. The `compile` speed-up option compiles each
  of them (without it, the whole model is compiled)
- (optional) the config class's `from_dict(values)`: the config of a saved dict (reads older
  formats, `config_from_dict`)
- (optional) the config class's `INFO_FIELDS`: fields unrelated to the weight shapes (e.g. the
  feature count seen in training). They are not compared when resuming and are not logged as part
  of the model structure of a training run (`info_fields`)
"""

from __future__ import annotations

import importlib
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

# Name -> "module:Class" (imported on first use) or the class itself
MODEL_TYPES: dict[str, str | type[nn.Module]] = {
    "column_row_transformer": "tabsil.models.column_row_transformer:ColumnRowTransformer",
}
# Old name -> current name, accepted when reading checkpoints, run params and Hub config.json files
MODEL_TYPE_ALIASES: dict[str, str] = {}
# The model type used when a checkpoint, run or Hub repo does not record one
DEFAULT_MODEL_TYPE = "column_row_transformer"


def canonical_model_type(model_type: str) -> str:
    """Map an old name to the current one. Unknown names are returned as is (`model_class`
    raises the ValueError)."""
    return MODEL_TYPE_ALIASES.get(model_type, model_type)


def model_class(model_type: str) -> type[nn.Module]:
    name = canonical_model_type(model_type)
    if name not in MODEL_TYPES:
        raise ValueError(f"model_type must be one of {tuple(MODEL_TYPES)}: {model_type}")
    target = MODEL_TYPES[name]
    if isinstance(target, str):
        module, name = target.split(":")
        target = getattr(importlib.import_module(module), name)
    return target


def config_fields(model_type: str) -> tuple[str, ...]:
    """The field names of the model's config."""
    return tuple(f.name for f in fields(model_class(model_type).config_class))


def info_fields(model_type: str) -> tuple[str, ...]:
    """The model config fields unrelated to the weight shapes (the config class's `INFO_FIELDS`).
    They are not compared when resuming from a checkpoint and are not logged as part of the model
    structure of a training run."""
    return tuple(getattr(model_class(model_type).config_class, "INFO_FIELDS", ()))


def config_dict(config: Any) -> dict:
    """The model config as the dict saved in checkpoints and Hub config.json files."""
    return config.to_dict()


def config_from_dict(model_type: str, values: dict) -> Any:
    """The model config of a saved dict. Missing fields take the defaults. The config class's
    `from_dict` reads it if there is one (older formats); otherwise a field this code does not know
    (saved by newer code or another branch) raises a ValueError naming it."""
    cls = model_class(model_type).config_class
    if hasattr(cls, "from_dict"):
        return cls.from_dict(values)
    unknown = sorted(set(values) - {f.name for f in fields(cls)})
    if unknown:
        raise ValueError(
            f"the {model_type} model config has fields this code does not know: {unknown}"
            " (the checkpoint may have been saved by newer code or another branch)"
        )
    return cls(**values)


def build_model(model_type: str, config: Any) -> nn.Module:
    """`config` is the model's config dataclass or a dict with the same contents."""
    if not is_dataclass(config):
        config = config_from_dict(model_type, config)
    return model_class(model_type)(config)


def state_config(state: dict) -> dict | None:
    """The model config stored in a checkpoint (None if absent). `save_checkpoint` files keep it
    under `config`, training checkpoints under `model_config`. Older code treats any state with
    `config` as a `save_checkpoint` file and looks for `state_dict`, so training checkpoints use a
    different key and older code still reads them through the run params."""
    return state.get("model_config", state.get("config"))


def state_weights(state: dict) -> dict:
    """The checkpoint weights: `state_dict` in `save_checkpoint` files, `model` in training
    checkpoints."""
    return state["state_dict"] if "state_dict" in state else state["model"]


def model_from_state(state: dict) -> tuple[nn.Module, torch.Tensor | None]:
    """(model, regression bucket borders) from a checkpoint holding a model config
    (`state_config`). Reads both `save_checkpoint` files and training checkpoints. The model type
    is the `model_type` stored in the file (without one, `DEFAULT_MODEL_TYPE`)."""
    config = state_config(state)
    if config is None:
        raise ValueError(
            "the checkpoint has no model config (a training checkpoint older than stored configs)"
        )
    model = build_model(state.get("model_type", DEFAULT_MODEL_TYPE), config)
    model.load_state_dict(state_weights(state))
    return model.eval(), state["borders"]


def load_checkpoint(
    path: str | Path, map_location: str | torch.device = "cpu"
) -> tuple[nn.Module, torch.Tensor | None]:
    """Load a `save_checkpoint` file, or a training checkpoint that stores the model config."""
    return model_from_state(torch.load(path, map_location=map_location, weights_only=True))


def save_checkpoint(
    path: str | Path, model: nn.Module, borders: torch.Tensor | None = None
) -> None:
    """Save the weights, the config and (for regression) the bar distribution borders."""
    payload = {
        "model_type": model.model_type,
        "config": config_dict(model.config),
        "state_dict": model.state_dict(),
        "borders": borders,
    }
    torch.save(payload, path)


def randomize_zero_weights(model: nn.Module, std: float = 0.3) -> nn.Module:
    """Fill the zero-initialized weight matrices (the last Linear of residual branches, attention
    output projections, ...) with N(0, std²). At init those branches add nothing to the output, so
    they must be filled to follow the path from input to output in an untrained model (notebooks,
    tests). 1-D parameters such as biases and norms are left as they are. Uses the global torch RNG
    (seed it with `torch.manual_seed` beforehand)."""
    with torch.no_grad():
        for p in model.parameters():
            if p.dim() > 1 and (p == 0).all():
                nn.init.normal_(p, std=std)
    return model
