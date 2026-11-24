"""Upload trained models (`tabsil.models` registry models) to the Hugging Face Hub and download
them (`tabsil.from_pretrained` is `from_pretrained` here).

On the Hub the weights are in `model.safetensors` and the config in `config.json`, so loading
needs no pickle.
The regression borders (the buckets the quantile outputs are turned into) go into
`model.safetensors` too, under the name `borders`. config.json files and checkpoints written by
minipfn for the same structure are read as well (`ColumnRowTransformerConfig.from_dict`).

    from tabsil import from_pretrained
    model, borders = from_pretrained("aiden-jeon/tabsil")
    model, borders = from_pretrained("aiden-jeon/tabsil@step-100k")

For inference alone this function is not needed: `TabSilClassifier()`·`TabSilRegressor()`
(`tabsil.recipe`) download the published weights (`model_or_pretrained`).

`load_model` reads a Hub repo, a local `save_pretrained` folder or a checkpoint file the same way
(the evaluation and upload scripts use it through `tabsil_train.hub.load_model`, which also reads
MLflow run artifacts, `runs:/...`). Hub repos are written as `hf://owner/name`, so a mistyped local
path is not sent to the Hub. Intermediate checkpoints live on branches of the same repo
(`step-100k`, ...) and are loaded as `hf://owner/name@step-100k`.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file, save_file
from torch import nn

from tabsil.models import (
    DEFAULT_MODEL_TYPE,
    build_model,
    config_dict,
    model_from_state,
    state_config,
)

WEIGHTS_NAME = "model.safetensors"
CONFIG_NAME = "config.json"
BORDERS_KEY = "borders"
HUB_PREFIX = "hf://"
# MLflow run artifacts, read by `tabsil_train.hub.load_model`
RUNS_PREFIX = "runs:/"


def _is_local_path(repo_id_or_path: str | Path) -> bool:
    """Whether the argument is a local path rather than a Hub repo id (`owner/name`)."""
    if isinstance(repo_id_or_path, Path) or Path(repo_id_or_path).exists():
        return True
    s = str(repo_id_or_path)
    return s.count("/") != 1 or s.startswith((".", "/", "~")) or s.endswith((".pt", ".json"))


def save_pretrained(
    model: nn.Module, borders: torch.Tensor | None, out_dir: str | Path, **extra
) -> Path:
    """Write `model.safetensors` and `config.json` to `out_dir`. `extra` adds keys to config.json
    (`inference`: the inference settings for these weights)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tensors = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    if borders is not None:
        tensors[BORDERS_KEY] = borders.detach().cpu().contiguous()
    save_file(tensors, out_dir / WEIGHTS_NAME)
    # the model serves both tasks with one set of weights
    config = {
        "task": "multitask",
        "model_type": model.model_type,
        "model": config_dict(model.config),
        **extra,
    }
    (out_dir / CONFIG_NAME).write_text(json.dumps(config, indent=2) + "\n")
    return out_dir


def from_pretrained(
    repo_id_or_path: str | Path, revision: str | None = None
) -> tuple[nn.Module, torch.Tensor | None]:
    """Load (model, regression borders) from a Hub repo id or a local `save_pretrained` folder.
    Pass an intermediate checkpoint branch as `revision` or as `owner/name@step-100k`."""
    if _is_local_path(repo_id_or_path):
        local = Path(repo_id_or_path).expanduser()
        weights_path, config_path = local / WEIGHTS_NAME, local / CONFIG_NAME
    else:
        repo_id, _, at = str(repo_id_or_path).partition("@")
        if at:
            if revision is not None and revision != at:
                raise ValueError(f"revision given twice: {repo_id_or_path}, {revision}")
            revision = at
        weights_path = hf_hub_download(repo_id, WEIGHTS_NAME, revision=revision)
        config_path = hf_hub_download(repo_id, CONFIG_NAME, revision=revision)
    config = json.loads(Path(config_path).read_text())
    tensors = load_file(weights_path)
    borders = tensors.pop(BORDERS_KEY, None)
    model = build_model(config.get("model_type", DEFAULT_MODEL_TYPE), config["model"])
    model.load_state_dict(tensors)
    if "inference" in config:
        # inference settings for these weights (number of members, member preprocessing kinds);
        # the recipe's inference classes read them as defaults (`tabsil.recipe.default_inference`)
        model.inference_config = config["inference"]
    return model.eval(), borders


def model_or_pretrained(
    model: nn.Module | str | Path | None, default_repo: str, task: str
) -> tuple[nn.Module, torch.Tensor | None]:
    """Resolve the `model` an inference class receives into (model, regression borders).

    A module gives (that module, None); a Hub repo id (`owner/name`, `owner/name@branch`) or a
    local folder is loaded from there; None loads `default_repo` (the published weights).
    ValueError if the loaded model does not serve `task`."""
    if isinstance(model, nn.Module):
        return model, None
    spec = default_repo if model is None else model
    loaded = from_pretrained(spec)
    if task not in (served := served_tasks(*loaded)):
        raise ValueError(f"{spec} is a {'·'.join(served)} model (needs a {task} model)")
    return loaded


def served_tasks(model: nn.Module, borders: torch.Tensor | None) -> tuple[str, ...]:
    """The tasks the model serves (`model.tasks`; regression only with the regression borders)."""
    return tuple(t for t in model.tasks if t != "regression" or borders is not None)


def load_model(
    spec: str | Path, task: str, dst: str | Path | None = None, *, name: str | None = None
) -> tuple[nn.Module, torch.Tensor | None]:
    """Load a checkpoint onto the CPU. `spec` is one of:

    - `hf://owner/name`: a Hub repo (uploaded with `save_pretrained`). Append `@<revision>` to
      load a branch, tag or commit (e.g. `hf://owner/name@step-100k`)
    - a local `save_pretrained` folder, or a file saved with `tabsil.models.save_checkpoint`

    Checkpoints written during training (`checkpoints/step_XXXXXXX.pt`, `latest.pt`) build the
    model from the model type and config stored in them. MLflow run artifacts (`runs:/...`) are
    read by `tabsil_train.hub.load_model` (the `tabsil-train` package). Messages name the
    checkpoint `name` (default: `spec`). `dst` (the download folder of MLflow artifacts) is not used
    here; it is kept so that calls written for the earlier `load_model(spec, task, dst)` still work.
    """
    spec = str(spec)
    name = spec if name is None else name
    if spec.startswith(HUB_PREFIX):
        repo_id, _, revision = spec.removeprefix(HUB_PREFIX).partition("@")
        return _check_task(from_pretrained(repo_id, revision=revision or None), task, name)
    if spec.startswith(RUNS_PREFIX):
        raise SystemExit(
            f"{name} is an MLflow run artifact: load it with `tabsil_train.hub.load_model`"
            " (the tabsil-train package)."
        )
    path = Path(spec)
    if not path.exists():
        raise SystemExit(f"{name} does not exist (pass a Hub repo as {HUB_PREFIX}owner/name).")
    if path.is_dir():
        return _check_task(from_pretrained(path), task, name)
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state_config(state) is None:
        raise SystemExit(f"{name} has no model config (saved before checkpoints recorded one).")
    return _check_task(model_from_state(state), task, name)


def _check_task(loaded: tuple[nn.Module, torch.Tensor | None], task: str, spec: str):
    if task not in (served := served_tasks(*loaded)):
        raise SystemExit(f"{spec} is a {'·'.join(served)} model (needs a {task} model).")
    return loaded
