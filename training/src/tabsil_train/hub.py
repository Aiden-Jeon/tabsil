"""Checkpoints for the training, evaluation and upload scripts. `load_model` reads a Hub repo, a
local `save_pretrained` folder, a checkpoint file or an MLflow run artifact (`runs:/...`) the same
way; `run_architecture` gives the model structure a training run logged as params.

    model, borders = load_model("runs:/<run_id>/checkpoints/latest.pt", "classification", out_dir)
    model, borders = load_model("hf://aiden-jeon/tabsil@stage2-10k", "regression", out_dir)
    structure = run_architecture(run.data.params)  # replace(config, **structure)

Hub repos are written as `hf://owner/name`, so a mistyped local path is not sent to the Hub and
a local folder named like the repo is not read in its place. Intermediate checkpoints live on
branches of the same repo (`stage1-150k`, `stage2-10k`) and are loaded as
`hf://owner/name@stage2-10k`. The Hub format (`save_pretrained`, `from_pretrained`) is in
`tabsil.hub`.
"""

from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path

import torch
from torch import nn

from tabsil import hub
from tabsil.models import (
    DEFAULT_MODEL_TYPE,
    canonical_model_type,
    config_fields,
    info_fields,
    model_from_state,
    state_config,
)
from tabsil_train.training.config import TrainConfig

HUB_PREFIX = "hf://"
# MLflow run artifacts
RUNS_PREFIX = "runs:/"
# file where `download_artifact` records the downloaded artifact (to skip re-downloading into the
# same folder)
DOWNLOADED_NAME = ".downloaded.json"


def _model_params(model_type: str) -> tuple[str, ...]:
    # the model-structure settings a training run logs as params (fields of both the model config
    # and TrainConfig). Fields that do not affect the weight shapes (`info_fields`) are not
    # structure and are left out
    train_fields = {f.name for f in fields(TrainConfig)} - set(info_fields(model_type))
    return tuple(name for name in config_fields(model_type) if name in train_fields)


def _parse_param(default, text: str):
    # MLflow params are strings. bool("False") is True, so bools are parsed separately
    if isinstance(default, bool):
        if text not in ("True", "False"):
            raise ValueError(f"not a bool param: {text}")
        return text == "True"
    return type(default)(text)


def download_artifact(uri: str, dst: str | Path) -> Path:
    """Download an MLflow artifact (`runs:/<run_id>/<path>`) under `dst` and return its path.

    If the same `uri` was fully downloaded into the same `dst` before, it is not downloaded again,
    so rerunning the upload script after a card edit does not fetch a large checkpoint again. The
    record is written only after the download finishes, so a partial download is never reused."""
    import mlflow

    dst = Path(dst)
    record = dst / DOWNLOADED_NAME
    if record.is_file():
        saved = json.loads(record.read_text())
        if saved.get("uri") == uri and Path(saved["path"]).exists():
            print(f"reusing the downloaded artifact: {saved['path']}")
            return Path(saved["path"])
    dst.mkdir(parents=True, exist_ok=True)
    record.unlink(missing_ok=True)
    path = Path(mlflow.artifacts.download_artifacts(artifact_uri=uri, dst_path=str(dst)))
    record.write_text(json.dumps({"uri": uri, "path": str(path)}) + "\n")
    return path


def load_model(
    spec: str | Path, task: str, dst: str | Path
) -> tuple[nn.Module, torch.Tensor | None]:
    """Load a checkpoint onto the CPU. `spec` is one of:

    - `runs:/<run_id>/<path>`: a file or folder a training run logged to MLflow (downloaded into
      `dst`; an earlier download in the same `dst` is reused, see `download_artifact`)
    - `hf://owner/name`: a Hub repo (uploaded with `save_pretrained`). Append `@<revision>` to
      load a branch, tag or commit (e.g. `hf://owner/name@step-100k`). It is downloaded from the
      Hub also when the working directory has a folder `owner/name`
    - a local `save_pretrained` folder, or a file saved with `tabsil.models.save_checkpoint`

    Checkpoints written during training (`checkpoints/step_XXXXXXX.pt`, `latest.pt`) build the
    model from the model type and config stored in them. A path that does not exist, a checkpoint
    without a model config and a model that does not serve `task` end the script (SystemExit)
    with a message naming `spec`.
    """
    spec = str(spec)
    if spec.startswith(HUB_PREFIX):
        repo_id, _, revision = spec.removeprefix(HUB_PREFIX).partition("@")
        loaded = hub.from_pretrained(repo_id, revision=revision or None, from_hub=True)
    else:
        path = download_artifact(spec, dst) if spec.startswith(RUNS_PREFIX) else Path(spec)
        if not path.exists():
            raise SystemExit(f"{spec} does not exist (pass a Hub repo as {HUB_PREFIX}owner/name).")
        if path.is_dir():
            loaded = hub.from_pretrained(path)
        else:
            state = torch.load(path, map_location="cpu", weights_only=True)
            if state_config(state) is None:
                raise SystemExit(
                    f"{spec} has no model config (saved before checkpoints recorded one)."
                )
            loaded = model_from_state(state)
    if task not in (served := hub.served_tasks(*loaded)):
        raise SystemExit(f"{spec} is a {'·'.join(served)} model (needs a {task} model).")
    return loaded


def run_architecture(params: dict[str, str]) -> dict:
    """The model structure from a training run's MLflow params: `model_type` and the model's
    structure settings (TrainConfig fields).

    Values the run did not record use the config defaults. Apply the result to a training config
    with `replace(config, **...)`.
    """
    model_type = canonical_model_type(params.get("model_type", DEFAULT_MODEL_TYPE))
    defaults = TrainConfig(model_type=model_type)
    names = _model_params(model_type)
    return {
        "model_type": model_type,
        **{
            name: _parse_param(
                getattr(defaults, name), params.get(name, str(getattr(defaults, name)))
            )
            for name in names
        },
    }


def weights_digest(model: nn.Module, borders: torch.Tensor | None = None) -> str:
    """SHA-1 of a model's weights (every state_dict tensor's name, dtype, shape and bytes, in
    name order) and of the regression borders: two checkpoints at different paths with the same
    weights get the same digest, and new weights at an old path a new one."""
    import hashlib

    digest = hashlib.sha1()
    tensors = sorted(model.state_dict().items())
    if borders is not None:
        tensors.append(("<borders>", torch.as_tensor(borders)))
    for name, t in tensors:
        t = t.detach().cpu().contiguous()
        digest.update(f"{name}|{t.dtype}|{tuple(t.shape)}|".encode())
        digest.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()
