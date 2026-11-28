"""Shared helpers of the v2 prior diagnostics (`README.md` in this folder)."""

from __future__ import annotations

import functools
import os

import numpy as np
import torch

DEVICE = os.environ.get("DIAG_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
# Output folder of the CSV files (run_all.py logs it to the MLflow run)
OUT = os.environ.get("DIAG_OUT", "/tmp/v2_prior_diag")
os.makedirs(OUT, exist_ok=True)
# The weights (and, through their config, the inference settings) the numbers are measured with
MODEL = os.environ.get("DIAG_MODEL", "aiden-jeon/tabsil@v1.1")


# `MODEL` specs that the training package's loader reads (`tabsil_train.hub.load_model`)
TRAINING_SPECS = ("runs:/", "hf://")


@functools.cache
def load_model() -> tuple[torch.nn.Module, torch.Tensor]:
    """(model, regression borders) of `MODEL`, loaded once per process. `MODEL` is a Hub repo id
    or a local folder (`tabsil.hub`), or a file a training run logged to MLflow
    (`runs:/<run id>/model/final.pt`), a Hub repo as `hf://owner/name@revision` or a `.pt` file:
    those go through `tabsil_train.hub.load_model`, which needs `training/src` on the path. A
    download is kept under `OUT/checkpoints` and reused by the next diagnostic of the job."""
    if MODEL.startswith(TRAINING_SPECS) or MODEL.endswith(".pt"):
        from tabsil_train.hub import load_model as load_spec

        model, borders = load_spec(MODEL, "regression", os.path.join(OUT, "checkpoints"))
    else:
        from tabsil.hub import model_or_pretrained

        model, borders = model_or_pretrained(MODEL, MODEL, "regression")
    if "classification" not in model.tasks:
        raise ValueError(f"{MODEL} does not serve classification")
    return model, borders


def log_density(log_probs: torch.Tensor, borders: torch.Tensor, y: np.ndarray) -> np.ndarray:
    """Per-row log density of y under `predict_distribution`'s output (log probabilities on
    borders in original units). As that docstring says, the outer buckets are half-normal tails
    (`FullSupportBarDistribution`); the inner buckets are uniform."""
    from tabsil.models.bar_distribution import FullSupportBarDistribution

    dist = FullSupportBarDistribution(borders.double().cpu())
    nll = dist(log_probs.double().cpu(), torch.as_tensor(y, dtype=torch.float64))
    return -nll.numpy()
