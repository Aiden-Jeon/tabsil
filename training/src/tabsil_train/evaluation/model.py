"""The model the ADBench, causal and impute runs load for their TabSil methods: one multitask
checkpoint (`--regressor-path`) that serves regression and classification."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

from tabsil.recipe import default_inference
from tabsil_train.benchmarks import _files
from tabsil_train.hub import load_model


def suite_model(
    args: argparse.Namespace,
    path: str | None,
    out_dir: Path,
    suite: str,
    classification_use: str,
    **options,
) -> tuple[nn.Module, torch.Tensor, int, str]:
    """(model, borders, member count, settings string) of a suite's TabSil methods.

    `path` is the checkpoint the command line gives the suite. The script ends if there is none
    (`suite` names the suite in the message) or if the model does not serve classification,
    which the suite needs for `classification_use`. The member count is `--n-estimators`, or the
    model's for regression. The settings string is the one written with the methods' rows
    (`_files.model_settings`, with the suite's `options`)."""
    if not path:
        raise SystemExit(f"{suite} needs --regressor-path (a model serving both tasks).")
    model, borders = load_model(path, "regression", out_dir / "checkpoints" / "regression")
    if "classification" not in model.tasks:
        raise SystemExit(f"{path} does not serve classification ({classification_use}).")
    n_estimators = args.n_estimators or default_inference(model, "regression")["n_estimators"]
    settings = _files.model_settings(model, borders, n_estimators, args.seed, **options)
    return model, borders, n_estimators, settings
