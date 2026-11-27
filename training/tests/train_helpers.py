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


def tiny_checkpoint(path: str | Path, seed: int = 0) -> str:
    """Save `tiny_model(seed)` and regression borders as a checkpoint file (one model serving
    both tasks, as `eval.py --regressor-path` takes); returns the path."""
    from tabsil.models import save_checkpoint

    save_checkpoint(path, tiny_model(seed), torch.linspace(-4, 4, 33))
    return str(path)


def in_process_workers(script, monkeypatch) -> list[list[str]]:
    """Run the worker processes of `eval.py --shards` (ADBench, impute) in this process: starting
    `python eval.py <argv>` calls `script.main(argv)` instead, and an exception is exit code 1
    (written to the worker's log). The queue's waits are skipped. Returns the list the started
    workers' argv are appended to."""
    import subprocess
    import time

    started: list[list[str]] = []

    class Worker:
        def __init__(self, cmd, stdout, stderr, env):
            argv = list(cmd[2:])  # after the interpreter and the script
            started.append(argv)
            try:
                script.main(argv)
                self.returncode = 0
            except (Exception, SystemExit) as error:
                print(repr(error), file=stdout)
                self.returncode = 1

        def poll(self):
            return self.returncode

    monkeypatch.setattr(subprocess, "Popen", Worker)
    monkeypatch.setattr(time, "sleep", lambda seconds: None)
    return started


def load_script(relative_path: str, module_name: str):
    """Import a script of the tabsil-train package (e.g. `scripts/train.py`, relative to
    `training/`) as a module."""
    path = TRAINING_DIR / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
