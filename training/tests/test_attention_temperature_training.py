"""Attention temperatures with the training stack: the log-scale parameters get their own
training metric groups, and the curriculum stage 1 code path trains them (the temperatures
themselves: tests/test_attention_temperature.py)."""

from dataclasses import replace

import numpy as np
import torch
import train_helpers as helpers

from tabsil_train import recipe
from tabsil_train.training.diagnostics import parameter_groups
from tabsil_train.training.loop import train


def test_diagnostics_groups_log_scale_separately():
    model = helpers.tabsil_smoke().build_model()
    groups = parameter_groups(model, {id(p) for p in model.parameters()})
    assert {
        "row_icl_log_scale/muon",
        "row_stage_log_scale/muon",
        "output_log_scale_classification/muon",
    } <= set(groups)
    assert not any("qk_norm" in g or "qassmax" in g for g in groups)


def test_curriculum_smoke_trains(tmp_path, monkeypatch):
    # Stage 1 code path at small size (all parameters Muon)
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    config = helpers.tabsil_smoke()
    stage = {k: v for k, v in recipe.TRAINING.items() if k not in ("batch_size", "max_features")}
    small = dict(
        max_features=8,
        min_rows=48,
        max_rows=80,
        log_uniform_rows=True,
        max_cells=80 * 8,
        steps=3,
        warmup_steps=1,
        val_every=3,
        log_every=1,
        checkpoint_every=3,
        keep_every=None,
        collapse_threshold=None,
        border_datasets=20,
        lr=8e-4,
    )
    result = train(replace(config, **{**stage, **small}), tmp_path, mlflow_log=False)
    assert all(np.isfinite(result["history"]["train_loss"]))
    state = torch.load(tmp_path / "latest.pt", weights_only=True)
    assert any(k.endswith("log_scale.head_scale") for k in state["model"])
