"""Activation checkpointing: `ColumnRowTransformer.activation_checkpointing_rows` and the
training setting `TrainConfig.activation_checkpointing_rows`. Enabling it keeps loss and
gradients the same, and it is not used for tables below the threshold, validation or inference."""

import inspect
from dataclasses import replace

import helpers
import numpy as np
import pytest
import torch

from tabsil import recipe
from tabsil.models import column_row_transformer as M
from tabsil.training import loop
from tabsil.training.config import TrainConfig
from tabsil.training.loop import train


def _model(**changes) -> M.ColumnRowTransformer:
    # the TabSil structure at tiny size
    torch.manual_seed(0)
    return M.ColumnRowTransformer(M.ColumnRowTransformerConfig.tiny(**changes)).train()


def _batch(task: str, b: int = 2, r: int = 40, c: int = 5, n_train: int = 30):
    g = torch.Generator().manual_seed(1)
    x = torch.randn(b, r, c, generator=g)
    if task == "classification":
        return x, torch.randint(0, 3, (b, n_train), generator=g).float()
    return x, torch.randn(b, n_train, generator=g)


def _grads(model, task, rows):
    model.activation_checkpointing_rows = rows
    model.zero_grad(set_to_none=True)
    x, y = _batch(task)
    loss = model(x, y, task).square().mean()
    loss.backward()
    grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    return loss.detach(), grads


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_checkpointing_same_gradients(task, monkeypatch):
    # float32 CPU: checkpointing recomputes the same forward, so loss and gradients are
    # bit-identical (classification decoder path that builds the score matrix). The default
    # decoder without a score matrix only differs in op order, so it is close
    model = _model()
    loss, grads = _grads(model, task, None)
    monkeypatch.setattr(M, "CHECKPOINT_FUSED_DECODER", False)
    same_loss, same = _grads(model, task, 0)
    assert torch.equal(loss, same_loss) and same.keys() == grads.keys()
    assert all(torch.equal(grads[n], same[n]) for n in grads)
    monkeypatch.setattr(M, "CHECKPOINT_FUSED_DECODER", True)
    close_loss, close = _grads(model, task, 0)
    assert close.keys() == grads.keys()
    assert torch.allclose(loss, close_loss, rtol=1e-5)
    flat = torch.cat([g.flatten() for g in grads.values()])
    diff = torch.cat([(close[n] - grads[n]).flatten() for n in grads])
    assert diff.norm() <= 1e-5 * flat.norm()


def test_checkpointing_only_large_training_tables(monkeypatch):
    # Only training forwards with more rows than the limit: once per block (column 1, row 1, ICL 2)
    calls = []
    real = M.checkpoint

    def counting(block, *args, **kwargs):
        calls.append(type(block).__name__)
        return real(block, *args, **kwargs)

    monkeypatch.setattr(M, "checkpoint", counting)
    model = _model()
    x, y = _batch("regression")  # 40 rows
    model.activation_checkpointing_rows = 40
    model(x, y, "regression").sum().backward()
    assert calls == []
    model.activation_checkpointing_rows = 39
    model(x, y, "regression").sum().backward()
    assert calls == ["InducedBlock", "Block", "Block", "Block"]
    calls.clear()
    with torch.no_grad():  # no gradient, nothing to recompute
        model(x, y, "regression")
    model.eval()  # validation/inference
    model(x, y, "regression")
    model(x, y, "regression", inference=True)
    assert calls == []


def test_checkpointing_config():
    config = TrainConfig.smoke()
    assert config.activation_checkpointing_rows is None  # off by default
    assert replace(config, activation_checkpointing_rows=10).activation_checkpointing_rows == 10
    tabsil = helpers.tabsil_smoke()
    assert replace(tabsil, activation_checkpointing_rows=0).activation_checkpointing_rows == 0
    with pytest.raises(ValueError, match="must be at least 0"):
        replace(tabsil, activation_checkpointing_rows=-1)
    # TabSil turns it on in stage 2 and 3 only, for tables above 2,048 rows
    s1, s2, s3 = (recipe.curriculum(n) for n in (1, 2, 3))
    assert s1.activation_checkpointing_rows is None
    assert s2.activation_checkpointing_rows == s3.activation_checkpointing_rows == 2_048
    # Not stored in the model config (checkpoint file, config.json)
    assert not hasattr(s3.model_config(), "activation_checkpointing_rows")


def _smoke(**changes):
    config = helpers.tabsil_smoke()
    small = dict(
        max_features=6,
        min_rows=48,
        max_rows=80,
        max_cells=80 * 6,
        log_uniform_rows=True,
        steps=4,
        warmup_steps=1,
        val_every=2,
        log_every=1,
        checkpoint_every=2,
        keep_every=None,
        collapse_threshold=None,
        border_datasets=20,
    )
    return replace(config, **{**small, **changes})


def test_checkpointing_trains_same(tmp_path, monkeypatch):
    # Training loop: a run with it on (all tables) has the same training loss as one with it off
    # (compared with the score-matrix decoder). Validation batches use the inference path
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    monkeypatch.setattr(M, "CHECKPOINT_FUSED_DECODER", False)
    inference_calls = []
    real = loop._validate

    signature = inspect.signature(real)

    def recording(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        inference_calls.append(bound.arguments["inference_rows"])
        return real(*args, **kwargs)

    monkeypatch.setattr(loop, "_validate", recording)
    off = train(_smoke(), tmp_path / "off", mlflow_log=False)
    assert set(inference_calls) == {None}
    inference_calls.clear()
    on = train(_smoke(activation_checkpointing_rows=0), tmp_path / "on", mlflow_log=False)
    assert set(inference_calls) == {0}
    assert on["history"]["train_loss"] == off["history"]["train_loss"]
    # Validation loss uses the inference path (row chunks, no score matrix): only op order differs
    assert np.allclose(on["history"]["val_loss"], off["history"]["val_loss"], rtol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="block compile needs a GPU")
def test_checkpointing_trains_compiled(tmp_path, monkeypatch):
    # Works with the per-block compile speedup option; the recomputed forward is compiled too
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    config = _smoke(activation_checkpointing_rows=60, compile=True, stop_step=3)
    out = train(config, tmp_path, mlflow_log=False)
    assert all(np.isfinite(out["history"]["train_loss"]))
