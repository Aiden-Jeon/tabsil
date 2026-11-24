"""The Kumo-Tabular parts of the model (`column_row_transformer` module docstring): interleaved
column/row stages and the regression quantile head with the training stack: the training loss,
the TabSil curriculum presets and the train script (the model parts alone: tests/)."""

from dataclasses import replace

import pytest
import torch
import train_helpers as helpers

from tabsil.models import model_from_state
from tabsil.models.column_row_transformer import ColumnRowTransformer, ColumnRowTransformerConfig
from tabsil_train import recipe
from tabsil_train.hub import run_architecture
from tabsil_train.training.loop import train
from tabsil_train.training.losses import prior_loss


def _config(**changes) -> ColumnRowTransformerConfig:
    return ColumnRowTransformerConfig.tiny(**changes)


def _inputs(c: int = 5, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(2, 40, c, generator=g)
    x[0, 3, 1] = float("nan")
    x[:, :, 2] = torch.randint(0, 4, (2, 40), generator=g).float()  # categorical column (codes)
    yc = torch.randint(0, 3, (2, 30), generator=g).float()
    yr = torch.randn(2, 30, generator=g)
    return x, yc, yr


def test_quantile_loss_of_a_fresh_model_is_finite():
    # A freshly built model outputs the initial quantiles; the training loss on them is finite
    model = ColumnRowTransformer(_config()).eval()
    x, _, yr = _inputs()
    out = model(x, yr, "regression")
    loss = prior_loss(out, torch.randn(2, 10), "regression")
    assert torch.isfinite(loss)


def test_curriculum_presets():
    one, two, three = (recipe.curriculum(n) for n in (1, 2, 3))
    for config in (one, two, three):
        assert {k: getattr(config, k) for k in recipe.STRUCTURE} == recipe.STRUCTURE
    assert (one.lr, two.lr, three.lr) == (8e-4, 1e-4, 2e-5)
    assert (one.micro_batch, two.micro_batch, three.micro_batch) == (4, 1, 1)
    assert (one.max_rows, two.max_rows, three.max_rows) == (1_024, 10_240, 60_000)
    # The wide column/row stages need activation checkpointing above 2,048 rows from stage 2
    assert one.activation_checkpointing_rows is None
    assert two.activation_checkpointing_rows == three.activation_checkpointing_rows == 2_048
    assert recipe.curriculum(1, steps=1000).warmup_steps == 10
    with pytest.raises(ValueError, match="curriculum stage"):
        recipe.curriculum(4)
    with pytest.raises(ValueError, match="step"):
        recipe.curriculum(1, steps=0)


def test_tabsil_smoke_checkpoint_round_trip(tmp_path, monkeypatch):
    # A small TabSil model trained and saved reads back with the same structure
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    small = dict(
        max_features=8,
        min_rows=48,
        max_rows=80,
        max_cells=80 * 8,
        steps=2,
        warmup_steps=1,
        val_every=2,
        log_every=1,
        checkpoint_every=2,
        keep_every=None,
        collapse_threshold=None,
        border_datasets=20,
    )
    config = helpers.tabsil_smoke(**small)
    train(config, tmp_path, mlflow_log=False)
    model, borders = model_from_state(torch.load(tmp_path / "latest.pt", weights_only=True))
    assert borders is not None and model.config == config.model_config()


def _script(name):
    return helpers.load_script("scripts/train.py", name)


def _clear_train_env(monkeypatch):
    for name in ("CONFIG_NAME", "LR", "MICRO_BATCH", "STEPS"):
        monkeypatch.delenv(f"TABSIL_{name}", raising=False)


def test_train_script_curriculum(monkeypatch):
    _clear_train_env(monkeypatch)
    script = _script("train_script_tabsil_curriculum")
    config = script.build_config(script.parse_args(["--config-name", "stage1"]))
    assert config == recipe.curriculum(1)
    # Stage 2 and 3 take the weights of the previous stage's run
    with pytest.raises(SystemExit):
        script.parse_args(["--config-name", "stage2"])
    source = run_architecture({k: str(v) for k, v in vars(recipe.curriculum(1)).items()})
    for n in (2, 3):
        args = script.parse_args([f"--config-name=stage{n}", "--init-run-id", "x"])
        assert script.build_config(args, source) == recipe.curriculum(n)
    # The weights of a run with another size are not taken
    args = script.parse_args(["--config-name", "stage2", "--init-run-id", "x"])
    other = replace(recipe.curriculum(1), n_cls=8, emsize=128)
    with pytest.raises(SystemExit, match="structure"):
        script.build_config(args, run_architecture({k: str(v) for k, v in vars(other).items()}))
