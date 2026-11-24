"""The Kumo-Tabular parts of the model (`column_row_transformer` module docstring): interleaved
column/row stages and the regression quantile head, plus the TabSil curriculum presets."""

import math
from dataclasses import replace

import helpers
import numpy as np
import pytest
import torch

from tabsil import recipe
from tabsil.hub import run_architecture
from tabsil.inference.ensemble import EnsembleClassifier, EnsembleRegressor
from tabsil.models import config_dict, model_from_state
from tabsil.models.column_row_transformer import ColumnRowTransformer, ColumnRowTransformerConfig
from tabsil.models.quantile_distribution import (
    N_QUANTILES,
    initial_bias,
    pinball_loss,
    quantile_levels,
    quantiles_to_bucket_probs,
)
from tabsil.training.loop import train
from tabsil.training.losses import prior_loss

# Tiny Kumo-style column/row stages (2 block pairs)
TINY_STAGES = {"column_blocks": 2, "row_blocks": 2}


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


def test_validation():
    with pytest.raises(ValueError, match="column blocks"):
        _config(column_blocks=1, row_blocks=2)
    with pytest.raises(ValueError, match="column blocks"):
        _config(column_blocks=0, row_blocks=0)
    model = ColumnRowTransformer(_config())
    x, yc, _ = _inputs()
    with pytest.raises(TypeError):
        model(x, yc)  # the task is required
    with pytest.raises(ValueError, match="task"):
        model(x, yc, "multitask")


# --- Interleaved column/row stages -----------------------------------------------------------


def test_interleaved_layout_runs_and_matches_chunked_inference():
    torch.manual_seed(0)
    model = ColumnRowTransformer(_config(**TINY_STAGES)).eval()
    x, yc, yr = _inputs()
    for task, y in (("classification", yc), ("regression", yr)):
        full = model(x, y, task)
        model.inference_chunk_cells = 37  # several row chunks, several column chunks
        model.inference_column_chunk = 2
        chunked = model(x, y, task, inference=True)
        model.inference_chunk_cells = ColumnRowTransformer.inference_chunk_cells
        model.inference_column_chunk = ColumnRowTransformer.inference_column_chunk
        torch.testing.assert_close(chunked, full)


def test_interleaved_layout_mixes_columns_before_the_next_column_block():
    # When interleaved, the second column block's summary sees other columns: changing just one
    # other column changes the first column's column-stage output
    torch.manual_seed(0)
    seen = {}

    def hook(module, args, out):
        seen["a"] = out.detach().clone()

    x, yc, _ = _inputs()
    x2 = x.clone()
    x2[:, :, 4] += 1.0
    model = ColumnRowTransformer(_config(**TINY_STAGES)).eval()
    model.column_blocks[1].register_forward_hook(hook)
    model(x, yc, "classification")
    first = seen["a"].view(2, 5, 40, -1)[:, 0]
    model(x2, yc, "classification")
    second = seen["a"].view(2, 5, 40, -1)[:, 0]
    assert not torch.equal(first, second)


def test_kumo_stages_size():
    # The row ICL is the TabPFN-3.5 one (width 1024); the cell, column and row stages are 256 wide
    model_config = recipe.curriculum(1).model_config()
    assert model_config.icl_dim == 1024 and model_config.emsize == 256
    with torch.device("meta"):
        counts = ColumnRowTransformer(model_config).num_parameters_by_stage()
    assert counts["column_stage"] + counts["row_stage"] > counts["cell_encoder"]


# --- Regression quantile head -------------------------------------------------------------


def test_pinball_loss_and_initial_quantiles():
    tau = quantile_levels(999)
    assert tau[0] == pytest.approx(0.001) and tau[-1] == pytest.approx(0.999)
    q = torch.tensor([[0.0, 1.0, 2.0]])
    y = torch.tensor([1.5])
    # τ = 0.25, 0.5, 0.75: 0.25·1.5, 0.5·0.5, 0.25·0.5 ((1 − τ)(q − y) when y is below q)
    expected = (0.25 * 1.5 + 0.5 * 0.5 + (1 - 0.75) * 0.5) / 3
    assert pinball_loss(q, y).item() == pytest.approx(expected)
    assert pinball_loss(q, torch.tensor([float("nan")])).item() == 0.0
    bias = initial_bias()
    assert bias[499] == pytest.approx(0.0, abs=1e-6) and bias[-1] == pytest.approx(3.0902, abs=1e-3)
    # A freshly built model outputs standard normal quantiles on every row (weight 0)
    model = ColumnRowTransformer(_config()).eval()
    x, _, yr = _inputs()
    out = model(x, yr, "regression")
    assert out.shape == (2, 10, N_QUANTILES)
    torch.testing.assert_close(out, bias.expand_as(out))
    loss = prior_loss(out, torch.randn(2, 10), "regression")
    assert torch.isfinite(loss)


def test_quantiles_to_bucket_probs():
    borders = torch.linspace(-6, 6, 2001)
    q = initial_bias()[None].repeat(3, 1)
    q[1] = q[1] * 2 + 1  # N(1, 4)
    q[2] = q[2].flip(0)  # unsorted quantiles are accepted too
    p = quantiles_to_bucket_probs(q, borders)
    assert p.shape == (3, 2000) and (p >= 0).all()
    torch.testing.assert_close(p.sum(-1), torch.ones(3))
    centers = (borders[1:] + borders[:-1]) / 2
    means = p @ centers
    assert means[0].item() == pytest.approx(0.0, abs=1e-3)
    assert means[1].item() == pytest.approx(1.0, abs=1e-2)
    torch.testing.assert_close(p[2], p[0])
    # CDF: 0.5 at border 0 (N(0, 1))
    cdf = torch.cumsum(p[0], 0)
    assert cdf[999].item() == pytest.approx(0.5, abs=2e-3)
    # Quantiles outside the borders go to the end buckets (no mass is lost)
    far = quantiles_to_bucket_probs(torch.full((1, 5), 100.0), borders)
    assert far[0, -1].item() == pytest.approx(1.0)


def test_quantile_regressor_predicts_through_the_ensemble():
    torch.manual_seed(0)
    model = ColumnRowTransformer(_config())
    borders = torch.linspace(-5, 5, 101)
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(60, 4)), rng.normal(size=60) * 3 + 10
    reg = EnsembleRegressor(model, borders, n_estimators=2, member_kinds=("standard",)).fit(X, y)
    pred = reg.predict(X[:7])
    # A fresh model predicts near the training target mean (standard normal quantiles in z space)
    assert pred.shape == (7,) and np.allclose(pred, y.mean(), atol=0.5)
    log_p, out_borders = reg.predict_distribution(X[:7])
    assert log_p.shape == (7, 100)
    torch.testing.assert_close(log_p.exp().sum(-1), torch.ones(7))
    assert out_borders.dtype == torch.float64


def test_test_time_methods_refuse_regression():
    from tabsil.inference.diag_scale import DiagScale

    borders = torch.linspace(-3, 3, 11)
    model = ColumnRowTransformer(_config())
    with pytest.raises(ValueError, match="DiagScale"):
        DiagScale(EnsembleRegressor(model, borders, n_estimators=2))
    DiagScale(EnsembleClassifier(model, n_estimators=2))  # the classification side works


# --- Training config, training script, smoke training ----------------------------------------


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


def test_saved_tabsil_config_round_trips():
    # Checkpoints and Hub config.json files hold the model config only (no structure name). The
    # rebuilt model has the same config and TabSil's inference settings
    from tabsil.models import config_from_dict

    model_config = recipe.curriculum(1).model_config()
    saved = config_dict(model_config)
    assert not {"name", "arch", "architecture", "task"} & set(saved)
    with torch.device("meta"):
        model = ColumnRowTransformer(config_from_dict("column_row_transformer", saved))
    assert model.config == model_config
    for task in ("classification", "regression"):
        assert recipe.default_inference(model, task)["max_features_per_member"] == 768


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


def test_attention_splits_large_batch(monkeypatch):
    # Attention is called in pieces when the batch exceeds the CUDA kernel grid limit
    # (`SDPA_MAX_BATCH`; flash attention failed with invalid argument on 8 tables × 10,240 rows).
    # Splitting keeps outputs and gradients the same (checked on CPU with a small limit)
    from tabsil.models import column_row_transformer as M

    torch.manual_seed(0)
    attn = M.Attention(64, 4, zero_out=False)
    x = torch.randn(7, 30, 64)
    runs = []
    for max_batch in (65_535, 3):
        monkeypatch.setattr(M, "SDPA_MAX_BATCH", max_batch)
        x_ = x.clone().requires_grad_()
        attn.zero_grad(set_to_none=True)
        out = attn(x_, n_context=20)
        out.square().mean().backward()
        runs.append([out, x_.grad, *(p.grad for p in attn.parameters() if p.grad is not None)])
    for a, b in zip(*runs, strict=True):
        assert torch.allclose(a, b, atol=1e-6)
    assert math.isfinite(float(runs[0][0].detach().sum()))
