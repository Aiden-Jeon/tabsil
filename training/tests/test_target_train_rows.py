"""Tests of the train-row target scaling (`TrainConfig.target_train_rows`,
`priors.transforms.TrainRowTargetScale`): the regression target standardized with the train rows,
as inference does, in the prior, the regression borders and the validation."""

from dataclasses import replace

import numpy as np
import pytest
import torch
from train_helpers import load_script, tabsil_smoke, tiny_model

from tabsil_train.priors import PriorBatch
from tabsil_train.priors import graph_scm as P
from tabsil_train.priors.pipeline import Group, Rng
from tabsil_train.priors.transforms import TrainRowTargetScale
from tabsil_train.priors.transforms.target import train_row_scale
from tabsil_train.training import reference as R
from tabsil_train.training.config import TrainConfig
from tabsil_train.training.loop import _validate

SIZES = dict(min_rows=60, max_rows=120, log_uniform_rows=True, max_cells=120 * 12, max_features=12)


def _group(y: torch.Tensor, n_train: int) -> Group:
    return Group(torch.randn(len(y), y.shape[1], 3), y, n_train, {})


def test_the_target_is_standardized_with_the_train_rows():
    torch.manual_seed(0)
    y = torch.randn(50, 3) * torch.tensor([1.0, 2.0, 0.5]) + torch.tensor([0.0, 3.0, -1.0])
    group = _group(y, 30)
    rng = Rng.from_seed(0)
    state = rng.numpy.bit_generator.state
    out = TrainRowTargetScale()(group, rng)
    assert rng.numpy.bit_generator.state == state  # no random numbers
    assert torch.equal(out.x, group.x)
    train = out.y[:30].double()
    assert torch.allclose(train.mean(0), torch.zeros(3, dtype=torch.float64), atol=1e-6)
    assert torch.allclose(train.std(0, correction=0), torch.ones(3, dtype=torch.float64))
    # the eval rows move with the same mean and sd (as at inference)
    mean, std = y[:30].mean(0), y[:30].std(0, correction=0)
    assert torch.allclose(out.y[30:], (y[30:] - mean) / std, atol=1e-5)


def test_tables_with_a_narrow_train_spread_are_drawn_again():
    y = torch.randn(40, 2)
    y[:20, 1] = 0.05 * torch.randn(20)  # train-row sd about 0.05 of the all-row units
    assert TrainRowTargetScale()(_group(y, 20), Rng.from_seed(0)) is None
    assert TrainRowTargetScale(min_train_std=0.0)(_group(y, 20), Rng.from_seed(0)) is not None
    with pytest.raises(ValueError):
        TrainRowTargetScale(-1.0)


def test_the_prior_scales_only_regression_targets():
    on = P.GraphSCMPrior("regression", seed=2, target_train_rows=True, **SIZES)
    assert isinstance(on._accept[-1], TrainRowTargetScale)
    batch = on.sample_batch(4)
    train = batch.y_train.double()
    assert torch.allclose(train.mean(1), torch.zeros(4, dtype=torch.float64), atol=1e-5)
    assert torch.allclose(train.std(1, correction=0), torch.ones(4, dtype=torch.float64))
    # off (v1): the targets are standardized over all rows, not with the train rows
    off = P.GraphSCMPrior("regression", seed=2, **SIZES)
    assert not any(isinstance(t, TrainRowTargetScale) for t in off._accept)
    # classification is unchanged by the option
    a = P.GraphSCMPrior("classification", seed=4, target_train_rows=True, **SIZES).sample_batch(3)
    b = P.GraphSCMPrior("classification", seed=4, **SIZES).sample_batch(3)
    assert torch.equal(a.x.nan_to_num(-9.0), b.x.nan_to_num(-9.0)) and torch.equal(a.y, b.y)


def test_the_option_reaches_the_prior_and_the_borders(monkeypatch):
    config = replace(
        tabsil_smoke(),
        target_train_rows=True,
        border_datasets=3,
        filter_unpredictable=False,
        **SIZES,
    )
    seen = []
    real = P.GraphSCMPrior

    def spy(*args, **kwargs):
        seen.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(P, "GraphSCMPrior", spy)
    P.make_borders(config, n_buckets=10, seed=0)
    config.make_prior("regression", seed=0)
    assert [kw["target_train_rows"] for kw in seen] == [True, True]
    # the reference validation set is drawn without it
    assert R.reference_config(config).target_train_rows is False


def test_borders_follow_the_scaled_targets():
    config = replace(tabsil_smoke(), border_datasets=6, filter_unpredictable=False, **SIZES)
    off = P.make_borders(config, 20, seed=1)
    on = P.make_borders(replace(config, target_train_rows=True), 20, seed=1)
    assert on.shape == off.shape and (on.diff() > 0).all() and not torch.equal(on, off)


def _batches(seed=0, b=2, r=30, n_train=20):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(b, r, 4, generator=g)
    y = 2.0 * x[..., 0] + 0.3 * torch.randn(b, r, generator=g) + 1.0
    return [PriorBatch(x, y, n_train)]


def _autocast():
    import contextlib

    return contextlib.nullcontext()


def test_validation_of_a_scaled_run_maps_the_loss_back_to_the_batch_units():
    model = tiny_model()
    batches = _batches()
    batch = batches[0]
    mean, scale = train_row_scale(batch.y, batch.n_train)
    scaled = [PriorBatch(batch.x, (batch.y - mean) / scale, batch.n_train)]
    plain = _validate(model, scaled, "cpu", _autocast, "regression")["val_loss"]
    mapped = _validate(model, batches, "cpu", _autocast, "regression", train_row_targets=True)
    # the same model outputs, each table's loss times its train-row sd
    from tabsil.models.quantile_distribution import pinball_loss

    with torch.no_grad():
        q = model(scaled[0].x, scaled[0].y[:, :20], task="regression")
        expected = (pinball_loss(q, scaled[0].y[:, 20:]) * scale).mean().item()
    assert mapped["val_loss"] == pytest.approx(expected, rel=1e-5)
    assert mapped["val_loss"] != pytest.approx(plain, rel=1e-3)
    # on targets already standardized with the train rows the mapping changes nothing
    again = _validate(model, scaled, "cpu", _autocast, "regression", train_row_targets=True)
    assert again["val_loss"] == pytest.approx(plain, rel=1e-5)
    # classification is not touched
    cls = [PriorBatch(batch.x, (batch.y > 1.0).float(), batch.n_train)]
    assert _validate(model, cls, "cpu", _autocast, "classification", train_row_targets=True) == (
        _validate(model, cls, "cpu", _autocast, "classification")
    )


def test_train_script_sets_the_option_and_a_resumed_run_keeps_it(monkeypatch):
    for name in ("TABSIL_CONFIG_NAME", "TABSIL_STEPS", "TABSIL_TARGET_TRAIN_ROWS"):
        monkeypatch.delenv(name, raising=False)
    script = load_script("scripts/train.py", "train_script_target")
    base = script.build_config(script.parse_args(["--config-name", "medium"]))
    assert base == TrainConfig.medium() and not base.target_train_rows
    flag = script.build_config(
        script.parse_args(["--config-name", "medium", "--target-train-rows"])
    )
    assert flag == replace(TrainConfig.medium(), target_train_rows=True)
    monkeypatch.setenv("TABSIL_TARGET_TRAIN_ROWS", "1")
    script = load_script("scripts/train.py", "train_script_target")
    assert script.build_config(script.parse_args(["--config-name", "medium"])) == flag
    monkeypatch.delenv("TABSIL_TARGET_TRAIN_ROWS")
    script = load_script("scripts/train.py", "train_script_target")
    resume = ["--config-name", "medium", "--resume-run-id", "abc"]
    # a run that had it on keeps it on
    args = script.parse_args(resume)
    script.inherit_recipe_options(args, {"target_train_rows": "True"})
    assert script.build_config(args).target_train_rows
    # a run that had it off (or predates it) cannot turn it on
    args = script.parse_args([*resume, "--target-train-rows"])
    with pytest.raises(SystemExit, match="target-train-rows"):
        script.inherit_recipe_options(args, {})
    args = script.parse_args(resume)
    script.inherit_recipe_options(args, {"target_train_rows": "False"})
    assert not script.build_config(args).target_train_rows
    # a run that takes only the weights uses the given value
    args = script.parse_args(["--config-name", "stage2", "--init-run-id", "abc"])
    script.inherit_recipe_options(args, {"target_train_rows": "True"})
    assert not args.target_train_rows


def test_scaled_targets_have_no_nan_with_missing_eval_targets():
    # a NaN eval target (no loss, `pinball_loss`) does not reach the train-row statistics
    y = torch.randn(1, 10)
    y[0, 8] = float("nan")
    mean, std = train_row_scale(y, 6)
    assert torch.isfinite(mean).all() and torch.isfinite(std).all()
    assert np.isclose(float(mean), float(y[0, :6].mean()), atol=1e-6)


def test_training_runs_with_the_option(tmp_path):
    from tabsil_train.training.loop import train

    config = replace(
        tabsil_smoke(),
        target_train_rows=True,
        filter_unpredictable=False,
        border_datasets=6,
        batch_size=4,
        micro_batch=4,
        min_rows=32,
        max_rows=32,
        max_features=4,
        max_cells=128,
        steps=2,
        warmup_steps=1,
        log_every=1,
        val_every=2,
        checkpoint_every=2,
    )
    result = train(config, tmp_path, mlflow_log=False)
    assert torch.isfinite(torch.tensor(result["history"]["train_loss"])).all()
