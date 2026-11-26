"""Training loop tests (smoke size, CPU)."""

from dataclasses import replace

import pytest
import torch
from train_helpers import load_script, tabsil_smoke

from tabsil.models import load_checkpoint
from tabsil.models.quantile_distribution import N_QUANTILES
from tabsil_train import recipe
from tabsil_train.training.config import TrainConfig
from tabsil_train.training.loop import _make_val_batches, check_source_checkpoint, train
from tabsil_train.training.stability import CollapseGuard
from tabsil_train.utils import lr_factor


def _smoke(**changes) -> TrainConfig:
    """The TabSil smoke config without the (slow) unpredictable-dataset filter, with few border
    datasets and small batches of narrow tables, so that the many short runs below stay fast."""
    small = {
        "filter_unpredictable": False,
        "border_datasets": 10,
        "batch_size": 4,
        "micro_batch": 4,
        "min_rows": 32,
        "max_rows": 32,
        "max_features": 4,
        "max_cells": 128,
    }
    return tabsil_smoke(**{**small, **changes})


def test_lr_schedule_warms_up_then_decays_to_zero():
    assert lr_factor(0, 10, 100) == 0.0
    assert lr_factor(10, 10, 100) == 1.0
    assert 0.0 < lr_factor(55, 10, 100) < 1.0
    assert lr_factor(100, 10, 100) == 0.0


def test_training_writes_checkpoint_with_borders_and_resumes(tmp_path):
    # one model learns both tasks: the checkpoint holds the regression borders
    config = _smoke(steps=4, val_every=2)
    result = train(config, tmp_path, mlflow_log=False)
    model, borders = load_checkpoint(result["final_path"])
    assert model.config.n_out == 10 and model.config.output_size("regression") == N_QUANTILES
    assert borders.shape == (5001,) and (borders[1:] > borders[:-1]).all()
    assert result["history"]["step"][-1] == 4
    assert torch.isfinite(torch.tensor(result["history"]["train_loss"])).all()

    longer = replace(config, steps=6)
    resumed = train(
        longer, tmp_path / "resumed", resume_from=tmp_path / "latest.pt", mlflow_log=False
    )
    assert resumed["history"]["step"][-1] == 6
    assert resumed["history"]["step"][0] == result["history"]["step"][0]  # appends old history


def test_init_from_takes_only_the_weights(tmp_path):
    config = _smoke(steps=4, val_every=2)
    train(config, tmp_path / "first", mlflow_log=False)
    init = tmp_path / "first" / "latest.pt"
    with pytest.raises(ValueError, match="init_from"):
        train(config, tmp_path / "x", resume_from=init, init_from=init, mlflow_log=False)
    # training at step 0 leaves the weights unchanged (lr is 0 on the first warmup step)
    frozen = replace(config, steps=1, warmup_steps=1, stop_step=1, collapse_threshold=None)
    result = train(frozen, tmp_path / "second", init_from=init, mlflow_log=False)
    assert result["history"]["step"] == [1]  # does not append the previous history
    before = torch.load(init, weights_only=True)
    after = torch.load(tmp_path / "second" / "latest.pt", weights_only=True)
    assert after["step"] == 1 and after["scheduler"]["last_epoch"] == 1
    # (the Muon weight decay is scaled by the lr too, so nothing moves)
    for name, tensor in before["model"].items():
        assert torch.equal(after["model"][name], tensor), name
    assert torch.equal(after["borders"], before["borders"])  # the borders come along


def test_chunked_training_continues_the_full_schedule(tmp_path):
    config = _smoke(steps=10, log_every=1)
    first = train(replace(config, stop_step=4), tmp_path / "a", mlflow_log=False)
    assert first["history"]["step"][-1] == 4
    state = torch.load(tmp_path / "a" / "latest.pt", weights_only=True)
    assert state["step"] == 4 and state["optimizer_type"] == "muon"
    # the lr follows the full 10-step schedule, not this run's 4 steps
    expected = config.lr * lr_factor(4, config.warmup_steps, config.steps)
    assert abs(state["scheduler"]["_last_lr"][0] - expected) < 1e-12

    second = train(
        config, tmp_path / "b", resume_from=tmp_path / "a" / "latest.pt", mlflow_log=False
    )
    assert second["history"]["step"] == list(range(1, 11))


def test_kept_checkpoints_are_not_overwritten(tmp_path):
    config = _smoke(steps=6, checkpoint_every=2, keep_every=4)
    train(config, tmp_path, mlflow_log=False)
    assert sorted(p.name for p in tmp_path.glob("step_*.pt")) == ["step_0000004.pt"]
    assert torch.load(tmp_path / "step_0000004.pt", weights_only=True)["step"] == 4
    assert torch.load(tmp_path / "latest.pt", weights_only=True)["step"] == 6

    # resuming from a kept step trains again from that step
    resumed = train(
        config, tmp_path / "b", resume_from=tmp_path / "step_0000004.pt", mlflow_log=False
    )
    assert resumed["history"]["step"][-1] == 6

    # keeps that step even when keep_every is not a multiple of checkpoint_every
    odd = replace(config, checkpoint_every=4, keep_every=3)
    train(odd, tmp_path / "c", mlflow_log=False)
    assert sorted(p.name for p in (tmp_path / "c").glob("step_*.pt")) == [
        "step_0000003.pt",
        "step_0000006.pt",
    ]


def test_val_batches_do_not_depend_on_micro_batch():
    config = _smoke()
    for task in config.tasks:
        a = _make_val_batches(replace(config, micro_batch=4), task, n_batches=2)
        b = _make_val_batches(replace(config, micro_batch=1), task, n_batches=2)
        for batch_a, batch_b in zip(a, b, strict=True):
            assert batch_a.x.shape[0] == config.val_batch
            assert torch.equal(batch_a.x.nan_to_num(), batch_b.x.nan_to_num())
            assert torch.equal(batch_a.y, batch_b.y)


def test_deadline_in_the_past_stops_at_the_next_log_step(tmp_path):
    config = _smoke(steps=10, log_every=2, deadline=0.0)
    result = train(config, tmp_path, mlflow_log=False)
    assert result["history"]["step"][-1] == 2
    assert torch.load(tmp_path / "latest.pt", weights_only=True)["step"] == 2


def test_micro_batch_shrinks_when_gpus_exceed_the_batch():
    config = TrainConfig()  # batch 256, micro 4
    for world_size, micro, accumulation in [(1, 4, 64), (8, 4, 8), (64, 4, 1), (128, 2, 1)]:
        scaled = config.for_world_size(world_size)
        assert scaled.batch_size == 256
        assert scaled.micro_batch == micro
        assert scaled.grad_accumulation(world_size) == accumulation


def test_collapse_guard_needs_a_sustained_jump():
    guard = CollapseGuard(threshold=0.5, patience=3)
    for v in [0.82, 0.80, 0.81, 0.79, 0.83, 0.80]:
        assert not guard.update(v)
    assert not guard.update(1.5)  # a single spike is not a collapse
    assert not guard.update(0.81) and not guard.suspect
    assert not guard.update(2.0) and not guard.update(2.0)
    assert guard.update(2.0)  # three in a row is a collapse


def test_training_rewinds_after_a_collapse(tmp_path, monkeypatch):
    calls = {"n": 0}
    original = CollapseGuard.update

    def fake_update(self, window_mean):
        calls["n"] += 1
        if calls["n"] == 7:  # treat the 7th window as one collapse
            return True
        return original(self, window_mean)

    monkeypatch.setattr(CollapseGuard, "update", fake_update)
    config = _smoke(steps=20, log_every=2, snapshot_every=4, val_every=20, keep_every=None)
    result = train(config, tmp_path, mlflow_log=False)
    steps = result["history"]["step"]
    assert steps[-1] == 20
    assert steps == sorted(set(steps))  # history of the rolled-back span is erased and rewritten
    # collapse at step 14 -> back to the older of the last two snapshots (8, 12), then on to 20
    assert calls["n"] == 7 + (20 - 8) // 2
    # the rollback count is saved in the checkpoint and mixed into the next chunk's data seed
    assert torch.load(tmp_path / "latest.pt", weights_only=True)["rewinds"] == 1


@pytest.mark.parametrize("min_scale, expected_scale", [(0.0, 0.25), (0.75, 0.75)])
def test_each_collapse_lowers_the_lr_by_the_rewind_factor(
    tmp_path, monkeypatch, min_scale, expected_scale
):
    calls = {"n": 0}
    original = CollapseGuard.update

    def fake_update(self, window_mean):
        calls["n"] += 1
        if calls["n"] in (7, 9):  # collapse at step 14, back to 8, and again at step 12
            return True
        return original(self, window_mean)

    monkeypatch.setattr(CollapseGuard, "update", fake_update)
    config = _smoke(
        steps=20, log_every=2, snapshot_every=4, rewind_lr_factor=0.5, val_every=20, keep_every=None
    )
    config = replace(config, rewind_min_lr=config.lr * min_scale)
    train(config, tmp_path, mlflow_log=False)
    state = torch.load(tmp_path / "latest.pt", weights_only=True)
    assert state["rewinds"] == 2
    # The second rollback snapshot (step 8) predates the first cut, but both lr cuts apply.
    # with a floor, it stops there
    base = config.lr * expected_scale
    assert abs(state["scheduler"]["base_lrs"][0] - base) < 1e-15
    expected = base * lr_factor(20, config.warmup_steps, config.steps)
    # every Muon param group follows the lowered lr
    for group in state["optimizer"]["param_groups"]:
        assert abs(group["lr"] - expected) < 1e-12
    assert state["scheduler"]["base_lrs"] == pytest.approx(
        [base] * len(state["scheduler"]["base_lrs"])
    )
    # the next chunk resumes (last-step checkpoint, so the resume-step check is skipped)
    assert abs(check_source_checkpoint(tmp_path / "latest.pt", config, resume=False) - base) < 1e-15


def test_curriculum_lowers_the_lr_on_collapse():
    assert all(recipe.curriculum(n).rewind_lr_factor == 0.5 for n in recipe.CURRICULUM)
    assert TrainConfig().rewind_lr_factor == 0.5
    for bad in (0.0, -0.5, 5.0):
        with pytest.raises(ValueError, match="rewind_lr_factor"):
            TrainConfig(rewind_lr_factor=bad)


def test_training_that_ends_suspect_saves_the_newest_healthy_state(tmp_path, monkeypatch):
    calls = {"n": 0}
    original = CollapseGuard.update

    def fake_update(self, window_mean):
        calls["n"] += 1
        if calls["n"] == 10:  # only the last window (step 20) jumps once
            self.bad += 1
            return False
        return original(self, window_mean)

    monkeypatch.setattr(CollapseGuard, "update", fake_update)
    config = _smoke(
        steps=20, log_every=2, snapshot_every=4, rewind_lr_factor=0.5, val_every=20, keep_every=None
    )
    result = train(config, tmp_path, mlflow_log=False)
    state = torch.load(tmp_path / "latest.pt", weights_only=True)
    assert state["step"] == 16  # the newest healthy state, not the older 12
    assert state["scheduler"]["base_lrs"][0] == config.lr  # not a collapse, so lr is unchanged
    assert result["history"]["step"][-1] == 16


def test_training_stops_after_too_many_rewinds(tmp_path):
    config = _smoke(
        steps=60,
        log_every=2,
        snapshot_every=4,
        val_every=60,
        keep_every=None,
        collapse_threshold=-10.0,  # every window is bad once healthy history exists
        max_rewinds=2,
    )
    with pytest.raises(RuntimeError, match="rewound more than"):
        train(config, tmp_path, mlflow_log=False)


def test_collapse_guard_counts_windows_without_finite_loss():
    guard = CollapseGuard(threshold=0.5, patience=2)
    assert not guard.update(float("inf")) and guard.suspect  # bad even without healthy history
    assert guard.update(float("inf"))


def test_resume_with_a_different_lr_rescales_the_rest_of_the_schedule(tmp_path):
    config = _smoke(steps=10, log_every=1)
    train(replace(config, stop_step=4), tmp_path / "a", mlflow_log=False)
    # used when lr is not given
    assert check_source_checkpoint(tmp_path / "a" / "latest.pt", config, resume=True) == config.lr
    lower = replace(config, lr=config.lr / 2)
    train(
        replace(lower, stop_step=6),
        tmp_path / "b",
        resume_from=tmp_path / "a" / "latest.pt",
        mlflow_log=False,
    )
    state = torch.load(tmp_path / "b" / "latest.pt", weights_only=True)
    assert abs(state["scheduler"]["base_lrs"][0] - lower.lr) < 1e-15
    expected = lower.lr * lr_factor(6, config.warmup_steps, config.steps)
    assert abs(state["scheduler"]["_last_lr"][0] - expected) < 1e-12
    assert abs(state["optimizer"]["param_groups"][0]["lr"] - expected) < 1e-12


def test_resume_past_the_stop_step_is_an_error(tmp_path):
    # resuming a 300k paper run with long_tables (20k steps) used to end without training and
    # overwrite that run's results; this is now an error
    config = _smoke(steps=4, val_every=2)
    train(config, tmp_path / "a", mlflow_log=False)
    with pytest.raises(ValueError, match="nothing to train"):
        train(config, tmp_path / "b", resume_from=tmp_path / "a" / "latest.pt", mlflow_log=False)
    with pytest.raises(ValueError, match="nothing to train"):
        train(
            replace(config, steps=10, stop_step=3),
            tmp_path / "c",
            resume_from=tmp_path / "a" / "latest.pt",
            mlflow_log=False,
        )


def test_resume_refuses_checkpoints_of_other_optimizers(tmp_path):
    # a checkpoint without a Muon optimizer state cannot be continued; its weights can still be
    # taken with init_from
    config = _smoke(steps=2, val_every=2, collapse_threshold=None)
    train(config, tmp_path / "a", mlflow_log=False)
    state = torch.load(tmp_path / "a" / "latest.pt", weights_only=True)
    longer = replace(config, steps=4)
    for name, optimizer_type in (("adam", "adam"), ("none", None)):
        other = {k: v for k, v in state.items() if k != "optimizer_type"}
        if optimizer_type is not None:
            other["optimizer_type"] = optimizer_type
        path = tmp_path / f"{name}.pt"
        torch.save(other, path)
        with pytest.raises(ValueError, match="Muon runs only"):
            check_source_checkpoint(path, longer, resume=True)
        check_source_checkpoint(path, longer, resume=False)  # the weights can be taken


def _train_script():
    return load_script("scripts/train.py", "train_script_loop")


def _clear_env(monkeypatch):
    for name in ("TABSIL_CONFIG_NAME", "TABSIL_STEPS", "TABSIL_EMSIZE", "TABSIL_N_CLS"):
        monkeypatch.delenv(name, raising=False)


def test_train_script_presets_are_the_tabsil_curriculum(monkeypatch):
    _clear_env(monkeypatch)
    script = _train_script()
    assert script.CONFIG_NAMES == ("smoke", "medium", "stage1", "stage2", "stage3")
    args = script.parse_args(["--config-name", "stage1"])
    assert script.build_config(args) == recipe.curriculum(1)
    # --steps sets the stage length and its warmup
    args = script.parse_args(["--config-name", "stage1", "--steps", "1000"])
    assert script.build_config(args) == recipe.curriculum(1, steps=1000)
    smoke = script.build_config(script.parse_args(["--config-name", "smoke"]))
    assert smoke == TrainConfig.smoke() == tabsil_smoke()
    medium = script.build_config(script.parse_args(["--config-name", "medium"]))
    assert medium == TrainConfig.medium()
    # later stages start from the previous stage's weights
    with pytest.raises(SystemExit):
        script.parse_args(["--config-name", "stage2"])
    args = script.parse_args(["--config-name", "stage2", "--init-run-id", "abc"])
    assert script.build_config(args) == recipe.curriculum(2)


def test_train_script_takes_only_a_source_run_of_the_same_structure(monkeypatch):
    from tabsil_train.hub import run_architecture

    _clear_env(monkeypatch)
    script = _train_script()
    stage1 = recipe.curriculum(1)
    params = {k: str(v) for k, v in vars(stage1).items()}
    source = run_architecture(params)
    assert {k: source[k] for k in recipe.STRUCTURE} == recipe.STRUCTURE
    args = script.parse_args(["--config-name", "stage2", "--init-run-id", "abc"])
    assert script.build_config(args, source) == recipe.curriculum(2)
    # a run of another size is refused
    other = run_architecture({**params, "column_blocks": "3", "row_blocks": "3"})
    with pytest.raises(SystemExit, match="tabsil-v1"):
        script.build_config(args, other)
    # smoke runs resume with their reduced size
    smoke = tabsil_smoke()
    source = run_architecture({k: str(v) for k, v in vars(smoke).items()})
    args = script.parse_args(["--config-name", "smoke", "--resume-run-id", "abc"])
    config = script.build_config(args, source)
    assert config == smoke and config.n_inducing == 8


def test_train_script_changes_the_model_size(monkeypatch):
    from tabsil_train.hub import run_architecture

    _clear_env(monkeypatch)
    script = _train_script()
    # the size arguments change the preset's structure; everything else stays the recipe's
    args = script.parse_args(["--config-name", "stage1", "--emsize", "192"])
    config = script.build_config(args)
    assert config == replace(recipe.curriculum(1), emsize=192)
    assert config.model_config().icl_dim == 4 * 192
    monkeypatch.setenv("TABSIL_EMSIZE", "512")
    monkeypatch.setenv("TABSIL_N_CLS", "2")
    config = script.build_config(script.parse_args(["--config-name", "stage1"]))
    assert (config.emsize, config.n_cls, config.model_config().icl_dim) == (512, 2, 1024)
    # the run logs them as params, so a later stage reads its structure back
    source = run_architecture({k: str(v) for k, v in vars(config).items()})
    assert (source["emsize"], source["n_cls"]) == (512, 2)
    monkeypatch.delenv("TABSIL_EMSIZE")
    monkeypatch.delenv("TABSIL_N_CLS")
    # a later stage from a run of another size needs the same values again
    args = script.parse_args(["--config-name", "stage2", "--init-run-id", "abc"])
    with pytest.raises(SystemExit, match="--emsize"):
        script.build_config(args, source)
    args = script.parse_args(
        ["--config-name", "stage2", "--init-run-id", "abc", "--emsize", "512", "--n-cls", "2"]
    )
    assert script.build_config(args, source) == replace(recipe.curriculum(2), emsize=512, n_cls=2)
    # a structure the model rejects stops before the run starts
    for bad, match in ((["--emsize", "130"], "does not divide"), (["--emsize", "100"], "RoPE")):
        with pytest.raises(SystemExit, match=match):
            script.build_config(script.parse_args(["--config-name", "stage1", *bad]))


def test_train_script_picks_the_prior_and_a_resumed_run_keeps_its_own(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.delenv("TABSIL_PRIOR_TYPE", raising=False)
    script = _train_script()
    prior = "graph_scm_hicard_missing"
    args = script.parse_args(["--config-name", "stage2", "--init-run-id", "abc"])
    assert script.build_config(args).prior_type == "graph_scm"
    # a new stage from a run's weights takes the given prior
    args = script.parse_args(
        ["--config-name", "stage2", "--init-run-id", "abc", "--prior-type", prior]
    )
    script.inherit_prior_type(args, {"prior_type": "graph_scm"})
    assert script.build_config(args) == replace(recipe.curriculum(2), prior_type=prior)
    monkeypatch.setenv("TABSIL_PRIOR_TYPE", prior)
    args = script.parse_args(["--config-name", "stage2", "--init-run-id", "abc"])
    assert script.build_config(args).prior_type == prior
    monkeypatch.delenv("TABSIL_PRIOR_TYPE")
    # a resumed run keeps its prior (runs without the param used graph_scm)
    args = script.parse_args(["--config-name", "stage2", "--resume-run-id", "abc"])
    script.inherit_prior_type(args, {"prior_type": prior})
    assert script.build_config(args).prior_type == prior
    args = script.parse_args(["--config-name", "stage2", "--resume-run-id", "abc"])
    script.inherit_prior_type(args, {})
    assert args.prior_type == "graph_scm"
    args = script.parse_args(
        ["--config-name", "stage2", "--resume-run-id", "abc", "--prior-type", prior]
    )
    with pytest.raises(SystemExit, match="prior_type"):
        script.inherit_prior_type(args, {"prior_type": "graph_scm"})
    with pytest.raises(SystemExit, match="prior_type"):
        script.build_config(script.parse_args(["--config-name", "smoke", "--prior-type", "nope"]))


def test_train_script_checks_the_source_checkpoint_before_the_run(tmp_path):
    # a checkpoint with a mismatched model config stops before opening an MLflow run
    script = _train_script()
    config = _smoke(steps=2, collapse_threshold=None)
    train(config, tmp_path / "a", mlflow_log=False)
    latest = str(tmp_path / "a" / "latest.pt")
    with pytest.raises(SystemExit, match="nhead"):
        script._check_source(latest, replace(config, nhead=4, steps=4), resume=True)
    with pytest.raises(SystemExit, match="nothing to train"):
        script._check_source(latest, config, resume=True)
    # a checkpoint without regression borders is not one of this model
    state = torch.load(latest, weights_only=True)
    state["borders"] = None
    torch.save(state, tmp_path / "no_borders.pt")
    with pytest.raises(SystemExit, match="regression borders"):
        script._check_source(str(tmp_path / "no_borders.pt"), config, resume=False)
    script._check_source(latest, config, resume=False)  # loading weights only ignores the step


def test_train_script_takes_weights_from_a_hub_folder(tmp_path, monkeypatch):
    # --init-hub: a save_pretrained folder (here local) becomes an init_from checkpoint
    from tabsil.hub import save_pretrained
    from tabsil.models import load_checkpoint

    _clear_env(monkeypatch)
    script = _train_script()
    config = _smoke(steps=2, collapse_threshold=None)
    train(config, tmp_path / "a", mlflow_log=False)
    model, borders = load_checkpoint(tmp_path / "a" / "latest.pt")
    save_pretrained(model, borders, tmp_path / "hub")
    path = script.hub_init_checkpoint(str(tmp_path / "hub"), config, tmp_path / "init.pt")
    result = train(config, tmp_path / "b", mlflow_log=False, init_from=path)
    trained, trained_borders = load_checkpoint(result["final_path"])
    assert torch.equal(trained_borders, borders)  # the borders carry over
    # another structure is refused before the run starts
    with pytest.raises(SystemExit, match="nhead"):
        script.hub_init_checkpoint(str(tmp_path / "hub"), replace(config, nhead=4), tmp_path / "x")
    # one weight source at a time; a later stage accepts --init-hub
    args = script.parse_args(["--config-name", "stage2", "--init-hub", "aiden-jeon/tabsil@x"])
    assert args.init_hub == "aiden-jeon/tabsil@x"
    for other in (["--init-run-id", "abc"], ["--resume-run-id", "abc"]):
        with pytest.raises(SystemExit):
            script.parse_args(["--config-name", "stage2", "--init-hub", "r", *other])


@pytest.mark.parametrize(
    "changes, match",
    [
        ({"min_rows": 1}, "min_rows"),
        ({"min_rows": 2048, "max_rows": 1024}, "min_rows"),
        ({"max_rows": 200_000}, "max_cells"),
        ({"min_train_fraction": 0.0}, "min_train_fraction"),
        ({"max_train_fraction": 1.0}, "max_train_fraction"),
        ({"max_features": 0}, "max_features"),
    ],
)
def test_bad_table_sizes_fail_when_the_config_is_built(changes, match):
    # used to crash inside the DataLoader worker on NaN conversion / division by zero
    with pytest.raises(ValueError, match=match):
        TrainConfig(**changes)


def test_classification_outputs_come_from_the_prior(monkeypatch):
    import types

    from tabsil_train import priors

    fake = types.SimpleNamespace(MAX_CLASSES=20, check_config=lambda config: None)
    monkeypatch.setitem(priors.PRIOR_TYPES, "fake", "fake_prior_module")
    monkeypatch.setitem(__import__("sys").modules, "fake_prior_module", fake)
    assert TrainConfig(prior_type="fake").n_out == 20
    assert TrainConfig().n_out == 10


def test_train_script_sets_the_tabicl_prior_options_from_env(monkeypatch):
    # env vars override TabICLv2 table size and prior settings; unset keeps the preset
    _clear_env(monkeypatch)
    script = _train_script()
    base = script.build_config(script.parse_args(["--config-name", "medium"]))
    assert base == TrainConfig.medium()
    env = {
        "TABSIL_MIN_ROWS": "400",
        "TABSIL_MAX_ROWS": "10240",
        "TABSIL_MIN_TRAIN_FRACTION": "0.79",
        "TABSIL_MAX_TRAIN_FRACTION": "0.81",
        "TABSIL_BINARY_PROB": "0.25",
        "TABSIL_FILTER_UNPREDICTABLE": "1",
    }
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    script = _train_script()
    config = script.build_config(script.parse_args(["--config-name", "medium"]))
    assert (config.min_rows, config.max_rows) == (400, 10240)
    assert (config.min_train_fraction, config.max_train_fraction) == (0.79, 0.81)
    assert config.binary_prob == 0.25 and config.filter_unpredictable
    # other settings are unchanged
    assert (
        replace(
            config,
            min_rows=base.min_rows,
            max_rows=base.max_rows,
            min_train_fraction=base.min_train_fraction,
            max_train_fraction=base.max_train_fraction,
            binary_prob=base.binary_prob,
        )
        == base
    )


def test_a_resumed_run_keeps_its_seed_steps_and_warmup(monkeypatch):
    _clear_env(monkeypatch)
    for name in ("TABSIL_SEED", "TABSIL_WARMUP_STEPS"):
        monkeypatch.delenv(name, raising=False)
    script = _train_script()
    params = {"seed": "1", "steps": "30000", "warmup_steps": "300"}
    args = script.parse_args(["--config-name", "stage1", "--resume-run-id", "abc"])
    script.inherit_schedule(args, params)
    config = script.build_config(args)
    assert (config.seed, config.steps, config.warmup_steps) == (1, 30000, 300)
    # values given on resume are used, but another seed is an error
    args = script.parse_args(
        ["--config-name", "stage1", "--resume-run-id", "abc", "--steps", "40000", "--seed", "1"]
    )
    script.inherit_schedule(args, params)
    assert (args.steps, args.seed, args.warmup_steps) == (40000, 1, 300)
    args = script.parse_args(["--config-name", "stage1", "--resume-run-id", "abc", "--seed", "2"])
    with pytest.raises(SystemExit, match="seed"):
        script.inherit_schedule(args, params)
    # runs without the params, and runs that only give their weights, keep the preset
    args = script.parse_args(["--config-name", "stage1", "--resume-run-id", "abc"])
    script.inherit_schedule(args, {})
    assert script.build_config(args) == recipe.curriculum(1)
    args = script.parse_args(["--config-name", "stage2", "--init-run-id", "abc"])
    script.inherit_schedule(args, params)
    assert script.build_config(args) == recipe.curriculum(2)
