"""Per-step-range config changes (`TrainConfig.phases`)."""

from dataclasses import replace

import helpers
import pytest
import torch

from tabsil.hub import run_architecture
from tabsil.training.config import TrainConfig, normalize_phases
from tabsil.training.loop import _phase_base_lr, train


def _smoke(**changes):
    return replace(
        helpers.tabsil_smoke(filter_unpredictable=False, border_datasets=10),
        steps=6,
        warmup_steps=1,
        val_every=3,
        log_every=1,
        checkpoint_every=3,
        keep_every=None,
        collapse_threshold=None,
        muon_weight_decay=0.0,
        **changes,
    )


PHASES = ((3, {"lr": 5e-5, "muon_weight_decay": 0.1}),)


def test_phase_value_follows_the_last_started_phase():
    config = replace(
        TrainConfig(lr=2e-4, muon_weight_decay=0.0),
        phases=((10, {"lr": 1e-4}), (20, {"muon_weight_decay": 0.1})),
    )
    assert config.phases == ((10, (("lr", 1e-4),)), (20, (("muon_weight_decay", 0.1),)))
    assert [config.phase_value("lr", s) for s in (-1, 9, 10, 25)] == [2e-4, 2e-4, 1e-4, 1e-4]
    assert [config.phase_value("muon_weight_decay", s) for s in (9, 19, 20)] == [0.0, 0.0, 0.1]
    # Round-trips from the MLflow param string to the same config
    assert normalize_phases(str(config.phases)) == config.phases
    assert normalize_phases("()") == () == normalize_phases("")


@pytest.mark.parametrize(
    "phases",
    [
        ((10, {"lr": 1e-4}), (10, {"lr": 5e-5})),  # start steps not increasing
        ((10, {"warmup_steps": 5}),),  # setting that cannot change
        ((10, {"lr": 0.0}),),
        ((10, {}),),
        ((10, {"muon_weight_decay": -0.1}),),
    ],
)
def test_phases_are_checked(phases):
    with pytest.raises(ValueError):
        TrainConfig(phases=phases)


def test_phase_base_lr_keeps_the_rewind_factor_and_its_floor():
    config = TrainConfig(lr=2e-4, phases=((10, {"lr": 1e-4}),))
    assert _phase_base_lr(config, 9, 1.0) == 2e-4 and _phase_base_lr(config, 10, 1.0) == 1e-4
    # A scale halved by collapse carries across phases (rewinding before a phase: phase lr x scale)
    assert _phase_base_lr(config, 15, 0.5) == pytest.approx(5e-5)
    assert _phase_base_lr(config, 5, 0.5) == pytest.approx(1e-4)
    # Entering a phase with a lowered scale does not go below rewind_min_lr (2.5e-5)
    assert _phase_base_lr(config, 15, 0.125) == pytest.approx(config.rewind_min_lr)
    # If the phase lr is already below the floor, use it
    low = TrainConfig(phases=((10, {"lr": 1e-5}),))
    assert _phase_base_lr(low, 15, 0.5) == pytest.approx(1e-5)


def test_phases_must_start_before_the_last_step():
    with pytest.raises(ValueError):
        TrainConfig(steps=100_000, phases=((125_000, {"lr": 1e-4}),))


def test_phases_take_a_dict_by_start_step():
    config = TrainConfig(phases={20: {"lr": 1e-4}, 10: {"muon_weight_decay": 0.1}})
    assert config.phases == ((10, (("muon_weight_decay", 0.1),)), (20, (("lr", 1e-4),)))


def test_training_switches_lr_and_weight_decay_at_the_phase(tmp_path):
    plain = train(_smoke(), tmp_path / "plain", mlflow_log=False)
    phased = train(_smoke(phases=PHASES), tmp_path / "phased", mlflow_log=False)
    # Identical up to the update where the phase starts (step 3), different after
    assert phased["history"]["train_loss"][:3] == plain["history"]["train_loss"][:3]
    assert phased["history"]["train_loss"][3:] != plain["history"]["train_loss"][3:]
    state = torch.load(tmp_path / "phased" / "latest.pt", weights_only=True)
    assert state["scheduler"]["base_lrs"][0] == pytest.approx(5e-5)
    groups = state["optimizer"]["param_groups"]
    assert all(g["weight_decay"] == 0.1 for g in groups)


def test_resume_enters_the_phase_at_the_resume_step(tmp_path):
    # Resuming from the checkpoint at the phase start step switches immediately
    train(_smoke(phases=PHASES, stop_step=3), tmp_path / "a", mlflow_log=False)
    before = torch.load(tmp_path / "a" / "latest.pt", weights_only=True)
    assert before["scheduler"]["base_lrs"][0] == pytest.approx(TrainConfig.smoke().lr)
    assert all(g["weight_decay"] == 0.0 for g in before["optimizer"]["param_groups"])
    train(
        _smoke(phases=PHASES),
        tmp_path / "b",
        resume_from=tmp_path / "a" / "latest.pt",
        mlflow_log=False,
    )
    after = torch.load(tmp_path / "b" / "latest.pt", weights_only=True)
    assert after["scheduler"]["base_lrs"][0] == pytest.approx(5e-5)
    assert all(g["weight_decay"] == 0.1 for g in after["optimizer"]["param_groups"])


def test_resume_past_a_phase_start_from_a_run_without_phases(tmp_path):
    # Resuming with phases from a checkpoint trained past the phase start (step 4) without
    # phases sets both lr and weight decay to the phase values
    train(_smoke(stop_step=4), tmp_path / "a", mlflow_log=False)
    assert torch.load(tmp_path / "a" / "latest.pt", weights_only=True)["step"] == 4
    train(
        _smoke(phases=PHASES),
        tmp_path / "b",
        resume_from=tmp_path / "a" / "latest.pt",
        mlflow_log=False,
    )
    after = torch.load(tmp_path / "b" / "latest.pt", weights_only=True)
    assert after["scheduler"]["base_lrs"][0] == pytest.approx(5e-5)
    assert all(g["weight_decay"] == 0.1 for g in after["optimizer"]["param_groups"])
    assert after["lr_scale"] == 1.0


def test_train_script_follows_the_source_phases(monkeypatch):
    for name in ("TABSIL_CONFIG_NAME", "TABSIL_LR"):
        monkeypatch.delenv(name, raising=False)
    script = helpers.load_script("scripts/train.py", "train_script_phases")
    # A run trained with phases (as the TabPFN-3.5 paper_run was), on the smoke structure
    config = replace(
        helpers.tabsil_smoke(),
        phases=((10, {"lr": 1e-4, "muon_weight_decay": 0.1}),),
    )
    # On resume, follow the original run's phases; a run without phases keeps the preset's
    # phases
    params = {k: str(v) for k, v in vars(config).items()}
    options = script.source_speed_options(params)
    assert options["phases"] == config.phases
    assert "phases" not in script.source_speed_options({**params, "phases": "()"})
    resume = script.parse_args(resume_args())
    resumed = script.build_config(resume, run_architecture(params), options)
    assert resumed.phases == config.phases
    # On resume, pre-phase lr continues the original run's lr (as set by --lr); --lr is rejected
    custom = {**params, "lr": "0.0003"}
    resumed = script.build_config(resume, run_architecture(custom), options)
    assert script.resume_phase_lr(resumed, resume, custom).lr == 3e-4
    with pytest.raises(SystemExit):
        script.resume_phase_lr(resumed, script.parse_args([*resume_args(), "--lr", "1e-4"]), custom)
    # For phases, the tag takes precedence over the param (it can change across resumes)
    tagged = script.source_params({"phases": "()"}, {"phases": str(config.phases)})
    assert normalize_phases(tagged["phases"]) == config.phases


def resume_args():
    return ["--config-name", "smoke", "--resume-run-id", "r"]
