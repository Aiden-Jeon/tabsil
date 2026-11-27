"""Training speedup option tests: LAWA, TF32/compile settings, time to target."""

from dataclasses import replace

import pytest
import torch
import train_helpers as helpers

from tabsil import models as registry
from tabsil_train.training.config import TrainConfig
from tabsil_train.training.loop import _average_states, train


def test_options_default_off():
    config = TrainConfig()
    assert not (config.compile or config.tf32)
    assert config.weight_average == 0
    with pytest.raises(ValueError, match="weight_average"):
        TrainConfig(weight_average=-1)


def _smoke(**changes):
    return replace(
        helpers.tabsil_smoke(filter_unpredictable=False, border_datasets=10),
        steps=6,
        val_every=3,
        log_every=3,
        checkpoint_every=3,
        keep_every=None,
        collapse_threshold=None,
        **changes,
    )


def test_weight_average_saves_the_mean_of_recent_weights(tmp_path):
    config = _smoke(weight_average=2, weight_average_every=2)
    train(config, tmp_path, mlflow_log=False)
    averaged, _ = registry.load_checkpoint(tmp_path / "final_avg.pt")
    final, _ = registry.load_checkpoint(tmp_path / "final.pt")
    # The average differs from the last weights (mean of steps 4 and 6)
    assert any(
        not torch.equal(a, b)
        for a, b in zip(averaged.state_dict().values(), final.state_dict().values(), strict=True)
    )


def test_average_states_keeps_dtypes():
    a = {"w": torch.tensor([1.0, 3.0]), "n": torch.tensor(2)}
    b = {"w": torch.tensor([3.0, 5.0]), "n": torch.tensor(2)}
    out = _average_states([a, b])
    assert torch.equal(out["w"], torch.tensor([2.0, 4.0])) and out["n"].dtype == torch.int64


def test_tf32_is_a_no_op_on_cpu(tmp_path):
    base = train(_smoke(), tmp_path / "a", mlflow_log=False)
    fast = train(_smoke(tf32=True), tmp_path / "b", mlflow_log=False)
    assert base["history"]["train_loss"] == fast["history"]["train_loss"]
    torch.set_float32_matmul_precision("highest")  # don't leak into other tests


def test_train_script_takes_the_speed_options(monkeypatch):
    for name in ("TABSIL_COMPILE", "TABSIL_TF32", "TABSIL_WEIGHT_AVERAGE"):
        monkeypatch.delenv(name, raising=False)
    script = helpers.load_script("scripts/train.py", "train_script_speed")
    config = script.build_config(script.parse_args(["--config-name", "smoke"]))
    assert not (config.compile or config.tf32) and config.weight_average == 0
    monkeypatch.setenv("TABSIL_TF32", "1")
    args = script.parse_args(
        ["--config-name", "smoke", "--compile", "--weight-average", "5", "--val-every", "7"]
    )
    config = script.build_config(args)
    assert config.compile and config.tf32 and config.weight_average == 5
    assert config.val_every == 7


def test_resume_keeps_train_seconds(tmp_path):
    train(_smoke(stop_step=3), tmp_path / "a", mlflow_log=False)
    state = torch.load(tmp_path / "a" / "latest.pt", weights_only=True)
    assert state["train_seconds"] > 0
    # On resume, training time continues (does not restart from 0)
    resumed = tmp_path / "resumed"
    train(_smoke(), resumed, resume_from=tmp_path / "a" / "latest.pt", mlflow_log=False)
    later = torch.load(resumed / "latest.pt", weights_only=True)
    assert later["train_seconds"] > state["train_seconds"]


def test_weight_average_waits_for_a_full_window(tmp_path):
    # Until the window (K entries) is full, nothing is averaged or saved
    config = _smoke(weight_average=4, weight_average_every=2)  # 6 steps give only 3
    train(config, tmp_path, mlflow_log=False)
    assert not (tmp_path / "final_avg.pt").exists()


def test_tf32_is_restored_after_training(tmp_path):
    torch.set_float32_matmul_precision("highest")
    train(_smoke(tf32=True), tmp_path, mlflow_log=False)
    assert torch.get_float32_matmul_precision() == "highest"


def test_train_script_reads_env_flags_and_keeps_the_source_options(monkeypatch):
    script = helpers.load_script("scripts/train.py", "train_script_speed2")
    for value, expected in (("True", True), ("yes", True), ("OFF", False), ("", False)):
        monkeypatch.setenv("TABSIL_COMPILE", value)
        assert script._env_flag("TABSIL_COMPILE") is expected
    monkeypatch.setenv("TABSIL_COMPILE", "enabled")
    with pytest.raises(SystemExit, match="TABSIL_COMPILE"):
        script._env_flag("TABSIL_COMPILE")
    monkeypatch.delenv("TABSIL_COMPILE")
    for name in ("TABSIL_TF32", "TABSIL_WEIGHT_AVERAGE", "TABSIL_CONFIG_NAME"):
        monkeypatch.delenv(name, raising=False)
    # Original run params (strings): enabled options stay on when resuming; numbers take the
    # argument if given
    params = {"compile": "True", "weight_average": "10", "tf32": "False"}
    options = script.source_speed_options(params)
    assert options == {"compile": True, "weight_average": 10, "tf32": False}
    from tabsil_train.hub import run_architecture

    source = run_architecture({k: str(v) for k, v in vars(helpers.tabsil_smoke()).items()})
    args = script.parse_args(["--config-name", "smoke", "--resume-run-id", "r"])
    config = script.build_config(args, source, options)
    assert config.compile and config.weight_average == 10
    args = script.parse_args(
        ["--config-name", "smoke", "--resume-run-id", "r", "--weight-average", "3", "--tf32"]
    )
    config = script.build_config(args, source, options)
    assert config.weight_average == 3 and config.tf32
