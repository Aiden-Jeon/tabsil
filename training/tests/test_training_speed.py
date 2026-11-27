"""Training speedup option tests: LAWA, TF32/compile settings, the collector freeze, time to
target."""

import gc
from dataclasses import replace

import mlflow
import pytest
import torch
import train_helpers as helpers

from tabsil import models as registry
from tabsil_train.training import loop
from tabsil_train.training.config import TrainConfig
from tabsil_train.training.loop import _average_states, train
from tabsil_train.training.speed import ValPoint, time_to_target, val_loss_at


def test_options_default_off():
    config = TrainConfig()
    assert not (config.compile or config.tf32)
    assert config.weight_average == 0
    assert config.gc_freeze  # the one speed setting that is on by default
    with pytest.raises(ValueError, match="weight_average"):
        TrainConfig(weight_average=-1)


def _smoke(**changes):
    settings = {
        "steps": 6,
        "val_every": 3,
        "log_every": 3,
        "checkpoint_every": 3,
        "keep_every": None,
        "collapse_threshold": None,
    }
    return replace(
        helpers.tabsil_smoke(filter_unpredictable=False, border_datasets=10),
        **{**settings, **changes},
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


def test_time_to_target():
    points = [ValPoint(1000, 100.0, 1.2), ValPoint(2000, 200.0, 1.0), ValPoint(3000, 300.0, 0.9)]
    assert val_loss_at(points, 2000) == 1.0
    assert time_to_target(points, 1.0).train_seconds == 200.0
    assert time_to_target(points, 0.5) is None
    with pytest.raises(ValueError, match="step 1500"):
        val_loss_at(points, 1500)


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


def _same(a, b) -> bool:
    """Whether two checkpoint values are equal, tensors bit for bit."""
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b, strict=True))
    return a == b


def _record_collector(monkeypatch) -> list:
    """Record the calls of `gc.collect`, `gc.freeze` and `gc.unfreeze` (still made) as (name,
    frozen objects after the call)."""
    calls = []

    def recorded(name):
        original = getattr(gc, name)

        def call(*args, **kwargs):
            out = original(*args, **kwargs)
            calls.append((name, gc.get_freeze_count()))
            return out

        return call

    for name in ("collect", "freeze", "unfreeze"):
        monkeypatch.setattr(gc, name, recorded(name))
    return calls


def test_gc_freeze_runs_at_every_log_step_and_does_not_change_the_run(tmp_path, monkeypatch):
    # The process pool that draws the regression borders gets the same hash seed in both runs
    monkeypatch.setenv("PYTHONHASHSEED", "1")
    monkeypatch.chdir(tmp_path)  # the local artifact store is a folder of the working directory
    assert gc.get_freeze_count() == 0
    calls = _record_collector(monkeypatch)
    # Log steps at 2, 4 and 6, with a validation and a checkpoint between them and the collapse
    # guard's snapshots on
    config = _smoke(log_every=2, collapse_threshold=0.5, snapshot_every=2)
    results, recorded, params = {}, {}, {}
    prev_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    try:
        mlflow.set_experiment("test-gc-freeze")
        for on in (False, True):
            calls.clear()
            with mlflow.start_run() as run:
                results[on] = train(replace(config, gc_freeze=on), tmp_path / str(on))
            recorded[on] = list(calls)
            params[on] = mlflow.MlflowClient().get_run(run.info.run_id).data.params
    finally:
        mlflow.set_tracking_uri(prev_uri)

    # On: one freeze per log step, each right after a collection (garbage is not frozen) and
    # holding the live objects. The end hands them back to the collector
    names = [name for name, _ in recorded[True]]
    freezes = [i for i, name in enumerate(names) if name == "freeze"]
    assert len(freezes) == 3
    assert all(names[i - 1] == "collect" and recorded[True][i][1] > 0 for i in freezes)
    assert names.count("unfreeze") == 1 and names.index("unfreeze") > freezes[-1]
    assert recorded[True][names.index("unfreeze")] == ("unfreeze", 0)
    assert gc.get_freeze_count() == 0
    # Off: the collector is left alone
    assert not {"freeze", "unfreeze"} & {name for name, _ in recorded[False]}
    # The run's params name the setting
    assert params[True]["gc_freeze"] == "True" and params[False]["gc_freeze"] == "False"

    # The same run: every logged train and validation loss, and the weights, the optimizer state
    # and the schedule bit for bit. The training time is the wall clock's
    assert results[True]["history"] == results[False]["history"]
    assert results[True]["history"]["step"] == [2, 4, 6]
    assert results[True]["history"]["val_step"] == [3, 6]
    state_on = torch.load(tmp_path / "True" / "latest.pt", weights_only=True)
    state_off = torch.load(tmp_path / "False" / "latest.pt", weights_only=True)
    assert state_on["step"] == 6
    assert {"model", "optimizer", "scheduler", "borders"} <= set(state_on)
    for key in set(state_on) - {"train_seconds"}:
        assert _same(state_on[key], state_off[key]), key


def test_gc_freeze_is_undone_when_training_fails(tmp_path, monkeypatch):
    def failing(*args, **kwargs):
        gc.freeze()  # as the loop does at a log step
        raise RuntimeError("training failed")

    monkeypatch.setattr(loop, "_train", failing)
    with pytest.raises(RuntimeError, match="training failed"):
        train(_smoke(), tmp_path, mlflow_log=False)
    assert gc.get_freeze_count() == 0


def test_train_script_turns_gc_freeze_off(monkeypatch):
    for name in ("TABSIL_GC_FREEZE", "TABSIL_CONFIG_NAME"):
        monkeypatch.delenv(name, raising=False)
    script = helpers.load_script("scripts/train.py", "train_script_gc_freeze")
    base = ["--config-name", "smoke"]
    assert script.build_config(script.parse_args(base)).gc_freeze  # on unless turned off
    assert not script.build_config(script.parse_args([*base, "--no-gc-freeze"])).gc_freeze
    for value, expected in (("0", False), ("off", False), ("1", True), ("", True)):
        monkeypatch.setenv("TABSIL_GC_FREEZE", value)
        assert script.build_config(script.parse_args(base)).gc_freeze is expected
    # an argument wins over the variable
    monkeypatch.setenv("TABSIL_GC_FREEZE", "0")
    assert script.build_config(script.parse_args([*base, "--gc-freeze"])).gc_freeze
    monkeypatch.setenv("TABSIL_GC_FREEZE", "maybe")
    with pytest.raises(SystemExit, match="TABSIL_GC_FREEZE"):
        script.parse_args(base)
    # a resumed run does not take the setting from the run it continues (set per chunk)
    monkeypatch.delenv("TABSIL_GC_FREEZE")
    from tabsil_train.hub import run_architecture

    smoke = helpers.tabsil_smoke(gc_freeze=False)
    params = {k: str(v) for k, v in vars(smoke).items()}
    args = script.parse_args([*base, "--resume-run-id", "r"])
    config = script.build_config(
        args, run_architecture(params), script.source_speed_options(params)
    )
    assert config.gc_freeze
