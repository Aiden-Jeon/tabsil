"""A resumed run keeps the table sizes and prior settings it was trained with: `DATA_OPTIONS` and
`resume_data_options` of the training script (`scripts/train.py`).

No model is trained here. A run is the dict of its MLflow params (strings, as MLflow stores them),
and where `main` is called, MLflow, the checkpoint and the training loop are replaced.
"""

import os
import re
import types
from dataclasses import asdict, replace

import pytest
from train_helpers import load_script

from tabsil_train import recipe
from tabsil_train.priors import PRIOR_TYPES, prior_module
from tabsil_train.training.config import TASKS, TrainConfig

RESUME = ["--resume-run-id", "abc"]
# The ten settings as a stage-2 run records them: the MLflow params of `tabsil-v2-s2-l20`
# (893293ed7d8d419780be2d178b8cc713), read on 2026-11-27. Ints, floats and bools are all strings
STAGE2_RUN = {
    "min_rows": "400",
    "max_rows": "10240",
    "min_train_fraction": "0.79",
    "max_train_fraction": "0.81",
    "binary_prob": "0.5",
    "filter_unpredictable": "True",
    "max_features": "100",
    "log_uniform_rows": "True",
    "max_cells": "1024000",
    "missing_dataset_prob": "0.3",
}
# A run whose ten settings all differ from every preset's (the last four have no argument, so
# only a config built in Python has them)
CUSTOM = {
    "min_rows": 400,
    "max_rows": 2048,
    "min_train_fraction": 0.5,
    "max_train_fraction": 0.7,
    "binary_prob": 0.25,
    "filter_unpredictable": False,
    "max_features": 50,
    "log_uniform_rows": True,
    "max_cells": 204_800,
    "missing_dataset_prob": 0.1,
}


@pytest.fixture
def script(monkeypatch):
    # every argument of the script also reads an environment variable
    for name in [name for name in os.environ if name.startswith("TABSIL_")]:
        monkeypatch.delenv(name)
    return load_script("scripts/train.py", "train_script_resume_data")


def _params(config: TrainConfig) -> dict[str, str]:
    """The params the training loop logs for `config` (MLflow stores `str(value)`)."""
    return {name: str(value) for name, value in asdict(config).items()}


def _data(script, config: TrainConfig) -> dict:
    return {name: getattr(config, name) for name in script.DATA_OPTIONS}


def _main_config(script, monkeypatch, tmp_path, argv, params) -> TrainConfig:
    """The config `main` hands to the training loop for a command that resumes, or takes the
    weights of, a run with `params`. The checkpoint download fails the test if `main` gets that
    far after it should have stopped."""
    run = types.SimpleNamespace(data=types.SimpleNamespace(params=params, tags={}))
    client = types.SimpleNamespace(get_run=lambda run_id: run)
    monkeypatch.setattr(script.mlflow, "MlflowClient", lambda: client)
    seen = {"downloads": 0}

    def download(**kwargs):
        seen["downloads"] += 1
        return str(tmp_path / "checkpoint.pt")

    def fake_train(config, out_dir, **kwargs):
        seen["config"] = config
        return {"history": {}, "final_path": None}

    monkeypatch.setattr(script.mlflow.artifacts, "download_artifacts", download)
    # the checkpoint's base lr is the config's (no lr was given)
    monkeypatch.setattr(script, "_check_source", lambda path, config, resume: config.lr)
    monkeypatch.setattr(script, "train", fake_train)
    monkeypatch.setenv("RANK", "1")  # not rank 0, so no MLflow run is opened
    try:
        script.main([*argv, "--out-dir", str(tmp_path / "out")])
    except SystemExit:
        assert seen["downloads"] == 0  # stopped before the checkpoint download
        raise
    return seen["config"]


class _Reads:
    """A training config that records which fields are read from it."""

    def __init__(self, config: TrainConfig) -> None:
        self.config, self.names = config, set()

    def __getattr__(self, name: str):
        self.names.add(name)
        return getattr(self.config, name)


def test_data_options_are_the_config_fields_the_priors_read(script):
    # every registered prior, both tasks: a field the prior reads from the training config is kept
    # by a resumed run, either here or as a recipe option (`target_train_rows`)
    read = set()
    for prior_type in PRIOR_TYPES:
        for task in TASKS:
            config = _Reads(TrainConfig(prior_type=prior_type))
            prior_module(prior_type).make_prior(config, task, 0)
            read |= config.names
    assert read == set(script.DATA_OPTIONS) | {"target_train_rows"}
    assert "target_train_rows" in script.RECIPE_OPTIONS
    assert len(set(script.DATA_OPTIONS)) == len(script.DATA_OPTIONS) == 10
    assert set(STAGE2_RUN) == set(CUSTOM) == set(script.DATA_OPTIONS)


def test_unset_settings_are_the_resumed_runs(script, monkeypatch, tmp_path):
    run = replace(TrainConfig.medium(), **CUSTOM)
    args = script.parse_args(["--config-name", "medium", *RESUME])
    assert script.resume_data_options(args, _params(run)) == CUSTOM
    # `main` builds the run's config again from the preset and the run's params
    argv = ["--config-name", "medium", *RESUME]
    assert _main_config(script, monkeypatch, tmp_path, argv, _params(run)) == run
    # also when resuming from a kept step, which writes to a new run
    argv = [*argv, "--resume-step", "5000"]
    assert _main_config(script, monkeypatch, tmp_path, argv, _params(run)) == run
    # a real run's params: what a stage-2 run records is what the stage-2 preset has
    stage2 = recipe.curriculum(2)
    assert {name: str(value) for name, value in _data(script, stage2).items()} == STAGE2_RUN
    args = script.parse_args(["--config-name", "stage2", *RESUME])
    assert script.resume_data_options(args, STAGE2_RUN) == _data(script, stage2)


@pytest.mark.parametrize(
    "flag, field, given, saved",
    [
        ("--min-rows", "min_rows", "512", "1024"),
        ("--max-rows", "max_rows", "4096", "1024"),
        ("--min-train-fraction", "min_train_fraction", "0.4", "0.3"),
        ("--max-train-fraction", "max_train_fraction", "0.95", "0.9"),
        ("--binary-prob", "binary_prob", "0.25", "0.5"),
    ],
)
def test_a_different_value_stops_and_names_the_field_and_both_values(
    script, monkeypatch, tmp_path, flag, field, given, saved
):
    params = _params(recipe.curriculum(1))
    message = re.escape(f"{flag} {given} differs from the resumed run's {field} {saved}")
    args = script.parse_args(["--config-name", "stage1", *RESUME, flag, given])
    with pytest.raises(SystemExit, match=message):
        script.resume_data_options(args, params)
    # `main` stops before it downloads the checkpoint (checked in `_main_config`)
    with pytest.raises(SystemExit, match=message):
        _main_config(
            script, monkeypatch, tmp_path, ["--config-name", "stage1", *RESUME, flag, given], params
        )
    # the same through the environment variable, which the message names
    env = "TABSIL_" + field.upper()
    monkeypatch.setenv(env, given)
    args = script.parse_args(["--config-name", "stage1", *RESUME])
    with pytest.raises(SystemExit, match=message) as stopped:
        script.resume_data_options(args, params)
    assert env in str(stopped.value) and "--init-run-id" in str(stopped.value)


def test_the_same_values_pass(script, monkeypatch, tmp_path):
    run = replace(recipe.curriculum(2), min_rows=500, binary_prob=0.25)
    same = ["--min-rows", "500", "--max-rows", "10240", "--binary-prob", "0.25"]
    same += ["--min-train-fraction", "0.79", "--max-train-fraction", "0.81"]
    argv = ["--config-name", "stage2", *RESUME, *same, "--filter-unpredictable"]
    assert script.resume_data_options(script.parse_args(argv), _params(run)) == _data(script, run)
    assert _main_config(script, monkeypatch, tmp_path, argv, _params(run)) == run
    # and through the environment, as a chunk submitted again with its overrides gives them
    env = {"MIN_ROWS": "500", "BINARY_PROB": "0.25", "FILTER_UNPREDICTABLE": "1"}
    for name, value in env.items():
        monkeypatch.setenv("TABSIL_" + name, value)
    argv = ["--config-name", "stage2", *RESUME]
    assert _main_config(script, monkeypatch, tmp_path, argv, _params(run)) == run


@pytest.mark.parametrize(
    "env",
    [
        # phase 1B (`tabsil-v2-1b-c-s0`) and L20 (`tabsil-v2-s2-l20`): their `env.TABSIL_*` tags
        {
            "TABSIL_CONFIG_NAME": "stage1",
            "TABSIL_STEPS": "30000",
            "TABSIL_BATCH_SIZE": "256",
            "TABSIL_REFERENCE_VAL_TABLES": "512",
            "TABSIL_NUM_WORKERS": "16",
            "TABSIL_COMPILE": "1",
            "TABSIL_TF32": "1",
            "TABSIL_MUON_DISTRIBUTED": "all",
            "TABSIL_SEED": "0",
        },
        {
            "TABSIL_CONFIG_NAME": "stage2",
            "TABSIL_STEPS": "20000",
            "TABSIL_INIT_HUB": "aiden-jeon/tabsil@stage1-150k",
            "TABSIL_MICRO_BATCH": "2",
            "TABSIL_PRIOR_TYPE": "graph_scm_hicard_missing",
            "TABSIL_NUM_WORKERS": "16",
            "TABSIL_COMPILE": "1",
            "TABSIL_TF32": "1",
            "TABSIL_MUON_DISTRIBUTED": "all",
            "TABSIL_SEED": "0",
            "TABSIL_WEIGHT_AVERAGE": "10",
            "TABSIL_WEIGHT_AVERAGE_EVERY": "100",
        },
    ],
)
def test_a_chunk_submitted_again_with_the_same_settings_trains_as_before(
    script, monkeypatch, tmp_path, env
):
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    first = script.build_config(script.parse_args([]))  # the first chunk, which logs the params
    # the next chunk: the same job with the run to resume in place of the weights to start from
    monkeypatch.delenv("TABSIL_INIT_HUB", raising=False)
    monkeypatch.setenv("TABSIL_RESUME_RUN_ID", "abc")
    monkeypatch.setenv("TABSIL_STOP_STEP", "10000")
    config = _main_config(script, monkeypatch, tmp_path, [], _params(first))
    assert config == replace(first, stop_step=10_000)


def test_taking_only_the_weights_is_not_affected(script, monkeypatch, tmp_path):
    # --init-run-id starts a new run: the preset's tables, or the given ones
    params = _params(replace(recipe.curriculum(1), **CUSTOM))
    argv = ["--config-name", "stage2", "--init-run-id", "abc"]
    assert script.resume_data_options(script.parse_args(argv), params) == {}
    assert _main_config(script, monkeypatch, tmp_path, argv, params) == recipe.curriculum(2)
    argv = [*argv, "--min-rows", "500", "--binary-prob", "0.1"]
    expected = replace(recipe.curriculum(2), min_rows=500, binary_prob=0.1)
    assert _main_config(script, monkeypatch, tmp_path, argv, params) == expected
    # a run that only gives its weights need not have recorded the settings either
    assert _main_config(script, monkeypatch, tmp_path, argv, {}) == expected


def test_on_off_settings_follow_the_resumed_run(script, monkeypatch, tmp_path):
    argv = ["--config-name", "stage1", *RESUME]
    off = _params(replace(recipe.curriculum(1), filter_unpredictable=False))
    on = _params(recipe.curriculum(1))
    # the filter is on in every preset; a run that had it off keeps it off
    assert not _main_config(script, monkeypatch, tmp_path, argv, off).filter_unpredictable
    assert _main_config(script, monkeypatch, tmp_path, argv, on).filter_unpredictable
    # the flag can only turn it on, so turning on what the run had off stops the script
    flag = [*argv, "--filter-unpredictable"]
    with pytest.raises(SystemExit, match="--filter-unpredictable is off in the resumed run"):
        script.resume_data_options(script.parse_args(flag), off)
    assert _main_config(script, monkeypatch, tmp_path, flag, on).filter_unpredictable
    monkeypatch.setenv("TABSIL_FILTER_UNPREDICTABLE", "1")
    with pytest.raises(SystemExit, match="TABSIL_FILTER_UNPREDICTABLE"):
        _main_config(script, monkeypatch, tmp_path, argv, off)
    monkeypatch.delenv("TABSIL_FILTER_UNPREDICTABLE")
    # log_uniform_rows has no argument: it is the run's, whatever the preset has
    assert not recipe.curriculum(1).log_uniform_rows and recipe.curriculum(2).log_uniform_rows
    log_uniform = _params(replace(recipe.curriculum(1), log_uniform_rows=True))
    assert _main_config(script, monkeypatch, tmp_path, argv, log_uniform).log_uniform_rows
    uniform = _params(replace(recipe.curriculum(2), log_uniform_rows=False))
    argv = ["--config-name", "stage2", *RESUME]
    assert not _main_config(script, monkeypatch, tmp_path, argv, uniform).log_uniform_rows


def test_a_run_that_did_not_record_the_settings_stops(script, monkeypatch, tmp_path):
    argv = ["--config-name", "stage1", *RESUME]
    stage1 = _params(recipe.curriculum(1))
    # a run of older code, or a run that logged no params: nothing says which tables it saw
    others = {k: v for k, v in stage1.items() if k not in script.DATA_OPTIONS}
    for params in ({}, others):
        with pytest.raises(SystemExit, match="did not record") as stopped:
            _main_config(script, monkeypatch, tmp_path, argv, params)
        assert all(name in str(stopped.value) for name in script.DATA_OPTIONS)
        assert "--init-run-id" in str(stopped.value)
    # one missing setting is enough, and it is the one named
    for missing in ("max_cells", "min_rows"):
        params = {k: v for k, v in stage1.items() if k != missing}
        with pytest.raises(SystemExit, match="did not record") as stopped:
            _main_config(script, monkeypatch, tmp_path, argv, params)
        named = [name for name in script.DATA_OPTIONS if name in str(stopped.value)]
        assert named == [missing]
    # giving the value does not replace the record: it cannot be compared with anything
    params = {k: v for k, v in stage1.items() if k != "min_rows"}
    with pytest.raises(SystemExit, match="did not record"):
        _main_config(script, monkeypatch, tmp_path, [*argv, "--min-rows", "1024"], params)


def test_another_config_name_keeps_the_runs_tables(script, monkeypatch, tmp_path):
    stage1, stage2 = recipe.curriculum(1, steps=30_000), recipe.curriculum(2)
    assert _data(script, stage1) != _data(script, stage2)
    # a stage-1 run resumed with the stage-2 preset trains on stage-1 tables, and the reverse
    argv = ["--config-name", "stage2", *RESUME]
    config = _main_config(script, monkeypatch, tmp_path, argv, _params(stage1))
    assert _data(script, config) == _data(script, stage1)
    assert (config.min_rows, config.max_rows, config.max_cells) == (1024, 1024, 102_400)
    assert not config.log_uniform_rows
    argv = ["--config-name", "stage1", *RESUME]
    config = _main_config(script, monkeypatch, tmp_path, argv, _params(stage2))
    assert _data(script, config) == _data(script, stage2)
    # the stage-3 preset on a stage-2 run: the longer tables of stage 3 do not come in
    argv = ["--config-name", "stage3", *RESUME]
    config = _main_config(script, monkeypatch, tmp_path, argv, _params(stage2))
    assert (config.max_rows, config.max_cells) == (10_240, 1_024_000)
    # an override that belongs to the other preset is a different value, so it stops
    argv = ["--config-name", "stage2", *RESUME, "--max-rows", "10240"]
    with pytest.raises(SystemExit, match="max_rows 1024"):
        _main_config(script, monkeypatch, tmp_path, argv, _params(stage1))


@pytest.mark.parametrize(
    "fractions, binary_prob",
    [
        ((0.1 + 0.2, 0.9), 1 / 3),  # 0.30000000000000004: the param keeps every digit
        ((1 / 3, 2 / 3), 0.0),
        ((1e-05, 0.5), 1.0),
        ((0.79, 0.81), 1),  # an int in a float field is logged as "1"
    ],
)
def test_params_are_compared_as_values_of_the_fields_type(script, fractions, binary_prob):
    low, high = fractions
    run = TrainConfig(min_train_fraction=low, max_train_fraction=high, binary_prob=binary_prob)
    params = _params(run)
    # a float written by `str` reads back as the same float, and equals the argument's value
    argv = ["--config-name", "stage1", *RESUME, "--min-rows", "1024"]
    argv += ["--min-train-fraction", repr(low), "--max-train-fraction", repr(high)]
    argv += ["--binary-prob", repr(float(binary_prob))]
    data = script.resume_data_options(script.parse_args(argv), params)
    assert data == _data(script, run)
    assert (data["min_train_fraction"], data["max_train_fraction"]) == (low, high)
    for name, kind in (("min_rows", int), ("binary_prob", float), ("log_uniform_rows", bool)):
        assert type(data[name]) is kind


def test_other_spellings_of_a_value_pass_and_bad_params_stop(script):
    params = _params(recipe.curriculum(1))
    argv = ["--config-name", "stage1", *RESUME]
    for spelling in ("0.5", ".5", "0.50", "5e-1"):
        args = script.parse_args([*argv, "--binary-prob", spelling])
        assert script.resume_data_options(args, params)["binary_prob"] == 0.5
    # a bool param is True or False: anything else stops instead of counting as off
    args = script.parse_args(argv)
    for name, text in (("filter_unpredictable", "1"), ("log_uniform_rows", "true")):
        with pytest.raises(SystemExit, match=f"{name} param"):
            script.resume_data_options(args, {**params, name: text})
    for name, text in (("max_rows", "1024.0"), ("binary_prob", "half")):
        with pytest.raises(SystemExit, match=f"{name} param"):
            script.resume_data_options(args, {**params, name: text})
