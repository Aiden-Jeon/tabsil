"""Tests of the reference validation set (`tabsil_train.training.reference`): fixed tables that do
not depend on the run's seed or prior, validated as the `ref_*` metrics."""

import random
from dataclasses import replace

import numpy as np
import pytest
import torch
from train_helpers import load_script, tabsil_smoke

from tabsil_train.training import reference as R
from tabsil_train.training.config import TrainConfig
from tabsil_train.training.loop import train

SMALL = dict(
    min_rows=40,
    max_rows=40,
    max_features=5,
    max_cells=200,
    border_datasets=10,
    batch_size=4,
    micro_batch=4,
)


def _config(**changes) -> TrainConfig:
    return tabsil_smoke(**{**SMALL, "reference_val_tables": 16, **changes})


def _same(a, b) -> bool:
    return len(a) == len(b) and all(
        p.n_train == q.n_train
        and torch.equal(p.x.nan_to_num(-999.0), q.x.nan_to_num(-999.0))
        and torch.equal(p.y, q.y)
        for p, q in zip(a, b, strict=True)
    )


TASKS = ("classification", "regression")


def test_reference_tables_do_not_depend_on_the_seed_or_the_prior():
    base = R.reference_sets(_config(), TASKS, n_jobs=2)
    assert all(len(base[t]) == 2 for t in TASKS)
    assert all(b.x.shape[0] == R.REFERENCE_BATCH for t in TASKS for b in base[t])
    assert not _same(base["classification"], base["regression"])
    for changes in (
        {"seed": 7, "prior_type": "graph_scm_realism", "val_batch": 4, "micro_batch": 2},
        {"binary_prob": 0.0, "missing_dataset_prob": 0.0, "filter_unpredictable": False},
    ):
        other = R.reference_sets(_config(**changes), TASKS, n_jobs=2)
        assert all(_same(other[t], base[t]) for t in TASKS), changes
    # the table sizes are the run's
    other = R.reference_batches(_config(min_rows=48, max_rows=48), "regression")
    assert other[0].x.shape[1] == 48 and base["regression"][0].x.shape[1] == 40


def test_reference_tables_do_not_depend_on_the_process_count_and_extend():
    config = _config(reference_val_tables=48)  # two chunks of 4 batches and one of 2
    one = R.reference_batches(config, "regression", n_jobs=1)
    assert len(one) == 6
    assert _same(R.reference_batches(config, "regression", n_jobs=3), one)
    # a smaller set is the start of a larger one
    assert _same(R.reference_batches(_config(reference_val_tables=32), "regression"), one[:4])


def test_drawing_the_reference_tables_keeps_the_global_random_state():
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    R.reference_batches(_config(), "classification")
    after = (random.random(), np.random.random(), torch.rand(1).item())
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    assert after == (random.random(), np.random.random(), torch.rand(1).item())


def test_reference_val_tables_must_be_a_multiple_of_the_batch():
    assert TrainConfig().reference_val_tables == 0  # off by default
    with pytest.raises(ValueError, match="multiple of 8"):
        TrainConfig(reference_val_tables=12)
    with pytest.raises(ValueError, match="multiple of 8"):
        TrainConfig(reference_val_tables=-8)


def test_reference_validation_does_not_change_training(tmp_path, monkeypatch):
    logged = []
    monkeypatch.setattr(
        "tabsil_train.training.loop._validate_tasks",
        _recording(logged),
    )
    run = dict(steps=4, warmup_steps=1, log_every=2, val_every=2, checkpoint_every=4)
    off = train(_config(reference_val_tables=0, **run), tmp_path / "off", mlflow_log=False)
    on = train(_config(**run), tmp_path / "on", mlflow_log=False)
    assert off["history"] == on["history"]
    a = torch.load(tmp_path / "off" / "latest.pt", weights_only=True)["model"]
    b = torch.load(tmp_path / "on" / "latest.pt", weights_only=True)["model"]
    assert all(torch.equal(a[k], b[k]) for k in a)
    ref = [m for m in logged if "ref_loss" in m]
    assert len(ref) == 2  # every validation of the run with the set
    assert set(ref[0]) == {
        "ref_loss",
        "ref_loss_classification",
        "ref_acc_classification",
        "ref_loss_regression",
        "ref_loss_regression_train_rows",
    }
    # without the train-row scaling, the inference-form loss is a second computation
    assert ref[0]["ref_loss_regression_train_rows"] != ref[0]["ref_loss_regression"]
    assert all("val_loss_regression_train_rows" not in m for m in logged)


def test_a_scaled_run_reports_its_reference_loss_as_the_train_row_loss(tmp_path, monkeypatch):
    logged = []
    monkeypatch.setattr("tabsil_train.training.loop._validate_tasks", _recording(logged))
    run = dict(steps=2, warmup_steps=1, log_every=2, val_every=2, checkpoint_every=2)
    train(_config(target_train_rows=True, **run), tmp_path / "yctx", mlflow_log=False)
    (ref,) = [m for m in logged if "ref_loss" in m]
    assert ref["ref_loss_regression_train_rows"] == ref["ref_loss_regression"]
    assert ref["ref_loss"] == pytest.approx(
        (ref["ref_loss_classification"] + ref["ref_loss_regression"]) / 2
    )


def test_reference_tables_are_copied_out_of_shared_memory():
    sets = R.reference_sets(_config(), TASKS, n_jobs=2)
    assert not any(t.is_shared() for task in TASKS for b in sets[task] for t in (b.x, b.y))


def _recording(logged):
    from tabsil_train.training import loop

    original = loop._validate_tasks

    def validate(*args, **kwargs):
        out = original(*args, **kwargs)
        logged.append(dict(out))
        return out

    return validate


def test_the_digest_follows_the_tables():
    sets = R.reference_sets(_config(), TASKS, n_jobs=2)
    digest = R.reference_digest(sets)
    assert digest == R.reference_digest({t: list(sets[t]) for t in TASKS})
    assert digest != R.reference_digest({"classification": sets["classification"]})


def test_train_script_sets_the_reference_set_and_a_resumed_run_keeps_it(monkeypatch):
    for name in ("TABSIL_CONFIG_NAME", "TABSIL_STEPS", "TABSIL_REFERENCE_VAL_TABLES"):
        monkeypatch.delenv(name, raising=False)
    script = load_script("scripts/train.py", "train_script_reference")
    assert script.build_config(script.parse_args(["--config-name", "medium"])) == (
        TrainConfig.medium()
    )
    monkeypatch.setenv("TABSIL_REFERENCE_VAL_TABLES", "512")
    script = load_script("scripts/train.py", "train_script_reference")
    config = script.build_config(script.parse_args(["--config-name", "medium"]))
    assert config == replace(TrainConfig.medium(), reference_val_tables=512)
    monkeypatch.delenv("TABSIL_REFERENCE_VAL_TABLES")
    script = load_script("scripts/train.py", "train_script_reference")
    args = script.parse_args(["--config-name", "medium", "--resume-run-id", "abc"])
    script.inherit_reference_val(args, {"reference_val_tables": "256"})
    assert script.build_config(args).reference_val_tables == 256
    # a value given on resume is used
    args = script.parse_args(
        ["--config-name", "medium", "--resume-run-id", "abc", "--reference-val-tables", "64"]
    )
    script.inherit_reference_val(args, {"reference_val_tables": "256"})
    assert args.reference_val_tables == 64


def test_reference_tables_do_not_depend_on_the_hash_seed_of_the_training_process():
    import subprocess
    import sys

    code = (
        "from dataclasses import replace;"
        "from tabsil_train.training import reference as R;"
        "from tabsil_train.training.config import TrainConfig;"
        "c = replace(TrainConfig.smoke(), min_rows=40, max_rows=40, max_features=5, max_cells=200,"
        " border_datasets=10, batch_size=4, micro_batch=4, reference_val_tables=8);"
        "print(R.reference_digest({'regression': R.reference_batches(c, 'regression')}))"
    )
    digests = {
        subprocess.run(
            [sys.executable, "-c", code],
            env={**__import__("os").environ, "PYTHONHASHSEED": seed},
            check=True,
            capture_output=True,
            text=True,
        ).stdout.split()[-1]
        for seed in ("1", "2")
    }
    assert len(digests) == 1
