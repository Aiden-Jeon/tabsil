"""Tests of the `column_row_transformer` model with the training stack: the smoke training config,
training, the curriculum and the train / eval scripts (the model alone: tests/)."""

import json
import types
from dataclasses import replace

import pytest
import torch
import train_helpers as helpers

from tabsil import hub
from tabsil import models as registry
from tabsil.models.column_row_transformer import (
    MAX_CLASSES,
    ColumnRowTransformer,
    ColumnRowTransformerConfig,
)
from tabsil.models.quantile_distribution import N_QUANTILES
from tabsil_train import hub as train_hub
from tabsil_train import recipe
from tabsil_train.training.config import TrainConfig
from tabsil_train.training.loop import check_checkpoint, train

TASKS = ("classification", "regression")


def _model(**changes):
    """The smoke-size TabSil model, with the zero-initialized layers filled."""
    torch.manual_seed(0)
    model = helpers.tabsil_smoke(**changes).build_model()
    # The last layer of each residual branch starts at zero, so inputs do not reach the output at
    # first. Fill it in to expose the paths
    with torch.no_grad():
        for p in model.parameters():
            if (p == 0).all():
                p.normal_(0, 0.1)
    return model.eval()


def _data(task, b=2, r=40, c=7, n_train=25):
    torch.manual_seed(1)
    x = torch.randn(b, r, c)
    x[0, 3, 2] = float("nan")
    y = (
        torch.randint(0, 3, (b, n_train)).float()
        if task == "classification"
        else torch.randn(b, n_train)
    )
    return x, y


def test_smoke_model_matches_the_inference_tests():
    # tests/helpers.py builds the smoke model without the training stack; it must stay this one
    config = TrainConfig.smoke().model_config()
    assert config == ColumnRowTransformerConfig(
        emsize=32, nlayers=2, nhead=2, n_inducing=8, column_blocks=2, row_blocks=2, max_features=8
    )
    assert config.n_out == 10


def test_curriculum_builds_the_published_structure():
    sizes = {k: v for k, v in recipe.STRUCTURE.items() if k != "model_type"}
    assert recipe.curriculum(1).model_config() == ColumnRowTransformerConfig(
        **sizes, max_features=100
    )


def test_smoke_config_builds_and_runs():
    config = TrainConfig.smoke()
    assert config.model_type == "column_row_transformer"
    assert (config.emsize, config.nlayers, config.nhead) == (32, 2, 2)
    assert (config.n_inducing, config.column_blocks, config.row_blocks) == (8, 2, 2)
    # the default config is the published structure
    assert {k: getattr(TrainConfig(), k) for k in recipe.STRUCTURE} == recipe.STRUCTURE
    torch.manual_seed(0)
    model = registry.randomize_zero_weights(config.build_model()).eval()
    x, yc = _data("classification")
    _, yr = _data("regression")
    with torch.no_grad():
        assert model(x, yc, task="classification").shape == (2, 15, 10)
        out = model(x, yr, task="regression")
    assert out.shape == (2, 15, N_QUANTILES) and torch.isfinite(out).all()


def test_train_save_load_and_hub(tmp_path):
    config = helpers.tabsil_smoke(steps=3)
    result = train(config, tmp_path, mlflow_log=False)
    model, borders = registry.load_checkpoint(result["final_path"])
    assert isinstance(model, ColumnRowTransformer) and borders is not None
    assert model.config == config.model_config()
    state = torch.load(tmp_path / "latest.pt", weights_only=True)
    check_checkpoint(state, config, "resume_from")
    with pytest.raises(ValueError, match="different model config"):
        check_checkpoint(state, replace(config, emsize=16), "resume_from")
    no_borders = {**state, "borders": None}
    with pytest.raises(ValueError, match="regression borders"):
        check_checkpoint(no_borders, config, "resume_from")

    loaded, loaded_borders = hub.from_pretrained(
        hub.save_pretrained(model, borders, tmp_path / "pretrained")
    )
    assert isinstance(loaded, ColumnRowTransformer) and torch.equal(loaded_borders, borders)
    for name, tensor in model.state_dict().items():
        assert torch.equal(loaded.state_dict()[name], tensor), name
    # Rebuild the architecture from training run params (training-script resume). The feature count
    # seen in training is not architecture, so it is not part of the run architecture
    params = {k: str(v) for k, v in vars(config).items()}
    source = train_hub.run_architecture(params)
    assert "max_features" not in source
    rebuilt = replace(TrainConfig(), **source, max_features=config.max_features)
    assert rebuilt.model_config() == config.model_config()
    assert len(result["history"]["val_loss"]) == 1


def _script(name):
    return helpers.load_script(f"scripts/{name}.py", f"{name}_script_column_row")


def _clear_env(monkeypatch):
    for name in ("CONFIG_NAME", "BATCH_SIZE", "STEPS"):
        monkeypatch.delenv(f"TABSIL_{name}", raising=False)


def test_train_script_builds_tabsil_and_reads_its_run(monkeypatch):
    _clear_env(monkeypatch)
    script = _script("train")
    args = script.parse_args(["--config-name", "stage1"])
    config = script.build_config(args)
    assert config == recipe.curriculum(1)
    params = {k: str(v) for k, v in vars(config).items()}
    args = script.parse_args(["--config-name", "stage1", "--resume-run-id", "r"])
    resumed = script.build_config(args, train_hub.run_architecture(params))
    assert resumed.model_config() == config.model_config()
    smoke = script.build_config(script.parse_args(["--config-name", "smoke"]))
    assert smoke == helpers.tabsil_smoke()
    # a run of another size is not taken
    other = {k: str(v) for k, v in vars(replace(config, n_cls=8, emsize=128)).items()}
    with pytest.raises(SystemExit, match="structure"):
        script.build_config(args, train_hub.run_architecture(other))
    with pytest.raises(SystemExit):
        script.parse_args(["--config-name", "stage2"])  # needs --init-run-id
    with pytest.raises(SystemExit):
        script.parse_args(["--task", "classification"])  # the flag is gone


def test_train_script_takes_the_batch_size_and_keeps_it_on_resume(monkeypatch):
    # Resuming follows the source run's batch unless the argument is given
    _clear_env(monkeypatch)
    script = _script("train")
    base = ["--config-name", "stage1"]
    assert script.build_config(script.parse_args(base)).batch_size == 256
    monkeypatch.setenv("TABSIL_BATCH_SIZE", "128")
    config = script.build_config(script.parse_args(base))
    assert config.batch_size == 128 and config.grad_accumulation(32) == 1
    assert config.for_world_size(64).micro_batch == 2
    monkeypatch.delenv("TABSIL_BATCH_SIZE")
    params = {k: str(v) for k, v in vars(config).items()}
    resume = ["--config-name", "stage1", "--resume-run-id", "r"]
    for extra, expected in (([], 128), (["--batch-size", "64"], 64)):
        args = script.parse_args(resume + extra)
        script.inherit_batch_size(args, params)
        resumed = script.build_config(args, train_hub.run_architecture(params))
        assert resumed.batch_size == expected
    # Taking only the weights keeps the preset value
    args = script.parse_args(["--config-name", "stage2", "--init-run-id", "r"])
    script.inherit_batch_size(args, params)
    assert args.batch_size is None


def test_eval_script_takes_tabsil_inference_defaults():
    script = _script("eval")
    assert script.default_inference(_model(), "classification") == {
        "n_estimators": 8,
        "member_kinds": ("standard",),
        "max_features_per_member": 768,
        "feature_subsampling": "balanced",
    }


def test_eval_n_estimators_sets_the_member_count(monkeypatch, tmp_path):
    script = _script("eval")
    model = _model()
    monkeypatch.setattr(script, "load_model", lambda path, suite, dst: (model, None))
    args = script.parse_args(["--classifier-path", "x.pt"])
    assert script.make_estimator_factory("classification", args).n_estimators == 8
    args = script.parse_args(["--classifier-path", "x.pt", "--n-estimators", "4"])
    assert script.make_estimator_factory("classification", args).n_estimators == 4


def test_arena_limits_follow_the_recipe():
    # The TabArena / BeyondArena size limits are TabSil's supported size
    import argparse as _argparse

    from tabsil.recipe import SUPPORTED_SIZE

    assert SUPPORTED_SIZE == {"max_train_rows": 1_000_000, "max_features": 6_000}
    script = _script("eval")
    args = _argparse.Namespace(max_train_rows=None, max_features=None)
    limits = script.arena_limits({"classification": (_model(), None)}, args)
    # Classes beyond the 10 target embeddings share embeddings, up to MAX_CLASSES (issue #66)
    assert limits["binary"].max_train_rows == 1_000_000
    assert limits["binary"].max_features == 6_000
    assert limits["binary"].max_classes == MAX_CLASSES == 160
    # With per-member feature selection there is no feature limit (--max-features sets one)
    wide = script.parse_args(["--max-features-per-member", "6000"])
    assert wide.max_features_per_member == 6000
    limits = script.arena_limits({"classification": (_model(), None)}, wide)
    assert limits["binary"].max_features is None and limits["binary"].max_train_rows == 1_000_000
    wide.max_features = 9000
    limits = script.arena_limits({"classification": (_model(), None)}, wide)
    assert limits["binary"].max_features == 9000


def test_recipe_and_config():
    config = recipe.curriculum(1)
    assert config.tasks == TASKS and config.rewind_lr_factor == 0.5
    with pytest.raises(ValueError, match="stage"):
        recipe.curriculum(4)
    with pytest.raises(ValueError, match="task"):
        config.make_prior("multitask", 0)


def test_eval_seed_changes_members_and_the_arena_fingerprint(monkeypatch, tmp_path):
    script = _script("eval")
    args = script.parse_args(["--suite", "classification", "--seed", "3"])
    assert args.seed == 3 and script.parse_args([]).seed == 0
    model = _model()
    monkeypatch.setattr(script, "load_model", lambda *a, **k: (model, None))
    args.classifier_path, args.out_dir = "unused", str(tmp_path)
    make = script.make_estimator_factory("classification", args)
    estimator = make(types.SimpleNamespace(categorical_indices=[]))
    assert estimator.random_state == 3
    # seed 0 (the default) is left out of the fingerprint, another seed is recorded
    paths = {"classification": "unused"}
    factories = {"classification": types.SimpleNamespace(model=model, n_estimators=8)}

    def fingerprint(argv):
        return json.loads(script.arena_fingerprint(script.parse_args(argv), paths, factories))

    assert fingerprint(["--seed", "0"]) == {"paths": paths, "n_estimators": {"classification": 8}}
    assert fingerprint(["--seed", "3"])["seed"] == 3
    assert fingerprint(["--max-features-per-member", "6000"])["max_features_per_member"] == 6000


def test_tabsil_default_feature_cap_in_eval(monkeypatch, tmp_path):
    # The eval script takes the default inference cap of 768 columns per member, balanced
    # (issue #99). Arguments come first
    model = _model()
    script = _script("eval")

    def factory(argv, task="classification"):
        monkeypatch.setattr(script, "load_model", lambda *a, **k: (model, None))
        paths = ["--classifier-path", "u", "--regressor-path", "u", "--out-dir", str(tmp_path)]
        args = script.parse_args([*paths, *argv])
        f = script.make_estimator_factory(task, args)
        return args, f

    def caps(argv):
        _, f = factory(argv)
        return f.max_features_per_member, f.feature_subsampling

    args, f = factory([])
    assert (f.max_features_per_member, f.feature_subsampling) == (768, "balanced")
    est = f(types.SimpleNamespace(categorical_indices=[]))
    assert est.max_features_per_member == 768 and est.feature_subsampling == "balanced"
    paths = {"classification": "u"}
    printed = json.loads(script.arena_fingerprint(args, paths, {"classification": f}))
    # Results computed with all columns are not reused
    assert printed["max_features_per_member"] == 768
    assert caps(["--max-features-per-member", "6000"]) == (6000, "balanced")
    assert caps(["--max-features-per-member", "0"]) == (None, "balanced")  # off: all columns
    assert caps(["--test-time", "greedy"]) == (None, "balanced")
    with pytest.raises(SystemExit):
        script.parse_args(["--max-features-per-member", "-1"])
    # Off keeps the old default fingerprint and the arena feature limit
    off, f_off = factory(["--max-features-per-member", "0"])
    printed = json.loads(script.arena_fingerprint(off, paths, {"classification": f_off}))
    assert "max_features_per_member" not in printed
    caps_off = script.feature_caps(off, {"classification": f_off})
    limits = script.arena_limits({"classification": (model, None)}, off, caps_off)
    assert limits["binary"].max_features == 6_000
