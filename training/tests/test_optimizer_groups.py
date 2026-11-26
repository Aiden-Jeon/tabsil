"""Tests of the Muon + AdamW optimizer (`TrainConfig.optimizer="muon_adamw"`, `training.muon`):
Kumo-Tabular's split of the parameters, the AdamW update, weight decay phases, checkpoints and
resume, and the training script."""

from dataclasses import replace

import pytest
import torch
import train_helpers as helpers

from tabsil_train.training.config import TrainConfig
from tabsil_train.training.loop import _set_muon_weight_decay, check_source_checkpoint, train
from tabsil_train.training.muon import (
    ADAMW_WEIGHT_DECAY,
    Muon,
    adamw_parameters,
    muon_parameters,
)


def _model():
    torch.manual_seed(0)
    return helpers.tabsil_smoke().build_model()


def _kinds(model) -> dict[str, str]:
    decay, no_decay = adamw_parameters(model)
    return {
        n: "decay" if id(p) in decay else "no_decay" if id(p) in no_decay else "muon"
        for n, p in model.named_parameters()
    }


def test_the_split_follows_kumo_tabular():
    kinds = _kinds(_model())
    # Muon: the Linear weights of the stages
    for name in (
        "cell_encoder.fourier.weight",
        "column_blocks.0.summarize.qkv.weight",
        "row_blocks.0.attention.qkv.weight",
        "icl_layers.0.attention.out.weight",
    ):
        assert kinds[name] == "muon", name
    # AdamW with weight decay: target encoders, output layers, logit scales, other matrices
    for name in (
        "cell_target.classification.encode.weight",
        "icl_target.regression.encode.weight",
        "heads.regression.weight",
        "heads.classification.q_projection.weight",
        "icl_layers.0.attention.log_scale.head_scale",
        "heads.classification.log_scale.head_scale",
        "cls_tokens",
        "column_blocks.0.inducing",
    ):
        assert kinds[name] == "decay", name
    assert all(kinds[n] == "decay" for n in kinds if n.startswith("pre_head") and "mlp" in n)
    gates = [n for n in kinds if "log_scale.gate" in n]
    assert gates and all(kinds[n] == "decay" for n in gates)
    # AdamW without weight decay: the other 1-D parameters
    for name in ("cell_encoder.frequencies", "heads.regression.bias", "icl_norm.weight"):
        assert kinds[name] == "no_decay", name
    # every Muon parameter is a matrix, and nothing outside them is Muon's
    model = _model()
    muon = [p for n, p in model.named_parameters() if kinds[n] == "muon"]
    assert all(p.ndim == 2 for p in muon)


def test_the_optimizer_holds_every_parameter_once():
    model = _model()
    opt = Muon.for_model(model, lr=1e-3, weight_decay=0.02, adamw=True)
    seen = [id(p) for g in opt.param_groups for p in g["params"]]
    assert sorted(seen) == sorted(id(p) for p in model.parameters())
    adam = [g for g in opt.param_groups if g.get("adamw")]
    assert [g["weight_decay"] for g in adam] == [ADAMW_WEIGHT_DECAY, 0.0]
    assert all(g["betas"] == (0.9, 0.999) and g["eps"] == 1e-8 for g in adam)
    muon = [g for g in opt.param_groups if not g.get("adamw")]
    assert [g["split"] for g in muon] == [1, 3] and all(g["weight_decay"] == 0.02 for g in muon)
    # the default optimizer is unchanged: no AdamW groups and no new group keys
    plain = Muon.for_model(model, lr=1e-3)
    assert all("adamw" not in g for g in plain.param_groups)
    assert [p for g in plain.param_groups for p in g["params"]] == [
        p for s in (1, 3) for p, split in muon_parameters(model) if split == s
    ]


def _grads(model, step):
    gen = torch.Generator().manual_seed(step)
    for p in model.parameters():
        p.grad = torch.randn(p.shape, generator=gen)


def test_adamw_part_is_torch_adamw_and_muon_part_is_muon():
    model, adamw_model, muon_model = _model(), _model(), _model()
    opt = Muon.for_model(model, lr=1e-2, weight_decay=0.01, adamw=True)
    decay, no_decay = adamw_parameters(adamw_model)
    reference = torch.optim.AdamW(
        [
            {"params": [p for p in adamw_model.parameters() if id(p) in decay]},
            {"params": [p for p in adamw_model.parameters() if id(p) in no_decay]},
        ],
        lr=1e-2,
        weight_decay=ADAMW_WEIGHT_DECAY,
        foreach=False,
    )
    reference.param_groups[1]["weight_decay"] = 0.0
    plain = Muon.for_model(muon_model, lr=1e-2, weight_decay=0.01)
    for step in range(3):
        for m in (model, adamw_model, muon_model):
            _grads(m, step)
            if step == 1:  # a part unused this step (the other task's layers)
                m.heads["regression"].weight.grad = None
        opt.step()
        reference.step()
        plain.step()
    kinds = _kinds(model)
    a, b, c = (dict(m.named_parameters()) for m in (model, adamw_model, muon_model))
    for name, kind in kinds.items():
        expected = c[name] if kind == "muon" else b[name]
        assert torch.allclose(a[name], expected, rtol=1e-6, atol=1e-8), name


def test_muon_weight_decay_changes_only_the_muon_groups():
    opt = Muon.for_model(_model(), lr=1e-3, weight_decay=0.01, adamw=True)
    _set_muon_weight_decay(opt, 0.1)
    assert [g["weight_decay"] for g in opt.param_groups] == [0.1, 0.1, ADAMW_WEIGHT_DECAY, 0.0]


def test_state_round_trip_keeps_the_adamw_moments():
    model = _model()
    opt = Muon.for_model(model, lr=1e-3, adamw=True)
    _grads(model, 0)
    opt.step()
    state = opt.state_dict()
    other = Muon.for_model(model, lr=5e-4, adamw=True)
    other.load_state_dict(state)
    head = model.heads["regression"].weight
    assert other.state[head]["step"] == 1
    assert torch.equal(other.state[head]["exp_avg"], opt.state[head]["exp_avg"])
    # a Muon-only state does not load into the split optimizer (different groups)
    with pytest.raises(ValueError):
        other.load_state_dict(Muon.for_model(model, lr=1e-3).state_dict())


def _smoke(**changes):
    small = dict(
        batch_size=4,
        micro_batch=4,
        min_rows=32,
        max_rows=32,
        max_features=4,
        max_cells=128,
        steps=4,
        warmup_steps=1,
        val_every=2,
        log_every=1,
        checkpoint_every=2,
        keep_every=None,
        collapse_threshold=None,
    )
    return replace(
        helpers.tabsil_smoke(filter_unpredictable=False, border_datasets=10),
        **{**small, **changes},
    )


def test_training_saves_and_resumes_the_split_optimizer(tmp_path):
    config = _smoke(optimizer="muon_adamw", phases={2: {"muon_weight_decay": 0.05}})
    full = train(config, tmp_path / "full", mlflow_log=False)
    assert torch.isfinite(torch.tensor(full["history"]["train_loss"])).all()
    state = torch.load(tmp_path / "full" / "latest.pt", weights_only=True)
    assert state["optimizer_type"] == "muon_adamw"
    groups = state["optimizer"]["param_groups"]
    assert [g.get("adamw", False) for g in groups] == [False, False, True, True]
    assert [g["weight_decay"] for g in groups] == [0.05, 0.05, ADAMW_WEIGHT_DECAY, 0.0]
    # stop at step 2 and resume: the first two steps match the single run
    train(replace(config, stop_step=2), tmp_path / "a", mlflow_log=False)
    resumed = train(
        config, tmp_path / "b", resume_from=tmp_path / "a" / "latest.pt", mlflow_log=False
    )
    assert resumed["history"]["train_loss"][:2] == full["history"]["train_loss"][:2]
    # a run resumes only from a checkpoint of its own optimizer; the weights can be taken
    longer = replace(config, steps=6)
    with pytest.raises(ValueError, match="Muon \\+ AdamW runs only"):
        check_source_checkpoint(_muon_checkpoint(tmp_path), longer, resume=True)
    with pytest.raises(ValueError, match="Muon runs only"):
        check_source_checkpoint(
            tmp_path / "a" / "latest.pt", replace(longer, optimizer="muon"), resume=True
        )
    check_source_checkpoint(_muon_checkpoint(tmp_path), longer, resume=False)
    init = train(longer, tmp_path / "init", init_from=_muon_checkpoint(tmp_path), mlflow_log=False)
    assert init["history"]["step"][-1] == 6


def _muon_checkpoint(tmp_path):
    path = tmp_path / "muon" / "latest.pt"
    if not path.exists():
        train(_smoke(stop_step=2), tmp_path / "muon", mlflow_log=False)
    return path


def test_diagnostics_name_the_adamw_groups(tmp_path, monkeypatch):
    from tabsil_train.training import loop

    seen = {}
    real = loop.parameter_groups

    def spy(model, muon_ids=None):
        groups = real(model, muon_ids)
        seen.update(groups)
        return groups

    monkeypatch.setattr(loop, "parameter_groups", spy)
    train(_smoke(optimizer="muon_adamw", steps=1, val_every=1), tmp_path, mlflow_log=False)
    assert any(n.endswith("/adam") for n in seen) and any(n.endswith("/muon") for n in seen)
    assert "output_regression/adam" in seen


def test_config_checks_the_optimizer():
    assert TrainConfig().optimizer == "muon"
    with pytest.raises(ValueError, match="optimizer"):
        TrainConfig(optimizer="adamw")


def test_train_script_sets_the_optimizer_and_a_resumed_run_keeps_it(monkeypatch):
    for name in ("TABSIL_CONFIG_NAME", "TABSIL_STEPS", "TABSIL_OPTIMIZER", "TABSIL_MUON_MOMENTUM"):
        monkeypatch.delenv(name, raising=False)
    script = helpers.load_script("scripts/train.py", "train_script_optimizer")
    assert script.build_config(script.parse_args(["--config-name", "medium"])) == (
        TrainConfig.medium()
    )
    expected = replace(TrainConfig.medium(), optimizer="muon_adamw", muon_momentum=0.95, lr=3e-4)
    argv = ["--config-name", "medium", "--optimizer", "muon_adamw", "--muon-momentum", "0.95"]
    assert script.build_config(script.parse_args([*argv, "--lr", "3e-4"])) == expected
    monkeypatch.setenv("TABSIL_OPTIMIZER", "muon_adamw")
    monkeypatch.setenv("TABSIL_MUON_MOMENTUM", "0.95")
    monkeypatch.setenv("TABSIL_LR", "3e-4")
    script = helpers.load_script("scripts/train.py", "train_script_optimizer")
    assert script.build_config(script.parse_args(["--config-name", "medium"])) == expected
    for name in ("TABSIL_OPTIMIZER", "TABSIL_MUON_MOMENTUM", "TABSIL_LR"):
        monkeypatch.delenv(name)
    script = helpers.load_script("scripts/train.py", "train_script_optimizer")
    resume = ["--config-name", "medium", "--resume-run-id", "abc"]
    args = script.parse_args(resume)
    script.inherit_recipe_options(args, {"optimizer": "muon_adamw"})
    assert script.build_config(args).optimizer == "muon_adamw"
    args = script.parse_args(resume)
    script.inherit_recipe_options(args, {})  # runs before the param were Muon runs
    assert args.optimizer == "muon"
    args = script.parse_args([*resume, "--optimizer", "muon"])
    with pytest.raises(SystemExit, match="optimizer"):
        script.inherit_recipe_options(args, {"optimizer": "muon_adamw"})
    # the momentum of the original run is inherited on resume, an explicit value wins
    assert script.source_speed_options({"muon_momentum": "0.95"}) == {"muon_momentum": 0.95}
