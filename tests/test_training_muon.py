"""Muon optimizer tests (issue #28): Muon on every parameter."""

from dataclasses import replace

import helpers
import pytest
import torch
from torch import nn

from tabsil.training.loop import check_resume_step, train
from tabsil.training.muon import MUON_MOMENTUM, Muon, _newton_schulz, muon_parameters
from tabsil.utils import lr_factor


def _splits(model) -> dict[str, int]:
    """Parameter name -> pieces Muon splits it into (`muon_parameters`)."""
    ids = {id(p): s for p, s in muon_parameters(model)}
    return {n: ids[id(p)] for n, p in model.named_parameters()}


def test_muon_takes_every_parameter():
    # learned tokens (inducing points, CLS) and scalars are Muon's too, as matrices
    model = helpers.tabsil_smoke().build_model()
    assert [p for p, _ in muon_parameters(model)] == list(model.parameters())
    splits = _splits(model)
    assert splits["column_blocks.0.summarize.qkv.weight"] == 3
    assert splits["cls_tokens"] == splits["column_blocks.0.inducing"] == 1
    assert splits["cell_encoder.frequencies"] == 1


def _as_matrix(t: torch.Tensor) -> torch.Tensor:
    # the (first dim, rest) matrix Muon orthogonalizes (0-D: (1, 1))
    return t.reshape(1, 1) if t.ndim == 0 else t.reshape(t.shape[0], -1)


def test_muon_matches_torch_muon_on_the_matrices():
    # Muon on every parameter is bit for bit torch.optim.Muon on each parameter's (first dim,
    # rest) matrix, with qkv split into its q, k and v pieces (the scheduler's lr reaches every
    # group)
    torch.manual_seed(0)
    model = helpers.tabsil_smoke().build_model()
    opt = Muon.for_model(model, lr=1e-3, weight_decay=0.01)
    pairs = muon_parameters(model)
    pieces = {  # model parameter -> pieces trained apart
        p: [nn.Parameter(w.clone()) for w in _as_matrix(p.detach()).chunk(s)] for p, s in pairs
    }
    ref = torch.optim.Muon(
        [w for p, _ in pairs for w in pieces[p]],
        lr=1e-3,
        weight_decay=0.01,
        momentum=MUON_MOMENTUM,
        adjust_lr_fn="match_rms_adamw",
    )
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_factor(s, 2, 6))
    ref_sched = torch.optim.lr_scheduler.LambdaLR(ref, lambda s: lr_factor(s, 2, 6))
    for _ in range(6):
        for p, ws in pieces.items():
            p.grad = torch.randn_like(p)
            for w, g in zip(ws, _as_matrix(p.grad).chunk(len(ws)), strict=True):
                w.grad = g.clone()
        opt.step()
        sched.step()
        ref.step()
        ref_sched.step()
        opt.zero_grad()
    for name, p in model.named_parameters():
        assert torch.equal(_as_matrix(p), torch.cat(list(pieces[p]))), name
    assert {g["lr"] for g in opt.param_groups} == {ref.param_groups[0]["lr"]}


def test_qkv_is_orthogonalized_per_block():
    # Each piece moves toward an orthogonal matrix: done jointly, each piece's singular values
    # are well below 1
    torch.manual_seed(0)
    g = torch.randn(3 * 16, 16)
    split = torch.cat([_newton_schulz(u) for u in g.split(16)]).float()
    joint = _newton_schulz(g).float()
    for i in range(3):
        block = slice(16 * i, 16 * (i + 1))
        assert torch.linalg.svdvals(split[block]).min() > 0.5
        assert torch.linalg.svdvals(joint[block]).mean() < 0.75


def test_muon_splits_multihead_attention_projections():
    # nn.MultiheadAttention's input projection (packed q/k/v) is split into three
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Linear(1, 8)
            self.block = nn.MultiheadAttention(8, 2, batch_first=True)

        def repeated_blocks(self):
            return [self.block]

    splits = _splits(Model())
    assert splits["block.in_proj_weight"] == 3
    assert splits["block.out_proj.weight"] == splits["embed.weight"] == 1
    assert splits["block.in_proj_bias"] == 1


def test_muon_state_round_trip():
    torch.manual_seed(0)
    model = helpers.tabsil_smoke().build_model()
    opt = Muon.for_model(model, lr=1e-3)
    for p in model.parameters():
        p.grad = torch.randn_like(p)
    opt.step()
    assert len(opt.state) == len(list(model.parameters()))
    state = opt.state_dict()
    # On resume, weight decay uses the current setting (the training loop sets lr)
    other = Muon.for_model(model, lr=5e-4, weight_decay=0.05)
    other.load_state_dict(state)
    assert [g["lr"] for g in other.param_groups] == [1e-3] * len(other.param_groups)
    assert [g["weight_decay"] for g in other.param_groups] == [0.05, 0.05]
    # minipfn's MuonAdam format with no Adam part (muon_scope="all") loads; with one it does not
    legacy = Muon.for_model(model, lr=5e-4)
    legacy.load_state_dict({"muon": state, "adam": None})
    assert legacy.state_dict()["state"].keys() == state["state"].keys()
    adam = torch.optim.Adam(model.parameters()).state_dict()
    with pytest.raises(ValueError, match="Adam part"):
        legacy.load_state_dict({"muon": state, "adam": adam})
    with pytest.raises(ValueError):  # a plain Adam state (one group) is not a Muon state
        legacy.load_state_dict(adam)


def _smoke(**changes):
    return replace(
        helpers.tabsil_smoke(filter_unpredictable=False, border_datasets=10),
        steps=6,
        val_every=3,
        log_every=1,
        checkpoint_every=3,
        keep_every=None,
        collapse_threshold=None,
        **changes,
    )


def test_muon_trains_and_resumes_like_one_run(tmp_path):
    full = train(_smoke(), tmp_path / "full", mlflow_log=False)
    assert all(torch.isfinite(torch.tensor(full["history"]["train_loss"])))
    state = torch.load(tmp_path / "full" / "latest.pt", weights_only=True)
    assert state["optimizer_type"] == "muon" and "param_groups" in state["optimizer"]
    # Stop at step 3 and resume. The data seed changes on resume, so the loss matches a single
    # run only up to step 3
    train(_smoke(stop_step=3), tmp_path / "a", mlflow_log=False)
    resumed = train(
        _smoke(), tmp_path / "b", resume_from=tmp_path / "a" / "latest.pt", mlflow_log=False
    )
    assert resumed["history"]["step"] == list(range(1, 7))
    assert resumed["history"]["train_loss"][:3] == full["history"]["train_loss"][:3]
    assert all(torch.isfinite(torch.tensor(resumed["history"]["train_loss"])))


def test_resume_needs_a_muon_checkpoint(tmp_path):
    train(_smoke(stop_step=3), tmp_path / "muon", mlflow_log=False)
    state = torch.load(tmp_path / "muon" / "latest.pt", weights_only=True)
    check_resume_step(state, _smoke())
    with pytest.raises(ValueError, match="adam"):
        check_resume_step({**state, "optimizer_type": "adam"}, _smoke())
    with pytest.raises(ValueError, match="schedule_free"):
        check_resume_step({**state, "schedule_free": True}, _smoke())
    # Checkpoints from before this key existed are Adam
    old = {k: v for k, v in state.items() if k != "optimizer_type"}
    with pytest.raises(ValueError, match="adam"):
        check_resume_step(old, _smoke())


def test_train_script_takes_the_muon_weight_decay(monkeypatch):
    from tabsil.hub import run_architecture

    for name in ("TABSIL_MUON_WEIGHT_DECAY", "TABSIL_CONFIG_NAME"):
        monkeypatch.delenv(name, raising=False)
    script = helpers.load_script("scripts/train.py", "train_script_muon")
    config = script.build_config(script.parse_args(["--config-name", "smoke"]))
    assert config.muon_weight_decay == 0.01
    monkeypatch.setenv("TABSIL_MUON_WEIGHT_DECAY", "0.05")
    config = script.build_config(script.parse_args(["--config-name", "smoke"]))
    assert config.muon_weight_decay == 0.05
    monkeypatch.delenv("TABSIL_MUON_WEIGHT_DECAY")
    # On resume, follow the original run's value; an explicit argument wins
    options = script.source_speed_options({"muon_weight_decay": "0.0"})
    assert options == {"muon_weight_decay": 0.0}
    source = run_architecture({k: str(v) for k, v in vars(helpers.tabsil_smoke()).items()})
    args = script.parse_args(["--config-name", "smoke", "--resume-run-id", "r"])
    assert script.build_config(args, source, options).muon_weight_decay == 0.0
    args = script.parse_args(
        ["--config-name", "smoke", "--resume-run-id", "r", "--muon-weight-decay", "0.1"]
    )
    assert script.build_config(args, source, options).muon_weight_decay == 0.1
