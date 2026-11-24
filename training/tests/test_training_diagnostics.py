"""Tests for training diagnostics (`training.diagnostics`)."""

import json
from dataclasses import replace

import mlflow
import pytest
import torch
import train_helpers as helpers
from torch import nn

from tabsil_train.training.config import TrainConfig
from tabsil_train.training.diagnostics import (
    TIMING_FIELDS,
    GroupNormWindow,
    RankTiming,
    TaskGradientCosine,
    group_norms,
    parameter_groups,
    rank_timing_metrics,
    shared_groups,
    write_rank_timing,
)
from tabsil_train.training.loop import train


def _multitask(**changes):
    # The TabSil smoke structure: its LogScale temperature parts get their own parameter groups
    config = helpers.tabsil_smoke(filter_unpredictable=False, border_datasets=10)
    return replace(
        config,
        steps=8,
        warmup_steps=1,
        log_every=4,
        val_every=8,
        checkpoint_every=8,
        keep_every=None,
        collapse_threshold=None,
        **changes,
    )


def test_groups_split_by_stage_task_and_optimizer():
    model = _multitask().build_model()
    muon_ids = {id(p) for p in model.parameters()}  # every parameter is Muon's
    groups = parameter_groups(model, muon_ids)
    # Every parameter appears in exactly one group
    ids = [id(p) for params in groups.values() for p in params]
    assert sorted(ids) == sorted(id(p) for p in model.parameters())
    assert {"column_stage/muon", "row_stage/muon", "row_icl/muon"} <= set(groups)
    assert not any(name.endswith("/adam") for name in groups)
    # Single-task parts (target encoder, output layer) are separate
    assert {"cell_encoder_classification/muon", "output_regression/muon"} <= set(groups)
    # Attention logit scale parts (LogScale a_h, gate) are separate within a stage
    assert {"row_icl_log_scale/muon", "column_stage_log_scale/muon"} <= set(groups)
    names = {id(p): n for n, p in model.named_parameters()}
    assert all("log_scale" in names[id(p)] for p in groups["row_icl_log_scale/muon"])
    assert not any("log_scale" in names[id(p)] for p in groups["row_icl/muon"])
    shared = shared_groups(groups, ("classification", "regression"))
    assert "row_icl/muon" in shared and "output_regression/muon" not in shared
    assert "row_icl_log_scale/muon" in shared
    # without optimizer ids no suffix is added
    assert "row_icl" in parameter_groups(model)


def test_group_norms_match_a_flat_norm_and_skip_unused():
    model = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 2))
    groups = {"a": list(model[0].parameters()), "b": list(model[1].parameters())}
    flat = torch.cat([p.flatten() for p in model[0].parameters()])
    assert group_norms(groups)["a"].item() == pytest.approx(flat.norm().item(), rel=1e-6)
    model(torch.randn(5, 3))[:, 0].sum().backward()
    model[1].weight.grad = None
    model[1].bias.grad = None
    assert set(group_norms(groups, grads=True)) == {"a"}


def test_grad_norm_window_keeps_mean_and_max_per_task():
    window = GroupNormWindow({})
    for task, value in [("classification", 1.0), ("regression", 5.0), ("classification", 3.0)]:
        window.update(task, {"g": torch.tensor(value)}, torch.tensor(value))
    m = window.metrics()
    assert m["grad_norm/classification/g"] == pytest.approx(2.0)
    assert m["grad_norm_max/classification/g"] == pytest.approx(3.0)
    assert m["grad_norm/regression/g"] == pytest.approx(5.0)
    assert m["grad_norm_median"] == pytest.approx(3.0) and m["grad_norm_max"] == pytest.approx(5.0)


def _set_grads(params, grads):
    for p, g in zip(params, grads, strict=True):
        p.grad = None if g is None else g.clone()


def test_task_gradient_cosine_over_four_steps():
    a, b = nn.Parameter(torch.zeros(3)), nn.Parameter(torch.zeros(2))
    head = nn.Parameter(torch.zeros(2))
    groups = {"shared": [a, b], "output_regression": [head]}
    cos = TaskGradientCosine(groups, ("classification", "regression"), log_every=10)
    assert list(cos.groups) == ["shared"]
    # Logged step (step 9, done 10) and the 3 steps before it
    assert [s for s in range(25) if cos.wants(s)] == [6, 7, 8, 9, 16, 17, 18, 19]
    g_cls = [torch.tensor([1.0, 0.0, 0.0]), torch.tensor([1.0, 0.0])]
    g_reg = [torch.tensor([-1.0, 0.0, 0.0]), torch.tensor([0.0, 1.0])]  # group 1 opposite, cos -1/2
    out = None
    for step in range(6, 10):
        task = ("classification", "regression")[step % 2]
        _set_grads([a, b], g_cls if task == "classification" else g_reg)
        head.grad = torch.ones(2) if task == "regression" else None
        out = cos.update(step, task, finite=True)
        assert (out is None) == (step < 9)
    assert out["grad_cos/shared"] == pytest.approx(-0.5, abs=1e-3)
    assert out["grad_cos_same/classification/shared"] == pytest.approx(1.0, abs=1e-3)
    assert out["grad_cos_same/regression/shared"] == pytest.approx(1.0, abs=1e-3)
    assert not cos.tasks  # cleared for the next window


def test_task_gradient_cosine_drops_an_incomplete_window():
    a = nn.Parameter(torch.zeros(2))
    cos = TaskGradientCosine({"shared": [a]}, ("classification", "regression"), log_every=10)
    a.grad = torch.ones(2)
    # A window with a non-finite step is not emitted
    assert cos.update(6, "classification", finite=True) is None
    assert cos.update(7, "regression", finite=False) is None
    assert cos.update(8, "classification", finite=True) is None
    assert cos.update(9, "regression", finite=True) is None
    # Starting mid-window (resume) does not collect until the next window
    assert cos.update(18, "classification", finite=True) is None
    assert cos.update(19, "regression", finite=True) is None
    assert not cos.tasks
    for step in range(26, 30):
        out = cos.update(step, ("classification", "regression")[step % 2], finite=True)
    assert out is not None


def _gathered(fwd, cells):
    """Build a `RankTiming.gather`-shaped record from (rank, step) times and cell counts."""
    fwd, cells = torch.tensor(fwd, dtype=torch.float64), torch.tensor(cells, dtype=torch.float64)
    ranks, steps = fwd.shape
    out = torch.zeros(ranks, steps, 2 + len(TIMING_FIELDS), dtype=torch.float64)
    out[..., 0] = torch.arange(steps, dtype=torch.float64)
    out[..., 2] = fwd
    out[..., 3] = 100.0 - fwd  # GPUs that finish early wait longer
    out[..., 4] = 10.0
    out[..., 5] = 5.0
    out[..., 6] = 3.0
    out[..., 7] = cells
    return out


def test_rank_timing_metrics_find_size_driven_waiting():
    cells = [[100.0, 300.0], [200.0, 100.0], [400.0, 200.0]]
    fwd = [[c / 10 for c in row] for row in cells]  # forward time proportional to cell count
    m = rank_timing_metrics(_gathered(fwd, cells))
    assert m["rank_fwd_cells_corr"] == pytest.approx(1.0)
    assert m["rank_bwd_fwd_corr"] == pytest.approx(-1.0)
    assert m["rank_slowest_is_largest"] == pytest.approx(1.0)
    assert m["rank_fwd_ms_slowest"] == pytest.approx(35.0)  # mean of 40 and 30 per step
    assert m["rank_fwd_imbalance"] == pytest.approx(35.0 / (1300 / 6 / 10))
    # With one GPU there are no cross-GPU metrics
    single = rank_timing_metrics(_gathered(fwd[:1], cells[:1]))
    assert "rank_fwd_cells_corr" not in single and "rank_fwd_ms" in single


def test_rank_timing_records_and_writes(tmp_path):
    timing = RankTiming(torch.device("cpu"))
    for step in range(3):
        for _ in range(2):  # gradient accumulation 2
            start = timing.mark()
            fwd = timing.mark()
            timing.add_micro(start, fwd, timing.mark(), torch.zeros(4, 10 + step, 3), 6)
        timing.step_done(step, step % 2)
    gathered = timing.gather(world_size=1)
    assert gathered.shape == (1, 3, 2 + len(TIMING_FIELDS))
    assert gathered[0, :, 7].tolist() == [2 * 4 * 10 * 3, 2 * 4 * 11 * 3, 2 * 4 * 12 * 3]
    assert not timing.records
    path = tmp_path / "t.jsonl"
    write_rank_timing(path, gathered, ("classification", "regression"))
    lines = [json.loads(s) for s in path.read_text().splitlines()]
    assert [line["task"] for line in lines] == ["classification", "regression", "classification"]
    assert lines[1]["rows"] == [11.0] and lines[1]["n_train"] == [6.0]


def test_grad_cosine_needs_four_steps_per_log():
    with pytest.raises(ValueError, match="log_every"):
        TrainConfig(grad_cosine=True, log_every=2)


def test_training_logs_diagnostics_without_changing_the_run(tmp_path, monkeypatch):
    # Process-pool workers that build regression borders use the same hash seed (as in the
    # training yaml; otherwise borders differ per run and the loss changes regardless of
    # diagnostics)
    monkeypatch.setenv("PYTHONHASHSEED", "1")
    plain = train(_multitask(), tmp_path / "plain", mlflow_log=False)
    prev_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    try:
        mlflow.set_experiment("test-diagnostics")
        with mlflow.start_run() as run:
            diag = train(_multitask(grad_cosine=True, rank_timing=True), tmp_path / "diag")
        metrics = mlflow.MlflowClient().get_run(run.info.run_id).data.metrics
    finally:
        mlflow.set_tracking_uri(prev_uri)
    # Diagnostics are read-only
    assert diag["history"]["train_loss"] == plain["history"]["train_loss"]
    for key in (
        "weight_norm/row_icl/muon",
        "grad_norm/classification/row_icl/muon",
        "grad_norm_max/regression/output_regression/muon",
        "grad_norm_median",
        "grad_cos/row_icl/muon",
        "grad_cos_same/classification/row_icl/muon",
        "rank_fwd_ms",
        "rank_cells",
    ):
        assert key in metrics, key
    assert -1.0 <= metrics["grad_cos/row_icl/muon"] <= 1.0
    lines = (tmp_path / "diag" / "rank_timing_from_0000000.jsonl").read_text().splitlines()
    assert [json.loads(s)["step"] for s in lines] == list(range(8))


def test_script_flags_turn_on_diagnostics(monkeypatch):
    for name in ("TABSIL_RANK_TIMING", "TABSIL_GRAD_COSINE", "TABSIL_CONFIG_NAME"):
        monkeypatch.delenv(name, raising=False)
    script = helpers.load_script("scripts/train.py", "train_script_diagnostics")
    base = ["--config-name", "smoke"]
    config = script.build_config(script.parse_args(base))
    assert not config.grad_cosine and not config.rank_timing
    config = script.build_config(script.parse_args([*base, "--grad-cosine", "--rank-timing"]))
    assert config.grad_cosine and config.rank_timing
    monkeypatch.setenv("TABSIL_RANK_TIMING", "1")
    config = script.build_config(script.parse_args(base))
    assert config.rank_timing and not config.grad_cosine
