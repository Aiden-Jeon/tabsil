"""Distributed Muon tests. Checks that updates computed across processes (gloo, CPU) are
bit-identical to a single-process computation."""

import os
import socket
from dataclasses import replace
from pathlib import Path

import helpers
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tabsil import recipe
from tabsil.training.config import TrainConfig
from tabsil.training.loop import train
from tabsil.training.muon import Muon, _matrix, _piece_cost, muon_parameters, shard_pieces

STEPS = 3


def _model():
    # TabSil: weights split into q/k/v (3 pieces), tall matrices (MLP first layer) and wide ones
    config = helpers.tabsil_smoke()
    torch.manual_seed(0)
    return config.build_model()


def _run(process_group=None) -> dict:
    """Weights and optimizer state after STEPS updates with the same gradients. At step 1 the
    first and last parameters have no gradient (layers of a task unused that step). 1-D
    parameters (flattened to an (n, 1) matrix) are sharded too."""
    model = _model()
    opt = Muon.for_model(model, lr=1e-2, weight_decay=0.01, process_group=process_group)
    params = list(model.parameters())
    for step in range(STEPS):
        gen = torch.Generator().manual_seed(step)
        for p in params:
            p.grad = torch.randn(p.shape, generator=gen)
        if step == 1:
            params[0].grad = None
            params[-1].grad = None
        opt.step()
    return {"model": model.state_dict(), "optimizer": opt.state_dict()}


def _worker(rank: int, world: int, port: int, group_size: int, threads: int, out: str) -> None:
    # CPU matmul summation order depends on thread count; match the single-process count
    # (if an earlier test trains without prior workers, `PriorBatches` sets this process to 1
    # thread)
    torch.set_num_threads(threads)
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        group = dist.group.WORLD
        if group_size < world:
            group, _ = dist.new_subgroups(group_size=group_size)
        torch.save(_run(group), Path(out) / f"rank{rank}.pt")
    finally:
        dist.destroy_process_group()


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _assert_same(a, b, path="") -> None:
    if isinstance(a, dict):
        assert a.keys() == b.keys(), path
        for k in a:
            _assert_same(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, list | tuple):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            _assert_same(x, y, f"{path}[{i}]")
    elif isinstance(a, torch.Tensor):
        assert a.dtype == b.dtype and torch.equal(a, b), path
    else:
        assert a == b, path


@pytest.mark.parametrize(("world", "group_size"), [(3, 3), (4, 2)])
def test_sharded_muon_matches_one_process(tmp_path, world, group_size):
    # Sharding across all GPUs (3, pieces not evenly divisible) or within nodes (4 as 2x2), every
    # process's weights and momentum state are bit-identical to the single-process run
    reference = _run()
    args = (world, _free_port(), group_size, torch.get_num_threads(), str(tmp_path))
    mp.spawn(_worker, args=args, nprocs=world)
    for rank in range(world):
        _assert_same(torch.load(tmp_path / f"rank{rank}.pt", weights_only=True), reference)


def test_shard_pieces_balances_the_work():
    # The pieces of the full-size TabSil model (q/k/v split) spread evenly over 8 and 32 GPUs
    with torch.device("meta"):
        model = recipe.curriculum(1).build_model()
    shapes = [
        (_matrix(p).shape[0] // s, _matrix(p).shape[1])
        for p, s in muon_parameters(model)
        for _ in range(s)
    ]
    assert len(shapes) == sum(s for _, s in muon_parameters(model))
    largest = max(_piece_cost(*shape) for shape in shapes)
    for n, bound in ((8, 1.05), (32, 1.1)):
        owner = shard_pieces(shapes, n)
        assert owner == shard_pieces(shapes, n)  # every GPU agrees
        loads = [0] * n
        for shape, rank in zip(shapes, owner, strict=True):
            loads[rank] += _piece_cost(*shape)
        mean = sum(loads) / n
        # LPT: no rank carries more than the mean plus the largest piece
        assert min(loads) > 0 and max(loads) <= mean + largest and max(loads) / mean < bound
    assert shard_pieces([(4, 4), (8, 8), (2, 2)], 2) == [1, 0, 1]  # largest to lightest rank
    assert _piece_cost(4, 2) == _piece_cost(2, 4) == 2 * 2 * (2 * 4 + 2)


def test_one_gpu_ignores_the_option(tmp_path):
    # With one GPU nothing is sharded (no process group is created either)
    smoke = replace(
        helpers.tabsil_smoke(),
        steps=4,
        val_every=2,
        log_every=1,
        collapse_threshold=None,
        filter_unpredictable=False,
        border_datasets=10,
    )
    # Training without prior workers switches this process to one thread (`PriorBatches`), so the
    # first run would see another CPU summation order than the second; run both with one thread
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        off = train(smoke, tmp_path / "off", mlflow_log=False)
        on = train(replace(smoke, muon_distributed="all"), tmp_path / "on", mlflow_log=False)
    finally:
        torch.set_num_threads(threads)
    assert on["history"] == off["history"]


def test_config_checks():
    assert TrainConfig().muon_distributed == "off"
    with pytest.raises(ValueError, match="muon_distributed"):
        TrainConfig(muon_distributed="gpu")
    TrainConfig(muon_distributed="node")


def test_train_script_takes_the_option(monkeypatch):
    # Enabled by flag/env var; resume follows the original run (off if absent); off disables it
    from tabsil.hub import run_architecture

    for name in ("TABSIL_MUON_DISTRIBUTED", "TABSIL_CONFIG_NAME"):
        monkeypatch.delenv(name, raising=False)
    script = helpers.load_script("scripts/train.py", "train_script_muon_distributed")
    monkeypatch.setenv("TABSIL_MUON_DISTRIBUTED", "node")
    config = script.build_config(script.parse_args(["--config-name", "smoke"]))
    assert config.muon_distributed == "node"
    monkeypatch.delenv("TABSIL_MUON_DISTRIBUTED")
    source = run_architecture({k: str(v) for k, v in vars(helpers.tabsil_smoke()).items()})

    def build(*flags, params):
        args = script.parse_args(["--config-name", "smoke", "--resume-run-id", "r", *flags])
        return script.build_config(args, source, script.source_speed_options(params))

    chunk01 = {"muon_weight_decay": "0.0"}  # run from before this option existed
    assert build(params=chunk01).muon_distributed == "off"
    assert build("--muon-distributed", "node", params=chunk01).muon_distributed == "node"
    chunk02 = {**chunk01, "muon_distributed": "node"}
    assert build(params=chunk02).muon_distributed == "node"
    assert build("--muon-distributed", "off", params=chunk02).muon_distributed == "off"
