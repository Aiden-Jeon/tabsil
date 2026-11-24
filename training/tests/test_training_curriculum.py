"""The TabICLv2 three-stage training setup (`tabsil.recipe.curriculum`) and its parts: Muon on
all parameters, the TabICLv2 table size rule, the binary share of class counts, prior filter."""

from dataclasses import replace

import numpy as np
import pytest
import torch
import train_helpers as helpers
from torch import nn

from tabsil_train import recipe
from tabsil_train.priors import graph_scm as P
from tabsil_train.priors.generators import graph_scm as G
from tabsil_train.priors.hyperparameters import ClassCount
from tabsil_train.priors.pipeline import Rng
from tabsil_train.priors.table_size import long_table_feature_cap, sample_split_size
from tabsil_train.training.config import TrainConfig
from tabsil_train.training.loop import train
from tabsil_train.training.muon import Muon, muon_parameters

# ---------------------------------------------------------------------------
# Table size
# ---------------------------------------------------------------------------


def _split(rng, **changes):
    kw = dict(
        min_rows=1024,
        max_rows=1024,
        min_train_fraction=0.3,
        max_train_fraction=0.9,
        max_features=100,
        max_cells=1024 * 100,
    )
    return sample_split_size(rng, **{**kw, **changes})


def test_stage1_sizes():
    # fixed 1,024 rows, 30-90% train, features U(1, 100)
    rng = np.random.default_rng(0)
    sizes = [_split(rng) for _ in range(2000)]
    assert all(n_train + n_eval == 1024 for n_train, n_eval, _ in sizes)
    train_frac = np.array([n_train for n_train, _, _ in sizes]) / 1024
    # int(1024 × 0.3) = 307, so the lower bound is 0.2998
    assert 0.299 <= train_frac.min() < 0.31 and 0.89 < train_frac.max() < 0.9
    features = np.array([f for _, _, f in sizes])
    assert features.min() == 1 and features.max() == 100
    assert abs(features.mean() - 50.5) < 2


def test_long_table_sizes():
    # stage 3: 400-60,000 rows log-uniform, 79-81% train, fewer max features for more rows
    rng = np.random.default_rng(1)
    rows, features = [], []
    for _ in range(3000):
        n_train, n_eval, n_features = _split(
            rng,
            min_rows=400,
            max_rows=60_000,
            min_train_fraction=0.79,
            max_train_fraction=0.81,
            log_uniform_rows=True,
            max_cells=60_000 * 100,
        )
        n_rows = n_train + n_eval
        assert 0.78 <= n_train / n_rows <= 0.81
        assert n_features <= long_table_feature_cap(n_rows, 100)
        rows.append(n_rows)
        features.append(n_features)
    rows = np.array(rows)
    assert 400 <= rows.min() < 450 and 50_000 < rows.max() <= 60_000
    # log-uniform: median near the geometric mean
    assert abs(np.log(np.median(rows)) - np.log(np.sqrt(400 * 60_000))) < 0.15
    assert max(f for r, f in zip(rows, features, strict=True) if r > 50_000) <= 20
    assert long_table_feature_cap(10_240, 100) == 100
    assert long_table_feature_cap(10_241, 100) == 80
    assert long_table_feature_cap(60_000, 10) == 10
    assert long_table_feature_cap(100_000, 100) == 10


def test_split_size_respects_cells_and_minimum():
    rng = np.random.default_rng(2)
    for _ in range(200):
        n_train, n_eval, n_features = _split(rng, max_cells=1024 * 7, min_train_rows=400)
        assert n_features <= 7 and n_train >= 400 and n_eval >= 1


# ---------------------------------------------------------------------------
# Number of classes
# ---------------------------------------------------------------------------


def test_class_count_binary_share():
    # binary_prob 0 keeps the old random stream; 0.5 gives binary 0.5 + 0.5/9
    old, new = Rng.from_seed(0), Rng.from_seed(0)
    a = [ClassCount(10)(old, 100, 5)["num_classes"] for _ in range(200)]
    b = [ClassCount(10, binary_prob=0.0)(new, 100, 5)["num_classes"] for _ in range(200)]
    assert a == b
    rng = Rng.from_seed(1)
    counts = np.array([ClassCount(10, 0.5)(rng, 100, 5)["num_classes"] for _ in range(20_000)])
    assert abs((counts == 2).mean() - (0.5 + 0.5 / 9)) < 0.015
    assert counts.max() == 10
    with pytest.raises(ValueError, match="binary_prob"):
        ClassCount(10, 1.5)


# ---------------------------------------------------------------------------
# prior
# ---------------------------------------------------------------------------


def _prior(task="classification", **changes):
    kw = dict(min_rows=60, max_rows=60, max_features=6, max_cells=600)
    return P.GraphSCMPrior(task, seed=0, **{**kw, **changes})


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_train_fraction_prior_batches(task):
    prior = _prior(task, min_rows=40, max_rows=80, log_uniform_rows=True)
    seen = set()
    for _ in range(6):
        batch = prior.sample_batch(3)  # x (dataset, row, feature)
        n_rows, n_features = batch.x.shape[1], batch.x.shape[2]
        assert 40 <= n_rows < 80 and 1 <= n_features <= 6
        assert 0 < batch.n_train < n_rows
        seen.add(n_rows - batch.n_train)
    assert len(seen) > 1  # the validation row count varies per batch
    # classification keeps at least two train rows (two classes can appear)
    assert prior.min_train_rows == (2 if task == "classification" else 1)


def test_size_checks():
    with pytest.raises(ValueError, match="min_train_fraction"):
        _prior(min_train_fraction=0.9, max_train_fraction=0.3)
    with pytest.raises(ValueError, match="max_cells"):
        _prior(max_cells=10)
    # the training config checks the sizes for both tasks (classification needs 2 train rows)
    with pytest.raises(ValueError, match="max_cells"):
        TrainConfig(min_rows=10_000_000, max_rows=10_000_000)
    with pytest.raises(ValueError, match="min_rows"):
        TrainConfig(min_rows=2, max_rows=2)
    # the cell cap also caps the feature count of a table
    narrow = _prior(max_cells=180)
    assert {narrow.sample_shape()[1] for _ in range(50)} <= {1, 2, 3}


def test_prior_filter_retries_unpredictable_datasets(monkeypatch):
    # graph/dataset filters are passed to TabICL PriorConfig; even if the dataset filter keeps
    # rejecting, each dataset is regenerated at most FILTER_ATTEMPTS times (cap lowered here)
    prior = _prior(filter_unpredictable=True)
    generator = prior.pipeline.generator
    assert generator._config.filter_unpredictable_graphs
    assert generator._config.filter_unpredictable_datasets
    calls = []

    def always(x, y, config, *, is_classif):
        calls.append(is_classif)
        return True

    monkeypatch.setattr(generator, "_should_filter", always)
    monkeypatch.setattr(G, "FILTER_ATTEMPTS", 6)
    batch = prior.sample_batch(2)
    assert batch.x.shape[0] == 2
    assert len(calls) >= 2 * 6 and all(calls)
    assert len(calls) % 6 == 0  # exactly the cap per dataset (accepted after that)
    off = _prior()
    assert not off.pipeline.generator._config.filter_unpredictable_datasets


def test_prior_filter_keeps_predictable_datasets(monkeypatch):
    prior = _prior("regression", filter_unpredictable=True)
    generator = prior.pipeline.generator
    calls = []
    monkeypatch.setattr(
        generator, "_should_filter", lambda *a, **k: calls.append(1) or False
    )  # always accept
    prior.sample_batch(3)
    assert len(calls) == 3


def test_real_filter_runs():
    # the real ExtraTrees filter also yields datasets (regression borders use the same filter)
    prior = _prior("classification", filter_unpredictable=True)
    batch = prior.sample_batch(2)
    assert torch.isfinite(batch.x).all() or torch.isnan(batch.x).any()
    borders = P.regression_borders(
        20,
        8,
        filter_unpredictable=True,
        min_rows=60,
        max_rows=60,
        max_features=6,
        max_cells=600,
    )
    assert borders.shape == (21,) and (borders.diff() > 0).all()


# ---------------------------------------------------------------------------
# Muon on all parameters
# ---------------------------------------------------------------------------


class _Toy(nn.Module):
    # 2-D matrices, 1-D bias/gain, 3-D tokens (no 0-D, to compare with TabICL Muon)
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(6, 5)
        self.gain = nn.Parameter(torch.ones(5))
        self.tokens = nn.Parameter(torch.randn(3, 2, 4))


def test_muon_takes_every_parameter():
    model = helpers.tabsil_smoke().build_model()
    muon = muon_parameters(model)
    assert [p for p, _ in muon] == list(model.parameters())
    # the qkv weights of the repeated blocks are split in three, the rest is one piece
    qkv = {id(m.qkv.weight) for m in model.modules() if hasattr(m, "qkv")}
    assert qkv and all((s == 3) == (id(p) in qkv) for p, s in muon)
    assert any(p.ndim == 1 for p, _ in muon)
    opt = Muon.for_model(model, lr=1e-2, weight_decay=0.01, momentum=0.9)
    assert all(g["momentum"] == 0.9 and g["weight_decay"] == 0.01 for g in opt.param_groups)
    assert sorted(g["split"] for g in opt.param_groups) == [1, 3]
    assert sum(len(g["params"]) for g in opt.param_groups) == len(muon)


def test_all_scope_matches_tabicl_muon():
    # Same update as Muon in the TabICLv2 training code (all parameters, momentum 0.9 = --beta1,
    # Nesterov, match_rms 0.2, weight decay 0.01). Sum vs mean momentum gives the same result since
    # orthogonalization normalizes the scale (only bf16 rounding differs)
    from tabicl.train._muon import Muon as TabICLMuon

    torch.manual_seed(0)
    ours, theirs = _Toy(), _Toy()
    theirs.load_state_dict(ours.state_dict())
    start = {n: p.detach().clone() for n, p in ours.named_parameters()}
    opt = Muon.for_model(ours, lr=1e-2, weight_decay=0.01, momentum=0.9)
    ref = TabICLMuon(
        param_groups=[dict(params=list(theirs.parameters()), use_muon=True)],
        lr=1e-2,
        weight_decay=0.01,
        matched_adamw_rms=0.2,
        momentum=0.9,
        nesterov=True,
        ns_steps=5,
    )
    for step in range(4):
        gen = torch.Generator().manual_seed(step)
        for a, b in zip(ours.parameters(), theirs.parameters(), strict=True):
            a.grad = torch.randn(a.shape, generator=gen)
            b.grad = a.grad.clone()
        opt.step()
        ref.step()
    for (name, a), b in zip(ours.named_parameters(), theirs.parameters(), strict=True):
        # differs only by the bf16 orthogonalization error (a few %) relative to 4 steps of movement
        moved = (a - start[name]).abs().max()
        assert (a - b).abs().max() < 0.05 * moved, name


def test_scalar_parameter_and_state_round_trip():
    class Scalar(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(3, 3)
            self.scale = nn.Parameter(torch.tensor(1.0))

    model = Scalar()
    opt = Muon.for_model(model, lr=0.1)
    for p in model.parameters():
        p.grad = torch.ones_like(p)
    before = model.scale.item()
    opt.step()
    # orthogonalizing (1, 1) leaves only the sign; lr scale 0.2·sqrt(1)
    assert model.scale.item() == pytest.approx(before - 0.1 * 0.2, rel=1e-2)
    state = opt.state_dict()
    again = Muon.for_model(model, lr=0.1, weight_decay=0.05)
    again.load_state_dict(state)
    assert again.state_dict()["state"].keys() == state["state"].keys()
    # the group settings other than lr stay as the new optimizer has them
    assert all(g["weight_decay"] == 0.05 for g in again.param_groups)
    # minipfn's MuonAdam state with Muon on every parameter (no Adam part) loads as well
    legacy = Muon.for_model(model, lr=0.1)
    legacy.load_state_dict({"muon": state, "adam": None})
    buffers = [s["momentum_buffer"] for s in legacy.state.values()]
    expected = [s["momentum_buffer"] for s in opt.state.values()]
    assert all(torch.equal(a, b) for a, b in zip(buffers, expected, strict=True))
    # a state with an Adam part (minipfn muon_scope="blocks") cannot be continued
    with pytest.raises(ValueError, match="Adam part"):
        legacy.load_state_dict({"muon": state, "adam": {"state": {}, "param_groups": []}})
    with pytest.raises(ValueError, match="Muon optimizer state"):
        legacy.load_state_dict({"state": {}})


def test_config_checks():
    assert TrainConfig().muon_momentum == 0.9 and TrainConfig().muon_weight_decay == 0.01
    with pytest.raises(ValueError, match="muon_momentum"):
        TrainConfig(muon_momentum=1.0)
    with pytest.raises(ValueError, match="muon_weight_decay"):
        TrainConfig(muon_weight_decay=-0.1)
    with pytest.raises(ValueError, match="muon_distributed"):
        TrainConfig(muon_distributed="rack")
    with pytest.raises(ValueError, match="binary_prob"):
        TrainConfig(binary_prob=-0.1)


# ---------------------------------------------------------------------------
# Recipe and training
# ---------------------------------------------------------------------------


def test_curriculum_recipe():
    s1, s2, s3 = (recipe.curriculum(n) for n in (1, 2, 3))
    for c in (s1, s2, s3):
        assert c.tasks == ("classification", "regression")
        assert c.muon_momentum == 0.9 and c.muon_weight_decay == 0.01 and c.batch_size == 256
        assert c.max_features == 100 and c.prior_type == "graph_scm"
        assert c.binary_prob == 0.5 and c.filter_unpredictable and c.phases == ()
        assert c.warmup_steps == c.steps // 100
        # cell cap = max rows x 100 (the feature cap of long tables is the TabICLv2 table)
        assert c.max_cells == c.max_rows * 100
    # the TrainConfig defaults are stage 1
    assert s1 == TrainConfig()
    assert (s1.steps, s1.lr, s1.grad_clip, s1.micro_batch) == (250_000, 8e-4, 10.0, 4)
    assert (s1.min_rows, s1.max_rows, s1.min_train_fraction, s1.max_train_fraction) == (
        1024,
        1024,
        0.3,
        0.9,
    )
    assert (s2.steps, s2.lr, s2.grad_clip, s2.max_rows, s2.log_uniform_rows) == (
        20_000,
        1e-4,
        10.0,
        10_240,
        True,
    )
    assert (s3.steps, s3.lr, s3.grad_clip, s3.max_rows) == (5_000, 2e-5, 1.0, 60_000)
    # datasets per task match TabICLv2 (500K steps x batch 64)
    assert s1.steps * s1.batch_size // 2 == 500_000 * 64
    assert s2.steps * s2.batch_size // 2 == 40_000 * 64
    assert s3.steps * s3.batch_size // 2 == 10_000 * 64
    with pytest.raises(ValueError, match="stage"):
        recipe.curriculum(4)
    with pytest.raises(ValueError, match="step"):
        recipe.curriculum(1, steps=0)


def _smoke_curriculum(base=None, **changes):
    # Every stage-1 code path at a small size: Muon on all parameters, train_fraction, binary share,
    # prior filter. `base` defaults to the TabSil smoke config
    config = helpers.tabsil_smoke() if base is None else base
    stage = {k: v for k, v in recipe.TRAINING.items() if k != "batch_size"}
    small = dict(
        max_features=8,
        min_rows=48,
        max_rows=80,
        log_uniform_rows=True,
        max_cells=80 * 8,
        steps=4,
        warmup_steps=1,
        val_every=2,
        log_every=1,
        checkpoint_every=2,
        keep_every=None,
        collapse_threshold=None,
        border_datasets=20,
    )
    return replace(config, **{**stage, **small, **changes})  # arguments override recipe values


def test_curriculum_trains_and_resumes(tmp_path, monkeypatch):
    # regression borders are built in a process pool, so their string hash seed must be fixed for
    # runs to share borders (TabICL prior; see the `generators.graph_scm` comment)
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    full = train(_smoke_curriculum(), tmp_path / "full", mlflow_log=False)
    assert all(np.isfinite(full["history"]["train_loss"]))
    state = torch.load(tmp_path / "full" / "latest.pt", weights_only=True)
    # one Muon state with a momentum buffer per parameter
    assert len(state["optimizer"]["state"]) == len(state["model"])
    assert state["optimizer_type"] == "muon"  # all parameters use Muon
    # resuming changes the data seed, so losses match only up to the stop step
    train(_smoke_curriculum(stop_step=2), tmp_path / "a", mlflow_log=False)
    resumed = train(
        _smoke_curriculum(),
        tmp_path / "b",
        resume_from=tmp_path / "a" / "latest.pt",
        mlflow_log=False,
    )
    assert resumed["history"]["step"] == list(range(1, 5))
    assert resumed["history"]["train_loss"][:2] == full["history"]["train_loss"][:2]
    assert all(np.isfinite(resumed["history"]["train_loss"]))


def test_train_script_stages(monkeypatch):
    for name in ("TABSIL_CONFIG_NAME", "TABSIL_INIT_RUN_ID", "TABSIL_STEPS"):
        monkeypatch.delenv(name, raising=False)
    script = helpers.load_script("scripts/train.py", "train_script_curriculum")
    config = script.build_config(script.parse_args(["--config-name", "stage1"]))
    assert config == recipe.curriculum(1)
    for stage in ("stage2", "stage3"):  # stage 2 and 3 need the previous stage's weights
        with pytest.raises(SystemExit):
            script.parse_args(["--config-name", stage])
    # Only the given step count (overriding the yaml default 300) and its warmup change
    args = ["--config-name", "stage3", "--steps", "500", "--init-run-id", "stage2"]
    config = script.build_config(script.parse_args(args))
    assert config == recipe.curriculum(3, steps=500)
    # Resuming continues the Muon momentum and weight decay of the source run. Options of minipfn
    # runs that TabSil no longer has are not carried over
    params = {"muon_momentum": "0.9", "muon_weight_decay": "0.01", "muon_scope": "all"}
    params |= {"optimizer": "muon", "logit_softcap": "None", "schedule_free": "False"}
    options = script.source_speed_options(params)
    assert options == {"muon_momentum": 0.9, "muon_weight_decay": 0.01}
