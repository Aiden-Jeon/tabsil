"""Tests of the `column_row_transformer` model (TabSil's backbone) and of the TabSil recipe
around it."""

import json
import types
from dataclasses import replace

import helpers
import numpy as np
import pytest
import torch
from sklearn.datasets import load_breast_cancer, load_diabetes

from tabsil import hub, recipe
from tabsil import models as registry
from tabsil.inference.ensemble import EnsembleClassifier, EnsembleRegressor
from tabsil.models.column_row_transformer import (
    MAX_CLASSES,
    ColumnRowTransformer,
    ColumnRowTransformerConfig,
    RetrievalDecoder,
    ecdf_midrank,
    group_offsets,
)
from tabsil.models.quantile_distribution import N_QUANTILES
from tabsil.training.config import N_BUCKETS, TrainConfig
from tabsil.training.loop import check_checkpoint, train

TASKS = ("classification", "regression")
BORDERS = torch.linspace(-3, 3, N_BUCKETS + 1)


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


def _out(task):
    return 10 if task == "classification" else N_QUANTILES


def test_paper_size_matches_the_report():
    # TabSil (interleaved Kumo stages, quantile head): 221.5M
    config = recipe.curriculum(1).model_config()
    with torch.device("meta"):
        model = ColumnRowTransformer(config)
    assert abs(model.num_parameters() / 1e6 - 221.5) / 221.5 < 0.02
    assert config.icl_dim == 1024


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


def test_group_offsets_and_ecdf():
    assert group_offsets(3) == (0, 1, 3)
    x = torch.tensor([[[1.0], [2.0], [2.0], [float("nan")], [2.0], [9.0]]])
    u = ecdf_midrank(x, n_train=4)[0, :, 0]
    # Train values 1, 2, 2 (3 after dropping one missing): 1 → 0.5/3, 2 → (1 + 1)/3, missing → 0.5,
    # 9 → 1
    assert torch.allclose(u, torch.tensor([0.5 / 3, 2 / 3, 2 / 3, 0.5, 2 / 3, 1.0]))


@pytest.mark.parametrize("task", TASKS)
def test_test_rows_do_not_see_each_other(task):
    model = _model()
    x, y = _data(task)
    out = model(x, y, task)
    assert out.shape == (2, 15, _out(task)) and torch.isfinite(out).all()
    changed = x.clone()
    changed[:, 30] += 5.0  # test row 5
    other = model(changed, y, task)
    keep = [i for i in range(15) if i != 5]
    assert torch.equal(other[:, keep], out[:, keep])
    assert not torch.equal(other[:, 5], out[:, 5])
    # Train rows change the test-row outputs
    changed = x.clone()
    changed[:, 3] += 5.0
    assert not torch.allclose(model(changed, y, task), out)


def test_retrieval_decoder_gives_probabilities_over_seen_classes():
    model = _model()
    x, y = _data("classification")
    # logit is log(max(p, 1e-5) + 3e-5), so the sum slightly exceeds 1 (max 1.0004 for 10 classes)
    p = model(x, y, "classification").exp()
    assert torch.allclose(p.sum(-1), torch.ones(2, 15), atol=1e-3)
    # Classes absent from train rows keep only the clip value (no per-class-count parameters)
    assert (p[..., 3:] < 1e-4).all()


def test_every_parameter_gets_a_gradient():
    # DDP hangs on parameters unused in forward (without find_unused_parameters). The two
    # tasks together use every parameter
    model = _model().train()
    loss = 0.0
    for task in TASKS:
        x, y = _data(task)
        # Classification logits are log p, so logsumexp is 0. Use one class's log probability
        loss = loss + model(x, y, task)[..., 0].sum()
    loss.backward()
    assert [n for n, p in model.named_parameters() if p.grad is None] == []


def test_test_row_multi_query_matches_expanded_heads():
    # Test rows fold heads into the query length. Equivalent to repeating key/value per head
    import torch.nn.functional as F

    attention = _model().icl_layers[0].attention
    x = torch.randn(2, 30, attention.qkv.in_features)
    n = 20
    out = attention(x, n_context=n)
    d = x.shape[-1]
    w = attention.qkv.weight
    q = attention.q_norm(attention._heads(F.linear(x, w[:d])))
    k = attention.k_norm(attention._heads(F.linear(x[:, :n], w[d : 2 * d])))
    v = attention._heads(F.linear(x[:, :n], w[2 * d :]))
    q = attention.log_scale(q, n)
    test = F.scaled_dot_product_attention(q[:, :, n:], k[:, :1].expand_as(k), v[:, :1].expand_as(v))
    expected = attention.out(test.transpose(1, 2).reshape(2, 10, d))
    assert torch.allclose(out[:, n:], expected, atol=1e-5)


def test_retrieval_decoder_projects_only_what_it_uses():
    model = _model()
    decoder = model.heads["classification"]
    h = torch.randn(2, 30, model.config.icl_dim)
    y = torch.randint(0, 3, (2, 20)).float()
    heads, dh = model.config.decoder_heads, model.config.decoder_head_dim
    # both query and key built for every row
    q, k = decoder.q_projection(h), decoder.k_projection(h)
    q = decoder.log_scale(q[:, 20:].reshape(2, -1, heads, dh).transpose(1, 2), 20)
    k = k[:, :20].reshape(2, -1, heads, dh).transpose(1, 2)
    weights = torch.softmax((q @ k.transpose(-1, -2)).float() / dh**0.5, dim=-1)
    onehot = torch.nn.functional.one_hot(y.long(), model.config.n_out).float()
    expected = torch.log((weights @ onehot[:, None]).mean(1).clamp_min(1e-5) + 3e-5)
    assert torch.allclose(decoder(h, y, 20), expected, atol=1e-5)


def test_ecdf_of_columns_constant_on_train_rows_is_flat():
    # A column constant on the train rows has, like z, the same ECDF on every row, so the test-only
    # signal is lost
    model = _model()
    x = torch.randn(1, 30, 4)
    x[0, :20, 1] = 3.0  # constant on the 20 train rows, different on test rows
    x[0, 20:, 1] = torch.linspace(-5, 5, 10)
    seen = {}
    hook = model.cell_encoder.register_forward_hook(lambda m, args, out: seen.update(u=args[2]))
    model._embed_cells(x, 20)
    hook.remove()
    # slot 0 of feature j's group is column j itself
    assert torch.equal(seen["u"][0, :, 1, 0], torch.full((30,), 0.5))
    assert not torch.equal(seen["u"][0, :, 0, 0], torch.full((30,), 0.5))


def test_bad_options_raise():
    with pytest.raises(ValueError, match="nhead"):
        ColumnRowTransformerConfig(nhead=7)
    with pytest.raises(ValueError, match="column blocks"):
        ColumnRowTransformerConfig(column_blocks=3, row_blocks=2)


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
    source = hub.run_architecture(params)
    assert "max_features" not in source
    rebuilt = replace(TrainConfig(), **source, max_features=config.max_features)
    assert rebuilt.model_config() == config.model_config()
    assert len(result["history"]["val_loss"]) == 1


def test_ensembles_and_default_inference():
    # TabSil's inference defaults (8 standard members, 768 columns per member, balanced) for any
    # model without saved settings
    tabsil = {
        "n_estimators": 8,
        "member_kinds": ("standard",),
        "max_features_per_member": 768,
        "feature_subsampling": "balanced",
    }
    X, y = load_breast_cancer(return_X_y=True)
    X[::7, 3] = np.nan
    model = _model()
    defaults = recipe.default_inference(model, "classification")
    assert defaults == tabsil
    clf = EnsembleClassifier(model, **defaults).fit(X[:100], y[:100])
    proba = clf.predict_proba(X[100:130])
    assert proba.shape == (30, 2) and np.allclose(proba.sum(1), 1, atol=1e-5)

    Xr, yr = load_diabetes(return_X_y=True)
    reg_model = _model()
    defaults = recipe.default_inference(reg_model, "regression")
    assert defaults == tabsil
    reg = EnsembleRegressor(reg_model, BORDERS, **defaults)
    assert reg.max_features_per_member == 768 and reg.feature_subsampling == "balanced"
    pred = reg.fit(Xr[:100], yr[:100]).predict(Xr[100:120])
    assert pred.shape == (20,) and np.isfinite(pred).all()
    assert len(reg.preprocessors_) == 8
    # Settings saved with the weights come first
    reg_model.inference_config = {"n_estimators": 2, "member_kinds": ["standard"]}
    caps = {"max_features_per_member": 768, "feature_subsampling": "balanced"}
    assert recipe.default_inference(reg_model, "regression") == {
        "n_estimators": 2,
        "member_kinds": ("standard",),
        **caps,  # keys missing from the saved settings use the recipe's
    }
    reg_model.inference_config = {"max_features_per_member": None}
    assert recipe.default_inference(reg_model, "regression")["max_features_per_member"] is None


def test_standard_members_keep_values():
    from tabsil.inference.preprocessing import Member, MemberPreprocessor

    X = np.random.default_rng(0).normal(size=(50, 3))
    prep = MemberPreprocessor(Member(None, False, seed=0), [], 160).fit(X)
    out = prep.transform(X, np.random.default_rng(1))
    # 3 original columns + 1 sample id, shuffled
    assert out.shape == (50, 4)
    kept = out[:, np.argsort(prep.order)[:3]]
    assert np.allclose(kept, X)


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
    resumed = script.build_config(args, hub.run_architecture(params))
    assert resumed.model_config() == config.model_config()
    smoke = script.build_config(script.parse_args(["--config-name", "smoke"]))
    assert smoke == helpers.tabsil_smoke()
    # a run of another size is not taken
    other = {k: str(v) for k, v in vars(replace(config, n_cls=8, emsize=128)).items()}
    with pytest.raises(SystemExit, match="structure"):
        script.build_config(args, hub.run_architecture(other))
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
        resumed = script.build_config(args, hub.run_architecture(params))
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


def test_fresh_model_reads_its_input():
    # Even in a fresh model (zero-init layers intact), row vectors depend on the input
    # so gradients flow. With a large CLS token or a zero-init row-stage output layer, all rows were
    # nearly identical and gradients were 1e-8
    torch.manual_seed(0)
    model = helpers.tabsil_smoke().build_model()
    x, y = _data("classification")
    out = model(x, y, "classification")
    assert out.std(dim=1).mean() > 1e-3  # different prediction per test row
    out[..., 0].sum().backward()
    assert model.row_blocks[0].attention.qkv.weight.grad.norm() > 1e-4
    assert model.heads["classification"].q_projection.weight.grad.norm() > 1e-4


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

    assert recipe.SUPPORTED_SIZE == {"max_train_rows": 1_000_000, "max_features": 6_000}
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


# ---------------------------------------------------------------------------
# One model for classification and regression
# ---------------------------------------------------------------------------
def test_model_needs_a_task_and_shares_everything_but_targets_and_heads():
    model = _model()
    assert model.tasks == TASKS
    x, yc = _data("classification")
    _, yr = _data("regression")
    assert model(x, yc, task="classification").shape == (2, 15, 10)
    assert model(x, yr, task="regression").shape == (2, 15, N_QUANTILES)
    with pytest.raises(TypeError):
        model(x, yc)
    # Only the target encoders (cell and ICL), the pre-output MLP and the output layer are per task
    per_task = {"cell_target", "icl_target", "pre_head", "heads"}
    for task, other in (("classification", "regression"), ("regression", "classification")):
        model.zero_grad()
        y = yc if task == "classification" else yr
        model(x, y, task=task)[..., 0].sum().backward()
        unused = {n for n, p in model.named_parameters() if p.grad is None}
        assert unused == {
            n
            for n, _ in model.named_parameters()
            if n.split(".")[0] in per_task and f".{other}" in n
        }


def test_labels_reach_the_icl_stage():
    # The target embedding enters both the cell stage and the row ICL input (as in the public
    # inference code)
    model = _model()
    assert set(model.icl_target) == set(TASKS)
    x, y = _data("regression")
    out = model(x, y, task="regression")
    with torch.no_grad():
        model.icl_target["regression"].encode.weight.mul_(3.0)
    assert not torch.allclose(model(x, y, task="regression"), out)


def test_recipe_and_config():
    config = recipe.curriculum(1)
    assert config.tasks == TASKS and config.rewind_lr_factor == 0.5
    with pytest.raises(ValueError, match="stage"):
        recipe.curriculum(4)
    with pytest.raises(ValueError, match="task"):
        config.make_prior("multitask", 0)


def test_ensembles_and_hub(tmp_path):
    model = _model()
    X, y = load_breast_cancer(return_X_y=True)
    clf = EnsembleClassifier(model, **recipe.default_inference(model, "classification"))
    proba = clf.fit(X[:80], y[:80]).predict_proba(X[80:100])
    assert proba.shape == (20, 2) and np.allclose(proba.sum(1), 1, atol=1e-5)
    Xr, yr = load_diabetes(return_X_y=True)
    reg = EnsembleRegressor(model, BORDERS, **recipe.default_inference(model, "regression"))
    assert np.isfinite(reg.fit(Xr[:80], yr[:80]).predict(Xr[80:100])).all()

    folder = hub.save_pretrained(model, BORDERS, tmp_path / "pretrained")
    assert json.loads((folder / "config.json").read_text())["task"] == "multitask"
    for task in TASKS:
        loaded, loaded_borders = hub.model_or_pretrained(str(folder), "unused", task)
        assert loaded.tasks == model.tasks and torch.equal(loaded_borders, BORDERS)
    assert hub.served_tasks(model, None) == ("classification",)  # no regression without borders


def test_parameter_counts_by_stage_and_task_add_up():
    config = recipe.curriculum(1).model_config()
    with torch.device("meta"):
        model = ColumnRowTransformer(config)
    by_stage = model.num_parameters_by_stage()
    assert list(by_stage) == ["cell_encoder", "column_stage", "row_stage", "row_icl", "output"]
    assert sum(by_stage.values()) == model.num_parameters()
    by_task = model.num_parameters_by_task()
    assert set(by_task) == {"shared", *TASKS}
    assert sum(by_task.values()) == model.num_parameters()
    assert all(v > 0 for v in by_task.values())


def test_tiny_config_and_randomize_zero_weights():
    config = ColumnRowTransformerConfig.tiny(n_out=10)
    assert (config.emsize, config.n_cls, config.nlayers, config.n_out) == (16, 2, 2, 10)
    torch.manual_seed(0)
    model = registry.randomize_zero_weights(ColumnRowTransformer(config))
    assert not any(p.dim() > 1 and (p == 0).all() for p in model.parameters())
    norms = [p for n, p in model.named_parameters() if n.endswith("row_norm.weight")]
    assert norms and all((p == 1).all() for p in norms)  # 1-D parameters unchanged
    x, y = _data("classification")
    with torch.no_grad():
        assert model(x, y, "classification").shape == (2, 15, 10)


# ---------------------------------------------------------------------------
# Inference on large tables and more than 10 classes
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("task", TASKS)
def test_inference_path_matches_the_training_path(task, monkeypatch):
    # The inference path (stages 0-2 in row/column chunks, decoder without the score matrix) matches
    # the training path
    model = _model()
    x, y = _data(task, r=41, c=8)
    x[1, 30, 4] = float("inf")
    x[:, :, 6] = 2.0  # column constant on train rows
    with torch.no_grad():
        full = model(x, y, task)
        # chunk sizes that do not divide the row/column counts (the last chunk is smaller)
        monkeypatch.setattr(model, "inference_chunk_cells", 2 * 8 * 6)
        monkeypatch.setattr(model, "inference_column_chunk", 3)
        chunked = model(x, y, task, inference=True)
        monkeypatch.setattr(model, "inference_chunk_cells", 10**9)
        fused = model(x, y, task, inference=True)
    assert torch.allclose(chunked, full, atol=1e-5), (chunked - full).abs().max()
    assert torch.allclose(fused, full, atol=1e-5)
    if task == "regression":  # regression shares the decoder; only stages 0-2 are chunked
        assert torch.equal(fused, full)


def test_fused_retrieval_matches_the_score_matrix_for_many_classes():
    # Folding the one-hot in head-dim slices: same result with more classes than head dim (8)
    torch.manual_seed(0)
    decoder = RetrievalDecoder(dim=16, heads=2, head_dim=8, n_classes=3, hidden=8).eval()
    with torch.no_grad():
        decoder.log_scale.head_scale.copy_(torch.tensor([0.7, 1.9]))
    h, y = torch.randn(2, 30, 16), torch.randint(0, 20, (2, 22)).float()
    with torch.no_grad():
        plain = decoder(h, y, 22, n_classes=20)
        fused = decoder(h, y, 22, n_classes=20, fused=True)
    assert plain.shape == fused.shape == (2, 8, 20)
    assert torch.allclose(fused, plain, atol=1e-5)
    assert torch.allclose(plain.exp().sum(-1), torch.ones(2, 8), atol=1e-3)


def test_more_classes_than_target_embeddings_share_embeddings():
    model = _model()  # 10 target embeddings
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 5))
    y = np.digitize(X[:, 0], np.quantile(X[:, 0], np.linspace(0, 1, 14)[1:-1]))  # 13 classes
    labels = np.array([f"c{i:02d}" for i in range(13)])[y]
    clf = EnsembleClassifier(model, n_estimators=3, member_kinds="standard")
    clf.fit(X[:200], labels[:200])
    proba = clf.predict_proba(X[200:])
    assert list(clf.classes_) == sorted(set(labels)) and proba.shape == (100, 13)
    assert np.allclose(proba.sum(1), 1, atol=1e-3)
    # Target embedding gets the group index (shuffled label mod 10); decoder gets the shuffled label
    y_member = torch.as_tensor(clf.y_trains_[0], dtype=torch.float32)[None]
    grouped, extra = clf._model_targets(y_member)
    assert grouped.max() < 10 and torch.equal(grouped, y_member % 10)
    assert extra["n_classes"] == 13 and torch.equal(extra["y_classes"], y_member)
    # More classes than the limit (MAX_CLASSES) are rejected
    with pytest.raises(ValueError, match="exceed what the model accepts"):
        EnsembleClassifier(model, n_estimators=1).fit(X[:200], np.arange(200) % 161)


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


def test_forward_rejects_labels_outside_the_target_embeddings():
    model = _model()  # 10 target embeddings
    x, y = torch.randn(1, 30, 4), torch.arange(20).float()[None] % 13  # 13 classes
    task = "classification"
    with pytest.raises(ValueError, match="out of range"):
        model(x, y, task)  # without grouping, labels 10-12 merge into 9
    with pytest.raises(ValueError, match="given together"):
        model(x, y % 10, task, y_classes=y)
    with pytest.raises(ValueError, match="within the target embeddings"):
        model(x, y, task, y_classes=y, n_classes=13)
    assert model(x, y % 10, task, y_classes=y, n_classes=13).shape == (1, 10, 13)
    assert model.max_classes == 160


def test_diag_scale_scales_queries_inside_the_attention_forward():
    # qkv weights are used directly, so forward hooks do not fire: attention applies δ in forward
    import copy

    from tabsil.inference.diag_scale import add_diag_scale

    model = _model()
    adapted = copy.deepcopy(model)
    deltas = add_diag_scale(adapted)
    n_attention = sum(1 for m in model.modules() if type(m).__name__ == "Attention")
    assert len(deltas) == n_attention > 0
    assert all(not m.qkv._forward_hooks for m in adapted.modules() if hasattr(m, "diag_scale"))
    x, yc = _data("classification")
    _, yr = _data("regression")
    with torch.no_grad():
        for task, y in (("classification", yc), ("regression", yr)):
            assert torch.equal(model(x, y, task=task), adapted(x, y, task=task))  # δ = 0
        for d in deltas:
            d.normal_(0, 0.5)
        task = "classification"
        assert not torch.allclose(model(x, yc, task=task), adapted(x, yc, task=task))
    adapted(x, yr, task="regression").square().mean().backward()
    assert all(d.grad is not None for d in deltas)
    assert set(adapted.state_dict()) - set(model.state_dict()) == {
        n for n, _ in adapted.named_parameters() if n.endswith("diag_scale")
    }


def test_diag_scale_trains_the_model():
    from tabsil.inference.diag_scale import DiagScale

    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 4))
    model = _model()
    y = np.arange(120) % 13  # more classes than target embeddings (10): forward with grouped labels
    estimator = EnsembleClassifier(model, n_estimators=2)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    fitted = DiagScale(estimator, epochs=2, patience=2).fit(X[:90], y[:90])
    info = fitted.adaptation_
    assert info["train_loss"] and len(info["val_curve"]) == 3 and info["n_params"] > 0
    assert fitted.predict_proba(X[90:]).shape == (30, 13)
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())  # original model


def test_tabsil_classes_share_the_multitask_repo(tmp_path, monkeypatch):
    # The published weights are one multitask model: both classes download the same repo
    from tabsil import TabSilClassifier, TabSilRegressor

    model = _model()
    folder = hub.save_pretrained(model, BORDERS, tmp_path / "tabsil")
    asked = []
    real = hub.from_pretrained

    def fake(spec, revision=None):
        asked.append(str(spec))
        repo = str(spec).partition("@")[0]
        return real(folder if repo == recipe.HUB_REPO else spec, revision)

    monkeypatch.setattr(hub, "from_pretrained", fake)
    clf, reg = TabSilClassifier(), TabSilRegressor()
    assert recipe.HUB_REPO == "aiden-jeon/tabsil"
    assert asked == [recipe.HUB_REPO, recipe.HUB_REPO]
    for loaded in (clf.model, reg.model):
        assert loaded.config == model.config
        for name, tensor in model.state_dict().items():
            assert torch.equal(loaded.state_dict()[name], tensor), name
    assert torch.equal(reg.borders.cpu(), BORDERS)
    # TabSil's inference settings (8 standard members)
    assert (clf.n_estimators, clf.member_kinds) == (8, ("standard",))
    assert (reg.n_estimators, reg.member_kinds) == (8, ("standard",))
    TabSilClassifier(recipe.HUB_REPO + "@stage2-20k")
    assert asked[-1] == recipe.HUB_REPO + "@stage2-20k"
    # A folder without borders is not taken for regression
    cls_only = hub.save_pretrained(_model(), None, tmp_path / "cls")
    with pytest.raises(ValueError, match="regression"):
        TabSilRegressor(str(cls_only))
    with pytest.raises(ValueError, match="borders"):
        TabSilRegressor(model)


def test_tabsil_default_feature_cap_in_classes_and_eval(monkeypatch, tmp_path):
    # The default inference puts at most 768 columns per member, balanced (issue #99). Arguments
    # come first
    from tabsil import TabSilClassifier, TabSilRegressor

    model = _model()
    clf = TabSilClassifier(model)
    assert clf.max_features_per_member == 768 and clf.feature_subsampling == "balanced"
    clf = TabSilClassifier(model, max_features_per_member=None, feature_subsampling="random")
    assert clf.max_features_per_member is None and clf.feature_subsampling == "random"
    clf = TabSilClassifier(model, n_estimators=3, member_kinds="standard")
    assert (clf.n_estimators, clf.member_kinds) == (3, ("standard",))
    reg = TabSilRegressor(_model(), BORDERS)
    assert (reg.n_estimators, reg.member_kinds) == (8, ("standard",))
    # The plain ensemble has no cap
    assert EnsembleClassifier(model, n_estimators=2).max_features_per_member is None

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


def test_diag_scale_views_keep_feature_subsampling():
    from tabsil.inference.diag_scale import _views

    model = _model()
    est = EnsembleClassifier(
        model, n_estimators=2, max_features_per_member=3, feature_subsampling="balanced"
    )
    views = _views(est, model, n_views=4, random_state=0)
    assert views.max_features_per_member == 3 and views.feature_subsampling == "balanced"
