"""Tests of the `column_row_transformer` model (TabSil's backbone) and of the TabSil recipe
around it (inference side; training and the scripts are in training/tests)."""

import json

import helpers
import numpy as np
import pytest
import torch
from sklearn.datasets import load_breast_cancer, load_diabetes

from tabsil import hub, recipe
from tabsil import models as registry
from tabsil.inference.ensemble import EnsembleClassifier, EnsembleRegressor
from tabsil.models.column_row_transformer import (
    ColumnRowTransformer,
    ColumnRowTransformerConfig,
    RetrievalDecoder,
    ecdf_midrank,
    group_offsets,
)
from tabsil.models.quantile_distribution import N_QUANTILES

TASKS = ("classification", "regression")
BORDERS = torch.linspace(-3, 3, 5000 + 1)  # the training borders (tabsil-train `N_BUCKETS`)


def _model(**changes):
    """The smoke-size TabSil model, with the zero-initialized layers filled."""
    torch.manual_seed(0)
    model = helpers.smoke_model(**changes)
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


def _published_config():
    # the published structure (`recipe.STRUCTURE`); tabsil-train checks the curriculum builds it
    sizes = {k: v for k, v in recipe.STRUCTURE.items() if k != "model_type"}
    return ColumnRowTransformerConfig(**sizes)


def test_paper_size_matches_the_report():
    # TabSil (interleaved Kumo stages, quantile head): 221.5M
    config = _published_config()
    with torch.device("meta"):
        model = ColumnRowTransformer(config)
    assert abs(model.num_parameters() / 1e6 - 221.5) / 221.5 < 0.02
    assert config.icl_dim == 1024


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
    out = prep.transform(X, np.random.default_rng(1).standard_normal((len(X), 1)))
    # 3 original columns + 1 sample id, shuffled
    assert out.shape == (50, 4)
    kept = out[:, np.argsort(prep.order)[:3]]
    assert np.allclose(kept, X)


def test_fresh_model_reads_its_input():
    # Even in a fresh model (zero-init layers intact), row vectors depend on the input
    # so gradients flow. With a large CLS token or a zero-init row-stage output layer, all rows were
    # nearly identical and gradients were 1e-8
    torch.manual_seed(0)
    model = helpers.smoke_model()
    x, y = _data("classification")
    out = model(x, y, "classification")
    assert out.std(dim=1).mean() > 1e-3  # different prediction per test row
    out[..., 0].sum().backward()
    assert model.row_blocks[0].attention.qkv.weight.grad.norm() > 1e-4
    assert model.heads["classification"].q_projection.weight.grad.norm() > 1e-4


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
    config = _published_config()
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
    assert asked == [recipe.HUB_DEFAULT, recipe.HUB_DEFAULT]  # the same repo, at the tag v1
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


def test_tabsil_default_feature_cap_in_classes():
    # The default inference puts at most 768 columns per member, balanced (issue #99). Arguments
    # come first (the eval script's defaults: training/tests)
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
