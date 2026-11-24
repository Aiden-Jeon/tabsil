"""Checkpoint loading and Hub format tests (no MLflow or Hub server)."""

import json
import subprocess
import sys

import mlflow
import pytest
import torch
from helpers import tabsil_smoke

from tabsil import hub
from tabsil.models import config_dict, save_checkpoint


@pytest.fixture
def chdir_tmp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # keep relative paths from hitting real repo folders


def _model():
    # one multitask model; saved without borders it serves classification only
    torch.manual_seed(0)
    return tabsil_smoke().build_model().eval()


def _same_weights(a, b):
    for name, tensor in a.state_dict().items():
        assert torch.equal(b.state_dict()[name], tensor), name


def _training_state(tmp_path, model):
    path = tmp_path / "step_0000100.pt"
    torch.save({"model": model.state_dict(), "step": 100, "borders": None}, path)
    return path


def test_save_and_load_pretrained_keeps_weights_and_borders(tmp_path):
    clf = _model()
    out = hub.save_pretrained(clf, None, tmp_path / "clf", inference={"n_estimators": 2})
    assert json.loads((out / hub.CONFIG_NAME).read_text())["inference"] == {"n_estimators": 2}
    loaded, borders = hub.from_pretrained(out)
    assert borders is None and not loaded.training
    _same_weights(clf, loaded)

    reg, edges = _model(), torch.linspace(-3, 3, 5001)
    loaded, borders = hub.from_pretrained(hub.save_pretrained(reg, edges, tmp_path / "reg"))
    assert torch.equal(borders, edges)
    _same_weights(reg, loaded)


def test_load_model_reads_every_kind_of_checkpoint(tmp_path):
    clf = _model()
    save_checkpoint(tmp_path / "final.pt", clf)
    hub.save_pretrained(clf, None, tmp_path / "pretrained")
    for spec in (tmp_path / "final.pt", tmp_path / "pretrained"):
        loaded, borders = hub.load_model(str(spec), "classification", tmp_path)
        assert borders is None
        _same_weights(clf, loaded)
    with pytest.raises(SystemExit, match="needs a regression model"):
        hub.load_model(str(tmp_path / "pretrained"), "regression", tmp_path)


def test_training_checkpoint_without_model_config_is_refused(tmp_path):
    path = _training_state(tmp_path, _model())
    with pytest.raises(SystemExit, match="no model config"):
        hub.load_model(str(path), "classification", tmp_path)


def test_training_checkpoint_from_run_carries_its_model_config(tmp_path, monkeypatch):
    from tabsil.training.loop import train

    config = tabsil_smoke(steps=1, border_datasets=20)
    train(config, tmp_path / "run", mlflow_log=False)
    path = tmp_path / "run" / "latest.pt"
    seen = {}

    def download(artifact_uri, dst_path):
        seen["uri"] = artifact_uri
        return str(path)

    monkeypatch.setattr(mlflow.artifacts, "download_artifacts", download)
    uri = "runs:/abc123/checkpoints/latest.pt"
    loaded, borders = hub.load_model(uri, "classification", tmp_path)
    assert seen["uri"] == uri and not loaded.training
    state = torch.load(path, weights_only=True)
    assert torch.equal(borders, state["borders"])  # training checkpoints keep the borders
    for name, tensor in state["model"].items():
        assert torch.equal(loaded.state_dict()[name], tensor), name
    # with its borders the same checkpoint serves regression too
    assert hub.served_tasks(*hub.load_model(uri, "regression", tmp_path)) == (
        "classification",
        "regression",
    )


def test_hub_needs_the_hf_prefix_and_missing_local_paths_fail(tmp_path, monkeypatch, chdir_tmp):
    seen = []
    monkeypatch.setattr(
        hub,
        "from_pretrained",
        lambda spec, revision=None: seen.append((spec, revision)) or (_model(), None),
    )
    hub.load_model("hf://owner/name", "classification", tmp_path)
    assert seen == [("owner/name", None)]
    hub.load_model(
        "hf://owner/name@step-100k", "classification", tmp_path
    )  # intermediate checkpoint branch
    assert seen[-1] == ("owner/name", "step-100k")
    with pytest.raises(SystemExit, match="hf://"):
        hub.load_model("models/clf", "classification", tmp_path)  # mistyped local path
    assert len(seen) == 2


def test_borders_keep_their_dtype(tmp_path):
    edges = torch.linspace(-3, 3, 5001, dtype=torch.float64)
    _, borders = hub.from_pretrained(hub.save_pretrained(_model(), edges, tmp_path))
    assert borders.dtype == torch.float64 and torch.equal(borders, edges)


def test_run_uri_to_a_pretrained_folder(tmp_path, monkeypatch):
    clf = _model()
    folder = hub.save_pretrained(clf, None, tmp_path / "pretrained")
    monkeypatch.setattr(
        mlflow.artifacts, "download_artifacts", lambda artifact_uri, dst_path: str(folder)
    )
    loaded, _ = hub.load_model("runs:/abc/model", "classification", tmp_path)
    _same_weights(clf, loaded)


def test_config_json_holds_the_model_config(tmp_path):
    # config.json holds the model sizes only; the fixed structure (minipfn's option fields) is not
    # written
    tabsil = _model()
    borders = torch.linspace(-3, 3, 5001)
    out = hub.save_pretrained(tabsil, borders, tmp_path / "tabsil")
    saved = json.loads((out / hub.CONFIG_NAME).read_text())
    assert saved["task"] == "multitask" and saved["model_type"] == "column_row_transformer"
    assert not {"task", "stage_layout", "regression_head", "qassmax_hidden"} & set(saved["model"])
    assert saved["model"] == config_dict(tabsil.config)
    loaded, loaded_borders = hub.from_pretrained(out)
    assert loaded.config == tabsil.config and torch.equal(loaded_borders, borders)
    assert hub.served_tasks(loaded, loaded_borders) == ("classification", "regression")
    assert hub.served_tasks(loaded, None) == ("classification",)
    _same_weights(tabsil, loaded)


def test_inference_classes_load_weights_without_from_pretrained(tmp_path, monkeypatch):
    # Users run public weights, so the inference classes fetch them directly: no argument means
    # TabSil's public repo; a string is a Hub repo id (`owner/name@branch`) or a local folder
    from tabsil import TabSilClassifier, TabSilRegressor
    from tabsil.recipe import HUB_REPO

    assert HUB_REPO == "aiden-jeon/tabsil"
    model = _model()
    borders = torch.linspace(-3, 3, 5001)
    tabsil_dir, cls_dir = tmp_path / "tabsil", tmp_path / "cls"
    hub.save_pretrained(model, borders, tabsil_dir, inference={"n_estimators": 3})
    torch.manual_seed(1)
    cls_model = tabsil_smoke().build_model().eval()
    hub.save_pretrained(cls_model, None, cls_dir)  # without borders: classification only

    # Return a local folder instead of the Hub; record whether it was called with the public repo
    asked = []
    real = hub.from_pretrained

    def fake(spec, revision=None):
        asked.append(str(spec))
        return real(tabsil_dir if str(spec).partition("@")[0] == HUB_REPO else spec, revision)

    monkeypatch.setattr(hub, "from_pretrained", fake)

    clf = TabSilClassifier()
    _same_weights(model, clf.model)
    assert clf.n_estimators == 3  # the inference settings saved with the weights come first
    assert clf.member_kinds == ("standard",) and clf.max_features_per_member == 768
    assert clf.feature_subsampling == "balanced"
    reg = TabSilRegressor(n_estimators=2)
    _same_weights(model, reg.model)
    assert torch.equal(reg.borders.cpu(), borders) and reg.n_estimators == 2
    assert asked == [HUB_REPO, HUB_REPO]
    # A branch, local folder or model can also be passed directly
    TabSilClassifier(HUB_REPO + "@step-100k")
    assert asked[-1].endswith("@step-100k")
    _same_weights(cls_model, TabSilClassifier(str(cls_dir)).model)
    assert TabSilRegressor(model, borders).model is model

    # Wrong inputs are reported
    with pytest.raises(ValueError, match="is a classification model"):
        TabSilRegressor(str(cls_dir))
    with pytest.raises(ValueError, match="borders"):
        TabSilRegressor(model)
    with pytest.raises(ValueError, match="borders"):
        TabSilRegressor(str(tabsil_dir), borders)


def test_import_tabsil_loads_the_hub_and_recipe_on_first_use():
    # `import tabsil` does not import the Hub or inference code; the names load on first use
    code = (
        "import sys, tabsil\n"
        "assert 'tabsil.hub' not in sys.modules and 'tabsil.recipe' not in sys.modules\n"
        "from tabsil import TabSilClassifier, TabSilRegressor, from_pretrained\n"
        "import tabsil.hub, tabsil.recipe\n"
        "assert from_pretrained is tabsil.hub.from_pretrained\n"
        "assert TabSilClassifier is tabsil.recipe.TabSilClassifier\n"
        "assert TabSilRegressor is tabsil.recipe.TabSilRegressor\n"
        "assert 'tabsil.training.config' not in sys.modules and 'mlflow' not in sys.modules\n"
        "assert {'TabSilClassifier', 'from_pretrained'} <= set(dir(tabsil))\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_from_pretrained_reads_the_branch_after_at(monkeypatch, tmp_path):
    model = _model()
    hub.save_pretrained(model, None, tmp_path)
    calls = []

    def download(repo_id, filename, revision=None):
        calls.append((repo_id, revision))
        return str(tmp_path / filename)

    monkeypatch.setattr(hub, "hf_hub_download", download)
    _same_weights(model, hub.from_pretrained("owner/name@step-100k")[0])
    hub.from_pretrained("owner/name", revision="step-200k")
    assert set(calls) == {("owner/name", "step-100k"), ("owner/name", "step-200k")}
    with pytest.raises(ValueError, match="revision"):
        hub.from_pretrained("owner/name@step-100k", revision="step-200k")
