"""`tabsil_train.hub.load_model`: Hub repos (`hf://`), `save_pretrained` folders, checkpoint files
and MLflow run checkpoints (`runs:/`), with no Hub or MLflow server."""

import mlflow
import pytest
import torch
from train_helpers import tabsil_smoke

from tabsil import hub
from tabsil.models import save_checkpoint
from tabsil_train import hub as train_hub


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


def test_load_model_reads_every_kind_of_checkpoint(tmp_path):
    clf = _model()
    save_checkpoint(tmp_path / "final.pt", clf)
    hub.save_pretrained(clf, None, tmp_path / "pretrained")
    for spec in (tmp_path / "final.pt", tmp_path / "pretrained"):
        loaded, borders = train_hub.load_model(str(spec), "classification", tmp_path)
        assert borders is None
        _same_weights(clf, loaded)
    with pytest.raises(SystemExit, match="needs a regression model"):
        train_hub.load_model(str(tmp_path / "pretrained"), "regression", tmp_path)


def test_training_checkpoint_without_model_config_is_refused(tmp_path):
    path = _training_state(tmp_path, _model())
    with pytest.raises(SystemExit, match="no model config"):
        train_hub.load_model(str(path), "classification", tmp_path)


def test_hub_needs_the_hf_prefix_and_missing_local_paths_fail(tmp_path, monkeypatch, chdir_tmp):
    seen = []
    monkeypatch.setattr(
        hub,
        "from_pretrained",
        lambda spec, revision=None: seen.append((spec, revision)) or (_model(), None),
    )
    train_hub.load_model("hf://owner/name", "classification", tmp_path)
    assert seen == [("owner/name", None)]
    # intermediate checkpoint branch
    train_hub.load_model("hf://owner/name@step-100k", "classification", tmp_path)
    assert seen[-1] == ("owner/name", "step-100k")
    with pytest.raises(SystemExit, match="hf://"):
        train_hub.load_model("models/clf", "classification", tmp_path)  # mistyped local path
    assert len(seen) == 2


def test_hf_spec_without_a_revision_is_main(tmp_path, monkeypatch):
    # The Hub tag a tabsil release pins is the default of `TabSilClassifier()` only: a repo given
    # as `hf://owner/name` keeps its own revision (`main`)
    from tabsil.recipe import HUB_REPO

    hub.save_pretrained(_model(), torch.linspace(-3, 3, 5001), tmp_path / "repo")
    calls = []

    def download(repo_id, filename, revision=None):
        calls.append((repo_id, revision))
        return str(tmp_path / "repo" / filename)

    monkeypatch.setattr(hub, "hf_hub_download", download)
    train_hub.load_model(f"hf://{HUB_REPO}", "classification", tmp_path)
    assert set(calls) == {(HUB_REPO, None)}


def test_training_checkpoint_from_run_carries_its_model_config(tmp_path, monkeypatch):
    from tabsil_train.training.loop import train

    config = tabsil_smoke(steps=1, border_datasets=20)
    train(config, tmp_path / "run", mlflow_log=False)
    path = tmp_path / "run" / "latest.pt"
    seen = {}

    def download(artifact_uri, dst_path):
        seen["uri"] = artifact_uri
        return str(path)

    monkeypatch.setattr(mlflow.artifacts, "download_artifacts", download)
    uri = "runs:/abc123/checkpoints/latest.pt"
    loaded, borders = train_hub.load_model(uri, "classification", tmp_path)
    assert seen["uri"] == uri and not loaded.training
    state = torch.load(path, weights_only=True)
    assert torch.equal(borders, state["borders"])  # training checkpoints keep the borders
    for name, tensor in state["model"].items():
        assert torch.equal(loaded.state_dict()[name], tensor), name
    # with its borders the same checkpoint serves regression too
    assert hub.served_tasks(*train_hub.load_model(uri, "regression", tmp_path)) == (
        "classification",
        "regression",
    )


def test_run_uri_to_a_pretrained_folder(tmp_path, monkeypatch):
    clf = _model()
    folder = hub.save_pretrained(clf, None, tmp_path / "pretrained")
    monkeypatch.setattr(
        mlflow.artifacts, "download_artifacts", lambda artifact_uri, dst_path: str(folder)
    )
    loaded, _ = train_hub.load_model("runs:/abc/model", "classification", tmp_path)
    _same_weights(clf, loaded)


def test_spec_given_as_a_path_and_a_missing_path(tmp_path):
    clf = _model()
    folder = hub.save_pretrained(clf, None, tmp_path / "pretrained")
    loaded, borders = train_hub.load_model(folder, "classification", tmp_path)  # a Path too
    assert borders is None
    _same_weights(clf, loaded)
    with pytest.raises(SystemExit, match="hf://"):
        train_hub.load_model(str(tmp_path / "missing"), "classification", tmp_path)


def test_run_uri_of_the_wrong_task_is_named_in_the_error(tmp_path, monkeypatch):
    folder = hub.save_pretrained(_model(), None, tmp_path / "pretrained")
    monkeypatch.setattr(
        mlflow.artifacts, "download_artifacts", lambda artifact_uri, dst_path: str(folder)
    )
    with pytest.raises(SystemExit, match="runs:/abc/model is a classification model"):
        train_hub.load_model("runs:/abc/model", "regression", tmp_path)
