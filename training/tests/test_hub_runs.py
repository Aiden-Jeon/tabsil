"""Loading MLflow run checkpoints (`runs:/`, `tabsil_train.hub.load_model`; no MLflow server)."""

import mlflow
import pytest
import torch
from train_helpers import tabsil_smoke

from tabsil import hub
from tabsil_train import hub as train_hub


def _model():
    # one multitask model; saved without borders it serves classification only
    torch.manual_seed(0)
    return tabsil_smoke().build_model().eval()


def _same_weights(a, b):
    for name, tensor in a.state_dict().items():
        assert torch.equal(b.state_dict()[name], tensor), name


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


def test_other_specs_go_to_tabsil(tmp_path):
    clf = _model()
    folder = hub.save_pretrained(clf, None, tmp_path / "pretrained")
    loaded, borders = train_hub.load_model(str(folder), "classification", tmp_path)
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
