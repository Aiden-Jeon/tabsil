"""Loading MLflow run checkpoints (`runs:/`, `tabsil_train.hub.load_model`; no MLflow server), and
what importing `tabsil_train.hub` loads."""

import subprocess
import sys

import mlflow
import pytest
import torch
from train_helpers import TRAINING_DIR, tabsil_smoke

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


def test_loading_models_does_not_import_the_training_loop():
    # Evaluation and the Hub upload import `tabsil_train.hub` to load models. Only
    # `hub_init_checkpoint` (a training run that starts from Hub weights) needs the training loop,
    # and imports it when called. A new process, because this one has the loop loaded already
    code = """
import importlib.util
import sys

import tabsil_train.hub

assert "tabsil_train.training.loop" not in sys.modules, "tabsil_train.hub imports the loop"
spec = importlib.util.spec_from_file_location("eval_script", sys.argv[1])
spec.loader.exec_module(importlib.util.module_from_spec(spec))
assert "tabsil_train.training.loop" not in sys.modules, "eval.py imports the loop"
"""
    done = subprocess.run(
        [sys.executable, "-c", code, str(TRAINING_DIR / "scripts" / "eval.py")],
        capture_output=True,
        text=True,
        timeout=600,  # a blocked import fails the test instead of hanging the run
    )
    assert done.returncode == 0, done.stderr
