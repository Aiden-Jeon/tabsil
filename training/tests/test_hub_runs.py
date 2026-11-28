"""`tabsil_train.hub.load_model`: Hub repos (`hf://`), `save_pretrained` folders, checkpoint files
and MLflow run checkpoints (`runs:/`), with no Hub or MLflow server; and what importing
`tabsil_train.hub` loads."""

import subprocess
import sys

import mlflow
import pytest
import torch
from train_helpers import TRAINING_DIR, tabsil_smoke

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
        lambda spec, revision=None, **_: seen.append((spec, revision)) or (_model(), None),
    )
    train_hub.load_model("hf://owner/name", "classification", tmp_path)
    assert seen == [("owner/name", None)]
    # intermediate checkpoint branch
    train_hub.load_model("hf://owner/name@step-100k", "classification", tmp_path)
    assert seen[-1] == ("owner/name", "step-100k")
    with pytest.raises(SystemExit, match="hf://"):
        train_hub.load_model("models/clf", "classification", tmp_path)  # mistyped local path
    assert len(seen) == 2


def test_hf_spec_is_the_hub_when_a_local_folder_has_the_repo_name(tmp_path, monkeypatch, chdir_tmp):
    # `hf://owner/name` is the Hub repo, also when the working directory holds a folder
    # `owner/name` (a local download of the repo, say). The folder is read when given as a path
    local = _model()
    torch.manual_seed(1)
    remote = tabsil_smoke().build_model().eval()
    with pytest.raises(AssertionError):
        _same_weights(local, remote)  # two different models
    hub.save_pretrained(local, None, "owner/name")  # ./owner/name: the cwd is tmp_path
    hub.save_pretrained(remote, None, tmp_path / "hub")
    calls = []

    def download(repo_id, filename, revision=None):
        calls.append((repo_id, revision))
        return str(tmp_path / "hub" / filename)

    monkeypatch.setattr(hub, "hf_hub_download", download)
    for spec in ("hf://owner/name@stage2-10k", "hf://owner/name"):
        _same_weights(remote, train_hub.load_model(spec, "classification", tmp_path)[0])
    assert set(calls) == {("owner/name", "stage2-10k"), ("owner/name", None)}
    calls.clear()
    _same_weights(local, train_hub.load_model("owner/name", "classification", tmp_path)[0])
    assert not calls


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
