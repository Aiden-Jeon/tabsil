"""Hub format tests (no Hub server). `load_model` (Hub repos, folders, checkpoint files and MLflow
run checkpoints) is `tabsil_train.hub.load_model`, tested in training/tests/test_hub_runs.py."""

import json
import subprocess
import sys

import pytest
import torch
from helpers import smoke_model

from tabsil import hub
from tabsil.models import config_dict


def _model():
    # one multitask model; saved without borders it serves classification only
    torch.manual_seed(0)
    return smoke_model().eval()


def _same_weights(a, b):
    for name, tensor in a.state_dict().items():
        assert torch.equal(b.state_dict()[name], tensor), name


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


def test_borders_keep_their_dtype(tmp_path):
    edges = torch.linspace(-3, 3, 5001, dtype=torch.float64)
    _, borders = hub.from_pretrained(hub.save_pretrained(_model(), edges, tmp_path))
    assert borders.dtype == torch.float64 and torch.equal(borders, edges)


def test_config_json_holds_the_model_config(tmp_path):
    # config.json holds the model sizes only; the fixed structure is not written
    tabsil = _model()
    borders = torch.linspace(-3, 3, 5001)
    out = hub.save_pretrained(tabsil, borders, tmp_path / "tabsil")
    saved = json.loads((out / hub.CONFIG_NAME).read_text())
    assert saved["task"] == "multitask" and saved["model_type"] == "column_row_transformer"
    assert not {"task", "stage_layout", "regression_head"} & set(saved["model"])
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
    from tabsil.recipe import HUB_DEFAULT, HUB_REPO

    assert HUB_REPO == "aiden-jeon/tabsil" and HUB_DEFAULT == "aiden-jeon/tabsil@v1.1"
    model = _model()
    borders = torch.linspace(-3, 3, 5001)
    tabsil_dir, cls_dir = tmp_path / "tabsil", tmp_path / "cls"
    hub.save_pretrained(model, borders, tabsil_dir, inference={"n_estimators": 3})
    torch.manual_seed(1)
    cls_model = smoke_model().eval()
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
    assert clf.member_kinds == ("standard", "power", "robust", "quantile")
    assert clf.max_features_per_member == 768
    assert clf.feature_subsampling == "balanced"
    reg = TabSilRegressor(n_estimators=2)
    _same_weights(model, reg.model)
    assert torch.equal(reg.borders.cpu(), borders) and reg.n_estimators == 2
    assert asked == [HUB_DEFAULT, HUB_DEFAULT]  # the tag `v1`, not `main`
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


def test_import_tabsil_loads_the_hub_and_estimators_on_first_use():
    # `import tabsil` does not import the Hub or inference code; the names load on first use.
    # `tabsil.models` comes with it (`tabsil.models.load_checkpoint` after a bare `import tabsil`)
    code = (
        "import sys, tabsil\n"
        "assert not {'tabsil.hub', 'tabsil.recipe', 'tabsil.tasks'} & set(sys.modules)\n"
        "assert tabsil.models.load_checkpoint\n"
        "from tabsil import TabSilClassifier, TabSilRegressor, from_pretrained\n"
        "import tabsil.hub, tabsil.tasks.classification, tabsil.tasks.regression\n"
        "assert from_pretrained is tabsil.hub.from_pretrained\n"
        "assert TabSilClassifier is tabsil.tasks.classification.TabSilClassifier\n"
        "assert TabSilRegressor is tabsil.tasks.regression.TabSilRegressor\n"
        "assert 'tabsil_train' not in sys.modules and 'mlflow' not in sys.modules\n"
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


def test_default_weights_are_the_pinned_tag_and_other_revisions_pass_through(monkeypatch, tmp_path):
    # A tabsil release pins the weights it downloads by default: the Hub tag `v1.1`, so moving
    # `main` does not change what `TabSilClassifier()` runs. A repo id given by the caller keeps
    # its own revision: a bare repo id is `main`, `@branch` a branch, and `revision=` is passed as
    # is
    from tabsil import TabSilClassifier, TabSilRegressor
    from tabsil.recipe import HUB_REPO, HUB_REVISION

    assert HUB_REVISION == "v1.1"
    model = _model()
    hub.save_pretrained(model, torch.linspace(-3, 3, 5001), tmp_path)
    calls = []

    def download(repo_id, filename, revision=None):
        calls.append((repo_id, revision))
        return str(tmp_path / filename)

    monkeypatch.setattr(hub, "hf_hub_download", download)

    def revisions(load):
        calls.clear()
        load()
        return set(calls)

    assert revisions(TabSilClassifier) == {(HUB_REPO, "v1.1")}
    assert revisions(TabSilRegressor) == {(HUB_REPO, "v1.1")}
    assert revisions(lambda: TabSilClassifier(HUB_REPO)) == {(HUB_REPO, None)}
    assert revisions(lambda: TabSilClassifier(f"{HUB_REPO}@stage1-150k")) == {
        (HUB_REPO, "stage1-150k")
    }
    assert revisions(lambda: TabSilRegressor(f"{HUB_REPO}@stage2-10k")) == {
        (HUB_REPO, "stage2-10k")
    }
    assert revisions(lambda: hub.from_pretrained(HUB_REPO, revision="67ef600")) == {
        (HUB_REPO, "67ef600")
    }
