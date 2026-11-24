"""Model registry tests: TabSil's model type reads back, and other models also train, save and
go through the Hub."""

import json
import types
from dataclasses import asdict, dataclass, replace

import mlflow
import pytest
import torch
from helpers import tabsil_smoke
from torch import nn

from tabsil import hub
from tabsil import models as registry
from tabsil.models import save_checkpoint
from tabsil.models.column_row_transformer import ColumnRowTransformer, ColumnRowTransformerConfig
from tabsil.training.config import TrainConfig
from tabsil.training.loop import train


@dataclass
class TinyConfig:
    emsize: int = 8
    nhead: int = 2
    n_out: int = 10

    def to_dict(self) -> dict:
        return asdict(self)


class TinyModel(nn.Module):
    """Tiny model that only follows the registry contract: per-row feature mean -> Linear. It
    serves both tasks with
    the same `n_out` outputs (classes or quantiles)."""

    model_type = "tiny"
    config_class = TinyConfig
    tasks = ("classification", "regression")

    def __init__(self, config: TinyConfig) -> None:
        super().__init__()
        self.config = config
        self.hidden = nn.Linear(2, config.emsize)
        self.head = nn.Linear(config.emsize, config.n_out)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def decoder_weight(self) -> None:
        return None

    def forward(self, x: torch.Tensor, y_train: torch.Tensor, task: str) -> torch.Tensor:
        n_train = y_train.shape[1]
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        row = x.mean(-1, keepdim=True)
        y_mean = y_train.mean(1, keepdim=True)[:, None].expand(-1, x.shape[1], 1)
        h = torch.relu(self.hidden(torch.cat([row, y_mean.to(row.dtype)], dim=-1)))
        return self.head(h[:, n_train:])


@pytest.fixture
def tiny(monkeypatch):
    monkeypatch.setitem(registry.MODEL_TYPES, "tiny", TinyModel)


def _same_weights(a, b):
    for name, tensor in a.state_dict().items():
        assert torch.equal(b.state_dict()[name], tensor), name


def test_build_model_from_dataclass_or_dict():
    config = ColumnRowTransformerConfig.tiny()
    assert registry.MODEL_TYPES.keys() == {"column_row_transformer"}
    assert registry.DEFAULT_MODEL_TYPE == "column_row_transformer"
    assert isinstance(registry.build_model("column_row_transformer", config), ColumnRowTransformer)
    assert registry.build_model("column_row_transformer", config.to_dict()).config == config
    with pytest.raises(ValueError, match="model_type"):
        registry.build_model("tabpfn_v9", config)
    with pytest.raises(ValueError, match="model_type"):
        registry.build_model("cell_transformer", config)  # TabPFN v2 is gone
    with pytest.raises(ValueError, match="model_type"):
        TrainConfig(model_type="tabpfn_v9")
    assert TrainConfig().model_type == "column_row_transformer"


def test_checkpoint_records_the_model_type_and_files_without_it_read_as_the_default(tmp_path):
    model = tabsil_smoke().build_model()
    save_checkpoint(tmp_path / "new.pt", model)
    payload = torch.load(tmp_path / "new.pt", weights_only=True)
    assert payload["model_type"] == "column_row_transformer"
    del payload["model_type"]  # a file without a model type
    torch.save(payload, tmp_path / "old.pt")
    for name in ("new.pt", "old.pt"):
        loaded, _ = registry.load_checkpoint(tmp_path / name)
        assert isinstance(loaded, ColumnRowTransformer)
        _same_weights(model, loaded)


def test_hub_config_records_the_model_type(tmp_path):
    model = tabsil_smoke().build_model()
    out = hub.save_pretrained(model, None, tmp_path / "clf")
    config = json.loads((out / hub.CONFIG_NAME).read_text())
    # Old from_pretrained reads only config["model"], so model_type lives outside it
    assert config["model_type"] == "column_row_transformer" and "model_type" not in config["model"]
    del config["model_type"]  # a repo without a model type reads as the default
    (out / hub.CONFIG_NAME).write_text(json.dumps(config))
    loaded, _ = hub.from_pretrained(out)
    assert isinstance(loaded, ColumnRowTransformer)
    _same_weights(model, loaded)


def test_other_model_types_train_save_and_load(tmp_path, tiny):
    config = replace(tabsil_smoke(), model_type="tiny", steps=4)
    assert config.model_config() == TinyConfig(emsize=config.emsize, nhead=config.nhead, n_out=10)
    result = train(config, tmp_path, mlflow_log=False)
    model, borders = registry.load_checkpoint(result["final_path"])
    # Without output-layer weights, training metrics are skipped and the run still finishes
    assert isinstance(model, TinyModel) and borders is not None
    assert result["history"]["step"][-1] == 4

    loaded, _ = hub.from_pretrained(hub.save_pretrained(model, borders, tmp_path / "pretrained"))
    assert isinstance(loaded, TinyModel)
    _same_weights(model, loaded)


def _fake_run(monkeypatch, path, params):
    run = types.SimpleNamespace(data=types.SimpleNamespace(params=params))
    monkeypatch.setattr(
        mlflow.artifacts, "download_artifacts", lambda artifact_uri, dst_path: str(path)
    )
    monkeypatch.setattr(
        mlflow, "MlflowClient", lambda: types.SimpleNamespace(get_run=lambda run_id: run)
    )


def test_training_checkpoints_carry_the_model_settings(tmp_path, monkeypatch, tiny):
    # Build the model from the checkpoint alone, without run params (TinyConfig-only fields kept)
    config = replace(tabsil_smoke(), model_type="tiny", steps=2)
    train(config, tmp_path, mlflow_log=False)
    state = torch.load(tmp_path / "latest.pt", weights_only=True)
    assert state["model_type"] == "tiny"
    assert state["model_config"] == config.model_config().to_dict()
    # With a `config` key, old code treats it as a save_checkpoint file and breaks looking for
    # state_dict. Splitting the keys makes old code read run params instead
    assert "config" not in state and "state_dict" not in state
    _fake_run(monkeypatch, tmp_path / "latest.pt", params={})  # KeyError if params are read
    loaded, _ = hub.load_model("runs:/abc/checkpoints/latest.pt", "classification", tmp_path)
    assert isinstance(loaded, TinyModel)
    assert isinstance(registry.load_checkpoint(tmp_path / "latest.pt")[0], TinyModel)


def test_training_checkpoints_without_model_settings_are_refused(tmp_path, monkeypatch):
    # Checkpoints saved before the model config was recorded are not read through run params
    config = tabsil_smoke()
    path = tmp_path / "step_0000100.pt"
    torch.save({"model": config.build_model().state_dict(), "borders": None}, path)
    _fake_run(monkeypatch, path, {"emsize": "32"})
    with pytest.raises(SystemExit, match="no model config"):
        hub.load_model("runs:/abc/checkpoints/step_0000100.pt", "classification", tmp_path)
    with pytest.raises(SystemExit, match="no model config"):
        hub.load_model(str(path), "classification", tmp_path)


def test_init_and_resume_need_the_same_model(tmp_path):
    config = tabsil_smoke(steps=2, collapse_threshold=None)
    train(config, tmp_path / "a", mlflow_log=False)
    latest = tmp_path / "a" / "latest.pt"
    # If only the head count differs, weight shapes match and load_state_dict silently loads
    other_heads = replace(config, nhead=4, steps=4)
    for kw in ({"init_from": latest}, {"resume_from": latest}):
        with pytest.raises(ValueError, match="nhead"):
            train(other_heads, tmp_path / "b", mlflow_log=False, **kw)
    # a checkpoint without the regression borders is not a checkpoint of this model
    state = torch.load(latest, weights_only=True)
    state["borders"] = None
    torch.save(state, tmp_path / "no_borders.pt")
    with pytest.raises(ValueError, match="regression borders"):
        train(config, tmp_path / "c", init_from=tmp_path / "no_borders.pt", mlflow_log=False)
    # Same config loads as is (save_checkpoint files too)
    final = tmp_path / "a" / "final.pt"
    train(replace(config, steps=1), tmp_path / "d", init_from=final, mlflow_log=False)


def test_run_params_parse_bools():
    assert hub._parse_param(False, "True") is True
    assert hub._parse_param(True, "False") is False
    assert hub._parse_param(3, "24") == 24 and hub._parse_param("linear", "fourier") == "fourier"
    with pytest.raises(ValueError):
        hub._parse_param(False, "1")


def test_unknown_model_fields_are_named(tmp_path):
    # Checkpoints with fields from other branches/newer code fail naming the field, not TypeError
    from tabsil.training.loop import check_checkpoint

    config = tabsil_smoke()
    values = {**config.model_config().to_dict(), "n_cls_tokens": 4}
    with pytest.raises(ValueError, match=r"does not know: \['n_cls_tokens'\]"):
        registry.build_model("column_row_transformer", values)
    state = {
        "model_type": "column_row_transformer",
        "model_config": {**config.model_config().to_dict(), "n_cls_tokens": 4},
        "borders": torch.zeros(3),
    }
    with pytest.raises(ValueError, match="init_from: .*does not know"):
        check_checkpoint(state, config, "init_from")


def test_old_prior_type_name_reads_as_the_new_name():
    # Also reads the old prior name (tabicl) from training run params
    config = TrainConfig(prior_type="tabicl")
    assert (config.model_type, config.prior_type) == ("column_row_transformer", "graph_scm")
    params = {"model_type": "column_row_transformer", "max_features": "500"}
    arch = hub.run_architecture(params)
    assert arch["model_type"] == "column_row_transformer"
    # Feature count seen in training is not architecture: it changes when resuming on longer tables
    assert "max_features" not in arch


def test_max_features_is_saved_and_not_compared_on_resume(tmp_path):
    from tabsil.training.loop import check_checkpoint

    config = TrainConfig()  # TabSil stage 1: max_features 100
    assert registry.config_dict(config.model_config())["max_features"] == 100
    wide = replace(config, max_features=500)
    assert registry.config_dict(wide.model_config())["max_features"] == 500
    # Resume a checkpoint trained with 100 using 500
    state = {
        "model_type": "column_row_transformer",
        "model_config": registry.config_dict(config.model_config()),
        "borders": torch.zeros(3),
    }
    check_checkpoint(state, wide, "init_from")


def test_inference_defaults_for_another_model_type(tiny):
    # Inference classes use TabSil defaults for models added to the registry too
    from tabsil.recipe import INFERENCE, TabSilClassifier, default_inference

    model = TinyModel(TinyConfig())
    assert default_inference(model, "classification")["member_kinds"] == INFERENCE["member_kinds"]
    clf = TabSilClassifier(model)
    assert clf.member_kinds == INFERENCE["member_kinds"]
    assert clf.n_estimators == INFERENCE["n_estimators"]["classification"]
    assert clf.max_features_per_member == INFERENCE["max_features_per_member"]
