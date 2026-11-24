"""Model registry tests: TabSil's model type reads back, and other models go through checkpoints,
the Hub and the inference classes (training other models: training/tests)."""

import json
from dataclasses import asdict, dataclass

import pytest
import torch
from helpers import smoke_model
from torch import nn

from tabsil import hub
from tabsil import models as registry
from tabsil.models import save_checkpoint
from tabsil.models.column_row_transformer import ColumnRowTransformer, ColumnRowTransformerConfig


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


def test_checkpoint_records_the_model_type_and_files_without_it_read_as_the_default(tmp_path):
    model = smoke_model()
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
    model = smoke_model()
    out = hub.save_pretrained(model, None, tmp_path / "clf")
    config = json.loads((out / hub.CONFIG_NAME).read_text())
    # Old from_pretrained reads only config["model"], so model_type lives outside it
    assert config["model_type"] == "column_row_transformer" and "model_type" not in config["model"]
    del config["model_type"]  # a repo without a model type reads as the default
    (out / hub.CONFIG_NAME).write_text(json.dumps(config))
    loaded, _ = hub.from_pretrained(out)
    assert isinstance(loaded, ColumnRowTransformer)
    _same_weights(model, loaded)


def test_inference_defaults_for_another_model_type(tiny):
    # Inference classes use TabSil defaults for models added to the registry too
    from tabsil.recipe import INFERENCE, TabSilClassifier, default_inference

    model = TinyModel(TinyConfig())
    assert default_inference(model, "classification")["member_kinds"] == INFERENCE["member_kinds"]
    clf = TabSilClassifier(model)
    assert clf.member_kinds == INFERENCE["member_kinds"]
    assert clf.n_estimators == INFERENCE["n_estimators"]["classification"]
    assert clf.max_features_per_member == INFERENCE["max_features_per_member"]
