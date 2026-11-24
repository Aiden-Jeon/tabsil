"""Reading checkpoints and Hub config.json files written by minipfn for the TabSil structure
(`ColumnRowTransformerConfig.from_dict`, `fixed_fields_check`). minipfn MLflow run params and
training from a minipfn checkpoint are tested in training/tests (tabsil-train).

minipfn's config of this model had fields that TabSil fixes in the code (`FIXED_FIELDS`). They are
accepted only with the TabSil value; a minipfn dict that lacks one of them had it at minipfn's
default (minipfn left fields added after the first release out of the dict at their default).
"""

import json

import pytest
import torch
from helpers import smoke_model, smoke_model_config
from safetensors.torch import save_file

from tabsil.hub import from_pretrained
from tabsil.models import config_from_dict, model_from_state
from tabsil.models.column_row_transformer import FIXED_FIELDS, ColumnRowTransformerConfig

# `minipfn.models.config_dict` of the minipfn-v1 smoke model, exactly as minipfn 8a7c81c wrote it
# (`with_architecture(TrainConfig.smoke("multitask"), keep_size=True).build_model().config`)
MINIPFN_SMOKE_CONFIG = {
    "task": "multitask",
    "emsize": 32,
    "features_per_group": 3,
    "n_inducing": 8,
    "column_blocks": 2,
    "column_heads": 4,
    "n_cls": 4,
    "row_blocks": 2,
    "row_heads": 4,
    "rope_base": 100000.0,
    "nlayers": 2,
    "nhead": 2,
    "ff_factor": 2,
    "decoder_heads": 6,
    "decoder_head_dim": 64,
    "qassmax_hidden": 64,
    "qk_norm": True,
    "encoder": "fourier",
    "target_norm": True,
    "head_mlp": True,
    "n_out": 10,
    "n_buckets": 5000,
    "max_features": 8,
    "qk_norm_affine": False,
    "column_summary_temperature": "logscale",
    "row_temperature": "gated_logscale",
    "icl_temperature": "logscale",
    "decoder_temperature": "logscale",
    "stage_layout": "interleaved",
    "regression_head": "quantile",
}


def _smoke_model():
    torch.manual_seed(0)
    return smoke_model().eval()


def test_minipfn_config_dict_reads_as_the_tabsil_smoke_config():
    config = config_from_dict("column_row_transformer", MINIPFN_SMOKE_CONFIG)
    assert config == smoke_model_config()  # TabSil's smoke model (`TrainConfig.smoke()`)
    assert config.gate_hidden == 64  # renamed from qassmax_hidden
    # the dict is not changed
    assert "qassmax_hidden" in MINIPFN_SMOKE_CONFIG and "task" in MINIPFN_SMOKE_CONFIG


def test_minipfn_training_checkpoint_loads_and_takes_weights(tmp_path):
    model = _smoke_model()
    borders = torch.linspace(-3, 3, 11)
    state = {
        "model_type": "column_row_transformer",
        "model_config": dict(MINIPFN_SMOKE_CONFIG),
        "model": model.state_dict(),
        "borders": borders,
        "optimizer_type": "muon",
    }
    loaded, loaded_borders = model_from_state(state)
    assert loaded.config == model.config and torch.equal(loaded_borders, borders)
    # init_from (stage 2 from a minipfn stage 1 run) accepting it is tested in training/tests
    x, y = torch.randn(1, 30, 5), torch.randint(0, 3, (1, 20)).float()
    torch.testing.assert_close(loaded(x, y, "classification"), model(x, y, "classification"))


def test_minipfn_hub_folder_loads(tmp_path):
    model = _smoke_model()
    borders = torch.linspace(-3, 3, 11)
    tensors = {k: v.contiguous() for k, v in model.state_dict().items()} | {"borders": borders}
    save_file(tensors, tmp_path / "model.safetensors")
    config = {
        "task": "multitask",
        "model_type": "column_row_transformer",
        "model": MINIPFN_SMOKE_CONFIG,
        "inference": {"n_estimators": 8, "member_kinds": ["standard"]},
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    loaded, loaded_borders = from_pretrained(tmp_path)
    assert loaded.config == model.config and torch.equal(loaded_borders, borders)
    assert loaded.inference_config["n_estimators"] == 8


@pytest.mark.parametrize(
    "name, value",
    [
        ("task", "classification"),
        ("stage_layout", "sequential"),
        ("logit_softcap", 30.0),
        ("qk_norm_affine", True),
        ("row_temperature", "none"),
        ("icl_temperature", "qassmax"),
        ("regression_head", "bar"),
        ("encoder", "linear"),
        ("categorical_encoder", True),
        ("decoder_qk_norm", True),
        ("head_mlp", False),
    ],
)
def test_minipfn_config_of_another_structure_is_refused(name, value):
    with pytest.raises(ValueError, match=name):
        config_from_dict("column_row_transformer", {**MINIPFN_SMOKE_CONFIG, name: value})


@pytest.mark.parametrize("name", ["stage_layout", "regression_head", "qk_norm_affine"])
def test_missing_field_of_a_minipfn_dict_is_its_minipfn_default(name):
    # minipfn left a field at its default out of the dict: those defaults are not TabSil's
    values = {k: v for k, v in MINIPFN_SMOKE_CONFIG.items() if k != name}
    with pytest.raises(ValueError, match=name):
        config_from_dict("column_row_transformer", values)


def test_every_fixed_field_holds_the_tabsil_value_in_the_minipfn_dict():
    for name, (old_default, fixed) in FIXED_FIELDS.items():
        assert MINIPFN_SMOKE_CONFIG.get(name, old_default) == fixed, name


def test_tabsil_dict_has_no_fixed_fields_and_unknown_fields_raise():
    saved = ColumnRowTransformerConfig.tiny().to_dict()
    assert not set(FIXED_FIELDS) & set(saved)
    assert config_from_dict("column_row_transformer", saved) == ColumnRowTransformerConfig.tiny()
    with pytest.raises(ValueError, match="cell_dropout"):
        config_from_dict("column_row_transformer", {**saved, "cell_dropout": 0.1})
    # a removed field in a TabSil dict (no `task` marker) is an unknown field
    with pytest.raises(ValueError, match="stage_layout"):
        config_from_dict("column_row_transformer", {**saved, "stage_layout": "interleaved"})
