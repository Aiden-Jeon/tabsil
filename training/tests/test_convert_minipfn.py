"""`training/hub/convert_minipfn.py`: a checkpoint in minipfn's format (the config with the fields
TabSil fixes, the marker `task` and `qassmax_hidden`) becomes a `save_pretrained` folder with the
same tensors, and a config that is not TabSil's structure is refused."""

import pytest
import torch
from train_helpers import load_script, tiny_model

from tabsil import hub
from tabsil.models import config_dict
from tabsil_train.hub import weights_digest

convert = load_script("hub/convert_minipfn.py", "convert_minipfn_script")


def _minipfn_state(**config_changes):
    # the tiny model saved as minipfn did: every fixed field at TabSil's value, plus the old name
    model = tiny_model(seed=1)
    config = config_dict(model.config)
    config["qassmax_hidden"] = config.pop("gate_hidden")
    config.update({name: fixed for name, (_, fixed) in convert.FIXED_FIELDS.items()})
    config.update(config_changes)
    state = {
        "model_type": model.model_type,
        "config": config,
        "state_dict": model.state_dict(),
        "borders": torch.linspace(-4, 4, 33),
    }
    return model, state


def test_a_minipfn_checkpoint_becomes_a_folder_with_the_same_tensors(tmp_path):
    model, state = _minipfn_state()
    torch.save(state, tmp_path / "final.pt")
    digest = convert.main(["--checkpoint", str(tmp_path / "final.pt"), "--out-dir", str(tmp_path)])
    loaded, borders = hub.from_pretrained(tmp_path / "model")
    for name, tensor in model.state_dict().items():
        assert torch.equal(loaded.state_dict()[name], tensor), name
    assert torch.equal(borders, state["borders"])
    assert loaded.config.gate_hidden == model.config.gate_hidden  # renamed, not defaulted
    assert digest == weights_digest(model, state["borders"])


def test_missing_fixed_fields_take_minipfn_defaults_and_must_match():
    # minipfn left a field out at its default; for these the default is TabSil's value
    _, state = _minipfn_state()
    for name in ("logit_softcap", "column_spread_temperature", "decoder_qk_norm"):
        state["config"].pop(name)
    convert.convert(state)
    # a field whose minipfn default is not TabSil's value must be present
    _, state = _minipfn_state()
    state["config"].pop("qk_norm")
    with pytest.raises(SystemExit, match="qk_norm"):
        convert.convert(state)


def test_another_structure_and_a_tabsil_config_are_refused(tmp_path):
    _, state = _minipfn_state(encoder="linear")
    with pytest.raises(SystemExit, match="encoder"):
        convert.convert(state)
    model, state = _minipfn_state()
    state["config"] = config_dict(model.config)  # no marker: TabSil's own format
    with pytest.raises(SystemExit, match="not a minipfn config"):
        convert.convert(state)


def test_compare_needs_the_same_weights(tmp_path):
    model, state = _minipfn_state()
    torch.save(state, tmp_path / "final.pt")
    hub.save_pretrained(model, state["borders"], tmp_path / "same")
    convert.main(
        ["--checkpoint", str(tmp_path / "final.pt"), "--out-dir", str(tmp_path / "a")]
        + ["--compare", str(tmp_path / "same")]
    )
    hub.save_pretrained(tiny_model(seed=2), state["borders"], tmp_path / "other")
    with pytest.raises(SystemExit, match="differs"):
        convert.main(
            ["--checkpoint", str(tmp_path / "final.pt"), "--out-dir", str(tmp_path / "b")]
            + ["--compare", str(tmp_path / "other")]
        )
