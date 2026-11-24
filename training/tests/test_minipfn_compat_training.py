"""minipfn MLflow run params and checkpoints with the training stack: `run_architecture` reads a
minipfn run's params as the TabSil structure, and training takes a minipfn checkpoint's weights
(`init_from`). Loading minipfn checkpoints and Hub folders is tested in
tests/test_minipfn_compat.py.
"""

from dataclasses import replace

import pytest
import torch

from tabsil.models import model_from_state
from tabsil_train import recipe
from tabsil_train.hub import run_architecture
from tabsil_train.training.config import TrainConfig
from tabsil_train.training.loop import check_checkpoint

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
# The structure params of a minipfn-v1 stage 1 training run, as MLflow recorded them (strings)
MINIPFN_STAGE1_PARAMS = {
    "task": "multitask",
    "model_type": "column_row_transformer",
    "emsize": "256",
    "nlayers": "24",
    "nhead": "16",
    "features_per_group": "3",
    "encoder": "fourier",
    "n_inducing": "256",
    "n_cls": "4",
    "qk_norm": "True",
    "logit_softcap": "None",
    "qk_norm_affine": "False",
    "column_summary_temperature": "logscale",
    "column_spread_temperature": "none",
    "row_temperature": "gated_logscale",
    "icl_temperature": "logscale",
    "decoder_temperature": "logscale",
    "column_blocks": "6",
    "row_blocks": "6",
    "column_heads": "4",
    "row_heads": "4",
    "decoder_qk_norm": "False",
    "stage_layout": "interleaved",
    "regression_head": "quantile",
    "categorical_encoder": "False",
    "target_norm": "True",
    "head_mlp": "True",
    "max_features": "100",
    "optimizer": "muon",
    "muon_scope": "all",
}


def test_minipfn_config_dict_is_the_smoke_training_config():
    from tabsil.models import config_from_dict

    assert config_from_dict("column_row_transformer", MINIPFN_SMOKE_CONFIG) == (
        TrainConfig.smoke().model_config()
    )


def test_minipfn_training_checkpoint_is_taken_by_init_from():
    torch.manual_seed(0)
    model = TrainConfig.smoke().build_model().eval()
    state = {
        "model_type": "column_row_transformer",
        "model_config": dict(MINIPFN_SMOKE_CONFIG),
        "model": model.state_dict(),
        "borders": torch.linspace(-3, 3, 11),
        "optimizer_type": "muon",
    }
    assert model_from_state(state)[0].config == model.config
    # init_from (stage 2 from a minipfn stage 1 run) accepts it
    check_checkpoint(state, TrainConfig.smoke(), "init_from")


def test_minipfn_run_params_give_the_tabsil_structure():
    source = run_architecture(dict(MINIPFN_STAGE1_PARAMS))
    assert source == recipe.STRUCTURE
    assert replace(recipe.curriculum(2), **source) == recipe.curriculum(2)
    for name, value in (("stage_layout", "sequential"), ("logit_softcap", "30.0")):
        with pytest.raises(ValueError, match=name):
            run_architecture({**MINIPFN_STAGE1_PARAMS, name: value})
    # a run from before the field existed had minipfn's default
    older = {k: v for k, v in MINIPFN_STAGE1_PARAMS.items() if k != "regression_head"}
    with pytest.raises(ValueError, match="regression_head"):
        run_architecture(older)
