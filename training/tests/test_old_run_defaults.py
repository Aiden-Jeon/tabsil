"""The values that stand in for what an older training run did not record.

Three places read a training run's MLflow params and fill in a value for a param the run does not
have, because the run was made before the setting existed:

- `tabsil_train.hub.run_architecture`: the model structure, from the `TrainConfig` defaults
- `inherit_recipe_options` of the training script: `target_train_rows` and `optimizer`, from the
  `TrainConfig` defaults (runs made before these options existed, such as `tabsil-v2-s2-r10`,
  have neither param)
- `inherit_prior_type` of the training script: the prior, `DEFAULT_PRIOR_TYPE`

So these defaults are also how the older runs are read, and the values are written out here. A
test that fails here means a default changed: the runs that did not record the setting were
trained with the value below, and with the new default they would be resumed, continued or
evaluated as something else. Keep the old value for them (for example a fallback of its own in
the place that reads the run) before changing what new runs get. A new setting that is read this
way adds its line here: the value the runs before it were trained with.
"""

import os

import pytest
from train_helpers import load_script

from tabsil.models import DEFAULT_MODEL_TYPE
from tabsil_train.hub import run_architecture
from tabsil_train.priors import DEFAULT_PRIOR_TYPE
from tabsil_train.training.config import TrainConfig

# What a run without the param was trained with
OLD_RUN_STRUCTURE = {
    "model_type": "column_row_transformer",
    "emsize": 256,
    "nlayers": 24,
    "nhead": 16,
    "features_per_group": 3,
    "n_inducing": 256,
    "n_cls": 4,
    "column_blocks": 6,
    "row_blocks": 6,
    "column_heads": 4,
    "row_heads": 4,
}
OLD_RUN_RECIPE = {"target_train_rows": False, "optimizer": "muon"}
OLD_RUN_PRIOR = "graph_scm"
CHANGED = "a default that older runs are read with has changed (see this file's docstring)"


@pytest.fixture
def script(monkeypatch):
    # every argument of the script also reads an environment variable
    for name in [name for name in os.environ if name.startswith("TABSIL_")]:
        monkeypatch.delenv(name)
    return load_script("scripts/train.py", "train_script_old_runs")


def test_a_run_without_structure_params_has_the_v1_structure():
    assert run_architecture({}) == OLD_RUN_STRUCTURE, CHANGED
    assert DEFAULT_MODEL_TYPE == OLD_RUN_STRUCTURE["model_type"], CHANGED
    # a recorded value replaces only itself; the rest is still filled in
    assert run_architecture({"emsize": "128"}) == {**OLD_RUN_STRUCTURE, "emsize": 128}
    # the fill-in values are the config defaults, which is why a changed default shows up here
    defaults = TrainConfig()
    assert {name: getattr(defaults, name) for name in OLD_RUN_STRUCTURE} == OLD_RUN_STRUCTURE


def test_a_resumed_run_without_recipe_params_keeps_the_v1_recipe(script):
    args = script.parse_args(["--config-name", "stage1", "--resume-run-id", "abc"])
    script.inherit_recipe_options(args, {})
    assert {name: getattr(args, name) for name in script.RECIPE_OPTIONS} == OLD_RUN_RECIPE, CHANGED
    # turning on what such a run did not have is refused, as for a run that recorded it as off
    args = script.parse_args(
        ["--config-name", "stage1", "--resume-run-id", "abc", "--target-train-rows"]
    )
    with pytest.raises(SystemExit, match="target-train-rows is off"):
        script.inherit_recipe_options(args, {})
    args = script.parse_args(
        ["--config-name", "stage1", "--resume-run-id", "abc", "--optimizer", "muon_adamw"]
    )
    with pytest.raises(SystemExit, match="differs from the resumed run's muon$"):
        script.inherit_recipe_options(args, {})


def test_a_resumed_run_without_a_prior_param_stays_on_graph_scm(script):
    assert DEFAULT_PRIOR_TYPE == OLD_RUN_PRIOR, CHANGED
    args = script.parse_args(["--config-name", "stage1", "--resume-run-id", "abc"])
    script.inherit_prior_type(args, {})
    assert args.prior_type == OLD_RUN_PRIOR, CHANGED
    args = script.parse_args(
        ["--config-name", "stage1", "--resume-run-id", "abc", "--prior-type", "graph_scm_realism"]
    )
    with pytest.raises(SystemExit, match="resumed run's prior_type graph_scm$"):
        script.inherit_prior_type(args, {})
