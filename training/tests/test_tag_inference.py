"""`training/hub/tag_inference.py`: the `inference` entry a release tags onto the published weights
(no Hub access: only the config it writes)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "hub"))
import tag_inference  # noqa: E402

from tabsil import recipe  # noqa: E402


def test_tagged_inference_is_the_recipe_and_reads_back_as_the_defaults():
    inference = tag_inference.recipe_inference()
    assert inference == {
        "n_estimators": 8,
        "member_kinds": list(recipe.NUMERIC_CYCLE),
        "max_features_per_member": 768,
        "feature_subsampling": "balanced",
        "categories": "counts",
        "softmax_temperature": 1.0,
    }
    v1 = {"task": "multitask", "model_type": "column_row_transformer", "model": {"emsize": 256}}
    v1["inference"] = {"n_estimators": 8, "member_kinds": ["standard"]}
    new = tag_inference.tagged_config(v1, inference)
    assert {k: v for k, v in new.items() if k != "inference"} == {
        k: v for k, v in v1.items() if k != "inference"
    }

    class Saved:  # weights loaded from the tagged config.json
        inference_config = new["inference"]

    for task in ("classification", "regression"):
        got = recipe.default_inference(Saved(), task)
        assert got == {**inference, "member_kinds": recipe.NUMERIC_CYCLE}


def test_default_tag_is_the_release_revision():
    args = tag_inference.parse_args([])
    assert (args.repo_id, args.source, args.tag) == (recipe.HUB_REPO, "v1", recipe.HUB_REVISION)
