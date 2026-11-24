"""`tabsil.recipe`: the structure and the inference defaults (the curriculum: training/tests)."""

from tabsil import recipe


def test_structure_is_the_published_model():
    from tabsil.models.column_row_transformer import ColumnRowTransformerConfig

    sizes = {k: v for k, v in recipe.STRUCTURE.items() if k != "model_type"}
    # the model config defaults are the published structure
    assert ColumnRowTransformerConfig(**sizes) == ColumnRowTransformerConfig()
    assert recipe.STRUCTURE["model_type"] == recipe.MODEL_TYPE == "column_row_transformer"


def test_default_inference_prefers_the_saved_settings():
    class Model:
        inference_config = None

    model = Model()
    for task in ("classification", "regression"):
        assert recipe.default_inference(model, task) == {
            "n_estimators": 8,
            "member_kinds": ("standard",),
            "max_features_per_member": 768,
            "feature_subsampling": "balanced",
        }
    model.inference_config = {"n_estimators": 4, "member_kinds": ["standard"]}
    assert recipe.default_inference(model, "classification") == {
        "n_estimators": 4,
        "member_kinds": ("standard",),
        "max_features_per_member": 768,
        "feature_subsampling": "balanced",
    }
    assert recipe.resolve_member_kinds(recipe.NAME) == ("standard",)
    assert recipe.SUPPORTED_SIZE == {"max_train_rows": 1_000_000, "max_features": 6_000}
    assert recipe.HUB_REPO == "aiden-jeon/tabsil"
