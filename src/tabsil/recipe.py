"""TabSil recipe: the model structure and the inference defaults.

TabSil (architecture name `tabsil-v1`) is a tabular foundation model: one `column_row_transformer`
serves both classification and regression, trained only on synthetic tables from the TabICLv2
graph SCM prior (the `tabicl` package). The training curriculum and the prior are in the
`tabsil-train` package (`tabsil_train.recipe`, `tabsil_train.priors`).

Structure (`STRUCTURE`):

- backbone from the TabPFN-3.5 report: row in-context learning with 24 layers x 16 heads at width
  1024 (4 CLS x 256), Fourier + ECDF cell encoder, q/k RMSNorm, target LayerNorm, an MLP before
  the output layers, and a many-class retrieval decoder for classification
- from Kumo-Tabular (blog post and NVIDIA structured-data-models): 6 interleaved column/row stage
  pairs (cell width 256, 4 heads, 256 inducing points, 4 CLS tokens), Kumo attention temperatures
  (`logscale` / `gated_logscale`) with gain-free q/k RMSNorm and no logit soft-cap, and a
  regression head with 999 quantiles

Inference (`INFERENCE`, `default_inference`): `EnsembleClassifier` / `EnsembleRegressor` with 8
preprocessing members that cycle through 4 numeric transforms (`standard`, `power`, `robust`,
`quantile`), each seeing at most 768 original columns spread `balanced` over the members.
Categorical columns with more than 20 categories also get a log count column
(`categories="counts"`), and the classification softmax has temperature 1.0 (tabsil 0.1.1; 0.1.0
used `standard` members only, no count columns and temperature 0.9).

The non-size parts listed above (Fourier + ECDF encoder, gain-free q/k RMSNorm and the
temperatures, target LayerNorm, pre-head MLP, interleaved stages, retrieval decoder, quantile
head, one multitask model) are fixed in the code (`tabsil.models.column_row_transformer`); this
module holds the numbers.

    clf = TabSilClassifier().fit(x_train, y_train)  # `HUB_REPO` at the tag `HUB_REVISION`
    reg = TabSilRegressor(model, borders).fit(x_train, y_train)
"""

from __future__ import annotations

from torch import nn

from tabsil.inference.ensemble import resolve_member_kinds as _resolve_member_kinds

NAME = "tabsil-v1"
MODEL_TYPE = "column_row_transformer"

# ---------------------------------------------------------------------------
# Model structure
# ---------------------------------------------------------------------------
STRUCTURE = {
    "model_type": MODEL_TYPE,
    # TabPFN-3.5 backbone: 3 cells per token, row ICL 24 layers x 16 heads
    "features_per_group": 3,
    "nlayers": 24,
    "nhead": 16,
    # Kumo-Tabular (large) column/row stages: 6 interleaved pairs, cell width 256, 4 heads, 256
    # inducing points, 4 CLS tokens -> row vector 4 x 256 = 1024
    "emsize": 256,
    "column_blocks": 6,
    "row_blocks": 6,
    "column_heads": 4,
    "row_heads": 4,
    "n_inducing": 256,
    "n_cls": 4,
}
# Largest table TabSil is evaluated on (TabArena and BeyondArena skip larger splits). Same as the
# TabPFN-3.5 recommended limits, since the row ICL and everything after it are the same
SUPPORTED_SIZE = {"max_train_rows": 1_000_000, "max_features": 6_000}


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
# Kumo-Tabular's round-robin numeric transforms (`tabsil.inference.preprocessing`)
NUMERIC_CYCLE = ("standard", "power", "robust", "quantile")
# 8 members cycling through the numeric transforms, each with at most 768 original columns spread
# evenly over the members (`balanced`): TabSil trains on tables with at most 100 features, and
# putting every column of a table with thousands of columns into one member hurts a lot. A log
# count column for categorical columns with more than 20 categories, and softmax temperature 1.0
# (classification). TabArena-Lite (the repository's harness, v1 weights): Elo 1726 with 0.1.0's
# settings (`standard` members, codes only, temperature 0.9) and 1768 with these
INFERENCE = {
    "n_estimators": {"classification": 8, "regression": 8},
    "member_kinds": NUMERIC_CYCLE,
    "max_features_per_member": 768,
    "feature_subsampling": "balanced",
    "categories": "counts",
    "softmax_temperature": 1.0,
}
# Member kind presets selectable by name (`resolve_member_kinds`, e.g. `--member-kinds tabsil-v1`)
MEMBER_KIND_PRESETS = {NAME: INFERENCE["member_kinds"], "numeric-cycle": NUMERIC_CYCLE}


def resolve_member_kinds(value: str | tuple[str, ...] | list[str]) -> tuple[str, ...]:
    """Member preprocessing kinds: a preset name (`MEMBER_KIND_PRESETS`), a comma separated string,
    or a sequence of kinds."""
    return _resolve_member_kinds(value, MEMBER_KIND_PRESETS)


def default_inference(model: nn.Module, task: str) -> dict:
    """The inference settings of `model` for `task`: `{"n_estimators", "member_kinds",
    "max_features_per_member", "feature_subsampling", "categories", "softmax_temperature"}`, all
    arguments of `EnsembleClassifier` / `EnsembleRegressor` (regression ignores the temperature).
    Settings saved with the weights (`model.inference_config`, the `inference` entry of the Hub
    config.json) come first; missing keys use `INFERENCE`."""
    saved = getattr(model, "inference_config", None) or {}
    return {
        "n_estimators": int(saved.get("n_estimators", INFERENCE["n_estimators"][task])),
        "member_kinds": resolve_member_kinds(
            saved.get("member_kinds") or INFERENCE["member_kinds"]
        ),
        "max_features_per_member": saved.get(
            "max_features_per_member", INFERENCE["max_features_per_member"]
        ),
        "feature_subsampling": saved.get("feature_subsampling", INFERENCE["feature_subsampling"]),
        "categories": saved.get("categories", INFERENCE["categories"]),
        "softmax_temperature": float(
            saved.get("softmax_temperature", INFERENCE["softmax_temperature"])
        ),
    }


def inference_kwargs(
    model: nn.Module,
    task: str,
    n_estimators: int | None,
    member_kinds: str | tuple[str, ...] | list[str] | None,
) -> dict:
    """The ensemble arguments of `TabSilClassifier` / `TabSilRegressor` (`tabsil.tasks`):
    `n_estimators` and `member_kinds` given by the caller (not None) come first, then the model's
    inference settings (`default_inference`)."""
    out = default_inference(model, task)
    if n_estimators is not None:
        out["n_estimators"] = n_estimators
    if member_kinds is not None:
        out["member_kinds"] = resolve_member_kinds(member_kinds)
    return out


# The default Hub repo of `TabSilClassifier()` and `TabSilRegressor()` and the revision they
# download: the Hub tag `v1.1`, the v1 weights with tabsil 0.1.1's inference settings in config.json
# (`INFERENCE`; tabsil 0.1.0 downloads the tag `v1`, the same weights with its own settings), so a
# tabsil release keeps downloading the same weights and settings when `main` moves. One multitask
# model serves both tasks. A repo id passed by the caller keeps its own revision
# (`aiden-jeon/tabsil` is `main`, `aiden-jeon/tabsil@stage2-10k` a branch)
HUB_REPO = "aiden-jeon/tabsil"
HUB_REVISION = "v1.1"
HUB_DEFAULT = f"{HUB_REPO}@{HUB_REVISION}"
HUB_REPOS = {"classification": HUB_DEFAULT, "regression": HUB_DEFAULT}
