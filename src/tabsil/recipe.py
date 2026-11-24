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
`standard` preprocessing members, each seeing at most 768 original columns spread `balanced` over
the members.

The non-size parts listed above (Fourier + ECDF encoder, gain-free q/k RMSNorm and the
temperatures, target LayerNorm, pre-head MLP, interleaved stages, retrieval decoder, quantile
head, one multitask model) are fixed in the code (`tabsil.models.column_row_transformer`); this
module holds the numbers.

    clf = TabSilClassifier().fit(x_train, y_train)  # downloads the published weights
    reg = TabSilRegressor(model, borders).fit(x_train, y_train)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from tabsil.inference.ensemble import EnsembleClassifier, EnsembleRegressor
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
# 8 `standard` members, each with at most 768 original columns spread evenly over the members
# (`balanced`). TabSil trains on tables with at most 100 features, and putting every column of a
# table with thousands of columns into one member hurts a lot
INFERENCE = {
    "n_estimators": {"classification": 8, "regression": 8},
    "member_kinds": ("standard",),
    "max_features_per_member": 768,
    "feature_subsampling": "balanced",
}
# Member kind presets selectable by name (`resolve_member_kinds`, e.g. `--member-kinds tabsil-v1`)
MEMBER_KIND_PRESETS = {NAME: INFERENCE["member_kinds"]}


def resolve_member_kinds(value: str | tuple[str, ...] | list[str]) -> tuple[str, ...]:
    """Member preprocessing kinds: a preset name (`MEMBER_KIND_PRESETS`), a comma separated string,
    or a sequence of kinds."""
    return _resolve_member_kinds(value, MEMBER_KIND_PRESETS)


def default_inference(model: nn.Module, task: str) -> dict:
    """The inference settings of `model` for `task`: `{"n_estimators", "member_kinds",
    "max_features_per_member", "feature_subsampling"}`, all arguments of `EnsembleClassifier` /
    `EnsembleRegressor`. Settings saved with the weights (`model.inference_config`, the `inference`
    entry of the Hub config.json) come first; missing keys use `INFERENCE`."""
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
    }


def _inference_kwargs(model: nn.Module, task: str, n_estimators, member_kinds) -> dict:
    # Arguments given by the caller come first, then the model's inference settings
    out = default_inference(model, task)
    if n_estimators is not None:
        out["n_estimators"] = n_estimators
    if member_kinds is not None:
        out["member_kinds"] = resolve_member_kinds(member_kinds)
    return out


# Published TabSil weights (the default of `TabSilClassifier()` and `TabSilRegressor()`). One
# multitask model serves both tasks
HUB_REPO = "aiden-jeon/tabsil"
HUB_REPOS = {"classification": HUB_REPO, "regression": HUB_REPO}


def _load(model, task: str, repos: dict[str, str]) -> tuple[nn.Module, torch.Tensor | None]:
    # Not a model: download from a Hub repo id or a local folder (None: the published weights)
    from tabsil.hub import model_or_pretrained

    return model_or_pretrained(model, repos[task], task)


class TabSilClassifier(EnsembleClassifier):
    """TabSil classification: `EnsembleClassifier` with TabSil's inference defaults. Without `model`
    it downloads the published weights (`HUB_REPO`). Other weights can be given as a Hub repo id
    (`owner/name`, `owner/name@branch`), a local folder, or a model. The member count and kinds
    default to the model's settings (`default_inference`); arguments override them.

        clf = TabSilClassifier().fit(x_train, y_train)
    """

    hub_repos = HUB_REPOS  # the weights downloaded when no model is given

    def __init__(
        self,
        model: nn.Module | str | Path | None = None,
        *,
        n_estimators: int | None = None,
        member_kinds: str | tuple[str, ...] | None = None,
        **kwargs: Any,
    ) -> None:
        if not isinstance(model, nn.Module):
            model, _ = _load(model, "classification", self.hub_repos)
        inference = _inference_kwargs(model, "classification", n_estimators, member_kinds)
        super().__init__(model, **{**inference, **kwargs})


class TabSilRegressor(EnsembleRegressor):
    """TabSil regression: `EnsembleRegressor` with TabSil's inference defaults. Without `model` it
    downloads the published weights and their regression borders (`HUB_REPO`). Other weights can
    be given as a Hub repo id, a local folder (borders from there too), or a model with `borders`.

        reg = TabSilRegressor().fit(x_train, y_train)
    """

    hub_repos = HUB_REPOS  # the weights downloaded when no model is given

    def __init__(
        self,
        model: nn.Module | str | Path | None = None,
        borders: torch.Tensor | None = None,
        *,
        n_estimators: int | None = None,
        member_kinds: str | tuple[str, ...] | None = None,
        **kwargs: Any,
    ) -> None:
        if isinstance(model, nn.Module):
            if borders is None:
                raise ValueError("pass the regression borders together with a model")
        else:
            if borders is not None:
                raise ValueError("borders are only given with a model (downloads include them)")
            model, borders = _load(model, "regression", self.hub_repos)
        inference = _inference_kwargs(model, "regression", n_estimators, member_kinds)
        super().__init__(model, borders, **{**inference, **kwargs})
