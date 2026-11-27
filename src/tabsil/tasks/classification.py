"""TabSil classification: `TabSilClassifier`, the classification ensemble
(`tabsil.inference.ensemble.EnsembleClassifier`) with TabSil's weights and inference defaults
(`tabsil.recipe`).

    from tabsil import TabSilClassifier

    clf = TabSilClassifier().fit(x_train, y_train)  # the default Hub weights
    proba = clf.predict_proba(x_test)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from torch import nn

from tabsil.inference.ensemble import EnsembleClassifier
from tabsil.recipe import HUB_REPOS, inference_kwargs


class TabSilClassifier(EnsembleClassifier):
    """TabSil classification: `EnsembleClassifier` with TabSil's inference defaults. Without `model`
    it downloads the default Hub repo (`HUB_REPO`) at the tag `HUB_REVISION`. Other weights can
    be given as a Hub repo id (`owner/name` is `main`, `owner/name@branch` a branch, tag or
    commit), a local folder, or a model. The member count and
    kinds default to the model's settings (`default_inference`); arguments override them.

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
            # Not a model: download from a Hub repo id or a local folder (None: the default Hub
            # repo)
            from tabsil.hub import model_or_pretrained

            repo = self.hub_repos["classification"]
            model, _ = model_or_pretrained(model, repo, "classification")
        inference = inference_kwargs(model, "classification", n_estimators, member_kinds)
        super().__init__(model, **{**inference, **kwargs})
