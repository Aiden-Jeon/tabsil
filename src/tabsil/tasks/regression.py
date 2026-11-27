"""TabSil regression: `TabSilRegressor`, the regression ensemble
(`tabsil.inference.ensemble.EnsembleRegressor`) with TabSil's weights and inference defaults
(`tabsil.recipe`).

    from tabsil import TabSilRegressor

    reg = TabSilRegressor().fit(x_train, y_train)  # the default Hub weights
    mean = reg.predict(x_test)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from tabsil.inference.ensemble import EnsembleRegressor
from tabsil.recipe import HUB_REPOS, inference_kwargs


class TabSilRegressor(EnsembleRegressor):
    """TabSil regression: `EnsembleRegressor` with TabSil's inference defaults. Without `model` it
    downloads the weights and their regression borders from the default Hub repo (`HUB_REPO`) at
    the tag `HUB_REVISION`. Other weights can be given as a Hub repo id, a local folder (borders
    from there too), or a model with `borders`.

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
            # Not a model: download from a Hub repo id or a local folder (None: the default Hub
            # repo)
            from tabsil.hub import model_or_pretrained

            model, borders = model_or_pretrained(model, self.hub_repos["regression"], "regression")
        inference = inference_kwargs(model, "regression", n_estimators, member_kinds)
        super().__init__(model, borders, **{**inference, **kwargs})
