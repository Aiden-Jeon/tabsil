"""TabSil: a tabular foundation model (classification and regression in one model).

This package is inference: loading the weights and predicting. Modules on an axis are named after
the part they implement (generic nouns); TabSil's numbers and the combination of the parts are in
`recipe`.

- `models`: the model and its registry (column-row transformer, bar and quantile distributions,
  checkpoint save and load)
- `inference`: sklearn-style inference (preprocessing member ensembles, test-time methods)
- `hub`: save to and load from the Hugging Face Hub format (`save_pretrained`, `from_pretrained`)
- `recipe`: TabSil's structure and inference defaults (`TabSilClassifier`, `TabSilRegressor`)
- `utils`: device and seed helpers

Training (priors, training loop, curriculum), evaluation (AMLB/CTR23, TabArena, BeyondArena,
uncertainty metrics) and the Hub upload scripts are in the `tabsil-train` package (`training/` in
the repository, imported as `tabsil_train`).

The inference entry points are importable from here and loaded on first use (so `import tabsil`
does not import the Hub and inference code).

    from tabsil import TabSilClassifier

    clf = TabSilClassifier("path/to/folder").fit(x_train, y_train)  # a save_pretrained folder
    # without a model: the default Hub repo (`tabsil.recipe.HUB_REPO`, the v1 weights)
"""

from typing import TYPE_CHECKING

from tabsil.models.bar_distribution import (
    BarDistribution,
    FullSupportBarDistribution,
    get_bucket_borders,
)
from tabsil.utils import get_device, set_seed

if TYPE_CHECKING:
    from tabsil.hub import from_pretrained
    from tabsil.recipe import TabSilClassifier, TabSilRegressor

# Names loaded on first use -> the module defining them
_LAZY = {
    "from_pretrained": "tabsil.hub",
    "TabSilClassifier": "tabsil.recipe",
    "TabSilRegressor": "tabsil.recipe",
}


def __getattr__(name: str):
    if name in _LAZY:
        import importlib

        value = getattr(importlib.import_module(_LAZY[name]), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY))


__all__ = [
    "get_device",
    "set_seed",
    "BarDistribution",
    "FullSupportBarDistribution",
    "get_bucket_borders",
    "from_pretrained",
    "TabSilClassifier",
    "TabSilRegressor",
]
