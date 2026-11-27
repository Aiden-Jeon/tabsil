"""TabSil: a tabular foundation model (classification and regression in one model).

This package is inference: loading the weights and predicting. `models` and `inference` are named
after the part they implement (generic nouns), TabSil's numbers are in `recipe`, and `tasks`
combines them into the estimators users call.

- `tasks`: the estimators, one module per task. Classification and regression are the two tasks
  the model itself outputs; the other three are built by calling those two
  - `classification`: `TabSilClassifier`
  - `regression`: `TabSilRegressor`
  - `forecast`: time series forecasting with the regression model (`TabSilForecaster`)
  - `unsupervised`: per-column conditional distributions, row log densities, outlier scores and
    imputation (`TabSilUnsupervised`)
  - `causal`: treatment effects with TabSil base learners (S/T/X/DR meta-learners, `TabSilCATE`)
- `inference`: the engine under the estimators (sklearn-style preprocessing member ensembles)
- `models`: the model and its registry (column-row transformer, bar and quantile distributions,
  checkpoint save and load)
- `recipe`: TabSil's numbers (the structure, the inference defaults, the default Hub repo and tag)
- `hub`: save to and load from the Hugging Face Hub format (`save_pretrained`, `from_pretrained`)
- `utils`: device and seed helpers

Training (priors, training loop, curriculum), evaluation (AMLB/CTR23, TabArena, BeyondArena,
uncertainty metrics) and the Hub upload scripts are in the `tabsil-train` package (`training/` in
the repository, imported as `tabsil_train`).

The five estimators and `from_pretrained` are importable from here and loaded on first use (so
`import tabsil` does not import the Hub and inference code).

    from tabsil import TabSilClassifier

    clf = TabSilClassifier("path/to/folder").fit(x_train, y_train)  # a save_pretrained folder
    # without a model: the default Hub repo (`tabsil.recipe.HUB_REPO`) at the tag `HUB_REVISION`
    # (the v1 weights)
"""

from typing import TYPE_CHECKING

# `import tabsil` binds `tabsil.models`, as tabsil 0.1.x did (`tabsil.models.load_checkpoint`).
# The other submodules are imported by name (`import tabsil.hub`)
from tabsil import models as models
from tabsil.utils import get_device, set_seed

if TYPE_CHECKING:
    from tabsil.hub import from_pretrained
    from tabsil.tasks.causal import TabSilCATE
    from tabsil.tasks.classification import TabSilClassifier
    from tabsil.tasks.forecast import TabSilForecaster
    from tabsil.tasks.regression import TabSilRegressor
    from tabsil.tasks.unsupervised import TabSilUnsupervised

# Names loaded on first use -> the module defining them
_LAZY = {
    "from_pretrained": "tabsil.hub",
    "TabSilClassifier": "tabsil.tasks.classification",
    "TabSilRegressor": "tabsil.tasks.regression",
    "TabSilForecaster": "tabsil.tasks.forecast",
    "TabSilUnsupervised": "tabsil.tasks.unsupervised",
    "TabSilCATE": "tabsil.tasks.causal",
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
    "from_pretrained",
    "TabSilClassifier",
    "TabSilRegressor",
    "TabSilForecaster",
    "TabSilUnsupervised",
    "TabSilCATE",
]
