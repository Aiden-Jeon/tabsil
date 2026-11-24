"""Graph SCM generator: tables from the TabICLv2 graph SCM (soda-inria/tabicl, BSD-3-Clause).

A computation graph over a random DAG produces the features and the target, and already includes
categorical features, discretization, Kumaraswamy warping and regression targets. So y is the
class index for classification and a z-normalized value for regression (no transform to classes
is needed). The prior that adds the table size rule and missing values is
`tabsil_train.priors.graph_scm`.

TabICL uses the global numpy and torch random state (not `Rng`), so creating this generator seeds
them. With the same seed, a different Python string hash seed (`PYTHONHASHSEED`) still gives
different datasets.

`filter_unpredictable` (the TabICLv2 training scripts' `--filter_unpredictable_graphs True
--filter_unpredictable_datasets True`): graphs whose x nodes and y node share no ancestor are
redrawn (graph filter, inside the generator), and a finished dataset is dropped when ExtraTrees
OOB predictions are no better than the mean (bootstrap p ≥ 0.05; dataset filter,
`tabicl.prior._dataset.should_filter`). Like TabICLv2's graph prior
(`tabicl.prior._dataset.GraphPrior`), datasets are redrawn until a predictable one comes out
(there `while True`; here the next one is accepted after `FILTER_ATTEMPTS` tries so that it cannot
hang, and the mlp/tree prior `SCMPrior` only filters the first 6). Per the paper, about 35% of
classification and 25% of regression datasets are filtered in stage 1, and training converges
better. Off by default.
"""

from __future__ import annotations

import sys
import types
from typing import Literal

import numpy as np
import torch

from tabsil_train.priors.pipeline import Rng

# Most dataset-filter attempts for one dataset (TabICL's `GraphPrior` has no limit; this only
# prevents hanging). `SCMPrior` stops after 6 (`idx <= 5`)
FILTER_ATTEMPTS = 1000


def _graph_scm_classes():
    """Import TabICL's GraphSCM and PriorConfig.

    The `tabicl.prior` package imports xgboost at module level for its tree SCM prior. The graph
    prior does not use xgboost, so if xgboost cannot be imported a stand-in module is installed
    that raises only when used (on macOS xgboost needs a separate libomp). TabSil does not use the
    tree SCM prior.
    """
    try:
        import xgboost  # noqa: F401
    except Exception:
        stub = types.ModuleType("xgboost")

        class XGBRegressor:
            def __init__(self, *args, **kwargs):
                raise ImportError("tabicl's tree SCM prior needs xgboost (TabSil does not use it)")

        stub.XGBRegressor = XGBRegressor
        sys.modules["xgboost"] = stub
    from tabicl.prior._dataset import should_filter
    from tabicl.prior._graph_scm import GraphSCM as TabICLGraphSCM
    from tabicl.prior.graph_lib._config import PriorConfig

    return TabICLGraphSCM, PriorConfig, should_filter


class GraphSCM:
    """Graph SCM generator (`Generator`). Classification uses the `num_classes` hyperparameter.

    Datasets are built one at a time. When TabICL discards a dataset that produced NaN, y is all
    -100 (`tabsil_train.priors.transforms.filters.DropInvalid` then has the group redrawn).
    """

    def __init__(
        self,
        task: Literal["classification", "regression"],
        seed: int = 0,
        *,
        filter_unpredictable: bool = False,
    ) -> None:
        self.regression = task == "regression"
        self.filter_unpredictable = filter_unpredictable
        np.random.seed(seed)
        torch.manual_seed(seed)
        self._graph_scm, prior_config, self._should_filter = _graph_scm_classes()
        self._config = prior_config(
            filter_unpredictable_graphs=filter_unpredictable,
            filter_unpredictable_datasets=filter_unpredictable,
        )

    def __call__(
        self, n_rows: int, n_features: int, n_datasets: int, hp: dict, rng: Rng
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return stack_datasets(self._dataset, n_rows, n_features, n_datasets, hp)

    def _dataset(self, n_rows: int, n_features: int, hp: dict) -> tuple[torch.Tensor, ...]:
        make = self._graph_scm(
            regression=self.regression,
            seq_len=n_rows,
            num_features=n_features,
            max_features=n_features,
            num_classes=None if self.regression else hp["num_classes"],
            config=self._config,
        )
        return filtered(make, self, regression=self.regression)


def stack_datasets(
    dataset, n_rows: int, n_features: int, n_datasets: int, hp: dict
) -> tuple[torch.Tensor, ...]:
    """Stack n_datasets datasets, each built by `dataset(n_rows, n_features, hp)`, into a group:
    x (rows, datasets, features), y (rows, datasets)."""
    outs = [dataset(n_rows, n_features, hp) for _ in range(n_datasets)]
    return torch.stack([o[0] for o in outs], 1), torch.stack([o[1] for o in outs], 1)


def filtered(
    make, generator, *, regression: bool, attempts: int | None = None
) -> tuple[torch.Tensor, ...]:
    """Build one dataset with `make()`. With `generator.filter_unpredictable`, rebuild until it is
    predictable (after `attempts` filtered ones, default `FILTER_ATTEMPTS`, the next is accepted).
    Datasets discarded for NaN (y all -100) are returned unfiltered so that DropInvalid redraws
    them. `generator` is a generator with `filter_unpredictable`, `_should_filter` and `_config`."""
    out = make()
    if not generator.filter_unpredictable:
        return out
    for _ in range(FILTER_ATTEMPTS if attempts is None else attempts):
        x, y = out[0], out[1]
        if (y == -100).all() or not generator._should_filter(
            x, y, generator._config, is_classif=not regression
        ):
            break
        out = make()
    return out
