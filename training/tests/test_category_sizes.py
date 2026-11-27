"""The category sizes the graph SCM gives (the docstrings of `graph_scm.generator`,
`transforms.dependence` and `graph_scm`, and the comments of `graph_scm.variants`, state them): at
most 9 categories by default, up to 200 with `corrected_cat_sizes`."""

import pytest

from tabsil_train.priors.graph_scm.generator import GraphSCM, _graph_scm_classes
from tabsil_train.priors.pipeline import Rng


def _sizes(monkeypatch, corrected: bool, n_tables: int = 40) -> tuple[set, list]:
    _graph_scm_classes()  # imports tabicl.prior (with a stand-in for xgboost if it is missing)
    import tabicl.prior._graph_scm as scm

    calls, sizes = set(), []
    real = scm.sample_categorical_sizes

    def spy(n_features, context, max_cat_size=100):
        calls.add(max_cat_size)
        out = real(n_features, context, max_cat_size=max_cat_size)
        sizes.extend(s for s in out if s)
        return out

    monkeypatch.setattr(scm, "sample_categorical_sizes", spy)
    generator = GraphSCM("regression", seed=0, corrected_cat_sizes=corrected)
    for i in range(n_tables):
        generator(200, 12, 1, {}, Rng.from_seed(i))
    return calls, sizes


@pytest.mark.parametrize("corrected", [False, True])
def test_category_sizes(monkeypatch, corrected):
    calls, sizes = _sizes(monkeypatch, corrected)
    # the graph SCM asks for up to 200 categories (a per-table limit drawn from 2-200, then each
    # size from 2 to that limit, both log-uniform, so sizes above 100 are rare); without the
    # correction the limit is at most 9
    assert calls == {200}
    assert sizes and min(sizes) >= 2
    if corrected:
        assert 9 < max(sizes) <= 200
    else:
        assert max(sizes) <= 9
