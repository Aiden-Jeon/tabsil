"""BeyondArena (142 datasets, 98 classification and 44 regression; arXiv 2606.30410 "Beyond IID").

TabArena is mostly IID splits, so it rarely exposes where foundation models are weak. BeyondArena
collects datasets that vary in split type (IID, temporal, grouped), size (tiny to large) and
feature kind (low-dim, high-dim, text, high-cardinality). The data is the Hugging Face dataset
`TabArena/BeyondArena` (curation tool `TabArena/data-foundry`), downloaded by the `tabarena`
package.

The default split range is the official BeyondArena protocol `core2k` (2,000 splits, allocated
per dataset by cost). The public leaderboard (`data_beyondarena/` in Space `TabArena/leaderboard`)
uses the same range. `lite` (first split of each dataset) is for quick checks.

Per-subset scores (`SUITE.breakdown`): split type (random, temporal, grouped), size, feature kind.
Elo is computed separately per subset (as on the public leaderboard, each subset is one
evaluation).
"""

from __future__ import annotations

from tabsil_train.benchmarks.tabular.arena import Suite

_PUBLIC = (
    "data_beyondarena/subsets/split_{split}/size_{size}/features_{features}/tasks_all/"
    "website_leaderboard.csv"
)


def _public(split: str = "all", size: str = "all", features: str = "all") -> str:
    return _PUBLIC.format(split=split, size=size, features=features)


BREAKDOWN = {
    **{f"split_{s}": ((s,), _public(split=s)) for s in ("random", "temporal", "grouped")},
    **{f"size_{s}": ((s,), _public(size=s)) for s in ("tiny", "small", "medium", "large")},
    **{
        f"features_{f}": ((f,), _public(features=f))
        for f in ("low-dim", "high-dim", "text", "high-cardinality")
    },
}


def suite(splits: str = "core2k") -> Suite:
    """`splits`: "core2k" (official, default) or "lite" (first split of each dataset)."""
    base = ("core2k",) if splits == "core2k" else ("core2k", "lite")
    return Suite(
        name="beyondarena" if splits == "core2k" else "beyondarena-lite",
        context="BeyondArenaContext",
        subset=base,
        public_csv=_public(),
        breakdown={k: ((*base, *s), csv) for k, (s, csv) in BREAKDOWN.items()},
    )


SUITE = suite()
