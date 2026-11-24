"""TabArena-Lite (51 datasets from TabArena v0.1, 38 classification and 13 regression, first split
of each dataset only).

TabArena (Erickson et al. 2025, arXiv 2506.16791) evaluates 51 OpenML datasets, each over several
splits. Lite uses only the first split (repeat 0, fold 0), the same split as the "TabArena-Lite"
table on the public leaderboard (Hugging Face Space `TabArena/leaderboard`). We compare against
the table that includes imputed methods (`imputation_yes`), because datasets a model cannot run
on are filled the same way, with RandomForest (default).

Evaluate with `tabsil_train.benchmarks.arena.run(SUITE, ...)`.
"""

from __future__ import annotations

from tabsil_train.benchmarks.arena import Suite

_PUBLIC = (
    "data/entrants_models/imputation_yes/splits_lite/tasks_{tasks}/datasets_all/"
    "website_leaderboard.csv"
)

SUITE = Suite(
    name="tabarena-lite",
    context="TabArenaContext",
    subset=("lite",),
    public_csv=_PUBLIC.format(tasks="all"),
    breakdown={
        task: (("lite", task), _PUBLIC.format(tasks=task))
        for task in ("classification", "regression", "binary", "multiclass")
    },
)
