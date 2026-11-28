"""TabArena-Lite (51 datasets from TabArena v0.1, 38 classification and 13 regression, first split
of each dataset only), and the same 51 datasets with their first 3 splits (`SUITE_3SPLIT`).

TabArena (Erickson et al. 2025, arXiv 2506.16791) evaluates 51 OpenML datasets, each over several
splits. Lite uses only the first split (repeat 0, fold 0), the same split as the "TabArena-Lite"
table on the public leaderboard (Hugging Face Space `TabArena/leaderboard`). We compare against
the table that includes imputed methods (`imputation_yes`), because datasets a model cannot run
on are filled the same way, with RandomForest (default).

`SUITE_3SPLIT` ("tabarena-3split", 153 splits) keeps splits 0-2 of each dataset: repeat 0, the
three folds of TabArena's 3-fold cross-validation, so every training row of a dataset is a test
row once. It is the v2 experiment plan's evaluation set (`training/docs/v2-experiment-plan.md`
§3). Every baseline is scored on the same 153 splits (`Suite.splits`). The public leaderboard has
no table for these splits; `comparison.csv` shows the all-splits table next to them for reference.

Evaluate with `tabsil_train.benchmarks.tabular.arena.run(SUITE, ...)`.
"""

from __future__ import annotations

from tabsil_train.benchmarks.tabular.arena import Suite

_PUBLIC = (
    "data/entrants_models/imputation_yes/splits_lite/tasks_{tasks}/datasets_all/"
    "website_leaderboard.csv"
)
_PUBLIC_ALL = _PUBLIC.replace("splits_lite", "splits_all")

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

SUITE_3SPLIT = Suite(
    name="tabarena-3split",
    context="TabArenaContext",
    subset=(),
    public_csv=_PUBLIC_ALL.format(tasks="all"),
    breakdown={
        task: ((task,), _PUBLIC_ALL.format(tasks=task))
        for task in ("classification", "regression", "binary", "multiclass")
    },
    splits=3,
)
