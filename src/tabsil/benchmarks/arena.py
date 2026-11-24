"""Shared TabArena/BeyondArena evaluation: run an sklearn-style estimator on the official splits
and score it the same way as the public leaderboard.

Data, splits, metrics and aggregation come from the official TabArena package `tabarena` (extra
`arena`: `uv sync --no-dev --extra arena --group arena-dev`; it needs pandas 2.3, so it is
installed separately from the dev group). What we add is the wrapper that turns the estimator
into a system of that package (`ExternalSystemModel`, a method that does its own preprocessing
and validation), the rules for skipping datasets the model cannot handle, and the result format.

- Per-split metrics follow TabArena: ROC AUC for binary, log loss for multiclass, RMSE for
  regression; aggregation uses the error (`metric_error`, lower is better).
- Aggregation (`tabarena`'s `compare`): Elo (bootstrap 95% CI, reference method pinned to 1000:
  RandomForest (default) for TabArena, XGBoost (default) for BeyondArena), normalized score (best
  method per dataset 1, median method 0, negatives clipped to 0), improvability (mean % error
  above the best method), and rank. Per-split results of ~100 baseline methods are downloaded on
  first use into `~/.cache/tabarena`.
- Splits beyond the model limits (`ModelLimits`: train rows, features, classes, problem type) are
  not run. As on the leaderboard, they are filled with RandomForest (default) results (imputed)
  and the imputed fraction is reported.
- Difference from the public leaderboard: TabArena runs every method with 8-fold bagging inside
  the train split, while here the default is a single fit on the whole train split. With
  `bagging=k`, the train split is divided into k folds, k models are fit (each on the k-1 other
  folds) and their predictions are averaged.
- Text/datetime columns are not passed as-is: datetime columns become seconds, and non-numeric
  columns (including text) are passed as categorical (the estimator encodes them).

    from tabsil.benchmarks import arena, tabarena
    result = arena.run(tabarena.SUITE, make_estimator, limits=..., out_dir="out")

`make_estimator(problem_type, categorical_indices)` returns an estimator for `problem_type`
("binary", "multiclass", "regression"). Classifiers need `fit`, `predict_proba` and `classes_`;
regressors need `fit` and `predict`.
"""

from __future__ import annotations

import json
import math
import pickle
import sys
import urllib.parse
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROBLEM_TYPES = ("binary", "multiclass", "regression")
PUBLIC_LEADERBOARD = "https://huggingface.co/spaces/TabArena/leaderboard/resolve/main/"
# methods to compute win rates against {leaderboard display name: method name in `tabarena`
# results}. Result names get a "TA-" prefix or different casing per suite, so `_method_key`
# normalizes them. Methods without results are skipped
WINRATE_BASELINES = {
    "XGBoost (default)": "XGB (default)",
    "XGBoost (tuned + ensembled)": "XGB (tuned + ensemble)",
    "CatBoost (default)": "CAT (default)",
    "TabPFNv2 (default)": "TABPFNV2_GPU (default)",
    "RealTabPFN-2.5 (default)": "REALTABPFN-V2.5 (default)",
    "TabICLv2 (default)": "TABICLV2 (default)",
}


# temporal split: the time column is cut into this many contiguous intervals used as groups.
# Same as the default fold count of greedy selection (`ensemble_selection.N_FOLDS`). Missing
# times form their own group (-1), which may share a fold with another interval
TIME_INTERVALS = 8


def _time_values(t: pd.Series) -> pd.Series:
    """Time column as order-preserving numbers (missing/unparseable -> NaN). Ordered categoricals
    use category order, datetimes and date strings use timestamps, everything else is numeric."""
    import warnings

    if isinstance(t.dtype, pd.CategoricalDtype) and t.cat.ordered:
        return t.cat.codes.where(t.cat.codes >= 0).astype(float)
    if pd.api.types.is_bool_dtype(t) or pd.api.types.is_numeric_dtype(t):
        return pd.to_numeric(t, errors="coerce").astype(float)
    if not pd.api.types.is_datetime64_any_dtype(t):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # date format inference warning
            t = pd.to_datetime(t.astype(object), errors="coerce", utc=True)
    if getattr(t.dt, "tz", None) is not None:
        t = t.dt.tz_convert(None)
    return (t - pd.Timestamp(0)) / pd.Timedelta(seconds=1)


def split_groups(X: pd.DataFrame, metadata, n_intervals: int = TIME_INTERVALS) -> np.ndarray | None:
    """Per-row group ids from the task's split info. Grouped split: the group column (the value
    combination if several columns). Temporal split: contiguous intervals of the time column with
    similar row counts (equal times share an interval, missing times are -1). Neither (random
    split): None. Uses the same columns AutoGluon uses to build bagging folds on TabArena
    (`resolve_validation_splits`). If a column is missing or the time column cannot yield at
    least 2 intervals, warns and returns None (random folds)."""
    group_on = getattr(metadata, "group_on", None)
    time_on = getattr(metadata, "time_on", None)
    named = [*(group_on if isinstance(group_on, list | tuple) else [group_on]), time_on]
    missing = [c for c in named if c is not None and c not in X.columns]
    if missing:  # column absent from train rows: random folds (do not fail the split)
        print(
            f"split_groups: split columns {missing} not in train rows, using random folds",
            file=sys.stderr,
        )
        return None
    if group_on is not None:
        columns = list(group_on) if isinstance(group_on, list | tuple) else [group_on]
        codes = np.stack([pd.factorize(X[c], use_na_sentinel=False)[0] for c in columns], axis=1)
        return np.unique(codes, axis=0, return_inverse=True)[1].reshape(-1)
    if time_on is not None:
        ranks = _time_values(X[time_on]).rank(method="min")
        codes = pd.qcut(ranks, q=n_intervals, labels=False, duplicates="drop")
        if codes.nunique() < 2:
            print(
                f"split_groups: time column {time_on!r} ({X[time_on].dtype}) gives fewer than 2"
                " intervals, using random folds",
                file=sys.stderr,
            )
            return None
        return codes.fillna(-1).astype(int).to_numpy()
    return None


def _method_key(method: str) -> str:
    return str(method).removeprefix("TA-").upper()


@dataclass(frozen=True)
class Suite:
    """One benchmark: `tabarena` context, splits to evaluate, matching public leaderboard results.

    `subset` is the context's subset expression (e.g. `["lite"]`, `["core2k"]`), `public_csv` is
    a path under `PUBLIC_LEADERBOARD`. `breakdown` gives per-subset scores
    {name: (subset expression, public CSV)}.
    """

    name: str
    context: str  # "TabArenaContext" | "BeyondArenaContext"
    subset: tuple[str, ...]
    public_csv: str
    breakdown: dict[str, tuple[tuple[str, ...], str]] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelLimits:
    """Upper bounds on the splits a model can run. Splits beyond them are imputed. None means no
    limit."""

    max_train_rows: int | None = None
    max_features: int | None = None
    max_classes: int | None = None
    problem_types: tuple[str, ...] = PROBLEM_TYPES

    def skip_reason(self, row: pd.Series) -> str | None:
        """Why one task-grid row (one split) is not run; None if it is run."""
        if row["problem_type"] not in self.problem_types:
            return f"problem_type {row['problem_type']}"
        if self.max_train_rows is not None and row["max_train_rows"] > self.max_train_rows:
            return f"train rows {int(row['max_train_rows'])} > {self.max_train_rows}"
        if self.max_features is not None and row["n_features"] > self.max_features:
            return f"features {int(row['n_features'])} > {self.max_features}"
        n_classes = row["n_classes"]
        if self.max_classes is not None and n_classes > self.max_classes:
            return f"classes {int(n_classes)} > {self.max_classes}"
        return None


def make_context(suite: Suite):
    try:
        from tabarena import contexts
    except ImportError as e:  # pragma: no cover - only without the extra
        raise ImportError(
            "TabArena/BeyondArena evaluation needs the tabarena package: "
            "uv sync --no-dev --extra arena --group arena-dev"
        ) from e
    return getattr(contexts, suite.context)(backend="native")


Limits = ModelLimits | dict[str, ModelLimits]


def skip_reason(limits: Limits, row: pd.Series) -> str | None:
    """`limits` is one `ModelLimits` or one per problem type ("binary", ...).
    Problem types without an entry are not run."""
    if isinstance(limits, ModelLimits):
        return limits.skip_reason(row)
    if row["problem_type"] not in limits:
        return f"problem_type {row['problem_type']}"
    return limits[row["problem_type"]].skip_reason(row)


def task_grid(context, suite: Suite, limits: Limits) -> pd.DataFrame:
    """The suite's splits (dataset, fold, repeat, sizes) and why each is skipped (`skip_reason`)."""
    grid = context.task_metadata_collection.task_grid()
    grid = grid[_subset_mask(context, grid, suite.subset)].reset_index(drop=True)
    grid["skip_reason"] = [skip_reason(limits, row) for _, row in grid.iterrows()]
    return grid


def _subset_mask(context, grid: pd.DataFrame, subset: tuple[str, ...]) -> pd.Series:
    # each subset expression (AND-ed) maps to a context predicate; "!name" negates it
    mask = pd.Series(True, index=grid.index)
    for expr in subset:
        negate = expr.startswith("!")
        m = context.subset_predicates[expr.lstrip("!")](grid)
        mask &= ~m if negate else m
    return mask


def run(
    suite: Suite,
    make_estimator: Callable[[str, list[int]], Any],
    *,
    limits: Limits,
    out_dir: str | Path,
    method: str = "tabsil",
    bagging: int = 0,
    datasets: list[str] | None = None,
    breakdown: bool = True,
    fingerprint: str = "",
    results: list | None = None,
    results_path: str | Path | None = None,
) -> dict:
    """Run the suite's splits and score them. Writes results to `out_dir` and returns a summary.

    Split results are cached per split under `out_dir/runs/<id>`; calling again with the same
    `out_dir` skips finished splits. `<id>` is derived from `fingerprint` (whatever determines the
    estimator: checkpoint, inference settings, etc.; supplied by the caller), `method` and
    `bagging`, so results from a different checkpoint are never reused. With `datasets`, only
    those datasets are run (the rest are imputed). If no split was run or all failed, scores are
    left empty (None) and the other files are still written.

    Files written:
    - `tasks.csv`: whether each split was run (`skip_reason`; "failed" for splits that ran but
      failed)
    - `results_per_split.csv`: this method's split results (`metric_error` etc., incl. imputed)
    - `leaderboard.csv`: leaderboard with baselines (`tabarena` format, all methods)
    - `leaderboard_<name>.csv`: per-subset leaderboards (`breakdown`)
    - `comparison.csv`: public leaderboard (same splits) and this run's scores side by side per
      method
    - `summary.json`: this method's scores, imputed fraction, win rates, train/inference time of
      the splits actually run, and settings (if the public CSV cannot be fetched, the public
      columns of `comparison.csv` are empty)
    - `compare.log`: output of the `tabarena` aggregation

    When splitting across GPUs (`dataset_queue`, the eval script's `--shards`):
    - With `results_path`, run the splits, pickle the results (the list of dicts from `run_jobs`)
      to that file and skip scoring (returns an empty dict). Used by workers that take datasets
      one at a time from the queue. The file is renamed into place after writing, so if it
      exists it is complete
    - With `results`, do not run splits; only score those results (the merged worker results).
      `run_jobs` materializes task data first even when cached, so results are passed instead of
      re-reading the cache
    """
    from tabsil.benchmarks import _arena_system as system

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    context = make_context(suite)
    grid = task_grid(context, suite, limits)
    if datasets is not None:
        unknown = sorted(set(datasets) - set(grid["dataset"]))
        if unknown:
            raise ValueError(f"datasets not in {suite.name}: {unknown}")
        grid.loc[~grid["dataset"].isin(datasets), "skip_reason"] = "not selected"
    grid.to_csv(out_dir / "tasks.csv", index=False)
    todo = grid[grid["skip_reason"].isna()]
    print(
        f"{suite.name}: splits {len(grid)} ({grid['dataset'].nunique()} datasets), "
        f"run {len(todo)}, imputed {len(grid) - len(todo)}",
        flush=True,
    )

    run_id = _run_id(fingerprint, method, bagging)
    run_dir = out_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "fingerprint.txt").write_text(f"{method}\nbagging={bagging}\n{fingerprint}\n")

    if results is not None:
        results = list(results)
    elif len(todo):
        key = system.register_factory(make_estimator)
        experiments = system.experiments(suite, method, key, bagging)
        jobs = context.build_jobs(
            experiments,
            subset=list(suite.subset),
            dataset_names=sorted(todo["dataset"].unique()),
        )
        wanted = set(zip(todo["dataset"], todo["fold"], todo["repeat"], strict=True))
        jobs = [j for j in jobs if _job_split(j) in wanted]
        prefetch_tasks(context, jobs, what=suite.name)
        # do not stop if one split fails: it has no result and becomes imputed (`failed`)
        results = context.run_jobs(
            jobs,
            expname=str(run_dir),
            register=False,
            debug_mode=True,
            raise_on_failure=False,
        )
    else:
        results = []
    if results_path is not None:
        # rename after writing: the eval script's `--shards` treats the file's existence as done
        results_path = Path(results_path)
        partial = results_path.with_name(results_path.name + ".partial")
        with open(partial, "wb") as f:
            pickle.dump(results, f)
        partial.replace(results_path)
        return {}
    if results:
        context.register(results, scope_to_valid_tasks=False)
    pool = model_pool(context, method)

    log = out_dir / "compare.log"
    leaderboard, per_split = _quiet(
        context.compare,
        log,
        output_dir=None,
        subset=list(suite.subset),
        ta_results=pool,
        return_results=True,
    )
    name = _new_method_name(leaderboard, method)
    if name is None:
        print(f"{method}: no split has results, skipping scores", file=sys.stderr)
    fillna = context.fillna_method
    leaderboard.to_csv(out_dir / "leaderboard.csv", index=False)
    ours = with_imputed(per_split, name or method, fillna)
    ours.drop(columns=["method_metadata"], errors="ignore").to_csv(
        out_dir / "results_per_split.csv", index=False
    )
    # splits that ran but have no result (the fold in results is the split index)
    filled = ours[ours["imputed"]]
    imputed = set(zip(filled["dataset"], filled["fold"], strict=True))
    was_imputed = pd.Series(
        [(d, s) in imputed for d, s in zip(grid["dataset"], grid["split"], strict=True)],
        index=grid.index,
    )
    failed = grid["skip_reason"].isna() & was_imputed
    grid.loc[failed, "skip_reason"] = "failed"
    grid.to_csv(out_dir / "tasks.csv", index=False)
    if failed.any():
        print(f"{int(failed.sum())} failed splits (imputed): tasks.csv", file=sys.stderr)
    website = context.leaderboard_to_website_format(leaderboard=leaderboard)
    comparison = compare_with_public(website, try_public_leaderboard(suite.public_csv), name)
    comparison.to_csv(out_dir / "comparison.csv", index=False)

    summary = {
        "suite": suite.name,
        "method": name or method,
        "run_id": run_id,
        "bagging": bagging,
        "limits": _limits_dict(limits),
        "splits": len(grid),
        "datasets": int(grid["dataset"].nunique()),
        "run_splits": len(todo),
        "failed_splits": int(failed.sum()),
        "imputed_pct": 100.0 * float(ours["imputed"].mean()),
        "skip_reasons": _reason_counts(grid["skip_reason"]),
        **(_scores(leaderboard, name) if name else dict.fromkeys(SCORE_KEYS)),
        "winrate": winrates(per_split, ours),
        "time": run_times(ours),
        "breakdown": {},
    }
    if breakdown and name:
        for label, (subset, public_csv) in suite.breakdown.items():
            part = _quiet(_breakdown, log, context, pool, subset, name, public_csv)
            if part is None:
                continue
            board, scores = part
            board.to_csv(out_dir / f"leaderboard_{label}.csv", index=False)
            summary["breakdown"][label] = scores
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default)
    )
    return summary


def dataset_queue(grid: pd.DataFrame, subsample_rows: int | None = None) -> list[tuple[str, int]]:
    """Datasets with splits to run (no `skip_reason`), most expensive first, as (dataset, number
    of splits). A dataset's cost is the sum of its splits' cells (train rows x features). Sending
    expensive ones first leaves small work at the end, so GPUs that finish early do not idle for
    long. Ties are broken by name, so the same grid always gives the same order. With
    `subsample_rows` (`--subsample-train-rows`), train rows are capped at that value (the rows fit
    actually sees)."""
    todo = grid[grid["skip_reason"].isna()]
    rows = todo["max_train_rows"].astype(float)
    if subsample_rows is not None:
        rows = rows.clip(upper=float(subsample_rows))
    cells = rows * todo["n_features"].clip(lower=1).astype(float)
    cost = cells.groupby(todo["dataset"]).sum()
    splits = todo.groupby("dataset").size()
    order = sorted(cost.index, key=lambda d: (-cost[d], d))
    return [(d, int(splits[d])) for d in order]


def split_rows(context, subsample_rows: int | None = None) -> dict[str, int]:
    """Per dataset, the largest train + test row count over its splits (task metadata
    `num_instances_train`/`num_instances_test`; the grid has only train rows). This is the row
    count one inference forward sees. With `subsample_rows` (`--subsample-train-rows`), train rows
    are capped at that value. Empty dict if the metadata has no row counts."""
    frame = context.task_metadata_collection.to_dataframe()
    needed = {"tabarena_task_name", "num_instances_train", "num_instances_test"}
    if frame.empty or not needed <= set(frame.columns):
        return {}
    train = frame["num_instances_train"].astype(float)
    if subsample_rows is not None:
        train = train.clip(upper=float(subsample_rows))
    rows = train + frame["num_instances_test"].astype(float)
    return {d: int(n) for d, n in rows.groupby(frame["tabarena_task_name"]).max().items()}


def _run_id(fingerprint: str, method: str, bagging: int) -> str:
    import hashlib

    text = json.dumps([method, bagging, fingerprint])
    return hashlib.sha1(text.encode()).hexdigest()[:12]


def _reason_counts(reasons: pd.Series) -> dict:
    """Count splits by reason kind ("train rows 11640 > 10000" -> "train rows > 10000")."""
    kinds = reasons.fillna("run").str.replace(r" \d+ > ", " > ", regex=True)
    return kinds.value_counts().to_dict()


def _limits_dict(limits: Limits) -> dict:
    if isinstance(limits, ModelLimits):
        return asdict(limits)
    return {k: asdict(v) for k, v in limits.items()}


def _quiet(fn, log: Path, *args, **kwargs):
    # tabarena's aggregation prints the whole leaderboard to stdout; redirect it to a log file
    import contextlib

    with open(log, "a") as f, contextlib.redirect_stdout(f):
        return fn(*args, **kwargs)


def model_pool(context, method: str) -> pd.DataFrame:
    """Split results of the methods to compare: models only, as in the public leaderboard's default
    table (systems like AutoGluon excluded), plus this run's method (names starting with
    `method`)."""
    from tabarena.evaluation.entrants import DEFAULT_ENTRANT_POOL, filter_results_to_pool

    results = context.load_results()
    ours = results["method"].astype(str).str.startswith(method)
    pool = filter_results_to_pool(
        results, DEFAULT_ENTRANT_POOL, context.method_metadata_collection.info()
    )
    return pd.concat([pool[~pool.index.isin(results.index[ours])], results[ours]])


def _job_split(job) -> tuple[str, int, int]:
    task = job.task if hasattr(job, "task") else job
    return (task.dataset, int(task.fold), int(task.repeat))


def prefetch_tasks(context, jobs: list, *, what: str) -> None:
    """Download task data for `jobs` before running splits, retrying per task on network errors
    (`benchmarks._retry`).

    `run_jobs` downloads data per task while running splits (OpenML tasks), and errors at that
    stage are not caught by `raise_on_failure=False`, so the whole evaluation stops. With data
    prefetched, `run_jobs` only reads the cache. `make_context` passes no `cache_config`, so both
    use the default openml cache.
    - OpenML tasks (TabArena): task, split and dataset are fetched per task (`fetch_openml_task`).
    - Data Foundry tasks (BeyondArena, non-integer ids): fetched by `materialize`. Already
      downloaded datasets are skipped, so calling again fetches only what is left."""
    from tabsil.benchmarks._retry import fetch_openml_task, retry

    collection = context.task_metadata_collection.subset_to_jobs(jobs)
    task_ids: list[int] = []
    for ttm in collection:
        try:
            task_ids.append(int(ttm.task_id_str))
        except (TypeError, ValueError):
            continue
    for task_id in dict.fromkeys(task_ids):
        fetch_openml_task(task_id)
    retry(collection.materialize, what=f"{what} materialize")


def _new_method_name(leaderboard: pd.DataFrame, method: str) -> str | None:
    # leaderboard name of the registered method (system config name appended); None if no results
    names = [m for m in leaderboard["method"] if str(m).startswith(method)]
    if len(names) > 1:
        raise RuntimeError(f"several leaderboard methods start with {method}: {names}")
    return names[0] if names else None


SCORE_KEYS = (
    "elo",
    "elo_ci_plus",
    "elo_ci_minus",
    "score",
    "improvability",
    "rank",
    "harmonic_rank",
    "imputed_pct_leaderboard",
    "n_methods",
    "position",
)


def _scores(leaderboard: pd.DataFrame, name: str) -> dict:
    """This method's scores from the leaderboard, in public leaderboard units: score is
    1 - normalized error, improvability and imputed are %, harmonic rank is 1 / mrr."""
    row = leaderboard[leaderboard["method"] == name].iloc[0]
    return {
        "elo": float(row["elo"]),
        "elo_ci_plus": float(row["elo+"]),
        "elo_ci_minus": float(row["elo-"]),
        "score": 1.0 - float(row["normalized-error"]),
        "improvability": 100.0 * float(row["improvability"]),
        "rank": float(row["rank"]),
        "harmonic_rank": 1.0 / float(row["mrr"]),
        "imputed_pct_leaderboard": 100.0 * float(row["imputed"]),
        "n_methods": int(len(leaderboard)),
        "position": int((leaderboard["elo"] > row["elo"]).sum()) + 1,
    }


def with_imputed(per_split: pd.DataFrame, name: str, fillna: str) -> pd.DataFrame:
    """This method's split results with unrun splits filled from the `fillna` method
    (RandomForest (default)) (`imputed=True`), as the leaderboard does internally. The scope is
    all splits in `per_split`."""
    ours = per_split[per_split["method"] == name].copy()
    ours["imputed"] = False
    filler = per_split[per_split["method"] == fillna]
    done = set(zip(ours["dataset"], ours["fold"], strict=True))
    missing = filler[[k not in done for k in zip(filler["dataset"], filler["fold"], strict=True)]]
    missing = missing.assign(method=name, imputed=True, impute_method=fillna)
    out = pd.concat([ours, missing], ignore_index=True)
    return out.sort_values(["dataset", "fold"]).reset_index(drop=True)


TIME_COLUMNS = ("time_train_s", "time_infer_s")


def run_times(ours: pd.DataFrame) -> dict:
    """Mean and total train/inference time (s) of splits actually run (not imputed). `ours` is the
    output of `with_imputed`. Empty dict if there are no time columns."""
    run = ours[~ours["imputed"].astype(bool)]
    out = {}
    for col in TIME_COLUMNS:
        if col in run and len(run):
            out[f"{col}_mean"] = float(run[col].mean())
            out[f"{col}_total"] = float(run[col].sum())
    return out


def winrates(per_split: pd.DataFrame, ours: pd.DataFrame) -> dict:
    """Win rate against each of `WINRATE_BASELINES`, counting 1 per split with lower error and 0.5
    for ties. `ours` is the output of `with_imputed`. `all` includes imputed splits (same scope as
    the leaderboard); `run` covers only splits actually run."""
    ours = ours.set_index(["dataset", "fold"])
    out = {}
    keys = per_split["method"].map(_method_key)
    for baseline, internal in WINRATE_BASELINES.items():
        other = per_split[keys == _method_key(internal)]
        if not len(other):
            continue
        other = other.set_index(["dataset", "fold"])["metric_error"]
        both = ours.join(other.rename("other"), how="inner")
        win = np.where(
            both["metric_error"] < both["other"],
            1.0,
            np.where(both["metric_error"] == both["other"], 0.5, 0.0),
        )
        imputed = both["imputed"].astype(bool).to_numpy()
        out[baseline] = {
            "all": float(win.mean()) if len(win) else math.nan,
            "run": float(win[~imputed].mean()) if (~imputed).any() else math.nan,
            "splits": int(len(win)),
            "run_splits": int((~imputed).sum()),
        }
    return out


def _breakdown(context, pool, subset: tuple[str, ...], name: str, public_csv: str | None):
    try:
        board, per_split = context.compare(
            output_dir=None, subset=list(subset), ta_results=pool, return_results=True
        )
    except Exception as e:  # aggregation can fail when the subset has too few datasets
        print(f"subset {subset}: aggregation failed ({e})", file=sys.stderr, flush=True)
        return None
    if name not in set(board["method"]):
        return None
    ours = with_imputed(per_split, name, context.fillna_method)
    scores = {
        **_scores(board, name),
        "datasets": int(per_split["dataset"].nunique()),
        "splits": int(len(ours)),
        "imputed_pct": 100.0 * float(ours["imputed"].mean()),
        "winrate": winrates(per_split, ours),
        "time": run_times(ours),
    }
    if public_csv:
        website = context.leaderboard_to_website_format(leaderboard=board)
        board = compare_with_public(website, try_public_leaderboard(public_csv), name)
    return board, scores


def load_public_leaderboard(path: str) -> pd.DataFrame:
    """Public leaderboard CSV (`path` in the Hugging Face Space `TabArena/leaderboard`)."""
    url = PUBLIC_LEADERBOARD + urllib.parse.quote(path)
    df = pd.read_csv(url)
    df["name"] = df["Model"].map(display_name)
    return df


def try_public_leaderboard(path: str) -> pd.DataFrame | None:
    """`load_public_leaderboard`, or None if it cannot be fetched (network, missing combination),
    so hours of results are not lost over one public CSV."""
    try:
        return load_public_leaderboard(path)
    except Exception as e:
        print(f"could not fetch public leaderboard {path}: {e}", file=sys.stderr, flush=True)
        return None


def display_name(model: str) -> str:
    """Just the name from the leaderboard Model cell (`[name (default)](link) [x% IMPUTED]`)."""
    model = str(model)
    if model.startswith("[") and "](" in model:
        model = model[1 : model.index("](")]
    return model.split(" [")[0].strip()


def compare_with_public(
    website: pd.DataFrame, public: pd.DataFrame | None, name: str | None
) -> pd.DataFrame:
    """Per method, public leaderboard Elo/scores next to the values recomputed in this run.
    This run's method (`name`) has empty public values. If `public` is None, all public columns
    are empty. Sorted by Elo."""
    cols = {
        "Elo [⬆️]": "elo",
        "Elo 95% CI": "elo_ci",
        "Score [⬆️]": "score",
        "Improvability (%) [⬇️]": "improvability",
        "Imputed (%) [⬇️]": "imputed_pct",
    }
    ours = website.copy()
    ours["name"] = ours["Model"].map(display_name)
    if name is not None:
        ours.loc[ours["Model"].astype(str).str.contains(name, regex=False), "name"] = name
    ours = ours[["name", *[c for c in cols if c in ours]]].rename(columns=cols)
    if public is None:
        public = pd.DataFrame({"name": pd.Series(dtype=str), **{c: [] for c in cols}})
    theirs = public[["name", *[c for c in cols if c in public]]].rename(columns=cols)
    merged = ours.merge(theirs, on="name", how="outer", suffixes=("", "_public"))
    merged["is_new"] = merged["name"] == name
    return merged.sort_values("elo", ascending=False, na_position="last").reset_index(drop=True)


def _json_default(value: Any):
    if isinstance(value, np.integer | np.floating):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    raise TypeError(f"value not JSON serializable: {value!r}")
