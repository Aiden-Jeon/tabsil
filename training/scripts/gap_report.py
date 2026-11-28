"""Gap of TabSil to the top of the public TabArena-Lite leaderboard, per dataset, and the Elo of
candidate changes on the live pool (`tabsil_train.benchmarks.tabular.paired` reads the runs).

Three parts, all on split 0 of the 51 Lite datasets (38 classification, 13 regression):

1. Per dataset: the TabSil error of each run, the error of six reference entrants (Causilo 1.5,
   Kumo-Tabular (L), LimiX-2-1016, Seldon, Kumo-Tabular-Medium, TabPFN-3.5), the error ratio
   TabSil / reference, TabSil's rank among those seven, and the dataset properties. Summaries: G
   (the geometric mean of the ratios) with a bootstrap 95% CI, by task and by property bucket,
   and the datasets where TabSil loses most.
2. Elo on the live pool. The per-dataset errors of the live board are read from its pages
   database. The Elo method is TabArena's (`bencheval`, Bradley-Terry maximum likelihood on
   per-dataset wins, ties as half a win each, RandomForest (default) calibrated to 1000). The
   script reproduces the published Lite Elo of every entrant, inserts the TabSil runs as new
   entrants and runs the what-ifs: one dataset set to the Kumo-Tabular-Medium or best top-5
   error, the uniform or per-task error reduction that reaches rank 5, and matching
   Kumo-Tabular-Medium on a whole task.
3. Writes CSV files and one markdown summary to `--out-dir`.

Inputs and sources:
- TabSil runs: MLflow eval artifact `eval/tabarena-lite/results_per_split.csv` of each run
  (experiment 634945255951077; default runs 7cc31c48 (#14) and 138151a2 (v1.1)). Read with
  `paired.load_splits`, which refuses imputed or missing splits.
- The live board: Hugging Face Space `TabArena/leaderboard`, current `main` revision (its commit
  sha is recorded in the summary). Files: `data/pages.sqlite` (the per-dataset explorer page of
  `entrants_models/imputation_yes/splits_lite`) and that pool's `website_leaderboard.csv` (the
  published Elo). The decoder reads the database tables directly: `page` maps a page path to a
  list of `chunk` ids, a chunk is a byte range of an lzma `stream`, and the page text is the
  concatenation. The explorer's data are JavaScript constants `const CONFIG = {...};` and
  `const POINTS = [...];` (dataset index `d`, method index `m`, error `e`). The layout was read
  from the tables and the page text and checked against the published CSV (the 114 entrants match
  by name, after `ensembled` -> `ensemble`, and the reproduced Elo agrees with the published Elo).
  A sibling decoder on branch `dev-v2-codex-2` was read for the layout too; nothing is imported
  from it.
- Dataset properties: TabArena's task metadata from the `tabarena` package source on GitHub
  (`autogluon/tabarena`, commit 8e55fb9fd20f0ba453a7f4318c763911ec13d019):
  `TabArena-v0.1_tasks_metadata.csv` (split r0f0: train rows, features, classes, minority share)
  and `curated_tabarena_dataset_metadata.csv`
  (OpenML dataset id). Categorical count, declared levels and missing cells of the predictors
  come from the public OpenML REST API feature list (`/api/v1/json/data/features/<id>`).

The downloads go to `--cache-dir`, which must be a new directory outside the repository. With
`--offline` nothing is downloaded and every file must already be in the cache.

From the repository root (MLflow needs the Databricks profile):
  DATABRICKS_CONFIG_PROFILE=e2-demo-field-eng MLFLOW_TRACKING_URI=databricks \\
    uv run --frozen --package tabsil-train python training/scripts/gap_report.py \\
    --cache-dir /tmp/top5-gap-data --out-dir /tmp/top5-gap-out
  ... --offline                           (after one online run; no network needed)
  ... --run "TabSil #14=<run id>"         (repeat to replace the default two runs)
  ... --space-revision <sha>              (the board at a fixed commit; v2-top5-plan.md uses
                                           a525fe89b5824325055ad58ff3de5590047a2877)

The Elo of the published board is shown next to the reproduction and decides nothing.
"""

from __future__ import annotations

import argparse
import csv
import json
import lzma
import math
import re
import shutil
import sqlite3
import urllib.request
from contextlib import closing
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from tabsil_train.benchmarks.tabular import paired

SPACE = "TabArena/leaderboard"
SPACE_API = f"https://huggingface.co/api/spaces/{SPACE}"
SPACE_FILE = f"https://huggingface.co/spaces/{SPACE}/resolve/{{sha}}/{{path}}"
POOL = "entrants_models/imputation_yes/splits_lite"
PAGE_PATH = f"{POOL}/tasks_all/datasets_all/per_dataset_explorer.html"
PUBLISHED_PATH = f"data/{POOL}/tasks_all/datasets_all/website_leaderboard.csv"

TABARENA_SHA = "8e55fb9fd20f0ba453a7f4318c763911ec13d019"
TABARENA_RAW = (
    "https://raw.githubusercontent.com/autogluon/tabarena/{sha}/packages/tabarena/src/"
    "tabarena/benchmark/task/metadata/{path}"
)
TASK_METADATA = "sources/data/TabArena-v0.1_tasks_metadata.csv"
CURATED_METADATA = "data/curated_tabarena_dataset_metadata.csv"
OPENML_FEATURES = "https://www.openml.org/api/v1/json/data/features/{did}"
ARTIFACT = "eval/tabarena-lite/results_per_split.csv"
EXPERIMENT_NOTE = "experiment 634945255951077 on profile e2-demo-field-eng"

DEFAULT_RUNS = (
    ("TabSil #14", "7cc31c4845c442079d998a1c02839533"),
    ("TabSil v1.1", "138151a232db44af8a45be80b967c265"),
)
PUBLISHED_TABSIL = "TabSil (default)"
CALIBRATION = "RandomForest (default)"
CALIBRATION_ELO = 1000
REFERENCES = {
    "Causilo 1.5": "Causilo 1.5 (default)",
    "Kumo-Tabular (L)": "Kumo-Tabular (default)",
    "LimiX-2-1016": "LimiX-2-1016 (default)",
    "Seldon": "Seldon (default)",
    "Kumo-Tabular-Medium": "Kumo-Tabular-Medium (default)",
    "TabPFN-3.5": "TabPFN-3.5 (default)",
}
KUMO_MEDIUM = "Kumo-Tabular-Medium"
TARGET_RANK = 5
BOOTSTRAP_ROUNDS = 2000
BOOTSTRAP_SEED = 0
# bencheval's ridge on the log-strengths (negligible on a real field; pins down separable ones)
BT_RIDGE = 0.5 / (1e6 * math.log(10) ** 2)
GROUPS = {
    "all": ("binary", "multiclass", "regression"),
    "classification": ("binary", "multiclass"),
    "regression": ("regression",),
    "binary": ("binary",),
    "multiclass": ("multiclass",),
}
# Planning what-ifs (v2-top5-plan.md §4). A change to the primary run's errors is priced in Elo
# on the live pool in four ways:
# - a cut of every error by a share x with per-dataset noise: log ratio = -x + sd * N(0, 1), so
#   G is exp(-x); sd 0.01 is the spread of two stage-2 seeds on TabArena (v2-findings §4)
NOISE_SDS = (0.0, 0.01, 0.02)
NOISE_CUTS = (0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02)
NOISE_DRAWS = 10
NOISE_SEED = 0
# - another entrant's change, dataset by dataset, applied to the primary run (an analogy: the run
#   moves as that entrant moved). (before, after, label)
TRANSPLANTS = (
    ("Kumo-Tabular-Small (default)", "Kumo-Tabular-Medium (default)", "Kumo-Tabular S -> M"),
    ("Kumo-Tabular-Medium (default)", "Kumo-Tabular (default)", "Kumo-Tabular M -> L"),
    ("Kumo-Tabular-Small (default)", "Kumo-Tabular (default)", "Kumo-Tabular S -> L"),
    ("TabPFN-3.5-Fast (default)", "TabPFN-3.5 (default)", "TabPFN-3.5-Fast -> TabPFN-3.5"),
    ("Seldon-Mini (default)", "Seldon (default)", "Seldon-Mini -> Seldon"),
    ("Lattice-tiny (default)", "Lattice (default)", "Lattice-tiny -> Lattice"),
    ("TabPFN-3 (default)", "TabPFN-3.5 (default)", "TabPFN-3 -> TabPFN-3.5"),
    ("Causilo (default)", "Causilo 1.5 (default)", "Causilo -> Causilo 1.5"),
    ("RealTabPFN-2.5 (default)", "RealTabPFN-2.5 (tuned)", "RealTabPFN-2.5 default -> tuned"),
    (
        "RealTabPFN-2.5 (default)",
        "RealTabPFN-2.5 (tuned + ensemble)",
        "RealTabPFN-2.5 default -> tuned + ensemble",
    ),
    ("TabPFNv2 (default)", "TabPFNv2 (tuned + ensemble)", "TabPFNv2 default -> tuned + ensemble"),
)
# - a share of a Kumo-Tabular step (its per-dataset log ratios times the share), for a compute
#   multiplier m over the run's recipe: share = log m / log (the step's compute ratio), the power
#   law in compute (v2-task-formulations.md §11.5). The multipliers are v2-top5-plan.md's, from the
#   costs of its §9 at today's prices against #14's recipe (150K stage-1 and 10K stage-2 steps,
#   about 822 H100-h): 2.1 (250K stage-1 steps, the same stage 2) 1.49; one doubling 2; tier 3
#   without stage 3 (500K / 40K steps) 3.17 to 3.29. (label, before, after, compute ratio)
KUMO_STEPS = (
    ("Kumo-Tabular S -> M", "Kumo-Tabular-Small (default)", "Kumo-Tabular-Medium (default)", 4.57),
    ("Kumo-Tabular M -> L", "Kumo-Tabular-Medium (default)", "Kumo-Tabular (default)", 3.22),
)
COMPUTE_MULTIPLIERS = (
    ("2.1", 1.49),
    ("one doubling", 2.0),
    ("tier 3 low", 3.17),
    ("tier 3 high", 3.29),
)
# - on the datasets of a property bucket where the run is behind Kumo-Tabular-Medium, a share of
#   the log gap closed (1: Kumo-Tabular-Medium's error), or the best top-5 error
CLOSE_FRACTIONS = (0.25, 0.5, 1.0)
# - the change from the second run to the primary one, split by datasets: the second run with the
#   primary run's errors on these datasets only (the four that #14's prior was made for)
NAMED_DATASETS = (
    "polish_companies_bankruptcy",
    "anneal",
    "kddcup09_appetency",
    "Amazon_employee_access",
)


# ---------------------------------------------------------------- downloads and caches


def _get(url: str, dest: Path) -> Path:
    """Download `url` to `dest` (a new file in the cache; a generic client name, no user data)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "tabsil-gap-report"})
    with urllib.request.urlopen(request, timeout=120) as reply, open(tmp, "wb") as out:
        shutil.copyfileobj(reply, out)
    tmp.replace(dest)
    return dest


def cached(url: str, dest: Path, offline: bool) -> Path:
    """`dest` from the cache, downloaded from `url` first unless `offline`."""
    if dest.exists():
        return dest
    if offline:
        raise SystemExit(f"{dest} is not in the cache and --offline is set")
    return _get(url, dest)


def space_revision(cache: Path, offline: bool) -> str:
    """The commit sha of the Space's current `main`, recorded in the summary. Offline, the sha
    of the last online run is used."""
    api = cache / "space" / "api.json"
    if not offline:
        _get(SPACE_API, api)
    return json.loads(api.read_text())["sha"]


# ---------------------------------------------------------------- the live board


def _page_text(db_path: Path, page: str) -> str:
    """One page of the official page pack (format 1; see the module docstring)."""
    uri = db_path.resolve().as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as db:
        if db.execute("select value from meta where key='format'").fetchone() != ("1",):
            raise ValueError(f"{db_path}: unknown page-pack format")
        row = db.execute("select chunks from page where path=?", (page,)).fetchone()
        if row is None:
            raise KeyError(f"{db_path}: no page {page}")
        ids = [int(i) for i in row[0].split()]
        chunks = {
            i: db.execute("select stream, start, length from chunk where id=?", (i,)).fetchone()
            for i in set(ids)
        }
        streams = {}
        for stream in {c[0] for c in chunks.values()}:
            data = db.execute("select data from stream where id=?", (stream,)).fetchone()[0]
            streams[stream] = lzma.decompress(data)
    pieces = [streams[chunks[i][0]][chunks[i][1] : chunks[i][1] + chunks[i][2]] for i in ids]
    return b"".join(pieces).decode("utf-8")


def _js_constant(text: str, name: str):
    match = re.search(rf"^  const {name} = (.+);$", text, re.MULTILINE)
    if match is None:
        raise ValueError(f"the explorer page has no `const {name}`")
    return json.loads(match[1])


def _short_name(cell: str) -> str:
    """The entrant name of a published-leaderboard cell (`[Name](url) [35% IMPUTED]`)."""
    match = re.match(r"^\[(.+?)\]\(", cell)
    name = match.group(1) if match else cell.split(" [")[0]
    return name.replace("ensembled", "ensemble")


def load_board(cache: Path, sha: str, offline: bool):
    """(errors: entrants x datasets, NaN where the board has none; datasets: per-dataset info;
    published: Elo, imputed % and rank of each entrant, in board order)."""
    db_path = cached(
        SPACE_FILE.format(sha=sha, path="data/pages.sqlite"),
        cache / "space" / sha / "pages.sqlite",
        offline,
    )
    csv_path = cached(
        SPACE_FILE.format(sha=sha, path=PUBLISHED_PATH),
        cache / "space" / sha / "website_leaderboard.csv",
        offline,
    )
    text = _page_text(db_path, PAGE_PATH)
    config = _js_constant(text, "CONFIG")
    points = _js_constant(text, "POINTS")
    datasets = [d["name"] for d in config["datasets"]]
    methods = [m["name"] for m in config["methods"]]
    errors = pd.DataFrame(np.nan, index=methods, columns=datasets)
    for p in points:
        errors.iat[p["m"], p["d"]] = p["e"]
    with open(csv_path, encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    published = pd.DataFrame(
        {
            "name": [_short_name(r["Model"]) for r in rows],
            "elo": [float(r["Elo [⬆️]"]) for r in rows],
            "imputed_pct": [float(r["Imputed (%) [⬇️]"]) for r in rows],
        }
    ).set_index("name")
    missing = set(methods) ^ set(published.index)
    if missing:
        raise ValueError(f"entrants differ between the page and the CSV: {sorted(missing)}")
    info = pd.DataFrame(
        {
            "task": [d["task"] for d in config["datasets"]],
            "metric": [d["metric"] for d in config["datasets"]],
        },
        index=datasets,
    )
    return errors, info, published


def fill_missing(errors: pd.DataFrame) -> pd.DataFrame:
    """TabArena's `imputation_yes` pool: a method without a result on a dataset gets the
    RandomForest (default) error on that dataset (`fillna` of the evaluator). The imputed
    cells of the board are the ones this changes."""
    filled = errors.copy()
    for dataset in filled.columns:
        filled[dataset] = filled[dataset].fillna(filled.at[CALIBRATION, dataset])
    return filled


# ---------------------------------------------------------------- the TabSil runs


def load_run(cache: Path, run_id: str, offline: bool) -> pd.DataFrame:
    """A run's split-0 errors: columns dataset, error, metric, problem_type (one row each)."""
    local = cache / "mlflow" / run_id / "results_per_split.csv"
    if not offline and not local.exists():
        import mlflow

        downloaded = mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path=ARTIFACT, dst_path=str(cache / "mlflow" / run_id)
        )
        local = Path(downloaded)
    if not local.exists():
        raise SystemExit(f"{local} is not in the cache and --offline is set")
    # load_splits refuses imputed, missing or duplicate splits
    checked = paired.load_splits(local)
    if set(checked["split"]) != {0} or checked["dataset"].duplicated().any():
        raise SystemExit(f"{run_id}: the Lite table must hold one split 0 per dataset")
    raw = pd.read_csv(local)
    return pd.DataFrame(
        {
            "dataset": raw["dataset"],
            "error": raw["metric_error"],
            "metric": raw["metric"],
            "problem_type": raw["problem_type"],
        }
    ).set_index("dataset")


# ---------------------------------------------------------------- dataset properties


def _openml_features(cache: Path, did: int, offline: bool) -> list[dict]:
    """The OpenML feature list of a dataset (`/data/features/<id>`): name, type, target flag,
    declared levels of a nominal feature and its missing-value count."""
    path = cached(
        OPENML_FEATURES.format(did=did), cache / "openml" / f"features-{did}.json", offline
    )
    return json.loads(path.read_text())["data_features"]["feature"]


def load_properties(cache: Path, sha_metadata: str, offline: bool) -> pd.DataFrame:
    """One row per dataset (index: dataset name) with the properties of split 0.

    Rows, features, classes and the minority share come from TabArena's split r0f0 metadata.
    The predictor columns (the target excluded) come from OpenML's feature list: a nominal
    predictor's cardinality is the number of declared levels (the observed count can be lower;
    OpenML's own nominal statistics are missing for many datasets), and missing cells are the
    per-feature counts."""
    task_csv = cached(
        TABARENA_RAW.format(sha=sha_metadata, path=TASK_METADATA),
        cache / "tabarena" / sha_metadata / "tasks.csv",
        offline,
    )
    curated_csv = cached(
        TABARENA_RAW.format(sha=sha_metadata, path=CURATED_METADATA),
        cache / "tabarena" / sha_metadata / "curated.csv",
        offline,
    )
    tasks = pd.read_csv(task_csv)
    tasks = tasks[tasks["split_index"] == "r0f0"].set_index("dataset_name")
    curated = pd.read_csv(curated_csv).set_index("dataset_name")
    rows = []
    for name in tasks.index:
        t = tasks.loc[name]
        classification = t["problem_type"] != "regression"
        did = int(curated.loc[name, "dataset_id"])
        predictors = [
            f
            for f in _openml_features(cache, did, offline)
            if str(f.get("is_target")).lower() != "true"
            and str(f.get("is_ignore")).lower() != "true"
            and str(f.get("is_row_identifier")).lower() != "true"
        ]
        nominal = [f for f in predictors if f["data_type"] == "nominal"]
        levels = [len(f.get("nominal_value") or []) for f in nominal]
        missing_cells = sum(int(f.get("number_of_missing_values") or 0) for f in predictors)
        features = int(t["num_features"])
        if len(predictors) != features:
            raise ValueError(f"{name}: {len(predictors)} OpenML predictors, {features} in TabArena")
        rows.append(
            {
                "dataset": name,
                "dataset_id": did,
                "task": t["problem_type"],
                "train_rows": round(float(t["num_instances_train"])),
                "features": features,
                "cat_features": len(nominal),
                "cat_share": len(nominal) / features,
                "max_cardinality": max(levels) if levels else 0,
                "mean_cardinality": float(np.mean(levels)) if levels else 0.0,
                "missing_any": missing_cells > 0,
                "missing_cell_share": missing_cells / (float(t["num_instances"]) * features),
                "classes": int(t["num_classes"]) if classification else np.nan,
                "minority_share": float(t["target_minority_fraction"])
                if classification
                else np.nan,
            }
        )
    return pd.DataFrame(rows).set_index("dataset")


# ---------------------------------------------------------------- ratios, G and buckets


def geo_mean(values) -> float:
    values = np.asarray(values, dtype=float)
    return float(np.exp(np.mean(np.log(values))))


def bootstrap_ci(ratios, rounds: int = BOOTSTRAP_ROUNDS, seed: int = BOOTSTRAP_SEED):
    """95% bootstrap interval of G, resampling datasets (2,000 rounds, seed 0)."""
    logs = np.log(np.asarray(ratios, dtype=float))
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(logs), size=(rounds, len(logs)))
    means = np.exp(logs[picks].mean(axis=1))
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def summarize_group(ratios: pd.Series) -> dict:
    """G, its CI and the wins and losses of a group of datasets."""
    low, high = bootstrap_ci(ratios.to_numpy())
    return {
        "n": len(ratios),
        "G": geo_mean(ratios),
        "ci_low": low,
        "ci_high": high,
        "better": int((ratios < 1).sum()),
        "worse": int((ratios > 1).sum()),
    }


def top5_geo_error(errors: pd.DataFrame, top5: list[str]) -> pd.Series:
    """Per dataset, the geometric mean of the error of the board's top five (log-error mean)."""
    return np.exp(np.log(errors.loc[top5]).mean(axis=0))


def rank_among(value: float, others) -> int:
    """1 + the number of `others` with a strictly lower error (1 = best)."""
    return 1 + int(sum(o < value for o in others))


def bucket_frame(props: pd.DataFrame) -> pd.DataFrame:
    """Boolean/categorical property buckets, one column per property."""
    classification = props["task"] != "regression"
    return pd.DataFrame(
        {
            "train rows": pd.cut(
                props["train_rows"],
                [0, 1999, 10000, np.inf],
                labels=["<2K", "2K-10K", ">10K"],
            ).astype(str),
            "features": pd.cut(
                props["features"], [0, 20, 100, np.inf], labels=["<=20", "21-100", ">100"]
            ).astype(str),
            "categorical": np.where(props["cat_features"] > 0, "any", "none"),
            "max cardinality": np.where(
                props["max_cardinality"].isna(),
                "unknown",
                np.where(props["max_cardinality"] >= 21, ">=21", "<21"),
            ),
            "missing": np.where(props["missing_any"], "any", "none"),
            "minority share": np.where(
                classification,
                np.where(props["minority_share"] < 0.10, "<10%", ">=10%"),
                "n/a",
            ),
            "classes": np.where(
                classification,
                np.where(props["classes"] > 2, ">2", "2"),
                "n/a",
            ),
        },
        index=props.index,
    )


# ---------------------------------------------------------------- Elo (TabArena's method)


def bradley_terry(errors: pd.DataFrame) -> pd.Series:
    """Elo of every method (rows of `errors`, lower is better; complete, one unit per column).

    Each dataset is one battle per pair of methods; a pair's result is the wins of each method
    on that dataset, ties as half a win. With every method on every dataset, the
    Bradley-Terry likelihood depends on the data only through each method's total wins, which
    are read from the average ranks (wins = n - rank). The maximum is found in log-strengths and
    scaled to the Elo scale (400 per factor 10), then shifted so that RandomForest (default) is
    1000. This is `bencheval`'s rank path (`EloHelper.compute_mle_elo_from_ranks`).
    """
    if errors.isna().to_numpy().any():
        raise ValueError("the Elo fit needs every method on every dataset (fill the missing first)")
    n, n_tasks = errors.shape
    ranks = errors.rank(axis=0, method="average").to_numpy()
    wins = (n - ranks).sum(axis=1)
    upper = np.triu_indices(n, 1)
    off_diagonal = ~np.eye(n, dtype=bool)

    def objective(t: np.ndarray):
        top = np.maximum(t[:, None], t[None, :])
        log_sum = top + np.log(np.exp(t[:, None] - top) + np.exp(t[None, :] - top))
        log_likelihood = wins @ t - n_tasks * log_sum[upper].sum()
        win_probability = np.exp(t[:, None] - log_sum)
        gradient = wins - n_tasks * np.where(off_diagonal, win_probability, 0.0).sum(axis=1)
        return -log_likelihood + BT_RIDGE * (t @ t), -gradient + 2 * BT_RIDGE * t

    fit = minimize(
        objective,
        np.zeros(n),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 10000, "ftol": 1e-15, "gtol": 1e-12},
    )
    strength = fit.x - fit.x.mean()
    elo = pd.Series(400 * strength / math.log(10) + 1000, index=errors.index)
    return elo + (CALIBRATION_ELO - elo[CALIBRATION])


def rank_of(elo: pd.Series, name: str) -> int:
    return 1 + int((elo > elo[name]).sum())


def pool_elo(pool: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """(Elo, rank) of every entrant of a filled pool."""
    elo = bradley_terry(pool)
    rank = pd.Series({name: rank_of(elo, name) for name in elo.index})
    return elo, rank


def with_error(pool: pd.DataFrame, name: str, changes: pd.Series) -> pd.DataFrame:
    """The pool with `name`'s errors replaced on the datasets in `changes` (a Series)."""
    out = pool.copy()
    for dataset, value in changes.items():
        out.at[name, dataset] = value
    return out


def scaled(pool: pd.DataFrame, name: str, factor: pd.Series | float) -> pd.DataFrame:
    """The pool with `name`'s errors multiplied by `factor` (scalar or per dataset)."""
    out = pool.copy()
    out.loc[name] = out.loc[name] * factor
    return out


def rank_with_reduction(pool: pd.DataFrame, name: str, mask: pd.Series, r: float) -> int:
    """The rank of `name` when its errors on the datasets in `mask` are multiplied by 1 - r."""
    factor = pd.Series(np.where(mask.to_numpy(), 1 - r, 1.0), index=pool.columns)
    _, rank = pool_elo(scaled(pool, name, factor))
    return int(rank[name])


def reduction_to_rank(pool: pd.DataFrame, name: str, mask: pd.Series) -> float | None:
    """The smallest uniform reduction r of `name`'s errors on the datasets in `mask` (errors
    times 1 - r) at which it reaches TARGET_RANK. None if TARGET_RANK is not reached by r = 0.99
    (a scope that cannot lift the rank alone: the rank stops rising once the method is best on
    every dataset of the scope). Bisection; rank is monotone in r on a real pool."""
    if rank_with_reduction(pool, name, mask, 0.0) <= TARGET_RANK:
        return 0.0
    if rank_with_reduction(pool, name, mask, 0.99) > TARGET_RANK:
        return None
    lo, hi = 0.0, 0.99
    for _ in range(40):
        mid = (lo + hi) / 2
        if rank_with_reduction(pool, name, mask, mid) <= TARGET_RANK:
            hi = mid
        else:
            lo = mid
    return hi


# ---------------------------------------------------------------- output tables


def crowd(
    pool: pd.DataFrame, name: str, exclude=(), top: int = 15, widths=(0.01, 0.02)
) -> pd.DataFrame:
    """The share of (dataset, entrant) pairs, over the `top` board entrants with the highest Elo
    (`name` and the other inserted runs in `exclude` left out), whose error is within each
    relative width of `name`'s error on that dataset."""
    elo, _ = pool_elo(pool)
    skip = {name, *exclude}
    others = [m for m in elo.sort_values(ascending=False).index if m not in skip][:top]
    log_ratio = np.log(pool.loc[others] / pool.loc[name])
    rows = [
        {"entrants": top, "within": w, "share of pairs": float((log_ratio.abs() < w).mean().mean())}
        for w in widths
    ]
    return pd.DataFrame(rows)


def noise_whatif(
    pool: pd.DataFrame,
    name: str,
    cuts=NOISE_CUTS,
    sds=NOISE_SDS,
    draws=NOISE_DRAWS,
    seed=NOISE_SEED,
) -> pd.DataFrame:
    """Elo gain and rank of `name` when every error is multiplied by exp(-x + sd * z), z standard
    normal per dataset (`draws` draws per cell, one draw for sd 0)."""
    base, _ = pool_elo(pool)
    rng = np.random.default_rng(seed)
    rows = []
    for sd in sds:
        for x in cuts:
            gains, ranks = [], []
            for _ in range(1 if sd == 0 else draws):
                log_ratio = -x + sd * rng.standard_normal(pool.shape[1])
                factor = pd.Series(np.exp(log_ratio), index=pool.columns)
                elo, rank = pool_elo(scaled(pool, name, factor))
                gains.append(elo[name] - base[name])
                ranks.append(int(rank[name]))
            rows.append(
                {
                    "sd": sd,
                    "G": math.exp(-x),
                    "elo gain mean": float(np.mean(gains)),
                    "elo gain min": float(np.min(gains)),
                    "elo gain max": float(np.max(gains)),
                    "rank median": float(np.median(ranks)),
                    "share rank <= 5": float(np.mean(np.array(ranks) <= TARGET_RANK)),
                }
            )
    return pd.DataFrame(rows)


def transplant_whatif(pool: pd.DataFrame, errors: pd.DataFrame, name: str, pairs=TRANSPLANTS):
    """`name`'s errors times another entrant's error ratio after / before, dataset by dataset.
    `errors` is the filled board; pairs with an entrant that is not on it are skipped."""
    base, _ = pool_elo(pool)
    rows = []
    for before, after, label in pairs:
        if before not in errors.index or after not in errors.index:
            continue
        log_ratio = np.log(errors.loc[after] / errors.loc[before])
        elo, rank = pool_elo(scaled(pool, name, np.exp(log_ratio)))
        rows.append(
            {
                "change": label,
                "G": float(np.exp(log_ratio.mean())),
                "sd of log ratio": float(log_ratio.std()),
                "datasets better": int((log_ratio < 0).sum()),
                "elo": float(elo[name]),
                "elo gain": float(elo[name] - base[name]),
                "rank": int(rank[name]),
            }
        )
    return pd.DataFrame(rows)


def share_whatif(pool, errors, name, steps=KUMO_STEPS, multipliers=COMPUTE_MULTIPLIERS):
    """`name`'s errors times a share of each Kumo step's per-dataset ratios, the share set by a
    compute multiplier over the run's recipe (a scenario by analogy, not a forecast)."""
    base, _ = pool_elo(pool)
    rows = []
    for label, before, after, ratio in steps:
        if before not in errors.index or after not in errors.index:
            continue
        log_ratio = np.log(errors.loc[after] / errors.loc[before])
        for run, multiplier in multipliers:
            share = math.log(multiplier) / math.log(ratio)
            elo, rank = pool_elo(scaled(pool, name, np.exp(share * log_ratio)))
            rows.append(
                {
                    "step": label,
                    "run": run,
                    "compute multiplier": multiplier,
                    "share of the step": share,
                    "G": float(np.exp(share * log_ratio.mean())),
                    "elo gain": float(elo[name] - base[name]),
                    "rank": int(rank[name]),
                }
            )
    return pd.DataFrame(rows)


def close_gap(run: pd.Series, target: pd.Series, mask: pd.Series, fraction: float) -> pd.Series:
    """`run` with a `fraction` of its log gap to `target` closed on the datasets in `mask` where
    it is behind (fraction 1: the target's error there)."""
    out = run.copy()
    behind = mask & (target < run)
    out[behind] = np.exp((1 - fraction) * np.log(run[behind]) + fraction * np.log(target[behind]))
    return out


def bucket_whatif(pool, name, buckets, kumo, best_top5, fractions=CLOSE_FRACTIONS):
    """Elo of `name` with its gap closed on the datasets of each property bucket."""
    base, _ = pool_elo(pool)
    run = pool.loc[name]
    masks = {"all": pd.Series(True, index=pool.columns)}
    for column in buckets.columns:
        for value in sorted(buckets[column].unique()):
            if value in ("n/a", "unknown"):
                continue
            masks[f"{column} {value}"] = (buckets[column] == value).reindex(pool.columns)
    # the four buckets where TabSil is furthest behind (v2-top5-plan.md §4.1), together
    masks["missing any, features 21-100, minority share <10% or categorical none"] = (
        (buckets["missing"] == "any")
        | (buckets["features"] == "21-100")
        | (buckets["minority share"] == "<10%")
        | (buckets["categorical"] == "none")
    ).reindex(pool.columns)
    rows = []
    for label, mask in masks.items():
        mask = mask.fillna(False).astype(bool)
        row = {"bucket": label, "datasets": int(mask.sum())}
        row["behind Kumo-Tabular-Medium"] = int((mask & (kumo < run)).sum())
        targets = [(f"{f:.0%} of the gap to Kumo-Tabular-Medium", kumo, f) for f in fractions]
        targets.append(("best top-5", best_top5, 1.0))
        for column, target, fraction in targets:
            changed = pool.copy()
            changed.loc[name] = close_gap(run, target, mask, fraction)
            elo, rank = pool_elo(changed)
            row[f"{column}: elo gain"] = float(elo[name] - base[name])
            row[f"{column}: rank"] = int(rank[name])
        rows.append(row)
    return pd.DataFrame(rows)


def split_whatif(
    errors: pd.DataFrame, before: pd.Series, after: pd.Series, datasets
) -> pd.DataFrame:
    """Elo of `before`, of `after`, and of `before` with `after`'s errors on `datasets` (each one
    alone and all together), each inserted alone into the filled board `errors`."""

    def elo_of(values: pd.Series) -> tuple[float, int]:
        pool = errors.copy()
        pool.loc["what-if"] = values
        elo, rank = pool_elo(pool)
        return float(elo["what-if"]), int(rank["what-if"])

    datasets = [d for d in datasets if d in before.index]
    cases = [("before", before), ("after", after)]
    for d in datasets:
        changed = before.copy()
        changed[d] = after[d]
        cases.append((f"before with after on {d}", changed))
    changed = before.copy()
    changed[datasets] = after[datasets]
    cases.append(("before with after on all of these", changed))
    rows = []
    for label, values in cases:
        elo, rank = elo_of(values)
        rows.append({"case": label, "elo": elo, "rank": rank})
    out = pd.DataFrame(rows)
    out["elo gain over before"] = out["elo"] - out.loc[0, "elo"]
    return out


def compare_buckets(arm: pd.Series, control: pd.Series, buckets: pd.DataFrame) -> pd.DataFrame:
    """G of `arm` against `control` (split-0 errors per dataset) overall and per property bucket."""
    log_ratio = np.log(arm / control)
    rows = [{"bucket": "all", "datasets": len(log_ratio), "G": float(np.exp(log_ratio.mean()))}]
    for column in buckets.columns:
        for value in sorted(buckets[column].unique()):
            if value in ("n/a", "unknown"):
                continue
            members = buckets.index[buckets[column] == value]
            rows.append(
                {
                    "bucket": f"{column} {value}",
                    "datasets": len(members),
                    "G": float(np.exp(log_ratio[members].mean())),
                }
            )
    return pd.DataFrame(rows)


def markdown_table(frame: pd.DataFrame, floatfmt: str = ".3f") -> str:
    """A plain markdown table (no extra dependency)."""
    cols = [str(c) for c in frame.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for row in frame.itertuples(index=False):
        cells = []
        for v in row:
            if isinstance(v, (float, np.floating)):
                cells.append("" if np.isnan(v) else format(float(v), floatfmt))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--offline", action="store_true", help="use the cache only")
    parser.add_argument(
        "--run",
        action="append",
        metavar="LABEL=RUN_ID",
        help="a TabSil run (repeat); default: TabSil #14 and TabSil v1.1",
    )
    parser.add_argument(
        "--space-revision",
        metavar="SHA",
        help="a commit of the Space to read instead of its current main (the plan's numbers: "
        "a525fe89b5824325055ad58ff3de5590047a2877)",
    )
    parser.add_argument(
        "--compare",
        action="append",
        default=[],
        metavar="LABEL=ARM_ID:CONTROL_ID[+CONTROL_ID]",
        help="G of one run against another (or the geometric mean of several) by property bucket, "
        "from their split-0 errors (repeat); e.g. R10 against #14's run and N10",
    )
    parser.add_argument(
        "--no-planning",
        action="store_true",
        help="skip the planning what-ifs (noise, transplants, buckets, split; a few minutes)",
    )
    args = parser.parse_args(argv)
    if args.run:
        runs = []
        for item in args.run:
            label, _, run_id = item.partition("=")
            if not label or not run_id:
                parser.error(f"--run takes LABEL=RUN_ID, got {item!r}")
            runs.append((label, run_id))
        args.runs = tuple(runs)
    else:
        args.runs = DEFAULT_RUNS
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cache = args.cache_dir
    cache.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    sha = args.space_revision or space_revision(cache, args.offline)
    errors, info, published = load_board(cache, sha, args.offline)
    props = load_properties(cache, TABARENA_SHA, args.offline)
    datasets = list(errors.columns)
    if set(props.index) != set(datasets):
        raise SystemExit("the dataset names of the board and of the metadata differ")
    props = props.loc[datasets]
    top5 = list(published.sort_values("elo", ascending=False).index[:5])
    refs = {label: errors.loc[name] for label, name in REFERENCES.items()}

    runs = {}
    for label, run_id in args.runs:
        frame = load_run(cache, run_id, args.offline).loc[datasets]
        if (frame["metric"] != info.loc[datasets, "metric"]).any():
            raise SystemExit(f"{label}: a dataset's metric differs from the board's")
        runs[label] = frame

    per = pd.DataFrame(index=datasets)
    per["task"] = props["task"]
    per["metric"] = info.loc[datasets, "metric"]
    for label, frame in runs.items():
        per[f"error {label}"] = frame["error"]
    for label, values in refs.items():
        per[f"error {label}"] = values
    per["error top-5 geo-mean"] = top5_geo_error(errors, top5)
    for label in runs:
        for ref in list(refs) + ["top-5 geo-mean"]:
            per[f"ratio {label} / {ref}"] = per[f"error {label}"] / per[f"error {ref}"]
        # rank among the six references and this run (1 = lowest error on the dataset)
        per[f"rank {label} of 7"] = [
            rank_among(per.at[d, f"error {label}"], [per.at[d, f"error {r}"] for r in refs])
            for d in datasets
        ]
    per = per.join(props.drop(columns=["task"]))

    # the Elo of the board, reproduced with TabArena's method
    board_elo = bradley_terry(fill_missing(errors))
    reproduced = pd.DataFrame(
        {
            "published": published["elo"],
            "reproduced": board_elo,
            "imputed_pct": published["imputed_pct"],
        }
    ).loc[errors.index]
    reproduced["diff"] = reproduced["reproduced"] - reproduced["published"]
    max_diff = float(reproduced["diff"].abs().max())

    pool = fill_missing(errors)
    for label, frame in runs.items():
        pool.loc[label] = frame["error"]
    elo, rank = pool_elo(pool)
    pool_table = pd.DataFrame(
        {
            "published_elo": published["elo"].reindex(pool.index),
            "reproduced_elo": board_elo.reindex(pool.index),
            "pool_elo": elo,
            "pool_rank": rank,
        }
    ).sort_values("pool_elo", ascending=False)
    pool_table["inserted"] = pool_table.index.isin(list(runs))

    # the what-ifs use the primary run (the first of --run)
    primary = next(iter(runs))
    kumo = KUMO_MEDIUM
    kumo_name = REFERENCES[kumo]
    best_top5 = errors.loc[top5].min(axis=0)
    marginal = []
    for d in datasets:
        for kind, value in (
            ("Kumo-Tabular-Medium", errors.at[kumo_name, d]),
            ("best top-5", best_top5[d]),
        ):
            changed = with_error(pool, primary, pd.Series({d: value}))
            e2, r2 = pool_elo(changed)
            marginal.append(
                {
                    "dataset": d,
                    "task": props.at[d, "task"],
                    "set to": kind,
                    "error before": pool.at[primary, d],
                    "error after": value,
                    "elo": e2[primary],
                    "rank": int(r2[primary]),
                    "elo gain": e2[primary] - elo[primary],
                }
            )
    marginal = pd.DataFrame(marginal).sort_values(["set to", "elo gain"], ascending=[True, False])
    base_elo, base_rank = float(elo[primary]), int(rank[primary])

    reductions = []
    scopes = {
        "all": ("binary", "multiclass", "regression"),
        "binary": ("binary",),
        "multiclass": ("multiclass",),
        "regression": ("regression",),
        "classification": ("binary", "multiclass"),
    }
    for scope, tasks in scopes.items():
        mask = props["task"].isin(tasks)
        r = reduction_to_rank(pool, primary, mask)
        reductions.append(
            {
                "scope": scope,
                "datasets": int(mask.sum()),
                "reduction %": float("nan") if r is None else 100 * r,
                "rank at 99% reduction": rank_with_reduction(pool, primary, mask, 0.99),
            }
        )
    reductions = pd.DataFrame(reductions)
    matched = []
    for task in ("binary", "multiclass", "regression"):
        mask = props["task"] == task
        changed = pool.copy()
        changed.loc[primary, mask.to_numpy()] = errors.loc[kumo_name, mask.to_numpy()].to_numpy()
        e2, r2 = pool_elo(changed)
        matched.append(
            {
                "matched KTM-M on": task,
                "datasets": int(mask.sum()),
                "elo": e2[primary],
                "rank": int(r2[primary]),
            }
        )
    matched = pd.DataFrame(matched)

    # G, the buckets and the gap tables
    buckets = bucket_frame(props)
    group_rows, bucket_rows = [], []
    for label in runs:
        for group, tasks in GROUPS.items():
            members = props.index[props["task"].isin(tasks)]
            for ref in list(refs) + ["top-5 geo-mean"]:
                ratios = per.loc[members, f"ratio {label} / {ref}"]
                group_rows.append(
                    {"run": label, "group": group, "reference": ref, **summarize_group(ratios)}
                )
        for column in buckets.columns:
            for value in sorted(buckets[column].unique()):
                members = buckets.index[buckets[column] == value]
                if len(members) == 0:
                    continue
                for ref in list(refs) + ["top-5 geo-mean"]:
                    ratios = per.loc[members, f"ratio {label} / {ref}"]
                    bucket_rows.append(
                        {
                            "run": label,
                            "property": column,
                            "bucket": value,
                            "reference": ref,
                            "n": len(ratios),
                            "G": geo_mean(ratios),
                        }
                    )
    group_table = pd.DataFrame(group_rows)
    bucket_table = pd.DataFrame(bucket_rows)

    # the planning what-ifs (v2-top5-plan.md §4)
    planning = {}
    if not args.no_planning:
        board = fill_missing(errors)
        best_filled = pool.loc[top5].min(axis=0)
        planning["crowd"] = crowd(pool, primary, exclude=list(runs))
        planning["noise"] = noise_whatif(pool, primary)
        planning["transplant"] = transplant_whatif(pool, board, primary)
        planning["share"] = share_whatif(pool, board, primary)
        planning["buckets"] = bucket_whatif(
            pool, primary, buckets, pool.loc[kumo_name], best_filled
        )
        if len(runs) > 1:
            second = list(runs)[1]
            planning["split"] = split_whatif(
                board, runs[second]["error"], runs[primary]["error"], NAMED_DATASETS
            )
    comparisons = []
    for item in args.compare:
        label, _, ids = item.partition("=")
        arm_id, _, control_ids = ids.partition(":")
        if not (label and arm_id and control_ids):
            raise SystemExit(f"--compare takes LABEL=ARM_ID:CONTROL_ID[+CONTROL_ID], got {item!r}")
        arm = load_run(cache, arm_id, args.offline).loc[datasets]["error"]
        controls = [
            load_run(cache, c, args.offline).loc[datasets]["error"] for c in control_ids.split("+")
        ]
        control = np.exp(np.mean([np.log(c) for c in controls], axis=0))
        table = compare_buckets(arm, pd.Series(control, index=datasets), buckets)
        table.insert(0, "comparison", label)
        comparisons.append(table)
    if comparisons:
        planning["compare"] = pd.concat(comparisons, ignore_index=True)

    gap_cols = [
        "task",
        "metric",
        f"error {primary}",
        f"error {kumo}",
        f"ratio {primary} / {kumo}",
        "error top-5 geo-mean",
        f"ratio {primary} / top-5 geo-mean",
        "train_rows",
        "features",
        "cat_share",
        "max_cardinality",
        "missing_any",
        "classes",
        "minority_share",
    ]
    top_kumo = per.sort_values(f"ratio {primary} / {kumo}", ascending=False).head(10)
    top_top5 = per.sort_values(f"ratio {primary} / top-5 geo-mean", ascending=False).head(10)

    write_outputs(
        args.out_dir,
        per,
        group_table,
        bucket_table,
        pool_table,
        marginal,
        reductions,
        matched,
        reproduced,
        top_kumo[gap_cols],
        top_top5[gap_cols],
        dict(
            sha=sha,
            max_diff=max_diff,
            primary=primary,
            base_elo=base_elo,
            base_rank=base_rank,
            runs=args.runs,
            pool_size=len(pool),
        ),
    )
    if planning:
        write_planning(args.out_dir, planning, primary, list(runs))


def write_outputs(
    out_dir,
    per,
    group_table,
    bucket_table,
    pool_table,
    marginal,
    reductions,
    matched,
    reproduced,
    top_kumo,
    top_top5,
    meta,
) -> None:
    per.to_csv(out_dir / "per_dataset.csv", float_format="%.6g")
    group_table.to_csv(out_dir / "group_G.csv", index=False, float_format="%.4f")
    bucket_table.to_csv(out_dir / "bucket_G.csv", index=False, float_format="%.4f")
    pool_table.to_csv(out_dir / "elo_pool.csv", float_format="%.2f")
    marginal.to_csv(out_dir / "marginal_elo.csv", index=False, float_format="%.4f")
    reductions.to_csv(out_dir / "reduction.csv", index=False, float_format="%.3f")
    matched.to_csv(out_dir / "matched_task.csv", index=False, float_format="%.2f")
    reproduced.to_csv(out_dir / "elo_reproduction.csv", float_format="%.2f")
    top_kumo.to_csv(out_dir / "top10_gap_kumo_medium.csv", float_format="%.6g")
    top_top5.to_csv(out_dir / "top10_gap_top5.csv", float_format="%.6g")
    (out_dir / "summary.md").write_text(
        summary_markdown(
            group_table,
            bucket_table,
            pool_table,
            marginal,
            reductions,
            matched,
            reproduced,
            top_kumo,
            top_top5,
            meta,
            per,
        ),
        encoding="utf-8",
    )


def write_planning(out_dir: Path, planning: dict, primary: str, runs: list[str]) -> None:
    """The planning what-ifs as CSV files and a section appended to `summary.md`."""
    for key, frame in planning.items():
        frame.to_csv(out_dir / f"whatif_{key}.csv", index=False, float_format="%.4f")
    sections = {
        "crowd": (
            f"How close the top entrants are to {primary}: the share of (dataset, entrant) pairs "
            "within a relative error width:",
            ".3f",
        ),
        "noise": (
            f"Every row changes the errors of {primary} only and refits the live pool. A cut x of "
            "every error with per-dataset noise: log ratio = -x + sd * N(0, 1), "
            f"{NOISE_DRAWS} draws per cell with sd > 0 (seed {NOISE_SEED}).",
            ".3f",
        ),
        "transplant": (
            f"Another entrant's change, dataset by dataset, applied to {primary} (an analogy):",
            ".3f",
        ),
        "share": (
            f"A share of a Kumo-Tabular step applied to {primary}, set by a compute multiplier "
            "over its recipe (a scenario by analogy):",
            ".3f",
        ),
        "buckets": (
            f"Gap closed on the datasets of a property bucket where {primary} is behind "
            "Kumo-Tabular-Medium (a share of the log gap), or set to the best top-5 error:",
            ".1f",
        ),
        "compare": (
            "Runs against runs (`--compare`), G of split-0 errors by property bucket:",
            ".4f",
        ),
        "split": (
            f"The change from {runs[1] if len(runs) > 1 else ''} (before) to {primary} (after), "
            "split by datasets:",
            ".1f",
        ),
    }
    out = ["", "## Planning what-ifs (v2-top5-plan.md §4)", ""]
    for key, (text, floatfmt) in sections.items():
        if key in planning:
            out += [text, "", markdown_table(planning[key], floatfmt), ""]
    with open(out_dir / "summary.md", "a") as f:
        f.write("\n".join(out))


def summary_markdown(
    group_table,
    bucket_table,
    pool_table,
    marginal,
    reductions,
    matched,
    reproduced,
    top_kumo,
    top_top5,
    meta,
    per,
) -> str:
    primary = meta["primary"]
    out = [f"# Gap of TabSil to the TabArena-Lite top ({date.today().isoformat()})", ""]
    out += [
        "Sources: TabArena Space `" + SPACE + "` revision `" + meta["sha"] + "` (pages.sqlite,"
        " imputation_yes / splits_lite); TabArena metadata at autogluon/tabarena `"
        + TABARENA_SHA
        + "`; OpenML feature API; MLflow runs "
        + ", ".join(f"{label} `{run_id}`" for label, run_id in meta["runs"])
        + f" ({EXPERIMENT_NOTE}, artifact `{ARTIFACT}`).",
        "",
        f"## 3a. Elo reproduction\n\nLargest absolute difference to the published Lite Elo over "
        f"{len(reproduced)} entrants: **{meta['max_diff']:.2f}** (published values are rounded "
        "to integers; the TabArena fill of missing results is the RandomForest (default) error).",
        "",
    ]
    out += ["## 3b. Live pool with the TabSil runs inserted", ""]
    table = pool_table.reset_index().rename(columns={"index": "entrant"}).head(12)
    out += [markdown_table(table, ".1f"), ""]
    rows = pool_table.loc[list(dict(meta["runs"]).keys()) + [PUBLISHED_TABSIL]]
    out += ["Elo in the same fit (rank among " + str(meta["pool_size"]) + " entrants):", ""]
    out += [markdown_table(rows.reset_index().rename(columns={"index": "entrant"}), ".1f"), ""]
    out += ["## Summaries of G (TabSil / reference; below 1 is better)", ""]
    for label in dict(meta["runs"]):
        sub = group_table[(group_table["run"] == label) & (group_table["reference"] == KUMO_MEDIUM)]
        sub2 = group_table[
            (group_table["run"] == label) & (group_table["reference"] == "top-5 geo-mean")
        ]
        merged = sub[["group", "n", "G", "ci_low", "ci_high", "better", "worse"]].merge(
            sub2[["group", "G"]], on="group", suffixes=(" vs Kumo-Medium", " vs top-5 geo")
        )
        out += [f"Run {label}:", "", markdown_table(merged, ".3f"), ""]
    allref = group_table[(group_table["run"] == primary) & (group_table["group"] == "all")]
    out += [
        f"All datasets, run {primary}, G against each reference:",
        "",
        markdown_table(
            allref[["reference", "n", "G", "ci_low", "ci_high", "better", "worse"]], ".3f"
        ),
        "",
    ]
    out += [
        "## Property buckets (run " + primary + ", G against Kumo-Tabular-Medium and top-5)",
        "",
    ]
    bk = bucket_table[
        (bucket_table["run"] == primary)
        & bucket_table["reference"].isin([KUMO_MEDIUM, "top-5 geo-mean"])
    ]
    wide = bk.pivot(index=["property", "bucket"], columns="reference", values="G")
    wide = wide[[KUMO_MEDIUM, "top-5 geo-mean"]].reset_index()
    counts = bk[bk["reference"] == KUMO_MEDIUM].set_index(["property", "bucket"])["n"]
    wide.insert(
        2,
        "n",
        counts.reindex(pd.MultiIndex.from_frame(wide[["property", "bucket"]]))
        .to_numpy()
        .astype(int),
    )
    out += [markdown_table(wide, ".3f"), ""]
    out += ["## Top-10 gap, run " + primary + ", against Kumo-Tabular-Medium", ""]
    out += [markdown_table(top_kumo.reset_index().rename(columns={"index": "dataset"}), ".4g"), ""]
    out += ["## Top-10 gap, run " + primary + ", against the top-5 geometric mean", ""]
    out += [markdown_table(top_top5.reset_index().rename(columns={"index": "dataset"}), ".4g"), ""]
    out += [
        f"## 3c. Marginal Elo of run {primary} (base Elo {meta['base_elo']:.1f}, rank "
        f"{meta['base_rank']})",
        "",
    ]
    for kind in ("Kumo-Tabular-Medium", "best top-5"):
        sub = marginal[marginal["set to"] == kind].head(15)
        out += [
            f"Error on one dataset set to the {kind} error, top 15 by Elo gain:",
            "",
            markdown_table(
                sub[["dataset", "task", "error before", "error after", "elo", "rank", "elo gain"]],
                ".4g",
            ),
            "",
        ]
    out += [
        "## 3d. Reduction to rank 5 and matching Kumo-Tabular-Medium",
        "",
        "Reduction r: the smallest uniform cut of the run's errors on the scope that gives rank "
        f"{TARGET_RANK} (blank: not reached; the last column is the rank at a 99% cut, which "
        "shows where the rank stops rising).",
        "",
        markdown_table(reductions, ".2f"),
        "",
        markdown_table(matched, ".1f"),
        "",
    ]
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    main()
