"""ADBench unsupervised outlier detection: 12 classical tables x 3 seeds, TabSil's chain-rule log
density (`tabsil.tasks.unsupervised`) against the official TabPFN v2 extension and pyod baselines.

The protocol, the 12 tables, the comparison numbers and the Elo are those of the fm-agent-bench
anomaly track (`docs/anomaly.md`, `fab/bench/anomaly/`, results at commit
e4235ac8e6c42ac6b781ac41f8d2102fe5ba1e33, copied to `adbench_reference/`):

- data: ADBench (Minqi824/ADBench @ 3dac822, BSD-2) `adbench/datasets/Classical/<k>_<name>.npz`,
  downloaded once and checked against `SHA256` (cache: `$TABSIL_ADBENCH_DIR`, default
  `~/.cache/tabsil/adbench`)
- tables (`DATASETS`): Classical minus image tables, fewer than 1,000 rows, outside 5-40
  features, fewer than 20 test anomalies, mostly duplicate rows, and repeats of one source table
- split (`adbench_split`, ADBench `DataGenerator.generator` in the unsupervised setting): per seed
  1, 2, 3 (`SPLITS` s1-s3) the global numpy/random seed is set, tables over 10,000 rows are
  subsampled to 10,000, `train_test_split(test_size=0.3, stratify=y)` without a random_state, and
  MinMax scaling fitted on the training rows. The training rows keep their anomalies without
  labels (a contaminated context)
- metrics: test AUROC (main, Elo error 1 - AUROC) and AUPRC (`average_precision_score`)
- methods (`METHODS`): `tabsil_unsup` (TabSilUnsupervised, 10 column orders, score = -log
  density) and `iforest` (scikit-learn IsolationForest with pyod's `IForest` defaults and the
  global numpy seed, to check the split row for row against ADBench Table D4 and the
  fm-agent-bench run). A `+clean` suffix fits on the training rows labeled normal only: it uses
  training labels, so it is a diagnostic of the context contamination and stays out of the Elo
- Elo (`elo`, the fm-agent-bench `fab/bench/tabarena/elo.py` computation): Bradley-Terry on the
  (table, split) errors, each table weighing 1 in total, IsolationForest = 1000, 95% interval from
  200 bootstraps over tables. Pool: the reference's 10 pyod methods and `tabpfn_unsup` (cells the
  reference imputed with the IsolationForest value stay imputed, `imputed`) plus TabSil

`run` writes one row per (table, split, method) to `results.csv` as it goes and skips rows
already there, so a rerun resumes. A row is reused only with the settings it was computed with
(`opts`; for TabSil the weights, member count, seed, number of column orders and saved inference
settings, `_files.model_settings`): a rerun with another model into the same folder stops, and
IsolationForest rows are reused whatever the model. A failed (table, split, method) is reported
and the run ends with an error after the others; nothing is imputed for our methods.
"""

from __future__ import annotations

import os
import random
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from tabsil_train.benchmarks import _elo, _files
from tabsil_train.benchmarks._files import download_verified

ADBENCH_REPO = "Minqi824/ADBench"
ADBENCH_COMMIT = "3dac8221081e190f157d78e93bfa8867f90d0965"
ADBENCH_URL = (
    f"https://raw.githubusercontent.com/{ADBENCH_REPO}/{ADBENCH_COMMIT}/adbench/datasets/Classical"
)
# fm-agent-bench commit of the copied comparison results (`adbench_reference/`)
REFERENCE_COMMIT = "e4235ac8e6c42ac6b781ac41f8d2102fe5ba1e33"
REFERENCE_DIR = Path(__file__).parent / "adbench_reference"

# table name (ADBench Table B1) -> npz file name (number_name)
DATASETS = {
    "annthyroid": "2_annthyroid",
    "cardio": "6_cardio",
    "cover": "10_cover",
    "fault": "12_fault",
    "landsat": "19_landsat",
    "magic.gamma": "22_magic.gamma",
    "mammography": "23_mammography",
    "PageBlocks": "27_PageBlocks",
    "shuttle": "32_shuttle",
    "Waveform": "41_Waveform",
    "Wilt": "44_Wilt",
    "yeast": "47_yeast",
}
# sha256 of the npz files at ADBENCH_COMMIT
SHA256 = {
    "2_annthyroid": "e738f7c58a8740d2c4aac4c71ff49ce5e1d1aa16f44178135fc9c09df05fad1b",
    "6_cardio": "bc5009bde6930b4d298b594f31b34485a9a17210cff0e8fb00a8c3068b885a2d",
    "10_cover": "945a097c3542ae81a551bf1c06dcad78a755d94f84fcaf1d60687e42d1a656df",
    "12_fault": "ec27301f9a4cf7c5d8b22c360cf37ddb043b7fb062465723dfd8b34e2771e5dd",
    "19_landsat": "ba17e2158c07469597b89db49d3982c874c6b713ff6670721ddc88969da77987",
    "22_magic.gamma": "0be375984d7af9ea7a63b12206c240840d8b7daa3607021c5fb8756407dd86f1",
    "23_mammography": "a0308ac9712d78321073b1c2dc98dc37665c0f35c1cc33019416fe8f8d7de7ea",
    "27_PageBlocks": "10a9b4d5bb5d3b9083caab4c4062f665610eada703c3d927bf7c34b1f2d0d223",
    "32_shuttle": "26900b0f6be45d9ddb7a402c9b2f956ceda42d23a8ca4cf4406b8bc1352823d1",
    "41_Waveform": "6c9d867337ad1a2f8372c1d27f4c032f0401237e2041b7c45f00de8d319ff30f",
    "44_Wilt": "c70b72aa611ec87b20efb400943ef3ce4f8bf7aa58fcef825fbd28cc5cee3cc9",
    "47_yeast": "bbb15c873d78a4f6a589f46d0f75814595e2282d2ae7ce7779c750f276c3770d",
}
SPLITS = {"s1": 1, "s2": 2, "s3": 3}  # ADBench run.py seed_list
TEST_SIZE, MAX_ROWS, MIN_ROWS = 0.3, 10_000, 1_000  # ADBench data_generator.py
CLEAN = "+clean"
METHODS = ("tabsil_unsup", "iforest")
ANCHOR = "iforest"  # Elo 1000
OURS = "tabsil_unsup"
OFFICIAL = "tabpfn_unsup"
RESULT_COLUMNS = [
    "dataset", "split", "seed", "method", "auroc", "aupr", "seconds", "n_context", "n_test",
    "test_anomalies", "n_features", "opts",
]  # fmt: skip


def cache_dir() -> Path:
    return Path(os.environ.get("TABSIL_ADBENCH_DIR", Path.home() / ".cache" / "tabsil" / "adbench"))


def npz_path(dataset: str, root: Path | None = None) -> Path:
    """The cached npz of `dataset`, downloaded from ADBENCH_URL and checked against SHA256 the
    first time (`_files.download_verified`: a download with another hash is not cached)."""
    name = DATASETS[dataset]
    return download_verified(
        f"{ADBENCH_URL}/{name}.npz",
        (root or cache_dir()) / f"{name}.npz",
        SHA256[name],
        what=f"the ADBench @ {ADBENCH_COMMIT[:7]} file",
        timeout=120,
    )


def load(dataset: str, root: Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    d = np.load(npz_path(dataset, root), allow_pickle=False)  # downloaded file: no pickle
    return np.asarray(d["X"], dtype=float), np.asarray(d["y"]).astype(int)


def set_seed(seed: int) -> None:
    """ADBench `Utils.set_seed` (numpy and random global state; torch is seeded by our methods'
    own random_state)."""
    np.random.seed(seed)
    random.seed(seed)


def adbench_split(X: np.ndarray, y: np.ndarray, seed: int):
    """ADBench `DataGenerator.generator` (unsupervised, la=0, no noise) -> (X_train, X_test,
    y_train, y_test), MinMax fitted on the training rows. Same order of calls as the original:
    set_seed -> (more than 10,000 rows: set_seed again -> 10,000 rows without replacement) ->
    train_test_split without random_state (global numpy state). The 12 tables have at least 1,000
    rows, so ADBench's resampling of small tables is not here."""
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler

    if len(y) < MIN_ROWS:
        raise ValueError(f"{len(y)} rows < {MIN_ROWS}: ADBench resamples these (not used here)")
    set_seed(seed)
    if len(y) > MAX_ROWS:
        set_seed(seed)
        idx = np.random.choice(np.arange(len(y)), MAX_ROWS, replace=False)
        X, y = X[idx], y[idx]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, shuffle=True, stratify=y
    )
    scaler = MinMaxScaler().fit(X_train)
    return scaler.transform(X_train), scaler.transform(X_test), y_train, y_test


# --- methods --------------------------------------------------------------------------------


def base(method: str) -> str:
    return method.removesuffix(CLEAN)


def check_methods(methods: list[str]) -> list[str]:
    if bad := [m for m in methods if base(m) not in METHODS]:
        raise ValueError(f"unknown methods {bad} (known: {METHODS}, each optionally {CLEAN})")
    return methods


def iforest_scores(X_train: np.ndarray, X_test: np.ndarray, seed: int) -> np.ndarray:
    """pyod `IForest()` (n_estimators=100, max_samples='auto', contamination=0.1, random_state
    None): scikit-learn IsolationForest drawing from the global numpy state set to the seed, as
    ADBench. pyod's decision_function is -score_samples up to a constant, so the ranking is the
    same."""
    from sklearn.ensemble import IsolationForest

    set_seed(seed)
    model = IsolationForest(n_estimators=100, max_samples="auto", contamination=0.1).fit(X_train)
    return -model.score_samples(X_test)


def tabsil_scorer(model, borders, n_permutations: int, n_estimators: int | None, device: str,
                  seed_offset: int = 0) -> Callable:  # fmt: skip
    """fit(X_train) -> outlier_score(X_test) of `TabSilUnsupervised`, random_state = the ADBench
    seed + `seed_offset` (column orders and members)."""
    from tabsil.tasks.unsupervised import TabSilUnsupervised

    def score(X_train: np.ndarray, X_test: np.ndarray, seed: int) -> np.ndarray:
        est = TabSilUnsupervised(
            model,
            borders,
            n_permutations=n_permutations,
            n_estimators=n_estimators,
            random_state=seed + seed_offset,
            device=device,
        )
        return est.fit(X_train).outlier_score(X_test)

    return score


def job_cost(dataset: str, method: str) -> float:
    """Relative cost of a TabSil job (to start the expensive ones first): columns (one
    conditional per column and order) x rows after ADBench's 10,000-row cap."""
    X, _ = load(dataset)
    return X.shape[1] * min(len(X), MAX_ROWS) * (0.9 if method.endswith(CLEAN) else 1.0)


# --- run ------------------------------------------------------------------------------------


def read_results(path: Path) -> pd.DataFrame:
    return _files.read_results(path, RESULT_COLUMNS)


def row_settings(methods: list[str], opts: dict[str, str] | None) -> dict[str, str]:
    """Method -> the settings string (`opts`) of its rows: its base method's (a `+clean` row is
    computed with the settings of the method it is a diagnostic of)."""
    return {m: (opts or {}).get(base(m), "") for m in methods}


def run(
    scorers: dict[str, Callable],
    datasets: list[str],
    splits: list[str],
    methods: list[str],
    out_dir: Path,
    opts: dict[str, str] | None = None,
    log: Callable[[str], None] = print,
) -> tuple[pd.DataFrame, list[dict]]:
    """Score every (table, split, method) not yet in `out_dir/results.csv`, appending a row as
    each finishes. `scorers[base method](X_train, X_test, seed)` -> test scores (higher = more
    anomalous). `opts[base method]` is the settings string written with its rows; rows of
    `methods` already there with another string stop the run (`_files.check_settings`). Returns
    (all rows, failures)."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    results = _files.Results(out_dir, RESULT_COLUMNS, row_settings(methods, opts), "adbench", log)
    done = results.done
    finished = set(zip(done.dataset, done.split, done.method, strict=True))
    for dataset in datasets:
        X, y = load(dataset)
        for split in splits:
            seed = SPLITS[split]
            X_train, X_test, y_train, y_test = adbench_split(X, y, seed)
            for method in methods:
                if (dataset, split, method) in finished:
                    continue
                X_fit = X_train[y_train == 0] if method.endswith(CLEAN) else X_train
                t0 = time.time()
                try:
                    s = np.asarray(scorers[base(method)](X_fit, X_test, seed), dtype=float)
                    if s.shape != (len(X_test),) or not np.isfinite(s).all():
                        raise ValueError(
                            f"scores of shape {s.shape} with {int((~np.isfinite(s)).sum())} "
                            "non-finite values"
                        )
                except Exception as e:  # noqa: BLE001 - record and go on; the run fails at the end
                    results.fail(dict(dataset=dataset, split=split, method=method), e)
                    continue
                row = dict(
                    dataset=dataset, split=split, seed=seed, method=method,
                    auroc=float(roc_auc_score(y_test, s)),
                    aupr=float(average_precision_score(y_test, s)),
                    seconds=round(time.time() - t0, 1), n_context=len(X_fit), n_test=len(X_test),
                    test_anomalies=int(y_test.sum()), n_features=X.shape[1],
                    opts=(opts or {}).get(base(method), ""),
                )  # fmt: skip
                results.append([row])
                log(
                    f"[adbench] {dataset} {split} {method}: AUROC {row['auroc']:.4f} "
                    f"AUPRC {row['aupr']:.4f} ({row['seconds']:.0f}s, context {len(X_fit)}, "
                    f"test {len(X_test)})"
                )
    return results.finish()


# --- reference and comparison ---------------------------------------------------------------


def reference_results() -> pd.DataFrame:
    """The fm-agent-bench run (REFERENCE_COMMIT `results/anomaly.csv`): pyod's 10 methods,
    `iforest+clean`, `tabpfn_unsup` (TabPFN v2, 10 orders) and its `+clean`, one row per (table,
    split, method) (the latest; repeated rows hold the same score)."""
    ref = pd.read_csv(REFERENCE_DIR / "anomaly.csv")
    ref = ref.sort_values("timestamp").drop_duplicates(["dataset", "split", "model"], keep="last")
    return ref.rename(columns={"model": "method", "score": "auroc"})[
        ["dataset", "split", "seed", "method", "auroc", "aupr", "opts"]
    ].reset_index(drop=True)


def paper_reference() -> pd.DataFrame:
    """ADBench Table D4/D5 (seed 1-3 mean AUROC and AUPRC, pyod 1.0.0) for the 12 tables."""
    return pd.read_csv(REFERENCE_DIR / "anomaly_reference.csv")


def table_d4_check(ours: pd.DataFrame, method: str = ANCHOR) -> pd.DataFrame:
    """Per table: our seed-mean AUROC of `method` against ADBench Table D4 (`match`: same to two
    decimals, |diff| <= 0.005) and against the fm-agent-bench run of the same splits."""
    ref = paper_reference()
    d4 = ref[ref.method == method].set_index("dataset").aucroc
    fab = reference_results()
    fab = fab[fab.method == method]
    g = ours[ours.method == method]
    mean = g.groupby("dataset").auroc.mean()
    out = pd.DataFrame({"ours": mean, "table_d4": d4.reindex(mean.index)})
    out["diff"] = out.ours - out.table_d4
    out["match"] = out["diff"].abs() <= 0.005
    per_split = g.merge(fab, on=["dataset", "split"], suffixes=("", "_fab"))
    out["max_split_diff_vs_fab"] = (
        (per_split.auroc - per_split.auroc_fab).abs().groupby(per_split.dataset).max()
    )
    out["splits"] = g.groupby("dataset").size()
    return out.reset_index(names="dataset")


# --- Elo (`_elo`: fm-agent-bench fab/bench/tabarena/elo.py) -----------------------------------


def elo(long: pd.DataFrame, anchor: str = ANCHOR, rounds: int = _elo.ROUNDS,
        seed: int = _elo.SEED) -> pd.DataFrame:  # fmt: skip
    """long: method, dataset, split, metric_error (+ imputed 0/1) without gaps -> per method elo,
    elo+, elo- (95% bootstrap over tables), rank, winrate, imputed share. Pairs compare within a
    (table, split), weighted 1 / (splits of the table) (`_elo.table`)."""
    key = ["method", "dataset", "split"]
    if long.duplicated(key).any():
        raise ValueError("repeated (method, dataset, split) rows")
    long = (
        long.assign(imputed=long.imputed.astype(float))
        if "imputed" in long
        else long.assign(imputed=0.0)
    )
    wide = long.pivot(index="method", columns=["dataset", "split"], values="metric_error")
    if wide.isna().any().any():
        raise ValueError(f"{int(wide.isna().sum().sum())} gaps: fill them first")
    imp = long.pivot(index="method", columns=["dataset", "split"], values="imputed")
    imp = imp.reindex_like(wide)
    names, _, weight = _elo.unit_weights(wide)
    out = _elo.table(wide, anchor, rounds, seed).assign(
        imputed=(imp.fillna(0.0).to_numpy(float) * weight).sum(1) / weight.sum(),
        datasets=len(names),
    )
    return out.sort_values("elo", ascending=False).rename_axis("method").reset_index()


def fill(long: pd.DataFrame, methods: list[str], anchor: str = ANCHOR) -> pd.DataFrame:
    """Every method on every (table, split) of the anchor; missing cells get the anchor's error
    and imputed = 1 (fm-agent-bench `fill`)."""
    ref = long[long.method == anchor][["dataset", "split", "metric_error"]]
    parts = []
    for m in methods:
        g = ref.merge(
            long[long.method == m][["dataset", "split", "metric_error"]],
            on=["dataset", "split"], how="left", suffixes=("_anchor", ""),
        )  # fmt: skip
        g["imputed"] = g.metric_error.isna().astype(float)
        g["metric_error"] = g.metric_error.fillna(g.metric_error_anchor)
        parts.append(g.drop(columns="metric_error_anchor").assign(method=m))
    return pd.concat(parts, ignore_index=True)


def elo_pool(ours: pd.DataFrame, methods: list[str] = (OURS,)) -> pd.DataFrame:
    """The Elo pool: the reference's methods without `+clean` (pyod 10 + `tabpfn_unsup`) and our
    `methods`, on the reference's (table, split) units."""
    ref = reference_results()
    ref = ref[~ref.method.str.endswith(CLEAN)]
    g = ours[ours.method.isin(methods)]
    long = pd.concat([ref, g], ignore_index=True)[["method", "dataset", "split", "auroc"]]
    long = long.assign(metric_error=1 - long.auroc)
    return fill(long, list(dict.fromkeys(long.method)))


def summary_table(ours: pd.DataFrame) -> pd.DataFrame:
    """Per table, seed-mean AUROC and AUPRC: ours (and `+clean`), the official TabPFN extension
    (and `+clean`), IsolationForest, ECOD, the best of the 10 pyod methods, and our rank among
    pyod 10 + `tabpfn_unsup` + ours."""
    ref = reference_results()
    both = pd.concat([ref, ours[ref.columns.intersection(ours.columns)]], ignore_index=True)
    auroc = both.pivot_table(index="dataset", columns="method", values="auroc", aggfunc="mean")
    aupr = both.pivot_table(index="dataset", columns="method", values="aupr", aggfunc="mean")
    pyod = [
        m for m in auroc.columns
        if not m.endswith(CLEAN) and m not in (OURS, OFFICIAL)
        and m in set(ref.method)
    ]  # fmt: skip
    out = pd.DataFrame(index=auroc.index)
    cols = [OURS, OURS + CLEAN, OFFICIAL, OFFICIAL + CLEAN, ANCHOR, "ecod"]
    for c in cols:
        if c in auroc:
            out[c] = auroc[c]
    out["best_pyod"] = auroc[pyod].max(axis=1)
    out["best_pyod_method"] = auroc[pyod].idxmax(axis=1)
    if OURS in auroc:
        pool = auroc[[*pyod, OFFICIAL, OURS]]
        out["rank"] = pool.rank(axis=1, ascending=False)[OURS]
        out[f"{OURS}-{OFFICIAL}"] = auroc[OURS] - auroc[OFFICIAL]
        out[f"aupr_{OURS}"] = aupr[OURS]
    out[f"aupr_{OFFICIAL}"] = aupr[OFFICIAL]
    out["aupr_iforest"] = aupr[ANCHOR]
    return out.reset_index()


# the files `report` writes
REPORT_FILES = ("table_d4.csv", "summary.csv", "elo.csv")


def report(ours: pd.DataFrame, out_dir: Path, methods: list[str] = (OURS,)) -> dict:
    """Write `table_d4.csv` (the split check, with IsolationForest rows), `summary.csv` (per table,
    with TabSil rows) and `elo.csv` (pool with IsolationForest = 1000), and return the main
    numbers. The Elo needs every (table, split) of our methods (it does not impute ours).

    `ours` is the rows to report: the caller passes those of the methods its run asked for (other
    rows of `results.csv` can be an earlier run's, of another model). A file an earlier report
    wrote and this one does not is removed, so the folder holds one report."""
    for name in REPORT_FILES:
        (out_dir / name).unlink(missing_ok=True)
    out: dict = {}
    if ANCHOR in set(ours.method):
        check = table_d4_check(ours)
        check.to_csv(out_dir / "table_d4.csv", index=False)
        out["iforest_table_d4_match"] = f"{int(check.match.sum())}/{len(check)}"
        out["iforest_max_split_diff_vs_fab"] = float(check.max_split_diff_vs_fab.max())
    if OURS not in set(ours.method):
        return out
    summary = summary_table(ours)
    summary.to_csv(out_dir / "summary.csv", index=False)
    g = ours[ours.method == OURS]
    out["mean_auroc"] = float(g.groupby("dataset").auroc.mean().mean())
    out["mean_aupr"] = float(g.groupby("dataset").aupr.mean().mean())
    clean = ours[ours.method == OURS + CLEAN]
    if len(clean):
        out["mean_auroc_clean"] = float(clean.groupby("dataset").auroc.mean().mean())
    complete = len(g) == len(DATASETS) * len(SPLITS)
    out["complete"] = complete
    if complete:
        table = elo(elo_pool(ours, list(methods)))
        table.to_csv(out_dir / "elo.csv", index=False)
        out["elo"] = table.set_index("method").elo.round(1).to_dict()
        out["imputed"] = table.set_index("method").imputed.round(4).to_dict()
    return out
