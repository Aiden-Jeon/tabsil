"""Missing value imputation: 10 OpenML tables x MCAR rates 10/30/50% x 3 seeds, TabSil
(`tabsil.TabSilUnsupervised.impute`, zero-shot) against the official TabPFN v2
extension and the usual baselines, every method on the same rows and masks.

- tables (`TABLES`): OpenML datasets used in imputation benchmarks (HyperImpute, GAIN, MissForest
  evaluations), 3 numeric only and 7 with categorical columns, at most 21 columns. The whole table
  is imputed (the OpenML target is a column like the others). The ARFF file is downloaded once from
  OpenML and checked against `SHA256` (cache: `$TABSIL_IMPUTE_DIR`, default
  `~/.cache/tabsil/impute`). Column kinds are the ARFF's: a nominal attribute is categorical (codes
  0..k-1 in the declared order), the rest numeric. Rows with a missing value in the file are
  dropped (adult only)
- split (`make_split`): per (table, seed) at most `MAX_ROWS` (2,000) rows without replacement
  (numpy `default_rng([seed, table id])`); per (table, rate, seed) every cell is masked
  independently with probability `rate` (MCAR, `default_rng([seed, rate %, table id, 1])`). The
  methods get the masked table and fill it (transductive: fit and impute the same rows)
- metrics on the masked cells only: `nrmse`, the mean over numeric columns of RMSE / standard
  deviation of the column (all rows of the split; the mean imputation is about 1) and `accuracy`,
  the mean over categorical columns of the share of masked cells imputed with the true class.
  Every method's categorical values are set to the nearest class observed in that column
  (integer codes; most methods already give one)
- methods (`METHODS`): `tabsil` (one evaluation gives `tabsil` = predictive mean and
  `tabsil_median`), `tabpfn_impute` (tabpfn-extensions at a049fd7
  `TabPFNUnsupervisedModel.impute` defaults: 10 orders, t = 1e-9, with TabPFN v2, and one fix of
  its missing-value handling, `official_model_class`), `tabpfn_simple` (`simple_impute` of the
  same commit, one fit per column), `mean` (mean / most frequent),
  `knn` (scikit-learn KNNImputer, 5 neighbors, on z-scored numeric and one-hot categorical
  columns; a category is the largest one-hot value), `mice` (chained equations, `impute_mice`:
  BayesianRidge for numeric and multinomial logistic regression for categorical columns, 10
  rounds) and `missforest` (`impute_missforest`, Stekhoven and Buehlmann 2012). Every method is
  given the table's categorical columns; both TabPFN extension methods add the numeric columns
  they infer as categorical (`extension_categorical`), and `tabsil_inferred` (TabSil's mean with
  those same column kinds) is the ablation for that difference
- random numbers: every method draws its own from the split's `seed`, so a second run on the same
  machine gives the same rows. Two baselines need help for that: MissForest predicts on one thread
  (`impute_missforest`), and both global generators the official extension draws from are seeded
  (`tabpfn_imputer`)

`run` writes one row per (table, rate, seed, method) to `results.csv` as it goes and skips rows
already there, so a rerun resumes. A row is reused only with the settings it was computed with
(`opts`; for TabSil the weights, member count, seed and saved inference settings,
`_files.model_settings`): a rerun with another model into the same folder stops, and the
baselines' rows are reused whatever the model. A method that raises or leaves a masked cell
non-finite is a failure: it is recorded in `failures.json` and the run ends with an error after
the others; nothing is filled in for it.
"""

from __future__ import annotations

import functools
import json
import os
import random
import time
import zlib
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from tabsil.tasks.unsupervised import nearest_class  # the same rule as TabSilUnsupervised.impute
from tabsil_train.benchmarks import _files
from tabsil_train.benchmarks._files import download_verified

OPENML_URL = "https://api.openml.org/data/v1/download"
# name -> (OpenML dataset id, ARFF file id, ARFF file name)
TABLES = {
    "airfoil_self_noise": (44957, 22111821, "airfoil_self_noise"),
    "concrete_compressive_strength": (44959, 22111823, "concrete_compressive_strength"),
    "wine-quality-red": (40691, 4965268, "wine-quality-red"),
    "california_housing": (44977, 22111841, "california_housing"),
    "yeast": (181, 3616, "yeast"),
    "abalone": (183, 3620, "abalone"),
    "adult": (1590, 1595261, "adult"),
    "credit-g": (31, 31, "credit-g"),
    "bank-marketing": (1461, 1586218, "bank-marketing"),
    "cmc": (23, 23, "cmc"),
}
# OpenML dataset id -> sha256 of its ARFF file (the md5 OpenML lists matches too)
SHA256 = {
    44957: "27c5c925f7f81911e9bd4664bba3656d2558d9ba301a6a0a617fac4b5797117a",
    44959: "510beb3f50cba7907815b611af40edf14eb75902995bb9d7bccb40db156cf8ec",
    40691: "f1179c9abed3a09cf4202caa08306f8b78c7e401bf37d31440fdeac003796314",
    44977: "8462bfde6053fab576dc98aa43ecd4cc7fb9d8af0337f72bdf60f0392734602a",
    181: "ae97eccd3ad5f9a998efec8acb2026f1748df6c8a392a5b2b0f321c2ef053933",
    183: "a75351597c04719e793dd5fc9bee926bae0f06e413fe77591f01e4688154988e",
    1590: "77aa1703717a29f0b5642e94c3ba1defd2486f0b34d4d8eccc1b37a5f7d226b0",
    31: "768be84ef749110d5fcac109a6541f2df190e8c6987e1dc432c0a4f29d0642f5",
    1461: "7381f5eec0cfcd20b70e70988ead1cff758fa1e9402e1f800ee8a4b59529e4b2",
    23: "9d8694d39e05d9d2963fccbe4f71e2af04c72c13de45f5dcbe775a586f214567",
}
RATES = (0.1, 0.3, 0.5)
SEEDS = (0, 1, 2)
MAX_ROWS = 2_000
# job method -> the result methods it writes (one TabSil evaluation gives the mean and the median)
METHODS = {
    "tabsil": ("tabsil", "tabsil_median"),
    "tabsil_inferred": ("tabsil_inferred",),
    "tabpfn_impute": ("tabpfn_impute",),
    "tabpfn_simple": ("tabpfn_simple",),
    "mean": ("mean",),
    "knn": ("knn",),
    "mice": ("mice",),
    "missforest": ("missforest",),
}
GPU_METHODS = ("tabsil", "tabsil_inferred", "tabpfn_impute", "tabpfn_simple")
OURS = "tabsil"
OFFICIAL = "tabpfn_impute"
# PriorLabs/tabpfn-extensions commit of the baselines (the 0.6.3 release fails `impute` on CUDA:
# its draws are written to the CPU table without `.to(device)`; a049fd7 has that fix)
TABPFN_EXTENSIONS_VERSION = "a049fd7"
RESULT_COLUMNS = [
    "dataset", "rate", "seed", "method", "nrmse", "accuracy", "seconds", "n_rows", "n_numeric",
    "n_categorical", "masked_numeric", "masked_categorical", "columns", "opts",
]  # fmt: skip


def cache_dir() -> Path:
    return Path(os.environ.get("TABSIL_IMPUTE_DIR", Path.home() / ".cache" / "tabsil" / "impute"))


def arff_path(name: str, root: Path | None = None) -> Path:
    """The cached ARFF file of table `name`, downloaded from OpenML and checked against its
    sha256 the first time (`_files.download_verified`: a download with another hash is not
    cached)."""
    dataset_id, file_id, file_name = TABLES[name]
    return download_verified(
        f"{OPENML_URL}/{file_id}/{file_name}.arff",
        (root or cache_dir()) / f"{name}.arff",
        SHA256[dataset_id],
        what=f"the pinned OpenML file of {name}",
    )


def parse_arff(text: str) -> tuple[np.ndarray, list[int], list[str]]:
    """(values (rows, columns) float64 with categorical codes, categorical column indices, column
    names) of an ARFF file. A nominal attribute's code is its value's position in the declared
    list. Rows with a missing value are dropped."""
    import arff  # liac-arff (an openml dependency)

    data = arff.loads(text)
    names, categorical, columns = [], [], []
    rows = data["data"]
    keep = [i for i, r in enumerate(rows) if all(v is not None for v in r)]
    for j, (name, kind) in enumerate(data["attributes"]):
        names.append(name)
        values = [rows[i][j] for i in keep]
        if isinstance(kind, list):
            code = {v: k for k, v in enumerate(kind)}
            categorical.append(j)
            columns.append(np.array([code[v] for v in values], dtype=np.float64))
        elif str(kind).upper() in ("NUMERIC", "REAL", "INTEGER"):
            columns.append(np.asarray(values, dtype=np.float64))
        else:
            raise ValueError(f"attribute {name}: unsupported ARFF type {kind}")
    return np.column_stack(columns), categorical, names


def load(name: str, root: Path | None = None) -> tuple[np.ndarray, list[int], list[str]]:
    return parse_arff(arff_path(name, root).read_text())


def table_key(name: str) -> int:
    return zlib.crc32(name.encode())


def make_split(name: str, X: np.ndarray, rate: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """(the rows of the split (at most MAX_ROWS), their MCAR mask (True = masked))."""
    key = table_key(name)
    if len(X) > MAX_ROWS:
        rows = np.random.default_rng([seed, key]).choice(len(X), MAX_ROWS, replace=False)
        X = X[np.sort(rows)]
    mask = np.random.default_rng([seed, round(rate * 100), key, 1]).random(X.shape) < rate
    if (mask.sum(0) > len(X) - 2).any():
        raise ValueError(f"{name} rate {rate} seed {seed}: a column keeps fewer than 2 values")
    return X, mask


# --- metrics --------------------------------------------------------------------------------


def score(X_true: np.ndarray, X_imp: np.ndarray, mask: np.ndarray, categorical: list[int]) -> dict:
    """nrmse (numeric columns: RMSE / std, averaged) and accuracy (categorical columns: share of
    masked cells with the true class, averaged), on the masked cells, and `columns`, the value of
    each column as JSON ({column index: nrmse or accuracy}). Categorical imputations are set to
    the nearest class observed in the column first."""
    nrmse, accuracy, columns = [], [], {}
    masked_numeric = masked_categorical = 0
    for j in range(X_true.shape[1]):
        m = mask[:, j]
        if not m.any():
            continue
        truth, guess = X_true[m, j], X_imp[m, j]
        if j in categorical:
            classes = np.unique(X_true[~m, j])
            accuracy.append(float((nearest_class(guess, classes) == truth).mean()))
            columns[j] = round(accuracy[-1], 5)
            masked_categorical += int(m.sum())
        else:
            std = X_true[:, j].std()
            if std > 0:
                nrmse.append(float(np.sqrt(np.mean((guess - truth) ** 2)) / std))
                columns[j] = round(nrmse[-1], 5)
                masked_numeric += int(m.sum())
    return {
        "nrmse": float(np.mean(nrmse)) if nrmse else np.nan,
        "accuracy": float(np.mean(accuracy)) if accuracy else np.nan,
        "masked_numeric": masked_numeric,
        "masked_categorical": masked_categorical,
        "columns": json.dumps(columns),
    }


# --- methods --------------------------------------------------------------------------------


def _encode(X: np.ndarray, categorical: list[int]):
    """z-scored numeric columns and one-hot categorical columns (a missing cell is missing in
    every one-hot column), and the inverse that takes a category as the largest one-hot value."""
    numeric = [j for j in range(X.shape[1]) if j not in categorical]
    mean, std = np.nanmean(X[:, numeric], 0), np.nanstd(X[:, numeric], 0)
    std = np.where(std > 0, std, 1.0)
    parts, blocks = [(X[:, numeric] - mean) / std], []
    start = len(numeric)
    for j in categorical:
        classes = np.unique(X[~np.isnan(X[:, j]), j])
        onehot = (X[:, j, None] == classes[None]).astype(float)
        onehot[np.isnan(X[:, j])] = np.nan
        parts.append(onehot)
        blocks.append((j, classes, slice(start, start + len(classes))))
        start += len(classes)

    def decode(Z: np.ndarray) -> np.ndarray:
        out = np.empty_like(X)
        out[:, numeric] = Z[:, : len(numeric)] * std + mean
        for j, classes, cols in blocks:
            out[:, j] = classes[Z[:, cols].argmax(1)]
        return out

    return np.hstack(parts), decode


def impute_mean(X: np.ndarray, categorical: list[int], seed: int) -> np.ndarray:
    out = X.copy()
    for j in range(X.shape[1]):
        col, miss = X[:, j], np.isnan(X[:, j])
        if j in categorical:
            values, counts = np.unique(col[~miss], return_counts=True)
            out[miss, j] = values[counts.argmax()]
        else:
            out[miss, j] = col[~miss].mean()
    return out


def impute_knn(X: np.ndarray, categorical: list[int], seed: int) -> np.ndarray:
    from sklearn.impute import KNNImputer

    Z, decode = _encode(X, categorical)
    return decode(KNNImputer(n_neighbors=5).fit_transform(Z))


def _linear_features(current: np.ndarray, columns: list[int], categorical: list[int]):
    """Features of a linear model from `columns` of the current imputation: numeric columns
    z-scored, categorical columns one-hot (all their classes)."""
    parts = []
    for k in columns:
        col = current[:, k]
        if k in categorical:
            parts.append((col[:, None] == np.unique(col)[None]).astype(float))
        else:
            parts.append(((col - col.mean()) / (col.std() or 1.0))[:, None])
    return np.hstack(parts) if parts else np.zeros((len(current), 1))


def impute_mice(X: np.ndarray, categorical: list[int], seed: int, max_iter: int = 10) -> np.ndarray:
    """MICE by chained equations without posterior draws (scikit-learn `IterativeImputer`'s
    defaults: mean / most frequent start, 10 rounds, columns with fewer missing cells first): each
    round imputes every column with a missing cell from the other columns' current imputation
    (numeric z-scored, categorical one-hot), with BayesianRidge for a numeric column and a
    multinomial LogisticRegression for a categorical one. A column is never predicted from its
    own one-hot block: on one-hot columns, `IterativeImputer` predicts each from its siblings,
    which are missing in the same rows, and every category stays at the column means (the most
    frequent class)."""
    from sklearn.linear_model import BayesianRidge, LogisticRegression

    miss = np.isnan(X)
    current = impute_mean(X, categorical, seed)
    order = [j for j in np.argsort(miss.sum(0), kind="stable") if miss[:, j].any()]
    for _ in range(max_iter):
        for j in order:
            others = [k for k in range(X.shape[1]) if k != j]
            F = _linear_features(current, others, categorical)
            obs = ~miss[:, j]
            y = X[obs, j]
            if j in categorical:
                if len(np.unique(y)) == 1:
                    current[miss[:, j], j] = y[0]
                    continue
                model = LogisticRegression(max_iter=1000, random_state=seed)
            else:
                model = BayesianRidge()
            model.fit(F[obs], y)
            current[miss[:, j], j] = model.predict(F[miss[:, j]])
    return current


def n_jobs() -> int:
    """Parallel jobs of MissForest's forest fits: `$TABSIL_IMPUTE_N_JOBS` (set per worker by
    `eval.py --suite impute --shards`), else -1 (every core). The imputation is the same for
    every value."""
    return int(os.environ.get("TABSIL_IMPUTE_N_JOBS", -1))


def impute_missforest(
    X: np.ndarray, categorical: list[int], seed: int, n_trees: int = 100, max_iter: int = 10
) -> np.ndarray:
    """MissForest (Stekhoven and Buehlmann 2012, the R package's defaults): start from the mean /
    most frequent value, then per round impute each column (fewest missing first) with a random
    forest (100 trees, sqrt(columns) per split; a classifier for categorical columns) fitted on
    its observed rows with the current imputation of the other columns; stop when the change of
    both the numeric and the categorical part grows (or after 10 rounds) and return the
    imputation before that round (`missForest`'s `stopCriterion`: go on while the change of a
    kind present in the table shrinks). The numeric change is sum((new - old)^2) / sum(new^2) over
    the numeric columns, the categorical change the share of changed missing categorical cells.
    Categorical features enter the forests as codes."""
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    miss = np.isnan(X)
    current = impute_mean(X, categorical, seed)
    order = [j for j in np.argsort(miss.sum(0), kind="stable") if miss[:, j].any()]
    numeric = [j for j in range(X.shape[1]) if j not in categorical]
    kinds = [cols for cols in (numeric, list(categorical)) if miss[:, cols].any()]
    previous_change = [np.inf] * len(kinds)
    for _ in range(max_iter):
        previous = current.copy()
        for j in order:
            others = [k for k in range(X.shape[1]) if k != j]
            kind = RandomForestClassifier if j in categorical else RandomForestRegressor
            forest = kind(
                n_estimators=n_trees, max_features="sqrt", random_state=seed, n_jobs=n_jobs()
            )
            forest.fit(current[~miss[:, j]][:, others], X[~miss[:, j], j])
            # predict on one thread: with several, scikit-learn adds the trees' outputs in the
            # order the threads finish, so the last bits differ from call to call, and with them
            # the round at which the loop stops
            forest.set_params(n_jobs=1)
            current[miss[:, j], j] = forest.predict(current[miss[:, j]][:, others])
        change = []
        for cols in kinds:
            new, old = current[:, cols], previous[:, cols]
            if cols is numeric:
                change.append(float(np.sum((new - old) ** 2) / max(np.sum(new**2), 1e-300)))
            else:
                change.append(float(np.mean(new[miss[:, cols]] != old[miss[:, cols]])))
        if not any(c < p for c, p in zip(change, previous_change, strict=True)):
            return previous
        previous_change = change
    return current


def tabpfn_models(device: str):
    """TabPFN v2 classifier and regressor with their defaults (tabpfn `ModelVersion.V2`).
    TabPFN's usage telemetry is turned off."""
    os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")
    from tabpfn import TabPFNClassifier, TabPFNRegressor
    from tabpfn.constants import ModelVersion

    clf = TabPFNClassifier.create_default_for_version(ModelVersion.V2, device=device)
    reg = TabPFNRegressor.create_default_for_version(ModelVersion.V2, device=device)
    return clf, reg


def official_model_class():
    """`TabPFNUnsupervisedModel` with one fix: `use_classifier_` counts a column's classes over
    its observed values. At a049fd7 `impute_` and `sample_from_model_prediction_` pass the
    fit column with its missing values, and `torch.unique` counts every NaN as a value, so a
    categorical column missing in the fit rows is routed to the regression code with the
    classifier and `impute` raises (`predict() got an unexpected keyword argument
    'output_type'`). `density_` already drops the missing values first; the fix does the same in
    the other two calls, nothing else changes."""
    import torch
    from tabpfn_extensions.unsupervised import TabPFNUnsupervisedModel

    class Official(TabPFNUnsupervisedModel):
        def use_classifier_(self, column_idx, y):
            if torch.is_tensor(y):
                y = y[~torch.isnan(y)] if y.is_floating_point() else y
            else:
                y = np.asarray(y)
                y = y[~np.isnan(y)] if np.issubdtype(y.dtype, np.floating) else y
            return super().use_classifier_(column_idx, y)

    return Official


def tabpfn_imputer(device: str, simple: bool) -> Callable:
    """The official extension: `TabPFNUnsupervisedModel(clf, reg).fit(X).impute(X)` with its
    defaults and the given categorical columns (`official_model_class`: with the missing-value
    fix of `use_classifier_`), or `simple_impute(X, clf, reg, categorical_features)`. Both add the
    numeric columns they infer as categorical (at most 40 values, more than 10 rows per value);
    such columns are still scored as numeric. `seed` seeds the two global generators the official
    code draws from: torch's (its samples) and Python's `random` (its column orders)."""

    def impute(X: np.ndarray, categorical: list[int], seed: int) -> np.ndarray:
        import torch

        os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")
        from tabpfn_extensions.unsupervised import simple_impute

        torch.manual_seed(seed)
        random.seed(seed)
        clf, reg = tabpfn_models(device)
        if simple:
            return np.asarray(simple_impute(X, clf, reg, categorical_features=list(categorical)))
        model = official_model_class()(tabpfn_clf=clf, tabpfn_reg=reg)
        model.set_categorical_features(list(categorical))
        out = model.fit(X).impute(X)
        return out.cpu().numpy().astype(np.float64)

    return impute


def extension_categorical(X: np.ndarray, categorical: list[int]) -> list[int]:
    """The columns the TabPFN extension treats as categorical on `X` (a049fd7
    `infer_categorical_features` on a numpy array): the given ones and every column with at most
    40 distinct values (missing counted as one, `np.unique`) and more than 10 rows per value."""
    out = []
    for j in range(X.shape[1]):
        n_unique = len(np.unique(X[:, j]))
        if j in categorical or (0 < n_unique <= 40 and X.shape[0] / n_unique > 10):
            out.append(j)
    return out


def tabsil_imputer(
    model, borders, n_estimators: int | None, device: str, seed_offset: int = 0,
    extension_kinds: bool = False,
):  # fmt: skip
    """`TabSilUnsupervised(categorical_features=...).fit(X)`, then one `missing_conditionals(X)`
    filled two ways: {"tabsil": mean, "tabsil_median": median}. random_state = seed +
    `seed_offset`. With `extension_kinds`, the categorical columns are those the TabPFN extension
    uses (`extension_categorical`) and the result is {"tabsil_inferred": mean}; scoring still
    follows the table's kinds."""
    from tabsil.tasks.unsupervised import TabSilUnsupervised

    def impute(X: np.ndarray, categorical: list[int], seed: int) -> dict[str, np.ndarray]:
        kinds = extension_categorical(X, categorical) if extension_kinds else list(categorical)
        est = TabSilUnsupervised(
            model, borders, categorical_features=kinds, n_estimators=n_estimators,
            random_state=seed + seed_offset, device=device,
        ).fit(X)  # fmt: skip
        conds = est.missing_conditionals(X)
        if extension_kinds:
            return {"tabsil_inferred": est.impute_from(X, conds, "mean")}
        return {
            "tabsil": est.impute_from(X, conds, "mean"),
            "tabsil_median": est.impute_from(X, conds, "median"),
        }

    return impute


CPU_IMPUTERS = {
    "mean": impute_mean,
    "knn": impute_knn,
    "mice": impute_mice,
    "missforest": impute_missforest,
}


def check_methods(methods: list[str]) -> list[str]:
    if bad := [m for m in methods if m not in METHODS]:
        raise ValueError(f"unknown methods {bad} (known: {list(METHODS)})")
    return methods


def job_cost(name: str, method: str) -> float:
    """Relative cost of a job, to start the expensive ones first: the official extension fits
    about 10 x columns^2 models, the others about one per column."""
    columns = n_columns(name)
    return columns**2 * 10 if method == OFFICIAL else float(columns)


@functools.cache
def n_columns(name: str) -> int:
    return len(load(name)[2])


# --- run ------------------------------------------------------------------------------------


def finished_keys_list(rows: pd.DataFrame) -> list[tuple]:
    """(table, rate rounded to 4 digits, seed, method) of each result row, in order. The rate is
    cast first: the frame of a missing results file has object columns, which pandas 2 does not
    round."""
    rates = rows.rate.astype(float).round(4)
    return list(zip(rows.dataset, rates, rows.seed.astype(int), rows.method, strict=True))


def finished_keys(rows: pd.DataFrame) -> set[tuple]:
    """The set of `finished_keys_list`."""
    return set(finished_keys_list(rows))


def new_rows(rows: pd.DataFrame, existing: pd.DataFrame) -> pd.DataFrame:
    """The rows of `rows` whose (table, rate, seed, method) is not in `existing` (to merge a
    worker's results without repeating rows an earlier run merged)."""
    merged = finished_keys(existing)
    new = np.array([k not in merged for k in finished_keys_list(rows)], dtype=bool)
    return rows[new]  # a boolean mask, also when `rows` is empty (the columns stay)


def read_results(path: Path) -> pd.DataFrame:
    return _files.read_results(path, RESULT_COLUMNS)


def row_settings(methods: list[str], opts: dict[str, str] | None) -> dict[str, str]:
    """Result method -> the settings string (`opts`) of its rows: its job method's
    (`tabsil_median` rows are written by the `tabsil` job)."""
    return {out: (opts or {}).get(m, "") for m in methods for out in METHODS[m]}


def run(
    imputers: dict[str, Callable],
    datasets: list[str],
    rates: list[float],
    seeds: list[int],
    methods: list[str],
    out_dir: Path,
    opts: dict[str, str] | None = None,
    log: Callable[[str], None] = print,
) -> tuple[pd.DataFrame, list[dict]]:
    """Impute every (table, rate, seed, job method) whose rows are not all in
    `out_dir/results.csv`, appending rows as each finishes. `imputers[method](X_masked,
    categorical, seed)` -> the filled table, or {result method: filled table}. `opts[method]` is
    the settings string written with its rows; rows of `methods` already there with another
    string stop the run (`_files.check_settings`). Returns (all rows, failures)."""
    results = _files.Results(out_dir, RESULT_COLUMNS, row_settings(methods, opts), "impute", log)
    finished = finished_keys(results.done)
    for dataset in datasets:
        X_all, categorical, _ = load(dataset)
        for rate in rates:
            for seed in seeds:
                X, mask = make_split(dataset, X_all, rate, seed)
                X_masked = np.where(mask, np.nan, X)
                for method in methods:
                    outputs = METHODS[method]
                    if all((dataset, round(rate, 4), seed, m) in finished for m in outputs):
                        continue
                    t0 = time.time()
                    try:
                        result = imputers[method](X_masked.copy(), list(categorical), seed)
                        if not isinstance(result, dict):
                            result = {method: result}
                        if set(result) != set(outputs):
                            raise ValueError(f"results {sorted(result)}, expected {outputs}")
                        for m, X_imp in result.items():
                            X_imp = np.asarray(X_imp, dtype=np.float64)
                            if X_imp.shape != X.shape or not np.isfinite(X_imp[mask]).all():
                                raise ValueError(
                                    f"{m}: shape {X_imp.shape} with "
                                    f"{int((~np.isfinite(X_imp[mask])).sum())} non-finite "
                                    "masked cells"
                                )
                    except Exception as e:  # noqa: BLE001 - record and go on; the run fails
                        results.fail(dict(dataset=dataset, rate=rate, seed=seed, method=method), e)
                        continue
                    seconds = round(time.time() - t0, 1)
                    rows = []
                    for m, X_imp in result.items():
                        s = score(X, np.asarray(X_imp, dtype=np.float64), mask, categorical)
                        rows.append(
                            dict(dataset=dataset, rate=rate, seed=seed, method=m, **s,
                                 seconds=seconds, n_rows=len(X),
                                 n_numeric=X.shape[1] - len(categorical),
                                 n_categorical=len(categorical), opts=(opts or {}).get(method, ""))
                        )  # fmt: skip
                        log(
                            f"[impute] {dataset} {rate} {seed} {m}: nrmse {s['nrmse']:.4f} "
                            f"accuracy {s['accuracy']:.4f} ({seconds:.0f}s)"
                        )
                    results.append(rows)
    return results.finish()


# --- report ---------------------------------------------------------------------------------


def result_methods(methods: list[str]) -> list[str]:
    """The result methods the job methods write (`tabsil` also writes `tabsil_median`)."""
    return [out for m in methods for out in METHODS[m]]


def in_grid(
    rows: pd.DataFrame, datasets: list[str], rates: list[float], seeds: list[int],
    methods: list[str] | None = None,
) -> pd.DataFrame:  # fmt: skip
    """The rows of the (table, rate, seed) grid and, if given, of the result `methods` (a
    results file can hold rows of earlier runs outside them)."""
    keep = (
        rows.dataset.isin(datasets)
        & rows.rate.astype(float).round(4).isin([round(r, 4) for r in rates])
        & rows.seed.astype(int).isin(seeds)
    )
    if methods is not None:
        keep &= rows.method.isin(methods)
    return rows[keep]


def coverage(
    rows: pd.DataFrame, datasets: list[str], rates: list[float], seeds: list[int],
    methods: list[str] | None = None,
) -> dict[str, float]:  # fmt: skip
    """Per result method (`methods`, default the methods in `rows`), the share of (table, rate,
    seed) units without a result: 0 when the method ran everywhere, 1 when it has no row at all
    (nothing is filled in for a missing unit)."""
    units = len(datasets) * len(rates) * len(seeds)
    want = {(d, round(r, 4), s) for d in datasets for r in rates for s in seeds}
    out = {}
    for m in sorted(set(rows.method) if methods is None else methods):
        have = {(d, r, s) for d, r, s, mm in finished_keys(rows) if mm == m}
        out[m] = 1 - len(have & want) / units
    return out


def per_table(rows: pd.DataFrame, metric: str) -> pd.DataFrame:
    """(table, rate) x method: the seed mean of `metric`."""
    g = rows.dropna(subset=[metric])
    return g.pivot_table(index=["dataset", "rate"], columns="method", values=metric, aggfunc="mean")


def summary_table(rows: pd.DataFrame) -> pd.DataFrame:
    """Per rate and method: the table mean of the seed-mean nrmse and accuracy (tables with that
    kind of column), and the average rank over (table, seed) units (1 = best; nrmse ascending,
    accuracy descending)."""
    out = []
    for metric, ascending in (("nrmse", True), ("accuracy", False)):
        g = rows.dropna(subset=[metric])
        means = per_table(g, metric).groupby(level="rate").mean()
        wide = g.pivot_table(index=["dataset", "rate", "seed"], columns="method", values=metric)
        ranks = wide.rank(axis=1, ascending=ascending).groupby(level="rate").mean()
        for rate in means.index:
            for m in means.columns:
                out.append(dict(rate=rate, method=m, metric=metric, mean=means.loc[rate, m],
                                rank=ranks.loc[rate, m],
                                tables=int(per_table(g, metric).loc[(slice(None), rate), m]
                                           .notna().sum())))  # fmt: skip
    return pd.DataFrame(out)


def report(
    rows: pd.DataFrame, out_dir: Path, datasets: list[str], rates: list[float], seeds: list[int],
    methods: list[str] | None = None,
) -> dict:  # fmt: skip
    """Write `summary.csv` (per rate), `per_table_nrmse.csv` and `per_table_accuracy.csv` of the
    rows in the (table, rate, seed) grid and the result `methods` (default all in `rows`), and
    return the main numbers with each method's share of missing units."""
    rows = in_grid(rows, datasets, rates, seeds, methods)
    out: dict = {"missing_units": coverage(rows, datasets, rates, seeds, methods)}
    if not len(rows):
        return out
    summary = summary_table(rows)
    summary.to_csv(out_dir / "summary.csv", index=False)
    per_table(rows, "nrmse").round(5).to_csv(out_dir / "per_table_nrmse.csv")
    per_table(rows, "accuracy").round(5).to_csv(out_dir / "per_table_accuracy.csv")
    for metric in ("nrmse", "accuracy"):
        g = summary[summary.metric == metric]
        out[metric] = {
            f"{r:g}": g[g.rate == r].set_index("method")["mean"].round(4).to_dict()
            for r in sorted(g.rate.unique())
        }
    return out
