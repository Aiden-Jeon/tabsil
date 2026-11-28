"""Real tables as training data: the OpenML corpus of item 1.2 of the v2 top-5 plan
(`training/docs/v2-top5-plan.md`), for the short continuation of item 2.2.

A dataset is a random subsample of one corpus table with the table's own target, given to the
model as inference gives it a table (`tabsil.inference.ensemble`): the predictors are encoded on
the train rows (`to_numeric`: categories seen in the train rows become codes, others missing) and
go through the preprocessing of one ensemble member (`MemberPreprocessor`: the outlier clip,
one-hot blocks, a random member kind of `MEMBER_KINDS`, category counts, the sample identifier and
the column shuffle), fitted on the train rows. The model never sees a real table any other way.

- **Tables.** The classification prior draws the corpus's binary tables with probability
  `binary_prob` (the training config's) and its multiclass tables otherwise; the regression prior
  draws its regression tables. Tables are drawn uniformly within their kind.
- **Sizes.** The rows of a batch are drawn as the graph SCM prior draws them (`min_rows` to
  `max_rows`, log-uniform with `log_uniform_rows`), capped by the smallest table of the batch; the
  train rows are `min_train_fraction` to `max_train_fraction` of them. Rows are a random subset of
  the table's rows (those with a target). A member wider than `max_features` keeps a random
  `max_features` of its columns; in a batch of several tables every table keeps a random subset of
  the narrowest width.
- **Targets.** Classification: the classes of the train rows, in a random order (as the members'
  class permutations); an eval row of a class the train rows lack gets `IGNORE_INDEX`. A subsample
  whose train rows hold one class is drawn again. Regression: standardized over all rows, or over
  the train rows with `target_train_rows`, as the graph SCM prior does.
- **The corpus** is the MLflow artifact `CORPUS_URI`, pinned by the digest of its manifest
  (`MANIFEST_SHA256`; the manifest holds each table's digest, checked on load). It is downloaded
  once per machine into a folder named by that digest, under a file lock, so the processes of one
  node share one copy. The name `real_tables` therefore fixes the data, as every registered prior
  does.

The regression borders are the start checkpoint's: this prior continues a model (`init_from`) and
has no borders of its own (`make_borders` raises).
"""

from __future__ import annotations

import fcntl
import hashlib
import math
import os
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from tabsil.inference.preprocessing import (
    MEMBER_KINDS,
    Member,
    MemberPreprocessor,
    apply_encoder,
    row_hashes,
    row_identifier,
    to_numeric,
)
from tabsil_train.priors.base import IGNORE_INDEX, PriorBatch

# The corpus of item 1.2 (training/scripts/real_corpus.py), logged to MLflow, and its manifest's
# SHA-256
# (MLflow run `tabsil-real-corpus` in the `tabsil` experiment: 79 tables, binary 15, multiclass 12,
# regression 52; built by training/scripts/real_corpus.py at 1ee94c9)
CORPUS_URI = "runs:/90572b03b0e341deb11f15c447343676/corpus"
MANIFEST_SHA256 = "fb5ba33b345c769859ae4bbe39bbe13b0be0172bd658960867d16d1f82d6a1ec"
MAX_CLASSES = 10  # the corpus keeps tables of 2 to 10 classes (the graph SCM prior's maximum)
CATEGORIES = "counts"  # many-category columns as at inference (`--categories counts`)
OPTIONS = {
    "corpus": CORPUS_URI,
    "manifest_sha256": MANIFEST_SHA256,
    "member_kinds": list(MEMBER_KINDS),
    "categories": CATEGORIES,
}
TASK_TYPES = {"classification": ("binary", "multiclass"), "regression": ("regression",)}
# attempts at a usable subsample (two train classes, a regression target that varies)
MAX_ATTEMPTS = 50


@dataclass
class Table:
    name: str
    X: pd.DataFrame  # the predictors: numeric columns, and strings for categorical ones
    y: np.ndarray  # class labels as strings, or float regression targets


def corpus_folder() -> Path:
    """The local corpus folder: `CORPUS_URI` downloaded once per machine (module docstring)."""
    from tabsil_train.hub import download_artifact

    root = Path(os.environ.get("TMPDIR", "/tmp")) / "tabsil-real-tables" / MANIFEST_SHA256[:16]
    root.mkdir(parents=True, exist_ok=True)
    with open(root / "lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return download_artifact(CORPUS_URI, root)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_tables(
    folder: Path, manifest_sha256: str, kinds: tuple[str, ...] | None = None
) -> dict[str, list[Table]]:
    """The corpus tables by task type (`binary`, `multiclass`, `regression`; only `kinds` if
    given), every file read checked against its digest. ValueError if a digest differs."""
    manifest_path = folder / "manifest.csv"
    if _sha256(manifest_path) != manifest_sha256:
        raise ValueError(f"{manifest_path} is not the pinned corpus manifest")
    tables: dict[str, list[Table]] = {kind: [] for kinds in TASK_TYPES.values() for kind in kinds}
    for row in pd.read_csv(manifest_path).itertuples():
        if kinds is not None and row.task_type not in kinds:
            continue
        path = folder / row.file
        if _sha256(path) != row.sha256:
            raise ValueError(f"{path} differs from the manifest")
        frame = pd.read_parquet(path)
        target = frame[row.target]
        keep = target.notna().to_numpy(copy=True)
        if row.task_type == "regression":
            y = pd.to_numeric(target[keep], errors="coerce").to_numpy(np.float64)
            keep[np.flatnonzero(keep)[~np.isfinite(y)]] = False
            y = y[np.isfinite(y)]
        else:
            y = target[keep].astype(str).to_numpy()
        X = frame.drop(columns=[row.target])[keep].reset_index(drop=True)
        tables[row.task_type].append(Table(f"{row.openml_id} {row.name}", X, y))
    return tables


_TABLES: dict[tuple[str, tuple[str, ...]], dict[str, list[Table]]] = {}


def corpus_tables(kinds: tuple[str, ...]) -> dict[str, list[Table]]:
    """`load_tables` of the pinned corpus for the task types `kinds`, once per process (a data
    worker draws one task, so it holds that task's tables only)."""
    key = (MANIFEST_SHA256, kinds)
    if key not in _TABLES:
        _TABLES[key] = load_tables(corpus_folder(), MANIFEST_SHA256, kinds)
    return _TABLES[key]


class RealTablesPrior:
    """The `task` prior on corpus tables (module docstring). `sample_batch(batch_size)` gives a
    `PriorBatch` of `batch_size` datasets of one shape. `tables` is the corpus by task type; None
    reads the pinned corpus (`corpus_tables`) at the first batch, so that building the prior
    downloads nothing."""

    def __init__(
        self,
        task: str,
        tables: dict[str, list[Table]] | None,
        seed: int,
        *,
        min_rows: int,
        max_rows: int,
        log_uniform_rows: bool,
        max_features: int,
        max_cells: int,
        min_train_fraction: float,
        max_train_fraction: float,
        binary_prob: float,
        target_train_rows: bool,
    ) -> None:
        if task not in TASK_TYPES:
            raise ValueError(f"task is one of {tuple(TASK_TYPES)}: {task}")
        self.task = task
        self.tables = tables
        self._kinds: list[tuple[str, list[Table]]] | None = None
        self.rng = np.random.default_rng(seed)
        self.min_rows, self.max_rows = min_rows, max_rows
        self.log_uniform_rows = log_uniform_rows
        self.max_features, self.max_cells = max_features, max_cells
        self.min_train_fraction, self.max_train_fraction = min_train_fraction, max_train_fraction
        self.binary_prob = binary_prob
        self.target_train_rows = target_train_rows

    @property
    def kinds(self) -> list[tuple[str, list[Table]]]:
        """(task type, tables) of this task's types that have tables, in `TASK_TYPES` order."""
        if self._kinds is None:
            types = TASK_TYPES[self.task]
            tables = corpus_tables(types) if self.tables is None else self.tables
            kinds = [(kind, tables.get(kind, [])) for kind in types]
            self._kinds = [(kind, group) for kind, group in kinds if group]
            if not self._kinds:
                raise ValueError(f"the corpus has no {self.task} table")
        return self._kinds

    def _table(self) -> Table:
        group = self.kinds[0][1]
        if len(self.kinds) == 2 and self.rng.random() >= self.binary_prob:
            group = self.kinds[1][1]  # multiclass (TASK_TYPES order)
        return group[int(self.rng.integers(len(group)))]

    def _rows(self, available: int) -> tuple[int, int]:
        low, high = self.min_rows, self.max_rows
        if self.log_uniform_rows:
            n_rows = round(math.exp(self.rng.uniform(math.log(low), math.log(high))))
        else:
            n_rows = int(self.rng.integers(low, high + 1))
        n_rows = max(2, min(n_rows, available))
        fraction = self.rng.uniform(self.min_train_fraction, self.max_train_fraction)
        n_train = min(max(1, round(n_rows * fraction)), n_rows - 1)
        return n_rows, n_train

    def _targets(self, table: Table, train: np.ndarray, rows: np.ndarray) -> np.ndarray | None:
        """The model targets of `rows` (train rows first), or None if the subsample is unusable."""
        if self.task == "regression":
            y = table.y[rows]
            ref = table.y[train] if self.target_train_rows else y
            std = ref.std()
            return None if std < 1e-12 else (y - ref.mean()) / std
        classes = np.unique(table.y[train])
        if len(classes) < 2 or len(classes) > MAX_CLASSES:
            return None
        order = self.rng.permutation(len(classes))
        labels = table.y[rows]
        index = np.searchsorted(classes, labels)
        known = (index < len(classes)) & (classes[np.minimum(index, len(classes) - 1)] == labels)
        out = np.full(len(rows), float(IGNORE_INDEX))
        out[known] = order[index[known]]
        return out

    def _dataset(self, table: Table, n_rows: int, n_train: int):
        """(x (rows, columns), y (rows)) of one subsample of `table`, or None if unusable."""
        rows = self.rng.choice(len(table.y), size=n_rows, replace=False)
        train = rows[:n_train]
        y = self._targets(table, train, rows)
        if y is None:
            return None
        x_train, cat_ix, encoder = to_numeric(table.X.iloc[train], None)
        x_eval = apply_encoder(table.X.iloc[rows[n_train:]], cat_ix, encoder)
        member = Member(
            class_permutation=None,
            target_power=False,
            seed=int(self.rng.integers(0, 2**31 - 1)),
            kind=str(self.rng.choice(MEMBER_KINDS)),
            categories=CATEGORIES,
        )
        prep = MemberPreprocessor(member, cat_ix, self.max_features).fit(x_train)
        identifier = np.random.default_rng(member.seed + 2).standard_normal((n_train, 1))
        x = np.concatenate(
            [
                prep.transform(x_train, identifier),
                prep.transform(x_eval, row_identifier(row_hashes(x_eval), member.seed + 3)),
            ]
        )
        return self._columns(x, self.max_features), y

    def _columns(self, x: np.ndarray, width: int) -> np.ndarray:
        if x.shape[1] <= width:
            return x
        return x[:, np.sort(self.rng.choice(x.shape[1], size=width, replace=False))]

    def sample_batch(self, batch_size: int) -> PriorBatch:
        for _ in range(MAX_ATTEMPTS):
            tables = [self._table() for _ in range(batch_size)]
            n_rows, n_train = self._rows(min(len(t.y) for t in tables))
            with warnings.catch_warnings():
                # empty columns of a subsample warn in the column statistics; inference ignores them
                warnings.simplefilter("ignore", RuntimeWarning)
                datasets = [self._dataset(t, n_rows, n_train) for t in tables]
            if any(d is None for d in datasets):
                continue
            width = min(x.shape[1] for x, _ in datasets)
            # the cell cap of the training config (rows are cut, not columns, as a table has few)
            n_rows = min(n_rows, max(n_train + 1, self.max_cells // max(1, width)))
            x = np.stack([self._columns(x, width)[:n_rows] for x, _ in datasets])
            y = np.stack([y[:n_rows] for _, y in datasets])
            return PriorBatch(
                torch.as_tensor(x, dtype=torch.float32),
                torch.as_tensor(y, dtype=torch.float32),
                n_train,
            )
        raise RuntimeError(f"no usable {self.task} subsample after {MAX_ATTEMPTS} attempts")


def check_config(config) -> None:
    """Whether the size settings of the training config can be drawn (ValueError otherwise)."""
    if not 2 <= config.min_rows <= config.max_rows:
        raise ValueError(f"real_tables needs 2 <= min_rows <= max_rows: {config.min_rows}")
    if not 0 < config.min_train_fraction <= config.max_train_fraction < 1:
        raise ValueError("real_tables needs 0 < min_train_fraction <= max_train_fraction < 1")
    if config.max_features < 2:
        raise ValueError(f"real_tables needs max_features >= 2: {config.max_features}")


def make_prior(config, task: str, seed: int) -> RealTablesPrior:
    """The `task` prior on the pinned corpus with the size settings of the training config."""
    return RealTablesPrior(
        task,
        None,
        seed,
        min_rows=config.min_rows,
        max_rows=config.max_rows,
        log_uniform_rows=config.log_uniform_rows,
        max_features=config.max_features,
        max_cells=config.max_cells,
        min_train_fraction=config.min_train_fraction,
        max_train_fraction=config.max_train_fraction,
        binary_prob=config.binary_prob,
        target_train_rows=config.target_train_rows,
    )


def make_borders(config, n_buckets: int, seed: int, n_jobs: int = 1) -> torch.Tensor:
    raise ValueError(
        "real_tables continues a model: its regression borders are the start checkpoint's"
        " (train from --init-run-id or --init-hub)"
    )
