"""Task-shaped tables: an ordinary prior table rearranged so that its train and eval rows stand in
the relation of a task's context and queries (`training/docs/v2-task-formulations.md` sections 3
and 4, `training/docs/v2-prior-task-plan.md` tier 2). Both are off by default in the graph SCM
prior.

- `SortedSplit` (regression; forecasting): the rows are ordered by one column before the cut, so
  the eval rows lie beyond the train rows in that column, as a forecast horizon lies beyond its
  context in the running index. The target is standardized again with the train rows, as
  inference does, so eval targets can lie outside the train targets' range. With a random
  ordering column they rarely do (in a sample of the probe's tables the median share of eval
  targets beyond the train range was 0), so half of the tables are ordered by the column the
  target follows most: those are the tables with a trend.
- `OutlierTable` (classification; outlier detection): the train rows are data rows labeled 0,
  some of them hidden anomalies, and reference rows labeled 1; the eval rows are data rows with
  their true label (0 inlier, 1 anomaly). The model's answer is then the probability of the
  reference class, and a small dense group among the data rows still has to get it.

`OutlierTable` follows the designs read for it. Contaminated contexts and balanced queries:
TACTIC (arXiv 2603.14171, Section 3: anomalies of 0.05 to 0.3 times the nominal rows, or none;
equal numbers of nominal and anomalous queries) and ICLAD (arXiv 2603.19497, Section 4.1: an
anomaly ratio of U(0, 0.4)). Anomalies as whole classes of an SCM table (`mode`): TACTIC's first
prior and ICLAD's structural anomalies. Perturbed rows (`local`), uniform rows (`global`) and
shifted tight groups (`cluster`): ICLAD's perturbations of selected features, and the three
mechanisms of TACTIC's second prior (local, cluster and global anomalies of a Gaussian mixture,
"inspired by AdBench"); the forms here are this module's, on SCM rows instead of a mixture.
`dependency` (values that occur, in combinations that may not) is this module's addition. The
reference rows as a second class are Hastie, Tibshirani and Friedman, The Elements of
Statistical Learning, section 14.2.4; neither model has them, since both end in an output of
their own.
"""

from __future__ import annotations

import numpy as np
import torch

from tabsil_train.priors.pipeline import Group, Rng
from tabsil_train.priors.transforms.target import MIN_TRAIN_STD

# --- SortedSplit ---------------------------------------------------------------------------
# A column can order the rows if at least this share of its values are distinct
MIN_DISTINCT_SHARE = 0.5
# Eval targets, in train-row standard deviations, are kept within the regression borders
MAX_ABS_TARGET = 20.0

# --- OutlierTable --------------------------------------------------------------------------
ANOMALY_KINDS = ("mode", "local", "cluster", "global", "dependency")
REFERENCE_KINDS = ("uniform", "marginals")
# Share of the train rows that are reference rows (the inference wrapper uses 0.5)
REFERENCE_SHARE = (0.2, 0.5)
# Hidden anomalies among the context's data rows: none with probability CLEAN_PROB, else a share
# of U(0, MAX_CONTAMINATION) (ICLAD's range; the 12 ADBench tables of the suite have 1% to 35%)
CLEAN_PROB = 0.1
MAX_CONTAMINATION = 0.4
# The bounding box of uniform rows is widened by this share of each column's range on each side
PAD = 0.1
# A column with at most this many distinct values is discrete: uniform rows take its values
DISCRETE_LEVELS = 32
# Fewest inlier rows of a context
MIN_CONTEXT_INLIERS = 8
# The nominal classes of a `mode` table hold at least a share of U(low, high) of its rows
NOMINAL_SHARE = (0.6, 0.9)


def ordering_columns(x: np.ndarray) -> np.ndarray:
    """The columns that can order the rows: finite, with at least `MIN_DISTINCT_SHARE` of their
    values distinct."""
    n = len(x)
    keep = [
        j
        for j in range(x.shape[1])
        if np.isfinite(x[:, j]).all() and len(np.unique(x[:, j])) >= MIN_DISTINCT_SHARE * n
    ]
    return np.asarray(keep, dtype=int)


def rank_correlation(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """|Spearman correlation| of each column of x with y (0 for a constant y); ties share their
    mean rank."""
    from scipy.stats import rankdata

    ranks = rankdata(x, axis=0)
    target = rankdata(y)
    ranks -= ranks.mean(0)
    target -= target.mean()
    norm = np.sqrt((ranks**2).sum(0) * (target**2).sum())
    return np.abs(ranks.T @ target) / np.where(norm > 0, norm, np.inf)


def sorted_split(
    x: np.ndarray,
    y: np.ndarray,
    n_train: int,
    r: np.random.Generator,
    index_prob: float,
    target_prob: float = 0.0,
    min_train_std: float = MIN_TRAIN_STD,
) -> tuple[np.ndarray, np.ndarray] | None:
    """One table (x (rows, features), y (rows)) with its rows ordered by one of the columns that
    have enough distinct values (`ordering_columns`), ascending or descending, and y standardized
    with the first `n_train` rows (ddof 0) and clipped at `MAX_ABS_TARGET`. The ordering column
    is, with probability `target_prob`, the one with the largest rank correlation to the target
    (a table with a trend: the eval targets continue beyond the train targets), and otherwise a
    random one (the target need not follow it). With probability `index_prob` the ordering column
    is replaced by the row index, the form of a forecaster's running index. None if the train-row
    sd of y is at most `min_train_std` (the table is drawn again, as `TrainRowTargetScale` does).
    A table without a column to order by is returned as it is."""
    n = len(y)
    columns = ordering_columns(x)
    if len(columns) == 0:
        return x, y
    if r.random() < target_prob:
        column = int(columns[np.argmax(rank_correlation(x[:, columns], y))])
    else:
        column = int(r.choice(columns))
    order = np.argsort(x[:, column], kind="stable")
    if r.random() < 0.5:
        order = order[::-1]
    x, y = x[order].copy(), y[order].astype(np.float64)
    if r.random() < index_prob:
        x[:, column] = np.arange(n)
    mean, std = y[:n_train].mean(), y[:n_train].std()
    if std <= min_train_std:
        return None
    return x, np.clip((y - mean) / std, -MAX_ABS_TARGET, MAX_ABS_TARGET)


class SortedSplit:
    """With probability `prob` per dataset, order its rows by one column and standardize the
    target with the train rows (`sorted_split`): half of the time (`target_prob`) by the column
    the target follows most, and half of the time (`index_prob`) with that column replaced by the
    row index. The group is dropped (drawn again) if a sorted dataset's train-row target sd is
    too small."""

    def __init__(self, prob: float, index_prob: float = 0.5, target_prob: float = 0.5) -> None:
        if not all(0.0 <= p <= 1.0 for p in (prob, index_prob, target_prob)):
            raise ValueError(
                f"probabilities must be between 0 and 1: {prob}, {index_prob}, {target_prob}"
            )
        self.prob = prob
        self.index_prob = index_prob
        self.target_prob = target_prob

    def __call__(self, group: Group, rng: Rng) -> Group | None:
        r = rng.numpy
        xs, ys = [], []
        for d in range(group.y.shape[1]):
            x, y = group.x[:, d], group.y[:, d]
            if r.random() < self.prob:
                out = sorted_split(
                    x.double().cpu().numpy(),
                    y.double().cpu().numpy(),
                    group.n_train,
                    r,
                    self.index_prob,
                    self.target_prob,
                )
                if out is None:
                    return None
                x = torch.as_tensor(out[0], dtype=x.dtype, device=x.device)
                y = torch.as_tensor(out[1], dtype=y.dtype, device=y.device)
            xs.append(x)
            ys.append(y)
        return group.with_tensors(torch.stack(xs, 1), torch.stack(ys, 1))


# ---------------------------------------------------------------------------
# Outlier tables
# ---------------------------------------------------------------------------
def _discrete(x: np.ndarray) -> list[np.ndarray | None]:
    """Per column: its distinct values if it is discrete (`DISCRETE_LEVELS`), else None."""
    out = []
    for j in range(x.shape[1]):
        values = np.unique(x[np.isfinite(x[:, j]), j])
        out.append(values if 0 < len(values) <= DISCRETE_LEVELS else None)
    return out


def uniform_rows(x: np.ndarray, n: int, r: np.random.Generator) -> np.ndarray:
    """n rows uniform in the bounding box of x widened by `PAD`; a discrete column is uniform over
    its values."""
    lo, hi = np.nanmin(x, 0), np.nanmax(x, 0)
    span = hi - lo
    out = r.uniform(lo - PAD * span, hi + PAD * span, size=(n, x.shape[1]))
    for j, values in enumerate(_discrete(x)):
        if values is not None:
            out[:, j] = r.choice(values, n)
    return out


def marginal_rows(x: np.ndarray, n: int, r: np.random.Generator) -> np.ndarray:
    """n rows whose columns are drawn independently from the columns of x: the columns'
    distributions without their dependence."""
    return np.column_stack([x[r.integers(0, len(x), n), j] for j in range(x.shape[1])])


def local_rows(x: np.ndarray, n: int, r: np.random.Generator) -> np.ndarray:
    """n rows of x with a few columns moved: 1 to a third of the columns (the same count for the
    table), each by 2 to 5 of its standard deviations up or down; a discrete column takes
    another of its values instead."""
    f = x.shape[1]
    out = x[r.integers(0, len(x), n)].copy()
    std = np.nanstd(x, 0)
    std = np.where(std > 0, std, 1.0)
    count = int(r.integers(1, max(1, f // 3) + 1))
    discrete = _discrete(x)
    for i in range(n):
        for j in r.permutation(f)[:count]:
            values = discrete[j]
            if values is None:
                out[i, j] += r.choice([-1.0, 1.0]) * r.uniform(2.0, 5.0) * std[j]
            elif len(values) > 1:
                out[i, j] = r.choice(values[values != out[i, j]])
    return out


def cluster_center(x: np.ndarray, r: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """(center, width per column) of a tight group off the bulk of x: 1.5 to 3 standard deviations
    from the mean in every column, 0.05 to 0.3 wide. A discrete column has one value."""
    f = x.shape[1]
    mean, std = np.nanmean(x, 0), np.nanstd(x, 0)
    center = mean + r.choice([-1.0, 1.0], f) * r.uniform(1.5, 3.0, f) * std
    width = r.uniform(0.05, 0.3) * std
    for j, values in enumerate(_discrete(x)):
        if values is not None:
            center[j], width[j] = r.choice(values), 0.0
    return center, width


def dependency_rows(x: np.ndarray, n: int, r: np.random.Generator) -> np.ndarray:
    """n rows of x in which 30% to 100% of the columns (per row) are replaced by other rows'
    values, each column from a row of its own: values that occur, in combinations that may
    not."""
    f = x.shape[1]
    out = x[r.integers(0, len(x), n)].copy()
    replace = r.random((n, f)) < r.uniform(0.3, 1.0, (n, 1))
    donors = marginal_rows(x, n, r)
    return np.where(replace, donors, out)


def nominal_classes(y: np.ndarray, r: np.random.Generator) -> np.ndarray | None:
    """The classes whose rows are the inliers of a `mode` table: classes in random order until
    they hold a share of U(`NOMINAL_SHARE`) of the rows, leaving at least one class out. None with
    fewer than two classes."""
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        return None
    order = r.permutation(len(classes))
    target = r.uniform(*NOMINAL_SHARE) * len(y)
    k = int(np.searchsorted(np.cumsum(counts[order]), target) + 1)
    return classes[order[: min(k, len(classes) - 1)]]


def _counts(
    n_train: int, n_eval: int, inliers: int, anomalies: int | None, r: np.random.Generator
) -> tuple[int, int, int, int] | None:
    """(inlier queries, anomalous queries, context inliers, context anomalies) for a table with
    `inliers` inlier rows and `anomalies` rows to take anomalies from (None: as many as wanted).
    Half of the queries are anomalies; the rest of the train rows are reference rows. None if the
    context would have fewer than `MIN_CONTEXT_INLIERS` inliers or a query kind would be empty."""
    q_out = n_eval // 2 if anomalies is None else min(n_eval // 2, anomalies)
    q_in = n_eval - q_out
    n_data = n_train - round(r.uniform(*REFERENCE_SHARE) * n_train)
    contamination = 0.0 if r.random() < CLEAN_PROB else r.uniform(0.0, MAX_CONTAMINATION)
    c_out = round(contamination * n_data)
    if anomalies is not None:
        c_out = min(c_out, anomalies - q_out)
    c_in = min(n_data - c_out, inliers - q_in)
    if q_in < 1 or q_out < 1 or c_in < MIN_CONTEXT_INLIERS:
        return None
    return q_in, q_out, c_in, c_out


def outlier_table(
    x: np.ndarray,
    y: np.ndarray,
    n_train: int,
    r: np.random.Generator,
    kinds: tuple[str, ...] = ANOMALY_KINDS,
    reference_kinds: tuple[str, ...] = REFERENCE_KINDS,
) -> tuple[np.ndarray, np.ndarray] | None:
    """One classification table (x (rows, features), y (rows) class labels) turned into an outlier
    table of the same shape (module docstring): (x, y) with y 0 for the context's data rows and
    the inlier queries and 1 for the reference rows and the anomalous queries. A `mode` table
    whose classes do not give enough rows becomes a `local` one. None if the table is too small
    for a context with `MIN_CONTEXT_INLIERS` inliers."""
    n, f = x.shape
    n_eval = n - n_train
    kind = str(r.choice(kinds))
    inliers, pool, counts = r.permutation(n), None, None
    if kind == "mode":
        nominal = nominal_classes(y, r)
        if nominal is not None:
            is_nominal = np.isin(y, nominal)
            inliers = r.permutation(np.flatnonzero(is_nominal))
            pool = r.permutation(np.flatnonzero(~is_nominal))
            counts = _counts(n_train, n_eval, len(inliers), len(pool), r)
        if counts is None:
            kind, inliers, pool = "local", r.permutation(n), None
    if counts is None:
        counts = _counts(n_train, n_eval, n, None, r)
    if counts is None:
        return None
    q_in, q_out, c_in, c_out = counts
    n_ref = n_train - c_in - c_out
    query_in, context_in = x[inliers[:q_in]], x[inliers[q_in : q_in + c_in]]
    base = x[inliers]  # what the synthetic anomalies are built from
    if kind == "mode":
        anomalies = x[pool[: q_out + c_out]]
    elif kind == "local":
        anomalies = local_rows(base, q_out + c_out, r)
    elif kind == "cluster":
        center, width = cluster_center(base, r)
        anomalies = center + width * r.standard_normal((q_out + c_out, f))
    elif kind == "global":
        anomalies = uniform_rows(base, q_out + c_out, r)
    elif kind == "dependency":
        anomalies = dependency_rows(base, q_out + c_out, r)
    else:
        raise ValueError(f"unknown anomaly kind {kind!r}: one of {ANOMALY_KINDS}")
    query_out, context_out = anomalies[:q_out], anomalies[q_out:]
    data = np.vstack([context_in, context_out])
    reference_kind = str(r.choice(reference_kinds))
    if reference_kind == "uniform":
        reference = uniform_rows(data, n_ref, r)
    elif reference_kind == "marginals":
        reference = marginal_rows(data, n_ref, r)
    else:
        raise ValueError(f"unknown reference kind {reference_kind!r}: one of {REFERENCE_KINDS}")
    context = np.vstack([data, reference])
    context_y = np.r_[np.zeros(len(data)), np.ones(n_ref)]
    queries = np.vstack([query_in, query_out])
    query_y = np.r_[np.zeros(q_in), np.ones(q_out)]
    a, b = r.permutation(n_train), r.permutation(n_eval)
    return np.vstack([context[a], queries[b]]), np.r_[context_y[a], query_y[b]]


class OutlierTable:
    """With probability `prob` per dataset, turn a classification table into an outlier table
    (`outlier_table`). A table too small to be turned stays as it is. It runs after
    `CoverClasses`, which would otherwise reorder its rows."""

    def __init__(
        self,
        prob: float,
        kinds: tuple[str, ...] = ANOMALY_KINDS,
        reference_kinds: tuple[str, ...] = REFERENCE_KINDS,
    ) -> None:
        if not 0.0 <= prob <= 1.0:
            raise ValueError(f"prob must be between 0 and 1: {prob}")
        if not kinds or set(kinds) - set(ANOMALY_KINDS):
            raise ValueError(f"anomaly kinds are chosen from {ANOMALY_KINDS}: {kinds}")
        if not reference_kinds or set(reference_kinds) - set(REFERENCE_KINDS):
            raise ValueError(
                f"reference kinds are chosen from {REFERENCE_KINDS}: {reference_kinds}"
            )
        self.prob = prob
        self.kinds = tuple(kinds)
        self.reference_kinds = tuple(reference_kinds)

    def __call__(self, group: Group, rng: Rng) -> Group | None:
        r = rng.numpy
        xs, ys = [], []
        for d in range(group.y.shape[1]):
            x, y = group.x[:, d], group.y[:, d]
            if r.random() < self.prob:
                out = outlier_table(
                    x.double().cpu().numpy(),
                    y.double().cpu().numpy(),
                    group.n_train,
                    r,
                    self.kinds,
                    self.reference_kinds,
                )
                if out is not None:
                    x = torch.as_tensor(out[0], dtype=x.dtype, device=x.device)
                    y = torch.as_tensor(out[1], dtype=y.dtype, device=y.device)
            xs.append(x)
            ys.append(y)
        return group.with_tensors(torch.stack(xs, 1), torch.stack(ys, 1))
