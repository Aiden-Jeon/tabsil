"""Predictive-distribution quality and uncertainty metrics: CRPS, intervals and PIT for regression
bar distributions, ECE for classification probabilities, and (conditional) coverage of split
conformal prediction.

Point-prediction metrics (ROC AUC, RMSE) say nothing about whether the distribution fits or whether
the uncertainty can be trusted. Everything here is computed from checkpoint predictions alone, with
no training.

A regression distribution is built once with `bar_histogram(probs, borders, tails)`: bucket
probabilities `probs` (n, n_buckets) and borders `borders`. Borders are (n_buckets + 1,) when shared
by all rows, as in TabPFN's bar distribution, or (n, n_buckets + 1) when they differ per row. Models
that output quantiles (GBDT quantile regression, TabICL, Gaussian parameters converted to quantiles)
are converted with `quantile_histogram(levels, values)` into per-row-border histograms. Inner
buckets have uniform density, so the CDF is linear between borders. The two end buckets follow
`tails`.

- `"half_normal"` (default): matches the distribution the model is trained with
  (`FullSupportBarDistribution`). The first bucket's mass spreads left from `borders[1]` and the
  last bucket's mass spreads right from `borders[-2]` as a half-normal, with scale chosen so the
  median equals that bucket's width (so there is mass beyond the outermost borders). This is also
  the tail used by `EnsembleRegressor.predict_distribution`.
- `"uniform"`: end buckets are uniform too (zero mass beyond the outermost borders). This is how
  ScoringBench (arXiv 2603.29928) scores TabPFN's bar distribution; use it to compare with their
  numbers.

The (log prob, borders) from `EnsembleRegressor.predict_distribution` can be passed directly after
`np.exp`. A zero-width bucket (equal borders) is treated as an atom at that point.

- `crps`: CRPS = ∫ (F(t) − 1{t ≥ y})² dt. Exact: the CDF is linear on inner buckets and the
  half-normal tail integral has a closed form.
- `interval_scores`: coverage, width and interval score of the central `level` interval
  [Q((1−level)/2), Q((1+level)/2)] (Gneiting & Raftery 2007: width + (2/α)·miss distance,
  α = 1 − level). As a proxy for conditional coverage, also reports the lowest coverage across
  interval-width quintiles (`coverage_min_by_width`): do wide and narrow intervals cover at the
  same rate (same idea as SSCS for classification)? Rows with equal width land in the same bin, so
  the result does not depend on row order.
- `pit`: PIT F(y). Uniform(0, 1) under a calibrated distribution; the distance is reported as the
  KS statistic (`pit_ks`).

Classification probabilities `proba` (n, n_classes) follow the order of `classes` (the estimator's
`classes_`). Test rows whose class was absent from the training fold are dropped with the same rule
as ROC AUC (`amlb.drop_unseen_classes`).

- `ece`: top-label expected calibration error (15 equal-width confidence bins), log loss, Brier.
- `conformal`: split conformal prediction. The model is already fit on the training fold, so test
  fold predictions are randomly split (`n_repeats` times) into a calibration half and an evaluation
  half, with no extra training (the two halves are exchangeable). Nonconformity scores are LAC
  (1 − p_y, Sadinle et al. 2019) and randomized APS (sum of probabilities larger than p_y + u · p_y,
  Romano et al. 2020). Reported: marginal coverage, mean set size, SSCS (size-stratified coverage,
  Angelopoulos et al. 2021: the lowest coverage across set-size groups 0-1, 2-3, 4-10, ..., using
  only groups with at least `min_group` rows), and class-conditional coverage gap (mean over
  classes of |coverage − (1 − α)|, Ding et al. 2023). Evaluation halves from all repeats are pooled
  before grouping by size and class.

`regression_metrics` and `classification_metrics` collect the metrics of one fold into a dict
(`amlb.evaluate_task(..., uncertainty=True)`). If no rows remain, everything is NaN.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import special, stats

from tabsil_train.benchmarks.amlb import drop_unseen_classes

COVERAGE_LEVELS = (0.5, 0.8, 0.9, 0.95)
CONFORMAL_ALPHA = 0.1
CONFORMAL_SCORES = ("lac", "aps")
TAILS = ("half_normal", "uniform")
DEFAULT_TAILS = "half_normal"
CRPS_CHUNK_ROWS = 512  # rows per (rows, n_buckets) array built by crps at once

# median of half-normal(scale 1) = Φ⁻¹(0.75). Same scale rule as `FullSupportBarDistribution`
_HALF_NORMAL_MEDIAN = float(special.ndtri(0.75))
_INV_SQRT_PI = 1.0 / np.sqrt(np.pi)


# ---------------------------------------------------------------------------
# Regression: bucket histograms
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BarHistogram:
    """Normalized bucket probabilities and the CDF at the borders. Built by `bar_histogram`."""

    probs: np.ndarray  # (n, B), each row sums to 1
    borders: np.ndarray  # (n, B + 1), float64. A `np.broadcast_to` view when shared by all rows
    cdf: np.ndarray  # (n, B + 1), CDF at the borders (0 and 1 at the ends). Same for any tails
    tails: str
    scale_left: (
        np.ndarray
    )  # (n,), per-row scale of the left half-normal tail (first width / median)
    scale_right: np.ndarray  # (n,), scale of the right tail
    shared_borders: np.ndarray | None = None  # the (B + 1,) borders if shared by all rows


def bar_histogram(probs, borders, tails: str = DEFAULT_TAILS) -> BarHistogram:
    """Build a histogram from bucket probabilities (n, B) and borders. Borders are (B + 1,) when
    shared by all rows, or (n, B + 1) per row (convert quantile predictions with
    `quantile_histogram` first)."""
    if tails not in TAILS:
        raise ValueError(f"tails must be one of {TAILS}: {tails}")
    probs = np.atleast_2d(np.asarray(probs, dtype=np.float64))
    borders = np.asarray(borders, dtype=np.float64)
    n, n_buckets = probs.shape
    if borders.shape not in ((n_buckets + 1,), (n, n_buckets + 1)):
        raise ValueError(
            f"borders must have shape (n_buckets + 1,) or (n_rows, n_buckets + 1): "
            f"{borders.shape}, {probs.shape}"
        )
    if np.any(np.diff(borders, axis=-1) < 0):
        raise ValueError("borders must be non-decreasing")
    if tails == "half_normal" and (
        n_buckets < 2
        or np.any(borders[..., 1] <= borders[..., 0])
        or np.any(borders[..., -1] <= borders[..., -2])
    ):
        raise ValueError("half-normal tails need at least 2 buckets and positive-width end buckets")
    shared = borders if borders.ndim == 1 else None
    # compute tail scales once (broadcast a constant when borders are shared)
    scale_left = (borders[..., 1] - borders[..., 0]) / _HALF_NORMAL_MEDIAN
    scale_right = (borders[..., -1] - borders[..., -2]) / _HALF_NORMAL_MEDIAN
    if shared is not None:
        borders = np.broadcast_to(shared, (n, n_buckets + 1))  # no copy
        scale_left, scale_right = np.full(n, scale_left), np.full(n, scale_right)
    probs = probs.clip(min=0.0)
    probs = probs / probs.sum(axis=1, keepdims=True)
    cdf = np.concatenate([np.zeros((n, 1)), np.cumsum(probs, axis=1)], axis=1)
    cdf[:, -1] = 1.0
    return BarHistogram(probs, borders, cdf, tails, scale_left, scale_right, shared)


def quantile_histogram(levels, values) -> tuple[np.ndarray, np.ndarray]:
    """Convert quantile predictions into a bucket histogram with per-row borders (for
    `bar_histogram`).

    `levels` (K,) are increasing probability levels in (0, 1) and `values` (n, K) are the per-row
    quantiles. Assuming the CDF passes through (values[:, k], levels[k]) and is linear in between,
    the mass of the bucket between two neighbouring quantiles is the difference of their levels
    (how ScoringBench scores quantiles as CDF points). Then `quantile(h, levels)` returns `values`
    exactly. Crossing quantiles (from GBDTs fit separately per level) are sorted per row
    (rearrangement, Chernozhukov et al. 2010). Equal quantiles form a zero-width bucket, i.e. an
    atom.

    The mass levels[0] below the lowest quantile and 1 − levels[-1] above the highest go to the two
    end buckets. Their width (the half-normal tail median) is set so the tail density matches the
    density of the adjacent inner bucket. If that bucket has zero width, the mean inner density is
    used; if all quantiles are equal, a tiny width is used.

    Returns (bucket probabilities (n, K + 1), borders (n, K + 2)).
    """
    levels = np.asarray(levels, dtype=np.float64)
    values = np.atleast_2d(np.asarray(values, dtype=np.float64))
    if levels.ndim != 1 or len(levels) < 2 or values.shape[1] != len(levels):
        raise ValueError(
            f"quantiles must have shape (n_rows, n_levels >= 2): {levels.shape}, {values.shape}"
        )
    if levels[0] <= 0 or levels[-1] >= 1 or np.any(np.diff(levels) <= 0):
        raise ValueError("probability levels must be strictly increasing in (0, 1)")
    if not np.isfinite(values).all():
        raise ValueError("quantiles contain NaN or infinity")
    q = np.sort(values, axis=1)
    p_buckets = np.concatenate([levels[:1], np.diff(levels), 1 - levels[-1:]])
    probs = np.broadcast_to(p_buckets, (len(q), len(p_buckets))).copy()
    # width per unit mass (inverse density). Fallback order: adjacent bucket → inner mean → tiny
    inner = (q[:, -1] - q[:, 0]) / (levels[-1] - levels[0])
    tiny = 1e-9 * np.maximum(1.0, np.abs(q).max(axis=1))

    def tail_width(p_tail: float, gap: np.ndarray, p_gap: float) -> np.ndarray:
        per_prob = np.where(gap > 0, gap / p_gap, inner)
        # density at the join of a half-normal (scale s, mass p), p·√(2/π)/s, equals 1/per_prob
        width = _HALF_NORMAL_MEDIAN * p_tail * np.sqrt(2 / np.pi) * per_prob
        return np.maximum(width, tiny)

    left = tail_width(levels[0], q[:, 1] - q[:, 0], levels[1] - levels[0])
    right = tail_width(1 - levels[-1], q[:, -1] - q[:, -2], levels[-1] - levels[-2])
    borders = np.concatenate([(q[:, 0] - left)[:, None], q, (q[:, -1] + right)[:, None]], axis=1)
    return probs, borders


def _bucket_index(borders: np.ndarray, y: np.ndarray, shared: np.ndarray | None) -> np.ndarray:
    """Per row, k with borders[k] ≤ y < borders[k + 1] (clipped to 0..B − 1). borders (m, B + 1)."""
    if shared is not None:
        k = np.searchsorted(shared, y, side="right") - 1
    else:
        k = (borders <= y[:, None]).sum(axis=1) - 1
    return k.clip(0, borders.shape[1] - 2)


def _upper_tail(u):
    """Q(u) = 1 − Φ(u)."""
    return special.ndtr(-u)


def _normal_pdf(u):
    return np.exp(-0.5 * u * u) / np.sqrt(2 * np.pi)


def _fraction(lo: np.ndarray, hi: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Position of y within [lo, hi] (0-1). For a zero-width bucket, 1 if y is at or above it."""
    width = hi - lo
    safe = np.where(width > 0, width, 1.0)
    return np.where(width > 0, (y - lo) / safe, (y >= lo).astype(float)).clip(0, 1)


def pit(h: BarHistogram, y) -> np.ndarray:
    """F(y) (n,)."""
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    b, rows = h.borders, np.arange(len(y))
    k = _bucket_index(b, y, h.shared_borders)
    out = h.cdf[rows, k] + h.probs[rows, k] * _fraction(b[rows, k], b[rows, k + 1], y)
    if h.tails == "half_normal":
        p_left, p_right = h.probs[:, 0], h.probs[:, -1]
        left = p_left * 2 * _upper_tail((b[:, 1] - y) / h.scale_left)
        right = 1 - p_right * 2 * _upper_tail((y - b[:, -2]) / h.scale_right)
        out = np.where(y < b[:, 1], left, np.where(y > b[:, -2], right, out))
    return out.clip(0.0, 1.0)


def pit_ks(pit_values) -> float:
    """KS statistic between PIT and Uniform(0, 1) (0 means a calibrated distribution)."""
    return float(stats.kstest(np.asarray(pit_values, dtype=np.float64), "uniform").statistic)


def quantile(h: BarHistogram, q) -> np.ndarray:
    """Quantiles (n, len(q)). Linear CDF on inner buckets; exact inverse on half-normal tails."""
    q = np.atleast_1d(np.asarray(q, dtype=np.float64))
    n, n_buckets = h.probs.shape
    # find bucket k with cdf[k] < q <= cdf[k + 1] for all rows at once: adding 2i to row i's CDF
    # (0-1) and concatenating gives an increasing array, so one searchsorted suffices (zero-mass
    # buckets are skipped)
    offset = 2.0 * np.arange(n)[:, None]
    flat = (h.cdf + offset).ravel()
    pos = np.searchsorted(flat, (q[None, :] + offset).ravel(), side="left").reshape(n, len(q))
    k = (pos - np.arange(n)[:, None] * (n_buckets + 1) - 1).clip(0, n_buckets - 1)
    rows = np.arange(n)[:, None]
    p = h.probs[rows, k]
    frac = np.where(p > 0, (q[None, :] - h.cdf[rows, k]) / np.where(p > 0, p, 1.0), 0.0)
    b = h.borders
    out = b[rows, k] + frac.clip(0.0, 1.0) * (b[rows, k + 1] - b[rows, k])
    if h.tails == "half_normal":
        p_left, p_right = h.probs[:, :1], h.probs[:, -1:]
        in_left, in_right = k == 0, k == n_buckets - 1
        # left: p0 · 2Q(u) = q, right: 1 − p_last · 2Q(u) = q. Cells outside the tail get
        # Φ⁻¹(0.5) = 0, and the Φ⁻¹ argument is clipped below 1 − 1e-16 (finite u at q = 0, 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            arg_left = np.where(in_left, 1 - q[None, :] / (2 * p_left), 0.5)
            arg_right = np.where(in_right, 1 - (1 - q[None, :]) / (2 * p_right), 0.5)
        u_left = special.ndtri(np.nan_to_num(arg_left, nan=0.5).clip(0.5, 1 - 1e-16))
        u_right = special.ndtri(np.nan_to_num(arg_right, nan=0.5).clip(0.5, 1 - 1e-16))
        out = np.where(in_left, b[:, 1:2] - h.scale_left[:, None] * u_left, out)
        out = np.where(in_right, b[:, -2:-1] + h.scale_right[:, None] * u_right, out)
    return out


def _segment_square_integral(a, b, h, width):
    """∫ (F − h)² dt over [l, r] where F is linear from a to b."""
    a, b = a - h, b - h
    return width * (a * a + a * b + b * b) / 3.0


def _sq_tail_integral(a):
    """∫_a^∞ Q(u)² du = −aQ(a)² + 2φ(a)Q(a) − Q(√2·a)/√π (a ≥ 0)."""
    q = _upper_tail(a)
    return -a * q * q + 2 * _normal_pdf(a) * q - _INV_SQRT_PI * _upper_tail(np.sqrt(2) * a)


def _right_tail_crps(p, start, scale, y):
    """∫_c^∞ (F − H)² dt over a half-normal tail (mass p) spreading right from c = start.
    On the tail, F(t) = 1 − 2p·Q(u), u = (t − c) / scale."""
    a = np.maximum((y - start) / scale, 0.0)  # if y is left of c, H = 1 over the whole tail
    q_int = _normal_pdf(0.0) - _normal_pdf(a) + a * _upper_tail(a)  # ∫_0^a Q du
    tail_sq = _sq_tail_integral(a)
    below = a - 4 * p * q_int + 4 * p * p * (_sq_tail_integral(0.0) - tail_sq)  # H = 0 on [0, a]
    return scale * (below + 4 * p * p * tail_sq)


def crps(h: BarHistogram, y) -> np.ndarray:
    """Per-row CRPS (original units). Computed in chunks of `CRPS_CHUNK_ROWS` rows."""
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    return np.concatenate(
        [
            _crps_rows(h, slice(i, i + CRPS_CHUNK_ROWS), y[i : i + CRPS_CHUNK_ROWS])
            for i in range(0, len(y), CRPS_CHUNK_ROWS)
        ]
    )


def _crps_rows(h: BarHistogram, rows: slice, y: np.ndarray) -> np.ndarray:
    probs, cdf, b = h.probs[rows], h.cdf[rows], h.borders[rows]
    # buckets integrated with a linear CDF: all for uniform, inner ones only for half-normal
    lo, hi = (0, b.shape[1] - 1) if h.tails == "uniform" else (1, b.shape[1] - 2)
    left, right = b[:, lo:hi], b[:, lo + 1 : hi + 1]
    width = right - left
    a, c = cdf[:, lo:hi], cdf[:, lo + 1 : hi + 1]
    # H = 1 if the whole bucket lies right of y, else H = 0
    total = np.where(
        left >= y[:, None],
        _segment_square_integral(a, c, 1.0, width),
        _segment_square_integral(a, c, 0.0, width),
    )
    # split the bucket containing y at y
    inside = np.flatnonzero((y > b[:, lo]) & (y < b[:, hi]))
    if len(inside) and hi > lo:
        b_in, y_in, r = b[inside], y[inside], np.arange(len(inside))
        k = _bucket_index(b_in, y_in, h.shared_borders).clip(lo, hi - 1)
        b_k, b_k1 = b_in[r, k], b_in[r, k + 1]
        f_y = cdf[inside, k] + probs[inside, k] * _fraction(b_k, b_k1, y_in)
        split = _segment_square_integral(cdf[inside, k], f_y, 0.0, y_in - b_k)
        split += _segment_square_integral(f_y, cdf[inside, k + 1], 1.0, b_k1 - y_in)
        total[inside, k - lo] = split
    out = total.sum(axis=1)
    if h.tails == "uniform":
        # beyond the outermost borders F is 0 or 1 and differs from H by 1
        return out + (b[:, 0] - y).clip(min=0.0) + (y - b[:, -1]).clip(min=0.0)
    # the left tail is the right-tail formula after flipping t → −t
    out += _right_tail_crps(probs[:, -1], b[:, -2], h.scale_right[rows], y)
    out += _right_tail_crps(probs[:, 0], -b[:, 1], h.scale_left[rows], -y)
    return out


def interval_scores(
    h: BarHistogram, y, levels=COVERAGE_LEVELS, n_width_bins: int = 5
) -> dict[str, float]:
    """Per central interval: coverage, mean width, interval score, min coverage by width bin."""
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    alphas = 1.0 - np.asarray(levels, dtype=np.float64)
    bounds = quantile(h, np.concatenate([alphas / 2, 1 - alphas / 2]))  # all levels at once
    out = {}
    for j, (level, alpha) in enumerate(zip(levels, alphas, strict=True)):
        lo, hi = bounds[:, j], bounds[:, len(levels) + j]
        covered = (y >= lo) & (y <= hi)
        width = hi - lo
        score = width + (2 / alpha) * ((lo - y).clip(min=0) + (y - hi).clip(min=0))
        name = f"{round(level * 100)}"
        out[f"coverage_{name}"] = float(covered.mean())
        out[f"width_{name}"] = float(width.mean())
        out[f"interval_score_{name}"] = float(score.mean())
        out[f"coverage_min_by_width_{name}"] = min_coverage_by_width(covered, width, n_width_bins)
    return out


def min_coverage_by_width(
    covered: np.ndarray, width: np.ndarray, n_bins: int = 5, min_rows: int = 10
) -> float:
    """Lowest coverage across n_bins width quantile bins (only bins with at least `min_rows` rows).
    Rows with equal width get the same average rank and land in the same bin (order-independent)."""
    if len(covered) < 2 * n_bins:
        return float("nan")
    rank = stats.rankdata(width, method="average")  # 1 ~ n
    bins = np.minimum(((rank - 0.5) / len(width) * n_bins).astype(int), n_bins - 1)
    per_bin = [covered[bins == j].mean() for j in range(n_bins) if (bins == j).sum() >= min_rows]
    return float(min(per_bin)) if per_bin else float("nan")


def regression_metrics(
    probs, borders, y, y_scale: float = 1.0, levels=COVERAGE_LEVELS, tails: str = DEFAULT_TAILS
) -> dict[str, float]:
    """Regression distribution metrics for one fold. Borders are (B + 1,) or per-row (n, B + 1).
    CRPS, widths and interval scores divided by `y_scale` (e.g. the training-target std) are also
    reported with a `_scaled` suffix (for averaging across datasets)."""
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if len(y) == 0:
        return {}
    h = bar_histogram(probs, borders, tails)  # normalize and build the CDF once
    out = {"crps": float(crps(h, y).mean())}
    out.update(interval_scores(h, y, levels))
    out["pit_ks"] = pit_ks(pit(h, y))
    scaled = {
        f"{k}_scaled": v / y_scale
        for k, v in out.items()
        if k == "crps" or k.startswith(("width_", "interval_score_"))
    }
    return {**out, **scaled}


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def _seen_rows(proba, y_true, classes) -> tuple[np.ndarray, np.ndarray]:
    """Keep only rows whose class appeared in the training fold (`amlb.drop_unseen_classes`) and
    map labels to indices into `classes`."""
    y_true, proba = drop_unseen_classes(y_true, proba, classes)
    proba = np.asarray(proba, dtype=np.float64)
    proba = proba / proba.sum(axis=1, keepdims=True)
    index = {c: i for i, c in enumerate(np.asarray(classes).tolist())}
    return proba, np.array([index[c] for c in y_true.tolist()], dtype=int)


def ece(proba, labels, n_bins: int = 15) -> float:
    """Top-label ECE. `labels` are column indices into `proba`."""
    proba, labels = np.asarray(proba, dtype=np.float64), np.asarray(labels)
    if len(labels) == 0:
        return float("nan")
    conf = proba.max(axis=1)
    correct = proba.argmax(axis=1) == labels
    bins = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    total = 0.0
    for b in np.unique(bins):
        m = bins == b
        total += m.sum() * abs(correct[m].mean() - conf[m].mean())
    return float(total / len(labels))


def conformal_scores(proba: np.ndarray, kind: str, u: np.ndarray | None = None) -> np.ndarray:
    """Nonconformity scores for every class (n, n_classes): the score if that class were the label.

    APS uses the original randomized definition: score of class k = sum of probabilities larger
    than p_k + u · p_k (u ~ Uniform(0, 1), one per row). Without randomization (u = 1), binary
    classification with accuracy below 1 − α gets threshold 1 and every set becomes all classes.
    """
    if kind == "lac":
        return 1.0 - proba
    if kind == "aps":
        u = np.ones(len(proba)) if u is None else np.asarray(u, dtype=np.float64)
        above = (proba[:, None, :] * (proba[:, None, :] > proba[:, :, None])).sum(-1)
        return above + u[:, None] * proba
    raise ValueError(f"conformal score must be one of {CONFORMAL_SCORES}: {kind}")


# SSCS set-size groups (Angelopoulos et al. 2021): 0-1, 2-3, 4-10, 11-100, 101+.
# 0 and 1 share a group because when accuracy exceeds 1 − α, empty sets appear to bring coverage
# down to 1 − α; an empty set has coverage 0 by definition, so a separate group would pin SSCS at 0
SSCS_SIZE_BINS = (0, 2, 4, 11, 101)


def size_stratified_coverage(covered: np.ndarray, sizes: np.ndarray, min_group: int = 10) -> float:
    """Lowest coverage across set-size groups (`SSCS_SIZE_BINS`), with at least `min_group` rows."""
    groups = np.searchsorted(SSCS_SIZE_BINS, sizes, side="right")
    per_group = [
        covered[groups == g].mean() for g in np.unique(groups) if (groups == g).sum() >= min_group
    ]
    return float(min(per_group)) if per_group else float("nan")


def conformal_threshold(cal_scores: np.ndarray, alpha: float) -> float:
    """The ⌈(n + 1)(1 − α)⌉-th calibration score; infinity (all classes) if that rank exceeds n."""
    n = len(cal_scores)
    rank = int(np.ceil((n + 1) * (1 - alpha)))
    if rank > n:
        return float("inf")
    return float(np.sort(cal_scores)[rank - 1])


_CONFORMAL_NAMES = ("coverage", "set_size", "sscs", "class_coverage_gap")


def conformal(
    proba,
    labels,
    alpha: float = CONFORMAL_ALPHA,
    kind: str = "lac",
    n_repeats: int = 20,
    min_group: int = 10,
    seed: int = 0,
) -> dict[str, float]:
    """Split conformal prediction: repeatedly split rows at random into calibration/eval halves."""
    proba, labels = np.asarray(proba, dtype=np.float64), np.asarray(labels)
    n = len(labels)
    if n < 20:
        return dict.fromkeys(_CONFORMAL_NAMES, float("nan"))
    rng = np.random.default_rng(seed)
    scores = conformal_scores(proba, kind, u=rng.random(n))
    true_scores = scores[np.arange(n), labels]
    covered, sizes, eval_labels = [], [], []
    for _ in range(n_repeats):
        perm = rng.permutation(n)
        cal, ev = perm[: n // 2], perm[n // 2 :]
        q = conformal_threshold(true_scores[cal], alpha)
        sets = scores[ev] <= q
        covered.append(sets[np.arange(len(ev)), labels[ev]])
        sizes.append(sets.sum(axis=1))
        eval_labels.append(labels[ev])
    covered, sizes = np.concatenate(covered), np.concatenate(sizes)
    eval_labels = np.concatenate(eval_labels)
    by_class = [covered[eval_labels == c].mean() for c in np.unique(eval_labels)]
    return {
        "coverage": float(covered.mean()),
        "set_size": float(sizes.mean()),
        "sscs": size_stratified_coverage(covered, sizes, min_group),
        "class_coverage_gap": float(np.mean(np.abs(np.array(by_class) - (1 - alpha)))),
    }


def classification_metrics(
    proba, y_true, classes, alpha: float = CONFORMAL_ALPHA, seed: int = 0
) -> dict[str, float]:
    """Classification metrics for one fold: ECE, log loss, Brier, conformal (LAC, APS)."""
    proba, labels = _seen_rows(proba, y_true, classes)
    if len(labels) == 0:  # no rows left: NaN, like score_fold
        names = ["ece", "log_loss", "brier"]
        names += [f"{kind}_{k}" for kind in CONFORMAL_SCORES for k in _CONFORMAL_NAMES]
        return dict.fromkeys(names, float("nan"))
    rows = np.arange(len(labels))
    onehot = np.zeros_like(proba)
    onehot[rows, labels] = 1.0
    out = {
        "ece": ece(proba, labels),
        "log_loss": float(-np.log(proba[rows, labels].clip(min=1e-15)).mean()),
        "brier": float(((proba - onehot) ** 2).sum(axis=1).mean()),
    }
    for kind in CONFORMAL_SCORES:
        result = conformal(proba, labels, alpha=alpha, kind=kind, seed=seed)
        out.update({f"{kind}_{k}": v for k, v in result.items()})
    return out


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

DATASET_KEYS = ["suite", "task_id", "dataset"]


def suite_means(per_dataset) -> dict[str, float]:
    """Dataset means of one suite's per-dataset metric table (`DATASET_KEYS` columns + metrics).
    For regression metrics with units (CRPS, width, interval score) only the `_scaled` variants
    (divided by the training-target std) are averaged; all-NaN columns are dropped."""
    values = per_dataset.drop(columns=[c for c in DATASET_KEYS if c in per_dataset.columns])
    values = values.dropna(axis=1, how="all")
    unscaled = {c.removesuffix("_scaled") for c in values.columns if c.endswith("_scaled")}
    means = values.drop(columns=[c for c in unscaled if c in values.columns]).mean()
    return {k: float(v) for k, v in means.items()}


def dataset_means(rows, exclude=()) -> pd.DataFrame:
    """Average per-fold rows (dicts from `amlb.evaluate_task`, including `suite`) over folds per
    dataset (`DATASET_KEYS`). Columns in `fold` and `exclude` are not averaged."""
    df = pd.DataFrame(rows)
    skip = {*DATASET_KEYS, "fold", *exclude}
    columns = [c for c in df.columns if c not in skip]
    return df.groupby(DATASET_KEYS)[columns].mean().reset_index()
