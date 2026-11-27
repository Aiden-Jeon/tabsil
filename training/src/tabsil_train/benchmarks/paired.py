"""Paired comparison of two evaluation runs on the same splits (v2 experiment plan §3,
`training/docs/v2-experiment-plan.md`).

The unit is the dataset. A run's error on a dataset is the mean of TabArena's error
(`metric_error`: 1 - ROC AUC for binary, log loss for multiclass, RMSE for regression) over its
splits, and the two runs must have the same splits. For run a (control) and run b (arm), with
r_d = error_b / error_a per dataset:

- G: the geometric mean of r_d, exp(mean ln r_d). Below 1 means b is better
- a 95% bootstrap CI of G, resampling datasets (2,000 rounds by default, fixed seed)
- the Wilcoxon signed-rank p (two-sided) of ln r_d against 0
- the number of datasets where b is better (r_d < 1), worse (r_d > 1) and equal

reported for all datasets, classification (binary and multiclass), regression, binary and
multiclass. Several suites (TabArena with 3 splits, the development suite) can be pooled: their
datasets are put together, each named `<suite>/<dataset>`. Elo is carried along if the source has
it and decides nothing.

Inputs are the per-split tables the eval script writes: `results_per_split.csv` of a TabArena /
BeyondArena run (columns dataset, fold = split index, metric_error, problem_type, imputed) or
`eval_rows.csv` of an AMLB / development-suite run (dataset, fold, metric_error, problem_type).
An imputed split or a missing error is an error, not a value: a comparison with them would
compare the stand-in method.

    from tabsil_train.benchmarks import paired
    a = paired.load_splits("a/results_per_split.csv")
    b = paired.load_splits("b/results_per_split.csv")
    print(paired.format_report(paired.compare({"tabarena-lite": (a, b)})))
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

BOOTSTRAP_ROUNDS = 2000
BOOTSTRAP_SEED = 0
# a dataset error of exactly 0 (a perfect ROC AUC on every split) is raised to this before the
# ratio; two zeros are a ratio of 1
ERROR_FLOOR = 1e-6
GROUPS = {
    "all": ("binary", "multiclass", "regression"),
    "classification": ("binary", "multiclass"),
    "regression": ("regression",),
    "binary": ("binary",),
    "multiclass": ("multiclass",),
}
COLUMNS = ["dataset", "split", "problem_type", "error"]


def load_splits(path: str | Path, allow_imputed: bool = False) -> pd.DataFrame:
    """One run's per-split errors (columns `COLUMNS`) from a `results_per_split.csv` (arenas)
    or `eval_rows.csv` (AMLB, development suite). Raises on imputed splits (unless
    `allow_imputed`) and on missing errors."""
    frame = pd.read_csv(path)
    return splits_frame(frame, str(path), allow_imputed)


def splits_frame(frame: pd.DataFrame, label: str, allow_imputed: bool = False) -> pd.DataFrame:
    """`load_splits` on a frame already read (`label` names it in errors)."""
    missing = {"dataset", "fold", "metric_error", "problem_type"} - set(frame.columns)
    if missing:
        raise ValueError(
            f"{label}: no {sorted(missing)} column; the paired report needs `metric_error` and "
            "`problem_type` (AMLB runs log them since the development suite was added)"
        )
    if "imputed" in frame.columns and not allow_imputed:
        imputed = frame[frame["imputed"].astype(bool)]
        if len(imputed):
            names = ", ".join(
                f"{d} split {s}" for d, s in zip(imputed["dataset"], imputed["fold"], strict=True)
            )
            raise ValueError(f"{label}: {len(imputed)} imputed splits ({names})")
    out = frame.rename(columns={"fold": "split", "metric_error": "error"})[COLUMNS].copy()
    bad = out[~np.isfinite(out["error"].astype(float))]
    if len(bad):
        names = ", ".join(
            f"{d} split {s}" for d, s in zip(bad["dataset"], bad["split"], strict=True)
        )
        raise ValueError(f"{label}: no error on {names}")
    if out.duplicated(["dataset", "split"]).any():
        raise ValueError(f"{label}: a (dataset, split) appears twice")
    unknown = set(out["problem_type"]) - set(GROUPS["all"])
    if unknown:
        raise ValueError(f"{label}: unknown problem types {sorted(unknown)}")
    return out


def mean_splits(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """The mean of several runs as one run: on every split, the mean of the runs' errors. For a
    control that is the mean of repeated runs (two seeds of one branch, plan §4 phase 1A). The
    runs must have the same (dataset, split) pairs and problem types."""
    if not frames:
        raise ValueError("no runs to average")
    keys = ["dataset", "split"]
    first = frames[0].set_index(keys).sort_index()
    errors = [first["error"]]
    for frame in frames[1:]:
        other = frame.set_index(keys).sort_index()
        if not first.index.equals(other.index):
            raise ValueError("the runs to average have different splits")
        if (first["problem_type"] != other["problem_type"]).any():
            raise ValueError("the runs to average disagree on a dataset's problem type")
        errors.append(other["error"])
    out = first.copy()
    out["error"] = np.mean([e.to_numpy(dtype=float) for e in errors], axis=0)
    return out.reset_index()[COLUMNS]


def dataset_ratios(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """Per dataset: the split-mean errors of a and b, the split count and r = b / a. The two
    runs must have the same (dataset, split) pairs and problem types."""
    keys = ["dataset", "split"]
    pairs_a = set(map(tuple, a[keys].to_numpy().tolist()))
    pairs_b = set(map(tuple, b[keys].to_numpy().tolist()))
    if pairs_a != pairs_b:
        only_a, only_b = sorted(pairs_a - pairs_b), sorted(pairs_b - pairs_a)
        raise ValueError(
            f"the two runs have different splits: only in a {only_a[:5]}"
            f"{'...' if len(only_a) > 5 else ''}, only in b {only_b[:5]}"
            f"{'...' if len(only_b) > 5 else ''}"
        )
    merged = a.merge(b, on=keys, suffixes=("_a", "_b"))
    if (merged["problem_type_a"] != merged["problem_type_b"]).any():
        raise ValueError("the two runs disagree on a dataset's problem type")
    per = (
        merged.groupby("dataset")
        .agg(
            problem_type=("problem_type_a", "first"),
            splits=("split", "size"),
            error_a=("error_a", "mean"),
            error_b=("error_b", "mean"),
        )
        .reset_index()
    )
    both_zero = (per["error_a"] == 0) & (per["error_b"] == 0)
    floor_a = per["error_a"].clip(lower=ERROR_FLOOR)
    floor_b = per["error_b"].clip(lower=ERROR_FLOOR)
    per["ratio"] = np.where(both_zero, 1.0, floor_b / floor_a)
    return per


def summarize(
    ratios: pd.DataFrame,
    rounds: int = BOOTSTRAP_ROUNDS,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """The §3 statistics of one set of per-dataset ratios (`dataset_ratios`)."""
    from scipy.stats import wilcoxon

    log_r = np.log(ratios["ratio"].to_numpy(dtype=float))
    n = len(log_r)
    out = {
        "datasets": int(n),
        "splits": int(ratios["splits"].sum()),
        "better": int((log_r < 0).sum()),
        "worse": int((log_r > 0).sum()),
        "equal": int((log_r == 0).sum()),
    }
    if n == 0:
        return {**out, "G": None, "ci_low": None, "ci_high": None, "wilcoxon_p": None}
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n, size=(rounds, n))
    boot = np.exp(log_r[draws].mean(axis=1))
    if np.all(log_r == 0):
        p = 1.0
    else:
        p = float(wilcoxon(log_r, zero_method="wilcox", alternative="two-sided").pvalue)
    return {
        **out,
        "G": float(np.exp(log_r.mean())),
        "ci_low": float(np.quantile(boot, 0.025)),
        "ci_high": float(np.quantile(boot, 0.975)),
        "wilcoxon_p": p,
    }


def compare(
    suites: Mapping[str, tuple[pd.DataFrame, pd.DataFrame]],
    rounds: int = BOOTSTRAP_ROUNDS,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """The paired report of a and b over one or more suites {suite: (a splits, b splits)}.
    Returns {"suites": {suite: {group: stats}}, "pooled": {group: stats} (with more than one
    suite), "ratios": per-dataset frame (column `suite`)}."""
    frames = []
    for name, (a, b) in suites.items():
        per = dataset_ratios(a, b)
        per.insert(0, "suite", name)
        frames.append(per)
    ratios = pd.concat(frames, ignore_index=True)

    def groups(frame: pd.DataFrame) -> dict:
        return {
            group: summarize(frame[frame["problem_type"].isin(kinds)], rounds, seed)
            for group, kinds in GROUPS.items()
        }

    report = {
        "suites": {name: groups(ratios[ratios["suite"] == name]) for name in suites},
        "ratios": ratios,
    }
    if len(suites) > 1:
        report["pooled"] = groups(ratios)
    return report


def format_report(report: dict, elo: Mapping[str, Mapping[str, tuple]] | None = None) -> str:
    """Markdown tables of `compare`'s result: one row per group, per suite and pooled. `elo` adds
    the Elo of a and b per suite and group {suite: {group: (elo a, elo b)}}."""
    lines = []
    blocks = list(report["suites"].items())
    if "pooled" in report:
        blocks.append(("pooled", report["pooled"]))
    for name, groups in blocks:
        lines += [
            f"### {name}",
            "",
            "| Group | Datasets (splits) | G (95% CI) | Wilcoxon p | Better / worse / equal"
            " | Elo a → b |",
            "|---|---|---|---|---|---|",
        ]
        for group, s in groups.items():
            if not s["datasets"]:
                continue
            e = (elo or {}).get(name, {}).get(group)
            elo_text = f"{e[0]:.0f} → {e[1]:.0f}" if e and None not in e else ""
            lines.append(
                f"| {group} | {s['datasets']} ({s['splits']}) | {s['G']:.3f}"
                f" ({s['ci_low']:.3f}-{s['ci_high']:.3f}) | {s['wilcoxon_p']:.3g}"
                f" | {s['better']} / {s['worse']} / {s['equal']} | {elo_text} |"
            )
        lines.append("")
    return "\n".join(lines)


def to_json(report: dict) -> str:
    """`compare`'s result without the per-dataset frame, as JSON."""
    return json.dumps({k: v for k, v in report.items() if k != "ratios"}, indent=2)
