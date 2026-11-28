"""ADBench with the outlier arms of `arms.py` (plan item 1.3): the tables, splits and metrics of
the outlier suite (`tabsil_train.benchmarks.unsupervised.adbench`: 12 tables x 3 seeds, the
contaminated training rows as the context, test AUROC and AUPRC) with other scores of the same
rows. `iforest` is the suite's IsolationForest, as a check that the splits are the suite's.

`ADBENCH_ARMS` names the scorers (comma-separated names of `arms.outlier_scorers`, or `iforest`),
`ADBENCH_DATASETS` the tables (default: all 12). One row per (table, split, arm) goes to
`adbench_arms.csv` as it is computed; the mean AUROC per arm over the 36 cells is the number the
suite reports as its mean.

`ADBENCH_COMBINE` (arm names joined by `+`, for example `ctx64+ref-uniform`) adds an arm that is
the mean of those arms' ranks of the test rows (`rank_mean`): two scores of different kinds, a
density and a class probability, have no common scale, and their ranks do.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from arms import outlier_scorers
from common import DEVICE, OUT, load_model
from sklearn.metrics import average_precision_score, roc_auc_score

from tabsil_train.benchmarks.unsupervised import adbench


def rank_mean(scores: list[np.ndarray]) -> np.ndarray:
    """The mean over `scores` of each row's rank within its score (ties share their mean rank),
    divided by the row count: higher = more anomalous, as the scores."""
    from scipy.stats import rankdata

    return np.mean([rankdata(s) / len(s) for s in scores], axis=0)


def main() -> None:
    names = [a for a in os.environ.get("ADBENCH_ARMS", "iforest").split(",") if a]
    combine = [a for a in os.environ.get("ADBENCH_COMBINE", "").split("+") if a]
    datasets = [d for d in os.environ.get("ADBENCH_DATASETS", "").split(",") if d]
    datasets = datasets or list(adbench.DATASETS)
    model, borders = load_model()
    scorers = outlier_scorers(model, borders, DEVICE)
    scorers["iforest"] = lambda X_train, X_test, seed: {
        "iforest": adbench.iforest_scores(X_train, X_test, seed)
    }
    unknown = sorted(set(names) - set(scorers))
    if unknown:
        sys.exit(f"unknown ADBENCH_ARMS {unknown}; choose from {list(scorers)}")
    path = Path(OUT) / "adbench_arms.csv"
    rows = []
    for dataset in datasets:
        X, y = adbench.load(dataset)
        for split, seed in adbench.SPLITS.items():
            X_train, X_test, _, y_test = adbench.adbench_split(X, y, seed)
            cell: dict[str, np.ndarray] = {}
            for name in [*names, "+".join(combine)] if combine else names:
                t0 = time.time()
                if combine and name == "+".join(combine):
                    missing = sorted(set(combine) - set(cell))
                    if missing:
                        sys.exit(f"ADBENCH_COMBINE names arms that were not computed: {missing}")
                    scores = {name: rank_mean([cell[a] for a in combine])}
                else:
                    scores = scorers[name](X_train, X_test, seed)
                    cell |= scores
                seconds = round(time.time() - t0, 1)
                for arm, s in scores.items():
                    s = np.asarray(s, dtype=float)
                    if s.shape != (len(X_test),) or not np.isfinite(s).all():
                        raise ValueError(f"{dataset} {split} {arm}: scores are not finite")
                    rows.append(
                        dict(
                            dataset=dataset,
                            split=split,
                            seed=seed,
                            arm=arm,
                            auroc=float(roc_auc_score(y_test, s)),
                            aupr=float(average_precision_score(y_test, s)),
                            seconds=seconds,
                            n_context=len(X_train),
                            n_test=len(X_test),
                        )
                    )
                    print(
                        f"[adbench] {dataset} {split} {arm}: AUROC {rows[-1]['auroc']:.4f}"
                        f" ({seconds:.0f}s)",
                        flush=True,
                    )
                pd.DataFrame(rows).to_csv(path, index=False)
    df = pd.DataFrame(rows)
    by_table = df.pivot_table(index="dataset", columns="arm", values="auroc", sort=False)
    print(f"\nAUROC by table (mean of the splits)\n{by_table.round(4).to_string()}")
    mean = df.groupby("arm", sort=False)[["auroc", "aupr"]].mean()
    mean["seconds"] = df.groupby("arm", sort=False).seconds.sum()
    print(f"\nmean over {df.groupby('arm').size().iloc[0]} cells\n{mean.round(4).to_string()}")


if __name__ == "__main__":
    main()
