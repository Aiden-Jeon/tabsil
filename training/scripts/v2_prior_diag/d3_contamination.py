"""D3: how does TabSilUnsupervised's outlier score change as the context gets contaminated?

Inliers: a nonlinear 6-dimensional manifold (2 latent factors through random tanh maps + noise).
Outlier kinds (`D3_KINDS`, comma-separated):
- global: uniform in 1.2x the inlier bounding box (scattered, each outlier on its own)
- local: inliers with 2 random columns shifted by 3 sd
- cluster: a tight blob (0.15 sd) of similar anomalies about 2 sd off the inlier mean, the same
  blob in the context and the test rows
- curve: points on a 1-dimensional curve (every column 2 tanh(z1 + z2)) shifted 1.5 sd from the
  inlier mean
Context contamination 0, 2, 5, 10, 20% of 600 rows. Test: 300 inliers + 30 outliers of the same
kind. Reported: AUROC of TabSil (-log density, 3 column orders) and of IsolationForest, and the
mean log density of the test inliers and outliers.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd
from common import DEVICE, OUT, load_model
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score

from tabsil import TabSilUnsupervised

D, N_CTX, N_IN, N_OUT = 6, 600, 300, 30
SEEDS = range(3)
FRACTIONS = (0.0, 0.02, 0.05, 0.1, 0.2)
KINDS = os.environ.get("D3_KINDS", "global,local,cluster,curve").split(",")


def inliers(n, rng, W1, W2):
    z = rng.normal(size=(n, 2))
    return np.tanh(z @ W1) @ W2 + 0.1 * rng.normal(size=(n, D))


def outliers(kind, n, rng, ref, center_rng):
    mu, sd = ref.mean(0), ref.std(0)
    if kind == "global":
        lo, hi = ref.min(0), ref.max(0)
        mid, half = (lo + hi) / 2, (hi - lo) / 2 * 1.2
        return rng.uniform(mid - half, mid + half, size=(n, D))
    if kind == "local":
        x = ref[rng.integers(0, len(ref), n)].copy()
        for i in range(n):
            cols = rng.choice(D, 2, replace=False)
            x[i, cols] += 3 * sd[cols] * rng.choice([-1, 1], 2)
        return x
    if kind == "cluster":
        # center_rng is re-seeded per call, so the context and the test rows share one blob
        c = mu + 2.0 * sd * center_rng.choice([-1, 1], D) * center_rng.uniform(0.7, 1.0, D)
        return c + 0.15 * sd * rng.normal(size=(n, D))
    if kind == "curve":
        z = rng.normal(size=(n, 2))
        return 2.0 * np.tanh(z.sum(1, keepdims=True)) * np.ones((1, D)) + mu + 1.5 * sd
    raise ValueError(kind)


def main() -> None:
    weights, borders = load_model()
    rows = []
    t0 = time.time()
    for kind in KINDS:
        for frac in FRACTIONS:
            for seed in SEEDS:
                rng = np.random.default_rng(seed)
                W1, W2 = rng.normal(size=(2, 4)), rng.normal(size=(4, D))
                ref = inliers(5000, np.random.default_rng(100 + seed), W1, W2)
                # draw order: context inliers, context outliers, shuffle, test inliers, test
                # outliers (the order of the reported run)
                n_bad = int(round(frac * N_CTX))
                ctx_good = inliers(N_CTX - n_bad, rng, W1, W2)
                ctx_bad = outliers(kind, n_bad, rng, ref, np.random.default_rng(200 + seed))
                ctx = np.vstack([ctx_good, ctx_bad])
                ctx = ctx[rng.permutation(len(ctx))]
                test_good = inliers(N_IN, rng, W1, W2)
                test_bad = outliers(kind, N_OUT, rng, ref, np.random.default_rng(200 + seed))
                test = np.vstack([test_good, test_bad])
                lab = np.r_[np.zeros(N_IN), np.ones(N_OUT)]
                scorer = TabSilUnsupervised(
                    weights, borders, device=DEVICE, n_permutations=3, random_state=seed
                )
                ld = scorer.fit(ctx).log_density(test)
                iso = IsolationForest(random_state=seed).fit(ctx)
                rows.append(
                    dict(
                        kind=kind,
                        frac=frac,
                        seed=seed,
                        auroc_tabsil=roc_auc_score(lab, -ld),
                        auroc_iforest=roc_auc_score(lab, -iso.score_samples(test)),
                        inlier_logdens=ld[:N_IN].mean(),
                        outlier_logdens=ld[N_IN:].mean(),
                    )
                )
            print(f"{kind} {frac} done {time.time() - t0:.0f}s", file=sys.stderr, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "d3_contamination.csv"), index=False)
    summary = df.groupby(["kind", "frac"]).mean(numeric_only=True).drop(columns="seed")
    print(summary.round(3).to_string())


if __name__ == "__main__":
    main()
