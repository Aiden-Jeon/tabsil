"""D5: how does the error of TabSil's treatment effect estimates grow as overlap shrinks?
(`training/docs/v2-task-formulations.md` section 6, plan item 1.5)

Tables with a known effect: 5 standard normal features, potential outcomes from two random tanh
networks (the outcome without treatment, mu0, with unit variance, and the effect tau with mean 1
and standard deviation 0.5), noise of standard deviation 0.5. The treatment's propensity is
sigmoid(`STEEPNESS` * u) clipped to [c, 1 - c], where u is a standardized combination of two
features that also enter the outcome. c = 0.5 is a randomized experiment; at c = 0.02 almost every
row has a propensity of 0.02 or 0.98. `N_TRAIN` rows are observed and `N_TEST` rows are scored.

Reported per clip (`CLIPS`) and learner, over `REALIZATIONS` tables: the root PEHE (the root mean
squared error of the estimated effect against tau) on the test rows, and on the test rows whose
unclipped propensity is below 0.1 or above 0.9 (`pehe_edge`: the rows where one arm is rare).
Learners: TabSil's S-learner and T-learner (`TabSilCATE`).
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

N_TRAIN, N_TEST, N_FEATURES = 1000, 1000, 5
CLIPS = (0.5, 0.3, 0.2, 0.1, 0.05, 0.02)
REALIZATIONS = range(10)
LEARNERS = ("s", "t")
STEEPNESS = 4.0
NOISE = 0.5
MIN_GROUP = 5  # rows a treatment arm needs (the T-learner fits each arm)


def network(rng: np.random.Generator, X: np.ndarray) -> np.ndarray:
    """A random one-hidden-layer tanh network of the features, standardized over the rows."""
    hidden = np.tanh(X @ rng.normal(size=(X.shape[1], 8)) + rng.normal(size=8))
    out = hidden @ rng.normal(size=8)
    return (out - out.mean()) / out.std()


def make_table(seed: int, clip: float) -> dict:
    """One table: features, treatment, outcome of the observed rows, features and true effect of
    the test rows, and the unclipped propensity of the test rows. The functions and the features
    depend on `seed` only, so the clips of one realization share them."""
    rng = np.random.default_rng([seed, 11])
    n = N_TRAIN + N_TEST
    X = rng.normal(size=(n, N_FEATURES))
    mu0 = network(rng, X)
    tau = 1.0 + 0.5 * network(rng, X)
    direction = rng.normal(size=2)
    u = X[:, :2] @ direction / np.linalg.norm(direction)
    raw = 1.0 / (1.0 + np.exp(-STEEPNESS * u))
    noise, uniform = rng.normal(0, NOISE, n), rng.random(n)
    propensity = np.clip(raw, clip, 1 - clip)
    t = (uniform < propensity).astype(int)
    y = mu0 + t * tau + noise
    train, test = slice(0, N_TRAIN), slice(N_TRAIN, n)
    return dict(X=X[train], t=t[train], y=y[train], X_test=X[test], tau=tau[test], raw=raw[test])


def pehe(estimate: np.ndarray, tau: np.ndarray) -> float:
    return float(np.sqrt(np.mean((estimate - tau) ** 2)))


def main() -> None:
    from common import DEVICE, OUT, load_model

    from tabsil import TabSilCATE

    model, borders = load_model()
    rows = []
    t0 = time.time()
    for clip in CLIPS:
        for seed in REALIZATIONS:
            table = make_table(seed, clip)
            if min(table["t"].sum(), (1 - table["t"]).sum()) < MIN_GROUP:
                raise ValueError(f"clip {clip} seed {seed}: a treatment arm has too few rows")
            edge = (table["raw"] < 0.1) | (table["raw"] > 0.9)
            for learner in LEARNERS:
                est = TabSilCATE(model, borders, learner=learner, device=DEVICE, random_state=seed)
                effect = est.fit(table["X"], table["t"], table["y"]).effect(table["X_test"])
                rows.append(
                    dict(
                        clip=clip,
                        seed=seed,
                        learner=learner,
                        pehe=pehe(effect, table["tau"]),
                        pehe_edge=pehe(effect[edge], table["tau"][edge]),
                        ate_error=float(effect.mean() - table["tau"].mean()),
                        treated=float(table["t"].mean()),
                        edge_share=float(edge.mean()),
                    )
                )
        print(f"clip {clip} done {time.time() - t0:.0f}s", file=sys.stderr, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "d5_overlap.csv"), index=False)
    for value in ("pehe", "pehe_edge"):
        wide = df.pivot_table(index="clip", columns="learner", values=value, sort=False)
        spread = df.pivot_table(
            index="clip", columns="learner", values=value, aggfunc="sem", sort=False
        )
        print(f"\nroot {value} (mean of {len(REALIZATIONS)} tables)\n{wide.round(3).to_string()}")
        print(f"standard error\n{spread.round(3).to_string()}")


if __name__ == "__main__":
    main()
