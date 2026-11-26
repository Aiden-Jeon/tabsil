"""D2: is TabSil calibrated when the target is unpredictable (independent of x)?

Hypothesis: the v1 prior drops unpredictable datasets (filter_unpredictable), so on
near-independent conditionals (which the chain rule of TabSilUnsupervised calls often) the
predictive distribution may be overconfident. Measured as test NLL against
- the true marginal density (oracle, the best any model can do when y is independent of x)
- a Gaussian KDE fitted to the train targets only (a model that ignores x)
A predictable control (y = sin(2 x0) + noise) checks the measurement against the true conditional.
Classification: random labels with marginal p, log loss against the marginal's entropy.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd
from common import DEVICE, OUT, load_model, log_density
from scipy import stats

from tabsil import TabSilClassifier, TabSilRegressor

N_TEST = 500
SEEDS = range(3)


def targets(kind: str, n: int, rng):
    if kind == "normal":
        return rng.normal(0, 1, n), stats.norm(0, 1).logpdf
    if kind == "lognormal":
        return rng.lognormal(0, 0.8, n), stats.lognorm(0.8).logpdf
    if kind == "student_t3":
        return rng.standard_t(3, n), stats.t(3).logpdf
    if kind == "bimodal":
        z = rng.random(n) < 0.3
        y = np.where(z, rng.normal(-2, 0.4, n), rng.normal(1.5, 0.7, n))
        return y, lambda v: np.logaddexp(
            np.log(0.3) + stats.norm(-2, 0.4).logpdf(v),
            np.log(0.7) + stats.norm(1.5, 0.7).logpdf(v),
        )
    raise ValueError(kind)


def main() -> None:
    model, borders = load_model()
    reg = TabSilRegressor(model, borders, device=DEVICE)
    rows = []
    t0 = time.time()
    for n_train in (100, 500, 2000):
        for n_feat in (1, 5):
            for kind in ("normal", "lognormal", "student_t3", "bimodal", "predictable"):
                for seed in SEEDS:
                    rng = np.random.default_rng(seed)
                    n = n_train + N_TEST
                    x = rng.normal(size=(n, n_feat))
                    if kind == "predictable":
                        y = np.sin(2 * x[:, 0]) + rng.normal(0, 0.3, n)
                        true_lp = stats.norm(np.sin(2 * x[n_train:, 0]), 0.3).logpdf(y[n_train:])
                        kde_lp = np.full(N_TEST, np.nan)
                    else:
                        y, logpdf = targets(kind, n, rng)
                        true_lp = logpdf(y[n_train:])
                        kde = stats.gaussian_kde(y[:n_train])
                        kde_lp = np.log(np.maximum(kde(y[n_train:]), 1e-300))
                    reg.fit(x[:n_train], y[:n_train])
                    lp, b = reg.predict_distribution(x[n_train:])
                    m_lp = log_density(lp, b, y[n_train:])
                    rows.append(
                        dict(
                            task="reg",
                            kind=kind,
                            n_train=n_train,
                            n_feat=n_feat,
                            seed=seed,
                            nll_model=-m_lp.mean(),
                            nll_true=-true_lp.mean(),
                            nll_kde=np.nan if kind == "predictable" else -kde_lp.mean(),
                        )
                    )
                elapsed = time.time() - t0
                print(
                    f"reg {n_train} {n_feat} {kind} done {elapsed:.0f}s",
                    file=sys.stderr,
                    flush=True,
                )

    clf = TabSilClassifier(model, device=DEVICE)
    p = np.array([0.6, 0.3, 0.1])
    ent = -(p * np.log(p)).sum()
    for n_train in (100, 500, 2000):
        for n_feat in (1, 5):
            for seed in SEEDS:
                rng = np.random.default_rng(seed)
                n = n_train + N_TEST
                x = rng.normal(size=(n, n_feat))
                y = rng.choice(3, size=n, p=p)
                clf.fit(x[:n_train], y[:n_train])
                proba = clf.predict_proba(x[n_train:])
                # probability of the true class through classes_ (a class missing from the train
                # rows has no column and gets probability 0)
                col = {c: j for j, c in enumerate(clf.classes_)}
                p_true = np.array(
                    [proba[i, col[c]] if c in col else 0.0 for i, c in enumerate(y[n_train:])]
                )
                ll = -np.log(np.maximum(p_true, 1e-12)).mean()
                # model that ignores x: train label frequencies (stored in the nll_kde column)
                freq = np.bincount(y[:n_train], minlength=3) / n_train
                ll_freq = -np.log(np.maximum(freq[y[n_train:]], 1e-12)).mean()
                rows.append(
                    dict(
                        task="clf",
                        kind="random3",
                        n_train=n_train,
                        n_feat=n_feat,
                        seed=seed,
                        nll_model=ll,
                        nll_true=ent,
                        nll_kde=ll_freq,
                    )
                )
        print(f"clf {n_train} done {time.time() - t0:.0f}s", file=sys.stderr, flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "d2_calibration.csv"), index=False)
    df["excess_vs_true"] = df.nll_model - df.nll_true
    df["excess_vs_kde"] = df.nll_model - df.nll_kde
    columns = ["nll_model", "nll_true", "nll_kde", "excess_vs_true", "excess_vs_kde"]
    g = df.groupby(["task", "kind", "n_train", "n_feat"])[columns].mean()
    print(g.round(3).to_string())


if __name__ == "__main__":
    main()
