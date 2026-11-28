"""D2, rare class: what probability does TabSilClassifier give a class with 1 to 3 context rows
when the labels are independent of the features?

`training/docs/v2-task-formulations.md` section 2 derives that the training split, which keeps a
row of every class among the context rows, lowers a class's odds by n_c / (n_c + 1) against the
posterior, so that with labels that do not depend on the features and a uniform prior on the class
probabilities the output is near the class's share of the context rows, n_c / n, and not the
posterior's (n_c + 1) / (n + K).

Tables: n context rows of K classes, features of standard normal noise. The rarest class has n_c
context rows (`RARE`), the other classes share the rest equally. 500 query rows of the same noise.
Reported per (K, n, n_c, feature count), mean over the query rows and the seeds:

- `p_model`: the predicted probability of the rarest class
- `share`: n_c / n, and `posterior`: (n_c + 1) / (n + K)
- `pseudo_count`: the a for which (n_c + a) / (n + K a) equals `p_model`; 0 is the share of the
  context rows and 1 the posterior under a uniform prior
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd
from common import DEVICE, OUT, load_model

from tabsil import TabSilClassifier

N_QUERY = 500
SEEDS = range(5)
CLASSES = (2, 3, 10)
CONTEXT_ROWS = (50, 200, 1000)
RARE = (1, 2, 3, 10)
FEATURES = (1, 5)


def labels(n: int, n_classes: int, n_rare: int, rng: np.random.Generator) -> np.ndarray:
    """n labels in random order: class 0 has `n_rare` rows, the others share the rest (the first
    of them take the remainder)."""
    rest = np.arange(n - n_rare) % (n_classes - 1) + 1
    return rng.permutation(np.r_[np.zeros(n_rare, int), rest])


def pseudo_count(p: float, n: int, n_classes: int, n_rare: int) -> float:
    """a with (n_rare + a) / (n + n_classes a) = p."""
    return (p * n - n_rare) / (1 - n_classes * p)


def main() -> None:
    model, _ = load_model()
    clf = TabSilClassifier(model, device=DEVICE)
    rows = []
    t0 = time.time()
    for n_classes in CLASSES:
        for n in CONTEXT_ROWS:
            for n_rare in RARE:
                # every other class needs more rows than the rarest
                if (n - n_rare) // (n_classes - 1) <= n_rare:
                    continue
                for n_feat in FEATURES:
                    for seed in SEEDS:
                        rng = np.random.default_rng([seed, n_classes, n, n_rare, n_feat])
                        x = rng.normal(size=(n + N_QUERY, n_feat))
                        y = labels(n, n_classes, n_rare, rng)
                        clf.random_state = seed
                        proba = clf.fit(x[:n], y).predict_proba(x[n:])
                        p = float(proba[:, list(clf.classes_).index(0)].mean())
                        rows.append(
                            dict(
                                classes=n_classes,
                                n=n,
                                n_rare=n_rare,
                                n_feat=n_feat,
                                seed=seed,
                                p_model=p,
                                share=n_rare / n,
                                posterior=(n_rare + 1) / (n + n_classes),
                                pseudo_count=pseudo_count(p, n, n_classes, n_rare),
                            )
                        )
            print(f"{n_classes} {n} done {time.time() - t0:.0f}s", file=sys.stderr, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "d2_rare_class.csv"), index=False)
    columns = ["p_model", "share", "posterior", "pseudo_count"]
    by_cell = df.groupby(["classes", "n", "n_rare", "n_feat"])[columns].mean()
    print(by_cell.round(4).to_string())
    by_count = df.groupby("n_rare")["pseudo_count"].agg(["mean", "std", "count"])
    print(f"\npseudo count by the rarest class's context rows\n{by_count.round(3).to_string()}")


if __name__ == "__main__":
    main()
