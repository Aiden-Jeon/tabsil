"""Bradley-Terry Elo of a pool of methods, shared by the suites that rank TabSil against copied
reference results (`adbench`, `fevbench`).

The computation is fm-agent-bench's `fab/bench/tabarena/elo.py` (TabArena's): every pair of
methods is compared on every unit (a split of a table or task), the lower error wins and a tie
counts half; the units of one group (table, task) weigh 1 in total; the anchor method is moved to
1000; the 95% interval comes from bootstrap fits over groups.

- `fit`: the maximum likelihood fit on an error matrix
- `table`: Elo, interval, mean rank and win rate per method from a methods x units table

Each suite keeps its own `elo` function around `table`: the input column names, the checks and
the extra columns (`imputed`, the group count) differ.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# fm-agent-bench elo.py: Elo points per factor 10 in odds, the level of the fit before the anchor
# is moved, bootstrap rounds and their seed
SCALE, INIT, ROUNDS, SEED = 400, 1000.0, 200, 0


def fit(err: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Bradley-Terry maximum likelihood (logistic regression, C = 1e6): err (methods, units)
    without gaps, weight per unit -> uncalibrated Elo per method. In a pair the lower error wins,
    ties count half."""
    from sklearn.linear_model import LogisticRegression

    m = err.shape[0]
    i, j = np.triu_indices(m, 1)
    a, b = err[i], err[j]
    win_i = ((a < b) + 0.5 * (a == b)) @ weight
    win_j = ((a > b) + 0.5 * (a == b)) @ weight
    x = np.zeros((len(i), m))
    x[np.arange(len(i)), i], x[np.arange(len(i)), j] = np.log(10), -np.log(10)
    keep = (win_i + win_j) > 0
    X = np.vstack([x[keep], x[keep]])
    y = np.r_[np.ones(keep.sum()), np.zeros(keep.sum())]
    w = np.r_[win_i[keep], win_j[keep]]
    lr = LogisticRegression(fit_intercept=False, C=1e6, tol=1e-10, max_iter=1000)
    lr.fit(X, y, sample_weight=w)
    return SCALE * lr.coef_[0] + INIT


def unit_weights(wide: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(group names, each unit's group index, each unit's weight) of a methods x units table whose
    columns are (group, split): the splits of a group weigh 1 in total."""
    groups = wide.columns.get_level_values(0).to_numpy()
    names, index = np.unique(groups, return_inverse=True)
    return names, index, 1.0 / np.bincount(index)[index]


def table(
    wide: pd.DataFrame, anchor: str, rounds: int = ROUNDS, seed: int = SEED, min_ci: int = 0
) -> pd.DataFrame:
    """Per method of `wide` (methods x units errors without gaps, columns (group, split)): `elo`
    with `anchor` at 1000, the 95% interval `elo+` / `elo-` (the 2.5% / 97.5% quantile of `rounds`
    bootstrap fits over groups minus the point estimate; NaN without rounds or with fewer than
    `min_ci` groups), and the weighted mean `rank` and `winrate`."""
    names, index, weight = unit_weights(wide)
    err = wide.to_numpy(float)
    raw = fit(err, weight)
    if rounds > 0 and len(names) >= min_ci:
        rng = np.random.default_rng(seed)
        boots = []
        for _ in range(rounds):
            pick = np.bincount(rng.integers(0, len(names), len(names)), minlength=len(names))
            w = weight * pick[index]
            boots.append(fit(err[:, w > 0], w[w > 0]))
        lo, hi = np.quantile(boots, 0.025, axis=0), np.quantile(boots, 0.975, axis=0)
    else:
        lo = hi = np.full_like(raw, np.nan)
    m = len(wide)
    rank = wide.rank(axis=0, method="average").to_numpy()
    return pd.DataFrame(
        {
            "elo": raw - raw[list(wide.index).index(anchor)] + INIT,
            "elo+": np.clip(hi - raw, 0, None),
            "elo-": np.clip(raw - lo, 0, None),
            "rank": (rank * weight).sum(1) / weight.sum(),
            "winrate": ((m - rank) / (m - 1) * weight).sum(1) / weight.sum(),
        },
        index=wide.index,
    )
