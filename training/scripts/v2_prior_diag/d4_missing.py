"""D4: imputation error of `TabSilUnsupervised.impute` when the cells are not missing completely
at random (`training/docs/v2-task-formulations.md` section 5, plan item 1.4).

Every mechanism masks a target column at the same rate (`RATE`, 30%). A target column j has one
driver column k; `r` is a column's standard normal score:

- `mcar`: the target is missing completely at random
- `by_other`: the target is missing with probability Phi(a + B_OTHER r_k): a split on a feature
- `other_obs`: the target is missing completely at random, and the driver is missing with
  probability Phi(a + B_OTHER r_j) in the rows that observe the target. This is missing at random
  in Rubin's sense, and the rows to impute all show the driver (section 5's counter-example)
- `other_all`: the same with the driver's mask in every row, so that whether the driver is shown
  tells about the target in the rows to impute too. This is the mechanism of `mask_by_target` in
  the prior `graph_scm_hicard_missing`
- `self`: the target is missing with probability Phi(a + B_SELF r_j)

`a` is set so that the rate is `RATE` (Phi(a / sqrt(1 + b^2)) = RATE).

Part `gaussian`: tables of `N_ROWS` rows with a standard normal target (column 0), a driver with
correlation `RHO` to it (column 1) and two independent noise columns, where the values the wrapper
converges to are known. Per mechanism and seed, on the missing target cells: the mean signed error
(`bias`, imputed minus true; also split by whether the row shows the driver) and the RMSE of

- `model`: TabSil
- `ols`: least squares on the rows that observe the target, with the driver (0 where it is
  missing), an indicator of its being missing and the noise columns: the linear version of what a
  fit on the complete cases gives
- `full`: RHO times the driver (0 where it is missing), the mean under the full distribution
  given the shown driver, which ignores what the masks tell

and for `self` the closed form of the bias of the complete cases' conditional mean
(`closed_bias`, section 5, averaged over the rows to impute).

Part `suite`: the imputation suite's tables (`tabsil_train.benchmarks.unsupervised.impute`, its
rows per seed). The numeric columns are paired by their absolute Spearman correlation, largest
first; in each pair the column with the lower index is the target and the other its driver, and
scores are the normal scores of the ranks. The NRMSE (RMSE over the column's standard deviation,
mean over the target columns) and the mean standardized signed error are computed on the missing
target cells for TabSil and, with `D4_BASELINES=1`, for MissForest and the mean.

`D4_PARTS` (default `gaussian,suite`) and `D4_TABLES` choose what runs.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata, spearmanr

MECHANISMS = ("mcar", "by_other", "other_obs", "other_all", "self")
RATE = 0.3
B_OTHER, B_SELF = 1.5, 1.0
SEEDS = (0, 1, 2)
N_ROWS, RHO = 2000, 0.6


def offset(b: float, rate: float = RATE) -> float:
    """a with E[Phi(a + b r)] = rate for a standard normal r."""
    return float(norm.ppf(rate) * np.sqrt(1 + b * b))


def normal_scores(X: np.ndarray) -> np.ndarray:
    """Each column's values replaced by the standard normal quantile of their rank (ties share
    their mean rank)."""
    ranks = np.column_stack([rankdata(X[:, j]) for j in range(X.shape[1])])
    return norm.ppf((ranks - 0.5) / len(X))


def make_mask(
    mechanism: str,
    scores: np.ndarray,
    pairs: list[tuple[int, int]],
    rng: np.random.Generator,
) -> np.ndarray:
    """The mask (True = missing) of one mechanism for the (target, driver) column `pairs`."""
    n = len(scores)
    mask = np.zeros(scores.shape, dtype=bool)
    for j, k in pairs:
        u, v = rng.random(n), rng.random(n)
        if mechanism == "mcar":
            mask[:, j] = u < RATE
        elif mechanism == "by_other":
            mask[:, j] = u < norm.cdf(offset(B_OTHER) + B_OTHER * scores[:, k])
        elif mechanism == "self":
            mask[:, j] = u < norm.cdf(offset(B_SELF) + B_SELF * scores[:, j])
        elif mechanism in ("other_obs", "other_all"):
            mask[:, j] = u < RATE
            hide = v < norm.cdf(offset(B_OTHER) + B_OTHER * scores[:, j])
            mask[:, k] = hide & ~mask[:, j] if mechanism == "other_obs" else hide
        else:
            raise ValueError(f"unknown mechanism {mechanism!r}: one of {MECHANISMS}")
    return mask


# ---------------------------------------------------------------------------
# Gaussian tables
# ---------------------------------------------------------------------------
def gaussian_table(seed: int) -> np.ndarray:
    rng = np.random.default_rng([seed, 4])
    z = rng.normal(size=(N_ROWS, 4))
    z[:, 1] = RHO * z[:, 0] + np.sqrt(1 - RHO**2) * z[:, 1]
    return z


def ols_imputation(X: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """The target (column 0) of every row from least squares on the rows that observe it:
    intercept, the driver (0 where it is missing), the driver's missing indicator, the other
    columns."""
    hidden = mask[:, 1]
    design = np.column_stack(
        [np.ones(len(X)), np.where(hidden, 0.0, X[:, 1]), hidden.astype(float), X[:, 2:]]
    )
    if not hidden.any():
        design = np.delete(design, 2, axis=1)
    observed = ~mask[:, 0]
    beta, *_ = np.linalg.lstsq(design[observed], X[observed, 0], rcond=None)
    return design @ beta


def self_masking_bias(driver: np.ndarray) -> np.ndarray:
    """E[x | driver, observed] - E[x | driver, missing] of the `self` mechanism at each driver
    value (section 5, "With a predictive column")."""
    s = np.sqrt(1 - RHO**2)
    scale = np.sqrt(1 + B_SELF**2 * s**2)
    alpha = (offset(B_SELF) + B_SELF * RHO * driver) / scale
    return -s * (B_SELF * s / scale) * norm.pdf(alpha) / (norm.cdf(alpha) * norm.sf(alpha))


def errors(prefix: str, imputed: np.ndarray, truth: np.ndarray, shown: np.ndarray) -> dict:
    """Bias and RMSE of the imputed target cells, the bias also by whether the row shows the
    driver (NaN for an empty group)."""
    e = imputed - truth
    return {
        f"{prefix}_bias": float(e.mean()),
        f"{prefix}_rmse": float(np.sqrt(np.mean(e**2))),
        f"{prefix}_bias_shown": float(e[shown].mean()) if shown.any() else np.nan,
        f"{prefix}_bias_hidden": float(e[~shown].mean()) if (~shown).any() else np.nan,
    }


def gaussian_references(X: np.ndarray, mask: np.ndarray, mechanism: str) -> dict:
    """The reference values of one Gaussian table and mask (no model)."""
    rows = mask[:, 0]
    shown = ~mask[rows, 1]
    truth = X[rows, 0]
    full = np.where(mask[:, 1], 0.0, RHO * X[:, 1])
    out = {
        "missing": int(rows.sum()),
        "driver_hidden": float(mask[rows, 1].mean()),
        **errors("ols", ols_imputation(X, mask)[rows], truth, shown),
        **errors("full", full[rows], truth, shown),
    }
    out["closed_bias"] = (
        float(self_masking_bias(X[rows, 1]).mean()) if mechanism == "self" else np.nan
    )
    return out


def run_gaussian(imputer) -> pd.DataFrame:
    rows = []
    for mechanism in MECHANISMS:
        for seed in SEEDS:
            X = gaussian_table(seed)
            mask = make_mask(mechanism, X, [(0, 1)], np.random.default_rng([seed, 5]))
            masked = np.where(mask, np.nan, X)
            imputed = imputer(masked, [], seed)["tabsil"]
            target = mask[:, 0]
            rows.append(
                dict(
                    mechanism=mechanism,
                    seed=seed,
                    **errors("model", imputed[target, 0], X[target, 0], ~mask[target, 1]),
                    **gaussian_references(X, mask, mechanism),
                )
            )
        print(f"gaussian {mechanism} done", file=sys.stderr, flush=True)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# The imputation suite's tables
# ---------------------------------------------------------------------------
def column_pairs(X: np.ndarray, categorical: list[int]) -> list[tuple[int, int]]:
    """(target, driver) pairs of numeric columns: the pair with the largest absolute Spearman
    correlation first, then the largest among the columns left, until fewer than two are left.
    Constant columns are left out."""
    numeric = [j for j in range(X.shape[1]) if j not in categorical and np.ptp(X[:, j]) > 0]
    if len(numeric) < 2:
        return []
    corr = np.abs(np.atleast_2d(spearmanr(X[:, numeric]).statistic))
    if corr.shape == (1, 1):  # two columns: scipy returns the one correlation
        corr = np.array([[1.0, corr[0, 0]], [corr[0, 0], 1.0]])
    np.fill_diagonal(corr, -1.0)
    pairs = []
    while (corr > -1).any():
        a, b = np.unravel_index(np.argmax(corr), corr.shape)
        pairs.append((numeric[min(a, b)], numeric[max(a, b)]))
        corr[[a, b], :] = -1.0
        corr[:, [a, b]] = -1.0
    return pairs


def suite_scores(X: np.ndarray, imputed: np.ndarray, mask: np.ndarray, targets: list[int]) -> dict:
    """NRMSE and mean standardized signed error over the target columns' missing cells."""
    nrmse, bias = [], []
    for j in targets:
        m = mask[:, j]
        e = (imputed[m, j] - X[m, j]) / X[:, j].std()
        nrmse.append(float(np.sqrt(np.mean(e**2))))
        bias.append(float(e.mean()))
    return {"nrmse": float(np.mean(nrmse)), "bias": float(np.mean(bias))}


def run_suite(imputers: dict, tables: list[str]) -> pd.DataFrame:
    from tabsil_train.benchmarks.unsupervised import impute

    rows = []
    for name in tables:
        full, categorical, _ = impute.load(name)
        for seed in SEEDS:
            X, _ = impute.make_split(name, full, RATE, seed)
            pairs = column_pairs(X, categorical)
            if not pairs:
                raise ValueError(f"{name}: fewer than two numeric columns")
            targets = [j for j, _ in pairs]
            scores = normal_scores(X)
            for mechanism in MECHANISMS:
                rng = np.random.default_rng([seed, impute.table_key(name), 6])
                mask = make_mask(mechanism, scores, pairs, rng)
                masked = np.where(mask, np.nan, X)
                for method, imputer in imputers.items():
                    t0 = time.time()
                    imputed = imputer(masked, categorical, seed)
                    imputed = imputed["tabsil"] if isinstance(imputed, dict) else imputed
                    if not np.isfinite(imputed[mask]).all():
                        raise ValueError(f"{name} {mechanism} {method}: non-finite imputation")
                    rows.append(
                        dict(
                            dataset=name,
                            mechanism=mechanism,
                            seed=seed,
                            method=method,
                            **suite_scores(X, imputed, mask, targets),
                            targets=len(targets),
                            seconds=round(time.time() - t0, 1),
                        )
                    )
        print(f"suite {name} done", file=sys.stderr, flush=True)
    return pd.DataFrame(rows)


def main() -> None:
    from common import DEVICE, OUT, load_model

    from tabsil_train.benchmarks.unsupervised import impute

    parts = os.environ.get("D4_PARTS", "gaussian,suite").split(",")
    model, borders = load_model()
    tabsil = impute.tabsil_imputer(model, borders, None, DEVICE)
    if "gaussian" in parts:
        df = run_gaussian(tabsil)
        df.to_csv(os.path.join(OUT, "d4_missing_gaussian.csv"), index=False)
        mean = df.groupby("mechanism", sort=False).mean(numeric_only=True).drop(columns="seed")
        print(f"Gaussian tables (mean of {len(SEEDS)} seeds)\n{mean.round(3).T.to_string()}")
    if "suite" in parts:
        tables = [t for t in os.environ.get("D4_TABLES", "").split(",") if t] or list(impute.TABLES)
        imputers = {"tabsil": tabsil}
        if os.environ.get("D4_BASELINES") == "1":
            imputers |= {"mean": impute.impute_mean, "missforest": impute.impute_missforest}
        df = run_suite(imputers, tables)
        df.to_csv(os.path.join(OUT, "d4_missing_suite.csv"), index=False)
        for value in ("nrmse", "bias"):
            wide = df.pivot_table(index="mechanism", columns="method", values=value, sort=False)
            print(
                f"\nsuite tables, {value} (mean of tables and seeds)\n{wide.round(3).to_string()}"
            )


if __name__ == "__main__":
    main()
