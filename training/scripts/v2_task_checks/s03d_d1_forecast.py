"""Check for v2-task-formulations.md section 3, step 7 (the D1 table): seasonal naive and least
squares columns, which need no model weights.

Claim reproduced: "Seasonal naive and least squares reproduce D1's columns (3.674 and 0.368), so the
series are the same." Seasonal naive MASE 3.67 with trend capture 0.28, least squares MASE 0.37 with
trend capture 0.99 (linear trend, strength 15, seeds 0..3).

Design: the D1 series generator and the metrics of training/scripts/v2_prior_diag/d1_forecast.py,
copied here so that this check needs no model (the script imports the model; this one does not).
MASE: mean absolute error of the forecast over the 56 horizon steps divided by the in-sample
seasonal-naive (period 7) mean absolute error of the context. Trend capture: (mean forecast -
mean of the last 28 context values) / (true future mean - same). Least squares: OLS on
[1, index, weekly and monthly sin/cos] fitted to the context. Mean over seeds 0..3.
Tolerance: 0.002 on MASE (the stated values are to three decimals).
"""

import numpy as np

L, H = 365, 56


def series(strength, rng):
    t = np.arange(L + H, dtype=float)
    season = 2.0 * np.sin(2 * np.pi * t / 7) + 1.5 * np.sin(2 * np.pi * t / 30.4 + 1.0)
    trend = strength * t / H
    return 10 + trend + season + rng.normal(0, 1.0, L + H)


def design(t):
    return np.column_stack(
        [np.ones_like(t), t] + [f(2 * np.pi * t / p) for p in (7, 30.4) for f in (np.sin, np.cos)]
    )


rows = []
for seed in range(4):
    y = series(15.0, np.random.default_rng(seed))
    true = y[L:]
    last = y[L - 28 : L].mean()
    denom = true.mean() - last
    scale = np.abs(y[7:L] - y[: L - 7]).mean()
    snaive = np.array([y[L - 7 + (h % 7)] for h in range(H)])
    t = np.arange(L + H, dtype=float)
    beta, *_ = np.linalg.lstsq(design(t[:L]), y[:L], rcond=None)
    ols = design(t[L:]) @ beta
    rows.append(
        (
            np.abs(snaive - true).mean() / scale,
            (snaive.mean() - last) / denom,
            np.abs(ols - true).mean() / scale,
            (ols.mean() - last) / denom,
        )
    )
    print(
        f"seed {seed}: snaive MASE {rows[-1][0]:.3f} capture {rows[-1][1]:+.3f} | "
        f"OLS MASE {rows[-1][2]:.3f} capture {rows[-1][3]:+.3f}"
    )
r = np.array(rows).mean(axis=0)
print(
    f"mean over seeds: seasonal naive MASE {r[0]:.4f} (stated 3.674), "
    f"capture {r[1]:+.3f} (stated 0.28)"
)
print(
    f"                 least squares MASE {r[2]:.4f} (stated 0.368), "
    f"capture {r[3]:+.3f} (stated 0.99)"
)
assert abs(r[0] - 3.674) < 0.002, r[0]
assert abs(r[2] - 0.368) < 0.002, r[2]
print("PASS s03d")
