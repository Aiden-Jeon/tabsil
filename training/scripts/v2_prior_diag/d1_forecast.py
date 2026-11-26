"""D1: does TabSilForecaster extrapolate a trend beyond the context's index range?

Synthetic daily series: level + trend + weekly and monthly seasonality + noise. Trend strength is
the trend's rise over the horizon in units of the noise sd (0 = control). Metrics per series:
- trend capture: (mean forecast - mean of the last 28 context values) over the same for the
  true future mean (1 = the trend continues, 0 = flat)
- MASE (seasonal period 7) of TabSil's median, of seasonal naive, and of OLS on
  [index, weekly/monthly sin/cos] (a linear model that extrapolates by construction)
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd
from common import DEVICE, OUT

from tabsil import TabSilForecaster

L, H = 365, 56
SEEDS = range(4)


def series(kind: str, strength: float, rng):
    t = np.arange(L + H, dtype=float)
    noise_sd = 1.0
    season = 2.0 * np.sin(2 * np.pi * t / 7) + 1.5 * np.sin(2 * np.pi * t / 30.4 + 1.0)
    if kind == "linear":
        trend = strength * noise_sd * t / H
    elif kind == "exp":  # same rise over the horizon at the end, growing
        r = 0.01
        trend = strength * noise_sd * (np.exp(r * t) - 1) / (np.exp(r * (L + H)) - np.exp(r * L))
    elif kind == "piecewise":  # flat, then a slope change at 70% of the context
        k = int(0.7 * L)
        trend = strength * noise_sd * np.maximum(t - k, 0) / H
    else:
        raise ValueError(kind)
    y = 10 + trend + season + rng.normal(0, noise_sd, L + H)
    return y


def design(t):
    return np.column_stack(
        [np.ones_like(t), t] + [f(2 * np.pi * t / p) for p in (7, 30.4) for f in (np.sin, np.cos)]
    )


def main() -> None:
    fc = TabSilForecaster(device=DEVICE)
    rows = []
    t0 = time.time()
    ts = pd.date_range("2020-01-01", periods=L + H, freq="D")
    for kind in ("linear", "exp", "piecewise"):
        for strength in (0.0, 2.0, 5.0, 15.0):
            if kind != "linear" and strength == 0.0:
                continue
            for seed in SEEDS:
                rng = np.random.default_rng(seed)
                y = series(kind, strength, rng)
                ctx = pd.DataFrame({"timestamp": ts[:L], "target": y[:L]})
                pred = fc.fit(ctx).predict(horizon=H)
                p = pred["predictions"].to_numpy(float)[:H]
                true = y[L:]
                last = y[L - 28 : L].mean()
                denom = true.mean() - last
                capture = (p.mean() - last) / denom if abs(denom) > 0.5 else np.nan
                scale = np.abs(y[7:L] - y[: L - 7]).mean()
                snaive = np.array([y[L - 7 + (h % 7)] for h in range(H)])
                t = np.arange(L + H, dtype=float)
                beta, *_ = np.linalg.lstsq(design(t[:L]), y[:L], rcond=None)
                ols = design(t[L:]) @ beta
                rows.append(
                    dict(
                        kind=kind,
                        strength=strength,
                        seed=seed,
                        capture=capture,
                        mase_tabsil=np.abs(p - true).mean() / scale,
                        mase_snaive=np.abs(snaive - true).mean() / scale,
                        mase_ols=np.abs(ols - true).mean() / scale,
                        max_pred_minus_ctx_max=p.max() - y[:L].max(),
                        true_max_minus_ctx_max=true.max() - y[:L].max(),
                    )
                )
            print(f"{kind} {strength} done {time.time() - t0:.0f}s", file=sys.stderr, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "d1_forecast.csv"), index=False)
    summary = df.groupby(["kind", "strength"]).mean(numeric_only=True).drop(columns="seed")
    print(summary.round(3).to_string())


if __name__ == "__main__":
    main()
