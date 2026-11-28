"""D1: does TabSilForecaster extrapolate a trend beyond the context's index range?

Synthetic daily series: level + trend + weekly and monthly seasonality + noise. Trend strength is
the trend's rise over the horizon in units of the noise sd (0 = control). Metrics per series:
- trend capture: (mean forecast - mean of the last 28 context values) over the same for the
  true future mean (1 = the trend continues, 0 = flat)
- MASE (seasonal period 7) of TabSil's median, of seasonal naive, and of OLS on
  [index, weekly/monthly sin/cos] (a linear model that extrapolates by construction)
- the largest forecast minus the largest context value (the summary also gives its maximum over
  seeds)

The inference arms (`D1_ARMS`, comma-separated names of `arms.FORECAST_ARMS`, each optionally
with `+trend`) separate the model from the 0.1.1 inference recipe, which can also keep forecasts
inside the context range: the target-power members clip their distribution to the train target
range (`_clipped_inverse` in `tabsil.inference.ensemble`), and the "quantile" and "robust" feature
kinds saturate a running index beyond the train rows. The two arms of the first report are the
default:
- default: the weights' saved inference settings (0.1.1: 8 members, numeric-cycle kinds, half of
  them with the target power transform)
- plain: 2 members, both "standard" and without the target power transform (members 0 and 1 of
  the default cycle), so nothing in the recipe bounds the forecast
`arms.py` has the others: the two changes one at a time with 8 members, and a straight-line trend
fitted outside the model.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd
from arms import ArmForecaster
from common import DEVICE, OUT, load_model

from tabsil import TabSilForecaster

L, H = 365, 56
SEEDS = range(4)
ARMS = [a.strip() for a in os.environ.get("D1_ARMS", "default,plain").split(",") if a.strip()]


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
    model, borders = load_model()
    for arm in ARMS:
        run_arm(arm, ArmForecaster(model, borders, arm, device=DEVICE))


def run_arm(arm: str, fc: TabSilForecaster) -> None:
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
                        arm=arm,
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
            elapsed = time.time() - t0
            print(f"{arm} {kind} {strength} done {elapsed:.0f}s", file=sys.stderr, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, f"d1_forecast_{arm}.csv"), index=False)
    groups = df.groupby(["kind", "strength"])
    summary = groups.mean(numeric_only=True).drop(columns="seed")
    summary["max_pred_minus_ctx_max_seed_max"] = groups.max_pred_minus_ctx_max.max()
    print(f"arm {arm}\n{summary.round(3).to_string()}")


if __name__ == "__main__":
    main()
