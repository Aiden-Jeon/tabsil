"""Time series forecasting with the TabSil regression model (zero-shot, no retraining).

Each time step of a series becomes a table row of time features, and TabSil regression predicts
the rows of the forecast horizon from the context rows in one pass (the approach of TabPFN-TS,
Hoo et al. 2025, arXiv 2501.02945). Series are forecast one at a time, each with its own fit.

    fc = TabSilForecaster().fit(context)  # columns item_id, timestamp, target (+ covariates)
    pred = fc.predict(24)  # item_id, timestamp, predictions, "0.1" ... "0.9"

Features (`time_features`, the defaults of the TabPFN-TS fev wrapper: `RunningIndexFeature`,
`CalendarFeature`, `AutoSeasonalFeature`; no lags of the target):

- running index: 0, 1, 2, ... over the context and then the horizon
- calendar: the year, and sin/cos of second of minute, minute of hour, hour of day, day of week,
  day of month, day of year, week of year and month of year (the zero-based gluonts indices with
  the TabPFN-TS periods 60, 60, 24, 7, 30.5, 365, 52, 12; each sin/cos uses the period minus 1)
- seasonal: up to `MAX_SEASONAL_PERIODS` periods found in the context target by FFT (linear
  detrend, Hann window, zero padding x2, FFT peaks above 5% of the largest, periods rounded to
  integers), each as sin/cos of the running index; zero columns for periods not found
- known covariates: columns present in both the context and the `future` frame, as they are
  (non-numeric columns are categorical for the ensemble)

Context handling (our choices, following the TabPFN-TS fev wrapper where it has one):

- missing targets are filled forward, then backward, within a series; a series without any
  target value is all zeros. Missing covariate values are filled the same way within the context
  and within the horizon, and stay missing (NaN, which TabSil accepts) if a series has none
- a series uses at most its last `max_context_length` rows as context (default
  `DEFAULT_MAX_CONTEXT_LENGTH` = 4096, as TabPFN-TS and the TabICL forecaster). `None` uses up to
  TabSil's supported size (`tabsil.recipe.SUPPORTED_SIZE`, 1,000,000 training rows); a larger
  value is an error. The running index and the seasonal periods are computed on the kept rows
- a series whose context target is constant gets that value for every quantile and the point
  forecast (the model is not run: z-normalizing a constant target is undefined)

Prediction: the quantiles come from the mixture distribution of
`tabsil.recipe.TabSilRegressor.predict_distribution` (bucket probabilities on the common borders),
inverting its CDF with the density taken as uniform within a bucket. The point forecast is the
median of that distribution (`point_forecast="median"`, the forecast MASE and fev's median metrics
assume) or the regressor's mean (`"mean"`, `TabSilRegressor.predict`). The two outermost buckets
of TabSil's borders are wide (z beyond about ±4.3, up to ±22.6 of the context target), so a
quantile that lands there is coarse.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

DEFAULT_QUANTILE_LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
DEFAULT_MAX_CONTEXT_LENGTH = 4096
MAX_SEASONAL_PERIODS = 5
# (gluonts index feature, period) of the TabPFN-TS `CalendarFeature`
CALENDAR_PERIODS = (
    ("second_of_minute", 60.0),
    ("minute_of_hour", 60.0),
    ("hour_of_day", 24.0),
    ("day_of_week", 7.0),
    ("day_of_month", 30.5),
    ("day_of_year", 365.0),
    ("week_of_year", 52.0),
    ("month_of_year", 12.0),
)
POINT_FORECASTS = ("median", "mean")


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
def _calendar_index(timestamps: pd.DatetimeIndex, name: str) -> np.ndarray:
    """The zero-based calendar index of gluonts' `<name>_index` time features."""
    if name == "second_of_minute":
        return np.asarray(timestamps.second)
    if name == "minute_of_hour":
        return np.asarray(timestamps.minute)
    if name == "hour_of_day":
        return np.asarray(timestamps.hour)
    if name == "day_of_week":
        return np.asarray(timestamps.dayofweek)
    if name == "day_of_month":
        return np.asarray(timestamps.day) - 1
    if name == "day_of_year":
        return np.asarray(timestamps.dayofyear) - 1
    if name == "week_of_year":
        return np.asarray(timestamps.isocalendar().week, dtype=np.int64) - 1
    if name == "month_of_year":
        return np.asarray(timestamps.month) - 1
    raise ValueError(f"unknown calendar feature: {name}")


def seasonal_periods(
    values: np.ndarray, max_periods: int = MAX_SEASONAL_PERIODS, threshold: float = 0.05
) -> list[float]:
    """Dominant periods of a series by FFT, largest magnitude first (TabPFN-TS
    `AutoSeasonalFeature.find_seasonal_periods` with its default config).

    The series is linearly detrended, multiplied by a Hann window and zero-padded to twice its
    length. Local peaks of the FFT magnitude above `threshold` times the largest magnitude (the DC
    component excluded) are candidates (all bins if there is no such peak); the `max_periods`
    largest give periods 1 / frequency, rounded to integers, with zero and repeated periods
    dropped."""
    from scipy import fft
    from scipy.signal import find_peaks

    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    n = len(values)
    if n < 2:
        return []
    idx = np.arange(n)
    values = values - np.polyval(np.polyfit(idx, values, 1), idx)
    values = values * np.hanning(n)
    padded = np.zeros(2 * n)
    padded[:n] = values
    magnitudes = np.abs(fft.rfft(padded))
    freqs = np.fft.rfftfreq(2 * n, d=1.0)
    magnitudes[0] = 0.0
    peaks, _ = find_peaks(magnitudes, height=threshold * np.max(magnitudes))
    if len(peaks) == 0:
        peaks = np.arange(len(magnitudes))
    top = peaks[np.argsort(magnitudes[peaks])[::-1]][:max_periods]
    periods = np.zeros_like(freqs)
    periods[freqs > 0] = 1.0 / freqs[freqs > 0]
    top_periods = np.round(periods[top])
    keep = top_periods != 0
    top_periods, top = top_periods[keep], top[keep]
    if len(top_periods) > 0:
        first = np.unique(top_periods, return_index=True)[1]
        top_periods, top = top_periods[first], top[first]
    # largest magnitude first; ties keep the order above (as TabPFN-TS's stable sort)
    order = sorted(range(len(top)), key=lambda i: -magnitudes[top[i]])
    return [float(top_periods[i]) for i in order]


def time_features(
    timestamps: pd.DatetimeIndex | Sequence, context_target: np.ndarray
) -> pd.DataFrame:
    """Time features of one series (module docstring): rows are the context steps followed by
    the horizon steps (`timestamps`, in time order); the seasonal periods are found in
    `context_target` (the first `len(context_target)` rows)."""
    timestamps = pd.DatetimeIndex(timestamps)
    n = len(timestamps)
    index = np.arange(n)
    columns: dict[str, np.ndarray] = {
        "running_index": index.astype(float),
        "year": np.asarray(timestamps.year, dtype=float),
    }
    for name, period in CALENDAR_PERIODS:
        value = _calendar_index(timestamps, name).astype(np.int32)
        columns[f"{name}_sin"] = np.sin(2 * np.pi * value / (period - 1))
        columns[f"{name}_cos"] = np.cos(2 * np.pi * value / (period - 1))
    periods = seasonal_periods(context_target)
    for i in range(MAX_SEASONAL_PERIODS):
        if i < len(periods):
            columns[f"sin_#{i}"] = np.sin(2 * np.pi * index / periods[i])
            columns[f"cos_#{i}"] = np.cos(2 * np.pi * index / periods[i])
        else:
            columns[f"sin_#{i}"] = np.zeros(n)
            columns[f"cos_#{i}"] = np.zeros(n)
    return pd.DataFrame(columns)


# ---------------------------------------------------------------------------
# Quantiles of the bucket distribution
# ---------------------------------------------------------------------------
def bucket_quantiles(
    log_probs: torch.Tensor | np.ndarray, borders: torch.Tensor | np.ndarray, levels: Sequence
) -> np.ndarray:
    """Quantiles (n, len(levels)) of bucket distributions: log probabilities (n, K) on `borders`
    (K + 1,), with the density uniform within each bucket (the CDF is linear between borders)."""
    probs = np.exp(np.asarray(log_probs, dtype=np.float64))
    probs = probs / probs.sum(-1, keepdims=True)
    b = np.asarray(borders, dtype=np.float64)
    cdf = np.concatenate([np.zeros((len(probs), 1)), np.cumsum(probs, -1)], axis=-1)
    cdf[:, -1] = 1.0
    out = np.empty((len(probs), len(levels)))
    for j, q in enumerate(levels):
        # bucket k holds the quantile: cdf[k] < q <= cdf[k + 1]
        k = np.clip((cdf < q).sum(-1) - 1, 0, len(b) - 2)
        rows = np.arange(len(probs))
        p = probs[rows, k]
        share = np.where(p > 0, (q - cdf[rows, k]) / np.where(p > 0, p, 1.0), 0.0)
        out[:, j] = b[k] + np.clip(share, 0.0, 1.0) * (b[k + 1] - b[k])
    return out


# ---------------------------------------------------------------------------
# Forecaster
# ---------------------------------------------------------------------------
def _fill(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Fill missing values of `columns` forward, then backward."""
    if columns:
        frame[columns] = frame[columns].ffill().bfill()
    return frame


class TabSilForecaster:
    """Zero-shot forecasting with TabSil regression (module docstring for the features, context
    handling and quantiles).

    `model` and `borders` are given as for `tabsil.recipe.TabSilRegressor` (None: the default Hub
    repo at the v1 tag; a Hub repo id or a local folder; or a model with its `borders`); the
    weights are loaded once and shared by every series. `n_estimators` and the other keyword
    arguments go to each series' `TabSilRegressor` (default: the model's settings).

        fc = TabSilForecaster(device="cuda").fit(context)
        pred = fc.predict(24)
        pred = fc.predict(future=future)  # known covariates in `future` (and in `context`)
    """

    def __init__(
        self,
        model: nn.Module | str | Path | None = None,
        borders: torch.Tensor | None = None,
        *,
        max_context_length: int | None = DEFAULT_MAX_CONTEXT_LENGTH,
        point_forecast: str = "median",
        n_estimators: int | None = None,
        device: str = "cpu",
        random_state: int = 0,
        **regressor_kwargs: Any,
    ) -> None:
        from tabsil.recipe import HUB_REPOS, SUPPORTED_SIZE

        if point_forecast not in POINT_FORECASTS:
            raise ValueError(f"point_forecast is one of {POINT_FORECASTS}: {point_forecast}")
        limit = SUPPORTED_SIZE["max_train_rows"]
        if max_context_length is None:
            max_context_length = limit
        if not 1 <= max_context_length <= limit:
            raise ValueError(
                f"max_context_length must be between 1 and TabSil's supported size ({limit}"
                f" training rows): {max_context_length}"
            )
        if isinstance(model, nn.Module):
            if borders is None:
                raise ValueError("pass the regression borders together with a model")
        else:
            if borders is not None:
                raise ValueError("borders are only given with a model (downloads include them)")
            from tabsil.hub import model_or_pretrained

            model, borders = model_or_pretrained(model, HUB_REPOS["regression"], "regression")
        self.model = model
        self.borders = borders
        self.max_context_length = int(max_context_length)
        self.point_forecast = point_forecast
        self.n_estimators = n_estimators
        self.device = device
        self.random_state = random_state
        self.regressor_kwargs = regressor_kwargs

    def fit(
        self,
        context: pd.DataFrame,
        *,
        id_column: str = "item_id",
        timestamp_column: str = "timestamp",
        target_column: str = "target",
    ) -> TabSilForecaster:
        """Store the past of each series. `context` has a timestamp and a target column, an id
        column for several series (without it: one series), and optionally covariate columns
        (used only when `predict` gets them in `future`). Nothing is trained."""
        frame = context.copy()
        if id_column not in frame.columns:
            frame[id_column] = 0
        missing = {timestamp_column, target_column} - set(frame.columns)
        if missing:
            raise ValueError(f"context has no column {sorted(missing)}")
        frame[timestamp_column] = pd.to_datetime(frame[timestamp_column])
        frame[target_column] = pd.to_numeric(frame[target_column]).astype(float)
        self.id_column, self.timestamp_column = id_column, timestamp_column
        self.target_column = target_column
        self.covariates_ = [
            c for c in frame.columns if c not in (id_column, timestamp_column, target_column)
        ]
        self.series_ = {}
        for item, g in frame.groupby(id_column, sort=False):
            g = g.sort_values(timestamp_column, kind="stable").reset_index(drop=True)
            if g[timestamp_column].duplicated().any():
                raise ValueError(f"series {item!r} has repeated timestamps")
            self.series_[item] = g
        return self

    def predict(
        self,
        horizon: int | None = None,
        *,
        future: pd.DataFrame | None = None,
        quantile_levels: Sequence[float] = DEFAULT_QUANTILE_LEVELS,
        freq: str | None = None,
    ) -> pd.DataFrame:
        """Forecast every series: `horizon` steps after its last timestamp (at `freq`, default
        the frequency pandas infers from the series), or the rows of `future` (id and timestamp
        columns, plus the future values of known covariates). Returns one row per (series, step):
        the id, the timestamp, `predictions` (the point forecast) and one column per quantile
        level (`str(level)`), series in `fit` order."""
        if not hasattr(self, "series_"):
            raise RuntimeError("call fit before predict")
        if (horizon is None) == (future is None):
            raise ValueError("pass exactly one of horizon and future")
        levels = [float(q) for q in quantile_levels]
        if not levels or any(not 0 < q < 1 for q in levels):
            raise ValueError(f"quantile levels must be in (0, 1): {quantile_levels}")
        futures = self._futures(horizon, future, freq)
        frames = []
        for item, past in self.series_.items():
            steps = futures[item]
            point, quantiles = self._predict_series(past, steps, levels)
            frame = steps[[self.timestamp_column]].copy()
            frame.insert(0, self.id_column, item)
            frame["predictions"] = point
            for j, q in enumerate(levels):
                frame[str(q)] = quantiles[:, j]
            frames.append(frame)
        return pd.concat(frames, ignore_index=True)

    def _futures(self, horizon, future, freq) -> dict:
        ids, ts = self.id_column, self.timestamp_column
        out = {}
        if future is not None:
            future = future.copy()
            if ids not in future.columns:
                future[ids] = 0
            future[ts] = pd.to_datetime(future[ts])
            groups = dict(tuple(future.groupby(ids, sort=False)))
            for item, past in self.series_.items():
                if item not in groups:
                    raise ValueError(f"future has no rows for series {item!r}")
                steps = groups[item].sort_values(ts, kind="stable").reset_index(drop=True)
                if steps[ts].iloc[0] <= past[ts].iloc[-1]:
                    raise ValueError(f"future of series {item!r} starts within its context")
                out[item] = steps
            unknown = set(groups) - set(self.series_)
            if unknown:
                raise ValueError(f"future has series not in the context: {sorted(unknown)[:5]}")
            return out
        if horizon < 1:
            raise ValueError(f"horizon must be positive: {horizon}")
        for item, past in self.series_.items():
            step = freq or (pd.infer_freq(past[ts]) if len(past) >= 3 else None)
            if step is None:
                raise ValueError(
                    f"cannot infer the frequency of series {item!r}: pass freq or future"
                )
            start = past[ts].iloc[-1] + pd.tseries.frequencies.to_offset(step)
            out[item] = pd.DataFrame({ts: pd.date_range(start, periods=horizon, freq=step)})
        return out

    def _predict_series(
        self, past: pd.DataFrame, steps: pd.DataFrame, levels: list[float]
    ) -> tuple[np.ndarray, np.ndarray]:
        from tabsil.recipe import TabSilRegressor

        target, ts = self.target_column, self.timestamp_column
        covariates = [c for c in self.covariates_ if c in steps.columns]
        past = _fill(past[[ts, target, *covariates]].copy(), [target, *covariates])
        past[target] = past[target].fillna(0.0)  # a series without any target value
        past = past.iloc[-self.max_context_length :].reset_index(drop=True)
        steps = _fill(steps[[ts, *covariates]].copy(), covariates)
        y = past[target].to_numpy(float)
        h = len(steps)
        if np.all(y == y[0]):
            return np.full(h, y[0]), np.full((h, len(levels)), y[0])
        features = time_features(pd.concat([past[ts], steps[ts]], ignore_index=True), y)
        if covariates:
            values = pd.concat([past[covariates], steps[covariates]], ignore_index=True)
            features = pd.concat([values, features], axis=1)
        x_past, x_future = features.iloc[: len(past)], features.iloc[len(past) :]
        reg = TabSilRegressor(
            self.model,
            self.borders,
            n_estimators=self.n_estimators,
            device=self.device,
            random_state=self.random_state,
            **self.regressor_kwargs,
        ).fit(x_past, y)
        if self.point_forecast == "mean":
            mean, log_probs, borders = reg.predict_full(x_future)
        else:
            log_probs, borders = reg.predict_distribution(x_future)
        if not (torch.isfinite(log_probs).all() and torch.isfinite(borders).all()):
            raise FloatingPointError(
                "TabSilRegressor.predict_distribution returned a non-finite distribution"
            )
        quantiles = bucket_quantiles(log_probs, borders, [*levels, 0.5])
        point = mean if self.point_forecast == "mean" else quantiles[:, -1]
        return np.asarray(point, dtype=float), quantiles[:, :-1]
