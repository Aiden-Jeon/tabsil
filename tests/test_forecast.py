"""Forecasting with TabSil regression (`tabsil.forecast`), run on a tiny randomized model."""

import numpy as np
import pandas as pd
import pytest
import torch
from helpers import tiny_model

from tabsil.models.quantile_distribution import bucket_quantiles
from tabsil.recipe import SUPPORTED_SIZE
from tabsil.tasks.forecast import (
    DEFAULT_QUANTILE_LEVELS,
    TabSilForecaster,
    seasonal_periods,
    time_features,
)

BORDERS = torch.linspace(-4, 4, 33)
LEVELS = [str(q) for q in DEFAULT_QUANTILE_LEVELS]


def _forecaster(**kwargs) -> TabSilForecaster:
    return TabSilForecaster(tiny_model(), BORDERS, n_estimators=2, **kwargs)


def _context(n_series: int = 3, length: int = 60, freq: str = "h") -> pd.DataFrame:
    rng = np.random.default_rng(0)
    frames = []
    for i in range(n_series):
        t = np.arange(length + 2 * i)  # series of different lengths
        y = 5 * i + np.sin(2 * np.pi * t / 24) + 0.1 * rng.normal(size=len(t))
        ts = pd.date_range("2024-01-01", periods=len(t), freq=freq)
        frames.append(pd.DataFrame({"item_id": f"s{i}", "timestamp": ts, "target": y}))
    return pd.concat(frames, ignore_index=True)


def test_predict_shape_columns_and_timestamps():
    context = _context()
    pred = _forecaster().fit(context).predict(5)
    assert list(pred.columns) == ["item_id", "timestamp", "predictions", *LEVELS]
    assert len(pred) == 3 * 5
    assert list(dict.fromkeys(pred.item_id)) == ["s0", "s1", "s2"]
    for item, g in pred.groupby("item_id"):
        last = context[context.item_id == item].timestamp.max()
        expected = pd.date_range(last + pd.Timedelta(hours=1), periods=5, freq="h")
        assert list(g.timestamp) == list(expected)
    assert np.isfinite(pred[["predictions", *LEVELS]].to_numpy()).all()


def test_quantiles_are_monotone_and_the_median_is_the_point_forecast():
    pred = _forecaster().fit(_context()).predict(6)
    q = pred[LEVELS].to_numpy()
    assert (np.diff(q, axis=1) >= -1e-9).all()
    assert np.allclose(pred["predictions"], pred["0.5"])


def test_mean_point_forecast_is_the_regressor_mean():
    context = _context(n_series=1)
    median = _forecaster().fit(context).predict(4)
    mean = _forecaster(point_forecast="mean").fit(context).predict(4)
    assert np.allclose(median[LEVELS], mean[LEVELS])  # same distribution
    assert not np.allclose(median["predictions"], mean["predictions"])


def test_same_seed_gives_the_same_forecast():
    context = _context()
    a = _forecaster(random_state=3).fit(context).predict(4)
    b = _forecaster(random_state=3).fit(context).predict(4)
    pd.testing.assert_frame_equal(a, b)


def test_series_longer_than_the_context_limit_uses_its_last_rows():
    context = _context(n_series=1, length=80)
    limited = _forecaster(max_context_length=30).fit(context).predict(4)
    last_rows = _forecaster(max_context_length=30).fit(context.tail(30)).predict(4)
    pd.testing.assert_frame_equal(limited, last_rows)
    full = _forecaster(max_context_length=None).fit(context).predict(4)
    assert not np.allclose(full["predictions"], limited["predictions"])


def test_context_limit_cannot_exceed_the_supported_size():
    with pytest.raises(ValueError, match="supported size"):
        _forecaster(max_context_length=SUPPORTED_SIZE["max_train_rows"] + 1)
    with pytest.raises(ValueError, match="supported size"):
        _forecaster(max_context_length=0)


def test_constant_series_and_missing_targets():
    context = _context(n_series=2)
    context.loc[context.item_id == "s0", "target"] = 2.5
    context.loc[3:6, "target"] = np.nan  # filled forward within the series
    pred = _forecaster().fit(context).predict(3)
    s0 = pred[pred.item_id == "s0"]
    assert np.allclose(s0[["predictions", *LEVELS]], 2.5)
    assert np.isfinite(pred[pred.item_id == "s1"][LEVELS].to_numpy()).all()


def test_future_frame_with_a_known_covariate():
    context = _context(n_series=2, length=40)
    context["temp"] = np.arange(len(context), dtype=float)
    context["unknown"] = 1.0  # not in future: not used
    future = (
        context.groupby("item_id")
        .timestamp.max()
        .reset_index()
        .assign(timestamp=lambda d: d.timestamp + pd.Timedelta(hours=1), temp=7.0)
    )
    fc = _forecaster().fit(context)
    pred = fc.predict(future=future)
    assert len(pred) == 2 and list(pred.timestamp) == list(future.timestamp)
    other = fc.predict(future=future.assign(temp=-50.0))
    assert not np.allclose(pred["predictions"], other["predictions"])  # the covariate is used
    with pytest.raises(ValueError, match="exactly one"):
        fc.predict(3, future=future)
    with pytest.raises(ValueError, match="no rows for series"):
        fc.predict(future=future[future.item_id == "s0"])


def test_single_series_without_id_column():
    context = _context(n_series=1).drop(columns="item_id")
    pred = _forecaster().fit(context).predict(2, quantile_levels=[0.05, 0.95])
    assert list(pred.columns) == ["item_id", "timestamp", "predictions", "0.05", "0.95"]


def test_frequency_must_be_known():
    context = _context(n_series=1, length=20)
    context = context.drop(index=[5, 9, 10])  # irregular timestamps
    fc = _forecaster().fit(context)
    with pytest.raises(ValueError, match="frequency"):
        fc.predict(3)
    assert len(fc.predict(3, freq="h")) == 3


def test_time_features_columns_and_values():
    ts = pd.date_range("2024-03-04 05:00", periods=250, freq="h")  # a Monday
    features = time_features(ts, np.sin(2 * np.pi * np.arange(240) / 12))
    assert features.shape == (250, 28)
    assert np.array_equal(features["running_index"], np.arange(250))
    assert (features["year"] == 2024).all()
    # hour 5 of the day with the TabPFN-TS period 24 - 1
    assert np.isclose(features["hour_of_day_sin"][0], np.sin(2 * np.pi * 5 / 23))
    assert np.isclose(features["day_of_week_cos"][0], 1.0)  # Monday is index 0
    assert np.isclose(features["sin_#0"][3], np.sin(2 * np.pi * 3 / 12))
    assert (features["sin_#4"] == 0).all()  # one period found, the rest zero columns


def test_seasonal_periods_finds_the_period():
    t = np.arange(24 * 20)
    # periods on an FFT bin of the twice-padded length (others round to the nearest bin)
    y = 0.05 * t + np.sin(2 * np.pi * t / 24) + 0.5 * np.sin(2 * np.pi * t / 120)
    periods = seasonal_periods(y)
    assert periods[:2] == [24.0, 120.0]
    assert seasonal_periods(np.array([1.0])) == []


def test_bucket_quantiles_inverts_the_cdf():
    borders = torch.tensor([0.0, 1.0, 2.0, 4.0])
    probs = torch.tensor([[0.5, 0.25, 0.25]])
    q = bucket_quantiles(probs.log(), borders, [0.25, 0.5, 0.625, 0.875, 1e-9])
    assert np.allclose(q, [[0.5, 1.0, 1.5, 3.0, 0.0]], atol=1e-6)
    # an empty bucket is skipped
    q = bucket_quantiles(torch.tensor([[0.5, 0.0, 0.5]]).clamp_min(1e-30).log(), borders, [0.75])
    assert np.allclose(q, [[3.0]], atol=1e-6)
