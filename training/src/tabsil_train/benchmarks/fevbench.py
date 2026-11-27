"""fev-bench forecasting evaluation of TabSil (`tabsil.forecast.TabSilForecaster`, zero-shot).

Tasks, splits and metrics follow fm-agent-bench P7 (`docs/forecast.md` at commit
`FM_AGENT_BENCH_COMMIT`), which compared public tabular FMs on fev-bench:

- tasks (`TASKS`): 10 univariate fev-bench tasks (Shchur et al., arXiv 2509.26468), their
  definitions copied from autogluon/fev `benchmarks/fev_bench/tasks.yaml` at `FEV_COMMIT`. Data:
  the HF dataset `autogluon/fev_datasets`, which fev reads at its current revision; the
  evaluation stops if that is not `DATASETS_REVISION` (`check_datasets_revision`)
- splits: every rolling window of each task (fev's `iter_windows`); the context is everything
  before the window's cutoff (TabSil keeps the last `max_context_length` rows of each series)
- metrics: fev 0.10.0's own `compute_metrics`, SQL (scaled quantile loss over the 9 levels
  0.1..0.9, fev-bench's `eval_metric`, the main metric) and MASE (point forecast), plus WQL and
  WAPE (epsilon 1). A task's score is the mean over its windows (fev's `evaluation_summary`)
- covariates as fm-agent-bench ran TabPFN-TS: known dynamic covariates are features (their
  future values are given), past dynamic and static covariates are not used

Comparison numbers (copied, not rerun):

- `fev_windows.csv`: per-window SQL / MASE / WQL / WAPE of the 7 methods fm-agent-bench ran
  (Seasonal Naive, AutoETS, Chronos-2, Chronos-Bolt, TimesFM-2.5, TabPFN-TS (v2), TabICL-TS),
  its `results/forecast.csv` (`scorer=fev`, the full run) at `FM_AGENT_BENCH_COMMIT`
- `fev_public.csv`: the public fev-bench task means of 8 methods (fm-agent-bench
  `results/forecast_reference.csv`, fev results at `ecce671`); AutoARIMA and TabPFN-TS-3 come
  only from here

Our Seasonal Naive (`seasonal_naive`, the fev statsforecast wrapper's model reimplemented in
numpy) is scored on the same windows and must match fm-agent-bench's per-window values
(`check_seasonal_naive`) before the TabSil numbers are used.

Elo (`elo`): fm-agent-bench's TabArena-style Bradley-Terry Elo with Seasonal Naive = 1000
(`ANCHOR`), 200 bootstrap rounds over tasks for the 95% interval. Two pools, SQL and MASE each:

- `mean`: one unit per task (the window mean), the fm-agent-bench FM pool's unit: the 7 methods
  above + AutoARIMA and TabPFN-TS-3 (public task means) + TabSil. Only when every window ran
- `windows`: one unit per (task, window), each task's windows weighted 1 / window count: the 7
  methods + TabSil

Nothing is imputed: a failed series, window or task stops the evaluation with an error.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from tabsil_train.benchmarks import _elo

FEV_VERSION = "0.10.0"
# autogluon/fev commit of the task definitions and the public results
FEV_COMMIT = "ecce67120aae4827e9702b978a9e9327ecde81d0"
# autogluon/fev_datasets (HF dataset) revision
DATASETS_REPO = "autogluon/fev_datasets"
DATASETS_REVISION = "d3e7e720f4158eab03ebcb021f1cce779cd92acc"
# Aiden-Jeon/fm-agent-bench commit of the copied results (fev_windows.csv, fev_public.csv)
FM_AGENT_BENCH_COMMIT = "b1b010105881e3b0589ec86a1e6ecc7a463412ad"

ANCHOR = "Seasonal Naive"
METHOD = "TabSil"
METRICS = ("SQL", "MASE", "WQL", "WAPE")
ELO_METRICS = ("SQL", "MASE")
QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# Public-only methods added to the `mean` pool
PUBLIC_ONLY = ("AutoARIMA", "TabPFN-TS-3")
# Largest relative difference allowed between our Seasonal Naive and fm-agent-bench's
SNAIVE_TOLERANCE = 1e-6

_COMMON = {
    "eval_metric": "SQL",
    "extra_metrics": ["MASE", {"name": "WAPE", "epsilon": 1.0}, {"name": "WQL", "epsilon": 1.0}],
    "quantile_levels": QUANTILE_LEVELS,
}
# fev-bench task definitions (fev @ FEV_COMMIT, benchmarks/fev_bench/tasks.yaml), in the order
# of fm-agent-bench docs/forecast.md section 2. Copyright Amazon.com, Inc. or its affiliates,
# Apache License 2.0
TASKS: dict[str, dict] = {
    "redset_1H": {
        "horizon": 24,
        "num_windows": 10,
        "seasonality": 24,
        "static_columns": ["subset"],
    },
    "solar_with_weather_1H": {
        "horizon": 24,
        "num_windows": 20,
        "seasonality": 24,
        "known_dynamic_columns": [
            "wind_speed",
            "day_length",
            "humidity",
            "rain_1h",
            "snow_1h",
            "temp",
            "pressure",
        ],
        "past_dynamic_columns": ["global_horizontal_irradiance", "clouds_all"],
    },
    "ercot_1D": {"horizon": 28, "num_windows": 20, "seasonality": 7, "static_columns": []},
    "epf_be": {
        "horizon": 24,
        "num_windows": 20,
        "seasonality": 24,
        "known_dynamic_columns": ["Generation forecast", "System load forecast"],
    },
    "rohlik_orders_1D": {
        "horizon": 61,
        "num_windows": 5,
        "initial_cutoff": "2023-05-01",
        "seasonality": 7,
        "target": "orders",
        "known_dynamic_columns": [
            "holiday",
            "shops_closed",
            "winter_school_holidays",
            "school_holidays",
        ],
        "past_dynamic_columns": [
            "shutdown",
            "mini_shutdown",
            "blackout",
            "mov_change",
            "frankfurt_shutdown",
            "precipitation",
            "snow",
            "user_activity_1",
            "user_activity_2",
        ],
        "static_columns": [],
    },
    "solar_1D": {"horizon": 28, "num_windows": 10, "seasonality": 1, "static_columns": []},
    "ecdc_ili": {"horizon": 13, "num_windows": 10, "seasonality": 1, "static_columns": []},
    "hospital": {"horizon": 12, "num_windows": 4, "seasonality": 12, "static_columns": []},
    "us_consumption_1M": {
        "horizon": 12,
        "num_windows": 10,
        "seasonality": 12,
        "static_columns": [],
    },
    "hospital_admissions_1D": {
        "horizon": 28,
        "num_windows": 20,
        "seasonality": 7,
        "static_columns": [],
    },
}

_HERE = Path(__file__).resolve().parent
WINDOWS_CSV = _HERE / "fev_windows.csv"
PUBLIC_CSV = _HERE / "fev_public.csv"


# ---------------------------------------------------------------------------
# Tasks and predictions
# ---------------------------------------------------------------------------
def check_datasets_revision() -> str:
    """The current revision of the HF dataset fev reads (fev loads a config without a revision).
    RuntimeError if it is not `DATASETS_REVISION` (the data the reference results used), unless
    fev reads a mirror (`FEV_DATASETS_PREFIX`), which is then the caller's responsibility."""
    if os.environ.get("FEV_DATASETS_PREFIX"):
        return f"mirror {os.environ['FEV_DATASETS_PREFIX']}"
    from huggingface_hub import HfApi

    sha = HfApi().dataset_info(DATASETS_REPO).sha
    if sha != DATASETS_REVISION:
        raise RuntimeError(
            f"{DATASETS_REPO} is at {sha}, not {DATASETS_REVISION} (the revision of the"
            " reference results): the data may differ"
        )
    return sha


def make_task(name: str):
    """The fev `Task` of a task name in `TASKS`."""
    import fev

    if name not in TASKS:
        raise ValueError(f"unknown fev task {name!r}: one of {list(TASKS)}")
    return fev.Task(dataset_path=DATASETS_REPO, dataset_config=name, **_COMMON, **TASKS[name])


def window_indices(task, last: int | None = None) -> list[int]:
    """Window indices to evaluate: all, or the `last` ones."""
    n = task.num_windows
    if last is None or last >= n:
        return list(range(n))
    if last < 1:
        raise ValueError(f"windows must be >= 1: {last}")
    return list(range(n - last, n))


def _to_predictions(task, ids: list, rows: list[dict]):
    """fev predictions (a DatasetDict keyed by the target) from per-series rows in fev's order."""
    import datasets

    for item, row in zip(ids, rows, strict=True):
        values = np.asarray([row["predictions"], *[row[str(q)] for q in QUANTILE_LEVELS]])
        if not np.isfinite(values).all():
            raise ValueError(f"{task.task_name}: non-finite forecast for series {item!r}")
    return datasets.DatasetDict({task.target_columns[0]: datasets.Dataset.from_list(rows)})


def predict_tabsil(forecaster, task, window):
    """TabSil forecasts of one window: the context is the window's past (target and known dynamic
    covariates), the horizon its future rows."""
    import fev

    # as_univariate renames the target column to "target" (rohlik_orders_1D's is "orders")
    past, future, _ = fev.convert_input_data(window, "pandas", as_univariate=True)
    ids, ts, target = task.id_column, task.timestamp_column, "target"
    known = list(task.known_dynamic_columns)
    forecaster.fit(
        past[[ids, ts, target, *known]], id_column=ids, timestamp_column=ts, target_column=target
    )
    pred = forecaster.predict(future=future[[ids, ts, *known]], quantile_levels=QUANTILE_LEVELS)
    order = list(dict.fromkeys(future[ids]))
    if list(dict.fromkeys(pred[ids])) != order:
        raise ValueError(f"{task.task_name}: forecast series order differs from fev's")
    rows = []
    for _, g in pred.groupby(ids, sort=False):
        rows.append(
            {
                "predictions": g["predictions"].tolist(),
                **{str(q): g[str(q)].tolist() for q in QUANTILE_LEVELS},
            }
        )
    return _to_predictions(task, order, rows)


_SEASONALITY = {
    "YE": 1,
    "QE": 4,
    "ME": 12,
    "SME": 24,
    "W": 1,
    "D": 7,
    "B": 5,
    "bh": 9,
    "h": 24,
    "min": 1440,
}
_MAJOR_FREQ = {"MS": "ME", "M": "ME", "QS": "QE", "Q": "QE", "YS": "YE", "Y": "YE", "H": "h"}


def freq_seasonality(freq: str) -> int:
    """Seasonality of a pandas frequency (autogluon.timeseries `get_seasonality`, which the fev
    statsforecast wrapper uses: 24 for hourly, 7 for daily, 12 for monthly, ...)."""
    offset = pd.tseries.frequencies.to_offset(freq)
    base = offset.name.split("-")[0]
    base = _MAJOR_FREQ.get(base, base)
    seasonality, remainder = divmod(_SEASONALITY.get(base, 1), offset.n)
    return seasonality if not remainder else 1


def seasonal_naive(task, window, max_context_length: int = 1000):
    """Seasonal Naive forecasts of one window, as the fev statsforecast wrapper
    (`models/statsforecast`, `{"model": "seasonal_naive"}`) computes them with statsforecast 2.0.1
    `SeasonalNaive` (reimplemented; Apache License 2.0, Nixtla):

    - season m = max(task seasonality, frequency seasonality); 1 if m > `max_context_length`
    - missing targets are filled forward, the rest with 0; the last `max_context_length` values
    - forecast: the last m values repeated (missing ones 0); quantile q: forecast -/+ z sigma_h
      with z = Phi^-1(0.5 + level / 200), level = round(|q - 0.5| * 200), sigma_h = sigma
      sqrt(floor(step / m) + 1), sigma = sqrt(sum of squared seasonal differences / (n - m))
    """
    import fev
    from scipy.stats import norm

    m = max(task.seasonality, freq_seasonality(task.freq))
    if m > max_context_length:
        m = 1
    past, _, _ = fev.convert_input_data(window, "nixtla", as_univariate=True)
    h = window.horizon
    ids, rows = [], []
    for item, g in past.groupby("unique_id", sort=False):
        y = g["y"].ffill().fillna(0.0).to_numpy()
        if y.dtype not in (np.float32, np.float64):
            y = y.astype(np.float32)
        y = y[-max_context_length:]
        n = len(y)
        season = np.full(m, np.nan, dtype=y.dtype)
        season[: min(m, n)] = y[-min(m, n) :]
        mean = np.tile(season, -(-h // m))[:h]
        residuals = y[m:] - y[: n - m] if n > m else np.array([], dtype=y.dtype)
        sigma = np.sqrt(np.nansum(residuals**2) / (n - m)) if n - m > 0 else 0.0
        sigma_h = sigma * np.sqrt(np.floor(np.arange(h) / m) + 1)
        row = {"predictions": np.nan_to_num(mean, nan=0.0).tolist()}
        for q in QUANTILE_LEVELS:
            z = norm.ppf(0.5 + round(abs(q - 0.5) * 200) / 200)
            value = mean - z * sigma_h if q < 0.5 else mean + z * sigma_h
            row[str(q)] = np.nan_to_num(value, nan=0.0).tolist()
        ids.append(item)
        rows.append(row)
    return _to_predictions(task, ids, rows)


def score(task, window, predictions) -> dict:
    """fev's metrics of one window's predictions (SQL, MASE, WQL, WAPE)."""
    from fev.metrics import get_metric

    metrics = [get_metric(m) for m in [task.eval_metric, *task.extra_metrics]]
    clean = task.clean_and_validate_predictions(predictions)
    return window.compute_metrics(clean, metrics, task.seasonality, task.quantile_levels)


def evaluate_task(
    name: str,
    methods: dict[str, Callable],
    windows: int | None = None,
    log: Callable[[str], None] = print,
) -> list[dict]:
    """Score `methods` ({name: predict(task, window) -> fev predictions}) on a task's windows (all,
    or the last `windows`). One row per (method, window): task, window, cutoff, metrics,
    seconds (prediction time), series count."""
    task = make_task(name)
    rows = []
    for i in window_indices(task, windows):
        window = task.get_window(i)
        for method, predict in methods.items():
            t0 = time.perf_counter()
            preds = predict(task, window)
            seconds = time.perf_counter() - t0
            metrics = score(task, window, preds)
            rows.append(
                {
                    "method": method,
                    "task": name,
                    "window": i,
                    "cutoff": str(window.cutoff),
                    **{k: float(metrics[k]) for k in METRICS},
                    "seconds": seconds,
                    "series": len(preds[task.target_columns[0]]),
                }
            )
            log(
                f"  {name} window {i + 1}/{task.num_windows} {method}: SQL {metrics['SQL']:.4f}"
                f" MASE {metrics['MASE']:.4f} ({seconds:.1f}s, {rows[-1]['series']} series)"
            )
    return rows


# ---------------------------------------------------------------------------
# Reference results and checks
# ---------------------------------------------------------------------------
def reference_windows() -> pd.DataFrame:
    """fm-agent-bench's per-window results of 7 methods (`fev_windows.csv`)."""
    return pd.read_csv(WINDOWS_CSV)


def reference_public() -> pd.DataFrame:
    """Public fev-bench task means (`fev_public.csv`)."""
    return pd.read_csv(PUBLIC_CSV)


def check_seasonal_naive(ours: pd.DataFrame, tolerance: float = SNAIVE_TOLERANCE) -> dict:
    """Our Seasonal Naive (rows with method `ANCHOR + " (ours)"`) against fm-agent-bench's per
    window. Returns the largest relative difference per metric; ValueError beyond `tolerance`."""
    ref = reference_windows().query("method == @ANCHOR")
    mine = ours[ours.method == f"{ANCHOR} (ours)"]
    merged = mine.merge(ref, on=["task", "window"], suffixes=("", "_ref"), how="left")
    if merged[[f"{m}_ref" for m in METRICS]].isna().any().any():
        raise ValueError("fm-agent-bench has no Seasonal Naive value for some windows")
    if (merged.cutoff != merged.cutoff_ref.astype(str)).any():
        raise ValueError("window cutoffs differ from fm-agent-bench's")
    out = {}
    for m in METRICS:
        rel = (merged[m] - merged[f"{m}_ref"]).abs() / merged[f"{m}_ref"].abs().clip(lower=1e-12)
        out[m] = float(rel.max())
    if max(out.values()) > tolerance:
        raise ValueError(f"our Seasonal Naive differs from fm-agent-bench's: {out}")
    return out


# ---------------------------------------------------------------------------
# Elo (`_elo`: fm-agent-bench fab/bench/tabarena/elo.py `elo`, the TabArena computation)
# ---------------------------------------------------------------------------
MIN_CI_TASKS = 10  # fewer tasks: no bootstrap interval


def elo(
    long: pd.DataFrame,
    anchor: str = ANCHOR,
    rounds: int = _elo.ROUNDS,
    seed: int = _elo.SEED,
    min_ci: int = MIN_CI_TASKS,
) -> pd.DataFrame:
    """Elo per method from `long` (method, task, split, error; no missing cells): each task's
    splits weighted 1 / split count, `anchor` moved to 1000. The 95% interval (`elo+`, `elo-`)
    is the 2.5% / 97.5% quantile of `rounds` bootstrap fits over tasks minus the point estimate
    (empty with fewer than `min_ci` tasks). Also the weighted mean rank and win rate
    (`_elo.table`)."""
    key = ["method", "task", "split"]
    if long.duplicated(key).any():
        raise ValueError("repeated (method, task, split) rows")
    wide = long.pivot(index="method", columns=["task", "split"], values="error")
    if wide.isna().any().any():
        missing = wide.isna().stack()
        raise ValueError(f"missing results (nothing is imputed): {missing[missing].index[:5]}")
    if anchor not in wide.index:
        raise ValueError(f"the anchor {anchor!r} is not in the pool")
    names, _, _ = _elo.unit_weights(wide)
    out = _elo.table(wide, anchor, rounds, seed, min_ci).assign(tasks=len(names))
    return out.sort_values("elo", ascending=False)


def pools(ours: pd.DataFrame, metric: str) -> dict[str, pd.DataFrame]:
    """Elo input (method, task, split, error) per pool (module docstring) for `metric`, from our
    per-window rows (method `METHOD`). The `mean` pool needs every window of each task."""
    ref = reference_windows()
    mine = ours[ours.method == METHOD]
    units = mine[["task", "window"]]
    ref = ref.merge(units, on=["task", "window"])
    if len(ref) != len(units) * ref.method.nunique():
        raise ValueError("fm-agent-bench lacks some of the evaluated windows")
    both = pd.concat([ref, mine], ignore_index=True)
    out = {
        "windows": both.assign(split=both.window.map("w{}".format), error=both[metric])[
            ["method", "task", "split", "error"]
        ]
    }
    complete = all(
        set(g.window) == set(range(TASKS[t]["num_windows"])) for t, g in mine.groupby("task")
    )
    if complete:
        means = both.groupby(["method", "task"], as_index=False)[metric].mean()
        public = reference_public()
        public = public[public.method.isin(PUBLIC_ONLY) & public.task.isin(set(mine.task))]
        if len(public) != len(PUBLIC_ONLY) * mine.task.nunique():
            raise ValueError("the public results lack some of the evaluated tasks")
        means = pd.concat([means, public[["method", "task", metric]]], ignore_index=True)
        out["mean"] = means.assign(split="all", error=means[metric])[
            ["method", "task", "split", "error"]
        ]
    return out


def comparison(ours: pd.DataFrame) -> pd.DataFrame:
    """Task means (SQL, MASE) of TabSil, our Seasonal Naive and the reference methods on the
    evaluated windows, with each method's SQL relative to Seasonal Naive (`SQL_vs_snaive`)."""
    units = ours.loc[ours.method == METHOD, ["task", "window"]].drop_duplicates()
    ref = reference_windows().merge(units, on=["task", "window"])
    rows = pd.concat([ours[ref.columns.intersection(ours.columns)], ref], ignore_index=True)
    table = rows.groupby(["task", "method"], as_index=False)[list(METRICS)].mean()
    snaive = table[table.method == ANCHOR].set_index("task")["SQL"]
    table["SQL_vs_snaive"] = table.SQL / table.task.map(snaive)
    table["windows"] = table.task.map(units.groupby("task").size())
    return table


def run(
    forecaster,
    tasks: list[str] | None = None,
    windows: int | None = None,
    out_dir: str | Path = "tabsil_eval/fev",
    log: Callable[[str], None] = print,
) -> dict:
    """Evaluate `forecaster` (a `TabSilForecaster`) and our Seasonal Naive on `tasks` (default:
    all of `TASKS`), every window or the last `windows` of each. Writes `windows.csv` (per window),
    `comparison.csv` (task means next to the reference methods), `elo.csv` and `summary.json`
    under `out_dir`, and returns the summary."""
    tasks = list(TASKS) if tasks is None else tasks
    unknown = set(tasks) - set(TASKS)
    if unknown:
        raise ValueError(f"unknown fev tasks: {sorted(unknown)} (known: {list(TASKS)})")
    revision = check_datasets_revision()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    methods = {
        METHOD: lambda task, window: predict_tabsil(forecaster, task, window),
        f"{ANCHOR} (ours)": seasonal_naive,
    }
    rows, seconds = [], {}
    for k, name in enumerate(tasks, start=1):
        log(f"[{k}/{len(tasks)}] {name}")
        t0 = time.perf_counter()
        rows += evaluate_task(name, methods, windows, log)
        seconds[name] = time.perf_counter() - t0
        pd.DataFrame(rows).to_csv(out_dir / "windows.csv", index=False)  # progress so far
    ours = pd.DataFrame(rows)
    snaive_check = check_seasonal_naive(ours)
    log(f"our Seasonal Naive vs fm-agent-bench, largest relative difference: {snaive_check}")
    table = comparison(ours)
    table.to_csv(out_dir / "comparison.csv", index=False)
    elos = []
    for metric in ELO_METRICS:
        for pool, long in pools(ours, metric).items():
            elos.append(elo(long).reset_index().assign(pool=pool, metric=metric))
    elo_table = pd.concat(elos, ignore_index=True)
    elo_table = elo_table[["pool", "metric", *elo_table.columns.drop(["pool", "metric"])]]
    elo_table.to_csv(out_dir / "elo.csv", index=False)
    tabsil = ours[ours.method == METHOD]
    summary = {
        "tasks": tasks,
        "windows": int(len(tabsil)),
        "all_windows": windows is None,
        "series_forecasts": int(tabsil.series.sum()),
        "seconds": {"total": float(sum(seconds.values())), **seconds},
        "tabsil_seconds": float(tabsil.seconds.sum()),
        "seasonal_naive_max_rel_diff": snaive_check,
        "task_mean": {
            m: table[table.method == METHOD].set_index("task")[m].to_dict() for m in METRICS
        },
        "elo": {
            f"{r.pool}_{r.metric}": {"elo": r.elo, "elo+": r["elo+"], "elo-": r["elo-"]}
            for _, r in elo_table[elo_table.method == METHOD].iterrows()
        },
        "imputed": 0.0,
        "fev_version": FEV_VERSION,
        "fev_commit": FEV_COMMIT,
        "datasets_revision": revision,
        "fm_agent_bench_commit": FM_AGENT_BENCH_COMMIT,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    return summary
