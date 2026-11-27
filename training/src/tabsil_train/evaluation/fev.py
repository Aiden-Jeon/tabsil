"""`eval.py --suite fev`: fev-bench forecasting with the model, tasks and windows of the command
line (`tabsil_train.benchmarks.forecast.fevbench` is the benchmark: tasks, metrics, reference
results, Elo)."""

from __future__ import annotations

import argparse
from pathlib import Path

import mlflow
import pandas as pd

from tabsil import TabSilForecaster
from tabsil.recipe import default_inference
from tabsil_train.benchmarks.forecast import fevbench
from tabsil_train.hub import load_model

# the `--suite` name: the results folder under --out-dir and the MLflow artifact folder
SUITE = "fev"


def run_fev(args: argparse.Namespace, out_dir: Path, use_mlflow: bool) -> None:
    """Forecasting on fev-bench (`tabsil_train.benchmarks.forecast.fevbench`): every window of each
    task, our Seasonal Naive check, the comparison table and the Elo pools."""
    if args.regressor_path is None:
        raise SystemExit("--suite fev needs --regressor-path.")
    tasks = [t.strip() for t in args.datasets.split(",") if t.strip()] if args.datasets else None
    model, borders = load_model(args.regressor_path, "regression", out_dir / "checkpoints" / "fev")
    context = {}
    if args.max_context_length is not None:
        context["max_context_length"] = args.max_context_length or None
    forecaster = TabSilForecaster(
        model,
        borders,
        n_estimators=args.n_estimators,
        device=args.device,
        random_state=args.seed,
        **context,
    )
    print(
        f"fev: tasks {tasks or list(fevbench.TASKS)}, windows {args.fev_windows or 'all'},"
        f" context {forecaster.max_context_length}, device {args.device}",
        flush=True,
    )
    if use_mlflow:
        mlflow.log_params(
            {
                "suite": args.suite,
                "regressor_path": args.regressor_path,
                "seed": args.seed,
                "n_estimators": args.n_estimators
                or default_inference(model, "regression")["n_estimators"],
                "max_context_length": forecaster.max_context_length,
                "fev_windows": args.fev_windows or "all",
                "fev_tasks": ",".join(tasks or fevbench.TASKS),
            }
        )
    summary = fevbench.run(
        forecaster, tasks, args.fev_windows, out_dir / SUITE, log=lambda s: print(s, flush=True)
    )
    table = pd.read_csv(out_dir / SUITE / "comparison.csv")
    show = ["TabSil", "Seasonal Naive", "TabPFN-TS (v2)", "TabICL-TS", "Chronos-2"]
    for metric in ("SQL", "MASE"):
        wide = table.pivot(index="task", columns="method", values=metric)
        print(f"\ntask mean {metric}:\n{wide[[m for m in show if m in wide]].round(4)}")
    elo = pd.read_csv(out_dir / SUITE / "elo.csv")
    print(f"\nElo (Seasonal Naive = 1000):\n{elo.round(1).to_string(index=False)}")
    print(f"TabSil time {summary['tabsil_seconds']:.0f}s, total {summary['seconds']['total']:.0f}s")
    if use_mlflow:
        mlflow.log_metrics(
            {
                **{f"fev_elo_{k}": v["elo"] for k, v in summary["elo"].items()},
                **{
                    f"fev_{m}_{t}": v
                    for m, by_task in summary["task_mean"].items()
                    for t, v in by_task.items()
                },
                "fev_tabsil_seconds": summary["tabsil_seconds"],
                "fev_imputed_pct": 100 * summary["imputed"],
            }
        )
        for path in sorted((out_dir / SUITE).glob("*.csv")):
            mlflow.log_artifact(str(path), f"eval/{SUITE}")
        mlflow.log_artifact(str(out_dir / SUITE / "summary.json"), f"eval/{SUITE}")
