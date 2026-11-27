"""`eval.py --suite causal`: the treatment effect benchmarks with the model, realizations and
methods of the command line (`tabsil_train.benchmarks.causal` is the benchmark: data, methods,
report)."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlflow
import pandas as pd

from tabsil.recipe import default_inference
from tabsil_train.benchmarks import _files, causal
from tabsil_train.hub import load_model

# the `--suite` name: the results folder under --out-dir and the MLflow artifact folder
SUITE = "causal"


def run_causal(args: argparse.Namespace, out_dir: Path, use_mlflow: bool) -> None:
    """Treatment effects on IHDP and ACIC 2016 (`tabsil_train.benchmarks.causal`). Ends with an
    error if any (benchmark, realization, method) failed or has no result, after the others ran
    and the summary was written."""
    benchmarks = args.datasets.split(",") if args.datasets else list(causal.BENCHMARKS)
    if bad := [b for b in benchmarks if b not in causal.BENCHMARKS]:
        raise SystemExit(f"causal benchmarks are {list(causal.BENCHMARKS)}: {bad}")
    try:
        plan = {b: causal.parse_realizations(args.causal_realizations, b) for b in benchmarks}
        methods = causal.check_methods(
            args.causal_methods.split(",") if args.causal_methods else list(causal.METHODS)
        )
    except ValueError as e:
        raise SystemExit(str(e)) from None
    estimators: dict = {m: causal.econml_estimator(m) for m in methods if m in causal.ECONML}
    opts = {m: "EconML defaults, GBM base models" for m in estimators}
    params = {
        "suite": args.suite,
        "benchmarks": ",".join(benchmarks),
        "realizations": args.causal_realizations or "all",
        "methods": ",".join(methods),
    }
    if ours := [m for m in methods if m in causal.TABSIL]:
        if not args.regressor_path:
            raise SystemExit("--suite causal needs --regressor-path (a model serving both tasks).")
        model, borders = load_model(
            args.regressor_path, "regression", out_dir / "checkpoints" / "regression"
        )
        if "classification" not in model.tasks:
            raise SystemExit(f"{args.regressor_path} does not serve classification (propensity).")
        n_estimators = args.n_estimators or default_inference(model, "regression")["n_estimators"]
        settings = _files.model_settings(model, borders, n_estimators, args.seed)
        for m in ours:
            estimators[m] = causal.tabsil_estimator(
                model, borders, causal.TABSIL[m], n_estimators, args.device, 1000 * args.seed
            )
            opts[m] = settings
        params |= {"model": args.regressor_path, "n_estimators": n_estimators, "seed": args.seed}
    print(f"causal: {plan if args.causal_realizations else benchmarks} x {methods}, "
          f"{args.device}", flush=True)  # fmt: skip
    if use_mlflow:
        mlflow.log_params(params)
    suite_dir = out_dir / SUITE
    t0 = time.time()
    rows, failures = causal.run(
        estimators, plan, methods, suite_dir, opts, log=lambda s: print(s, flush=True)
    )
    summary = causal.report(rows, suite_dir, plan, methods)
    summary |= {"failures": len(failures), "seconds": round(time.time() - t0, 1)}
    (suite_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    table = pd.read_csv(suite_dir / "summary.csv")
    print(f"\n{table.round(3).to_string(index=False)}", flush=True)
    print(f"missing {summary['missing_pct']:.1f}%, imputed {summary['imputed_pct']:.1f}%, "
          f"failures {len(failures)}, {summary['seconds']:.0f}s", flush=True)  # fmt: skip
    if use_mlflow:
        metrics = {
            f"causal_{key.replace('/', '_')}_{m}": float(v)
            for key, by_metric in summary["methods"].items()
            for m, v in by_metric.items()
        }
        metrics |= {"causal_missing_pct": summary["missing_pct"], "causal_failures": len(failures)}
        mlflow.log_metrics(metrics)
        mlflow.log_artifacts(str(suite_dir), SUITE)
    if failures:
        raise SystemExit(f"causal: {len(failures)} failed cells: {failures}")
    if summary["missing_pct"]:
        raise SystemExit(f"causal: {summary['missing_pct']:.1f}% of the planned cells are missing")
