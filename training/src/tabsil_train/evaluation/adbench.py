"""`eval.py --suite adbench`: ADBench outlier detection with the model, splits and methods of the
command line (`tabsil_train.benchmarks.unsupervised.adbench` is the benchmark: data, protocol,
report).

- `run_adbench`: loads the model, runs the (table, split, method) cells in this process or, with
  `--shards`, the TabSil ones as worker processes, writes the report and logs to MLflow
- `run_adbench_shards`: the `--shards` queue of TabSil jobs
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

import mlflow
import pandas as pd

from tabsil_train.benchmarks import _files
from tabsil_train.benchmarks.unsupervised import adbench
from tabsil_train.evaluation.model import suite_model
from tabsil_train.evaluation.shards import run_shard_queue, shard_devices, strip_flags
from tabsil_train.hub import download_artifact

# the `--suite` name: the results folder under --out-dir and the MLflow artifact folder
SUITE = "adbench"


def run_adbench(args: argparse.Namespace, out_dir: Path, use_mlflow: bool) -> None:
    """ADBench unsupervised outlier detection (`tabsil_train.benchmarks.unsupervised.adbench`). Ends
    with an error if any (table, split, method) failed, after the others ran and the report was
    written."""
    datasets = list(adbench.DATASETS) if args.datasets is None else args.datasets.split(",")
    splits = args.adbench_splits.split(",")
    methods = adbench.check_methods(args.adbench_methods.split(","))
    if bad := [d for d in datasets if d not in adbench.DATASETS]:
        raise SystemExit(f"tables not in ADBench's 12: {bad} (known: {list(adbench.DATASETS)})")
    if bad := [s for s in splits if s not in adbench.SPLITS]:
        raise SystemExit(f"splits are {list(adbench.SPLITS)}: {bad}")
    scorers: dict = {"iforest": adbench.iforest_scores}
    opts = {"iforest": "sklearn IsolationForest = pyod IForest defaults"}
    params = {
        "suite": args.suite,
        "datasets": ",".join(datasets),
        "splits": ",".join(splits),
        "methods": ",".join(methods),
    }
    if any(adbench.base(m) == adbench.OURS for m in methods):
        path = args.regressor_path or args.classifier_path
        # with --shards the workers score; this process loads the model for its settings string
        model, borders, n_estimators, settings = suite_model(
            args, path, out_dir, "adbench", "categorical columns",
            n_permutations=args.n_permutations,
        )  # fmt: skip
        if args.shards == 1:
            scorers[adbench.OURS] = adbench.tabsil_scorer(
                model, borders, args.n_permutations, n_estimators, args.device, 100 * args.seed
            )
        opts[adbench.OURS] = settings
        params |= {"model": path, "n_permutations": args.n_permutations,
                   "n_estimators": n_estimators, "seed": args.seed}  # fmt: skip
        del model, borders  # the scorer holds them (none with --shards)
    print(f"adbench: {len(datasets)} tables x {splits} x {methods}, {args.device}", flush=True)
    if use_mlflow:
        mlflow.log_params(params)
    suite_dir = out_dir / SUITE
    t0 = time.time()
    if args.shards > 1:
        # IsolationForest (seconds) here, TabSil's (table, split, method) jobs on the GPUs
        cpu = [m for m in methods if adbench.base(m) != adbench.OURS]
        _, failures = adbench.run(scorers, datasets, splits, cpu, suite_dir, opts)
        gpu = [m for m in methods if adbench.base(m) == adbench.OURS]
        failures += run_adbench_shards(args, datasets, splits, gpu, suite_dir, out_dir, opts)
        rows = adbench.read_results(suite_dir / "results.csv")
    else:
        rows, failures = adbench.run(scorers, datasets, splits, methods, suite_dir, opts)
    # the report is of this run's methods: rows of other methods in results.csv are an earlier
    # run's and were not checked against this run's settings
    rows = rows[rows.method.isin(methods)]
    summary = adbench.report(rows, suite_dir, [m for m in methods if m == adbench.OURS])
    summary |= {"failures": len(failures), "seconds": round(time.time() - t0, 1)}
    (suite_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps(summary, indent=1, default=str), flush=True)
    for name in ("summary.csv", "elo.csv"):
        if (suite_dir / name).exists():
            print(f"\n{name}\n{pd.read_csv(suite_dir / name).round(4).to_string(index=False)}")
    if use_mlflow:
        metrics = {f"adbench_{k}": v for k, v in summary.items() if isinstance(v, int | float)}
        metrics |= {f"adbench_elo_{k}": v for k, v in summary.get("elo", {}).items()}
        mlflow.log_metrics({re.sub(r"[^\w\-. /]", "_", k): float(v) for k, v in metrics.items()})
        mlflow.log_artifacts(str(suite_dir), SUITE)
    if failures:
        raise SystemExit(f"adbench: {len(failures)} failed (table, split, method): {failures}")


def run_adbench_shards(
    args: argparse.Namespace,
    datasets: list[str],
    splits: list[str],
    methods: list[str],
    suite_dir: Path,
    out_dir: Path,
    opts: dict[str, str] | None = None,
) -> list[dict]:
    """ADBench `--shards`: one job per (table, split, method) not yet in `results.csv`, most
    expensive first, each a worker process (`eval.py --suite adbench` on one GPU, without MLflow)
    started as soon as a GPU is free (`run_shard_queue`). A finished job's rows are appended to
    `results.csv`; a failed job is returned as a failure (with its log under `shards/`). Rows of
    `methods` with other settings than `opts` (base method -> settings string) stop the run."""
    # the settings are checked here as well as in `adbench.run`: a job whose row is already
    # there never reaches a worker
    settings = adbench.row_settings(methods, opts)
    results = _files.Results(suite_dir, adbench.RESULT_COLUMNS, settings, SUITE)
    done = results.done
    finished = set(zip(done.dataset, done.split, done.method, strict=True))
    jobs = [
        (d, s, m) for d in datasets for s in splits for m in methods if (d, s, m) not in finished
    ]
    jobs.sort(key=lambda j: -adbench.job_cost(j[0], j[2]))
    devices = shard_devices(args)
    path = args.regressor_path or args.classifier_path
    if path and path.startswith("runs:/"):  # the parent's download, reused by every worker
        path = str(download_artifact(path, out_dir / "checkpoints" / "regression"))
    base = strip_flags(
        args.argv,
        {
            "--shards": 1,
            "--datasets": 1,
            "--adbench-splits": 1,
            "--adbench-methods": 1,
            "--out-dir": 1,
            "--device": 1,
            "--mlflow": 0,
            "--regressor-path": 1,
            "--classifier-path": 1,
            "--heartbeat-minutes": 1,
        },
    )
    # a worker sees one GPU (CUDA_VISIBLE_DEVICES), so cuda:<i> becomes cuda
    device = "cuda" if args.device.startswith("cuda") else args.device
    env = {k: v for k, v in os.environ.items() if not k.startswith("MLFLOW_RUN")}
    failures: list[dict] = []

    def command(job: tuple) -> tuple[str, list[str]]:
        d, s, m = job
        return f"{d}_{s}_{m}", ["--regressor-path", path, "--datasets", d,
                                "--adbench-splits", s, "--adbench-methods", m,
                                "--device", device]  # fmt: skip

    def on_done(job: tuple, returncode: int, job_dir: Path) -> bool:
        rows = adbench.read_results(job_dir / SUITE / "results.csv")
        if returncode == 0 and len(rows):
            results.append(rows)
            return True
        d, s, m = job
        failures.append(dict(dataset=d, split=s, method=m, error=f"exit {returncode}"))
        return False

    print(f"adbench shards: {len(jobs)} jobs -> {len(devices)} GPUs", flush=True)
    run_shard_queue(
        "adbench", jobs, command, on_done, devices, args.script, base, env, suite_dir / "shards",
        args.heartbeat_minutes,
    )  # fmt: skip
    return failures
