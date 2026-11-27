"""`eval.py --suite impute`: missing value imputation with the model and the (table, rate, seed,
method) grid of the command line (`tabsil_train.benchmarks.impute` is the benchmark: data, masks,
methods, report).

- `impute_grid`: the grid, checked
- `run_impute`: loads the model, runs the grid in this process or, with `--shards`, as worker
  processes, writes the report and logs to MLflow
- `run_impute_shards`: the `--shards` queue (GPU jobs, and the CPU methods in one more process),
  with `impute_worker_threads` CPU threads per worker
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from tabsil_train.benchmarks import _files, impute
from tabsil_train.evaluation.model import suite_model
from tabsil_train.evaluation.shards import run_shard_queue, shard_devices, strip_flags
from tabsil_train.hub import download_artifact

# the `--suite` name: the results folder under --out-dir and the MLflow artifact folder
SUITE = "impute"


def impute_grid(args: argparse.Namespace):
    """(tables, rates, seeds, methods) of `--suite impute`, checked."""
    datasets = list(impute.TABLES) if args.datasets is None else args.datasets.split(",")
    if bad := [d for d in datasets if d not in impute.TABLES]:
        raise SystemExit(f"tables not in the impute suite: {bad} (known: {list(impute.TABLES)})")
    rates = [float(r) for r in args.impute_rates.split(",")]
    if bad := [r for r in rates if not 0 < r < 1]:
        raise SystemExit(f"rates must be in (0, 1): {bad}")
    seeds = [int(s) for s in args.impute_seeds.split(",")]
    try:
        methods = impute.check_methods(args.impute_methods.split(","))
    except ValueError as e:
        raise SystemExit(str(e)) from None
    return datasets, rates, seeds, methods


def run_impute(args: argparse.Namespace, out_dir: Path, use_mlflow: bool) -> None:
    """Missing value imputation (`tabsil_train.benchmarks.impute`). Ends with an error if a
    (table, rate, seed, method) failed."""
    datasets, rates, seeds, methods = impute_grid(args)
    imputers: dict = dict(impute.CPU_IMPUTERS)
    opts = {
        "knn": "KNNImputer(n_neighbors=5), z-scored + one-hot",
        "mice": "chained: BayesianRidge / LogisticRegression, 10 rounds, z-scored + one-hot",
        "missforest": "100 trees, max_features=sqrt, max_iter=10",
    }
    params = {"suite": args.suite, "datasets": ",".join(datasets),
              "rates": ",".join(map(str, rates)), "seeds": ",".join(map(str, seeds)),
              "methods": ",".join(methods)}  # fmt: skip
    for m in ("tabpfn_impute", "tabpfn_simple"):
        imputers[m] = impute.tabpfn_imputer(args.device, simple=m == "tabpfn_simple")
        opts[m] = f"tabpfn-extensions {impute.TABPFN_EXTENSIONS_VERSION}, TabPFN v2 defaults"
    # `impute` rows from before its column orders followed the seed are another computation
    opts[impute.OFFICIAL] += ", column orders from the seed"
    ours = [m for m in methods if m.startswith(impute.OURS)]
    path = args.regressor_path or args.classifier_path
    if ours:
        # with --shards the workers impute; this process loads the model for its settings string
        model, borders, n_estimators, settings = suite_model(
            args, path, out_dir, "impute", "categorical columns"
        )
        for m in ours:
            if args.shards == 1:
                imputers[m] = impute.tabsil_imputer(
                    model, borders, n_estimators, args.device, 100 * args.seed,
                    extension_kinds=m == "tabsil_inferred",
                )  # fmt: skip
            opts[m] = settings
        params |= {"model": path, "n_estimators": n_estimators, "seed": args.seed}
        del model, borders  # the imputers hold them (none with --shards)
    print(f"impute: {len(datasets)} tables x rates {rates} x seeds {seeds} x {methods}, "
          f"{args.device}", flush=True)  # fmt: skip
    if use_mlflow:
        mlflow.log_params(params)
    suite_dir = out_dir / SUITE
    t0 = time.time()
    if args.shards > 1:
        failures = run_impute_shards(
            args, datasets, rates, seeds, methods, suite_dir, out_dir, use_mlflow, opts
        )
    else:
        _, failures = impute.run(imputers, datasets, rates, seeds, methods, suite_dir, opts)
    rows = impute.read_results(suite_dir / "results.csv")
    summary = impute.report(rows, suite_dir, datasets, rates, seeds, impute.result_methods(methods))
    summary |= {"failures": len(failures), "seconds": round(time.time() - t0, 1)}
    (suite_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps(summary, indent=1, default=str), flush=True)
    if (suite_dir / "summary.csv").exists():
        table = pd.read_csv(suite_dir / "summary.csv").round(4)
        print(f"\nsummary.csv\n{table.to_string(index=False)}", flush=True)
    if use_mlflow:
        metrics = {f"impute_{k}": v for k, v in summary.items() if isinstance(v, int | float)}
        for metric in ("nrmse", "accuracy"):
            for rate, by_method in summary.get(metric, {}).items():
                metrics |= {f"impute_{metric}_{rate}_{m}": v for m, v in by_method.items()}
        metrics = {k: v for k, v in metrics.items() if np.isfinite(v)}
        mlflow.log_metrics({re.sub(r"[^\w\-. /]", "_", k): float(v) for k, v in metrics.items()})
        mlflow.log_artifacts(str(suite_dir), SUITE)
    if failures:
        raise SystemExit(f"impute: {len(failures)} failed (table, rate, seed, method): {failures}")


# thread count variables of the numeric libraries a worker uses (torch reads OMP_NUM_THREADS;
# TABSIL_IMPUTE_N_JOBS is MissForest's n_jobs, `tabsil_train.benchmarks.impute.n_jobs`)
WORKER_THREAD_VARIABLES = (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "TABSIL_IMPUTE_N_JOBS",
)  # fmt: skip


def impute_worker_threads(workers: int, cores: int | None = None) -> int:
    """CPU threads per worker process: the cores (`os.sched_getaffinity` where available, else
    `os.cpu_count`) split evenly over `workers`, at least 1."""
    if cores is None:
        try:
            cores = len(os.sched_getaffinity(0))
        except AttributeError:  # macOS
            cores = os.cpu_count() or 1
    return max(1, cores // max(workers, 1))


def _result_rows(job_dir: Path) -> int:
    """Result rows a running impute worker has written so far."""
    path = job_dir / SUITE / "results.csv"
    if not path.exists():
        return 0
    with open(path) as f:
        return max(0, sum(1 for _ in f) - 1)


def run_impute_shards(
    args: argparse.Namespace,
    datasets: list[str],
    rates: list[float],
    seeds: list[int],
    methods: list[str],
    suite_dir: Path,
    out_dir: Path,
    use_mlflow: bool = False,
    opts: dict[str, str] | None = None,
) -> list[dict]:
    """`--suite impute --shards N`: the GPU methods as one job per (table, rate, method) not yet
    complete in `results.csv`, most expensive first, each a worker process (`eval.py --suite
    impute` on one GPU, without MLflow) started as soon as a GPU is free, and the CPU methods in
    one worker process (`--device cpu`) running at the same time (`run_shard_queue`). Every
    worker gets an equal share of the CPU cores (`impute_worker_threads`): TabPFN's preprocessing
    and torch's CPU work run on the CPU, and N workers each using every core oversubscribe it
    (the GPUs then sit idle). A finished job's rows are appended to `results.csv` (and logged to
    MLflow, so a run cut short keeps them); a failed job is returned as a failure (log under
    `shards/`). Rows of `methods` with other settings than `opts` (job method -> settings string)
    stop the run."""
    # the settings are checked here as well as in `impute.run`: a job whose rows are already
    # there never reaches a worker
    settings = impute.row_settings(methods, opts)
    results = _files.Results(suite_dir, impute.RESULT_COLUMNS, settings, SUITE)
    finished = impute.finished_keys(results.done)
    # the workers keep their failures in their own folders: a failures.json here is an earlier
    # run's without --shards
    (suite_dir / "failures.json").unlink(missing_ok=True)

    def complete(d: str, r: float, m: str) -> bool:
        return all((d, round(r, 4), s, out) in finished for s in seeds for out in impute.METHODS[m])

    gpu = [m for m in methods if m in impute.GPU_METHODS]
    cpu = [m for m in methods if m not in impute.GPU_METHODS]
    jobs = [(d, r, m) for d in datasets for r in rates for m in gpu if not complete(d, r, m)]
    jobs.sort(key=lambda j: -impute.job_cost(j[0], j[2]))
    devices = shard_devices(args)
    path = args.regressor_path or args.classifier_path
    if path and path.startswith("runs:/"):  # the parent's download, reused by every worker
        path = str(download_artifact(path, out_dir / "checkpoints" / "regression"))
    base = strip_flags(
        args.argv,
        {"--shards": 1, "--datasets": 1, "--impute-rates": 1, "--impute-methods": 1,
         "--out-dir": 1, "--device": 1, "--mlflow": 0, "--regressor-path": 1,
         "--classifier-path": 1, "--heartbeat-minutes": 1},
    )  # fmt: skip
    model_args = ["--regressor-path", path] if path else []
    device = "cuda" if args.device.startswith("cuda") else args.device
    threads = impute_worker_threads(len(devices) + (1 if cpu else 0))
    env = {k: v for k, v in os.environ.items() if not k.startswith("MLFLOW_RUN")}
    env |= {k: str(threads) for k in WORKER_THREAD_VARIABLES}
    failures: list[dict] = []

    def command(job: tuple) -> tuple[str, list[str]]:
        d, r, m = job
        return f"{d}_{r}_{m}", [*model_args, "--datasets", d, "--impute-rates", str(r),
                                "--impute-methods", m, "--device", device]  # fmt: skip

    def on_done(job: tuple, returncode: int, job_dir: Path) -> bool:
        rows = impute.read_results(job_dir / SUITE / "results.csv")
        # a worker resumes from its folder under shards/, which stays between runs, so its file
        # can hold rows merged by an earlier run: append only the new ones
        rows = impute.new_rows(rows, impute.read_results(results.path))
        # nor rows of other settings an earlier run left there (the worker stops on them)
        rows = rows[~_files.other_settings(rows, settings)]
        if len(rows):  # rows a failed job finished still count
            results.append(rows)
            if use_mlflow:
                mlflow.log_artifact(str(results.path), SUITE)
        if returncode != 0:
            failures.append(dict(job=job, error=f"exit {returncode}"))
        return returncode == 0

    side = []
    if cpu:
        side = [(("cpu", ",".join(cpu)), "cpu", [
            "--datasets", ",".join(datasets), "--impute-rates", ",".join(map(str, rates)),
            "--impute-methods", ",".join(cpu), "--device", "cpu",
        ])]  # fmt: skip
    print(
        f"impute shards: {len(jobs)} GPU jobs -> {len(devices)} GPUs, CPU methods {cpu}, "
        f"{threads} CPU threads per worker",
        flush=True,
    )
    run_shard_queue(
        "impute", jobs, command, on_done, devices, args.script, base, env, suite_dir / "shards",
        args.heartbeat_minutes, side=side, progress=lambda d: f"{_result_rows(d)} rows",
    )  # fmt: skip
    return failures
