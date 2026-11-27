"""Time the generation of single training tables of one or more priors, as the prior workers of
the training loop make them (`tabsil_train.training.data.PriorBatches`: one process, one torch
thread, `sample_batch(micro_batch)` with the table sizes of a curriculum stage).

The training loop's DataLoader hands tables over in worker order, so one table that takes much
longer than the others holds back its GPU, and through the gradient all-reduce every other GPU.
The mean generation time does not show this; the tail does. This script measures the distribution
and where the time goes:

- `scm_s`, `scm_calls`: the TabICL graph SCM draws (one per attempt)
- `filter_s`, `filter_calls`, `filtered`: the unpredictable-dataset filter (an ExtraTrees fit per
  attempt, `tabicl.prior._dataset.should_filter`) and how many attempts it rejected
- `groups`: pipeline attempts (more than 1 when a transform dropped the dataset)
- `cpu_s`: the process's CPU time. Well below `seconds` means the process waited for a CPU
- the rest of `seconds` is the transforms

Every process writes one CSV row per table. A table that has started is always finished and
recorded, so slow tables are not cut off at the time limit. The rows are merged into
`prior_timing.csv.gz`, summarized on stdout and logged to the MLflow run (`MLFLOW_RUN_ID` on AI
Runtime). `queue_sim.py` in this folder reads the CSV.

    python prior_timing.py --priors graph_scm_hicard_missing graph_scm_realism --minutes 20

A prior given as `name={"option": value, ...}` is the `graph_scm` prior with those
`GraphSCMPrior` options, to time single options that no registered prior turns on alone:

    python prior_timing.py --priors 'cat_sizes={"corrected_cat_sizes": true}'

Run it on AI Runtime (`prior_timing.yaml`), not locally.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import multiprocessing as mp
import os
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

TASKS = ("classification", "regression")
COUNTERS = ("groups", "scm_calls", "scm_s", "filter_calls", "filter_s", "filtered")
FIELDS = (
    "prior",
    "task",
    "worker",
    "index",
    "start_s",
    "seconds",
    "cpu_s",
    "rows",
    "n_train",
    "features",
    *COUNTERS,
)
NUMERIC = FIELDS[2:]
# Tail thresholds of the summary, in seconds
TAIL_SECONDS = (5, 10, 20, 30, 60, 120)


def instrument(prior) -> dict:
    """Count and time the generator calls of `prior` (a `GraphSCMPrior`). Returns the counters,
    which the caller resets per table. The prior's random draws are unchanged."""
    stats = dict.fromkeys(COUNTERS, 0)
    generator = prior.pipeline.generator

    def counted_generator(*args, **kwargs):
        stats["groups"] += 1
        return generator(*args, **kwargs)

    prior.pipeline.generator = counted_generator
    make_scm, should_filter = generator._graph_scm, generator._should_filter

    def timed_scm(*args, **kwargs):
        make = make_scm(*args, **kwargs)

        def call():
            start = time.perf_counter()
            out = make()
            stats["scm_s"] += time.perf_counter() - start
            stats["scm_calls"] += 1
            return out

        return call

    def timed_filter(*args, **kwargs):
        start = time.perf_counter()
        rejected = should_filter(*args, **kwargs)
        stats["filter_s"] += time.perf_counter() - start
        stats["filter_calls"] += 1
        stats["filtered"] += bool(rejected)
        return rejected

    generator._graph_scm, generator._should_filter = timed_scm, timed_filter
    return stats


def parse_prior(spec: str) -> tuple[str, dict | None]:
    """(name, options): a registered prior type (options None), or `name={json options}` for the
    `graph_scm` prior with those options."""
    name, sep, options = spec.partition("=")
    return (name, json.loads(options)) if sep else (spec, None)


def worker(args: tuple) -> str:
    """Generate tables of one prior and task until `seconds` have passed; one CSV row each."""
    spec, stage, task, index, seed, seconds, path = args
    import torch

    from tabsil_train import recipe
    from tabsil_train.priors import graph_scm

    torch.set_num_threads(1)  # as PriorBatches does in a DataLoader worker
    prior_type, options = parse_prior(spec)
    # a different seed per process and task, the same for every prior
    prior_seed = hash((seed, TASKS.index(task), index)) % (2**31)
    if options is None:
        config = replace(recipe.curriculum(stage), prior_type=prior_type)
        prior = config.make_prior(task, prior_seed)
    else:
        config = recipe.curriculum(stage)
        prior = graph_scm.make_prior(config, task, prior_seed, **options)
    stats = instrument(prior)
    begin = time.perf_counter()
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        n = 0
        while (start := time.perf_counter()) - begin < seconds:
            for key in stats:
                stats[key] = 0
            cpu = time.process_time()
            batch = prior.sample_batch(config.micro_batch)
            took, cpu = time.perf_counter() - start, time.process_time() - cpu
            _, rows, features = batch.x.shape
            row = [prior_type, task, index, n, round(start - begin, 3), round(took, 4)]
            row += [round(cpu, 4), rows, batch.n_train, features]
            writer.writerow(row + [round(stats[key], 4) for key in COUNTERS])
            f.flush()
            n += 1
    return path


def summarize(rows: np.ndarray) -> dict[str, float]:
    """Summary of one prior and task: `rows` is (tables, `NUMERIC` fields)."""
    col = {name: rows[:, i] for i, name in enumerate(NUMERIC)}
    seconds = col["seconds"]
    out = {
        "tables": float(len(seconds)),
        "mean_s": seconds.mean(),
        "p50_s": np.quantile(seconds, 0.5),
        "p90_s": np.quantile(seconds, 0.9),
        "p99_s": np.quantile(seconds, 0.99),
        "p999_s": np.quantile(seconds, 0.999),
        "max_s": seconds.max(),
        "cpu_share": col["cpu_s"].sum() / seconds.sum(),
        "scm_calls_mean": col["scm_calls"].mean(),
        "scm_calls_max": col["scm_calls"].max(),
        "filtered_share": col["filtered"].sum() / max(col["filter_calls"].sum(), 1),
        "time_share_scm": col["scm_s"].sum() / seconds.sum(),
        "time_share_filter": col["filter_s"].sum() / seconds.sum(),
        "redrawn_share": (col["groups"] > 1).mean(),
    }
    for limit in TAIL_SECONDS:
        over = seconds > limit
        out[f"share_over_{limit}s"] = over.mean()
        out[f"time_share_over_{limit}s"] = seconds[over].sum() / seconds.sum()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--priors", nargs="+", required=True)
    parser.add_argument("--stage", type=int, default=2, help="curriculum stage (table sizes)")
    parser.add_argument("--minutes", type=float, default=20.0, help="per prior")
    # half of the CPUs by default: the training nodes run their prior workers far below the CPU
    # count, and a fully loaded node would inflate the times
    parser.add_argument("--processes", type=int, default=None, help="default: half of the CPUs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=os.environ.get("DIAG_OUT", "/tmp/v2_prior_diag"))
    args = parser.parse_args()
    processes = args.processes or max(1, (os.cpu_count() or 2) // 2)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    import mlflow

    merged = out / "prior_timing.csv.gz"
    with mlflow.start_run(run_id=os.environ.get("MLFLOW_RUN_ID")):
        mlflow.log_params(
            {
                "priors": ",".join(args.priors),
                "stage": args.stage,
                "minutes_per_prior": args.minutes,
                "processes": processes,
                "cpu_count": os.cpu_count(),
                "seed": args.seed,
            }
        )
        with gzip.open(merged, "wt", newline="") as f:
            csv.writer(f).writerow(FIELDS)
        for spec in args.priors:
            prior_type, options = parse_prior(spec)
            if options is not None:
                mlflow.log_param(f"options_{prior_type}", json.dumps(options, sort_keys=True))
            # the processes alternate between the tasks, as a GPU's workers are split between them
            jobs = [
                (spec, args.stage, TASKS[i % 2], i, args.seed, args.minutes * 60)
                + (str(out / f"{prior_type}_{i}.csv"),)
                for i in range(processes)
            ]
            start = time.time()
            with mp.get_context("spawn").Pool(processes) as pool:
                paths = pool.map(worker, jobs, chunksize=1)
            minutes = (time.time() - start) / 60
            print(f"== {prior_type}: {processes} processes, {minutes:.1f} min", flush=True)
            table = []
            for path in paths:
                with open(path, newline="") as f:
                    table += list(csv.reader(f))
            with gzip.open(merged, "at", newline="") as f:
                csv.writer(f).writerows(table)
            for task in TASKS:
                rows = np.array(
                    [[float(v) for v in line[2:]] for line in table if line[1] == task],
                    dtype=np.float64,
                )
                if not len(rows):
                    continue
                summary = summarize(rows)
                print(f"-- {prior_type} {task}")
                for key, value in summary.items():
                    print(f"   {key:24s} {value:.5g}")
                mlflow.log_metrics(
                    {f"{prior_type}/{task}/{k}": float(v) for k, v in summary.items()}
                )
                slowest = rows[np.argsort(-rows[:, NUMERIC.index("seconds")])[:10]]
                print("   slowest tables (" + ", ".join(NUMERIC[3:]) + "):")
                for line in slowest:
                    print("     " + ", ".join(f"{v:g}" for v in line[3:]), flush=True)
            # log after every prior, so a later failure or the job timeout does not lose it
            mlflow.log_artifact(str(merged), "diag")


if __name__ == "__main__":
    main()
