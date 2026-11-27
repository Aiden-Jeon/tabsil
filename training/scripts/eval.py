"""Evaluate TabSil on the AMLB benchmark, TabArena-Lite, BeyondArena, fev-bench forecasting,
ADBench outlier detection, treatment effect benchmarks (IHDP, ACIC 2016) and missing
value imputation.

The AMLB benchmark is the TabPFN v2 paper's: 29 AMLB classification and 28 AMLB/CTR23 regression
tasks, each scored as the mean over the OpenML task's 10-fold cross-validation (ROC AUC for
classification, negative RMSE for regression) and printed next to the paper's TabPFN score. The
model is a checkpoint saved by `tabsil.models.save_checkpoint`, a checkpoint a training run logged
to MLflow (`runs:/<run_id>/checkpoints/step_XXXXXXX.pt`), or a Hub repo id (`hf://owner/name`,
uploaded with `tabsil.hub.save_pretrained`).

From the repository root (`EVAL=training/scripts/eval.py`, `RUN="uv run --package tabsil-train"`):
  $RUN python $EVAL --classifier-path clf.pt --regressor-path reg.pt   # 57 tasks x 10 folds
  $RUN python $EVAL --suite classification --classifier-path clf.pt --tasks 359955 --folds 2
  $RUN python $EVAL --suite classification --classifier-path runs:/<run_id>/checkpoints/latest.pt

Every fold row (`eval_rows.csv`) also has TabArena's error (`metric_error`: 1 - ROC AUC for binary,
log loss for multiclass, RMSE for regression) and `problem_type`, which the paired report
(`training/scripts/paired_report.py`) compares between two runs. `--suite dev` is the development
suite of the v2 experiment plan: the 57 tasks without the 9 whose data is also in TabArena's 51
(`tabsil_train.benchmarks.tabular.amlb.TABARENA_OVERLAP`), 23 classification and 25 regression
tasks.

The ensemble member count and preprocessing come from the model (`tabsil.recipe.default_inference`:
the Hub config.json `inference` entry first, otherwise 8 `standard` members with at most 768
columns per member chosen `balanced`). `--n-estimators` changes the member count and `--seed` the
members' random numbers (feature and class order, identifier column): re-evaluating the same model
with another seed measures the benchmark noise. Arena result caches are per seed, and per checkpoint
path, weights (a digest of the state dict and borders) and applied inference settings. The settings
each task ran with are logged as the MLflow params `inference_classification` and
`inference_regression` (JSON).

`--uncertainty` (AMLB) adds predictive distribution and uncertainty metrics per fold
(`tabsil_train.benchmarks.tabular.uncertainty`): for regression CRPS, central interval
(50/80/90/95%) coverage, width and interval score, the minimum coverage over width quintiles and the
PIT KS statistic; for classification ECE, log loss, Brier and split conformal (LAC, APS,
alpha = 0.1) coverage, set size, SSCS and class-conditional coverage gap. The per-dataset table is
`eval_uncertainty_per_dataset.csv` and the suite means are `<suite>_mean_<metric>` (regression CRPS,
widths and interval scores are averaged only as `_scaled`, divided by the training target standard
deviation).

`--suite tabarena-lite`, `tabarena-3split`, `beyondarena` and `beyondarena-lite` are TabArena-Lite,
the same 51 datasets with their first 3 splits (153 splits; every baseline is scored on the same
splits) and BeyondArena (`tabsil_train.benchmarks.tabular.arena`, needs the `arena` extra, which
needs pandas 2.3 and is installed apart from the dev group:
`uv sync --package tabsil-train --no-dev --extra arena --group arena-dev`, with the same options
for `uv run`). The official splits run and Elo, normalized score,
improvability and win rates are computed as on the public leaderboard. Splits of a task without a
checkpoint, or beyond TabSil's supported size (`tabsil.recipe.SUPPORTED_SIZE`; classes up to what
the model takes, `max_classes_of`), are imputed (`--max-train-rows`, `--max-features` change the
limits). One fit uses the whole training split. `--datasets` runs only some datasets (the rest
imputed). Results go to `<out-dir>/<suite>/`.

  $RUN --no-dev --extra arena --group arena-dev python $EVAL --suite tabarena-lite \
    --classifier-path hf://aiden-jeon/tabsil@v1.1 --regressor-path hf://aiden-jeon/tabsil@v1.1

Columns per member: without `--max-features-per-member` the model's setting applies (768 columns
chosen `balanced`, per task model). `--max-features-per-member 0` turns the cap off (all columns;
the arena feature limit is then the supported size). With a cap, the TabArena / BeyondArena feature
limit is lifted, so splits wider than the supported size run.

`--subsample-train-rows N` uses at most N random training rows per fit as the context
(`tabsil_train.benchmarks.tabular.subsample`; stratified by class for classification with at least
one row per class, drawn with `--seed`, the same rows for every member). `--max-train-rows` filters
splits by their original training row count.

`--suite fev` is time series forecasting on 10 fev-bench tasks
(`tabsil_train.benchmarks.forecast.fevbench`, needs the `fev` extra, installed apart from the dev
group like the arena extra:
`uv sync --package tabsil-train --no-dev --extra fev --group arena-dev`): `tabsil.TabSilForecaster`
with the `--regressor-path` model on every rolling window, scored with fev's SQL and MASE and put in
an Elo pool (Seasonal Naive = 1000) with the per-window results fm-agent-bench recorded for 7
methods. Our Seasonal Naive is recomputed on the same windows and must match those. `--datasets`
picks tasks, `--fev-windows N` only the last N windows of each (a quick check; the task-mean Elo
pool then drops out). A failed series, window or task stops the run (nothing is imputed). Results go
to `<out-dir>/fev/`.

  $RUN --no-dev --extra fev --group arena-dev python $EVAL --suite fev \
    --regressor-path hf://aiden-jeon/tabsil@v1 --datasets ecdc_ili --fev-windows 2

`--suite causal` is treatment effect estimation on IHDP (100 realizations) and ACIC 2016 (10
simulations) (`tabsil_train.benchmarks.causal.causal`, needs the `causal` extra for the EconML
baselines: `uv sync --package tabsil-train --extra causal`): `tabsil.TabSilCATE` with the S-, T-, X-
and DR-learner on the `--regressor-path` model (it must serve classification too, for the
propensity), and EconML's causal forest and GBM-based meta-learners on the same splits. √PEHE and
the ATE error within-sample and out-of-sample. `--datasets` picks the benchmarks (ihdp, acic2016),
`--causal-realizations` the realizations (`1-5,9`), `--causal-methods` the methods. Results go to
`<out-dir>/causal/` and resume from its `results.csv`; a failed cell ends the run with an error
after the others (nothing is imputed).

  $RUN --extra causal python $EVAL --suite causal --regressor-path hf://aiden-jeon/tabsil@v1 \
    --datasets ihdp --causal-realizations 1-3

`--latency` (arenas) records each fit's one-row prediction time (median of `LATENCY_REPEATS`) in
`<out-dir>/fits_<suite>.jsonl`, measured in a hook outside TabArena's train and inference timing.

Time is reported as on the TabArena leaderboard: train time per 1K train rows and prediction time
per 1K test rows (s), averaged over a dataset's splits, median over datasets
(`time_{train,infer}_s_per_1K_median`). The arenas put it in `summary.json` (with the raw seconds
and the baselines' value on the same splits, `time_baselines`) and `comparison.csv`; AMLB writes
each fold's fit and prediction time to `<out-dir>/fits_<suite>.jsonl` and `eval_rows.csv`.

`--suite adbench` is unsupervised outlier detection on 12 ADBench tables x 3 seeds
(`tabsil_train.benchmarks.unsupervised.adbench`): `tabsil.TabSilUnsupervised` fitted on the
unlabeled (contaminated) training rows scores the test rows, next to the official TabPFN v2
extension and pyod's methods from the fm-agent-bench run, with IsolationForest = 1000 Elo. One
multitask model serves both column kinds (`--regressor-path`, or `--classifier-path`).
`--adbench-methods` adds IsolationForest (the split check against ADBench Table D4) and the `+clean`
diagnostic (context = training rows labeled normal; kept out of the Elo); `--datasets` and
`--adbench-splits` pick tables and seeds. Results go to `<out-dir>/adbench/` and resume from its
`results.csv`.

  $RUN python $EVAL --suite adbench --regressor-path hf://aiden-jeon/tabsil@v1 \
    --datasets yeast --adbench-splits s1

`--suite impute` is missing value imputation on 10 OpenML tables x MCAR rates 10/30/50% x 3 seeds
(`tabsil_train.benchmarks.unsupervised.impute`): `TabSilUnsupervised.impute` (zero-shot, the
`--regressor-path` model serving both column kinds; mean and median from one evaluation) against the
official TabPFN v2 extension (`impute` and `simple_impute`, needs the `impute` extra:
`uv sync --all-packages --extra impute`), mean / most frequent, KNN, MICE and MissForest, all run on
the same rows and masks and scored on the masked cells (numeric: RMSE / std, categorical: accuracy).
`--impute-methods`, `--impute-rates`, `--impute-seeds` and `--datasets` pick the grid. A failed
(table, rate, seed, method) is reported and the run ends with an error (nothing is filled in).
With `--shards N` the GPU methods run as (table, rate, method) jobs on N GPUs and the CPU
baselines in one more process at the same time. Results go to `<out-dir>/impute/` and resume from
its `results.csv`.

  $RUN --extra impute python $EVAL --suite impute --regressor-path hf://aiden-jeon/tabsil@v1 \
    --datasets airfoil_self_noise --impute-rates 0.3 --impute-seeds 0

The ADBench, causal and impute suites reuse a `results.csv` row of a TabSil method only when it
was computed with this run's settings (the row's `opts`: a digest of the weights rather than the
checkpoint path, the member count, `--seed`, the suite's options and the inference settings saved
with the weights). Otherwise the run stops: use another `--out-dir`. Rows of the baselines
(IsolationForest, EconML, mean, KNN, ...) are reused whatever the model.

The fev, ADBench, causal and impute runs are in `tabsil_train.evaluation`, one module per suite
(`shards` is the `--shards` worker queue of ADBench and impute). The AMLB and TabArena /
BeyondArena runs are in this file.

On AI Runtime (`training/scripts/eval.yaml`, `training/scripts/eval_arena.yaml`) results are
logged to the platform's MLflow run (`MLFLOW_RUN_ID`). Locally, `--mlflow` creates a new run in the
`TABSIL_EXPERIMENT` experiment (default `tabsil`).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch

from tabsil.inference.ensemble import (
    EnsembleClassifier,
    EnsembleRegressor,
    max_classes_of,
    max_features_of,
)
from tabsil.inference.preprocessing import CATEGORY_ENCODINGS
from tabsil.recipe import SUPPORTED_SIZE, default_inference, resolve_member_kinds
from tabsil_train.benchmarks.tabular import amlb as bm
from tabsil_train.benchmarks.tabular import uncertainty as unc
from tabsil_train.benchmarks.tabular.subsample import SubsampledEstimator
from tabsil_train.evaluation.adbench import run_adbench
from tabsil_train.evaluation.causal import run_causal
from tabsil_train.evaluation.fev import run_fev
from tabsil_train.evaluation.impute import run_impute
from tabsil_train.evaluation.shards import clock as _clock
from tabsil_train.evaluation.shards import strip_flags as _strip_flags
from tabsil_train.hub import load_model, weights_digest
from tabsil_train.utils import mlflow_eval_run

ARENA_SUITES = ("tabarena-lite", "tabarena-3split", "beyondarena", "beyondarena-lite")
FEV_SUITE = "fev"
ADBENCH_SUITE = "adbench"
CAUSAL_SUITE = "causal"
IMPUTE_SUITE = "impute"
DEV_SUITE = "dev"
LATENCY_REPEATS = 5


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--suite",
        choices=[
            "classification",
            "regression",
            "all",
            DEV_SUITE,
            *ARENA_SUITES,
            FEV_SUITE,
            ADBENCH_SUITE,
            CAUSAL_SUITE,
            IMPUTE_SUITE,
        ],
        default="all",
    )
    parser.add_argument("--tasks", help="comma-separated OpenML task ids (chosen within the suite)")
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument(
        "--classifier-path",
        help="classification checkpoint (file, folder, runs:/ URI, hf://owner/name)",
    )
    parser.add_argument(
        "--regressor-path",
        help="regression checkpoint (file, folder, runs:/ URI, hf://owner/name)",
    )
    parser.add_argument(
        "--n-estimators", type=int, default=None, help="ensemble members (default: the model's)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="random seed of the ensemble members (feature and class order, identifier column;"
        " change it to measure one model's evaluation noise). With --subsample-train-rows it"
        " also picks the training rows",
    )
    parser.add_argument(
        "--uncertainty",
        action="store_true",
        help="AMLB: also report predictive distribution and uncertainty metrics (regression"
        " CRPS, coverage, PIT; classification ECE, conformal)",
    )
    parser.add_argument(
        "--latency",
        action="store_true",
        help="TabArena / BeyondArena: record each fit's one-row prediction time in"
        " fits_<suite>.jsonl",
    )
    parser.add_argument(
        "--datasets",
        help="TabArena / BeyondArena: comma-separated dataset names (the rest imputed); fev:"
        " comma-separated task names (only those are evaluated and ranked); ADBench: the tables"
        " to run; causal: the benchmarks (ihdp, acic2016; default both); impute: the tables to"
        " run",
    )
    parser.add_argument(
        "--causal-realizations",
        default=None,
        help="causal: realizations of each benchmark, comma-separated numbers and ranges"
        " (1-5,9; default all: IHDP 1-100, ACIC 2016 1-10)",
    )
    parser.add_argument(
        "--causal-methods",
        default=None,
        help="causal: comma-separated methods (tabsil_s, tabsil_t, tabsil_x, tabsil_dr,"
        " econml_cf, econml_s, econml_t, econml_x, econml_dr; default all)",
    )
    parser.add_argument(
        "--fev-windows",
        type=int,
        default=None,
        help="fev: evaluate only the last N rolling windows of each task (default: all)",
    )
    parser.add_argument(
        "--max-context-length",
        type=int,
        default=None,
        help="fev: context rows per series (default: TabSilForecaster's, 4096; 0: up to the"
        " supported size)",
    )
    parser.add_argument(
        "--adbench-splits",
        default="s1,s2,s3",
        help="ADBench: comma-separated splits (s1, s2, s3 = ADBench seeds 1, 2, 3)",
    )
    parser.add_argument(
        "--adbench-methods",
        default="tabsil_unsup,tabsil_unsup+clean,iforest",
        help="ADBench: comma-separated methods (tabsil_unsup, iforest; +clean = context of the"
        " training rows labeled normal, a diagnostic outside the Elo)",
    )
    parser.add_argument(
        "--impute-methods",
        default="tabsil,tabsil_inferred,tabpfn_impute,tabpfn_simple,mean,knn,mice,missforest",
        help="impute: comma-separated methods (tabsil also writes tabsil_median)",
    )
    parser.add_argument(
        "--impute-rates", default="0.1,0.3,0.5", help="impute: comma-separated MCAR rates"
    )
    parser.add_argument("--impute-seeds", default="0,1,2", help="impute: comma-separated seeds")
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=10,
        help="ADBench: column orders averaged by the chain-rule log density",
    )
    parser.add_argument(
        "--shards",
        type=int,
        default=1,
        help="TabArena / BeyondArena: start one worker per GPU and hand out datasets one at a"
        " time from a queue (most expensive first; at most the GPU count, 1: one process)",
    )
    parser.add_argument(
        "--heartbeat-minutes",
        type=float,
        default=5.0,
        help="--shards: print progress (finished splits, per-worker dataset and elapsed time)"
        " at this interval (minutes)",
    )
    parser.add_argument(
        "--host-memory-budget-gb",
        type=float,
        default=None,
        help="--shards: hand out large tables that keep cell state on the CPU only while the"
        " concurrent total stays within this many GB (default: 50%% of the container memory"
        " limit (cgroup), or 50%% of physical memory (/proc/meminfo) if there is no limit or"
        " it cannot be read; 0: off). Only the queue order changes",
    )
    # internal: worker that runs datasets received on stdin one at a time (started by `--shards`)
    parser.add_argument("--shard-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--max-train-rows",
        type=int,
        help="TabArena / BeyondArena: skip (impute) splits with more original training rows"
        " (default: the model's architecture; counted before --subsample-train-rows)",
    )
    parser.add_argument(
        "--subsample-train-rows",
        type=int,
        default=None,
        help="per fit, use at most this many random training rows as the context"
        " (classification stratified by class, at least one row per class, drawn with --seed,"
        " the same rows for every member). Applies to the splits left"
        " after --max-train-rows (default: no subsampling)",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        help="TabArena / BeyondArena: feature count limit (default: the model's architecture)",
    )
    parser.add_argument(
        "--max-features-per-member",
        type=int,
        default=None,
        help="with more features than this, each ensemble member picks this many (official"
        " TabPFN's max_features_per_estimator). Default: the model's (TabSil: 768)."
        " 0 turns it off (all columns). With a cap, the TabArena / BeyondArena feature limit"
        " is lifted (--max-features sets it again)",
    )
    parser.add_argument(
        "--method-name", default="tabsil", help="method name in TabArena / BeyondArena results"
    )
    # inference options (`tabsil.inference.ensemble`, `tabsil.inference.preprocessing`); the
    # defaults are the model's
    parser.add_argument(
        "--softmax-temperature",
        type=float,
        default=None,
        help="classification: divide the logits by this before the softmax (default: the model's;"
        " TabSil 1.0)",
    )
    parser.add_argument(
        "--member-kinds",
        default=None,
        help="member numeric transforms: a preset (numeric-cycle) or comma-separated kinds from"
        " standard,power,robust,quantile (default: the model's; TabSil numeric-cycle)",
    )
    parser.add_argument(
        "--categories",
        choices=CATEGORY_ENCODINGS,
        default=None,
        help="categorical columns that stay codes with more than 20 categories: keep the codes"
        " only, or add a log count column (default: the model's; TabSil counts)",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", default="tabsil_eval")
    parser.add_argument("--mlflow", action="store_true", help="log to MLflow locally as well")
    args = parser.parse_args(argv)
    if args.max_features_per_member is not None and args.max_features_per_member < 0:
        parser.error("--max-features-per-member must be >= 0 (0: off).")
    if args.subsample_train_rows is not None and args.subsample_train_rows < 1:
        parser.error("--subsample-train-rows must be >= 1.")
    if args.fev_windows is not None and args.fev_windows < 1:
        parser.error("--fev-windows must be >= 1.")
    return args


def selected_tasks(args: argparse.Namespace) -> list[tuple[str, int]]:
    both = args.suite in ("all", DEV_SUITE)
    suites = ["classification", "regression"] if both else [args.suite]
    chosen = None if args.tasks is None else {int(t) for t in args.tasks.split(",")}
    out = []
    for suite in suites:
        ids = bm.DEV_SUITE[suite] if args.suite == DEV_SUITE else bm.task_names(suite)
        for task_id in ids:
            if chosen is None or task_id in chosen:
                out.append((suite, task_id))
    if chosen is not None:
        unknown = sorted(chosen - {task_id for _, task_id in out})
        if unknown:
            raise SystemExit(f"task ids not in --suite {args.suite}: {unknown}")
    if not out:
        raise SystemExit("no tasks to evaluate.")
    return out


def make_estimator_factory(suite: str, args: argparse.Namespace):
    path = args.classifier_path if suite == "classification" else args.regressor_path
    if path is None:
        flag = "--classifier-path" if suite == "classification" else "--regressor-path"
        raise SystemExit(f"{suite} evaluation needs {flag}.")
    # load on the CPU; only the model moves to the device at inference
    model, borders = load_model(path, suite, Path(args.out_dir) / "checkpoints" / suite)
    # defaults: the model's inference setting (Hub config.json `inference`, else from the
    # architecture)
    inference = default_inference(model, suite)
    member_kinds = inference["member_kinds"]
    if getattr(args, "member_kinds", None):
        member_kinds = resolve_member_kinds(args.member_kinds)
    n_estimators = args.n_estimators or inference["n_estimators"]
    fits_log = fits_path(args)
    options = inference_options(args, suite)

    def make(task: bm.OpenMLTask):
        kw = {
            "n_estimators": n_estimators,
            "categorical_features_indices": task.categorical_indices,
            "member_kinds": member_kinds,
            "device": args.device,
            "random_state": args.seed,
            "max_features_per_member": make.max_features_per_member,
            "feature_subsampling": make.feature_subsampling,
            # the model's inference settings, then the command line's
            "categories": inference["categories"],
            **(
                {"softmax_temperature": inference["softmax_temperature"]}
                if suite == "classification"
                else {}
            ),
            **inference_options(args, suite),
        }
        if suite == "classification":
            estimator = EnsembleClassifier(model, **kw)
        else:
            estimator = EnsembleRegressor(model, borders, **kw)
        if args.latency:
            record = {"suite": args.suite, "task": suite, "dataset": getattr(task, "name", None)}
            estimator = RecordedEstimator(estimator, record, fits_log, latency=args.latency)
        if args.subsample_train_rows is not None:
            # outermost: the ensemble and the fit records all see only the chosen rows
            estimator = SubsampledEstimator(
                estimator,
                args.subsample_train_rows,
                stratify=suite == "classification",
                seed=args.seed,
            )
        return estimator

    make.max_one_hot_width = max_features_of(model)  # the host memory estimate uses the same cap
    # `--categories counts` adds up to one column per categorical column (host memory estimate)
    make.count_columns = (getattr(args, "categories", None) or inference["categories"]) == "counts"
    make.n_estimators = n_estimators
    # this task's columns-per-member cap and subsampling method (`model_feature_cap`, TabSil
    # 768 / balanced)
    make.max_features_per_member, make.feature_subsampling = model_feature_cap(args, inference)
    make.model, make.borders = model, borders  # TabArena / BeyondArena size limits (arena_limits)
    # the inference settings the estimators get (model's, then the command line's): logged to
    # MLflow and part of the arena cache key
    make.inference = applied_inference(
        suite, n_estimators, member_kinds, make.max_features_per_member,
        make.feature_subsampling, {**inference, **options},
    )  # fmt: skip
    return make


def applied_inference(
    suite: str,
    n_estimators: int,
    member_kinds,
    max_features_per_member: int | None,
    feature_subsampling: str,
    settings: dict,
) -> dict:
    """The inference settings one task's estimators run with, as plain values (the `inference_*`
    MLflow params and the arena cache key)."""
    out = {
        "n_estimators": int(n_estimators),
        "member_kinds": list(member_kinds),
        "max_features_per_member": max_features_per_member,
        "feature_subsampling": feature_subsampling,
        "categories": settings["categories"],
    }
    if suite == "classification":
        out["softmax_temperature"] = float(settings["softmax_temperature"])
    return out


def inference_options(args: argparse.Namespace, suite: str | None = None) -> dict:
    """The inference options given on the command line (`--softmax-temperature`,
    `--categories`) as estimator arguments; options not given are left out (the model's settings
    apply). With `suite` ("classification" / "regression"), only those of that task (the
    temperature is classification only)."""
    out = {}
    temperature = getattr(args, "softmax_temperature", None)
    if temperature is not None and suite in (None, "classification"):
        out["softmax_temperature"] = temperature
    if getattr(args, "categories", None) is not None:
        out["categories"] = args.categories
    return out


def model_feature_cap(args: argparse.Namespace, inference: dict) -> tuple[int | None, str]:
    """A task's (columns per member cap, subsampling method). `--max-features-per-member` if given
    (0 turns it off: all columns), otherwise the model's inference setting (the recipe's
    `default_inference`, TabSil 768 `balanced`)."""
    cap = args.max_features_per_member
    subsampling = inference.get("feature_subsampling", "balanced")
    if cap == 0:
        return None, subsampling
    if cap is not None:
        return cap, subsampling
    return inference.get("max_features_per_member"), subsampling


def feature_caps(args: argparse.Namespace, factories: dict) -> dict:
    """Columns-per-member cap per task (`model_feature_cap`; `args`' value for non-factories)."""
    return {
        task: getattr(f, "max_features_per_member", args.max_features_per_member or None)
        for task, f in factories.items()
    }


def _one_or_by_task(values: dict):
    # one value if equal across tasks, else a per-task dict (fingerprint, MLflow param)
    distinct = set(values.values())
    return distinct.pop() if len(distinct) == 1 else dict(sorted(values.items()))


def fits_path(args: argparse.Namespace) -> Path:
    """Fit record file (one per suite, cleared by `main` on every run)."""
    return Path(args.out_dir) / f"fits_{args.suite}.jsonl"


class RecordedEstimator:
    """Wrap an estimator and write one line per fit to `path` (JSONL): `record` (suite, task,
    dataset), the table size, the fit time (and the wrapped estimator's `fit_record_`, if any).

    With `latency`, the one-row prediction time (ms, median of `LATENCY_REPEATS`) is measured in
    `after_predict` and written then. The TabArena system calls it with one test row from an
    untimed hook after prediction (`post_predict`). Other attributes (`predict`, `predict_proba`,
    `classes_`, ...) are the wrapped estimator's.
    """

    def __init__(self, estimator, record: dict, path: Path, latency: bool) -> None:
        self.estimator = estimator
        self.record = record
        self.path = path
        self.latency = latency

    def __getattr__(self, name: str):
        # during copy / pickle (no estimator yet), do not look up the wrapped estimator
        if name.startswith("__") or name == "estimator":
            raise AttributeError(name)
        return getattr(self.estimator, name)

    def fit(self, X, y, **fit_kwargs) -> RecordedEstimator:
        start = time.perf_counter()
        self.estimator.fit(X, y, **fit_kwargs)
        self.record_ = {
            **self.record,
            "rows": len(X),
            "features": int(np.shape(X)[1]),
            "fit_seconds": time.perf_counter() - start,
            **getattr(self.estimator, "fit_record_", {}),
        }
        if not self.latency:
            self._write()
        return self

    def after_predict(self, row) -> None:
        if not self.latency:
            return
        predict = (
            self.estimator.predict
            if self.record["task"] == "regression"
            else self.estimator.predict_proba
        )
        times = []
        for _ in range(LATENCY_REPEATS):
            t = time.perf_counter()
            predict(row)
            times.append(time.perf_counter() - t)
        self.record_["latency_ms"] = 1000 * float(np.median(times))
        self._write()

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a") as f:
            f.write(json.dumps(self.record_, default=float) + "\n")


def arena_suite(name: str):
    from tabsil_train.benchmarks.tabular import beyondarena, tabarena

    if name == "tabarena-lite":
        return tabarena.SUITE
    if name == "tabarena-3split":
        return tabarena.SUITE_3SPLIT
    return beyondarena.suite("lite" if name == "beyondarena-lite" else "core2k")


def arena_limits(models: dict, args: argparse.Namespace, caps: dict | None = None) -> dict:
    """Limits of the splits to run per problem type. A task without a checkpoint
    (classification / regression) is left out (imputed). Sizes are TabSil's supported size
    (`tabsil.recipe.SUPPORTED_SIZE`), the class count is what the model takes (`max_classes_of`:
    the output count; column_row_transformer groups label embeddings up to `MAX_CLASSES`). Tasks
    with a columns-per-member cap (`caps`, per task; `--max-features-per-member` if not given)
    have no feature limit."""
    from tabsil_train.benchmarks.tabular.arena import ModelLimits

    limits = {}
    for task, (model, _) in models.items():
        size = dict(SUPPORTED_SIZE)
        if args.max_train_rows is not None:
            size["max_train_rows"] = args.max_train_rows
        if args.max_features is not None:
            size["max_features"] = args.max_features
        elif (caps or {}).get(task, getattr(args, "max_features_per_member", None) or None):
            size["max_features"] = None  # each member picks its features, so no limit
        if task == "classification":
            lim = ModelLimits(**size, max_classes=max_classes_of(model))
            limits.update(binary=lim, multiclass=lim)
        else:
            limits["regression"] = ModelLimits(**size)
    return limits


def arena_fingerprint(args: argparse.Namespace, paths: dict, factories: dict) -> str:
    """The value that decides whether finished splits are reused: only with the same checkpoints
    (path and weights, `weights_digest`) and inference settings (the command line's, and the ones
    the estimators get, `applied_inference`, which include the settings saved with the weights).
    Command-line settings at their default are left out."""
    caps = feature_caps(args, factories)
    return json.dumps(
        {
            "paths": {t: paths[t] for t in factories},
            "weights": {t: weights_digest(f.model, f.borders) for t, f in factories.items()},
            "inference": {t: f.inference for t, f in factories.items()},
            "n_estimators": {t: f.n_estimators for t, f in factories.items()},
            **({"seed": args.seed} if args.seed else {}),
            **(
                {"max_features_per_member": _one_or_by_task(caps)}
                if any(v is not None for v in caps.values())
                else {}
            ),
            **(
                {"subsample_train_rows": args.subsample_train_rows}
                if getattr(args, "subsample_train_rows", None) is not None
                else {}
            ),
            **(
                {"member_kinds": list(resolve_member_kinds(args.member_kinds))}
                if getattr(args, "member_kinds", None)
                else {}
            ),
            **inference_options(args),
        },
        sort_keys=True,
    )


def arena_estimator_factory(args: argparse.Namespace, paths: dict):
    """The `make(problem_type, categorical)` the TabArena / BeyondArena system takes, and the
    per-task factories. Estimators come from the same `make_estimator_factory` as the AMLB
    evaluation (only the problem type and categorical columns differ)."""
    import types

    factories = {
        task: make_estimator_factory(task, args) for task, path in paths.items() if path is not None
    }
    if not factories:
        raise SystemExit(f"{args.suite} needs --classifier-path or --regressor-path.")

    def make(problem_type: str, categorical: list[int]):
        task = "regression" if problem_type == "regression" else "classification"
        return factories[task](types.SimpleNamespace(categorical_indices=categorical))

    return make, factories


def merge_worker_logs(args: argparse.Namespace, out_dir: Path, worker_dirs: list[Path]) -> None:
    """Append the fit records (jsonl) of this run's workers (`worker_dirs`) to this run's file.
    No folder glob: if an earlier run had more workers, their leftover folders would mix in."""
    for worker_dir in worker_dirs:
        part = worker_dir / fits_path(args).name
        if part.exists():
            with open(out_dir / part.name, "a") as f:
                f.write(part.read_text())


def run_arena(args: argparse.Namespace, out_dir: Path, use_mlflow: bool) -> dict:
    """Run TabArena-Lite / BeyondArena and return the scores (the `summary.json` content).
    Estimators come from the same `make_estimator_factory` as the AMLB evaluation (only the
    problem type and categorical columns differ)."""
    from tabsil_train.benchmarks.tabular import arena

    suite = arena_suite(args.suite)
    paths = {"classification": args.classifier_path, "regression": args.regressor_path}
    make, factories = arena_estimator_factory(args, paths)
    print(f"members: { {t: f.n_estimators for t, f in factories.items()} }")

    fingerprint = arena_fingerprint(args, paths, factories)
    models = {task: (f.model, f.borders) for task, f in factories.items()}
    datasets = None if args.datasets is None else args.datasets.split(",")
    limits = arena_limits(models, args, feature_caps(args, factories))
    args.feature_caps_logged = _one_or_by_task(feature_caps(args, factories))
    common = dict(limits=limits, method=args.method_name, fingerprint=fingerprint)
    if args.shard_worker:  # worker: run datasets received on stdin one at a time, save results
        for line in iter(sys.stdin.readline, ""):
            job = json.loads(line)
            print(f"shard worker: {job['dataset']} ({job['splits']} splits)", flush=True)
            # receipt marker: dying without it makes the parent hand the job to another worker
            Path(job["out_dir"]).mkdir(parents=True, exist_ok=True)
            Path(job["started"]).touch()
            arena.run(
                suite,
                make,
                out_dir=job["out_dir"],
                datasets=[job["dataset"]],
                results_path=job["results"],
                **common,
            )
        return {}
    results = None
    if args.shards > 1:
        cache_id = arena._run_id(fingerprint, args.method_name)  # for progress
        estimate = host_state_estimator(factories, feature_caps(args, factories))
        results = run_shards(args, suite, limits, datasets, out_dir, cache_id, estimate)
    summary = arena.run(
        suite, make, out_dir=out_dir / args.suite, datasets=datasets, results=results, **common
    )
    print_arena_summary(summary)
    if use_mlflow:
        log_arena_mlflow(args, summary, out_dir / args.suite, factories)
    check_failed_splits(summary)
    return summary


def check_failed_splits(summary: dict) -> None:
    """Stop (exit code 1) if a split ran but has no result: the method raised and TabArena caught
    it and imputed the split, so the scores are not this method's. Called after the summary and
    the MLflow logging, so the run keeps its outputs. Splits imputed because they exceed the size
    limits or are left out with `--datasets` are not failures."""
    failed = summary.get("failed") or []
    if failed:
        raise SystemExit(
            f"{len(failed)} splits failed and were imputed: {', '.join(failed)}. The scores above"
            " are not valid; the errors are in the log (TabArena's 'Experiment ... failed on',"
            " or shards/shard<i>.log with --shards)."
        )


def shard_worker_command(argv: list[str], worker_dir: Path) -> list[str]:
    """A worker's command arguments: the original ones without the sharding flags and MLflow
    logging, plus its own out-dir and `--shard-worker`. Datasets arrive one at a time on stdin."""
    base = _strip_flags(
        argv,
        {
            "--shards": 1,
            "--datasets": 1,
            "--out-dir": 1,
            "--mlflow": 0,
            "--heartbeat-minutes": 1,
            "--shard-worker": 0,
        },
    )
    return [*base, "--out-dir", str(worker_dir), "--shard-worker"]


def shard_jobs(
    queue: list[tuple[str, int]],
    out_dir: Path,
    suite: str,
    cache_id: str = "",
    host_bytes: dict[str, int] | None = None,
) -> list[dict]:
    """One worker job per queued dataset: dataset, split count, `arena.run` out-dir, results
    file, receipt marker file, and this run's split cache folder (`runs/<cache_id>`, for
    progress). Datasets with a positive value in `host_bytes` carry it (the CPU cell state
    estimate, `host_state_bytes`) as `host_bytes` (the `run_queue` budget).

    The out-dir is keyed by dataset, not worker, so a rerun reuses finished splits (cache)
    whichever worker takes it. Folder names get a hash of the name (so `a/b` and `a_b`, or names
    differing only in case, do not share a folder)."""
    import hashlib

    jobs = []
    for dataset, splits in queue:
        digest = hashlib.sha1(dataset.encode()).hexdigest()[:8]
        safe_name = re.sub(r"[^\w.-]", "_", dataset)
        folder = f"{safe_name}-{digest}"
        dataset_dir = out_dir / "shards" / "datasets" / folder / suite
        jobs.append(
            {
                "dataset": dataset,
                "splits": splits,
                "out_dir": str(dataset_dir),
                "results": str(dataset_dir / "results.pkl"),
                "started": str(dataset_dir / "started"),
                "cache": str(dataset_dir / "runs" / cache_id),
            }
        )
        if (host_bytes or {}).get(dataset):
            jobs[-1]["host_bytes"] = int(host_bytes[dataset])
    return jobs


def _splits_cached(job: dict) -> int:
    # for progress: the cache tabarena writes for each finished split of this run
    # (`runs/<id>/data/<method>/<task>/<repeat>_<fold>/results.pkl`)
    found = sum(1 for _ in Path(job["cache"]).glob("data/*/*/*/results.pkl*"))
    return min(found, job["splits"])


def run_queue(
    jobs: list[dict],
    spawn,
    n_workers: int,
    *,
    heartbeat_s: float = 300.0,
    poll_s: float = 2.0,
    labels: list[str] | None = None,
    host_budget: int | None = None,
) -> tuple[list[str], dict[int, str]]:
    """Hand `jobs` out in order, one at a time, to `n_workers` workers (`spawn(i)` returns a
    Popen with stdin as PIPE). A worker gets its next job once it writes the results file
    (`job["results"]`). When all jobs are done, closing stdin ends the workers. Every
    `heartbeat_s`, print finished splits and per-worker dataset and elapsed time (so the platform
    does not kill a silent job, and so people can follow progress).

    When a worker dies, the remaining workers take the rest. If it died before writing the
    receipt marker (`job["started"]`), its job is handed out again; after the marker it is not
    (so one dataset cannot kill every worker in turn). Returns the datasets without results and
    the failed workers {index: reason}; success means both are empty.

    With `host_budget` (bytes), large tables that keep cell state on the CPU (`job["host_bytes"]`,
    `host_state_bytes`) are handed out only while they fit in the budget together with the large
    tables already running. With no large table running, one is handed out even over budget (it
    runs alone). While a large table waits at the front, later large tables are held until it
    starts, so an expensive table is not pushed to the tail by smaller large ones (expensive
    first); idle workers meanwhile take small tables. Only the queue order changes (results are
    the same per dataset)."""
    labels = labels or [f"worker {i}" for i in range(n_workers)]
    pending = list(jobs)
    total = sum(j["splits"] for j in jobs)
    current: dict[int, tuple[dict, float]] = {}  # worker → (job, time received)
    done: set[str] = set()
    failed: dict[int, str] = {}
    waiting: set[str] = set()  # large tables held back by the budget (announced once)
    start = last_beat = time.monotonic()

    def host_in_use() -> int:
        return sum(job.get("host_bytes", 0) for job, _ in current.values())

    def next_job() -> int | None:
        # queue index of the next job: in order, large tables only within budget (docstring)
        if host_budget is None:
            return 0
        in_use = host_in_use()
        blocked = False  # a large table waits ahead: hold later large tables
        for k, job in enumerate(pending):
            need = job.get("host_bytes", 0)
            if not need:
                return k
            if not blocked and (not in_use or in_use + need <= host_budget):
                return k
            blocked = True
            if job["dataset"] not in waiting:
                waiting.add(job["dataset"])
                print(
                    f"shards: {job['dataset']} waits for host memory (cell state "
                    f"{need / 1e9:.0f} GB, in use {in_use / 1e9:.0f} / budget "
                    f"{host_budget / 1e9:.0f} GB)",
                    flush=True,
                )
        return None

    def assign(i: int, proc, k: int = 0) -> None:
        job = pending.pop(k)
        # so files of an earlier run are not taken for this run's
        for stale in (job["results"], job["started"]):
            Path(stale).unlink(missing_ok=True)
        try:
            proc.stdin.write(json.dumps(job) + "\n")
            proc.stdin.flush()
        except OSError:  # already dead (BrokenPipeError, ...): return the job for another worker
            pending.insert(k, job)
            failed.setdefault(i, "stdin closed")
            return
        current[i] = (job, time.monotonic())
        host = f", cell state {job['host_bytes'] / 1e9:.0f} GB" if job.get("host_bytes") else ""
        print(f"shards: {labels[i]} ← {job['dataset']} ({job['splits']} splits{host})", flush=True)

    def close(proc) -> None:
        if not proc.stdin.closed:
            try:
                proc.stdin.close()
            except OSError:
                pass

    def heartbeat(now: float) -> None:
        cached = {i: _splits_cached(job) for i, (job, _) in current.items()}
        finished = sum(j["splits"] for j in jobs if j["dataset"] in done)
        print(
            f"[heartbeat {_clock(now - start)}] splits {finished + sum(cached.values())}/{total} "
            f"(datasets {len(done)}/{len(jobs)}, queue {len(pending)})",
            flush=True,
        )
        if host_budget is not None and any(j.get("host_bytes") for j in jobs):
            print(
                f"  host cell state {host_in_use() / 1e9:.0f} / budget {host_budget / 1e9:.0f} GB",
                flush=True,
            )
        for i in range(n_workers):
            if i in current:
                job, since = current[i]
                elapsed = _clock(now - since)
                state = f"{job['dataset']} splits {cached[i]}/{job['splits']}, {elapsed}"
            else:
                state = "failed" if i in failed else ("idle" if pending else "done")
            print(f"  {labels[i]}: {state}", flush=True)

    procs = [spawn(i) for i in range(n_workers)]
    try:
        while True:
            for i, proc in enumerate(procs):
                if i in current and Path(current[i][0]["results"]).exists():
                    job, since = current.pop(i)
                    done.add(job["dataset"])
                    print(
                        f"shards: {labels[i]} done {job['dataset']} "
                        f"in {_clock(time.monotonic() - since)} "
                        f"({len(done)}/{len(jobs)} datasets)",
                        flush=True,
                    )
                code = proc.poll()
                if code is not None and (code or i in current):
                    failed.setdefault(i, f"exit {code}")
                    if i in current:
                        job, _ = current.pop(i)
                        if Path(job["started"]).exists():
                            failed[i] += f" while running {job['dataset']}"
                        else:  # died before receiving it: hand to another worker
                            pending.insert(0, job)
                    close(proc)
            # hand remaining jobs to idle workers (alive, no job)
            for i, proc in enumerate(procs):
                if pending and i not in current and i not in failed and proc.poll() is None:
                    k = next_job()
                    if k is None:  # every remaining job waits for budget
                        break
                    assign(i, proc, k)
            if not current and (not pending or all(i in failed for i in range(n_workers))):
                for proc in procs:
                    close(proc)
                if all(p.poll() is not None for p in procs):
                    break
            now = time.monotonic()
            if now - last_beat >= heartbeat_s:
                heartbeat(now)
                last_beat = now
            time.sleep(poll_s)
    finally:
        # if the parent stops midway (exception, Ctrl-C), do not leave workers holding GPUs
        for proc in procs:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
    heartbeat(time.monotonic())
    missing = [j["dataset"] for j in jobs if j["dataset"] not in done]
    return missing, failed


# without `--host-memory-budget-gb`, this fraction of the container memory limit (cgroup) goes
# to CPU cell state; the rest is left for the workers' other memory (tables, per-member
# preprocessed training rows, ...)
HOST_MEMORY_FRACTION = 0.5
CGROUP_MEMORY_FILES = (
    "/sys/fs/cgroup/memory.max",  # cgroup v2
    "/sys/fs/cgroup/memory/memory.limit_in_bytes",  # cgroup v1
)


def cgroup_memory_limit(files: tuple[str, ...] = CGROUP_MEMORY_FILES) -> int | None:
    """This container's memory limit (bytes). None if unreadable or unlimited ("max", or v1's
    huge value)."""
    for name in files:
        try:
            text = Path(name).read_text().strip()
        except OSError:
            continue
        if text.isdigit() and int(text) < 1 << 60:
            return int(text)
        return None  # "max", or v1's "no limit"
    return None


def physical_memory(meminfo: str = "/proc/meminfo") -> int | None:
    """Physical memory (bytes, `MemTotal`). None if unreadable."""
    try:
        for line in Path(meminfo).read_text().splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024  # kB
    except (OSError, ValueError, IndexError):
        pass
    return None


def host_memory_budget(
    args: argparse.Namespace,
    files: tuple[str, ...] = CGROUP_MEMORY_FILES,
    meminfo: str = "/proc/meminfo",
):
    """`run_queue`'s CPU cell state budget (bytes; None: off). `--host-memory-budget-gb` (<= 0:
    off), otherwise `HOST_MEMORY_FRACTION` of the cgroup limit. With no limit ("max": e.g. a
    container limited by its parent cgroup) or an unreadable one, the same fraction of physical
    memory; if that is unknown too, off."""
    given = getattr(args, "host_memory_budget_gb", None)
    if given is not None:
        return None if given <= 0 else int(given * 1e9)
    limit = cgroup_memory_limit(files) or physical_memory(meminfo)
    return None if limit is None else int(limit * HOST_MEMORY_FRACTION)


def host_state_estimator(factories: dict, caps: dict):
    """`(problem_type, rows, features, has categorical columns, train rows) -> bytes of cell state
    one member forward keeps on the CPU`, an upper estimate. The model's
    `inference_host_state_bytes` gets the member's input column count: the original columns (per
    member cap `caps`), up to the one-hot cap (`max_one_hot_width`) when there are categorical
    columns, plus the sample identifier. With the train rows it counts only the part of the state
    the offload keeps (the row chunks with train rows); without them, the whole state. None when no
    factory has a model (the queue then does not budget host memory)."""
    models = [getattr(f, "model", None) for f in factories.values()]
    if all(m is None for m in models):
        return None

    def task_of(problem_type: str) -> str:
        return "regression" if problem_type == "regression" else "classification"

    def estimate(
        problem_type: str,
        rows: int,
        n_features: int,
        categorical: bool = True,
        train_rows: int | None = None,
    ) -> int:
        factory = factories.get(task_of(problem_type))
        fn = getattr(getattr(factory, "model", None), "inference_host_state_bytes", None)
        if fn is None:
            return 0
        cap = caps.get(task_of(problem_type))
        columns = n_features if cap is None else min(n_features, cap)
        width = getattr(factory, "max_one_hot_width", None)
        if categorical and width:
            # expanded columns never exceed the larger of the original count and the cap
            # (`one_hot_sizes`)
            columns = max(columns, width)
        if categorical and getattr(factory, "count_columns", False):
            # plus at most one count column per original column (`--categories counts`)
            columns += n_features if cap is None else min(n_features, cap)
        if train_rows is None:
            return int(fn(rows, columns + 1))
        return int(fn(rows, columns + 1, train_rows=min(train_rows, rows)))

    return estimate


def host_state_bytes(grid, rows: dict[str, int], estimate) -> dict[str, int]:
    """CPU cell state estimate per dataset with splits to run (`estimate`; zeros left out). Rows
    are the largest split's training + test rows (`arena.split_rows`), or training rows x 1.5 if
    unknown (an upper bound up to a 2:1 train:test split). The largest split's training rows (the
    grid's `max_train_rows`, before any subsampling, so an upper bound) give the part of the state
    the offload keeps."""
    todo = grid[grid["skip_reason"].isna()]
    out = {}
    for dataset, part in todo.groupby("dataset"):
        n_rows = rows.get(dataset) or int(part["max_train_rows"].max() * 1.5)
        problem_type = part["problem_type"].iloc[0]
        # if unknown whether categorical columns exist (None, TabArena), assume they do
        categorical = part["has_categorical"].iloc[0] if "has_categorical" in part else None
        categorical = True if categorical is None or pd.isna(categorical) else bool(categorical)
        train_rows = int(part["max_train_rows"].max())
        need = estimate(
            problem_type, int(n_rows), int(part["n_features"].max()), categorical, train_rows
        )
        if need:
            out[dataset] = need
    return out


def run_shards(
    args, suite, limits, datasets, out_dir: Path, cache_id: str = "", host_estimate=None
) -> list:
    """Start one worker process per GPU and hand out datasets one at a time from a queue, most
    expensive first (`arena.dataset_queue`, `run_queue`). A worker takes the next dataset as soon
    as it finishes one, so no GPU idles early. Returns the workers' collected split results.
    Worker output goes to `<out-dir>/shards/shard<i>.log`; fit records (jsonl) are appended to
    this run's files at the end. Stops if any worker fails or any dataset has no results (so that
    dataset is not silently imputed).

    Large tables that `host_estimate(problem_type, rows, features)` (`host_state_estimator`) says
    keep cell state on the CPU are handed out only while the concurrent total fits in
    `--host-memory-budget-gb` (`host_memory_budget`)."""
    import pickle
    import subprocess

    from tabsil_train.benchmarks.tabular import arena

    n_gpus = torch.cuda.device_count()
    if n_gpus and args.shards > n_gpus:
        raise SystemExit(f"--shards {args.shards} exceeds the GPU count {n_gpus}.")
    # if the parent already picked GPUs (CUDA_VISIBLE_DEVICES), worker i uses the i-th of them
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    devices = visible.split(",") if visible else [str(i) for i in range(max(n_gpus, args.shards))]
    context = arena.make_context(suite)
    grid = arena.task_grid(context, suite, limits)
    if datasets is not None:
        # stop before starting workers (`arena.run` stops for the same reason when scoring)
        unknown = sorted(set(datasets) - set(grid["dataset"]))
        if unknown:
            raise SystemExit(f"datasets not in {suite.name}: {unknown}")
        grid = grid[grid["dataset"].isin(datasets)]
    subsample = getattr(args, "subsample_train_rows", None)
    queue = arena.dataset_queue(grid, subsample)
    host_bytes = {}
    if host_estimate is not None and queue:
        host_bytes = host_state_bytes(grid, arena.split_rows(context, subsample), host_estimate)
    jobs = shard_jobs(queue, out_dir, args.suite, cache_id, host_bytes)
    if not jobs:  # no splits to run: only score (all imputed)
        return []
    budget = host_memory_budget(args) if host_bytes else None
    if host_bytes:
        big = sorted(host_bytes.items(), key=lambda t: (-t[1], t[0]))
        limit = "off" if budget is None else f"{budget / 1e9:.0f} GB"
        print(
            f"shards: host cell state budget {limit}; "
            + ", ".join(f"{d} {b / 1e9:.0f} GB" for d, b in big),
            flush=True,
        )
    n_workers = min(args.shards, len(jobs))
    threads = max(1, (os.cpu_count() or 1) // n_workers)
    # point at the same folder so workers reuse checkpoints (runs:/) the parent downloaded
    checkpoints = (out_dir / "checkpoints").resolve()
    worker_dirs = [out_dir / "shards" / f"shard{i}" for i in range(n_workers)]
    logs = []

    def spawn(i: int):
        worker_dir = worker_dirs[i]
        worker_dir.mkdir(parents=True, exist_ok=True)
        link = worker_dir / "checkpoints"
        if checkpoints.is_dir() and not link.exists():
            link.symlink_to(checkpoints, target_is_directory=True)
        env = {k: v for k, v in os.environ.items() if k != "MLFLOW_RUN_ID"}
        env.update(CUDA_VISIBLE_DEVICES=devices[i], OMP_NUM_THREADS=str(threads))
        log = open(out_dir / "shards" / f"shard{i}.log", "w")  # noqa: SIM115 (open until the end)
        logs.append(log)
        cmd = shard_worker_command(args.argv, worker_dir)
        return subprocess.Popen(
            [sys.executable, __file__, *cmd],
            env=env,
            stdin=subprocess.PIPE,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )

    total = sum(j["splits"] for j in jobs)
    print(f"shards: {len(jobs)} datasets ({total} splits) → {n_workers} workers", flush=True)
    labels = [f"shard {i} (GPU {devices[i]})" for i in range(n_workers)]
    try:
        missing, failed = run_queue(
            jobs,
            spawn,
            n_workers,
            heartbeat_s=args.heartbeat_minutes * 60,
            labels=labels,
            host_budget=budget,
        )
    finally:
        for log in logs:
            log.close()
    if failed or missing:
        tails = {
            i: (out_dir / "shards" / f"shard{i}.log").read_text().splitlines()[-20:] for i in failed
        }
        raise SystemExit(
            f"shards {sorted(failed)} failed ({failed}), datasets without results {missing}: "
            f"{tails}"
        )
    results = []
    # collect results by dataset name (the same order whichever worker took each)
    for job in sorted(jobs, key=lambda j: j["dataset"]):
        with open(job["results"], "rb") as f:
            results.extend(pickle.load(f))
    merge_worker_logs(args, out_dir, worker_dirs)
    print(f"shards: {len(results)} split results from {n_workers} workers", flush=True)
    return results


def per_1k_text(t: dict, suffix: str = "") -> str:
    """TabArena leaderboard time (median s per 1K rows, `arena.run_times`) as "train / infer"."""
    nan = float("nan")
    train = t.get(f"time_train_s_per_1K_median{suffix}", nan)
    infer = t.get(f"time_infer_s_per_1K_median{suffix}", nan)

    def fmt(v: float) -> str:  # 3 significant digits for small times, no exponent for large
        return f"{v:.0f}" if abs(v) >= 1000 else f"{v:.3g}"

    return f"{fmt(train)}/{fmt(infer)}"


def print_arena_summary(summary: dict) -> None:
    def line(label: str, s: dict) -> str:
        if s.get("elo") is None:
            return f"{label:24s} no splits with results (imputed {s['imputed_pct']:.1f}%)"
        xgb = s.get("winrate", {}).get("XGBoost (default)", {})
        t = s.get("time", {})  # split means; per 1K rows as on the leaderboard
        return (
            f"{label:24s} elo {s['elo']:.0f} (+{s['elo_ci_plus']:.0f}/-{s['elo_ci_minus']:.0f}) "
            f"score {s['score']:.3f} improvability {s['improvability']:.2f}% "
            f"imputed {s['imputed_pct_leaderboard']:.1f}% "
            f"win vs XGB(default) {xgb.get('all', float('nan')):.3f} "
            f"position {s['position']}/{s['n_methods']} "
            f"time train {t.get('time_train_s_mean', float('nan')):.1f}s "
            f"infer {t.get('time_infer_s_mean', float('nan')):.1f}s "
            # run: splits we ran; leaderboard: all splits, imputed ones with the fill method's time
            f"s/1K train/infer run {per_1k_text(t)} leaderboard {per_1k_text(t, '_leaderboard')}"
        )

    def baselines(label: str, s: dict) -> str | None:
        # baselines' leaderboard time on the same splits (run: only the splits we ran)
        parts = [
            f"{b} run {per_1k_text(t)} leaderboard {per_1k_text(t, '_leaderboard')}"
            for b, t in s.get("time_baselines", {}).items()
        ]
        return f"{label:24s} s/1K train/infer: " + ", ".join(parts) if parts else None

    print(line(summary["suite"], summary))
    if (text := baselines(f"  {summary['suite']} vs", summary)) is not None:
        print(text)
    for label, s in summary["breakdown"].items():
        print(line(f"  {label}", s))


def applied_inference_params(factories: dict) -> dict:
    """MLflow params of the inference settings each task ran with (`applied_inference`, JSON):
    `inference_classification`, `inference_regression`."""
    return {f"inference_{t}": json.dumps(f.inference, sort_keys=True) for t, f in factories.items()}


def log_arena_mlflow(args, summary: dict, out_dir: Path, factories: dict) -> None:
    prefix = summary["suite"].replace("-", "_")
    mlflow.log_params(
        {
            "suite": args.suite,
            "method_name": args.method_name,
            "run_id_cache": summary["run_id"],
            "seed": args.seed,
            "max_features_per_member": getattr(
                args, "feature_caps_logged", args.max_features_per_member
            ),
            **{f"n_estimators_{t}": f.n_estimators for t, f in factories.items()},
            "subsample_train_rows": args.subsample_train_rows,
            "member_kinds": getattr(args, "member_kinds", None) or "default",
            **inference_options(args),
            **applied_inference_params(factories),
        }
    )
    paths = {"classifier_path": args.classifier_path, "regressor_path": args.regressor_path}
    mlflow.log_params({k: v for k, v in paths.items() if v is not None})

    def metrics(p: str, s: dict) -> dict:
        keys = ("elo", "score", "improvability", "rank", "imputed_pct_leaderboard")
        out = {f"{p}_{k}": s[k] for k in keys if s.get(k) is not None}
        out[f"{p}_imputed_pct_splits"] = s["imputed_pct"]
        out.update({f"{p}_{k}": v for k, v in s.get("time", {}).items()})
        for baseline, w in s.get("winrate", {}).items():
            key = baseline.split(" (")[0].lower().replace("-", "").replace(".", "")
            sub = baseline.split("(")[-1].rstrip(")").replace(" + ", "_").replace(" ", "_")
            if w["all"] == w["all"]:  # skip NaN
                out[f"{p}_winrate_{key}_{sub}"] = w["all"]
        return out

    mlflow.log_metrics(metrics(prefix, summary))
    for label, s in summary["breakdown"].items():
        mlflow.log_metrics(metrics(f"{prefix}_{label.replace('-', '_')}", s))
    for path in sorted(out_dir.glob("*.csv")) + [out_dir / "summary.json"]:
        mlflow.log_artifact(str(path), f"eval/{summary['suite']}")
    log_fits_artifact(args)


def log_fits_artifact(args: argparse.Namespace) -> None:
    path = fits_path(args)
    if path.exists():
        mlflow.log_artifact(str(path), "eval")


def summarize(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    keys = ["suite", "task_id", "dataset"]
    if "problem_type" in df:
        keys.append("problem_type")
    values = [c for c in ("score", "metric_error") if c in df]
    per_dataset = df.groupby(keys)[values].mean().reset_index()
    paper = {**bm.paper_scores("classification"), **bm.paper_scores("regression")}
    per_dataset["paper"] = per_dataset["task_id"].map(paper)
    per_dataset["diff"] = per_dataset["score"] - per_dataset["paper"]
    return per_dataset


def mlflow_run(use_mlflow: bool):
    """One evaluation run (`tabsil_train.utils.mlflow_eval_run`; the local experiment is
    `TABSIL_EXPERIMENT`, default `tabsil`)."""
    return mlflow_eval_run(use_mlflow, __file__, "TABSIL_", "tabsil")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.argv = list(sys.argv[1:] if argv is None else argv)  # for `--shards` worker commands
    args.script = __file__  # the script the ADBench / impute `--shards` workers run
    if args.shards < 1:
        raise SystemExit("--shards must be >= 1.")
    if args.shards > 1 and args.suite not in (*ARENA_SUITES, ADBENCH_SUITE, IMPUTE_SUITE):
        raise SystemExit(
            "--shards is for the TabArena / BeyondArena / ADBench / impute suites only."
        )
    if args.uncertainty and args.suite in (
        *ARENA_SUITES,
        FEV_SUITE,
        ADBENCH_SUITE,
        CAUSAL_SUITE,
        IMPUTE_SUITE,
    ):
        # the arenas report only the official package's metrics; stop rather than silently ignore
        raise SystemExit(
            "--uncertainty is for AMLB suites only (classification, regression, all, dev)."
        )
    if args.latency and args.suite not in ARENA_SUITES:
        raise SystemExit("--latency is for the TabArena / BeyondArena suites only.")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fits_path(args).unlink(missing_ok=True)  # keep only this run's fits
    use_mlflow = args.mlflow or "MLFLOW_RUN_ID" in os.environ
    if args.suite in ARENA_SUITES:
        with mlflow_run(use_mlflow):
            run_arena(args, out_dir, use_mlflow)
        return
    if args.suite == FEV_SUITE:
        with mlflow_run(use_mlflow):
            run_fev(args, out_dir, use_mlflow)
        return
    if args.suite == ADBENCH_SUITE:
        with mlflow_run(use_mlflow):
            run_adbench(args, out_dir, use_mlflow)
        return
    if args.suite == CAUSAL_SUITE:
        with mlflow_run(use_mlflow):
            run_causal(args, out_dir, use_mlflow)
        return
    if args.suite == IMPUTE_SUITE:
        with mlflow_run(use_mlflow):
            run_impute(args, out_dir, use_mlflow)
        return
    tasks = selected_tasks(args)  # exit on bad task ids before creating a run
    with mlflow_run(use_mlflow):
        run_amlb(args, out_dir, tasks, use_mlflow)


def run_amlb(args: argparse.Namespace, out_dir: Path, tasks: list, use_mlflow: bool) -> None:
    """Evaluate the AMLB benchmark (29 AMLB classification, 28 AMLB/CTR23 regression tasks) and
    compare with the TabPFN v2 paper's scores."""
    print(f"tasks: {len(tasks)}, folds: {args.folds}, device: {args.device}", flush=True)
    # load models first: the model sets the default columns-per-member cap (`model_feature_cap`)
    factories = {suite: make_estimator_factory(suite, args) for suite in {s for s, _ in tasks}}

    if use_mlflow:
        mlflow.log_params(
            {
                "suite": args.suite,
                "folds": args.folds,
                "seed": args.seed,
                "max_features_per_member": _one_or_by_task(feature_caps(args, factories)),
                "subsample_train_rows": args.subsample_train_rows,
                **{f"n_estimators_{suite}": f.n_estimators for suite, f in factories.items()},
                **applied_inference_params(factories),
            }
        )
        paths = {"classifier_path": args.classifier_path, "regressor_path": args.regressor_path}
        mlflow.log_params({k: v for k, v in paths.items() if v is not None})

    rows = []
    for i, (suite, task_id) in enumerate(tasks, start=1):
        start = time.time()
        task = bm.load_openml_task(task_id)
        task_rows = bm.evaluate_task(
            task, factories[suite], n_folds=args.folds, uncertainty=args.uncertainty
        )
        rows.extend({**r, "suite": suite} for r in task_rows)
        write_amlb_fits(args, suite, task, task_rows)
        score = float(np.nanmean([r["score"] for r in task_rows]))
        paper = bm.paper_scores(suite)[task_id]
        t = bm.time_medians(task_rows)
        print(
            f"[{i}/{len(tasks)}] {suite:14s} {task.name:40s} {task.X.shape} "
            f"score {score:.4f} paper {paper:.4f} s/1K train/infer {per_1k_text(t)} "
            f"({time.time() - start:.0f}s)",
            flush=True,
        )
        pd.DataFrame(rows).to_csv(out_dir / "eval_rows.csv", index=False)

    per_dataset = summarize(rows)
    per_dataset.to_csv(out_dir / "eval_per_dataset.csv", index=False)
    print(per_dataset.round(4).to_string(index=False))
    for suite, group in per_dataset.groupby("suite"):
        metric = "roc_auc" if suite == "classification" else "neg_rmse"
        times = bm.time_medians([r for r in rows if r["suite"] == suite])
        print(
            f"{suite}: datasets {len(group)}, mean score {group['score'].mean():.4f}, "
            f"paper {group['paper'].mean():.4f}, mean |diff| {group['diff'].abs().mean():.4f}, "
            f"s/1K train/infer {per_1k_text(times)}"
        )
        if use_mlflow:
            mlflow.log_metric(f"{suite}_mean_{metric}", group["score"].mean())
            mlflow.log_metric(f"{suite}_paper_mean_{metric}", group["paper"].mean())
            mlflow.log_metric(f"{suite}_mean_abs_diff", group["diff"].abs().mean())
            mlflow.log_metrics({f"{suite}_{k}": v for k, v in times.items()})
    artifacts = ["eval_rows.csv", "eval_per_dataset.csv"]
    if args.uncertainty:
        report_uncertainty(rows, out_dir, use_mlflow)
        artifacts.append("eval_uncertainty_per_dataset.csv")
    if use_mlflow:
        for name in artifacts:
            mlflow.log_artifact(str(out_dir / name), "eval")
        log_fits_artifact(args)


def write_amlb_fits(args: argparse.Namespace, suite: str, task, task_rows: list[dict]) -> None:
    """Append one line per fold to `fits_<suite>.jsonl`, in the arena fit record's format
    (`RecordedEstimator`: suite, task, dataset, rows fit saw, features, fit_seconds) plus the task
    id, fold, the fold's train / test rows, prediction time and both times per 1K fold rows
    (`amlb.per_1k`)."""
    path = fits_path(args)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for r in task_rows:
            record = {
                "suite": args.suite,
                "task": suite,
                "dataset": task.name,
                "task_id": task.task_id,
                "fold": r["fold"],
                "rows": r["fit_rows"],  # as the arena record: rows fit saw
                "features": int(task.X.shape[1]),
                "fit_seconds": r["fit_seconds"],
                "train_rows": r["train_rows"],
                "test_rows": r["test_rows"],
                "predict_seconds": r["predict_seconds"],
                **bm.per_1k(r),
            }
            f.write(json.dumps(record, default=float) + "\n")


def summarize_uncertainty(rows: list[dict]) -> pd.DataFrame:
    """Per-dataset fold means of the uncertainty metrics
    (`tabsil_train.benchmarks.tabular.uncertainty`)."""
    return unc.dataset_means(rows, exclude=("score", "metric_error", "problem_type", *bm.TIME_KEYS))


def report_uncertainty(rows: list[dict], out_dir: Path, use_mlflow: bool) -> None:
    """Save the per-dataset table and print / log the dataset means per suite
    (`<suite>_mean_<metric>`). Regression metrics with units (CRPS, widths, interval scores) are
    averaged only as `_scaled`, divided by the training target standard deviation."""
    per_dataset = summarize_uncertainty(rows)
    per_dataset.to_csv(out_dir / "eval_uncertainty_per_dataset.csv", index=False)
    for suite, group in per_dataset.groupby("suite"):
        means = unc.suite_means(group)
        print(f"{suite} uncertainty (datasets {len(group)}):")
        print(pd.Series(means).round(4).to_string())
        if use_mlflow:
            mlflow.log_metrics({f"{suite}_mean_{k}": v for k, v in means.items()})


if __name__ == "__main__":
    main()
