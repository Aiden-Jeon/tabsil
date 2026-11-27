"""Train TabSil (`tabsil_train.recipe`) on the graph SCM prior (or `--prior-type`) and log the run
to MLflow.

The same script runs on AI Runtime (`databricks air run -f training/scripts/train.yaml`) and
locally.
- Multiple GPUs: `torchrun --standalone --nproc_per_node=8 training/scripts/train.py`
- Multiple nodes: on each node `torchrun --nnodes=<node count> --node_rank=<rank>
  --master_addr=<address> --master_port=<port> --nproc_per_node=8 training/scripts/train.py`. The
  batch stays the same and the per-GPU micro batch shrinks automatically so it never exceeds
  batch / GPU count. To keep the datasets per GPU when adding GPUs, raise the batch with
  `TABSIL_BATCH_SIZE` (resuming keeps the original run's batch).
- Local: `uv run --package tabsil-train python training/scripts/train.py --config-name smoke`

The model is the TabSil structure (`tabsil.recipe.STRUCTURE`) trained as one model for
classification and regression (alternating per step). `--config-name stage1|stage2|stage3` is the
TabSil curriculum (`tabsil_train.recipe.curriculum`); `smoke` (local CPU, reduced size) and `medium`
(stage 1 with 20k steps) are `TrainConfig.smoke()` and `TrainConfig.medium()`. For size
experiments, `--emsize` and `--n-cls` (`TABSIL_EMSIZE`, `TABSIL_N_CLS`) change the cell width and
the CLS token count (the row ICL width is their product); unset, the preset's values are kept.
When resuming or taking weights from a run, that run must have the same structure: a run of another
size needs the same `--emsize` and `--n-cls` again.
The scale is set with arguments or environment variables (`TABSIL_CONFIG_NAME`, `TABSIL_STEPS`,
`TABSIL_BATCH_SIZE`, `TABSIL_MICRO_BATCH`, `TABSIL_NUM_WORKERS`, `TABSIL_PREFETCH_FACTOR`,
`TABSIL_LR`, `TABSIL_CHECKPOINT_EVERY`, `TABSIL_VAL_EVERY`, `TABSIL_LOG_EVERY`). The training
speed options (`TrainConfig` comments) are turned on with `--compile`, `--tf32`,
`--weight-average K` (or `TABSIL_COMPILE=1` etc., `TABSIL_WEIGHT_AVERAGE=K`). The optimizer is
Muon on every
parameter; its weight decay is `TABSIL_MUON_WEIGHT_DECAY` and its momentum `TABSIL_MUON_MOMENTUM`
(resuming keeps the original run's values). `--optimizer muon_adamw` (`TABSIL_OPTIMIZER`) puts
the target encoders, the output and the parameters that are not Linear weights on AdamW instead
(`tabsil_train.training.muon`); a resumed run keeps its optimizer. `--muon-distributed node|all` (or
`TABSIL_MUON_DISTRIBUTED`) splits Muon's Newton–Schulz iteration over the GPUs of a node or over
all GPUs (same results and checkpoint format; resuming keeps the original run's value, `off`
turns it off).
The table size and prior settings (`TrainConfig` comments) can be changed with
`TABSIL_MIN_ROWS`·`TABSIL_MAX_ROWS`·`TABSIL_MIN_TRAIN_FRACTION`·
`TABSIL_MAX_TRAIN_FRACTION`, `TABSIL_BINARY_PROB`, `TABSIL_FILTER_UNPREDICTABLE=1`
(unset values keep the preset). A resumed run keeps the tables it was trained on: these six
settings and the four that only the preset sets (`max_features`, `log_uniform_rows`, `max_cells`,
`missing_dataset_prob`) are taken from the original run's params (`DATA_OPTIONS`). A different
value given on resume is an error, and so is resuming a run that did not record them. The two
settings that fit those tables on a GPU are the run's as well (`MEMORY_OPTIONS`): its activation
checkpointing threshold, and its micro batch unless `--micro-batch` is given. A run is resumed
with the `--config-name` it was started with, because the preset also sets the gradient clipping
and the checkpoint intervals: another one is an error. A run that takes only the weights
(`--init-run-id`) uses the preset's or the given values, under any preset.
Training recipe options of the v2 experiments (`TrainConfig` comments; off by default, and a
resumed run keeps the original run's values): `--target-train-rows` (`TABSIL_TARGET_TRAIN_ROWS=1`)
standardizes the regression target again with the train rows, as inference does.
`--prior-type` (`TABSIL_PRIOR_TYPE`) trains on another registered prior (`tabsil_train.priors.
PRIOR_TYPES`, e.g. `graph_scm_hicard_missing`); unset, the preset's (`graph_scm`). A resumed run
keeps the prior it was trained on (a different `--prior-type` is an error); a run that takes only
the weights (`--init-run-id`) uses the preset's or the given one.
`--reference-val-tables N` (`TABSIL_REFERENCE_VAL_TABLES`) adds a validation set of N tables per
task that depends neither on the run's seed nor on its prior (`tabsil_train.training.reference`,
metrics `ref_*`), so that runs of different seeds and priors can be compared on it; a resumed run
keeps the original run's value.
The training diagnostics (`tabsil_train.training.diagnostics`) always log per-group weight and
gradient norms; `--grad-cosine` (`TABSIL_GRAD_COSINE=1`) adds the per-group cosine between the two
tasks' gradients, and `--rank-timing` (`TABSIL_RANK_TIMING=1`) adds per-GPU forward/backward time
and table size. They do not change the training computation and are not inherited on resume (set
them per chunk).
Python's garbage collector: at every log step each GPU's process collects once and freezes its
live objects (`gc.freeze()`), so that the collector's full passes stop pausing one GPU while the
others wait for it (`TrainConfig.gc_freeze`). It is on by default and does not change the training
computation; `--no-gc-freeze` (`TABSIL_GC_FREEZE=0`) turns it off, per chunk like the diagnostics.
If the train loss jumps and does not recover, training rewinds to a healthy state kept in memory
and continues on different data (collapse threshold `TABSIL_COLLAPSE_THRESHOLD`, default 0.5,
`off` disables it). Each rewind multiplies the base lr by `TABSIL_REWIND_LR_FACTOR` (0.5 for
TabSil).
To train in chunks, give `TABSIL_STOP_STEP` (the step to stop at) or `TABSIL_DEADLINE` (the time
to stop, ISO 8601, e.g. 2026-10-28T08:45+09:00). The lr schedule continues over the full `steps`.
On resume, a `TABSIL_LR` different from the checkpoint's rescales the remaining schedule to that
lr; without it the checkpoint's lr is kept.
`--resume-run-id` (or `TABSIL_RESUME_RUN_ID`) resumes from that run's `checkpoints/latest.pt`.
With `--resume-step` (or `TABSIL_RESUME_STEP`) it resumes from the
`checkpoints/step_XXXXXXX.pt` saved at that step (one is kept every `TABSIL_KEEP_EVERY` steps).
`--init-run-id` (or `TABSIL_INIT_RUN_ID`, with the saved step in `TABSIL_INIT_STEP`) takes only
the weights from that run's checkpoint and trains from step 0. stage2·stage3 need it (the previous
stage's run). The curriculum warmup is 1% of the step count given with `--steps`
(`TABSIL_STEPS`).
`--init-hub` (or `TABSIL_INIT_HUB`) takes the weights and regression borders from a Hub repo or a
local `save_pretrained` folder instead (`aiden-jeon/tabsil@stage1-150k`: the v1 stage-1 weights,
whose training run was written by minipfn and no longer loads here).
The executed code and command are saved in the run's `code/` and tags (`command`,
`env.TABSIL_*`).

The run name is `tabsil-v1-<config name>` (unless the platform passes a run, `MLFLOW_RUN_ID`).

Hash seed: the graph SCM prior (tabicl) iterates a set of strings when it builds a table, so the
generated tables depend on Python's hash seed. Training runs with `PYTHONHASHSEED=0`: when the
variable is not set, the script restarts itself with it (an explicit value is kept), and the prior
worker processes inherit it. `training/scripts/train.yaml` sets it as well.
"""

from __future__ import annotations

import argparse
import os
import shlex
import sys
from dataclasses import replace
from datetime import datetime
from functools import partial
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import torch  # noqa: E402

from tabsil_train import recipe  # noqa: E402
from tabsil_train.hub import run_architecture  # noqa: E402
from tabsil_train.priors import DEFAULT_PRIOR_TYPE, canonical_prior_type  # noqa: E402
from tabsil_train.training.config import (  # noqa: E402
    SPEED_OPTIONS,
    TrainConfig,
    normalize_phases,
)
from tabsil_train.training.loop import (  # noqa: E402
    check_checkpoint,
    check_source_checkpoint,
    train,
)
from tabsil_train.utils import log_run_source  # noqa: E402

# Generic sizes (smoke, medium) come from TrainConfig (a TabSil config with reduced size or steps).
# The curriculum stages come from the recipe
PRESETS = {
    "smoke": TrainConfig.smoke,
    "medium": TrainConfig.medium,
    **{f"stage{n}": partial(recipe.curriculum, n) for n in recipe.CURRICULUM},
}
CONFIG_NAMES = tuple(PRESETS)
CURRICULUM_NAMES = {f"stage{n}": n for n in recipe.CURRICULUM}
# The preset when neither --config-name nor TABSIL_CONFIG_NAME is given
DEFAULT_CONFIG_NAME = "smoke"
# Model size settings that can be changed (TrainConfig fields of the model structure). Unset values
# keep the preset's; a run that resumes or takes weights from another run is checked against them
STRUCTURE_ARGS = ("emsize", "n_cls")


TRUE_VALUES, FALSE_VALUES = ("1", "true", "yes", "on"), ("", "0", "false", "no", "off")


def _env_flag(name: str) -> bool:
    """An on/off environment variable (case-insensitive: 1/true/yes/on, 0/false/no/off, empty is
    off). Any other value stops the script instead of silently turning the option off."""
    value = os.environ.get(name, "").strip().lower()
    if value in TRUE_VALUES:
        return True
    if value in FALSE_VALUES:
        return False
    raise SystemExit(
        f"{name}={os.environ[name]!r} is not an on/off value ({TRUE_VALUES} or {FALSE_VALUES})"
    )


def _env_switch(name: str) -> bool | None:
    """An on/off environment variable (`_env_flag`) of an option that is on by default: None when
    it is unset or empty, so that the preset's value is kept."""
    return _env_flag(name) if os.environ.get(name, "").strip() else None


def _env(name: str, cast=str):
    value = os.environ.get(name)
    return cast(value) if value else None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # A run is resumed with the config name it was started with (`check_resume_config_name`)
    parser.add_argument(
        "--config-name",
        choices=CONFIG_NAMES,
        default=os.environ.get("TABSIL_CONFIG_NAME", DEFAULT_CONFIG_NAME),
    )
    parser.add_argument("--steps", type=int, default=_env("TABSIL_STEPS", int))
    parser.add_argument("--batch-size", type=int, default=_env("TABSIL_BATCH_SIZE", int))
    # A resumed run keeps the original run's micro batch unless one is given
    # (`resume_memory_options`)
    parser.add_argument("--micro-batch", type=int, default=_env("TABSIL_MICRO_BATCH", int))
    parser.add_argument("--num-workers", type=int, default=_env("TABSIL_NUM_WORKERS", int))
    # micro batches requested ahead per prior worker (TrainConfig comment); the data is the same
    parser.add_argument("--prefetch-factor", type=int, default=_env("TABSIL_PREFETCH_FACTOR", int))
    parser.add_argument("--lr", type=float, default=_env("TABSIL_LR", float))
    parser.add_argument("--stop-step", type=int, default=_env("TABSIL_STOP_STEP", int))
    parser.add_argument("--deadline", default=os.environ.get("TABSIL_DEADLINE"))
    parser.add_argument(
        "--checkpoint-every", type=int, default=_env("TABSIL_CHECKPOINT_EVERY", int)
    )
    parser.add_argument("--keep-every", type=int, default=_env("TABSIL_KEEP_EVERY", int))
    parser.add_argument("--val-every", type=int, default=_env("TABSIL_VAL_EVERY", int))
    parser.add_argument("--log-every", type=int, default=_env("TABSIL_LOG_EVERY", int))
    # Training speed options (TrainConfig comments). They can only be turned on; on resume, options
    # the original run turned on stay on (`source_speed_options`)
    for flag in ("compile", "tf32"):
        env = "TABSIL_" + flag.upper().replace("-", "_")
        parser.add_argument(f"--{flag}", action="store_true", default=_env_flag(env))
    # Training diagnostics (TrainConfig comments). They do not change the training computation, so
    # they are not inherited on resume (set them per chunk)
    for flag in ("grad-cosine", "rank-timing"):
        env = "TABSIL_" + flag.upper().replace("-", "_")
        parser.add_argument(f"--{flag}", action="store_true", default=_env_flag(env))
    # The collector freeze (TrainConfig comment) is on by default, so it has an off switch. It does
    # not change the training computation either and is set per chunk
    parser.add_argument(
        "--gc-freeze",
        action=argparse.BooleanOptionalAction,
        default=_env_switch("TABSIL_GC_FREEZE"),
    )
    parser.add_argument("--weight-average", type=int, default=_env("TABSIL_WEIGHT_AVERAGE", int))
    parser.add_argument(
        "--weight-average-every", type=int, default=_env("TABSIL_WEIGHT_AVERAGE_EVERY", int)
    )
    parser.add_argument(
        "--muon-weight-decay", type=float, default=_env("TABSIL_MUON_WEIGHT_DECAY", float)
    )
    parser.add_argument("--muon-momentum", type=float, default=_env("TABSIL_MUON_MOMENTUM", float))
    parser.add_argument(
        "--optimizer",
        choices=["muon", "muon_adamw"],
        default=os.environ.get("TABSIL_OPTIMIZER") or None,
    )
    parser.add_argument(
        "--muon-distributed",
        choices=["off", "node", "all"],
        default=os.environ.get("TABSIL_MUON_DISTRIBUTED") or None,
    )
    parser.add_argument(
        "--collapse-threshold", default=os.environ.get("TABSIL_COLLAPSE_THRESHOLD") or None
    )
    parser.add_argument(
        "--rewind-lr-factor", type=float, default=_env("TABSIL_REWIND_LR_FACTOR", float)
    )
    # TabICLv2 table size and prior settings (TrainConfig comments). Unset values keep the preset;
    # a resumed run keeps the original run's (`resume_data_options`)
    for name in ("min-rows", "max-rows"):
        env = "TABSIL_" + name.upper().replace("-", "_")
        parser.add_argument(f"--{name}", type=int, default=_env(env, int))
    for name in ("min-train-fraction", "max-train-fraction", "binary-prob"):
        env = "TABSIL_" + name.upper().replace("-", "_")
        parser.add_argument(f"--{name}", type=float, default=_env(env, float))
    # Model size (STRUCTURE_ARGS). On resume or with --init-run-id, pass the source run's values
    for name in STRUCTURE_ARGS:
        parser.add_argument(
            "--" + name.replace("_", "-"), type=int, default=_env("TABSIL_" + name.upper(), int)
        )
    parser.add_argument("--prior-type", default=os.environ.get("TABSIL_PRIOR_TYPE") or None)
    # The reference validation set (`tabsil_train.training.reference`). A resumed run keeps the
    # original run's value unless one is given
    parser.add_argument(
        "--reference-val-tables", type=int, default=_env("TABSIL_REFERENCE_VAL_TABLES", int)
    )
    parser.add_argument(
        "--filter-unpredictable",
        action="store_true",
        default=_env_flag("TABSIL_FILTER_UNPREDICTABLE"),
    )
    # Training recipe options of the v2 experiments (TrainConfig comments). A resumed run keeps the
    # original run's values (`inherit_recipe_options`)
    parser.add_argument(
        "--target-train-rows",
        action="store_true",
        default=_env_flag("TABSIL_TARGET_TRAIN_ROWS"),
    )
    # For short comparison runs: the seed changes the initial weights, training data and validation
    # set, and the warmup replaces the preset's length. A resumed run keeps the original run's
    # seed, steps and warmup (`inherit_schedule`)
    parser.add_argument("--seed", type=int, default=_env("TABSIL_SEED", int))
    parser.add_argument("--warmup-steps", type=int, default=_env("TABSIL_WARMUP_STEPS", int))
    parser.add_argument("--resume-run-id", default=os.environ.get("TABSIL_RESUME_RUN_ID"))
    parser.add_argument("--resume-step", type=int, default=_env("TABSIL_RESUME_STEP", int))
    parser.add_argument("--init-run-id", default=os.environ.get("TABSIL_INIT_RUN_ID"))
    parser.add_argument("--init-step", type=int, default=_env("TABSIL_INIT_STEP", int))
    parser.add_argument("--init-hub", default=os.environ.get("TABSIL_INIT_HUB") or None)
    parser.add_argument("--out-dir", default=os.environ.get("TABSIL_OUT_DIR", "tabsil_run"))
    args = parser.parse_args(argv)
    if args.resume_step is not None and not args.resume_run_id:
        parser.error("--resume-step requires --resume-run-id")
    if args.init_step is not None and not args.init_run_id:
        parser.error("--init-step requires --init-run-id")
    if sum(bool(v) for v in (args.init_run_id, args.init_hub, args.resume_run_id)) > 1:
        parser.error("give only one of --init-run-id, --init-hub and --resume-run-id")
    later_stage = CURRICULUM_NAMES.get(args.config_name, 1) > 1
    if later_stage and not (args.init_run_id or args.init_hub or args.resume_run_id):
        # Each stage starts from the previous stage's weights with a new optimizer (TabICLv2
        # `--only_load_model True`). The regression borders carry over as well
        parser.error(
            f"{args.config_name} requires --init-run-id (the previous stage's run), --init-hub"
            " or --resume-run-id"
        )
    return args


def resume_phase_lr(config: TrainConfig, args: argparse.Namespace, params: dict) -> TrainConfig:
    """The lr when resuming a run with phases. The phases set the lr, so `--lr` is rejected, and the
    lr before the first phase (`config.lr`) is the original run's (also for a run whose lr was
    first changed with `--lr`)."""
    if not config.phases:
        return config
    if args.lr is not None:
        raise SystemExit("a run with phases does not take --lr")
    return replace(config, lr=float(params["lr"]))


def source_params(params: dict[str, str], tags: dict[str, str]) -> dict[str, str]:
    """The original run's params. The phases can change across resumes, so the `phases` tag is
    used when present (an MLflow param cannot be changed after it is first logged)."""
    if "phases" in tags:
        return {**params, "phases": tags["phases"]}
    return params


def source_speed_options(params: dict[str, str]) -> dict:
    """The original run's training speed options (MLflow params, strings). Options the run does
    not have are left out."""
    defaults = TrainConfig()
    out = {}
    for name in SPEED_OPTIONS:
        if name in params:
            default = getattr(defaults, name)
            text = params[name]
            if name == "phases":
                # Inherit phases only when present (resuming a run without them must not clear
                # the preset's phases)
                if phases := normalize_phases(text):
                    out[name] = phases
                continue
            out[name] = text == "True" if isinstance(default, bool) else type(default)(text)
    return out


def run_config_name(tags: dict[str, str]) -> str | None:
    """The `--config-name` a training run was started with, from the tags `log_run_source` wrote
    for it, read as `parse_args` read the command: the argument in the `command` tag, else the
    `env.TABSIL_CONFIG_NAME` tag, else the default. None for a run without a `command` tag (it
    was not started by this script)."""
    if "command" not in tags:
        return None
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config-name")
    given, _ = parser.parse_known_args(shlex.split(tags["command"])[1:])
    return given.config_name or tags.get("env.TABSIL_CONFIG_NAME") or DEFAULT_CONFIG_NAME


def check_resume_config_name(args: argparse.Namespace, tags: dict[str, str]) -> None:
    """On resume, stop unless `--config-name` is the one the original run was started with
    (`run_config_name`). The preset sets more than the resumed run takes from the original run
    (the gradient clipping, the checkpoint and validation intervals), so under another preset the
    run would continue with settings no run was started with. A run that takes only the weights
    (`--init-run-id`) starts anew under any preset."""
    if not args.resume_run_id:
        return
    saved = run_config_name(tags)
    if saved is None:
        raise SystemExit(
            "the resumed run has no `command` tag, so the --config-name it was started with is not"
            " known. Check that --resume-run-id is a training run of this script; to train anew"
            " from its weights use --init-run-id"
        )
    if args.config_name != saved:
        raise SystemExit(
            f"--config-name {args.config_name} differs from the resumed run's config name {saved}."
            " A run is resumed with the preset it was started with, which also sets the gradient"
            f" clipping and the checkpoint intervals: give --config-name {saved} (or"
            f" TABSIL_CONFIG_NAME={saved}), or train anew from the run's weights with"
            " --init-run-id"
        )


def inherit_batch_size(args: argparse.Namespace, params: dict[str, str]) -> None:
    """On resume without `--batch-size`, use the original run's batch (instead of silently falling
    back to the preset's). Training anew from a run's weights (`--init-run-id`) keeps the preset
    value."""
    if args.resume_run_id and args.batch_size is None and "batch_size" in params:
        args.batch_size = int(params["batch_size"])


# What a resumed run keeps of its schedule and data stream (TrainConfig field -> MLflow param).
# Without these a resumed run fell back to seed 0 and the preset's step count: other tables, another
# validation set and another cosine. The curriculum's warmup follows the inherited step count
SCHEDULE_OPTIONS = ("seed", "steps", "warmup_steps")


def inherit_schedule(args: argparse.Namespace, params: dict[str, str]) -> None:
    """On resume, the original run's seed, step count and warmup (`SCHEDULE_OPTIONS`) where none is
    given. A different `--seed` stops the script: the resumed steps would draw another run's
    tables."""
    if not args.resume_run_id:
        return
    for name in SCHEDULE_OPTIONS:
        if name not in params:
            continue
        saved = int(params[name])
        if getattr(args, name) is None:
            setattr(args, name, saved)
        elif name == "seed" and args.seed != saved:
            raise SystemExit(f"--seed {args.seed} differs from the resumed run's {saved}")


def inherit_reference_val(args: argparse.Namespace, params: dict[str, str]) -> None:
    """On resume without `--reference-val-tables`, keep the original run's reference validation
    set (so that a chunked run logs the same `ref_*` metrics in every chunk)."""
    if args.resume_run_id and args.reference_val_tables is None:
        if "reference_val_tables" in params:
            args.reference_val_tables = int(params["reference_val_tables"])


# Training recipe options a resumed run keeps (TrainConfig field -> parser of its MLflow param).
# A run must not change them midway, so a different value given on resume stops the script. A run
# that takes only the weights (`--init-run-id`) uses the preset's or the given values
RECIPE_OPTIONS = {"target_train_rows": lambda text: text == "True", "optimizer": str}


def inherit_recipe_options(args: argparse.Namespace, params: dict[str, str]) -> None:
    """On resume, the original run's recipe options (`RECIPE_OPTIONS`; runs that predate a param
    used its default). An on/off option can only be turned on with its flag, so an option the run
    had on is kept on, and turning on one the run had off stops the script."""
    if not args.resume_run_id:
        return
    defaults = TrainConfig()
    for name, parse in RECIPE_OPTIONS.items():
        saved = parse(params[name]) if name in params else getattr(defaults, name)
        given = getattr(args, name)
        if isinstance(saved, bool):
            if given and not saved:
                raise SystemExit(f"--{name.replace('_', '-')} is off in the resumed run")
            setattr(args, name, saved)
        elif given is None:
            setattr(args, name, saved)
        elif given != saved:
            raise SystemExit(
                f"--{name.replace('_', '-')} {given} differs from the resumed run's {saved}"
            )


def inherit_prior_type(args: argparse.Namespace, params: dict[str, str]) -> None:
    """On resume, train on the original run's prior: without `--prior-type` its `prior_type`
    param (runs that predate the param used `graph_scm`) is taken, and a different one stops the
    script (one run must stay on one prior). Training anew from a run's weights (`--init-run-id`)
    keeps the preset or the given prior."""
    if not args.resume_run_id:
        return
    saved = canonical_prior_type(params.get("prior_type", DEFAULT_PRIOR_TYPE))
    if args.prior_type is None:
        args.prior_type = saved
    elif canonical_prior_type(args.prior_type) != saved:
        raise SystemExit(
            f"--prior-type {args.prior_type} differs from the resumed run's prior_type {saved}"
        )


# Table size and prior settings a resumed run keeps: the TrainConfig fields the prior reads to draw
# a table (`tabsil_train.priors.graph_scm.make_prior`: its `SIZE_FIELDS`, `PRIOR_FIELDS` and
# `missing_dataset_prob`), except `target_train_rows`, which is one of RECIPE_OPTIONS. The first
# six have an argument, the last four are set by the preset only. Every run of this script has
# recorded them, so unlike RECIPE_OPTIONS there is no value to assume for a run without the param
DATA_OPTIONS = (
    "min_rows",
    "max_rows",
    "min_train_fraction",
    "max_train_fraction",
    "binary_prob",
    "filter_unpredictable",
    "max_features",
    "log_uniform_rows",
    "max_cells",
    "missing_dataset_prob",
)


def _run_param(name: str, default, text: str):
    """The resumed run's MLflow param `text` as the type of the TrainConfig field `name` (params
    are strings; `default` is the field's default). Read as `tabsil_train.hub` reads the model
    structure: a bool is True or False, and a param that does not read as its type stops the
    script (it does not count as off)."""
    try:
        if isinstance(default, bool):
            if text not in ("True", "False"):
                raise ValueError(text)
            return text == "True"
        return type(default)(text)
    except ValueError:
        kind = type(default).__name__
        raise SystemExit(
            f"the resumed run's {name} param does not read as {kind}: {text!r}"
        ) from None


def resume_data_options(args: argparse.Namespace, params: dict[str, str]) -> dict:
    """On resume, the table size and prior settings to train with (`DATA_OPTIONS`): the original
    run's, read from its params and not from the preset, so that the run stays on the tables it
    was trained on (`build_config` puts them over the preset; also when the preset's values have
    changed since, or the run was started with other values than its preset's). A different
    value given on resume stops the script, and so does a run that did not record them
    (nothing says which tables it saw). An on/off setting follows `inherit_recipe_options`: its
    flag can only turn it on, so turning on what the run had off stops the script. Empty unless
    resuming: a run that takes only the weights (`--init-run-id`) uses the preset's or the given
    values."""
    if not args.resume_run_id:
        return {}
    missing = [name for name in DATA_OPTIONS if name not in params]
    if missing:
        raise SystemExit(
            f"the resumed run did not record {', '.join(missing)}, so the tables it was trained on"
            " are not known and cannot be kept. Check that --resume-run-id is a training run that"
            " logged its params; to train anew from its weights use --init-run-id"
        )
    defaults = TrainConfig()
    saved = {name: _run_param(name, getattr(defaults, name), params[name]) for name in DATA_OPTIONS}
    for name, value in saved.items():
        given = getattr(args, name, None)  # a setting without an argument is never given
        flag = "--" + name.replace("_", "-")
        keep = (
            "A resumed run keeps the table sizes and prior settings it was trained with: leave"
            f" {flag} and TABSIL_{name.upper()} unset to keep them, or train anew from the run's"
            " weights with --init-run-id"
        )
        if isinstance(value, bool):
            if given and not value:
                raise SystemExit(f"{flag} is off in the resumed run. {keep}")
        elif given is not None and given != value:
            raise SystemExit(
                f"{flag} {given} differs from the resumed run's {name} {value}. {keep}"
            )
    return saved


# The two settings that fit a run's tables on a GPU. A resumed run takes them from the original run
# as well: the preset picks them for its own tables, and the run may have been started with other
# ones. `activation_checkpointing_rows` has no argument. The micro batch is the run's unless one is
# given, which is allowed: a chunk on other GPUs may need another, and it sets how many tables
# share a size in a forward pass, not which tables are drawn. Every run of this script has
# recorded both
MEMORY_OPTIONS = ("micro_batch", "activation_checkpointing_rows")


def resume_memory_options(args: argparse.Namespace, params: dict[str, str]) -> dict:
    """On resume, the settings that fit the run's tables on a GPU (`MEMORY_OPTIONS`), from the
    original run's params: its activation checkpointing threshold and, unless `--micro-batch` is
    given, its micro batch (the per-GPU micro batch it trained with; `for_world_size` still
    shrinks it for more GPUs). A run that did not record them stops the script. Empty unless
    resuming: a run that takes only the weights (`--init-run-id`) uses the preset's or the given
    values."""
    if not args.resume_run_id:
        return {}
    missing = [name for name in MEMORY_OPTIONS if name not in params]
    if missing:
        raise SystemExit(
            f"the resumed run did not record {', '.join(missing)}, so the micro batch and the"
            " activation checkpointing it was trained with are not known and cannot be kept."
            " Check that --resume-run-id is a training run that logged its params; to train anew"
            " from its weights use --init-run-id"
        )
    text = params["activation_checkpointing_rows"]
    try:
        saved = {"activation_checkpointing_rows": None if text == "None" else int(text)}
    except ValueError:
        raise SystemExit(
            "the resumed run's activation_checkpointing_rows param does not read as int or None:"
            f" {text!r}"
        ) from None
    if args.micro_batch is None:
        default = TrainConfig().micro_batch
        saved["micro_batch"] = _run_param("micro_batch", default, params["micro_batch"])
    return saved


def build_config(
    args: argparse.Namespace,
    source: dict | None = None,
    speed_options: dict | None = None,
    resume_options: dict | None = None,
) -> TrainConfig:
    """The training config. `source` is the model structure of the run being resumed or whose
    weights are taken (`tabsil_train.hub.run_architecture`); it must match the preset's structure.
    `speed_options` are that run's training speed options (`source_speed_options`), and
    `resume_options` what it keeps over the preset when it is resumed: its table size and prior
    settings (`resume_data_options`) and the settings that fit them on a GPU
    (`resume_memory_options`)."""
    try:
        if args.config_name in CURRICULUM_NAMES:
            # The curriculum sets the warmup from the step count, so --steps goes to the recipe
            base = PRESETS[args.config_name](steps=args.steps)
        else:
            base = PRESETS[args.config_name]()
    except ValueError as e:
        raise SystemExit(str(e)) from None
    # The size arguments change the preset's structure first, so a source run is checked against
    # the structure actually asked for
    base = _replace(base, _structure_values(args))
    _check_structure(base)
    if source is None:
        return _overrides(base, args)
    differ = {k: (getattr(base, k), v) for k, v in source.items() if getattr(base, k) != v}
    if differ:
        raise SystemExit(
            f"the source run is not the {recipe.NAME} structure of --config-name"
            f" {args.config_name} with the given size arguments (expected value, run value):"
            f" {differ}. A run of another size needs the same --emsize and --n-cls again"
        )
    source_phases = (speed_options or {}).get("phases")
    if source_phases and base.phases and source_phases != base.phases:
        raise SystemExit(
            f"the original run's phases {source_phases} differ from those of --config-name"
            f" {args.config_name}: {base.phases}"
        )
    # Apply the original run's structure and options and the arguments at once (so the config
    # check never sees an intermediate state). What a resumed run keeps comes last: a given table
    # setting equals the run's (`resume_data_options`), an on/off one may be off in the run, and
    # the run's micro batch is only among them when none is given
    changes = {
        **source,
        **(speed_options or {}),
        **_override_values(args),
        **(resume_options or {}),
    }
    return _replace(base, changes)


def _overrides(config: TrainConfig, args: argparse.Namespace) -> TrainConfig:
    """Overwrite the config with the values given as arguments or environment variables."""
    return _replace(config, _override_values(args))


def _structure_values(args: argparse.Namespace) -> dict:
    """The model size values given as arguments or environment variables (STRUCTURE_ARGS)."""
    return {n: getattr(args, n) for n in STRUCTURE_ARGS if getattr(args, n) is not None}


def _check_structure(config: TrainConfig) -> None:
    """Stop before the run starts if the model config rejects the structure (heads that do not
    divide a width, an odd RoPE head dim; `ColumnRowTransformerConfig.__post_init__`)."""
    try:
        config.model_config()
    except ValueError as e:
        raise SystemExit(f"model structure: {e}") from None


def _replace(config: TrainConfig, changes: dict) -> TrainConfig:
    try:
        return replace(config, **changes)
    except ValueError as e:  # incompatible options (`TrainConfig.__post_init__`)
        raise SystemExit(f"training config: {e}") from None


def _override_values(args: argparse.Namespace) -> dict:
    """The values given as arguments or environment variables (unset ones are left out)."""
    overrides = {
        "steps": args.steps,
        "batch_size": args.batch_size,
        "micro_batch": args.micro_batch,
        "num_workers": args.num_workers,
        "prefetch_factor": args.prefetch_factor,
        "gc_freeze": args.gc_freeze,
        "lr": args.lr,
        "stop_step": args.stop_step,
        "checkpoint_every": args.checkpoint_every,
        "keep_every": args.keep_every,
        "val_every": args.val_every,
        "log_every": args.log_every,
        "weight_average": args.weight_average,
        "weight_average_every": args.weight_average_every,
        "muon_weight_decay": args.muon_weight_decay,
        "muon_momentum": args.muon_momentum,
        "optimizer": args.optimizer,
        "muon_distributed": args.muon_distributed,
        "min_rows": args.min_rows,
        "max_rows": args.max_rows,
        "min_train_fraction": args.min_train_fraction,
        "max_train_fraction": args.max_train_fraction,
        "binary_prob": args.binary_prob,
        "prior_type": args.prior_type,
        "reference_val_tables": args.reference_val_tables,
        "seed": args.seed,
        "warmup_steps": args.warmup_steps,
        **{
            name: True
            for name in (
                "compile",
                "tf32",
                "grad_cosine",
                "rank_timing",
                "filter_unpredictable",
                "target_train_rows",
            )
            if getattr(args, name)
        },
        "rewind_lr_factor": args.rewind_lr_factor,
        "deadline": datetime.fromisoformat(args.deadline).timestamp() if args.deadline else None,
    }
    overrides = {k: v for k, v in overrides.items() if v is not None}
    # `off` disables it (CollapseGuard)
    if args.collapse_threshold is not None:
        overrides["collapse_threshold"] = _optional_float(args.collapse_threshold)
    return overrides


def _check_source(path: str, config: TrainConfig, *, resume: bool) -> float:
    # Checked before the MLflow run is opened (so a failure neither overwrites the original run's
    # tags and code nor leaves an empty run). Also returns the base lr from the same read
    try:
        return check_source_checkpoint(path, config, resume=resume)
    except ValueError as e:
        raise SystemExit(str(e)) from None


def hub_init_checkpoint(spec: str, config: TrainConfig, path: Path) -> Path:
    """The weights and regression borders of a Hub repo (`repo@revision`) or a local
    `save_pretrained` folder, written to `path` as a `save_checkpoint` file for `init_from`, after
    checking that the model matches the config (before the MLflow run is opened)."""
    from tabsil.hub import from_pretrained
    from tabsil.models import save_checkpoint

    model, borders = from_pretrained(spec)
    if borders is None:
        raise SystemExit(f"--init-hub {spec} has no regression borders")
    save_checkpoint(path, model, borders)
    try:
        check_checkpoint(torch.load(path, map_location="cpu", weights_only=True), config, spec)
    except ValueError as e:
        raise SystemExit(str(e)) from None
    return path


def _optional_float(value: str) -> float | None:
    return None if value.lower() in ("off", "none") else float(value)


def plot_history(history: dict) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.plot(history["step"], history["train_loss"], color="C0", label="train (prior)")
    if history["val_step"]:
        ax.plot(history["val_step"], history["val_loss"], "o-", color="C1", label="val (prior)")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("mean of classification CE and regression pinball loss")
    ax.set_title("TabSil prior-fitting loss")
    ax.legend()
    return fig


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    rank = int(os.environ.get("RANK", "0"))
    is_main = rank == 0
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    source, speed_options, resume_options = None, None, None
    if args.init_run_id or args.resume_run_id:
        run = mlflow.MlflowClient().get_run(args.init_run_id or args.resume_run_id)
        check_resume_config_name(args, run.data.tags)
        params = source_params(run.data.params, run.data.tags)
        try:
            source = run_architecture(params)
        except ValueError as e:  # an unknown model type or a param that does not parse
            raise SystemExit(str(e)) from None
        speed_options = source_speed_options(params)
        if args.init_run_id:
            # Taking only the weights starts a new schedule at step 0, so the original run's phases
            # (absolute steps) are not inherited
            speed_options.pop("phases", None)
        inherit_batch_size(args, params)
        inherit_schedule(args, params)
        inherit_prior_type(args, params)
        inherit_reference_val(args, params)
        inherit_recipe_options(args, params)
        resume_options = {
            **resume_data_options(args, params),
            **resume_memory_options(args, params),
        }
    config = build_config(args, source, speed_options, resume_options)
    if args.resume_run_id:
        config = resume_phase_lr(config, args, params)  # checked before the checkpoint download

    init_path = None
    if args.init_run_id:
        name = "latest.pt" if args.init_step is None else f"step_{args.init_step:07d}.pt"
        init_path = mlflow.artifacts.download_artifacts(
            run_id=args.init_run_id,
            artifact_path=f"checkpoints/{name}",
            dst_path=str(out_dir / f"init_rank{rank}"),
        )
        _check_source(init_path, config, resume=False)
    elif args.init_hub:
        init_path = hub_init_checkpoint(args.init_hub, config, out_dir / f"init_rank{rank}.pt")

    resume_path = None
    if args.resume_run_id:
        name = "latest.pt" if args.resume_step is None else f"step_{args.resume_step:07d}.pt"
        resume_path = mlflow.artifacts.download_artifacts(
            run_id=args.resume_run_id,
            artifact_path=f"checkpoints/{name}",
            dst_path=str(out_dir / f"resume_rank{rank}"),
        )
        saved_lr = _check_source(resume_path, config, resume=True)
        if config.phases:
            # base lr = phase lr x the checkpoint's collapse factor (training loop; lr set above)
            pass
        elif args.lr is None:
            # Without an explicit lr, keep the checkpoint's lr rather than the config default
            config = replace(config, lr=saved_lr)
        elif args.lr > saved_lr and config.rewind_lr_factor < 1 and rank == 0:
            print(
                f"warning: resuming with lr {args.lr:.2e}, higher than the checkpoint's base lr"
                f" {saved_lr:.2e}. If the collapse rule lowered that lr, the reduction is lost",
                flush=True,
            )

    if is_main:
        print("config_name:", args.config_name)
        print(config)
        # Log to the run passed by air if any, else to the resumed run, else to a new run.
        # Resuming from a saved step writes to a new run so the original run's latest.pt and
        # later step_*.pt are not overwritten
        resume_into = args.resume_run_id if args.resume_step is None else None
        run_id = os.environ.get("MLFLOW_RUN_ID") or resume_into
        run_name = None if run_id else f"{recipe.NAME}-{args.config_name}"
        mlflow.start_run(run_id=run_id, run_name=run_name)
        print("MLflow run_id:", mlflow.active_run().info.run_id, flush=True)
        log_run_source(__file__, env_prefix="TABSIL_")

    result = train(
        config, out_dir, resume_from=resume_path, mlflow_log=is_main, init_from=init_path
    )

    if is_main:
        mlflow.log_figure(plot_history(result["history"]), "loss_curve.png")
        mlflow.end_run()
        print("final model:", result["final_path"])


if __name__ == "__main__":
    # Fix the hash seed before anything runs (module docstring); an explicit value is kept
    if "PYTHONHASHSEED" not in os.environ:
        os.environ["PYTHONHASHSEED"] = "0"
        os.execv(sys.executable, [sys.executable, *sys.orig_argv[1:]])
    main()
