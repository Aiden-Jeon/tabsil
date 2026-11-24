"""The shared prior-fitting training loop (the procedure of the TabPFN v2 paper, Methods,
"Training details").

Every optimizer step, each GPU takes a batch of `micro_batch` datasets from the prior and
minimizes the loss of predicting the held-out targets of the eval rows given the train rows. One
model learns both tasks, alternating per step: classification with cross-entropy, regression with
the pinball loss of its quantile outputs. The effective batch is

    batch_size = micro_batch × grad_accumulation × world_size

and grad_accumulation is derived from the other values. When micro_batch × world_size exceeds
batch_size (many GPUs), the per-GPU micro batch shrinks to batch_size / world_size. The optimizer
is Muon on every parameter (`training.muon`); the learning rate follows a linear warmup and a
cosine decay.

Three fixes from training on GPUs are kept:
- bf16 has no GradScaler, so a step with an inf/nan gradient is skipped
- the number of train rows changes per forward, so cuDNN attention is off (it builds a new graph
  per shape and is slow)
- there is at least 1 train row (with 0, only that GPU leaves attention weights unused and DDP
  synchronization breaks)
"""

from __future__ import annotations

import collections
import contextlib
import copy
import json
import math
import os
import shutil
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from tabsil.models import (
    DEFAULT_MODEL_TYPE,
    canonical_model_type,
    config_dict,
    config_from_dict,
    info_fields,
    save_checkpoint,
    state_config,
    state_weights,
)
from tabsil.priors import prior_module
from tabsil.training.config import N_BUCKETS, TrainConfig
from tabsil.training.data import PriorBatches
from tabsil.training.diagnostics import (
    GroupNormWindow,
    RankTiming,
    TaskGradientCosine,
    group_norms,
    parameter_groups,
    rank_timing_metrics,
    weight_norm_metrics,
    write_rank_timing,
)
from tabsil.training.losses import prior_loss
from tabsil.training.muon import Muon, shard_group
from tabsil.training.stability import CollapseGuard
from tabsil.utils import DistContext, lr_factor, setup_distributed


def _borders(config: TrainConfig, d: DistContext, device: torch.device) -> torch.Tensor:
    """The regression borders, computed on rank 0 and broadcast to every GPU. Inference turns the
    quantile outputs into bucket probabilities on them; checkpoints keep them."""
    borders = torch.empty(N_BUCKETS + 1, device=device)
    if d.is_main:
        start = time.time()
        borders = (
            prior_module(config.prior_type)
            .make_borders(
                config,
                N_BUCKETS,
                seed=20_000_000 + config.seed,
                n_jobs=max(1, min(32, (os.cpu_count() or 1) // 2)),
            )
            .to(device)
        )
        print(f"regression borders: {config.border_datasets} datasets, {time.time() - start:.0f}s")
    if d.world_size > 1:
        dist.broadcast(borders, src=0)
    return borders


def _make_val_batches(config: TrainConfig, task: str, n_batches: int = 8):
    """Fixed validation prior batches from a seed disjoint from training (val_batch x n_batches)."""
    prior = config.make_prior(task, 10_000_000 + config.seed)
    return [prior.sample_batch(config.val_batch) for _ in range(n_batches)]


@torch.no_grad()
def _validate(
    model, val_batches, device, autocast, task, inference_rows: int | None = None
) -> dict[str, float]:
    """Loss over the validation batches (plus accuracy for classification). With
    `inference_rows`, batches with more rows use the model's inference path (`inference=True`,
    `TrainConfig.activation_checkpointing_rows`)."""
    model.eval()
    total, correct, count = 0.0, 0, 0
    for batch in val_batches:
        x, y = batch.x.to(device), batch.y.to(device)
        kwargs = {"task": task}
        if inference_rows is not None and x.shape[1] > inference_rows:
            kwargs["inference"] = True
        with autocast():
            logits = model(x, y[:, : batch.n_train], **kwargs)
        y_eval = y[:, batch.n_train :]
        total += prior_loss(logits, y_eval, task).item()
        if task == "classification":
            correct += (logits.argmax(-1) == y_eval.long()).sum().item()
            count += y_eval.numel()
    model.train()
    out = {"val_loss": total / len(val_batches)}
    if task == "classification":
        out["val_acc"] = correct / max(1, count)
    return out


def _validate_tasks(model, val_sets, device, autocast, config) -> dict[str, float]:
    """Validation of each task: the metric names end with the task, and `val_loss` is the mean of
    the two task losses."""
    rows = config.activation_checkpointing_rows
    out = {}
    for task, batches in val_sets.items():
        val = _validate(model, batches, device, autocast, task, rows)
        out.update({f"{k}_{task}": v for k, v in val.items()})
    out["val_loss"] = sum(out[f"val_loss_{t}"] for t in val_sets) / len(val_sets)
    return out


def _average_states(states) -> dict:
    """Mean of several state_dicts (LAWA). Float tensors are summed in float32 and cast back."""
    out = {}
    for key, first in states[0].items():
        if first.is_floating_point():
            out[key] = (sum(s[key].float() for s in states) / len(states)).to(first.dtype)
        else:
            out[key] = first
    return out


def _set_base_lr(opt, sched, lr: float, step: int, verbose: bool) -> None:
    """Change the base lr (base_lrs, initial_lr) and set the current lr to its value at `step`.

    Loading optimizer and scheduler states also restores their lr, so changing the lr on resume
    or lowering it after a collapse rewind must fix both the base and the current lr. The
    warmup/cosine shape is unchanged. The same lr changes nothing.
    """
    old = sched.base_lrs[0]
    factor = sched.lr_lambdas[0](step)  # the scheduler's own schedule
    sched.base_lrs = [lr] * len(opt.param_groups)
    for group in opt.param_groups:
        group["initial_lr"] = lr
        group["lr"] = lr * factor
    sched._last_lr = [group["lr"] for group in opt.param_groups]
    if verbose and not math.isclose(old, lr, rel_tol=1e-12):
        print(f"lr: base {old:.2e} -> {lr:.2e} ({lr * factor:.3e} from step {step})", flush=True)


def _set_muon_weight_decay(opt: Muon, weight_decay: float) -> None:
    """Set the weight decay of every Muon param group."""
    for group in opt.param_groups:
        group["weight_decay"] = weight_decay


def _phase_base_lr(config: TrainConfig, step: int, lr_scale: float) -> float:
    """Base lr of update `step` (0-based) = the phase lr x the collapse factor
    (`TrainConfig.phases`). With a factor below 1 it does not go under `rewind_min_lr` (a phase lr
    already below it stays). This is the same floor as the collapse rule."""
    lr = config.phase_value("lr", step)
    if lr_scale == 1.0:
        return lr
    return max(lr * lr_scale, min(config.rewind_min_lr, lr))


def _apply_phase(
    opt, sched, config: TrainConfig, step: int, lr_scale: float, verbose: bool
) -> None:
    """Apply the phase settings (base lr, Muon weight decay) of update `step`. Used by resuming,
    rewinds and phase starts. Idempotent for the same step."""
    lr = _phase_base_lr(config, step, lr_scale)
    if lr != sched.base_lrs[0]:
        _set_base_lr(opt, sched, lr, step, verbose)
    _set_muon_weight_decay(opt, config.phase_value("muon_weight_decay", step))


def _resume_lr_scale(state: dict, config: TrainConfig) -> float:
    """The checkpoint's collapse factor. For a checkpoint trained without phases: its base lr /
    the run's lr (`config.lr`; the training script passes the original run's value)."""
    if "lr_scale" in state:
        return float(state["lr_scale"])
    return float(state["scheduler"]["base_lrs"][0]) / config.lr


def check_checkpoint(state: dict, config: TrainConfig, name: str) -> None:
    """Check that `state` (a training checkpoint or a `save_checkpoint` file) has the same task
    and model as `config`.

    `load_state_dict` silently accepts a different config with the same shapes (e.g. head count),
    so compare up front. Training checkpoints that predate the stored model config are checked
    for the task only.
    """
    # the model serves regression, so its checkpoints hold the regression borders
    if state.get("borders") is None:
        raise ValueError(f"{name} has no regression borders (not a checkpoint of this model)")
    saved_config = state_config(state)
    if saved_config is None:
        return
    saved_type = canonical_model_type(state.get("model_type", DEFAULT_MODEL_TYPE))
    if saved_type != canonical_model_type(config.model_type):
        raise ValueError(f"{name} is a {saved_type} model (config model_type: {config.model_type})")
    # Missing fields (omitted as defaults, or added after saving) are filled with their defaults
    # before comparing. config_from_dict stops on a field this code does not know, naming it
    try:
        saved = config_from_dict(saved_type, saved_config).to_dict()
    except ValueError as e:
        raise ValueError(f"{name}: {e}") from None
    expected = config.model_config().to_dict()
    # fields unrelated to weight shapes (features seen in training etc.) may change on resume
    skip = set(info_fields(saved_type))
    diff = {
        k: (saved.get(k), v) for k, v in expected.items() if k not in skip and saved.get(k) != v
    }
    if diff:
        raise ValueError(f"{name} has a different model config (saved, config): {diff}")


def check_resume_step(state: dict, config: TrainConfig) -> None:
    """Check that the checkpoint to resume is before this run's stop step; otherwise there is
    nothing to train.

    Resuming a run of a different config by mistake (e.g. another curriculum stage) would end
    without training and overwrite that run's results.
    """
    # minipfn checkpoints may come from other optimizers (Adam, schedule-free AdamW; runs from
    # before the key existed are Adam). Only a Muon optimizer state can be continued
    saved_optimizer = state.get("optimizer_type", "adam")
    if state.get("schedule_free", False) or saved_optimizer != "muon":
        kind = "schedule_free" if state.get("schedule_free", False) else saved_optimizer
        raise ValueError(
            f"the checkpoint to resume was trained with {kind}; TabSil continues Muon runs only"
            " (take its weights with init_from instead)"
        )
    step, stop = int(state["step"]), min(config.stop_step or config.steps, config.steps)
    if step >= stop:
        raise ValueError(
            f"nothing to train: the checkpoint to resume is already at step {step}, stop step"
            f" {stop} (check that you resume a run of the same config and steps/stop_step)"
        )


def check_source_checkpoint(path: str | Path, config: TrainConfig, *, resume: bool) -> float:
    """Check up front that a checkpoint to resume (`resume`) or take weights from works with this
    config (task, model, resume step). `train` checks the same, but the training script calls
    this before opening the MLflow run, so it neither overwrites the original run's tags and code
    nor leaves an empty run. ValueError on mismatch.

    Returns the checkpoint's base lr, which a resumed run keeps unless an lr is given. The
    checkpoint is read once here (mmap)."""
    state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    check_checkpoint(state, config, "resume_from" if resume else "init_from")
    if resume:
        check_resume_step(state, config)
    return float(state["scheduler"]["base_lrs"][0])


def _to_cpu(obj):
    """Copy the tensors of a state_dict to CPU (an in-memory snapshot)."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().to("cpu", copy=True)
    if isinstance(obj, dict):
        return {k: _to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_cpu(v) for v in obj]
    return obj


def train(
    config: TrainConfig,
    out_dir: str | Path,
    resume_from: str | Path | None = None,
    mlflow_log: bool = True,
    init_from: str | Path | None = None,
) -> dict:
    """Run prior-fitting and return the final checkpoint path and the loss history.

    `resume_from` continues the full training state (optimizer, schedule, step); `init_from`
    takes only the weights (and regression borders) and starts from step 0 with a new optimizer
    (e.g. the next curriculum stage).

    Only rank 0 logs to MLflow, into the active run if there is one (AI Runtime passes the run via
    `MLFLOW_RUN_ID`). Every `checkpoint_every` steps the full training state is saved to
    `out_dir/latest.pt` and logged as an MLflow artifact (`checkpoints/`); every `keep_every`
    steps it is also kept as a non-overwritten `step_XXXXXXX.pt`. When `CollapseGuard` detects a
    collapse, training rewinds to the healthy state kept in memory every `snapshot_every` steps
    and continues with a new data seed (MLflow `rewinds`). A possibly collapsing state is never
    checkpointed.
    """
    # catch errors visible from the arguments alone before opening the process group
    if resume_from is not None and init_from is not None:
        raise ValueError("resume_from and init_from cannot be given together")
    d, device = setup_distributed()
    # the tf32 speed option is process-wide, so restore it at the end
    # (so it does not leak into later training or evaluation in the same process)
    matmul_precision = torch.get_float32_matmul_precision()
    try:
        return _train(
            config,
            out_dir,
            d,
            device,
            resume_from=resume_from,
            mlflow_log=mlflow_log,
            init_from=init_from,
        )
    except BaseException:
        # a failure after building the model (checkpoint shape checks) leaves other GPUs
        # waiting, so close the process group and re-raise (the normal path closes it at the end
        # of `_train`)
        if d.world_size > 1 and dist.is_initialized():
            dist.destroy_process_group()
        raise
    finally:
        torch.set_float32_matmul_precision(matmul_precision)


def _train(
    config: TrainConfig,
    out_dir: str | Path,
    d: DistContext,
    device: torch.device,
    *,
    resume_from: str | Path | None,
    mlflow_log: bool,
    init_from: str | Path | None,
) -> dict:
    """The body of `train`. `train` opens the process group; it is closed here on success and by
    `train` on failure."""
    config = config.for_world_size(d.world_size)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    accumulation = config.grad_accumulation(d.world_size)
    use_cuda = device.type == "cuda"
    if use_cuda:
        torch.backends.cuda.enable_cudnn_sdp(False)
    if config.tf32:  # speed option (TrainConfig comment). Off: PyTorch default (highest)
        torch.set_float32_matmul_precision("high")

    torch.manual_seed(config.seed)  # every GPU starts from the same initial weights
    model = config.build_model().to(device)
    if config.activation_checkpointing_rows is not None:  # config check verified the class
        model.activation_checkpointing_rows = config.activation_checkpointing_rows
    # Muon on every parameter (`training.muon`), warmup + cosine. Distributed Muon splits the
    # orthogonalization among GPUs (every GPU creates the group together)
    group = None if config.muon_distributed == "off" else shard_group(config.muon_distributed)
    opt = Muon.for_model(
        model,
        lr=config.lr,
        weight_decay=config.muon_weight_decay,
        momentum=config.muon_momentum,
        process_group=group,
    )
    schedule = lambda s: lr_factor(s, config.warmup_steps, config.steps)  # noqa: E731
    sched = torch.optim.lr_scheduler.LambdaLR(opt, schedule)

    start_step, skipped_steps, borders, rewinds = 0, 0, None, 0
    # lr factor lowered by collapse rewinds (used with phases, `_phase_base_lr`)
    lr_scale = 1.0
    resumed_train_seconds = 0.0
    history: dict[str, list] = {"step": [], "train_loss": [], "val_step": [], "val_loss": []}
    if init_from is not None:
        state = torch.load(init_from, map_location=device, weights_only=True)
        check_checkpoint(state, config, "init_from")
        model.load_state_dict(state_weights(state))
        borders = state["borders"]
    if resume_from is not None:
        state = torch.load(resume_from, map_location=device, weights_only=True)
        check_checkpoint(state, config, "resume_from")
        check_resume_step(state, config)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        sched.load_state_dict(state["scheduler"])
        start_step = int(state["step"])
        history = state["history"]
        skipped_steps = int(state["skipped_steps"])
        # carry over the rewind count so a resumed run does not see post-collapse data again
        rewinds = int(state.get("rewinds", 0))
        resumed_train_seconds = float(state.get("train_seconds", 0.0))
        if config.phases:
            # with phases: the phase lr of the resume step x the checkpoint's collapse factor,
            # and that phase's weight decay
            lr_scale = _resume_lr_scale(state, config)
            _apply_phase(opt, sched, config, start_step, lr_scale, d.is_main)
        else:
            _set_base_lr(opt, sched, config.lr, start_step, d.is_main)
        borders = state["borders"]
    if borders is None:
        borders = _borders(config, d, device)
    # each step uses only one task's target encoder and output layer, so allow unused parameters
    # (every GPU uses the same task in the same step)
    ddp_model = (
        DistributedDataParallel(model, device_ids=[d.local_rank], find_unused_parameters=True)
        if d.world_size > 1
        else model
    )
    # Training diagnostics (`training.diagnostics`): per-group norms and the task gradient cosine
    # are computed on rank 0 only (after DDP averaging every GPU has the same gradient). Every GPU
    # measures its own timing
    norm_groups, grad_window, cosine = None, None, None
    if d.is_main:
        # every parameter is Muon's (the group names keep the `/muon` suffix of minipfn's runs)
        norm_groups = parameter_groups(model, {id(p) for p in model.parameters()})
        grad_window = GroupNormWindow(norm_groups)
        if config.grad_cosine:
            cosine = TaskGradientCosine(norm_groups, config.tasks, config.log_every)
    cosine_metrics: dict[str, float] = {}  # cosine for the next log
    timing = RankTiming(device) if config.rank_timing else None
    # one file per chunk (so a resumed MLflow run does not overwrite the previous chunk's file)
    timing_path = out_dir / f"rank_timing_from_{start_step:07d}.jsonl"
    # Training forward. With the compile option each repeated block is compiled (compiling the
    # whole model recompiles every batch because of Python branches on the table size). Models
    # without a block list are compiled whole
    train_forward = ddp_model
    if config.compile:
        # blocks of different widths (row stages, ICL) share forward code and are compiled once
        # per shape. Beyond the default cache limit (8) they silently fall back to eager, so
        # raise it
        torch._dynamo.config.cache_size_limit = max(64, torch._dynamo.config.cache_size_limit)
        blocks = model.repeated_blocks() if hasattr(model, "repeated_blocks") else []
        for block in blocks:
            block.compile(dynamic=True)
        if not blocks:
            train_forward = torch.compile(ddp_model, dynamic=True)
    # rank 0's validation copy: holds the LAWA average (training weights are untouched)
    eval_copy_model = None
    averages = collections.deque(maxlen=config.weight_average) if config.weight_average else None
    if averages is not None and start_step and d.is_main:
        # the weights to average live only in memory (K models are too large for a checkpoint)
        print(
            f"LAWA: resumed, so the averaging window is empty; it fills again"
            f" {config.weight_average * config.weight_average_every} steps after step"
            f" {start_step}",
            flush=True,
        )
    # pure training time (excluding validation, checkpoints, logging), for time-to-target. Kept in
    # checkpoints so it accumulates across resumes
    train_seconds = resumed_train_seconds

    def eval_copy(state: dict):
        nonlocal eval_copy_model
        if eval_copy_model is None:
            eval_copy_model = copy.deepcopy(model)
        eval_copy_model.load_state_dict(state)
        return eval_copy_model

    def autocast():
        if use_cuda:
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    # prior workers are split among the tasks (CPU use per GPU stays num_workers)
    workers = config.num_workers // len(config.tasks) if config.num_workers else 0
    if config.num_workers:
        workers = max(1, workers)
    loader_kwargs = {"num_workers": workers, "pin_memory": use_cuda}
    if workers > 0:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=4)

    def make_batches(from_step: int, rewinds: int) -> dict:
        # mix the start step into the seed so a resumed run does not see the same data again.
        # After a collapse rewind the rewind count is mixed in too, for new data. Also the task
        # index
        seed = (config.seed, from_step, d.rank) + ((rewinds,) if rewinds else ())
        return {
            task: iter(
                DataLoader(
                    PriorBatches(config, task, seed=seed + (i,)),
                    batch_size=None,
                    **loader_kwargs,
                )
            )
            for i, task in enumerate(config.tasks)
        }

    batches = make_batches(start_step, rewinds)

    mlf, val_sets = None, None
    if d.is_main:
        val_sets = {task: _make_val_batches(config, task) for task in config.tasks}
        if mlflow_log:
            import mlflow as mlf

            if mlf.active_run() is None:
                mlf.start_run()
            run_params = mlf.get_run(mlf.active_run().info.run_id).data.params
            if "steps" not in run_params:
                mlf.log_params(asdict(config))
                mlf.log_params(
                    {
                        "world_size": d.world_size,
                        "grad_accumulation": accumulation,
                        "n_parameters": model.num_parameters(),
                        "device": str(device),
                        "cpu_count": os.cpu_count(),
                    }
                )
            else:
                # params of a continued run cannot change, so only add settings added since
                new = {k: v for k, v in asdict(config).items() if k not in run_params}
                if new:
                    mlf.log_params(new)
            # phases can change on resume (adding a preset's phases to a run without them), so
            # also store them in a mutable tag. The training script reads the tag before the
            # param
            if config.phases or run_params.get("phases", "()") != "()":
                mlf.set_tag("phases", str(config.phases))

    guard = None
    if config.collapse_threshold is not None:
        guard = CollapseGuard(config.collapse_threshold, config.collapse_patience)
    rewinds_here = 0  # rewinds in this run (rewinds is the total including earlier chunks)
    snapshots: list[dict] = []  # healthy states kept in memory (latest 2)

    def snapshot(at_step: int) -> None:
        snapshots.append(
            {
                "model": _to_cpu(model.state_dict()),
                "optimizer": _to_cpu(opt.state_dict()),
                "scheduler": copy.deepcopy(sched.state_dict()),
                "step": at_step,
                "history": copy.deepcopy(history),
                "skipped_steps": skipped_steps,
                "healthy": list(guard.healthy),
            }
        )
        del snapshots[:-2]

    ddp_model.train()
    window_loss, window_wait, window_start, window_finite = 0.0, 0.0, time.time(), 0
    window_tasks = {t: [0.0, 0] for t in config.tasks}
    last_log_step, skip_reports = start_step, 0
    stop_step = min(config.stop_step or config.steps, config.steps)
    if guard is not None:
        snapshot(start_step)
    phase_starts = {start for start, _ in config.phases}
    step = start_step - 1
    while step + 1 < stop_step:
        step += 1
        if step in phase_starts:
            _apply_phase(opt, sched, config, step, lr_scale, d.is_main)
            if d.is_main:
                lr, wd = sched.base_lrs[0], config.phase_value("muon_weight_decay", step)
                print(
                    f"step {step}: phase start (base lr {lr:.2e}, muon_weight_decay {wd:g})",
                    flush=True,
                )
        t_step = time.time()
        # alternate the task per step (same task on every GPU; 1:1 ratio, our choice)
        task = config.tasks[step % len(config.tasks)]
        step_loss = torch.zeros((), device=device)
        micro_info = []
        for micro in range(accumulation):
            t_wait = time.time()
            x, y, n_train = next(batches[task])
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            window_wait += time.time() - t_wait
            sync = micro == accumulation - 1 or d.world_size == 1
            ctx = contextlib.nullcontext() if sync else ddp_model.no_sync()
            marks = (timing.mark(),) if timing is not None else ()
            with ctx:
                with autocast():
                    logits = train_forward(x, y[:, :n_train], task=task)
                loss = prior_loss(logits, y[:, n_train:], task) / accumulation
                if timing is not None:
                    marks += (timing.mark(),)
                loss.backward()
            if timing is not None:  # backward includes the last micro batch's all-reduce wait
                timing.add_micro(*marks, timing.mark(), x, n_train)
            step_loss += loss.detach()
            micro_info.append((loss.detach(), x, y, n_train))
        # per-group gradient norms are pre-clip values
        step_norms = group_norms(norm_groups, grads=True) if grad_window is not None else None
        # skip this step on an inf/nan gradient (after DDP averaging every GPU decides the same)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        finite = bool(torch.isfinite(grad_norm))
        if grad_window is not None and finite:
            grad_window.update(task, step_norms, grad_norm)
        # clipping scales the whole gradient by one factor, so the cosine equals the pre-clip one
        if cosine is not None and cosine.wants(step):
            if (found := cosine.update(step, task, finite)) is not None:
                cosine_metrics = found
        if finite:
            opt.step()
        else:
            skipped_steps += 1
            if skip_reports < 5:  # log per-rank batch info the first few times, to find the cause
                skip_reports += 1
                for loss_, x_, y_, n_ in micro_info:
                    finite_x = x_[torch.isfinite(x_)]
                    print(
                        f"[rank {d.rank}] step {step + 1} non-finite grad: loss {loss_.item():.4g} "
                        f"x {tuple(x_.shape)} n_train {n_} "
                        f"|x|max {finite_x.abs().max().item() if finite_x.numel() else 0:.3g} "
                        f"y [{y_.min().item():.3g}, {y_.max().item():.3g}]",
                        flush=True,
                    )
        opt.zero_grad(set_to_none=True)
        sched.step()
        if timing is not None:
            timing.step_done(step, config.tasks.index(task))

        if d.world_size > 1:
            dist.all_reduce(step_loss, op=dist.ReduceOp.AVG)
        if torch.isfinite(step_loss):
            window_loss += step_loss.item()
            window_finite += 1
            window_tasks[task][0] += step_loss.item()
            window_tasks[task][1] += 1
        train_seconds += time.time() - t_step  # the .item() above waited for the GPU

        done = step + 1
        # check the deadline every log_every steps, agreeing across GPUs to stop at the same step
        if config.deadline is not None and done % config.log_every == 0:
            late = torch.tensor(float(time.time() > config.deadline), device=device)
            if d.world_size > 1:
                dist.all_reduce(late, op=dist.ReduceOp.MAX)
            if late.item() > 0:
                stop_step = done
        last = done == stop_step

        rewound_at_end = False
        if guard is not None:
            collapsed = False
            if done % config.log_every == 0:
                # window_loss is post-all-reduce, so it is the same on every GPU
                collapsed = guard.update(window_loss / window_finite if window_finite else math.inf)
            # on a collapse, or when ending in a suspect state, rewind to a healthy state
            if collapsed or (last and guard.suspect):
                rewinds, rewinds_here = rewinds + 1, rewinds_here + 1
                if rewinds_here > config.max_rewinds:
                    raise RuntimeError(
                        f"step {done}: rewound more than {config.max_rewinds} times on collapse"
                    )
                # on a collapse take the older snapshot, to land before the collapse began; when
                # only ending in a suspect state (could be a single spike) take the latest one
                snap = snapshots[0] if collapsed else snapshots[-1]
                # the snapshot may predate an earlier lr cut, so lower from the pre-rewind lr
                lr_now = sched.base_lrs[0]
                model.load_state_dict(snap["model"])
                opt.load_state_dict(snap["optimizer"])
                sched.load_state_dict(snap["scheduler"])
                # ending after a single spike is not a collapse, so the lr stays
                if config.phases:
                    # lower only the factor and apply the phase lr and weight decay of the step
                    # rewound to (its phase even across a boundary; floor in `_phase_base_lr`)
                    if collapsed:
                        lr_scale *= config.rewind_lr_factor
                    _apply_phase(opt, sched, config, snap["step"], lr_scale, d.is_main)
                else:
                    lr_next = lr_now
                    if collapsed:  # lower down to the floor only (an lr already below it stays)
                        lr_next = min(
                            lr_now, max(lr_now * config.rewind_lr_factor, config.rewind_min_lr)
                        )
                    _set_base_lr(opt, sched, lr_next, snap["step"], d.is_main)
                history = copy.deepcopy(snap["history"])
                skipped_steps = snap["skipped_steps"]
                guard.healthy, guard.bad = list(snap["healthy"]), 0
                if averages is not None:  # drop weights that may have been collapsing
                    averages.clear()
                snapshots[:] = [snap]
                if d.is_main:
                    print(
                        f"step {done}: train loss collapse -> rewound to step {snap['step']}",
                        flush=True,
                    )
                    if mlf is not None:
                        mlf.log_metric("rewinds", rewinds, step=done)
                del batches
                batches = make_batches(snap["step"], rewinds)
                window_loss, window_wait, window_start, window_finite = 0.0, 0.0, time.time(), 0
                window_tasks = {t: [0.0, 0] for t in config.tasks}
                for diagnostic in (grad_window, cosine, timing):
                    if diagnostic is not None:  # discard records of the rewound range
                        diagnostic.reset()
                cosine_metrics = {}
                last_log_step = snap["step"]
                if not last:
                    step = snap["step"] - 1
                    continue
                done, rewound_at_end = snap["step"], True  # save the rewound state and stop
        log_now = (done % config.log_every == 0 or last) and not rewound_at_end
        # every GPU joins the timing gather (only rank 0 receives)
        timing_rows = timing.gather(d.world_size) if timing is not None and log_now else None
        if d.is_main and log_now:
            n = done - last_log_step
            elapsed = time.time() - window_start
            metrics = {
                "train_loss": window_loss / max(1, window_finite),
                "lr": sched.get_last_lr()[0],
                # the schedule's base lr; differs from config.lr after a resume change or rewinds
                "base_lr": sched.base_lrs[0],
                "grad_norm": float(grad_norm),
                "skipped_steps": skipped_steps,
                "sec_per_step": elapsed / n,
                "data_wait_frac": window_wait / max(elapsed, 1e-9),
                "datasets_per_sec": config.batch_size * n / max(elapsed, 1e-9),
            }
            # train_loss is the mean over the alternating tasks
            for t, (total, count) in window_tasks.items():
                if count:
                    metrics[f"train_loss_{t}"] = total / count
            # output layer weight norm; steady growth sharpens outputs and destabilizes training
            if (decoder := model.decoder_weight()) is not None:
                metrics["decoder_norm"] = decoder.norm().item()
            if use_cuda:
                metrics["gpu_mem_gb"] = torch.cuda.max_memory_allocated(device) / 1e9
            # diagnostics: per-group gradient norms (window mean and max), weight norms, task
            # gradient cosine, per-rank timing and table sizes
            metrics.update(grad_window.metrics())
            metrics.update(weight_norm_metrics(norm_groups))
            metrics.update(cosine_metrics)
            cosine_metrics = {}
            if timing_rows is not None:
                metrics.update(rank_timing_metrics(timing_rows))
                write_rank_timing(timing_path, timing_rows, config.tasks)
            history["step"].append(done)
            history["train_loss"].append(metrics["train_loss"])
            print(
                f"step {done}/{config.steps} loss {metrics['train_loss']:.4f} "
                f"lr {metrics['lr']:.2e} {metrics['sec_per_step']:.3f}s/step "
                f"wait {metrics['data_wait_frac']:.0%} skipped {skipped_steps}",
                flush=True,
            )
            if mlf is not None:
                mlf.log_metrics(metrics, step=done)
        if done % config.log_every == 0 or last:
            window_loss, window_wait, window_start, window_finite = 0.0, 0.0, time.time(), 0
            window_tasks = {t: [0.0, 0] for t in config.tasks}
            if grad_window is not None:
                grad_window.reset()
            last_log_step = done

        if (
            averages is not None
            and d.is_main
            and done % config.weight_average_every == 0
            and not rewound_at_end
        ):
            averages.append(_to_cpu(model.state_dict()))
        if d.is_main and (done % config.val_every == 0 or last) and not rewound_at_end:
            val = _validate_tasks(model, val_sets, device, autocast, config)
            # LAWA only once its window (K) is full; after a resume the window refills
            if averages is not None and len(averages) == config.weight_average:
                averaged = eval_copy(_average_states(list(averages)))
                avg = _validate_tasks(averaged, val_sets, device, autocast, config)
                val.update({f"{k}_avg": v for k, v in avg.items()})
            val["train_seconds"] = train_seconds
            history["val_step"].append(done)
            history["val_loss"].append(val["val_loss"])
            print("  val " + " ".join(f"{k} {v:.4f}" for k, v in val.items()), flush=True)
            if mlf is not None:
                mlf.log_metrics(val, step=done)

        keep = bool(config.keep_every) and done % config.keep_every == 0
        suspect = guard is not None and guard.suspect  # never save a possibly collapsing state
        if d.is_main and (done % config.checkpoint_every == 0 or keep or last) and not suspect:
            state = {
                # model type and config (the file alone rebuilds the model, without run params)
                "model_type": config.model_type,
                # `model_config`, not `config`, so older code does not mistake this for a
                # save_checkpoint file (`models.state_config`)
                "model_config": config_dict(model.config),
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "scheduler": sched.state_dict(),
                "step": done,
                "history": history,
                "skipped_steps": skipped_steps,
                "borders": borders,
                "rewinds": rewinds,
                "train_seconds": train_seconds,
                # optimizer kind (checked on resume, `check_resume_step`)
                "optimizer_type": "muon",
            }
            if config.phases:  # checkpoints without phases keep the plain format
                state["lr_scale"] = lr_scale
            torch.save(state, out_dir / "latest.pt")
            kept = None
            if keep:
                kept = out_dir / f"step_{done:07d}.pt"
                shutil.copyfile(out_dir / "latest.pt", kept)
            if mlf is not None:
                mlf.log_artifact(str(out_dir / "latest.pt"), "checkpoints")
                if kept is not None:
                    mlf.log_artifact(str(kept), "checkpoints")
                mlf.log_metric("checkpoint_step", done, step=done)
                if timing_path.exists():
                    mlf.log_artifact(str(timing_path), "diagnostics")
        if last:
            break
        if guard is not None and done % config.snapshot_every == 0 and not guard.suspect:
            snapshot(done)

    final_path = out_dir / "final.pt"
    if d.is_main:
        final_borders = None if borders is None else borders.cpu()
        save_checkpoint(final_path, model, final_borders)
        if averages is not None and len(averages) == config.weight_average:  # LAWA average
            save_checkpoint(
                out_dir / "final_avg.pt", eval_copy(_average_states(list(averages))), final_borders
            )
            if mlf is not None:
                mlf.log_artifact(str(out_dir / "final_avg.pt"), "model")
        (out_dir / "history.json").write_text(json.dumps(history))
        (out_dir / "train_config.json").write_text(json.dumps(asdict(config)))
        if mlf is not None:
            mlf.log_artifact(str(final_path), "model")
            mlf.log_artifact(str(out_dir / "history.json"))
            if timing_path.exists():
                mlf.log_artifact(str(timing_path), "diagnostics")
    if d.world_size > 1:
        dist.barrier()
        dist.destroy_process_group()
    return {"final_path": str(final_path), "history": history, "is_main": d.is_main}
