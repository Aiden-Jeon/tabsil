"""Training diagnostics. They only read weights, gradients and timings; the training
computation is unchanged.

- Per-group norms: parameters are grouped by stage (set by the model, `parameter_stage`) and
  optimizer group (`/muon`), with the weight norm and the pre-clip gradient norm per group. Parts
  used by one task only (target encoder, output layer) get their own `<stage>_<task>` group. The
  small parameters that set the attention logit scale (LogScale a_h and gate) get their own
  `<stage>_log_scale` group: in long runs the gradient concentrated there (checkpoint analysis),
  hidden inside the norm of the whole stage
- Task gradient cosine (option `grad_cosine`): multitask alternates the task per step, so the
  gradients of the 4 consecutive steps before a log step (cls, reg, cls, reg or the reverse) give
  the cosine of every group both tasks share. Cross-task is the mean of the three pairs 1 step
  apart, same-task the pairs 2 steps apart. The same-task value is the baseline for "difference
  from different batches"; a lower cross-task value means the two task gradients point in
  different directions in the shared parts
- Per-rank timing and table size (option `rank_timing`): each GPU measures the forward and
  backward time of a step and that step's table size (rows, train rows, features, cells), gathered
  to rank 0 at every log step. Backward time includes waiting for the slowest GPU in the DDP
  all-reduce
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn

# parts that set the attention logit scale: a parameter whose name contains one of these modules
# gets its own group within its stage
ATTENTION_SCALE_PARTS: dict[str, str] = {"log_scale": "log_scale"}


def parameter_groups(model: nn.Module, muon_ids: set[int] | None = None) -> dict[str, list]:
    """Group name -> parameter list (`model.named_parameters()` order).

    Names are `<stage>[_log_scale][_<task>][/muon|/adam]`. The stage is
    `model.parameter_stage(name)` (else the top-level module name). `_log_scale` marks attention
    logit scale parts (`ATTENTION_SCALE_PARTS`); the task is added for parts used by only that
    task in a multi-task model; the optimizer suffix only with Muon (`muon_ids`: ids of the
    parameters Muon handles)."""
    tasks = tuple(getattr(model, "tasks", ()))
    stage_of = getattr(model, "parameter_stage", None)
    groups: dict[str, list] = {}
    for name, p in model.named_parameters():
        group = stage_of(name) if stage_of is not None else name.split(".")[0]
        parts = name.split(".")
        scale = next((ATTENTION_SCALE_PARTS[q] for q in parts if q in ATTENTION_SCALE_PARTS), None)
        if scale is not None:
            group += f"_{scale}"
        if len(tasks) > 1:
            task = next((t for t in tasks if t in parts), None)
            if task is not None:
                group += f"_{task}"
        if muon_ids is not None:
            group += "/muon" if id(p) in muon_ids else "/adam"
        groups.setdefault(group, []).append(p)
    return groups


def shared_groups(groups: dict[str, list], tasks: tuple[str, ...]) -> dict[str, list]:
    """Groups shared by both tasks (names without a task suffix)."""
    return {
        name: params
        for name, params in groups.items()
        if not any(name.split("/")[0].endswith(f"_{t}") for t in tasks)
    }


@torch.no_grad()
def group_norms(groups: dict[str, list], grads: bool = False) -> dict[str, torch.Tensor]:
    """L2 norm of the weights (gradients with `grads`) per group, left on the GPU (read at log
    time). Groups without any gradient (parts of the task unused this step) are left out."""
    out = {}
    for name, params in groups.items():
        tensors = [p.grad if grads else p for p in params]
        tensors = [t for t in tensors if t is not None]
        if tensors:
            norms = torch._foreach_norm([t.float() for t in tensors])
            out[name] = torch.linalg.vector_norm(torch.stack(norms))
    return out


class GroupNormWindow:
    """Per-group pre-clip gradient norms over a log window: summed per step, mean and max at log.

    Collected per task (metric `grad_norm/<task>/<group>`). Also the window median and max of the
    total gradient norm (the pre-clip value from `clip_grad_norm_`); the single-step `grad_norm`
    misses spiky steps."""

    def __init__(self, groups: dict[str, list]) -> None:
        self.groups = groups
        self.reset()

    def reset(self) -> None:
        self.sums: dict[tuple[str, str], list] = {}  # (task, group) -> [sum, max, steps]
        self.totals: list[torch.Tensor] = []

    def update(self, task: str, norms: dict[str, torch.Tensor], total: torch.Tensor) -> None:
        """For steps with a finite gradient only. `norms` is `group_norms(..., grads=True)`."""
        for name, value in norms.items():
            entry = self.sums.get((task, name))
            if entry is None:
                self.sums[(task, name)] = [value, value, 1]
            else:
                entry[0] = entry[0] + value
                entry[1] = torch.maximum(entry[1], value)
                entry[2] += 1
        self.totals.append(total.detach().float())

    def metrics(self) -> dict[str, float]:
        out: dict[str, float] = {}
        if self.totals:
            totals = torch.stack(self.totals)
            out["grad_norm_median"] = totals.median().item()
            out["grad_norm_max"] = totals.max().item()
        if not self.sums:
            return out
        keys = list(self.sums)
        stacked = torch.stack(
            [torch.stack([self.sums[k][0] / self.sums[k][2], self.sums[k][1]]) for k in keys]
        ).tolist()  # a single read from the GPU
        for (task, name), (mean, peak) in zip(keys, stacked, strict=True):
            out[f"grad_norm/{task}/{name}"] = mean
            out[f"grad_norm_max/{task}/{name}"] = peak
        return out


def weight_norm_metrics(groups: dict[str, list]) -> dict[str, float]:
    norms = group_norms(groups)
    values = torch.stack(list(norms.values())).tolist()
    return {f"weight_norm/{name}": v for name, v in zip(norms, values, strict=True)}


class TaskGradientCosine:
    """Per-group cosine of the two task gradients (module docstring). Rank 0 only (after DDP
    averaging every GPU has the same gradient).

    Keeps the gradients (bf16) of the 3 steps (s-3, s-2, s-1) before log step s (0-based) and
    accumulates per-group dot products of each new gradient with the kept ones. When all 4 steps
    are in, it returns the cosines and clears. Positions come from the step number. If a gradient
    in the window is non-finite or a step is missing (a resume or rewind starting mid-window),
    that window is not reported."""

    SPAN = 4

    def __init__(self, groups: dict[str, list], tasks: tuple[str, ...], log_every: int) -> None:
        if len(tasks) != 2:
            raise ValueError(f"task gradient cosine needs multitask (two tasks): {tasks}")
        if log_every < self.SPAN:
            raise ValueError(f"log_every must be at least {self.SPAN}: {log_every}")
        self.log_every = log_every
        self.groups = shared_groups(groups, tasks)
        self.device = next(p for params in groups.values() for p in params).device
        self.reset()

    def reset(self) -> None:
        self.saved: list[tuple[str, list]] = []  # (task, per-group bf16 gradient lists)
        self.sq: list[torch.Tensor] = []  # per step, per-group ‖g‖² (G,)
        self.dots: dict[tuple[int, int], torch.Tensor] = {}  # (i, j) -> per-group dot (G,)
        self.tasks: list[str] = []

    def position(self, step: int) -> int | None:
        """Position of step (0-based) in the window (0-3; 3 is the log step, where done = step + 1
        is a multiple of log_every). None outside the window."""
        before_log = -(step + 1) % self.log_every  # steps left until the log step
        return self.SPAN - 1 - before_log if before_log < self.SPAN else None

    def wants(self, step: int) -> bool:
        """Whether to look at this step's gradient (the log step and the 3 steps before it)."""
        return self.position(step) is not None

    @torch.no_grad()
    def update(self, step: int, task: str, finite: bool) -> dict[str, float] | None:
        """Add the step's (pre-clip) gradient. Returns the metrics once 4 steps are in."""
        position = self.position(step)
        if position is None:
            return None
        if position == 0:
            self.reset()
        if not finite or position != len(self.tasks):
            self.reset()  # do not report this window
            return None
        # unused parameters keep their slot as None (to align with the kept gradients)
        current = {name: [p.grad for p in params] for name, params in self.groups.items()}
        names = list(self.groups)
        index = len(self.tasks)
        self.sq.append(torch.stack([_dot(current[n], current[n], self.device) for n in names]))
        for j, (_, old) in enumerate(self.saved):
            self.dots[(j, index)] = torch.stack(
                [_dot(old[n], current[n], self.device) for n in names]
            )
        self.tasks.append(task)
        if index + 1 < self.SPAN:
            self.saved.append(
                (
                    task,
                    {
                        n: [None if g is None else g.to(torch.bfloat16) for g in current[n]]
                        for n in names
                    },
                )
            )
            return None
        metrics = self._metrics(names)
        self.reset()
        return metrics

    def _metrics(self, names: list[str]) -> dict[str, float]:
        def cos(i: int, j: int) -> torch.Tensor:
            denom = (self.sq[i] * self.sq[j]).sqrt().clamp(min=1e-30)
            return self.dots[(i, j)] / denom

        cross = torch.stack([cos(i, i + 1) for i in range(self.SPAN - 1)]).mean(0)
        rows = [cross] + [cos(i, i + 2) for i in range(self.SPAN - 2)]
        values = torch.stack(rows).tolist()
        out = {f"grad_cos/{n}": v for n, v in zip(names, values[0], strict=True)}
        for i, row in enumerate(values[1:]):
            task = self.tasks[i]
            out.update({f"grad_cos_same/{task}/{n}": v for n, v in zip(names, row, strict=True)})
        return out


def _dot(a: list, b: list, device: torch.device) -> torch.Tensor:
    """Dot product of two tensor lists in the same order (summed in float32). None slots are 0."""
    total = torch.zeros((), device=device)
    for x, y in zip(a, b, strict=True):
        if x is not None and y is not None:
            total = total + torch.dot(x.flatten().float(), y.flatten().float())
    return total


# columns of the per-rank timing record (`RankTiming`)
TIMING_FIELDS = ("fwd_ms", "bwd_ms", "rows", "n_train", "features", "cells")


class RankTiming:
    """Per-GPU forward/backward time and table size per step (module docstring). Used on every
    GPU.

    On CUDA, GPU time is measured with events (read at gather time; reading the loss every step
    waits for the GPU, so the events are done by then); on CPU, with the wall clock. With gradient
    accumulation, micro-batch times and cell counts are summed and row and feature counts
    averaged. Every GPU calls `gather` at the same step."""

    def __init__(self, device: torch.device) -> None:
        self.cuda = device.type == "cuda"
        self.device = device
        self.reset()

    def reset(self) -> None:
        self.records: list[list] = []  # per step: [step, task index, micro records]
        self._micros: list = []

    def mark(self):
        """The current time point (one before forward, after forward and after backward)."""
        if self.cuda:
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            return event
        return time.perf_counter()

    def add_micro(self, start, fwd, end, x: torch.Tensor, n_train: int) -> None:
        b, rows, features = x.shape
        self._micros.append((start, fwd, end, rows, n_train, features, b * rows * features))

    def step_done(self, step: int, task_index: int) -> None:
        self.records.append([step, task_index, self._micros])
        self._micros = []

    def _elapsed(self, a, b) -> float:
        return a.elapsed_time(b) if self.cuda else (b - a) * 1000.0

    def _rows(self) -> torch.Tensor:
        out = []
        for step, task_index, micros in self.records:
            fwd = sum(self._elapsed(s, f) for s, f, *_ in micros)
            bwd = sum(self._elapsed(f, e) for _, f, e, *_ in micros)
            sizes = torch.tensor([m[3:] for m in micros], dtype=torch.float64)
            rows, n_train, features = sizes[:, :3].mean(0).tolist()
            cells = sizes[:, 3].sum().item()
            out.append([step, task_index, fwd, bwd, rows, n_train, features, cells])
        return torch.tensor(out, dtype=torch.float64).reshape(-1, 2 + len(TIMING_FIELDS))

    def gather(self, world_size: int) -> torch.Tensor | None:
        """Records of all GPUs (rank, step, 2 + columns) on rank 0 (None on others). Clears them."""
        mine = self._rows()
        self.reset()
        if world_size == 1:
            return mine[None]
        # NCCL gathers GPU tensors only and takes float64 as is (keeps step numbers exact)
        local = mine.to(self.device)
        out = [torch.empty_like(local) for _ in range(world_size)]
        dist.all_gather(out, local)
        if dist.get_rank() != 0:
            return None
        return torch.stack(out).cpu()


def rank_timing_metrics(gathered: torch.Tensor) -> dict[str, float]:
    """Metrics from the gathered records (rank R, step N, columns).

    - `rank_fwd_ms`, `rank_bwd_ms`: mean over GPUs and steps. `rank_fwd_ms_slowest`: mean forward
      of the slowest GPU per step. `rank_fwd_imbalance`: their ratio (1 is balanced)
    - `rank_cells`, `rank_cells_largest`, `rank_cells_imbalance`: the same for cell counts
    - `rank_fwd_cells_corr`: correlation of forward time and cell count after subtracting the
      per-step GPU mean. Near 1, forward differences between GPUs come from table sizes
    - `rank_bwd_fwd_corr`: the same correlation of backward and forward. Negative: GPUs that
      finish forward early wait longer in backward (all-reduce)
    - `rank_slowest_is_largest`: share of steps where the GPU with the slowest forward also has
      the most cells
    With a single GPU the cross-GPU metrics (correlations, share) are not reported."""
    if gathered.shape[1] == 0:
        return {}
    columns = {name: gathered[..., 2 + i] for i, name in enumerate(TIMING_FIELDS)}
    fwd, bwd, cells = columns["fwd_ms"], columns["bwd_ms"], columns["cells"]
    out = {
        "rank_fwd_ms": fwd.mean().item(),
        "rank_bwd_ms": bwd.mean().item(),
        "rank_fwd_ms_slowest": fwd.max(0).values.mean().item(),
        "rank_cells": cells.mean().item(),
        "rank_cells_largest": cells.max(0).values.mean().item(),
        "rank_rows": columns["rows"].mean().item(),
        "rank_features": columns["features"].mean().item(),
    }
    out["rank_fwd_imbalance"] = out["rank_fwd_ms_slowest"] / max(out["rank_fwd_ms"], 1e-12)
    out["rank_cells_imbalance"] = out["rank_cells_largest"] / max(out["rank_cells"], 1e-12)
    if gathered.shape[0] > 1:
        out["rank_fwd_cells_corr"] = _within_step_corr(fwd, cells)
        out["rank_bwd_fwd_corr"] = _within_step_corr(bwd, fwd)
        out["rank_slowest_is_largest"] = (fwd.argmax(0) == cells.argmax(0)).double().mean().item()
    return {k: v for k, v in out.items() if math.isfinite(v)}


def _within_step_corr(a: torch.Tensor, b: torch.Tensor) -> float:
    """Correlation of two (rank, step) values, minus per-step rank means (removes step effects)."""
    a = a - a.mean(0, keepdim=True)
    b = b - b.mean(0, keepdim=True)
    denom = (a.square().sum() * b.square().sum()).sqrt()
    return (a * b).sum().item() / denom.item() if denom > 0 else math.nan


def write_rank_timing(path: Path, gathered: torch.Tensor, tasks: tuple[str, ...]) -> None:
    """Append the gathered records as one JSON line per step; each column a list in rank order."""
    with path.open("a") as f:
        for n in range(gathered.shape[1]):
            line = {
                "step": int(gathered[0, n, 0]),
                "task": tasks[int(gathered[0, n, 1])],
            }
            for i, name in enumerate(TIMING_FIELDS):
                values = gathered[:, n, 2 + i].tolist()
                line[name] = [round(v, 3) for v in values] if name.endswith("_ms") else values
            f.write(json.dumps(line) + "\n")
