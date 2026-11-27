"""Time the parts of a training step that `TABSIL_RANK_TIMING` leaves out, without changing the
training code. It runs `training/scripts/train.py` in this process after wrapping

- the optimizer step (`Muon.step`), the gradient clipping and the per-group gradient norms, each
  between two `torch.cuda.synchronize()` calls,
- the prior DataLoader's `__next__`: the data wait of every GPU (`data_wait_frac` is rank 0's),
- Python's garbage collector (`gc.callbacks`): the pause of every collection, per generation.

The wrapped calls sit where the loop already waits for the GPU (the gradient all-reduce before
them, the loss read after them), so the timing does not move work. The training computation is
unchanged. Every GPU writes one JSON file with a record per optimizer step, and rank 0 logs them
to the MLflow run under `eff_probe/`. `step_breakdown.py` joins them with the rank timing records.

Options (environment variables):

- `TABSIL_EFF_GC`: empty (default) only measures. `freeze:<step>` collects once and calls
  `gc.freeze()` after that many optimizer steps, so that later full collections skip the objects
  that exist by then (the model, the optimizer and the compiled graphs). `off` disables the
  collector for the run.

    torchrun --standalone --nproc_per_node=8 training/scripts/v2_eff/step_probe.py

takes the same arguments and `TABSIL_*` variables as `train.py` (`timing.yaml` in this folder).
"""

from __future__ import annotations

import gc
import json
import os
import runpy
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from tabsil_train.training import loop, muon

TRAIN_SCRIPT = Path(__file__).resolve().parents[1] / "train.py"
# what a step record holds, in this order (milliseconds, except the first)
FIELDS = ("wall_s", "opt_ms", "clip_ms", "norms_ms", "wait_ms", "gc_ms")


class Probe:
    """The timings of one process (one GPU). A record is closed at the end of each optimizer
    step, so the data wait and the collector pauses of a step's micro batches fall into it."""

    def __init__(self) -> None:
        self.start = time.perf_counter()
        self.steps: list[list[float]] = []
        self.current = dict.fromkeys(FIELDS[1:], 0.0)
        self.gc_start = 0.0
        # per generation: collections, seconds in total, the longest one
        self.gc_stats = {g: [0, 0.0, 0.0] for g in range(3)}
        self.gc_events: list[tuple[int, int, float]] = []  # (step, generation, seconds) over 50 ms
        self.frozen_at: int | None = None

    def add(self, name: str, seconds: float) -> None:
        self.current[name] += seconds * 1000.0

    def close_step(self) -> None:
        now = time.perf_counter() - self.start
        self.steps.append([now, *(self.current[name] for name in FIELDS[1:])])
        self.current = dict.fromkeys(FIELDS[1:], 0.0)

    def on_gc(self, phase: str, info: dict) -> None:
        if phase == "start":
            self.gc_start = time.perf_counter()
            return
        seconds = time.perf_counter() - self.gc_start
        stats = self.gc_stats[info["generation"]]
        stats[0] += 1
        stats[1] += seconds
        stats[2] = max(stats[2], seconds)
        self.add("gc_ms", seconds)
        if seconds > 0.05:
            self.gc_events.append((len(self.steps), info["generation"], round(seconds, 4)))


PROBE = Probe()


def _synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _timed(name: str, function):
    """`function` with its wall time, GPU work included, added to the current step's `name`."""

    def wrapper(*args, **kwargs):
        _synchronize()
        start = time.perf_counter()
        out = function(*args, **kwargs)
        _synchronize()
        PROBE.add(name, time.perf_counter() - start)
        return out

    return wrapper


def _gc_mode() -> tuple[str, int]:
    """(`measure`, `freeze` or `off`, the optimizer step to freeze after) from `TABSIL_EFF_GC`."""
    value = os.environ.get("TABSIL_EFF_GC", "").strip().lower()
    if not value:
        return "measure", 0
    if value == "off":
        return "off", 0
    if value.startswith("freeze:"):
        return "freeze", int(value.split(":", 1)[1])
    raise SystemExit(f"TABSIL_EFF_GC={value!r} is not empty, off or freeze:<step>")


def _wrap_optimizer_step(mode: str, freeze_at: int) -> None:
    step = muon.Muon.step

    def timed_step(self, closure=None):
        _synchronize()
        start = time.perf_counter()
        out = step(self, closure)
        _synchronize()
        PROBE.add("opt_ms", time.perf_counter() - start)
        PROBE.close_step()
        if mode == "freeze" and len(PROBE.steps) == freeze_at:
            gc.collect()
            gc.freeze()
            PROBE.frozen_at = freeze_at
        return out

    muon.Muon.step = timed_step


class _TimedIterator:
    def __init__(self, iterator) -> None:
        self.iterator = iterator

    def __iter__(self):
        return self

    def __next__(self):
        start = time.perf_counter()
        batch = next(self.iterator)
        PROBE.add("wait_ms", time.perf_counter() - start)
        return batch


class _TimedLoader(DataLoader):
    def __iter__(self):
        return _TimedIterator(super().__iter__())


def _compilations() -> list[dict]:
    """The latest compilations torch.compile kept (at most 64): when each started, in seconds
    since this process started, how long it took and the function. Empty if this torch version
    does not expose them."""
    try:
        from torch._dynamo.utils import get_compilation_metrics

        out = []
        offset = time.time() - (time.perf_counter() - PROBE.start)  # Unix time of the start
        for metric in get_compilation_metrics():
            start = getattr(metric, "start_time", None)
            if start is None and getattr(metric, "start_time_us", None) is not None:
                start = metric.start_time_us / 1e6
            out.append(
                {
                    "start_s": None if start is None else round(start - offset, 2),
                    "seconds": getattr(metric, "entire_frame_compile_time_s", None),
                    "function": getattr(metric, "co_name", None),
                    "recompile_reason": str(getattr(metric, "recompile_reason", None))[:300],
                }
            )
        return out
    except Exception as error:  # a private API: report instead of failing the timing run
        return [{"error": f"{type(error).__name__}: {error}"}]


def _write(path: Path, rank: int, mode: str) -> None:
    record = {
        "rank": rank,
        "world_size": int(os.environ.get("WORLD_SIZE", "1")),
        "gc_mode": mode,
        "frozen_at": PROBE.frozen_at,
        "fields": FIELDS,
        "steps": [[round(v, 4) for v in step] for step in PROBE.steps],
        "gc": {
            str(generation): {
                "collections": n,
                "seconds": round(total, 4),
                "longest": round(top, 4),
            }
            for generation, (n, total, top) in PROBE.gc_stats.items()
        },
        "gc_events": PROBE.gc_events,
        "gc_objects": len(gc.get_objects()),
        "gc_frozen": gc.get_freeze_count(),
        "compilations": _compilations(),
    }
    path.write_text(json.dumps(record))
    summary = {k: record[k] for k in ("rank", "gc", "gc_objects", "gc_frozen", "frozen_at")}
    print("EFF_PROBE " + json.dumps(summary), flush=True)


def _log_to_mlflow(out_dir: Path, world_size: int, run_id: str | None) -> None:
    """Rank 0, after `train.py` closed the run: wait for the other GPUs' files (one node shares
    the folder) and log all of them to the run. `run_id` is the platform's run, read before
    `train.py` ran: `mlflow.end_run()` removes `MLFLOW_RUN_ID` from the environment."""
    if not run_id:
        return
    import mlflow

    deadline = time.time() + 60
    while len(list(out_dir.glob("eff_probe_rank*.json"))) < world_size and time.time() < deadline:
        time.sleep(1)
    with mlflow.start_run(run_id=run_id):
        for path in sorted(out_dir.glob("eff_probe_rank*.json")):
            mlflow.log_artifact(str(path), "eff_probe")


def main() -> None:
    mode, freeze_at = _gc_mode()
    if mode == "off":
        gc.disable()
    gc.callbacks.append(PROBE.on_gc)
    _wrap_optimizer_step(mode, freeze_at)
    torch.nn.utils.clip_grad_norm_ = _timed("clip_ms", torch.nn.utils.clip_grad_norm_)
    loop.group_norms = _timed("norms_ms", loop.group_norms)
    loop.DataLoader = _TimedLoader

    rank = int(os.environ.get("RANK", "0"))
    out_dir = Path(os.environ.get("TABSIL_OUT_DIR", "tabsil_run"))
    run_id = os.environ.get("MLFLOW_RUN_ID")
    try:
        runpy.run_path(str(TRAIN_SCRIPT), run_name="__main__")
    finally:
        out_dir.mkdir(parents=True, exist_ok=True)
        _write(out_dir / f"eff_probe_rank{rank}.json", rank, mode)
    if rank == 0:
        _log_to_mlflow(out_dir, int(os.environ.get("WORLD_SIZE", "1")), run_id)


if __name__ == "__main__":
    # train.py restarts itself to fix the hash seed; do it here so that the probe is not lost
    if "PYTHONHASHSEED" not in os.environ:
        os.environ["PYTHONHASHSEED"] = "0"
        os.execv(sys.executable, [sys.executable, *sys.orig_argv[1:]])
    main()
