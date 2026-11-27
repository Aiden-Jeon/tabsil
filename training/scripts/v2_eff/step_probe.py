"""Time the parts of a training step that `TABSIL_RANK_TIMING` leaves out, without changing the
training code. It runs `training/scripts/train.py` in this process after wrapping

- the optimizer step (`Muon.step`), the gradient clipping and the per-group gradient norms, each
  between two `torch.cuda.synchronize()` calls,
- the prior DataLoader's `__next__`: the data wait of every GPU (`data_wait_frac` is rank 0's),
- Python's garbage collector (`gc.callbacks`): the pause of every collection, per generation,
- the training loop's own full collection and `gc.freeze()` at its log steps
  (`TrainConfig.gc_freeze`, on by default): their time, apart from the collector's pauses, and
  the number of frozen objects after each freeze,
- the prior workers of every GPU: the passes of each worker's collector and, on Linux, the memory
  that is the worker's own. A worker is forked from its GPU's process and inherits all of its
  objects, and its collector walks those in every full pass unless they were frozen before the
  fork.

The wrapped calls sit where the loop already waits for the GPU (the gradient all-reduce before
them, the loss read after them), so the timing does not move work. The training computation is
unchanged. Every GPU writes one JSON file with a record per optimizer step, and rank 0 logs them
to the MLflow run under `eff_probe/`. `step_breakdown.py` joins them with the rank timing records.

Options (environment variables):

- `TABSIL_EFF_GC`: empty (default) measures the training loop as it is. The other two values
  replace the loop's own freeze, so they turn it off (they set `TABSIL_GC_FREEZE=0`, and stop if
  it is set to on): `freeze:<step>` collects once and calls `gc.freeze()` after that many
  optimizer steps, so that the two halves of the run compare the loop without a freeze and with
  one (the first measurement of the collector's pauses). `off` disables the collector for the
  run.
- `TABSIL_EFF_PREFORK_FREEZE=1`: the GPU's process collects and freezes right before each prior
  DataLoader forks its workers, so that the workers inherit a frozen heap. The training loop does
  not do this; the option measures what it would give.

    torchrun --standalone --nproc_per_node=8 training/scripts/v2_eff/step_probe.py

takes the same arguments and `TABSIL_*` variables as `train.py` (`timing.yaml` in this folder).
"""

from __future__ import annotations

import gc
import json
import multiprocessing
import os
import runpy
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, get_worker_info

from tabsil_train.training import loop, muon

TRAIN_SCRIPT = Path(__file__).resolve().parents[1] / "train.py"
# What a step record holds, in this order (milliseconds, except the first). `gc_ms` is the
# collector's own passes, which stop one GPU at a time. `freeze_ms` is a full collection called
# by the code and the freeze after it, which every GPU makes at the same step; the training loop's
# falls into the record of the step after its log step. Records made before the loop froze have no
# `freeze_ms`, and their `gc_ms` holds the probe's one collection
FIELDS = ("wall_s", "opt_ms", "clip_ms", "norms_ms", "wait_ms", "gc_ms", "freeze_ms")
# The folder where the prior workers of this process write their records (`_worker_init`). Not a
# `TABSIL_*` name: those are the run's settings, and `train.py` tags the run with them
WORKER_DIR = "EFF_PROBE_WORKER_DIR"


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
        # After every `gc.freeze()` (the training loop's at its log steps, or this probe's):
        # (optimizer steps done, frozen objects, seconds of the collection and the freeze). Frozen
        # objects that only grow would be a leak
        self.freezes: list[tuple[int, int, float]] = []
        # Every time a prior DataLoader forks its workers: (optimizer steps done, objects the
        # collector tracks and walks, frozen objects). The workers inherit both
        self.forks: list[tuple[int, int, int]] = []
        self.prefork_freeze = False  # `TABSIL_EFF_PREFORK_FREEZE`
        self.called_collection = False  # inside a full collection called by the code
        self.collection_seconds = 0.0  # the one before the next freeze
        self.counting_seconds = 0.0  # spent counting frozen objects: taken off this probe's clock

    def add(self, name: str, seconds: float) -> None:
        self.current[name] += seconds * 1000.0

    def close_step(self) -> None:
        now = time.perf_counter() - self.start - self.counting_seconds
        self.steps.append([now, *(self.current[name] for name in FIELDS[1:])])
        self.current = dict.fromkeys(FIELDS[1:], 0.0)

    def on_gc(self, phase: str, info: dict) -> None:
        if phase == "start":
            self.gc_start = time.perf_counter()
            return
        if self.called_collection:  # timed as `freeze_ms` (`_wrap_collector`)
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
    """(`measure`, `freeze` or `off`, the optimizer step to freeze after) from `TABSIL_EFF_GC`.
    `freeze` and `off` stand in for the training loop's own freeze, so they turn it off for
    `train.py` (`TABSIL_GC_FREEZE=0`) and stop if it was asked for."""
    value = os.environ.get("TABSIL_EFF_GC", "").strip().lower()
    if not value:
        return "measure", 0
    if value == "off":
        mode, step = "off", 0
    elif value.startswith("freeze:"):
        mode, step = "freeze", int(value.split(":", 1)[1])
    else:
        raise SystemExit(f"TABSIL_EFF_GC={value!r} is not empty, off or freeze:<step>")
    loop_freeze = os.environ.get("TABSIL_GC_FREEZE", "").strip().lower()
    if loop_freeze in ("1", "true", "yes", "on"):
        raise SystemExit(
            f"TABSIL_EFF_GC={value} measures the loop without its own freeze: unset"
            " TABSIL_GC_FREEZE or set it to 0"
        )
    os.environ["TABSIL_GC_FREEZE"] = "0"
    return mode, step


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


def _wrap_collector() -> None:
    """Keep the full collections that the code calls apart from the collector's own passes. The
    training loop calls `gc.collect()` and `gc.freeze()` at its log steps on every GPU at once,
    and `freeze:<step>` here does the same once. Their time goes to `freeze_ms` instead of
    `gc_ms`, and each freeze is recorded with the number of frozen objects after it. Counting
    walks the frozen objects (about 25 ms per million), so that time is taken off this probe's
    clock; the run's own `sec_per_step` still holds it."""
    collect, freeze = gc.collect, gc.freeze

    def timed_collect(generation: int = 2) -> int:
        if generation != 2:  # a partial collection (torch.compile calls one after compiling)
            return collect(generation)
        PROBE.called_collection = True
        start = time.perf_counter()
        try:
            return collect()
        finally:
            PROBE.called_collection = False
            PROBE.collection_seconds = time.perf_counter() - start
            PROBE.add("freeze_ms", PROBE.collection_seconds)

    def counted_freeze() -> None:
        start = time.perf_counter()
        freeze()
        seconds = time.perf_counter() - start
        PROBE.add("freeze_ms", seconds)
        frozen = gc.get_freeze_count()
        PROBE.counting_seconds += time.perf_counter() - start - seconds
        total = round(PROBE.collection_seconds + seconds, 4)
        PROBE.freezes.append((len(PROBE.steps), frozen, total))
        PROBE.collection_seconds = 0.0

    gc.collect, gc.freeze = timed_collect, counted_freeze


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


def _own_memory(source: str = "/proc/self/smaps_rollup") -> dict | None:
    """The memory of this process in megabytes (Linux, `/proc/self/smaps_rollup`). `rss` holds
    the pages it still shares with the process it was forked from. `private` is its own: what it
    allocated, and the inherited pages it wrote to, such as those of the objects its collector
    walked."""
    try:
        lines = Path(source).read_text().splitlines()
    except OSError:
        return None
    kb = {}
    for line in lines:
        name, _, rest = line.partition(":")
        parts = rest.split()
        if len(parts) == 2 and parts[1] == "kB":
            kb[name] = int(parts[0])
    private = kb.get("Private_Dirty", 0) + kb.get("Private_Clean", 0)
    megabytes = {"rss": kb.get("Rss", 0), "pss": kb.get("Pss", 0), "private": private}
    return {name: round(value / 1024, 1) for name, value in megabytes.items()}


class _WorkerProbe:
    """The collector of one prior worker (`_worker_init`): its passes per generation, its CPU
    time and its memory. The record is rewritten after every full pass and, after another
    collection, at most every 15 s; the memory is read at most that often too."""

    def __init__(self, path: Path, header: dict) -> None:
        self.path = path
        self.header = header
        self.start = time.perf_counter()
        self.stats = {g: [0, 0.0, 0.0] for g in range(3)}
        self.gc_start = 0.0
        self.written = 0.0
        self.memory_read = self.start
        self.memory_start = self.memory = _own_memory()
        self.write()

    def on_gc(self, phase: str, info: dict) -> None:
        if phase == "start":
            self.gc_start = time.perf_counter()
            return
        now = time.perf_counter()
        seconds = now - self.gc_start
        stats = self.stats[info["generation"]]
        stats[0] += 1
        stats[1] += seconds
        stats[2] = max(stats[2], seconds)
        if info["generation"] == 2 or now - self.written > 15:
            self.write()

    def write(self) -> None:
        self.written = time.perf_counter()
        if self.written - self.memory_read > 15:
            self.memory_read, self.memory = self.written, _own_memory()
        record = {
            **self.header,
            "seconds": round(self.written - self.start, 1),
            "cpu_seconds": round(time.process_time(), 2),
            "gc": {
                str(generation): {
                    "collections": n,
                    "seconds": round(total, 4),
                    "longest": round(top, 4),
                }
                for generation, (n, total, top) in self.stats.items()
            },
            "memory_start_mb": self.memory_start,
            "memory_mb": self.memory,
        }
        scratch = self.path.with_suffix(".tmp")
        try:
            scratch.write_text(json.dumps(record))
            scratch.replace(self.path)
        except OSError:  # the folder is gone (the run is ending): keep the last record
            pass


def _worker_init(worker_id: int) -> None:
    """The `worker_init_fn` of the prior DataLoaders: in place of the callback this worker
    inherited from its GPU's process, record the worker's own collector (`_WorkerProbe`)."""
    gc.callbacks[:] = [c for c in gc.callbacks if getattr(c, "__self__", None) is not PROBE]
    folder = os.environ.get(WORKER_DIR)
    if not folder:
        return
    try:
        rank = int(os.environ.get("RANK", "0"))
        task = getattr(get_worker_info().dataset, "task", None)
        header = {
            "rank": rank,
            "task": task,
            "worker": worker_id,
            "pid": os.getpid(),
            # what it inherited: the frozen objects, and the collector's counts per generation.
            # Counting the objects it tracks would write to every one of them
            "frozen_at_start": gc.get_freeze_count(),
            "gc_count_at_start": gc.get_count(),
        }
        name = f"eff_probe_worker_rank{rank}_{task}_{worker_id}_{os.getpid()}.json"
        gc.callbacks.append(_WorkerProbe(Path(folder) / name, header).on_gc)
    except Exception as error:  # the record is an extra: a worker must not stop for it
        print(f"EFF_PROBE no worker record: {type(error).__name__}: {error}", flush=True)


def _worker_records(folder: Path, rank: int) -> list[dict]:
    """The records of this GPU's prior workers, as each one last wrote it."""
    out = []
    for path in sorted(folder.glob(f"eff_probe_worker_rank{rank}_*.json")):
        try:
            out.append(json.loads(path.read_text()))
        except (OSError, ValueError):
            continue
    return out


class _TimedLoader(DataLoader):
    """The training loop's prior DataLoader with its data wait timed and its workers recorded
    (`_worker_init`). With `TABSIL_EFF_PREFORK_FREEZE=1` the process collects and freezes right
    before the workers are forked."""

    def __init__(self, *args, **kwargs) -> None:
        # Only forked workers inherit the process's objects (and take a function that is not
        # importable by name: `train.py` runs as `__main__` here)
        self.forked_workers = bool(kwargs.get("num_workers")) and (
            multiprocessing.get_start_method() == "fork"
        )
        if self.forked_workers:
            kwargs.setdefault("worker_init_fn", _worker_init)
        super().__init__(*args, **kwargs)

    def __iter__(self):
        if self.forked_workers:
            if PROBE.prefork_freeze:
                gc.collect()
                gc.freeze()
            start = time.perf_counter()
            PROBE.forks.append((len(PROBE.steps), len(gc.get_objects()), gc.get_freeze_count()))
            PROBE.counting_seconds += time.perf_counter() - start
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
        "freezes": PROBE.freezes,
        "prefork_freeze": PROBE.prefork_freeze,
        "forks": PROBE.forks,
        "workers": _worker_records(path.parent, rank),
        # At exit: the objects the collector tracks and walks, and the frozen ones it skips.
        # `train` unfreezes what the loop froze, so a run with the loop's freeze has 0 frozen
        # here and its frozen counts in `freezes`. The two add up to all tracked objects
        "gc_objects": len(gc.get_objects()),
        "gc_frozen": gc.get_freeze_count(),
        "compilations": _compilations(),
    }
    path.write_text(json.dumps(record))
    summary = {
        k: record[k] for k in ("rank", "gc", "gc_objects", "gc_frozen", "frozen_at", "freezes")
    }
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
    _wrap_collector()
    _wrap_optimizer_step(mode, freeze_at)
    torch.nn.utils.clip_grad_norm_ = _timed("clip_ms", torch.nn.utils.clip_grad_norm_)
    loop.group_norms = _timed("norms_ms", loop.group_norms)
    loop.DataLoader = _TimedLoader

    rank = int(os.environ.get("RANK", "0"))
    out_dir = Path(os.environ.get("TABSIL_OUT_DIR", "tabsil_run"))
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob(f"eff_probe_worker_rank{rank}_*"):  # of an earlier run in the folder
        stale.unlink(missing_ok=True)
    os.environ[WORKER_DIR] = str(out_dir.resolve())
    prefork = os.environ.get("TABSIL_EFF_PREFORK_FREEZE", "").strip().lower()
    PROBE.prefork_freeze = prefork in ("1", "true", "yes", "on")
    run_id = os.environ.get("MLFLOW_RUN_ID")
    try:
        runpy.run_path(str(TRAIN_SCRIPT), run_name="__main__")
    finally:
        _write(out_dir / f"eff_probe_rank{rank}.json", rank, mode)
    if rank == 0:
        _log_to_mlflow(out_dir, int(os.environ.get("WORLD_SIZE", "1")), run_id)


if __name__ == "__main__":
    # train.py restarts itself to fix the hash seed; do it here so that the probe is not lost
    if "PYTHONHASHSEED" not in os.environ:
        os.environ["PYTHONHASHSEED"] = "0"
        os.execv(sys.executable, [sys.executable, *sys.orig_argv[1:]])
    main()
