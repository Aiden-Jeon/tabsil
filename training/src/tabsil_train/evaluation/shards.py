"""`--shards` of the ADBench and impute suites: jobs run as worker processes of `eval.py`, one
per free GPU.

- `shard_devices`: the GPUs of `--shards N`
- `run_shard_queue`: starts the most expensive job left on every free GPU (and processes without
  a GPU next to them), merges a job's results when it ends and prints a heartbeat
- `strip_flags`: the parent's arguments without the ones a worker gets its own value for
- `clock`: elapsed seconds as h:mm:ss

The TabArena / BeyondArena `--shards` (`run_queue` in `eval.py`) is another scheduler: its workers
stay alive and take datasets from stdin.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

import torch


def strip_flags(argv: list[str], flags: dict[str, int]) -> list[str]:
    """Remove `flags` (name: value count) from `argv`, including the `--name=value` form."""
    out, i = [], 0
    while i < len(argv):
        name = argv[i].split("=", 1)[0]
        if name in flags:
            i += 1 if "=" in argv[i] else 1 + flags[name]
            continue
        out.append(argv[i])
        i += 1
    return out


def clock(seconds: float) -> str:
    """Elapsed `seconds` as h:mm:ss."""
    seconds = int(seconds)
    return f"{seconds // 3600}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"


def shard_devices(args: argparse.Namespace) -> list[str]:
    """The GPUs of `--shards N`: the first N of CUDA_VISIBLE_DEVICES (it lists every visible
    GPU), or 0..N-1. N above the GPU count is an error."""
    n_gpus = torch.cuda.device_count()
    if n_gpus and args.shards > n_gpus:
        raise SystemExit(f"--shards {args.shards} exceeds the GPU count {n_gpus}.")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    return (visible.split(",") if visible else [str(i) for i in range(args.shards)])[: args.shards]


def run_shard_queue(
    label: str,
    jobs: list[tuple],
    job_command: Callable[[tuple], tuple[str, list[str]]],
    on_done: Callable[[tuple, int, Path], bool],
    devices: list[str],
    script: str | Path,
    base: list[str],
    env: dict[str, str],
    log_dir: Path,
    heartbeat_minutes: float,
    side: list[tuple[tuple, str, list[str]]] = (),
    progress: Callable[[Path], str] | None = None,
) -> None:
    """Run `jobs` (most expensive first) as worker processes (`python <script>` with `base` and
    the job's arguments, `job_command(job)` -> (name, argv); `script` is the path of `eval.py`),
    one per free GPU of `devices` (CUDA_VISIBLE_DEVICES = that GPU), and the `side` processes
    ((job, name, argv), no GPU) at the same time. A worker writes to `log_dir/<name>/` and logs to
    `log_dir/<name>.log`. When it ends, `on_done(job, returncode, job_dir)` merges its results and
    returns whether it succeeded. Prints each finished or failed job and a heartbeat every
    `heartbeat_minutes` (`progress(job_dir)` adds a per-job note). If the queue stops midway (an
    exception from `on_done` or from starting a worker, Ctrl-C), the workers still running are
    killed; the rows they wrote stay in their folders."""
    log_dir.mkdir(parents=True, exist_ok=True)

    def start(name: str, argv: list[str], visible_devices: str):
        job_dir = log_dir / name
        cmd = [sys.executable, str(script), *base, *argv, "--out-dir", str(job_dir)]
        log = open(log_dir / f"{name}.log", "w")  # noqa: SIM115 (closed when done)
        proc = subprocess.Popen(
            cmd, stdout=log, stderr=subprocess.STDOUT,
            env={**env, "CUDA_VISIBLE_DEVICES": visible_devices},
        )  # fmt: skip
        return proc, job_dir, time.time(), log

    def finish(job, proc, job_dir, started, log) -> None:
        log.close()
        if on_done(job, proc.returncode, job_dir):
            print(f"{label} shards: {job} done in {clock(time.time() - started)}", flush=True)
        else:
            print(f"{label} shards: {job} FAILED (exit {proc.returncode})", flush=True)

    def note(job_dir: Path) -> str:
        return f" ({progress(job_dir)})" if progress else ""

    jobs = list(jobs)
    running: dict[str, tuple] = {}  # device -> (job, process, job dir, start, log file)
    side_running: dict[str, tuple] = {}  # name -> the same
    t0 = last = time.time()
    try:
        for job, name, argv in side:
            side_running[name] = (job, *start(name, argv, ""))
        while jobs or running or side_running:
            for dev in devices:
                if dev not in running and jobs:
                    job = jobs.pop(0)
                    name, argv = job_command(job)
                    running[dev] = (job, *start(name, argv, dev))
            time.sleep(5)
            for table in (running, side_running):
                for key, (job, proc, job_dir, started, log) in list(table.items()):
                    if proc.poll() is not None:
                        del table[key]
                        finish(job, proc, job_dir, started, log)
            if time.time() - last >= heartbeat_minutes * 60:
                last = time.time()
                busy = ", ".join(
                    f"GPU {dev}: {j[0]} {clock(last - j[3])}{note(j[2])}"
                    for dev, j in running.items()
                )
                others = "".join(
                    f"; {name} {clock(last - j[3])}{note(j[2])}" for name, j in side_running.items()
                )
                print(f"{label} shards [{clock(last - t0)}] {len(jobs)} queued; {busy}{others}",
                      flush=True)  # fmt: skip
    finally:
        # if the queue stops midway (an exception, Ctrl-C), do not leave workers holding GPUs
        for _, proc, _, _, log in (*running.values(), *side_running.values()):
            if proc.poll() is None:
                proc.kill()
                proc.wait()
            log.close()
