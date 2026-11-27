"""Simulate the training loop's data queue from measured table generation times
(`prior_timing.py`'s CSV) and report the step time it predicts, for the loop's settings and for
changes to them. No table is generated and no model runs: this only replays the measured times.

What is modelled (`tabsil_train.training.loop`, torch `DataLoader`):

- Every GPU has one DataLoader per task with `workers` processes and `prefetch` requests per
  worker outstanding. Request i goes to worker i mod workers, and a worker serves its requests
  in order, one table each. A new request is sent each time the loop takes a table.
- The loader hands tables over in request order (`in_order=True`, torch's default), so the loop
  waits for table i even when later tables are ready. With `--any-order` it takes whichever
  finished first (`DataLoader(in_order=False)`).
- A step takes `micro` tables of one task on every GPU (the tasks alternate per step), computes
  for `compute` seconds per table, and ends when the slowest GPU is done (the gradient
  all-reduce), plus `overhead` seconds.

Generation times are drawn with replacement from the measured tables of the prior and task.
`compute` and `overhead` are constants, so the simulated step time without any data wait is
`micro * compute + overhead`; the run-to-run spread of the GPU time is not modelled.

    python queue_sim.py prior_timing.csv.gz --prior graph_scm_realism --compute 0.21 \
        --overhead 1.6 --workers 8 --prefetch 4
"""

from __future__ import annotations

import argparse
import csv
import gzip
import heapq
from pathlib import Path

import numpy as np

TASKS = ("classification", "regression")


def load_times(path: str | Path, prior: str) -> dict[str, np.ndarray]:
    """Generation seconds per task of `prior` from a `prior_timing.py` CSV (several files are
    concatenated by the caller)."""
    opener = gzip.open if str(path).endswith(".gz") else open
    out: dict[str, list[float]] = {task: [] for task in TASKS}
    with opener(path, "rt", newline="") as f:
        for row in csv.DictReader(f):
            if row["prior"] == prior:
                out[row["task"]].append(float(row["seconds"]))
    return {task: np.array(values) for task, values in out.items()}


class Sampler:
    """Generation times drawn with replacement from the measured ones (drawn in blocks)."""

    def __init__(self, times: np.ndarray, rng: np.random.Generator) -> None:
        self.times, self.rng = times, rng
        self.block: list[float] = []

    def __call__(self) -> float:
        if not self.block:
            self.block = self.rng.choice(self.times, size=65_536).tolist()
        return self.block.pop()


class Loader:
    """One DataLoader of one GPU and task (module docstring)."""

    def __init__(self, draw: Sampler, workers: int, prefetch: int, in_order: bool) -> None:
        self.draw = draw
        self.workers, self.prefetch, self.in_order = workers, prefetch, in_order
        self.free_at = [0.0] * workers  # when each worker finishes its last accepted request
        self.pending = [0] * workers  # requests a worker has not delivered yet
        self.next_worker = 0
        self.sent = 0  # requests sent so far
        self.taken = 0  # tables handed to the loop so far
        self.finish: dict[int, float] = {}  # request -> finish time (request order)
        self.ready: list[tuple[float, int]] = []  # (finish time, worker) heap (any order)
        for _ in range(workers * prefetch):
            self._send(0.0)

    def _send(self, now: float) -> None:
        # torch sends to the next worker in turn; with in_order=False it skips a worker that
        # already holds `prefetch` requests, so a stuck worker does not collect more
        w = self.next_worker
        if not self.in_order:
            while self.pending[w] >= self.prefetch:
                w = (w + 1) % self.workers
        self.next_worker = (w + 1) % self.workers
        done = max(self.free_at[w], now) + self.draw()
        self.free_at[w] = done
        self.pending[w] += 1
        if self.in_order:
            self.finish[self.sent] = done
        else:
            heapq.heappush(self.ready, (done, w))
        self.sent += 1

    def take(self, now: float) -> float:
        """The time the loop gets its next table when it asks at `now`."""
        if self.in_order:
            done, w = self.finish.pop(self.taken), self.taken % self.workers
        else:
            done, w = heapq.heappop(self.ready)
        self.pending[w] -= 1
        self.taken += 1
        got = max(now, done)
        self._send(got)
        return got


def simulate(
    times: dict[str, np.ndarray],
    *,
    gpus: int,
    steps: int,
    workers: int,
    prefetch: int,
    in_order: bool,
    micro: int,
    compute: float,
    overhead: float,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    draws = {task: Sampler(times[task], rng) for task in TASKS}
    loaders = [
        [Loader(draws[task], workers, prefetch, in_order) for task in TASKS] for _ in range(gpus)
    ]
    waited, stalls = 0.0, 0
    step_seconds = np.empty(steps)
    now = 60.0  # the workers fill their queues while the model loads and compiles
    for step in range(steps):
        start, ends, step_wait = now, [], 0.0
        for gpu in range(gpus):
            loader, t = loaders[gpu][step % len(TASKS)], start
            for _ in range(micro):
                got = loader.take(t)
                step_wait = max(step_wait, got - t)
                waited += got - t
                t = got + compute
            ends.append(t)
        now = max(ends) + overhead
        stalls += step_wait > 1.0
        step_seconds[step] = now - start
    windows = step_seconds[: steps // 100 * 100].reshape(-1, 100).mean(1)
    return {
        "sec_per_step": float(step_seconds.mean()),
        "no_wait_sec_per_step": micro * compute + overhead,
        "window_p10": float(np.quantile(windows, 0.1)),
        "window_p90": float(np.quantile(windows, 0.9)),
        "steps_with_a_wait_over_1s": stalls / steps,
        "wait_frac_per_gpu": waited / gpus / float(step_seconds.sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("csv", nargs="+", help="prior_timing.csv(.gz) files")
    parser.add_argument("--prior", required=True)
    parser.add_argument("--gpus", type=int, default=32)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--workers", type=int, nargs="+", default=[8], help="per GPU and task")
    parser.add_argument("--prefetch", type=int, nargs="+", default=[4], help="per worker")
    parser.add_argument("--any-order", action="store_true", help="also DataLoader(in_order=False)")
    parser.add_argument("--micro", type=int, default=8, help="tables per GPU and step")
    parser.add_argument("--compute", type=float, required=True, help="GPU seconds per table")
    parser.add_argument("--overhead", type=float, required=True, help="seconds per step")
    parser.add_argument("--cap", type=float, help="cut generation times at this many seconds")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    times = {task: np.empty(0) for task in TASKS}
    for path in args.csv:
        for task, values in load_times(path, args.prior).items():
            times[task] = np.concatenate([times[task], values])
    for task, values in times.items():
        if not len(values):
            raise SystemExit(f"no {task} tables of {args.prior} in {args.csv}")
        if args.cap:
            times[task] = np.minimum(values, args.cap)
        print(f"{args.prior} {task}: {len(values)} tables, mean {values.mean():.2f} s")
    print("| Workers | Prefetch | Order | s / step | Windows p10-p90 | Steps waiting > 1 s |")
    print("|---|---|---|---|---|---|")
    for workers in args.workers:
        for prefetch in args.prefetch:
            for in_order in (True, False) if args.any_order else (True,):
                out = simulate(
                    times,
                    gpus=args.gpus,
                    steps=args.steps,
                    workers=workers,
                    prefetch=prefetch,
                    in_order=in_order,
                    micro=args.micro,
                    compute=args.compute,
                    overhead=args.overhead,
                    seed=args.seed,
                )
                print(
                    f"| {workers} | {prefetch} | {'request' if in_order else 'any'}"
                    f" | {out['sec_per_step']:.2f} | {out['window_p10']:.2f}-"
                    f"{out['window_p90']:.2f} | {out['steps_with_a_wait_over_1s']:.1%} |"
                )


if __name__ == "__main__":
    main()
