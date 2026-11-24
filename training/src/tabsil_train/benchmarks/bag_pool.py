"""Fit and predict bagging bags in worker processes, one per GPU (the eval script's
`--bag-gpus`).

`--shards` splits datasets across GPUs, so the largest dataset runs its k bags one after another
on a single GPU while the other GPUs sit idle. Here the k bags of one split are sent to the
workers at once and their predictions are collected in parallel. Datasets still run in sequence,
but every split uses all GPUs.

- A worker is a `spawn`ed process that sees a single GPU via `CUDA_VISIBLE_DEVICES`. It loads
  `script` (a file path), calls `builder(argv)` once to build the estimator factory
  `make(problem_type, categorical)` (so the model is loaded once), and keeps fitted estimators by
  key
- Bag i goes to worker i % n. With more bags than workers, they are sent in rounds of n
- An exception in fit/predict inside a worker fails only that split (same as in-process runs).
  If a worker process dies, the evaluation stops (`SystemExit`, so the remaining splits do not
  silently become imputed)
"""

from __future__ import annotations

import importlib.util
import itertools
import multiprocessing as mp
import os
import traceback
from collections.abc import Sequence
from typing import Any

import numpy as np


def _worker(script: str, builder: str, argv: list[str], conn) -> None:
    spec = importlib.util.spec_from_file_location("_tabsil_bag_worker_script", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    make = getattr(module, builder)(argv)
    fitted: dict[int, Any] = {}
    while True:
        try:
            msg = conn.recv()
        except EOFError:
            return
        op = msg[0]
        if op == "stop":
            return
        if op == "drop":
            for key in msg[1]:
                fitted.pop(key, None)
            continue
        try:
            if op == "fit":
                _, key, problem_type, categorical, X, y, groups = msg
                kwargs = {} if groups is None else {"groups": groups}
                estimator = make(problem_type, categorical).fit(X, y, **kwargs)
                fitted[key] = estimator
                classes = getattr(estimator, "classes_", None)
                conn.send(("ok", None if classes is None else np.asarray(classes)))
            elif op == "predict":
                _, key, X, proba = msg
                estimator = fitted[key]
                out = estimator.predict_proba(X) if proba else estimator.predict(X)
                conn.send(("ok", np.asarray(out)))
            else:
                conn.send(("error", f"unknown request: {op}"))
        except Exception:  # noqa: BLE001 (report failure for this split only)
            conn.send(("error", traceback.format_exc()))


class BagPool:
    """One worker per GPU (`devices`, values for `CUDA_VISIBLE_DEVICES`). `argv[i]` is passed to
    worker i's builder, and `threads` is each worker's CPU thread count."""

    def __init__(
        self,
        script: str,
        builder: str,
        argv: Sequence[list[str]],
        devices: Sequence[str],
        threads: int = 1,
    ) -> None:
        if len(argv) != len(devices) or not devices:
            raise ValueError("give one argv and one GPU per worker")
        ctx = mp.get_context("spawn")
        self.workers = []
        self._keys = itertools.count()
        for dev, args in zip(devices, argv, strict=True):
            parent, child = ctx.Pipe()
            # a spawned process inherits the environment at start time
            saved = {k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS")}
            os.environ.update(CUDA_VISIBLE_DEVICES=str(dev), OMP_NUM_THREADS=str(threads))
            try:
                process = ctx.Process(
                    target=_worker, args=(script, builder, list(args), child), daemon=True
                )
                process.start()
            finally:
                for k, v in saved.items():
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v
            child.close()
            self.workers.append((process, parent))

    def __len__(self) -> int:
        return len(self.workers)

    def _recv(self, i: int):
        process, conn = self.workers[i]
        try:
            status, value = conn.recv()
        except (EOFError, OSError) as e:
            raise SystemExit(f"bag worker {i} died (exit code {process.exitcode}): {e!r}") from e
        if status != "ok":
            raise RuntimeError(f"bag worker {i}:\n{value}")
        return value

    def _send(self, i: int, msg) -> None:
        process, conn = self.workers[i]
        try:
            conn.send(msg)
        except (BrokenPipeError, OSError) as e:
            raise SystemExit(f"bag worker {i} died (exit code {process.exitcode}): {e!r}") from e

    def _rounds(self, requests: list) -> list:
        """Send request i to worker i % n in rounds of n and collect answers in request order.
        A failure is raised only after the whole round is received (so no unread answers are
        left in a worker's pipe)."""
        n, out = len(self.workers), []
        for start in range(0, len(requests), n):
            chunk = requests[start : start + n]
            for j, msg in enumerate(chunk):
                self._send((start + j) % n, msg)
            answers, error = [], None
            for j in range(len(chunk)):
                try:
                    answers.append(self._recv((start + j) % n))
                except RuntimeError as e:
                    error = error or e
            if error is not None:
                raise error
            out.extend(answers)
        return out

    def fit(
        self, problem_type: str, categorical: list[int], parts: list[tuple[Any, Any, Any]]
    ) -> tuple[list[tuple[int, int]], list]:
        """Fit each bag's (X, y, groups). Returns ((worker, key) per bag, `classes_` or None per
        bag)."""
        n = len(self.workers)
        keys = [next(self._keys) for _ in parts]
        handles = [(i % n, key) for i, key in enumerate(keys)]
        requests = [
            ("fit", key, problem_type, categorical, X, y, groups)
            for key, (X, y, groups) in zip(keys, parts, strict=True)
        ]
        try:
            classes = self._rounds(requests)
        except RuntimeError:
            self.drop(handles)  # also drop the bags that did fit
            raise
        return handles, classes

    def predict(self, handles: list[tuple[int, int]], X: Any, proba: bool) -> list[np.ndarray]:
        """Predictions on `X` for each fitted bag (`predict_proba` if `proba`)."""
        n = len(self.workers)
        if any(w != i % n for i, (w, _) in enumerate(handles)):
            raise ValueError("pass handles in the order fit returned them")
        return self._rounds([("predict", key, X, proba) for _, key in handles])

    def drop(self, handles: list[tuple[int, int]]) -> None:
        by_worker: dict[int, list[int]] = {}
        for w, key in handles:
            by_worker.setdefault(w, []).append(key)
        for w, keys in by_worker.items():
            self._send(w, ("drop", keys))

    def close(self) -> None:
        for _, conn in self.workers:
            try:
                conn.send(("stop",))
            except (BrokenPipeError, OSError):
                pass
        for process, conn in self.workers:
            process.join(timeout=60)
            if process.is_alive():
                process.terminate()
            conn.close()
        self.workers = []
