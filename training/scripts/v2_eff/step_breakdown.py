"""Where a training step's time goes, from the records of one MLflow training run.

    DATABRICKS_CONFIG_PROFILE=e2-demo-field-eng MLFLOW_TRACKING_URI=databricks \\
        uv run --package tabsil-train python training/scripts/v2_eff/step_breakdown.py <run id> \\
        [--steps 100:300] [--stall-ms 300]

It reads the run's `sec_per_step`, the per-GPU records of `TABSIL_RANK_TIMING=1`
(`diagnostics/rank_timing_from_*.jsonl`: forward and backward time per GPU and step) and, if the
run was made with `step_probe.py`, its records (`eff_probe/`: optimizer step, gradient clipping,
norms, data wait and collector pauses per GPU and step). `--steps a:b` takes the steps a..b-1
(0-based, as in the records); the first 100 hold the compilation.

How the parts are defined, per optimizer step and averaged over the steps:

- forward: the mean over the GPUs of their forward time (summed over the step's micro batches)
- backward: the smallest backward time over the GPUs. Every GPU ends a step together, in the
  gradient all-reduce of the last micro batch, so the GPU with the smallest backward is the one
  the others waited for, and its backward holds no waiting
- waiting for the slowest GPU: the mean backward minus that smallest one. It is caused by whatever
  made one GPU late: a collector pause, a late table, or more work in its tables
- rest: the step time minus the mean forward and backward. With the probe it splits into the
  optimizer step, clipping, the norms (rank 0 only), each GPU's own data wait, the training
  loop's own collection and freeze (`TrainConfig.gc_freeze`) and what is left (the loss
  all-reduce, logging and the Python between the micro batches)

The backward of the GPU that did not wait belongs to that GPU's tables. Where the tables of the
slowest GPU are larger than the average, the waiting is understated by that difference.

The loop's collection and freeze at a log step falls into the record of the step after it. The
first one walks every object and is paid once per run (about a second); the later ones walk what
was made since. With the default `log_every` of 100 the first is in record 100, so `--steps
101:300` gives the steady state and `--steps 100:300` holds the one-time pass as well.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import mlflow
import numpy as np


def _download(client: mlflow.MlflowClient, run_id: str, folder: str, dst: Path) -> list[Path]:
    paths = []
    for artifact in client.list_artifacts(run_id, folder):
        local = mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path=artifact.path, dst_path=str(dst)
        )
        paths.append(Path(local))
    return sorted(paths)


def rank_timing(paths: list[Path]) -> dict[int, dict]:
    """step -> the record of that step (the last one if a rewind repeated it)."""
    records = {}
    for path in paths:
        for line in path.read_text().splitlines():
            record = json.loads(line)
            records[record["step"]] = record
    return records


def probe_records(paths: list[Path]) -> list[dict]:
    """The probe file of every GPU, in rank order."""
    return sorted((json.loads(path.read_text()) for path in paths), key=lambda r: r["rank"])


def breakdown(
    timing: dict[int, dict],
    probes: list[dict],
    sec_per_step: dict[int, float],
    start: int,
    stop: int,
    stall_ms: float,
) -> dict[str, float]:
    steps = [s for s in range(start, stop) if s in timing]
    if not steps:
        raise SystemExit(f"no rank timing records in steps {start}:{stop}")
    fwd = np.array([timing[s]["fwd_ms"] for s in steps])  # (steps, GPUs)
    bwd = np.array([timing[s]["bwd_ms"] for s in steps])
    cells = np.array([timing[s]["cells"] for s in steps])
    out: dict[str, float] = {"steps": len(steps), "gpus": fwd.shape[1]}
    # the logged windows that lie inside the range: a window ending at `done` holds the steps
    # done - log_every .. done - 1
    dones = sorted(sec_per_step)
    log_every = dones[1] - dones[0] if len(dones) > 1 else dones[0]
    windows = [sec_per_step[d] for d in dones if d - log_every >= start and d <= stop]
    if probes:
        wall = np.array([p["steps"] for p in probes])[:, :, 0]  # (GPUs, steps)
        first = max(start, 1)
        step_ms = float((wall[0, stop - 1] - wall[0, first - 1]) / (stop - first) * 1000.0)
    elif windows:
        step_ms = float(np.mean(windows) * 1000.0)
    else:
        raise SystemExit(f"no sec_per_step window inside steps {start}:{stop}")
    out["step_ms"] = step_ms
    if windows:
        out["logged_step_ms"] = float(np.mean(windows) * 1000.0)
    out["forward_ms"] = float(fwd.mean())
    out["backward_ms"] = float(bwd.min(1).mean())
    out["waiting_ms"] = float(bwd.mean() - bwd.min(1).mean())
    out["rest_ms"] = step_ms - float((fwd + bwd).mean())
    # steps in which one GPU's forward is far above the others' (a pause on that GPU)
    excess = fwd.max(1) - np.median(fwd, 1)
    out["forward_stall_steps"] = float((excess > stall_ms).mean())
    # steps in which one GPU lost time outside forward and backward (its data wait)
    total = fwd + bwd
    out["outside_stall_steps"] = float(((total.max(1) - total.min(1)) > stall_ms).mean())
    out["forward_imbalance"] = float(np.mean(fwd.max(1) / fwd.mean(1)))
    out["cells_imbalance"] = float(np.mean(cells.max(1) / cells.mean(1)))
    out["cells_per_step"] = float(cells.sum(1).mean())
    if probes:
        # the record names its fields: those made before the loop froze have no `freeze_ms`
        fields = list(probes[0]["fields"])
        values = np.array([p["steps"] for p in probes])[:, start:stop, :]  # (GPUs, steps, fields)
        for i, name in enumerate(fields[1:], start=1):
            ranks = values[:1] if name == "norms_ms" else values  # the norms are rank 0's
            out[name] = float(ranks[:, :, i].mean())
        # a pause delays the step by the longest one among the GPUs, not by their sum
        out["gc_max_ms"] = float(values[:, :, fields.index("gc_ms")].max(0).mean())
        out["wait_max_ms"] = float(values[:, :, fields.index("wait_ms")].max(0).mean())
        named = ("opt_ms", "clip_ms", "norms_ms", "wait_ms", "freeze_ms")
        out["other_ms"] = out["rest_ms"] - sum(out.get(name, 0.0) for name in named)
    return out


def report(run, parts: dict[str, float], probes: list[dict], label: str) -> None:
    params = run.data.params
    tables = int(params["batch_size"])
    gpus = int(parts["gpus"])
    step = parts["step_ms"]
    print(f"\n{label}: {int(parts['steps'])} steps, {gpus} GPUs")
    rows = [
        ("forward", "forward_ms"),
        ("backward (the GPU that did not wait)", "backward_ms"),
        ("waiting for the slowest GPU", "waiting_ms"),
    ]
    if probes:
        rows += [
            ("optimizer step", "opt_ms"),
            ("gradient clipping", "clip_ms"),
            ("per-group norms (rank 0)", "norms_ms"),
            ("data wait (own)", "wait_ms"),
            ("collection and freeze (the loop's or the probe's)", "freeze_ms"),
            ("other", "other_ms"),
        ]
        rows = [row for row in rows if row[1] in parts]
    else:
        rows.append(("rest (optimizer, clipping, norms, data wait, logging)", "rest_ms"))
    for name, key in rows:
        print(f"  {name:55s} {parts[key]:8.1f} ms  {parts[key] / step:6.1%}")
    print(f"  {'step':55s} {step:8.1f} ms", end="")
    if "logged_step_ms" in parts:
        print(f"  (logged sec_per_step {parts['logged_step_ms'] / 1000:.3f})", end="")
    print()
    print(
        f"  GPU-seconds per table {step / 1000 * gpus / tables:.4f}, per 1M cells"
        f" {step / 1000 * gpus / (parts['cells_per_step'] / 1e6):.3f},"
        f" H100-h per 1K steps {step / 1000 * gpus * 1000 / 3600:.2f}"
    )
    print(
        f"  steps with a forward pause on one GPU {parts['forward_stall_steps']:.0%}, with a GPU"
        f" late outside forward and backward {parts['outside_stall_steps']:.0%}; slowest forward /"
        f" mean {parts['forward_imbalance']:.2f}, most cells / mean {parts['cells_imbalance']:.2f}"
    )
    if probes:
        print(
            f"  collector pauses: {parts['gc_ms']:.1f} ms per GPU and step, longest among the GPUs"
            f" {parts['gc_max_ms']:.1f} ms per step; longest data wait among the GPUs"
            f" {parts['wait_max_ms']:.1f} ms per step"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_id")
    parser.add_argument("--steps", default="100:300", help="first:stop, 0-based steps")
    parser.add_argument("--stall-ms", type=float, default=300.0)
    args = parser.parse_args()
    start, stop = (int(v) for v in args.steps.split(":"))

    client = mlflow.MlflowClient()
    run = client.get_run(args.run_id)
    with tempfile.TemporaryDirectory() as tmp:
        timing = rank_timing(_download(client, args.run_id, "diagnostics", Path(tmp)))
        probes = probe_records(_download(client, args.run_id, "eff_probe", Path(tmp)))
    history = client.get_metric_history(args.run_id, "sec_per_step")
    sec_per_step = {m.step: m.value for m in history}
    params = run.data.params
    print(
        f"{run.data.tags.get('mlflow.runName')} ({args.run_id}): emsize {params.get('emsize')},"
        f" batch {params.get('batch_size')}, micro batch {params.get('micro_batch')},"
        f" accumulation {params.get('grad_accumulation')}, prior {params.get('prior_type')}"
    )
    frozen = probes[0].get("frozen_at") if probes else None
    if frozen is not None and start < frozen < stop:
        # the collector was frozen inside the range: report both sides
        ranges = [(start, frozen, "collector as it is"), (frozen, stop, "after gc.freeze()")]
    else:
        ranges = [(start, stop, f"steps {start}:{stop}")]
    for first, last, label in ranges:
        parts = breakdown(timing, probes, sec_per_step, first, last, args.stall_ms)
        report(run, parts, probes, label)
    if probes:
        # the collector's own full passes (the probe keeps the loop's apart, in `freezes`)
        pauses = [e for p in probes for e in p["gc_events"] if e[1] == 2 and e[0] >= start]
        seconds = [e[2] for e in pauses]
        # frozen objects are tracked too, and `gc_objects` leaves them out
        tracked = [p["gc_objects"] + p["gc_frozen"] for p in probes]
        print(
            f"\nfull passes of the collector over 50 ms from step {start}: {len(pauses)} over"
            f" {len(probes)} GPUs"
            + (f", mean {np.mean(seconds):.3f} s, longest {max(seconds):.3f} s" if pauses else "")
            + f"; tracked objects per GPU at the end {min(tracked):,}-{max(tracked):,}"
        )
        # records made before the probe recorded the freezes have none
        freezes = [p["freezes"] for p in probes if p.get("freezes")]
        if freezes:
            steps = [freeze[0] for freeze in freezes[0]]
            first, last = [f[0][1] for f in freezes], [f[-1][1] for f in freezes]
            took = np.array([[freeze[2] for freeze in rank] for rank in freezes])  # (GPUs, freezes)
            later = f", the later ones {took[:, 1:].max(0).mean():.3f} s" if len(steps) > 1 else ""
            print(
                f"freezes: {len(steps)} per GPU, after {steps[0]} to {steps[-1]} steps; frozen"
                f" objects per GPU {min(first):,}-{max(first):,} after the first and"
                f" {min(last):,}-{max(last):,} after the last; the first collection and freeze"
                f" took {took[:, 0].max():.3f} s{later} (the slowest GPU)"
            )


if __name__ == "__main__":
    main()
