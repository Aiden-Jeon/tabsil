"""Where TabSil's fit and predict time goes, on synthetic tables the size of TabArena-Lite datasets,
for the two inference paths of `tabsil.inference.ensemble`:

- `joint`: every predict runs each member's training and test rows through the model together,
  one member at a time (`train_context=False`, the path of tabsil 0.1.0)
- `context`: fit keeps each member's train context and predict runs only the test rows, members
  with the same column count in one batch (`train_context=True`, the default)

Per table and path it reports fit and predict seconds (the first predict after fit, as TabArena
times it, and the median of repeats), the seconds per 1K rows TabArena reports (fit per 1K
training rows, predict per 1K test rows), the peak device memory and the largest difference of
the predictions between the two paths. For `joint` the predict time is split into host input
preparation, stages 0-2 (cell encoder, column/row stages) and the rest (row ICL, output head),
and the part spent on the training rows is estimated by predicting one test row.

    python training/scripts/profile_inference.py --device cuda [--tables taiwanese,hiva]
    python training/scripts/profile_inference.py --device cpu --random-weights --tables diabetes

The weights default to the Hub tag the tabsil package downloads (`tabsil.recipe.HUB_DEFAULT`).
Results are printed and written to `<out-dir>/profile.json` and `profile.md` (and logged to the
platform's MLflow run when `MLFLOW_RUN_ID` is set).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import statistics
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from tabsil.inference.ensemble import EnsembleClassifier, EnsembleRegressor
from tabsil.models.column_row_transformer import ColumnRowTransformer, ColumnRowTransformerConfig
from tabsil.recipe import HUB_DEFAULT, default_inference

# name: (task, classes, train rows, test rows, numeric features, categorical features, categories).
# Sizes of TabArena-Lite datasets (fold 0: two thirds train, one third test)
TABLES = {
    "diabetes": ("binary", 2, 512, 256, 8, 0, 0),
    "credit-g": ("binary", 2, 667, 333, 7, 13, 10),
    "hazelnut": ("binary", 2, 1600, 800, 30, 0, 0),
    "churn": ("binary", 2, 3333, 1667, 15, 4, 10),
    "taiwanese": ("binary", 2, 4546, 2273, 94, 0, 0),
    "coil2000": ("binary", 2, 6548, 3274, 85, 0, 0),
    "HR_Analytics": ("binary", 2, 12772, 6386, 4, 8, 40),
    "credit_card": ("binary", 2, 20000, 10000, 23, 0, 0),
    "bank-marketing": ("binary", 2, 30141, 15070, 4, 9, 12),
    "Bioresponse": ("binary", 2, 2501, 1250, 1776, 0, 0),
    "kddcup09": ("binary", 2, 33333, 16667, 174, 38, 2000),
    "airline": ("binary", 2, 86587, 43293, 17, 4, 5),
    "GiveMeSomeCredit": ("binary", 2, 100000, 50000, 10, 0, 0),
    "students": ("multiclass", 3, 2949, 1475, 36, 0, 0),
    "SDSS17": ("multiclass", 3, 52035, 26018, 11, 0, 0),
    "hiva": ("multiclass", 3, 2563, 1282, 1617, 0, 0),
    "concrete": ("regression", 0, 687, 343, 8, 0, 0),
    "houses": ("regression", 0, 13760, 6880, 8, 0, 0),
    "superconductivity": ("regression", 0, 14175, 7088, 81, 0, 0),
    "diamonds": ("regression", 0, 35960, 17980, 6, 3, 8),
    "QSAR-TID-11": ("regression", 0, 3828, 1914, 1024, 0, 0),
}
PATHS = ("joint", "context")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--weights", default=HUB_DEFAULT, help="Hub repo id or local folder")
    parser.add_argument(
        "--random-weights", action="store_true", help="a freshly initialized full-size model"
    )
    parser.add_argument("--tables", help=f"comma-separated subset of {', '.join(TABLES)}")
    parser.add_argument("--paths", default=",".join(PATHS))
    parser.add_argument("--repeats", type=int, default=2, help="timed predicts after the first")
    parser.add_argument(
        "--max-repeat-seconds", type=float, default=10.0, help="no repeats above this predict time"
    )
    parser.add_argument("--out-dir", default="tabsil_profile")
    parser.add_argument(
        "--first-call",
        action="store_true",
        help="instead: the cost of the first predict on a new input shape (`first_call`)",
    )
    return parser.parse_args(argv)


def make_table(spec: tuple, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray, list[int]]:
    """Synthetic table: standard normal numeric columns (10% missing), categorical columns with
    `categories` levels, and a target from the first columns."""
    task, classes, n_train, n_test, n_num, n_cat, categories = spec
    rng = np.random.default_rng(seed)
    n = n_train + n_test
    data = {f"x{i}": rng.normal(size=n) for i in range(n_num)}
    for i in range(n_cat):
        data[f"c{i}"] = pd.Categorical(rng.integers(0, categories, n).astype(str))
    X = pd.DataFrame(data)
    signal = X[[c for c in X.columns[:3] if c.startswith("x")]].sum(axis=1).to_numpy()
    X = X.mask(rng.random(X.shape) < 0.1)
    noisy = signal + rng.normal(size=n)
    if task == "regression":
        y = noisy
    else:
        y = np.digitize(noisy, np.quantile(noisy, np.linspace(0, 1, classes + 1)[1:-1]))
    cat = [X.columns.get_loc(c) for c in X.columns if c.startswith("c")]
    return X, y, cat


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class Timers:
    """Seconds per label, added up by `timed` (synchronizing the device around each call)."""

    def __init__(self, device: torch.device) -> None:
        self.device, self.seconds = device, {}

    @contextlib.contextmanager
    def timed(self, label: str):
        sync(self.device)
        start = time.perf_counter()
        try:
            yield
        finally:
            sync(self.device)
            self.seconds[label] = self.seconds.get(label, 0.0) + time.perf_counter() - start


@contextlib.contextmanager
def joint_breakdown(model: ColumnRowTransformer, estimator, timers: Timers):
    """Time the joint path's host input preparation (and copy to the device) and stages 0-2 of
    each member forward (the rest of the forward is the row ICL and the output head)."""
    stages = {
        name: getattr(ColumnRowTransformer, name) for name in ("_rows", "_rows_interleaved_chunked")
    }

    def wrap(name, fn):
        def timed(self, *args, **kwargs):
            with timers.timed("stages"):
                return fn(self, *args, **kwargs)

        return timed

    real_inputs = estimator._iter_inputs

    def inputs(*args, **kwargs):
        it = real_inputs(*args, **kwargs)
        while True:
            with timers.timed("inputs"):
                item = next(it, None)
            if item is None:
                return
            yield item

    for name, fn in stages.items():
        setattr(ColumnRowTransformer, name, wrap(name, fn))
    estimator._iter_inputs = inputs
    try:
        yield
    finally:
        for name, fn in stages.items():
            setattr(ColumnRowTransformer, name, fn)
        del estimator._iter_inputs


def predict_fn(estimator, task: str):
    return estimator.predict if task == "regression" else estimator.predict_proba


def profile_path(
    model, borders, spec, X, y, cat, path: str, args, device
) -> tuple[dict, np.ndarray]:
    task, _, n_train, n_test = spec[:4]
    inference = default_inference(model, "regression" if task == "regression" else "classification")
    kw = {
        "n_estimators": inference["n_estimators"],
        "categorical_features_indices": cat,
        "max_features_per_member": inference["max_features_per_member"],
        "feature_subsampling": inference["feature_subsampling"],
        "device": str(device),
        "train_context": path == "context",
    }
    if task == "regression":
        estimator = EnsembleRegressor(model, borders, **kw)
    else:
        estimator = EnsembleClassifier(model, **kw)
    X_train, X_test, y_train = X[:n_train], X[n_train:], y[:n_train]
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    timers = Timers(device)
    with timers.timed("fit"):
        estimator.fit(X_train, y_train)
    predict = predict_fn(estimator, task)
    with timers.timed("predict"):
        pred = predict(X_test)
    first = timers.seconds["predict"]
    times = [first]
    if first <= args.max_repeat_seconds:
        for _ in range(args.repeats):
            t = Timers(device)
            with t.timed("predict"):
                predict(X_test)
            times.append(t.seconds["predict"])
    row = {
        "fit_s": timers.seconds["fit"],
        "predict_s": first,
        "predict_s_median": statistics.median(times),
        "fit_s_per_1K": 1000 * timers.seconds["fit"] / n_train,
        "predict_s_per_1K": 1000 * first / n_test,
        "contexts": estimator.contexts_ is not None if path == "context" else None,
    }
    if device.type == "cuda":
        row["peak_gb"] = torch.cuda.max_memory_allocated(device) / 1e9
    # where the time goes
    one_row = Timers(device)
    with one_row.timed("predict"):
        predict(X_test[:1])
    row["predict_one_row_s"] = one_row.seconds["predict"]
    if path == "joint":
        parts = Timers(device)
        with joint_breakdown(model, estimator, parts), parts.timed("predict"):
            predict(X_test)
        total = parts.seconds["predict"]
        row["joint_inputs_s"] = parts.seconds.get("inputs", 0.0)
        row["joint_stages_s"] = parts.seconds.get("stages", 0.0)
        row["joint_icl_head_s"] = total - row["joint_inputs_s"] - row["joint_stages_s"]
        row["joint_breakdown_total_s"] = total
        # the training rows' share: predicting one test row still runs every training row
        row["joint_train_rows_share"] = min(1.0, row["predict_one_row_s"] / max(total, 1e-9))
    del estimator
    return row, np.asarray(pred)


def load(args, device):
    if args.random_weights:
        torch.manual_seed(0)
        model = ColumnRowTransformer(ColumnRowTransformerConfig()).eval()
        return model, torch.linspace(-5, 5, 1001)
    from tabsil.hub import from_pretrained

    model, borders = from_pretrained(args.weights)
    return model.eval(), borders


def markdown(rows: list[dict]) -> str:
    lines = [
        "| table | task | train | test | features | path | fit s | predict s | fit s/1K train"
        " | predict s/1K test | 1-row predict s | peak GB | max diff |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['table']} | {r['task']} | {r['train']} | {r['test']} | {r['features']} |"
            f" {r['path']} | {r['fit_s']:.2f} | {r['predict_s']:.2f} | {r['fit_s_per_1K']:.3f} |"
            f" {r['predict_s_per_1K']:.3f} | {r['predict_one_row_s']:.3f} |"
            f" {r.get('peak_gb', float('nan')):.1f} | {r.get('max_diff', float('nan')):.1e} |"
        )
    lines += [
        "",
        "joint-path breakdown (seconds of one predict):",
        "",
        "| table | inputs (host + copy) | stages 0-2 | row ICL + head | training-row share |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        if r["path"] == "joint":
            lines.append(
                f"| {r['table']} | {r['joint_inputs_s']:.3f} | {r['joint_stages_s']:.3f} |"
                f" {r['joint_icl_head_s']:.3f} | {r['joint_train_rows_share']:.0%} |"
            )
    return "\n".join(lines) + "\n"


SDPA_BACKENDS = ("cudnn", "flash", "mem_efficient", "math")


@contextlib.contextmanager
def sdpa_backends(enabled: tuple[str, ...]):
    """Only the `enabled` SDPA backends (torch.backends.cuda flags), restored afterwards."""
    cuda = torch.backends.cuda
    flags = {
        "cudnn": (cuda.cudnn_sdp_enabled, cuda.enable_cudnn_sdp),
        "flash": (cuda.flash_sdp_enabled, cuda.enable_flash_sdp),
        "mem_efficient": (cuda.mem_efficient_sdp_enabled, cuda.enable_mem_efficient_sdp),
        "math": (cuda.math_sdp_enabled, cuda.enable_math_sdp),
    }
    before = {name: get() for name, (get, _) in flags.items()}
    for name, (_, set_) in flags.items():
        set_(name in enabled)
    try:
        yield
    finally:
        for name, (_, set_) in flags.items():
            set_(before[name])


def reset_peak(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def peak_gb(device: torch.device) -> float:
    return torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else 0.0


def first_call(model, borders, args, device, out_dir: Path) -> None:
    """Per table and path: the predict time on test-row counts not seen before (cold) and again
    (warm), with the default SDPA backends and with cuDNN attention off; a profiler table of one
    cold predict; and the prediction difference from predicting the test rows in two halves (the
    rounding that depends on the batch shape, as a reference for the joint/context difference)."""
    from torch.profiler import ProfilerActivity, profile

    names = args.tables.split(",") if args.tables else ["diabetes", "taiwanese", "credit_card"]
    backends = {
        "default": SDPA_BACKENDS,
        "no_cudnn": ("flash", "mem_efficient", "math"),
    }
    report, rows = [], []
    report.append(
        "sdpa flags: "
        + ", ".join(
            f"{b}={getattr(torch.backends.cuda, f'{b}_sdp_enabled')()}" for b in SDPA_BACKENDS
        )
    )
    for name in names:
        spec = TABLES[name]
        task, _, n_train, n_test = spec[:4]
        X, y, cat = make_table(spec)
        X_test = X[n_train:]
        for path in args.paths.split(","):
            for label, enabled in backends.items():
                with sdpa_backends(enabled):
                    inference = default_inference(
                        model, "regression" if task == "regression" else "classification"
                    )
                    kw = {
                        "n_estimators": inference["n_estimators"],
                        "categorical_features_indices": cat,
                        "max_features_per_member": inference["max_features_per_member"],
                        "feature_subsampling": inference["feature_subsampling"],
                        "device": str(device),
                        "train_context": path == "context",
                    }
                    est = (
                        EnsembleRegressor(model, borders, **kw)
                        if task == "regression"
                        else EnsembleClassifier(model, **kw)
                    )
                    peaks = {}
                    reset_peak(device)
                    t = Timers(device)
                    with t.timed("fit"):
                        est.fit(X[:n_train], y[:n_train])
                    peaks["peak_fit_gb"] = peak_gb(device)
                    predict = predict_fn(est, task)
                    reset_peak(device)
                    predict(X_test)
                    peaks["peak_predict_gb"] = peak_gb(device)
                    if path == "context":  # one member per forward
                        est.predict_batch_bytes, batch = 1, est.predict_batch_bytes
                        reset_peak(device)
                        predict(X_test)
                        peaks["peak_predict_one_member_gb"] = peak_gb(device)
                        est.predict_batch_bytes = batch
                    times = {}
                    # test-row counts this process has not predicted yet (cold), then the same
                    # count again (warm)
                    for k in (n_test - 1, n_test - 2, 5, 7) if device.type == "cuda" else (5,):
                        for when in ("cold", "warm"):
                            tt = Timers(device)
                            with tt.timed("p"):
                                full = predict(X_test[:k])
                            times[f"{k}_{when}"] = tt.seconds["p"]
                    halves = np.concatenate(
                        [predict(X_test[: n_test // 2]), predict(X_test[n_test // 2 :])]
                    )
                    full = np.asarray(predict(X_test))
                    row = {
                        "table": name,
                        "path": path,
                        "sdpa": label,
                        "fit_s": t.seconds["fit"],
                        **peaks,
                        **times,
                        "halves_max_diff": float(np.abs(halves - full).max()),
                    }
                    rows.append(row)
                    print(json.dumps(row), flush=True)
                    if label == "default":
                        k = 11  # a new test-row count: profile one cold predict
                        with profile(
                            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]
                        ) as prof:
                            predict(X_test[:k])
                            sync(device)
                        table = prof.key_averages().table(sort_by="cpu_time_total", row_limit=25)
                        report.append(f"## {name} {path}: one cold predict ({k} rows)\n{table}")
                        print(report[-1], flush=True)
                    del est
    (out_dir / "first_call.json").write_text(json.dumps(rows, indent=1))
    (out_dir / "first_call.txt").write_text("\n\n".join(report))
    if "MLFLOW_RUN_ID" in os.environ:
        import mlflow

        with mlflow.start_run(run_id=os.environ["MLFLOW_RUN_ID"]):
            mlflow.log_artifact(str(out_dir / "first_call.json"), "profile")
            mlflow.log_artifact(str(out_dir / "first_call.txt"), "profile")


def main(argv=None) -> None:
    args = parse_args(argv)
    device = torch.device(args.device)
    names = args.tables.split(",") if args.tables else list(TABLES)
    paths = args.paths.split(",")
    model, borders = load(args, device)
    model.to(device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.first_call:
        return first_call(model, borders, args, device, out_dir)
    # warm up the device and kernels outside the timings (TabArena warms up before the timed fit)
    warm = make_table(("binary", 2, 64, 16, 4, 0, 0))
    for path in paths:
        profile_path(model, borders, ("binary", 2, 64, 16), *warm, path, args, device)
    rows = []
    for name in names:
        spec = TABLES[name]
        X, y, cat = make_table(spec)
        preds = {}
        for path in paths:
            row, preds[path] = profile_path(model, borders, spec, X, y, cat, path, args, device)
            row.update(
                table=name,
                task=spec[0],
                train=spec[2],
                test=spec[3],
                features=X.shape[1],
                path=path,
            )
            rows.append(row)
        if len(preds) == 2:
            diff = np.abs(preds["joint"] - preds["context"])
            scale = 1.0 if spec[0] != "regression" else max(float(np.std(y)), 1e-12)
            for r in rows[-2:]:
                r["max_diff"] = float(diff.max() / scale)  # regression: in target std units
        for r in rows[-len(paths) :]:
            print(json.dumps(r), flush=True)
    (out_dir / "profile.json").write_text(json.dumps(rows, indent=1))
    (out_dir / "profile.md").write_text(markdown(rows))
    print(markdown(rows), flush=True)
    if "MLFLOW_RUN_ID" in os.environ:
        import mlflow

        with mlflow.start_run(run_id=os.environ["MLFLOW_RUN_ID"]):
            mlflow.log_artifact(str(out_dir / "profile.json"), "profile")
            mlflow.log_artifact(str(out_dir / "profile.md"), "profile")


if __name__ == "__main__":
    main()
