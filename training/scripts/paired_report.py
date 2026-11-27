"""Paired report of two evaluation runs (`tabsil_train.benchmarks.paired`; v2 experiment plan §3):
the geometric-mean error ratio G of b against a over datasets, its bootstrap CI, the Wilcoxon p
and the datasets better and worse, overall, for classification and for regression, per suite and
pooled.

A run is a per-split table: a local `results_per_split.csv` (TabArena / BeyondArena) or
`eval_rows.csv` (AMLB, `--suite dev`), or an MLflow run, `runs:/<run_id>` (the eval artifact is
found under `eval/`) or `runs:/<run_id>/<artifact path>`. Give `--a` and `--b` once per suite, in
the same order; more than one pair is also pooled. The pairs must not share a dataset (an error;
`--no-pool` gives the per-suite tables only, e.g. for the same suite at two seeds). Several runs
joined with `+` (`runs:/<id>+runs:/<id>`) are one side that is their mean: on every split, the
mean of the runs' errors (the control of a branch that was repeated with another seed). MLflow
runs need `DATABRICKS_CONFIG_PROFILE=<profile> MLFLOW_TRACKING_URI=databricks`.

G measures the weights alone only if a and b were evaluated the same way (plan §3). For MLflow
runs the report compares the evaluation seed, the fold count, the context cap
(`subsample_train_rows`) and the applied inference settings (`inference_classification`,
`inference_regression`) of a and b, and prints a warning line for each that differs. Runs that
are averaged must agree in these settings (an error). Local files carry no settings and are not
checked.

From the repository root:
  uv run --package tabsil-train python training/scripts/paired_report.py \
    --a runs:/<control 3-split run> --b runs:/<arm 3-split run> \
    --a runs:/<control dev run> --b runs:/<arm dev run> --out-dir paired_out

The Elo of arena runs (MLflow metrics `<suite>_elo`, `<suite>_<group>_elo`, or `summary.json`
next to a local file) is shown next to G and decides nothing.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from tabsil_train.benchmarks import paired

ARENA_FILE = "results_per_split.csv"
AMLB_FILE = "eval_rows.csv"
ELO_GROUPS = ("classification", "regression", "binary", "multiclass")
# run params that a and b must share for G to measure the weights alone (`settings_differences`)
SETTINGS = (
    "seed",
    "folds",
    "subsample_train_rows",
    "inference_classification",
    "inference_regression",
)


class Run:
    """One side of a comparison: its per-split errors, suite name, Elo per group and, for an
    MLflow run, its params (None for a local file). A source of several runs joined with `+` is
    their mean (`paired.mean_splits`): the runs must be of one suite and, as MLflow runs, agree in
    the evaluation settings (`SETTINGS`). The mean has no Elo, and its params are the first
    run's."""

    def __init__(self, source: str, allow_imputed: bool, cache: Path) -> None:
        self.source = source
        parts = [resolve(part, cache) for part in source.split("+")]
        suites = {suite for _, suite, _, _ in parts}
        if len(suites) > 1:
            raise SystemExit(f"{source}: the runs to average are of different suites {suites}")
        _, self.suite, elo, self.params = parts[0]
        self.elo = elo if len(parts) == 1 else {}
        for _, _, _, params in parts[1:]:
            if self.params is None or params is None:
                self.params = None  # a local file among them: nothing to compare
                continue
            if differences := settings_differences(self.params, params):
                raise SystemExit(f"{source}: the runs to average differ in {differences}")
        frames = [paired.load_splits(path, allow_imputed=allow_imputed) for path, *_ in parts]
        self.splits = frames[0] if len(frames) == 1 else paired.mean_splits(frames)


def resolve(source: str, cache: Path) -> tuple[Path, str, dict, dict | None]:
    """(local file, suite name, {group: Elo}, run params) of a local file or a `runs:/` source.
    A local file has no params (None), and its suite name is only a label: the folder's name for
    an arena file, "amlb" otherwise."""
    if not source.startswith("runs:/"):
        path = Path(source)
        suite = path.parent.name if path.name == ARENA_FILE else "amlb"
        summary = path.parent / "summary.json"
        return path, suite, _summary_elo(summary) if summary.exists() else {}, None
    import mlflow

    run_id, _, artifact = source.removeprefix("runs:/").partition("/")
    client = mlflow.MlflowClient()
    if not artifact:
        artifact = _find_artifact(client, run_id)
    local = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path=artifact, dst_path=str(cache / run_id)
    )
    run = client.get_run(run_id)
    if Path(artifact).name == ARENA_FILE:
        suite = Path(artifact).parent.name
    else:
        suite = run.data.params.get("suite")
        suite = "dev" if suite == "dev" else f"amlb-{suite}" if suite else "amlb"
    prefix = suite.replace("-", "_")
    metrics = run.data.metrics
    elo = {"all": metrics.get(f"{prefix}_elo")}
    elo.update({g: metrics.get(f"{prefix}_{g}_elo") for g in ELO_GROUPS})
    elo = {g: v for g, v in elo.items() if v is not None}
    return Path(local), suite, elo, dict(run.data.params)


def settings_differences(a: dict, b: dict) -> list[str]:
    """How the evaluation settings (`SETTINGS`) of two runs' params differ, one text per setting;
    empty if they agree. A setting only one run logged cannot be compared and is listed too
    (runs from before the applied inference settings were logged have no `inference_*`)."""
    out = []
    for key in SETTINGS:
        value_a, value_b = a.get(key), b.get(key)
        if value_a == value_b:
            continue
        if value_a is None or value_b is None:
            out.append(f"{key} is not logged by {'a' if value_a is None else 'b'}")
            continue
        if key.startswith("inference_"):
            fields_a, fields_b = json.loads(value_a), json.loads(value_b)
            for field in sorted(set(fields_a) | set(fields_b)):
                if fields_a.get(field) != fields_b.get(field):
                    out.append(f"{key}.{field}: a {fields_a.get(field)}, b {fields_b.get(field)}")
        else:
            out.append(f"{key}: a {value_a}, b {value_b}")
    return out


def pair_warnings(name: str, a: Run, b: Run) -> list[str]:
    """Warning lines of one pair of MLflow runs: another suite, other evaluation settings."""
    if a.params is None or b.params is None:
        return []
    differences = settings_differences(a.params, b.params)
    if a.suite != b.suite:
        differences.insert(0, f"suite: a {a.suite}, b {b.suite}")
    return [f"- warning ({name}): a and b differ in {d}" for d in differences]


def _find_artifact(client, run_id: str) -> str:
    """The run's per-split table under `eval/`: an arena suite's `results_per_split.csv` or the
    AMLB `eval_rows.csv`. More than one arena suite needs the path in the source."""
    found = []
    for entry in client.list_artifacts(run_id, "eval"):
        if entry.is_dir:
            names = [Path(a.path).name for a in client.list_artifacts(run_id, entry.path)]
            if ARENA_FILE in names:
                found.append(f"{entry.path}/{ARENA_FILE}")
        elif Path(entry.path).name == AMLB_FILE:
            found.append(entry.path)
    if len(found) != 1:
        raise SystemExit(
            f"runs:/{run_id}: {'no' if not found else 'more than one'} per-split table under"
            f" eval/ ({found}); give runs:/{run_id}/<artifact path>"
        )
    return found[0]


def _summary_elo(path: Path) -> dict:
    summary = json.loads(path.read_text())
    elo = {"all": summary.get("elo")}
    elo.update({g: s.get("elo") for g, s in summary.get("breakdown", {}).items()})
    return {g: v for g, v in elo.items() if v is not None}


def _unique_names(names: list[str]) -> list[str]:
    # two pairs of the same suite are told apart as suite, suite#2, ...
    out, seen = [], {}
    for n in names:
        seen[n] = seen.get(n, 0) + 1
        out.append(n if seen[n] == 1 else f"{n}#{seen[n]}")
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--a", action="append", required=True, help="control run (per suite)")
    parser.add_argument("--b", action="append", required=True, help="arm run (per suite)")
    parser.add_argument(
        "--name", action="append", help="suite name per pair (default: from the run)"
    )
    parser.add_argument(
        "--no-pool",
        action="store_true",
        help="per-suite tables only (pairs that share datasets, e.g. one suite at two seeds)",
    )
    parser.add_argument("--rounds", type=int, default=paired.BOOTSTRAP_ROUNDS)
    parser.add_argument("--seed", type=int, default=paired.BOOTSTRAP_SEED)
    parser.add_argument(
        "--allow-imputed",
        action="store_true",
        help="keep imputed splits (they compare the stand-in method; for a look only)",
    )
    parser.add_argument("--out-dir", help="write report.md, report.json and ratios.csv here")
    args = parser.parse_args(argv)
    if len(args.a) != len(args.b):
        parser.error("give --a and --b the same number of times (one pair per suite)")
    if args.name and len(args.name) != len(args.a):
        parser.error("give --name once per pair")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    with tempfile.TemporaryDirectory() as tmp:
        pairs = [
            (Run(a, args.allow_imputed, Path(tmp)), Run(b, args.allow_imputed, Path(tmp)))
            for a, b in zip(args.a, args.b, strict=True)
        ]
    names = _unique_names(args.name or [a.suite for a, _ in pairs])
    try:
        report = paired.compare(
            {n: (a.splits, b.splits) for n, (a, b) in zip(names, pairs, strict=True)},
            rounds=args.rounds,
            seed=args.seed,
            pool=not args.no_pool,
        )
    except ValueError as e:
        raise SystemExit(str(e)) from None
    elo = {
        n: {g: (a.elo.get(g), b.elo.get(g)) for g in ("all", *ELO_GROUPS)}
        for n, (a, b) in zip(names, pairs, strict=True)
    }
    lines = [
        f"- {n}: a = `{a.source}`, b = `{b.source}`" for n, (a, b) in zip(names, pairs, strict=True)
    ]
    for n, (a, b) in zip(names, pairs, strict=True):
        lines += pair_warnings(n, a, b)
    text = "\n".join(lines) + f"\n\n{paired.format_report(report, elo)}"
    print(text)
    if args.out_dir:
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "report.md").write_text(text)
        (out / "report.json").write_text(paired.to_json(report))
        report["ratios"].to_csv(out / "ratios.csv", index=False)


if __name__ == "__main__":
    main()
