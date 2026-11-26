"""Paired report of two evaluation runs (`tabsil_train.benchmarks.paired`; v2 experiment plan §3):
the geometric-mean error ratio G of b against a over datasets, its bootstrap CI, the Wilcoxon p
and the datasets better and worse, overall, for classification and for regression, per suite and
pooled.

A run is a per-split table: a local `results_per_split.csv` (TabArena / BeyondArena) or
`eval_rows.csv` (AMLB, `--suite dev`), or an MLflow run, `runs:/<run_id>` (the eval artifact is
found under `eval/`) or `runs:/<run_id>/<artifact path>`. Give `--a` and `--b` once per suite, in
the same order; more than one pair is also pooled. MLflow runs need
`DATABRICKS_CONFIG_PROFILE=<profile> MLFLOW_TRACKING_URI=databricks`.

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
import sys
import tempfile
from pathlib import Path

from tabsil_train.benchmarks import paired

ARENA_FILE = "results_per_split.csv"
AMLB_FILE = "eval_rows.csv"
ELO_GROUPS = ("classification", "regression", "binary", "multiclass")


class Run:
    """One side of a comparison: its per-split errors, suite name and Elo per group."""

    def __init__(self, source: str, allow_imputed: bool, cache: Path) -> None:
        self.source = source
        path, self.suite, self.elo = resolve(source, cache)
        self.splits = paired.load_splits(path, allow_imputed=allow_imputed)


def resolve(source: str, cache: Path) -> tuple[Path, str, dict]:
    """(local file, suite name, {group: Elo}) of a local file or a `runs:/` source."""
    if not source.startswith("runs:/"):
        path = Path(source)
        suite = path.parent.name if path.name == ARENA_FILE else "amlb"
        summary = path.parent / "summary.json"
        return path, suite, _summary_elo(summary) if summary.exists() else {}
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
        suite = run.data.params.get("suite", "amlb")
        suite = "dev" if suite == "dev" else f"amlb-{suite}"
    prefix = suite.replace("-", "_")
    metrics = run.data.metrics
    elo = {"all": metrics.get(f"{prefix}_elo")}
    elo.update({g: metrics.get(f"{prefix}_{g}_elo") for g in ELO_GROUPS})
    return Path(local), suite, {g: v for g, v in elo.items() if v is not None}


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
    for a, b in pairs:
        if a.suite != b.suite and not args.name:
            print(f"warning: {a.source} is {a.suite}, {b.source} is {b.suite}", file=sys.stderr)
    names = _unique_names(args.name or [a.suite for a, _ in pairs])
    report = paired.compare(
        {n: (a.splits, b.splits) for n, (a, b) in zip(names, pairs, strict=True)},
        rounds=args.rounds,
        seed=args.seed,
    )
    elo = {
        n: {g: (a.elo.get(g), b.elo.get(g)) for g in ("all", *ELO_GROUPS)}
        for n, (a, b) in zip(names, pairs, strict=True)
    }
    sources = "\n".join(
        f"- {n}: a = `{a.source}`, b = `{b.source}`" for n, (a, b) in zip(names, pairs, strict=True)
    )
    text = f"{sources}\n\n{paired.format_report(report, elo)}"
    print(text)
    if args.out_dir:
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "report.md").write_text(text)
        (out / "report.json").write_text(paired.to_json(report))
        report["ratios"].to_csv(out / "ratios.csv", index=False)


if __name__ == "__main__":
    main()
