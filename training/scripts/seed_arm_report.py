"""Report of arms trained from scratch with several seeds against a control with several seeds
(v2 experiment plan §3, "Arm trained from scratch (phase 1B)").

Every training run has one evaluation run per suite. A run's score is its mean log error over the
datasets (the splits of a dataset are averaged first, as in `tabsil_train.benchmarks.paired`). For
each arm the report gives, per suite and pooled over the suites:

- the score of every control and arm seed, and whether every arm seed is below every control seed
  (the rule's first test). Under exchangeable runs, k arm seeds below all n control seeds has the
  one-sided permutation probability 1 / C(n + k, k): 0.1 for 2 against 3 and 0.05 for 3 against 3
- G of the seed means: per dataset, the arm's error averaged over its seeds over the control's
  error averaged over its seeds, then the geometric mean over datasets, with the bootstrap CI,
  the Wilcoxon p and the datasets better and worse, for all datasets, classification and
  regression (`paired.summarize`)
- G of each arm seed against the control seed of the same number, where both exist. Runs of the
  same seed start from the same weights and prior stream, so these pairs share that part of the
  noise
- the rule's reading: `third seed` (every arm seed below every control seed, with fewer than 3
  arm seeds), `adopted on the evidence` (3 arm seeds, all below, G at most 0.98), `dropped`
  (G above 1.02 with a CI above 1), otherwise `not resolved` with whether G is at most 1.01 (then
  the arm is adopted only if it removes a train/inference mismatch or copies a published
  configuration, which the report cannot know)

The spec is a JSON file:

    {"control": {"0": {"tabarena-3split": "runs:/<id>", "dev": "runs:/<id>"}, "1": {...}},
     "arms": {"yctx": {"0": {"tabarena-3split": "runs:/<id>", "dev": "runs:/<id>"}, ...}}}

A source is what `paired_report.py` takes: a local per-split file or `runs:/<run_id>`. An imputed
split or a missing error is an error. The evaluation settings of every run are compared with the
first control seed's (`paired_report.settings_differences`) and differences are printed as
warnings. From the repository root:

    DATABRICKS_CONFIG_PROFILE=<profile> MLFLOW_TRACKING_URI=databricks \\
      uv run --package tabsil-train python training/scripts/seed_arm_report.py spec.json \\
      --out-dir seed_report
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from tabsil_train.benchmarks import paired

POOLED = "pooled"
REPORT_GROUPS = ("all", "classification", "regression")
# the rule's thresholds (plan §3)
G_EVIDENCE = 0.98
G_PUBLISHED = 1.01
G_DROP = 1.02


def _paired_report():
    """The `paired_report.py` script next to this one (its `Run` reads a source, and
    `settings_differences` compares the evaluation settings of two runs)."""
    path = Path(__file__).with_name("paired_report.py")
    spec = importlib.util.spec_from_file_location("paired_report", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def dataset_errors(splits: pd.DataFrame) -> pd.DataFrame:
    """Per dataset: problem type, split count and the split-mean error, floored as in the paired
    report (`paired.ERROR_FLOOR`)."""
    per = (
        splits.groupby("dataset")
        .agg(
            problem_type=("problem_type", "first"),
            splits=("split", "size"),
            error=("error", "mean"),
        )
        .reset_index()
    )
    per["error"] = per["error"].clip(lower=paired.ERROR_FLOOR)
    return per


def score(errors: pd.DataFrame, kinds: tuple[str, ...] = paired.GROUPS["all"]) -> float:
    """The run's score: the mean log error over its datasets of `kinds`."""
    part = errors[errors["problem_type"].isin(kinds)]
    if part.empty:
        return float("nan")
    return float(np.log(part["error"].to_numpy(dtype=float)).mean())


def mean_errors(runs: list[pd.DataFrame]) -> pd.DataFrame:
    """Per dataset, the error averaged over runs (seeds). Every run must have the same datasets
    and split counts."""
    first = runs[0].sort_values("dataset").reset_index(drop=True)
    out = first.copy()
    total = first["error"].to_numpy(dtype=float).copy()
    for other in runs[1:]:
        other = other.sort_values("dataset").reset_index(drop=True)
        same = (
            len(other) == len(first)
            and (other["dataset"] == first["dataset"]).all()
            and (other["splits"] == first["splits"]).all()
        )
        if not same:
            raise ValueError("the seeds of one side were not evaluated on the same splits")
        total += other["error"].to_numpy(dtype=float)
    out["error"] = total / len(runs)
    return out


def ratios(control: pd.DataFrame, arm: pd.DataFrame) -> pd.DataFrame:
    """The per-dataset frame `paired.summarize` takes, from two per-dataset error frames."""
    merged = control.merge(arm, on="dataset", suffixes=("_a", "_b"), validate="one_to_one")
    if len(merged) != len(control) or len(merged) != len(arm):
        raise ValueError("control and arm were not evaluated on the same datasets")
    if (merged["splits_a"] != merged["splits_b"]).any():
        raise ValueError("control and arm were not evaluated on the same splits")
    return pd.DataFrame(
        {
            "dataset": merged["dataset"],
            "problem_type": merged["problem_type_a"],
            "splits": merged["splits_a"],
            "error_a": merged["error_a"],
            "error_b": merged["error_b"],
            "ratio": merged["error_b"] / merged["error_a"],
            "floored": (merged["error_a"] <= paired.ERROR_FLOOR)
            | (merged["error_b"] <= paired.ERROR_FLOOR),
        }
    )


def group_stats(frame: pd.DataFrame) -> dict:
    return {
        group: paired.summarize(frame[frame["problem_type"].isin(paired.GROUPS[group])])
        for group in REPORT_GROUPS
    }


def permutation_p(n_control: int, n_arm: int) -> float:
    """The one-sided probability that all arm runs score below all control runs if the runs are
    exchangeable."""
    return 1.0 / math.comb(n_control + n_arm, n_arm)


def reading(all_below: bool, n_arm: int, stats: dict) -> str:
    """The rule's reading of one arm on one set of datasets (module docstring)."""
    g, low = stats["G"], stats["ci_low"]
    if g > G_DROP and low > 1.0:
        return "dropped (G above 1.02 with a CI above 1)"
    if all_below and n_arm < 3:
        return "third seed (every arm seed is below every control seed)"
    if all_below and g <= G_EVIDENCE:
        return "adopted on the evidence (3 seeds below every control seed, G at most 0.98)"
    limit = "at most" if g <= G_PUBLISHED else "above"
    return f"not resolved (G {limit} 1.01)"


def compare_arm(control: dict[str, dict], arm: dict[str, dict]) -> dict:
    """One arm against the control. `control` and `arm` are {seed: {suite: per-dataset errors}}.
    Returns {suite or "pooled": {"scores": ..., "all_below": ..., "p": ..., "stats": {group:
    summary}, "pairs": {seed: {group: summary}}, "reading": ...}}."""
    suites = list(next(iter(control.values())))
    parts = {suite: (suite,) for suite in suites}
    if len(suites) > 1:
        parts[POOLED] = tuple(suites)

    def errors(run: dict[str, pd.DataFrame], names: tuple[str, ...]) -> pd.DataFrame:
        frames = [run[name].assign(dataset=name + "/" + run[name]["dataset"]) for name in names]
        return pd.concat(frames, ignore_index=True)

    out = {}
    for part, names in parts.items():
        c = {seed: errors(run, names) for seed, run in control.items()}
        a = {seed: errors(run, names) for seed, run in arm.items()}
        scores = {
            "control": {seed: score(e) for seed, e in c.items()},
            "arm": {seed: score(e) for seed, e in a.items()},
        }
        by_task = {
            group: {
                side: {seed: score(e, paired.GROUPS[group]) for seed, e in runs.items()}
                for side, runs in (("control", c), ("arm", a))
            }
            for group in ("classification", "regression")
        }
        all_below = max(scores["arm"].values()) < min(scores["control"].values())
        frame = ratios(mean_errors(list(c.values())), mean_errors(list(a.values())))
        stats = group_stats(frame)
        out[part] = {
            "scores": scores,
            "scores_by_task": by_task,
            "all_below": bool(all_below),
            "p_if_all_below": permutation_p(len(c), len(a)),
            "stats": stats,
            "pairs": {seed: group_stats(ratios(c[seed], a[seed])) for seed in a if seed in c},
            "reading": reading(all_below, len(a), stats["all"]),
            "ratios": frame,
        }
    return out


def _cell(stats: dict) -> str:
    if stats["G"] is None:
        return "—"
    interval = f"{stats['ci_low']:.3f}-{stats['ci_high']:.3f}"
    counts = f"{stats['better']} / {stats['worse']}"
    return f"{stats['G']:.3f} ({interval}), p {stats['wilcoxon_p']:.2g}, {counts}"


def format_report(report: dict[str, dict]) -> str:
    """Markdown of `compare_arm` results per arm."""
    lines = []
    for arm, parts in report.items():
        lines += [f"## {arm}", ""]
        lines += [
            "| Datasets | Control scores | Arm scores | All arm below all control | Reading |"
        ]
        lines += ["|---|---|---|---|---|"]
        for part, r in parts.items():
            control = ", ".join(f"s{s} {v:.4f}" for s, v in r["scores"]["control"].items())
            arm_scores = ", ".join(f"s{s} {v:.4f}" for s, v in r["scores"]["arm"].items())
            below = f"yes (p {r['p_if_all_below']:.2g})" if r["all_below"] else "no"
            lines.append(f"| {part} | {control} | {arm_scores} | {below} | {r['reading']} |")
        lines += [
            "",
            "G of the seed means, arm / control (95% CI), Wilcoxon p, better / worse:",
            "",
        ]
        lines += ["| Datasets | All | Classification | Regression |", "|---|---|---|---|"]
        for part, r in parts.items():
            cells = " | ".join(_cell(r["stats"][g]) for g in REPORT_GROUPS)
            lines.append(f"| {part} | {cells} |")
        pairs = next(iter(parts.values()))["pairs"]
        if pairs:
            lines += ["", "G of the arm seed against the control seed of the same number:", ""]
            lines += [
                "| Datasets | Seed | All | Classification | Regression |",
                "|---|---|---|---|---|",
            ]
            for part, r in parts.items():
                for seed, stats in r["pairs"].items():
                    cells = " | ".join(_cell(stats[g]) for g in REPORT_GROUPS)
                    lines.append(f"| {part} | {seed} | {cells} |")
        lines.append("")
    return "\n".join(lines)


def to_json(report: dict[str, dict]) -> str:
    def plain(parts: dict) -> dict:
        return {part: {k: v for k, v in r.items() if k != "ratios"} for part, r in parts.items()}

    return json.dumps({arm: plain(parts) for arm, parts in report.items()}, indent=2)


def load(spec: dict, cache: Path) -> tuple[dict, dict, list[str]]:
    """({seed: {suite: errors}} of the control, {arm: the same}, warning lines) of a spec."""
    report = _paired_report()
    warnings: list[str] = []
    reference: dict[str, dict | None] = {}

    def side(runs: dict, label: str) -> dict:
        out = {}
        for seed, sources in runs.items():
            out[seed] = {}
            for suite, source in sources.items():
                run = report.Run(source, False, cache)
                out[seed][suite] = dataset_errors(run.splits)
                if run.params is None:
                    continue
                if suite not in reference:
                    reference[suite] = run.params
                    continue
                for d in report.settings_differences(reference[suite], run.params):
                    warnings.append(f"- warning ({label} seed {seed}, {suite}): {d}")
        return out

    control = side(spec["control"], "control")
    arms = {name: side(runs, name) for name, runs in spec["arms"].items()}
    suites = {tuple(sorted(run)) for run in control.values()}
    suites |= {tuple(sorted(run)) for arm in arms.values() for run in arm.values()}
    if len(suites) != 1:
        raise SystemExit(f"every seed needs the same suites: {sorted(suites)}")
    return control, arms, warnings


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("spec", help="JSON file (module docstring)")
    parser.add_argument("--out-dir", default=None, help="write report.md, report.json and ratios")
    args = parser.parse_args(argv)
    spec = json.loads(Path(args.spec).read_text())
    with tempfile.TemporaryDirectory() as cache:
        control, arms, warnings = load(spec, Path(cache))
    report = {name: compare_arm(control, arm) for name, arm in arms.items()}
    text = format_report(report)
    if warnings:
        text += "\n" + "\n".join(warnings) + "\n"
    print(text)
    if args.out_dir:
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "report.md").write_text(text)
        (out / "report.json").write_text(to_json(report))
        for name, parts in report.items():
            for part, r in parts.items():
                r["ratios"].to_csv(out / f"ratios_{name}_{part}.csv", index=False)


if __name__ == "__main__":
    main()
