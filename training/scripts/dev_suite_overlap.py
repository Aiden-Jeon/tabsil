"""Which of the 57 AMLB / CTR23 tasks (`tabsil_train.benchmarks.amlb`) have data that is also in
TabArena v0.1's 51 datasets: the check behind `amlb.TABARENA_OVERLAP` and the development suite
(`--suite dev` of the eval script).

The names are compared after lower-casing and keeping only letters and digits. Name matches are
printed first, then both lists with their sizes and sources, so a person can find the datasets
that are the same data under another name (`TABARENA_OVERLAP` has two: dna = splice and
wine-quality-white within wine_quality). The script ends with an error if `TABARENA_OVERLAP`
misses a name match or names a TabArena dataset that does not exist.

The TabArena list is the curated metadata of the `tabarena` package (the `arena` extra):
  uv run --package tabsil-train --no-dev --extra arena --group arena-dev \
    python training/scripts/dev_suite_overlap.py
or a copy of that CSV: `--metadata curated_tabarena_dataset_metadata.csv`.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from tabsil_train.benchmarks import amlb


def normalize(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def tabarena_metadata(path: str | None) -> pd.DataFrame:
    if path is None:
        import tabarena

        path = (
            Path(tabarena.__file__).parent
            / "benchmark/task/metadata/data/curated_tabarena_dataset_metadata.csv"
        )
    return pd.read_csv(path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--metadata", help="curated_tabarena_dataset_metadata.csv")
    args = parser.parse_args(argv)
    ta = tabarena_metadata(args.metadata)
    by_name = {normalize(n): n for n in ta["dataset_name"]}
    by_name.update(
        {
            normalize(n): d
            for n, d in zip(ta["openml_dataset_name"], ta["dataset_name"], strict=True)
        }
    )
    rows = []
    for task, table in (
        ("classification", amlb.PAPER_CLASSIFICATION),
        ("regression", amlb.PAPER_REGRESSION),
    ):
        for name, task_id, dataset_id, _ in table:
            rows.append(
                {
                    "task": task,
                    "name": name,
                    "task_id": task_id,
                    "dataset_id": dataset_id,
                    "tabarena": by_name.get(normalize(name)),
                }
            )
    ours = pd.DataFrame(rows)
    matches = ours[ours["tabarena"].notna()]
    print(f"name matches ({len(matches)}):")
    print(matches.to_string(index=False))
    print("\nAMLB / CTR23 tasks:")
    print(ours[["task", "name", "task_id", "dataset_id"]].to_string(index=False))
    print("\nTabArena v0.1 datasets:")
    columns = ["dataset_name", "problem_type", "num_instances", "num_features", "original_data_url"]
    print(ta[columns].to_string(index=False))
    print("\nTABARENA_OVERLAP:")
    for task_id, (dataset, how) in amlb.TABARENA_OVERLAP.items():
        print(f"  {task_id} {ours.set_index('task_id').loc[task_id, 'name']} -> {dataset} ({how})")
    missing = set(matches["task_id"]) - set(amlb.TABARENA_OVERLAP)
    unknown = {d for d, _ in amlb.TABARENA_OVERLAP.values()} - set(ta["dataset_name"])
    if missing or unknown:
        raise SystemExit(
            f"TABARENA_OVERLAP misses name matches {sorted(missing)} or names unknown"
            f" datasets {sorted(unknown)}"
        )
    n = {t: len(ids) for t, ids in amlb.DEV_SUITE.items()}
    print(
        f"\ndevelopment suite: {n['classification']} classification, {n['regression']} regression"
    )


if __name__ == "__main__":
    main()
