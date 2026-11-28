"""fev-bench with one forecast arm of `arms.py` (plan item 1.2): the forecasting suite
(`tabsil_train.benchmarks.forecast.fevbench`, as `eval.py --suite fev` runs it: every window of
its 10 tasks, the weights' member count, seed 0) with `arms.ArmForecaster` in place of
`TabSilForecaster`. The `default` arm repeats the suite.

`FEV_ARM` names the arm. `FEV_TASKS` (comma-separated) and `FEV_WINDOWS` (the last windows of each
task) shorten the run for a check. The suite's files (`windows.csv`, `comparison.csv`, `elo.csv`,
`summary.json`) are written as `fev_arms_<arm>_<file>` in the output folder.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pandas as pd
from arms import ArmForecaster
from common import DEVICE, OUT, load_model

from tabsil_train.benchmarks.forecast import fevbench

FILES = ("windows.csv", "comparison.csv", "elo.csv", "summary.json")


def main() -> None:
    arm = os.environ.get("FEV_ARM", "default")
    tasks = [t for t in os.environ.get("FEV_TASKS", "").split(",") if t] or None
    windows = int(os.environ["FEV_WINDOWS"]) if os.environ.get("FEV_WINDOWS") else None
    model, borders = load_model()
    forecaster = ArmForecaster(model, borders, arm, device=DEVICE, random_state=0)
    folder = Path(OUT) / "fev" / arm
    summary = fevbench.run(forecaster, tasks, windows, folder, log=lambda s: print(s, flush=True))
    for name in FILES:
        shutil.copy(folder / name, Path(OUT) / f"fev_arms_{arm}_{name}")
    table = pd.read_csv(folder / "comparison.csv")
    for metric in ("SQL", "MASE"):
        ours = table[table.method == fevbench.METHOD].set_index("task")[metric]
        print(f"\narm {arm}, task mean {metric}:\n{ours.round(4).to_string()}")
    elo = pd.read_csv(folder / "elo.csv")
    print(f"\narm {arm}, Elo (Seasonal Naive = 1000):\n{elo.round(1).to_string(index=False)}")
    print(f"TabSil time {summary['tabsil_seconds']:.0f}s")


if __name__ == "__main__":
    main()
