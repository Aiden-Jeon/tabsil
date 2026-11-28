"""Run the v2 prior diagnostics (the three of `DEFAULT`, or the names given) one after another and
log each one's outputs (`DIAG_OUT`: CSV, stdout, stderr) under `diag/` of the MLflow run (the
platform's run, `MLFLOW_RUN_ID`, on AI Runtime) as soon as it ends, so a later failure or the job
timeout does not lose them. Each diagnostic gets at most `DIAG_TIMEOUT` seconds. A failed
diagnostic does not stop the others, but the script exits non-zero at the end. The settings that
the diagnostics read from the environment (`SETTINGS`) are logged as parameters."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import mlflow

HERE = Path(__file__).resolve().parent
DEFAULT = ("d1_forecast", "d2_calibration", "d3_contamination")
DIAGNOSTICS = (*DEFAULT, "d2_rare_class", "d4_missing", "d5_overlap", "fev_arms", "adbench_arms")
SETTINGS = (
    "D1_ARMS", "D3_ARMS", "D3_KINDS", "D4_PARTS", "D4_TABLES", "D4_BASELINES", "FEV_ARM",
    "FEV_TASKS", "FEV_WINDOWS", "ADBENCH_ARMS", "ADBENCH_DATASETS", "ADBENCH_COMBINE",
)  # fmt: skip


def run(name: str, out_dir: Path, timeout: float) -> int:
    """Run one diagnostic; its stdout and stderr go to `<name>.out` / `<name>.err`."""
    out, err = out_dir / f"{name}.out", out_dir / f"{name}.err"
    with open(out, "w") as o, open(err, "w") as e:
        try:
            code = subprocess.run(
                [sys.executable, str(HERE / f"{name}.py")], stdout=o, stderr=e, timeout=timeout
            ).returncode
        except subprocess.TimeoutExpired:
            e.write(f"\ntimed out after {timeout:.0f} s\n")
            code = -1
    print(f"{name} exit={code}\n{out.read_text()}", flush=True)
    if code:
        print(err.read_text()[-3000:], flush=True)
    return code


def main(names: list[str]) -> None:
    unknown = sorted(set(names) - set(DIAGNOSTICS))
    if unknown:
        sys.exit(f"unknown diagnostics {unknown}; choose from {DIAGNOSTICS}")
    out_dir = Path(os.environ.setdefault("DIAG_OUT", "/tmp/v2_prior_diag"))
    out_dir.mkdir(parents=True, exist_ok=True)
    timeout = float(os.environ.get("DIAG_TIMEOUT", "1500"))
    failed = []
    with mlflow.start_run(run_id=os.environ.get("MLFLOW_RUN_ID")):
        mlflow.log_params(
            {
                "diagnostics": ",".join(names),
                "model": os.environ.get("DIAG_MODEL", "default"),
                **{k: os.environ[k] for k in SETTINGS if os.environ.get(k)},
            }
        )
        for name in names:
            if run(name, out_dir, timeout):
                failed.append(name)
            for path in sorted(out_dir.glob(f"{name}*")):
                mlflow.log_artifact(str(path), "diag")
    if failed:
        sys.exit(f"failed diagnostics: {failed}")


if __name__ == "__main__":
    main(sys.argv[1:] or list(DEFAULT))
