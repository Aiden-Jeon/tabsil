"""Run the v2 prior diagnostics (all three, or the names given) one after another and log their
outputs (`DIAG_OUT`: CSV, stdout, stderr) under `diag/` of the MLflow run (the platform's run,
`MLFLOW_RUN_ID`, on AI Runtime). A failed diagnostic does not stop the others, but the script
exits non-zero at the end."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import mlflow

HERE = Path(__file__).resolve().parent
DIAGNOSTICS = ("d1_forecast", "d2_calibration", "d3_contamination")


def main(names: list[str]) -> None:
    unknown = sorted(set(names) - set(DIAGNOSTICS))
    if unknown:
        sys.exit(f"unknown diagnostics {unknown}; choose from {DIAGNOSTICS}")
    out_dir = Path(os.environ.setdefault("DIAG_OUT", "/tmp/v2_prior_diag"))
    out_dir.mkdir(parents=True, exist_ok=True)
    failed = []
    for name in names:
        out, err = out_dir / f"{name}.out", out_dir / f"{name}.err"
        with open(out, "w") as o, open(err, "w") as e:
            code = subprocess.call([sys.executable, str(HERE / f"{name}.py")], stdout=o, stderr=e)
        print(f"{name} exit={code}\n{out.read_text()}", flush=True)
        if code:
            failed.append(name)
            print(err.read_text()[-3000:], flush=True)
    with mlflow.start_run(run_id=os.environ.get("MLFLOW_RUN_ID")):
        mlflow.log_params({"diagnostics": ",".join(names)})
        mlflow.log_artifacts(str(out_dir), "diag")
    if failed:
        sys.exit(f"failed diagnostics: {failed}")


if __name__ == "__main__":
    main(sys.argv[1:] or list(DIAGNOSTICS))
