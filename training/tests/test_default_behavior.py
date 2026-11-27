"""The default settings train exactly as before the v2 recipe options (`default_behavior.py`).

The values in `data/default_behavior.json` were written by `default_behavior.py` on commit 01973a5,
before the options existed: batches and regression borders of the `graph_scm`,
`graph_scm_hicard_missing` and `graph_scm_realism` priors, the validation batches, and a short
smoke training run (losses, weights, optimizer state). The fingerprints are compared with a small
relative tolerance so that a different BLAS does not fail the test; any change in the data or the
training computation moves them by far more.
"""

import json
import math
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPECTED = HERE / "data" / "default_behavior.json"


def _compare(expected, actual, path="") -> list[str]:
    if isinstance(expected, dict):
        if set(expected) != set(actual):
            return [f"{path}: keys {sorted(expected)} != {sorted(actual)}"]
        return [d for k in expected for d in _compare(expected[k], actual[k], f"{path}/{k}")]
    if isinstance(expected, list):
        if len(expected) != len(actual):
            return [f"{path}: length {len(expected)} != {len(actual)}"]
        return [
            d
            for i, (e, a) in enumerate(zip(expected, actual, strict=True))
            for d in _compare(e, a, f"{path}[{i}]")
        ]
    if isinstance(expected, float) and not isinstance(actual, bool):
        if math.isclose(expected, actual, rel_tol=1e-6, abs_tol=1e-9):
            return []
        return [f"{path}: {expected} != {actual}"]
    return [] if expected == actual else [f"{path}: {expected!r} != {actual!r}"]


def test_default_settings_train_as_before(tmp_path):
    out = tmp_path / "actual.json"
    env = {**os.environ, "PYTHONHASHSEED": "0"}  # the graph SCM prior depends on it
    subprocess.run(
        [sys.executable, str(HERE / "default_behavior.py"), str(out)],
        env=env,
        check=True,
        capture_output=True,
    )
    differences = _compare(json.loads(EXPECTED.read_text()), json.loads(out.read_text()))
    assert not differences, "\n".join(differences[:20])
