"""The README charts match the tables they are drawn from (assets/charts/make_charts.py)."""

import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "assets" / "charts" / "make_charts.py"


def test_charts_are_up_to_date():
    out = subprocess.run(
        [sys.executable, "-I", str(SCRIPT), "--check"], capture_output=True, text=True
    )
    assert out.returncode == 0, out.stdout + out.stderr
