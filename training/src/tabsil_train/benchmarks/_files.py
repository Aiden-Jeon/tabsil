"""Benchmark data files pinned by SHA256, and per-cell result tables, shared by the suites that
download their data at run time and write one result row per cell (`adbench`, `causal`).

- `download_verified`: a file is downloaded once into a cache. The download goes to `<name>.part`
  and is moved into place only when its SHA256 is the pinned one, so a truncated download or an
  error page is not cached (it is deleted and the call raises). A cached file is checked again on
  every use
- `read_results`: a suite's `results.csv` (appended one row per finished cell, so a rerun resumes),
  or an empty table with the suite's columns
"""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

import pandas as pd


def _check(path: Path, sha256: str, what: str) -> None:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != sha256:
        raise ValueError(f"{path}: sha256 {digest} is not {what} ({sha256})")


def download_verified(
    url: str, path: Path, sha256: str, *, what: str = "the pinned file", timeout: float = 300
) -> Path:
    """`path`, downloaded from `url` if it is not there yet, and checked against `sha256`. `what`
    names the expected file in the error message."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".part")
        with urllib.request.urlopen(url, timeout=timeout) as r:
            tmp.write_bytes(r.read())
        try:
            _check(tmp, sha256, what)
        except ValueError:
            tmp.unlink()
            raise
        tmp.replace(path)
    _check(path, sha256, what)
    return path


def read_results(path: Path, columns: list[str]) -> pd.DataFrame:
    """The result rows in `path`, or an empty table with `columns` if there are none yet."""
    if path.exists() and path.stat().st_size:
        return pd.read_csv(path)
    return pd.DataFrame(columns=columns)
