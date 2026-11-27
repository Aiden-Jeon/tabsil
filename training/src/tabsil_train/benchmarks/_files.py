"""Benchmark data files pinned by SHA256, and per-cell result tables, shared by the suites that
download their data at run time and write one result row per cell (`adbench`, `causal`, `impute`).

- `download_verified`: a file is downloaded once into a cache. The download goes to `<name>.part`
  and is moved into place only when its SHA256 is the pinned one, so a truncated download or an
  error page is not cached (it is deleted and the call raises). A cached file is checked again on
  every use
- `read_results`: a suite's `results.csv` (appended one row per finished cell, so a rerun resumes),
  or an empty table with the suite's columns
- `check_settings`: a rerun reuses a row only if it was computed with the same settings. Every row
  carries its method's settings string (`opts`); a row of a requested method with another string
  stops the run. A TabSil method's string is `model_settings` (the weights by digest, the member
  count, the seed, the inference settings saved with the weights), so another checkpoint in the
  same output folder is an error; a baseline's string does not name the model, so its rows are
  reused whatever the model
- `Results`: what the suites' `run` functions do with `results.csv` in one run (read and check
  the earlier rows, append a finished cell's rows, record a failed cell, write `failures.json`)
"""

from __future__ import annotations

import hashlib
import json
import traceback
import urllib.request
from collections.abc import Callable
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


def model_settings(model, borders, n_estimators: int, seed: int, **options) -> str:
    """The settings string (`opts`) of the rows a TabSil method computes with `model`: what the
    checkpoint and the command line put into them, as `key=value` words.

    - `weights`: the digest of the weights and borders (`tabsil_train.hub.weights_digest`), not
      the checkpoint path. New weights at an old path (`runs:/<id>/checkpoints/latest.pt`) give
      another string; the same weights at another path (the local copy `--shards` workers get)
      give the same one
    - `n_estimators` (the member count the estimators get), `seed` (the `--seed` offset) and the
      suite's own `options` (ADBench: `n_permutations`)
    - the inference settings saved with the weights (`tabsil.recipe.default_inference`), which
      `TabSilClassifier` and `TabSilRegressor` read: two Hub tags can hold the same weights with
      other settings

    The device is not in the string: rows computed on CUDA (bfloat16 autocast) and on the CPU
    (float32) differ slightly and are reused for each other.
    """
    from tabsil.recipe import default_inference
    from tabsil_train.hub import weights_digest

    values = {
        "weights": weights_digest(model, borders),
        "n_estimators": int(n_estimators),
        "seed": int(seed),
        **options,
    }
    # the same for both tasks, except the member count (given above) and the softmax temperature
    # (classification only)
    for key, value in default_inference(model, "classification").items():
        if key != "n_estimators":
            values[key] = "/".join(value) if isinstance(value, tuple | list) else value
    return " ".join(f"{key}={value}" for key, value in values.items())


def _opts(rows: pd.DataFrame) -> pd.Series:
    # an empty cell reads back as NaN; a method without settings has ""
    return rows.opts.astype(object).where(rows.opts.notna(), "").astype(str)


def other_settings(rows: pd.DataFrame, settings: dict[str, str]) -> pd.Series:
    """Per row: whether its method is in `settings` (result method -> its settings string in
    this run) and its `opts` is another string."""
    current = rows.method.map(lambda m: settings.get(m, ""))
    return rows.method.isin(list(settings)) & (_opts(rows) != current)


def check_settings(done: pd.DataFrame, settings: dict[str, str], path: Path) -> None:
    """ValueError if `done` (the rows of `path`) has a row of a method in `settings` (result
    method -> its settings string in this run) with another string: the row is of another model
    or other options, and resuming would mix it with this run's rows."""
    stale = other_settings(done, settings)
    if stale.any():
        method = done.method[stale].iloc[0]
        raise ValueError(
            f"{path} has {sorted(set(done.method[stale]))} results with other settings"
            f" ({_opts(done)[stale].iloc[0]!r}; this run: {settings[method]!r});"
            " use another output directory"
        )


class Results:
    """A suite's `results.csv` during one `run`: one row (or a few) per cell, appended as each
    cell finishes, so a rerun resumes after the cells already there.

    - opening creates `out_dir`, reads the rows of earlier runs (`done`) and stops if a method of
      this run has rows with other settings (`check_settings`; `settings`: result method -> its
      settings string)
    - `append` writes a finished cell's rows
    - `fail` records a cell that raised and logs it with its traceback; the run goes on and its
      caller ends with an error
    - `finish` writes `failures.json` if a cell failed and returns (all rows, failures)

    Which cells are finished and what counts as a bad output stay in the suite's `run`: the
    cell keys differ (an impute job writes several rows) and so do the checks.
    """

    def __init__(
        self,
        out_dir: Path,
        columns: list[str],
        settings: dict[str, str],
        label: str,
        log: Callable[[str], None] = print,
    ) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        self.out_dir = out_dir
        self.path = out_dir / "results.csv"
        self.columns = columns
        self.label = label
        self.log = log
        self.done = read_results(self.path, columns)
        check_settings(self.done, settings, self.path)
        self.failures: list[dict] = []

    def append(self, rows: list[dict]) -> None:
        header = not self.path.exists() or not self.path.stat().st_size
        table = pd.DataFrame(rows, columns=self.columns)
        table.to_csv(self.path, mode="a", header=header, index=False)

    def fail(self, cell: dict, error: Exception) -> None:
        """Record the failure of `cell` (its key columns, in the order they are logged)."""
        self.failures.append({**cell, "error": repr(error)})
        trace = "".join(traceback.format_exception(error))
        self.log(f"[{self.label}] {' '.join(map(str, cell.values()))} FAILED:\n{trace}")

    def finish(self) -> tuple[pd.DataFrame, list[dict]]:
        if self.failures:
            (self.out_dir / "failures.json").write_text(json.dumps(self.failures, indent=1))
        return read_results(self.path, self.columns), self.failures
