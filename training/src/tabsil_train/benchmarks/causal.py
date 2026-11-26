"""Treatment effect benchmarks: IHDP (100 realizations) and ACIC 2016 (10 simulations), TabSil
meta-learners (`tabsil.causal.TabSilCATE`) against EconML baselines run on the same splits.

Benchmarks (`BENCHMARKS`):

- `ihdp`: the IHDP semi-synthetic benchmark of Hill (2011) in the 100-realization files of
  Shalit et al. (2017) / CFRNet (`ihdp_npci_1-100.{train,test}.npz` from fredjo.com, the files
  CausalPFN uses). Each realization has its fixed 672 training and 75 test rows (25 covariates);
  the model is fitted on the training rows' factual outcome `yf`. Within-sample metrics are over
  the 672 training rows (Shalit's train + validation rows), out-of-sample over the 75 test rows
- `acic2016`: the ACIC 2016 competition data (Dorie et al. 2019) in the 10 simulations shipped
  with causallib (`x.csv`, `zymu_<i>.csv`, 4,802 rows, 58 covariates), split as CausalPFN's
  `benchmarks/acic2016.py`: rows permuted with `np.random.default_rng(42 + i - 1)` for simulation
  i, the first 90% (4,321) train, the rest (481) test; the categorical columns x_2, x_21, x_24
  become category codes appended after the other columns; the observed outcome is y1 for treated
  and y0 for control rows

The true effect is always the noiseless τ = mu1 − mu0 (Hill 2011, Shalit 2017). Metrics per
realization: √PEHE = sqrt(mean((τ̂ − τ)²)) and ε_ATE = |mean τ̂ − mean τ|, within-sample (`_in`)
and out-of-sample (`_out`), and the relative ATE error ε_ATE / |ATE| within-sample (`ate_rel_in`,
the metric of CausalPFN's tables; theirs fits on all rows). The summary is the mean ± standard
error over realizations. Published numbers are in `causal_reference.csv` (`reference`), with
their protocols.

Methods (`METHODS`):

- `tabsil_s`, `tabsil_t`, `tabsil_x`, `tabsil_dr`: `TabSilCATE` with that learner, the model's
  inference defaults (8 members), random_state = the realization number + the `--seed` offset
- `econml_cf`: EconML `CausalForestDML(discrete_treatment=True)` with its default nuisance models
  and 100 trees
- `econml_s`, `econml_t`, `econml_x`, `econml_dr`: EconML `SLearner`, `TLearner`, `XLearner`,
  `DRLearner`, every outcome / effect / final model scikit-learn `GradientBoostingRegressor()` and
  every propensity model `GradientBoostingClassifier()` (defaults: 100 trees of depth 3); the
  other EconML options at their defaults (DR: cv=2, min_propensity=1e-6)

Data is downloaded on first use into `$TABSIL_CAUSAL_DIR` (default `~/.cache/tabsil/causal`) and
checked against `SHA256`; it is not redistributed (IHDP has no stated data license; the ACIC files
are under CDLA-Sharing-1.0, LICENSES/ACIC2016-CDLA-Sharing-1.0.txt).

`run` appends one row per (benchmark, realization, method) to `results.csv` as it goes and skips
rows already there, so a rerun resumes. A failed cell (an exception or a non-finite effect) is
recorded in `failures.json`, the others still run, and the caller ends with an error; nothing is
imputed.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import traceback
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

IHDP_URL = "https://www.fredjo.com/files"
CAUSALLIB_COMMIT = "267e0c72df81ea8264ece46496107f99a599b60d"
ACIC_URL = (
    f"https://raw.githubusercontent.com/BiomedSciAI/causallib/{CAUSALLIB_COMMIT}"
    "/causallib/datasets/data/acic_challenge_2016"
)
SHA256 = {
    "ihdp_npci_1-100.train.npz": "750697c71b4f8d7a3aafff771b56a4ac4cd83ec649bf69afb04f8a5aee41a240",
    "ihdp_npci_1-100.test.npz": "a70a8acbcc4e8deb677cc9bf9e9dabeb17caaa37cdbb1d7ba06be7ffb929c41c",
    "x.csv": "0d6387ad45d23e54b11cbf967248fb68ad5e578db80d193208b8b4c98b1ed0c1",
    "zymu_1.csv": "002fad96bb55d54edec08a0af46ef4f8ad69e0cfe6294ab48f351f22976a6e76",
    "zymu_2.csv": "bc3a03bb461bb1629add09c01dfc77053cf36e8faf8e48a7645b35ddb4005d1c",
    "zymu_3.csv": "8090562d3fcf5a3b02ff5493592a2880b95b106e1c12e2c3a6e6c28ae8468ff6",
    "zymu_4.csv": "61b405c545a451b7e2ce31614150834f94aa4b27069e2755876ed6e11cfee5fd",
    "zymu_5.csv": "e350ac76d0b8e83f45f7f04a4b76b9a32a0d53098452075617e61ce62ad0463a",
    "zymu_6.csv": "a059950f0b51056660364eca14524a3d3254f93c608d8a1df9240c2c4179a7b5",
    "zymu_7.csv": "ac102c3f34680ff901c30fb95985297eea0f20efaf02108f287ceec60b6a2a1f",
    "zymu_8.csv": "bbcde6ad666640adcd39fd50cab00865d22b5c463ce9db6d0940d8eccd5afcf3",
    "zymu_9.csv": "02663b776fa87cd3d91ee7871e98c1b4d63579957b12ef9bb1ecf367b7afc366",
    "zymu_10.csv": "c8ee8ca537ef4315a0ee5753bd01d792a1b3cdacb1fd69e1288b2bb89944d1b7",
}
# benchmark -> number of realizations (numbered from 1)
BENCHMARKS = {"ihdp": 100, "acic2016": 10}
ACIC_CATEGORICAL = ("x_2", "x_21", "x_24")
ACIC_TEST_RATIO, ACIC_SEED = 0.1, 42  # CausalPFN ACIC2016Dataset defaults

TABSIL = {"tabsil_s": "s", "tabsil_t": "t", "tabsil_x": "x", "tabsil_dr": "dr"}
ECONML = ("econml_cf", "econml_s", "econml_t", "econml_x", "econml_dr")
METHODS = (*TABSIL, *ECONML)
METRICS = ("pehe_in", "pehe_out", "ate_err_in", "ate_err_out", "ate_rel_in")
RESULT_COLUMNS = [
    "benchmark", "realization", "method", *METRICS, "ate_true_in", "ate_hat_in", "ate_true_out",
    "ate_hat_out", "seconds", "n_train", "n_test", "n_treated", "n_features", "opts",
]  # fmt: skip
REFERENCE_PATH = Path(__file__).parent / "causal_reference.csv"


@dataclass
class Split:
    """One realization: training rows (X, t, y) and the true effects of training and test rows."""

    X_train: np.ndarray
    t_train: np.ndarray
    y_train: np.ndarray
    tau_train: np.ndarray
    X_test: np.ndarray
    tau_test: np.ndarray


# --- data ---------------------------------------------------------------------------------------


def cache_dir() -> Path:
    return Path(os.environ.get("TABSIL_CAUSAL_DIR", Path.home() / ".cache" / "tabsil" / "causal"))


def data_path(name: str, root: Path | None = None) -> Path:
    """The cached file `name`, downloaded the first time and checked against SHA256."""
    path = (root or cache_dir()) / name
    if not path.exists():
        url = f"{IHDP_URL}/{name}" if name.startswith("ihdp") else f"{ACIC_URL}/{name}"
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        with urllib.request.urlopen(url, timeout=300) as r:
            tmp.write_bytes(r.read())
        tmp.replace(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != SHA256[name]:
        raise ValueError(f"{path}: sha256 {digest} is not the pinned file ({SHA256[name]})")
    return path


def _ihdp(root: Path | None = None) -> tuple[dict, dict]:
    def load(name):
        with np.load(data_path(name, root), allow_pickle=False) as d:  # downloaded: no pickle
            return {k: d[k] for k in ("x", "t", "yf", "mu0", "mu1")}

    return load("ihdp_npci_1-100.train.npz"), load("ihdp_npci_1-100.test.npz")


def ihdp_split(realization: int, data: tuple[dict, dict] | None = None) -> Split:
    """IHDP realization 1..100: its own 672 training and 75 test rows."""
    if not 1 <= realization <= BENCHMARKS["ihdp"]:
        raise ValueError(f"IHDP realizations are 1..100: {realization}")
    train, test = data or _ihdp()
    r = realization - 1
    return Split(
        X_train=train["x"][..., r].astype(float),
        t_train=train["t"][:, r].astype(int),
        y_train=train["yf"][:, r].astype(float),
        tau_train=(train["mu1"][:, r] - train["mu0"][:, r]).astype(float),
        X_test=test["x"][..., r].astype(float),
        tau_test=(test["mu1"][:, r] - test["mu0"][:, r]).astype(float),
    )


def acic_covariates(root: Path | None = None) -> np.ndarray:
    """ACIC 2016 covariates as CausalPFN encodes them: the 55 other columns in file order, then
    the category codes of x_2, x_21, x_24 (pandas category codes: sorted categories)."""
    x = pd.read_csv(data_path("x.csv", root))
    x.columns = [f"x_{i + 1}" for i in range(x.shape[1])]
    numeric = [c for c in x.columns if c not in ACIC_CATEGORICAL]
    codes = [x[c].astype("category").cat.codes.rename(f"{c}_numeric") for c in ACIC_CATEGORICAL]
    return pd.concat([x[numeric], *codes], axis=1).to_numpy(dtype=float)


def acic_split(simulation: int, X: np.ndarray | None = None, root: Path | None = None) -> Split:
    """ACIC 2016 simulation 1..10 split 90/10 as CausalPFN (module docstring)."""
    if not 1 <= simulation <= BENCHMARKS["acic2016"]:
        raise ValueError(f"ACIC 2016 simulations are 1..10: {simulation}")
    X = acic_covariates(root) if X is None else X
    sim = pd.read_csv(data_path(f"zymu_{simulation}.csv", root))
    sim.columns = ["z", "y0", "y1", "mu0", "mu1"]
    t = sim["z"].to_numpy().astype(int)
    y = np.where(t == 1, sim["y1"], sim["y0"]).astype(float)
    tau = (sim["mu1"] - sim["mu0"]).to_numpy(dtype=float)
    idx = np.random.default_rng(ACIC_SEED + simulation - 1).permutation(len(X))
    cut = int(len(idx) * (1 - ACIC_TEST_RATIO))
    tr, te = idx[:cut], idx[cut:]
    return Split(X[tr], t[tr], y[tr], tau[tr], X[te], tau[te])


def splits(benchmark: str, realizations: list[int]):
    """Yield (realization, Split) for the given realizations of `benchmark` (data loaded once)."""
    if benchmark == "ihdp":
        data = _ihdp()
        for r in realizations:
            yield r, ihdp_split(r, data)
    elif benchmark == "acic2016":
        X = acic_covariates()
        for r in realizations:
            yield r, acic_split(r, X)
    else:
        raise ValueError(f"unknown benchmark {benchmark!r} (known: {list(BENCHMARKS)})")


# --- methods ------------------------------------------------------------------------------------

# An estimator: fit(split, seed) -> (effects on the training rows, effects on the test rows)
Estimator = Callable[[Split, int], tuple[np.ndarray, np.ndarray]]


def check_methods(methods: list[str]) -> list[str]:
    if bad := [m for m in methods if m not in METHODS]:
        raise ValueError(f"unknown methods {bad} (known: {list(METHODS)})")
    return methods


def tabsil_estimator(model, borders, learner: str, n_estimators: int | None, device: str,
                     seed_offset: int = 0) -> Estimator:  # fmt: skip
    """`TabSilCATE(learner)` fitted on the training rows; random_state = seed + `seed_offset`."""
    from tabsil.causal import TabSilCATE

    def estimate(split: Split, seed: int):
        est = TabSilCATE(
            model,
            borders,
            learner=learner,
            n_estimators=n_estimators,
            device=device,
            random_state=seed + seed_offset,
        ).fit(split.X_train, split.t_train, split.y_train)
        return est.effect(split.X_train), est.effect(split.X_test)

    return estimate


def econml_estimator(method: str) -> Estimator:
    """The EconML baseline `method` (module docstring), random_state = the seed."""

    def estimate(split: Split, seed: int):
        from econml.dml import CausalForestDML
        from econml.dr import DRLearner
        from econml.metalearners import SLearner, TLearner, XLearner
        from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

        def gbr():
            return GradientBoostingRegressor(random_state=seed)

        def gbc():
            return GradientBoostingClassifier(random_state=seed)

        if method == "econml_cf":
            est = CausalForestDML(discrete_treatment=True, random_state=seed)
        elif method == "econml_s":
            est = SLearner(overall_model=gbr())
        elif method == "econml_t":
            est = TLearner(models=gbr())
        elif method == "econml_x":
            est = XLearner(models=gbr(), propensity_model=gbc())
        elif method == "econml_dr":
            est = DRLearner(
                model_propensity=gbc(), model_regression=gbr(), model_final=gbr(),
                random_state=seed,
            )  # fmt: skip
        else:
            raise ValueError(f"not an EconML method: {method}")
        est.fit(split.y_train, split.t_train, X=split.X_train)
        return np.ravel(est.effect(split.X_train)), np.ravel(est.effect(split.X_test))

    return estimate


def metrics(split: Split, effect_in: np.ndarray, effect_out: np.ndarray) -> dict:
    """√PEHE and ε_ATE within-sample (training rows) and out-of-sample (test rows)."""
    out = {}
    for key, hat, tau in (("in", effect_in, split.tau_train), ("out", effect_out, split.tau_test)):
        hat = np.asarray(hat, dtype=float)
        if hat.shape != tau.shape or not np.isfinite(hat).all():
            raise ValueError(
                f"{key}-sample effects of shape {hat.shape} (expected {tau.shape}) with "
                f"{int((~np.isfinite(hat)).sum())} non-finite values"
            )
        out[f"pehe_{key}"] = float(np.sqrt(np.mean((hat - tau) ** 2)))
        out[f"ate_err_{key}"] = float(abs(hat.mean() - tau.mean()))
        out[f"ate_true_{key}"] = float(tau.mean())
        out[f"ate_hat_{key}"] = float(hat.mean())
    out["ate_rel_in"] = out["ate_err_in"] / abs(out["ate_true_in"])
    return out


# --- run ----------------------------------------------------------------------------------------


def read_results(path: Path) -> pd.DataFrame:
    if path.exists() and path.stat().st_size:
        return pd.read_csv(path)
    return pd.DataFrame(columns=RESULT_COLUMNS)


def parse_realizations(text: str | None, benchmark: str) -> list[int]:
    """`None`: all; else comma-separated numbers and ranges (`1-5,9`), within the benchmark's."""
    n = BENCHMARKS[benchmark]
    if not text:
        return list(range(1, n + 1))
    out: list[int] = []
    for part in text.split(","):
        lo, _, hi = part.strip().partition("-")
        out += range(int(lo), int(hi or lo) + 1)
    if bad := [r for r in out if not 1 <= r <= n]:
        raise ValueError(f"{benchmark} realizations are 1..{n}: {bad}")
    return sorted(set(out))


def run(
    estimators: dict[str, Estimator],
    plan: dict[str, list[int]],
    methods: list[str],
    out_dir: Path,
    opts: dict[str, str] | None = None,
    log: Callable[[str], None] = print,
) -> tuple[pd.DataFrame, list[dict]]:
    """Every (benchmark, realization, method) of `plan` (benchmark -> realizations) not yet in
    `out_dir/results.csv`, appending a row as each finishes. Returns (all rows, failures)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "results.csv"
    done = read_results(path)
    finished = set(zip(done.benchmark, done.realization.astype(int), done.method, strict=True))
    failures = []
    for benchmark, realizations in plan.items():
        todo = [r for r in realizations if any((benchmark, r, m) not in finished for m in methods)]
        for r, split in splits(benchmark, todo):
            for method in methods:
                if (benchmark, r, method) in finished:
                    continue
                t0 = time.time()
                try:
                    row = metrics(split, *estimators[method](split, r))
                except Exception as e:  # noqa: BLE001 - record and go on; the run fails at the end
                    failures.append(
                        dict(benchmark=benchmark, realization=r, method=method, error=repr(e))
                    )
                    log(f"[causal] {benchmark} {r} {method} FAILED:\n{traceback.format_exc()}")
                    continue
                row |= dict(
                    benchmark=benchmark, realization=r, method=method,
                    seconds=round(time.time() - t0, 2), n_train=len(split.X_train),
                    n_test=len(split.X_test), n_treated=int(split.t_train.sum()),
                    n_features=split.X_train.shape[1], opts=(opts or {}).get(method, ""),
                )  # fmt: skip
                pd.DataFrame([row], columns=RESULT_COLUMNS).to_csv(
                    path, mode="a", header=not path.exists() or not path.stat().st_size, index=False
                )
                log(
                    f"[causal] {benchmark} {r} {method}: sqrt PEHE in {row['pehe_in']:.3f} out "
                    f"{row['pehe_out']:.3f}, ATE err in {row['ate_err_in']:.3f} out "
                    f"{row['ate_err_out']:.3f} ({row['seconds']:.1f}s)"
                )
    if failures:
        (out_dir / "failures.json").write_text(json.dumps(failures, indent=1))
    return read_results(path), failures


# --- report -------------------------------------------------------------------------------------


def summary_table(rows: pd.DataFrame) -> pd.DataFrame:
    """Per (benchmark, method): realizations, mean and standard error of each metric, and the
    mean seconds per realization."""
    out = []
    for (benchmark, method), g in rows.groupby(["benchmark", "method"], sort=True):
        rec = {"benchmark": benchmark, "method": method, "realizations": len(g)}
        for m in METRICS:
            v = g[m].astype(float)
            rec[m] = v.mean()
            rec[f"{m}_se"] = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan
        rec["seconds"] = g["seconds"].astype(float).mean()
        out.append(rec)
    return pd.DataFrame(out)


def reference() -> pd.DataFrame:
    """Published numbers (`causal_reference.csv`): source, protocol, method, benchmark and metric
    columns as in `summary_table` (blank where the source has no number)."""
    return pd.read_csv(REFERENCE_PATH, comment="#")


def report(rows: pd.DataFrame, out_dir: Path, plan: dict[str, list[int]],
           methods: list[str]) -> dict:  # fmt: skip
    """Write `summary.csv` and return the summary dict: per benchmark and method the metric means
    and the share of planned cells without a result (`missing`, must be 0)."""
    rows = rows[rows.method.isin(methods) & rows.benchmark.isin(list(plan))]
    rows = rows[[r in plan[b] for b, r in zip(rows.benchmark, rows.realization, strict=True)]]
    table = summary_table(rows)
    table.to_csv(out_dir / "summary.csv", index=False)
    planned = sum(len(v) for v in plan.values()) * len(methods)
    summary = {
        "planned_cells": planned,
        "results": len(rows),
        "missing_pct": 100 * (planned - len(rows)) / planned if planned else 0.0,
        "imputed_pct": 0.0,  # nothing is ever imputed: a missing cell fails the run
        "methods": {},
    }
    for rec in table.to_dict("records"):
        key = f"{rec['benchmark']}/{rec['method']}"
        summary["methods"][key] = {m: round(rec[m], 4) for m in METRICS} | {
            "realizations": rec["realizations"],
            "seconds": round(rec["seconds"], 2),
        }
    return summary
