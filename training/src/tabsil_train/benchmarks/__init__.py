"""Evaluation benchmarks, one folder per task of `tabsil.tasks` (classification and regression
share `tabular`).

- `tabular`: `amlb` (the AMLB and OpenML-CTR23 suites from the TabPFN v2 paper), `tabarena` and
  `beyondarena` (TabArena-Lite and BeyondArena, with their shared code in `arena`), `uncertainty`
  (predictive-distribution and uncertainty metrics), `paired` (the paired comparison of two runs)
  and `subsample` (at most N training rows per fit)
- `forecast`: `fevbench`, time series forecasting on 10 fev-bench tasks
- `unsupervised`: `adbench` (ADBench outlier detection) and `impute` (missing value imputation)
- `causal`: `causal`, treatment effects on IHDP and ACIC 2016

The reference results copied into the repository (fev, ADBench, causal) are files next to the
module that reads them. This folder holds the helpers that more than one benchmark uses: `_elo`
(the Elo of a pool of methods; fev and ADBench), `_files` (data files pinned by SHA256 and the
per-cell `results.csv`; ADBench, impute and causal) and `_retry` (retries of the data downloads of
AMLB and the arenas; `_files` does not retry).
"""
