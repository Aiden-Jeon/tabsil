"""Evaluation benchmarks, one folder per task (the tasks of `tabsil.tasks`).

- `tabular`: classification and regression. `amlb` (the AMLB and OpenML-CTR23 suites from the
  TabPFN v2 paper), `tabarena` and `beyondarena` (TabArena-Lite and BeyondArena, with their shared
  code in `arena`), `uncertainty` (predictive-distribution and uncertainty metrics) and `subsample`
  (at most N training rows per fit)
- `forecast`: `fevbench`, time series forecasting on 10 fev-bench tasks
- `unsupervised`: `adbench` (ADBench outlier detection) and `impute` (missing value imputation)
- `causal`: `causal`, treatment effects on IHDP and ACIC 2016

A benchmark's reference results are files next to its module. Shared by the tasks, in this
folder: `_elo` (the Elo of a pool of methods), `_files` (data files pinned by SHA256 and the
per-cell `results.csv`) and `_retry` (retries of benchmark data downloads).
"""
