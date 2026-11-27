"""Benchmarks of `tabsil.TabSilUnsupervised` (`tabsil.tasks.unsupervised`): outlier detection and
imputation.

- `adbench`: unsupervised outlier detection on 12 ADBench tables x 3 seeds, with the comparison
  results copied from fm-agent-bench (`adbench_reference/`)
- `impute`: missing value imputation on 10 OpenML tables x MCAR rates 10/30/50% x 3 seeds (the
  TabPFN extension baselines require the `impute` extra)
"""
