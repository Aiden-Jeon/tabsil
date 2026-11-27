"""The runs `training/scripts/eval.py` starts for the task suites, one module per suite: what
the command line asks for (model, grid, `--shards`), the benchmark's run and report, and the
MLflow logging. The benchmarks themselves (data, protocol, metrics, reports) are in
`tabsil_train.benchmarks`.

- `fev`: `--suite fev` (`run_fev`)
- `adbench`: `--suite adbench` (`run_adbench`)
- `causal`: `--suite causal` (`run_causal`)
- `impute`: `--suite impute` (`run_impute`)
- `shards`: the `--shards` worker queue of the ADBench and impute suites

The AMLB and TabArena / BeyondArena runs are in `eval.py` itself.
"""
