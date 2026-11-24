"""Evaluation benchmarks.

- `amlb`: the AMLB and OpenML-CTR23 suites from the TabPFN v2 paper, with the paper's scores
- `tabarena`, `beyondarena`: TabArena-Lite and BeyondArena (same splits and aggregation as the
  public leaderboards; shared code lives in `arena`; requires the `tabarena` package from the
  `arena` extra)
- `uncertainty`: predictive-distribution and uncertainty metrics (regression CRPS, interval
  coverage, PIT; classification ECE, conformal coverage)
"""
