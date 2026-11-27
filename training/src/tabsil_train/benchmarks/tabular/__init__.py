"""Classification and regression benchmarks (`tabsil.tasks.classification`,
`tabsil.tasks.regression`).

- `amlb`: the AMLB and OpenML-CTR23 suites from the TabPFN v2 paper, with the paper's scores
- `tabarena`, `beyondarena`: TabArena-Lite and BeyondArena (same splits and aggregation as the
  public leaderboards; shared code lives in `arena`, and `_arena_system` wraps the estimator for
  the `tabarena` package; requires the `tabarena` package from the `arena` extra)
- `uncertainty`: predictive-distribution and uncertainty metrics (regression CRPS, interval
  coverage, PIT; classification ECE, conformal coverage)
- `paired`: the paired comparison of two evaluation runs on the same splits (the error ratio per
  dataset, its bootstrap interval and the Wilcoxon test; v2 experiment plan §3)
- `subsample`: `SubsampledEstimator`, which passes at most N training rows to fit (class-stratified
  for classification; for context-length experiments, `eval.py --subsample-train-rows`)
"""
