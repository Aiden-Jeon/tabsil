# v2 plan for priors and tasks: results

**Status:** tiers 1 and 2 are measured and reading points R1 and R2 are read, on 2026-11-28. Tier 3 is not started and no run is waiting.

> **Correction, 2026-11-28.** The user's goal is the accuracy on TabArena-Lite, and more diverse priors and tasks were asked for as the means to raise it. The plan took the task metrics (fev-bench, ADBench) as the goal, which was a misreading on the planner's side ([v2-prior-task-plan.md](v2-prior-task-plan.md), correction at its top). The numbers below stand as measurements, and the bars and reading points R1 and R2 are those of the plan as it was written. **What the user asked is answered in "TabArena-Lite of the fine-tuned weights": training on task-shaped tables alone makes TabArena-Lite much worse (Elo 1808 to 1252), and the test of mixing them into ordinary training has not been run.**

The plan is [v2-prior-task-plan.md](v2-prior-task-plan.md) and the derivations it tests are in [v2-task-formulations.md](v2-task-formulations.md). The code of every measurement is in [`training/scripts/v2_prior_diag`](../scripts/v2_prior_diag/README.md). "v1.1" is the released weights with tabsil 0.1.1's inference settings (`aiden-jeon/tabsil@v1.1`). "#14's run" and "N10" are the two seeds of the v2 candidate, stage-2 runs on `graph_scm_hicard_missing` (MLflow `277916ec05a3457aa6d07f2bc08a60c4` and `520d88d73cdc4e5ea90cfd27425be00d`, `model/final.pt`).

## Summary

| Task | Bar | v1.1 today | Best without training (tier 1) | Best after the fine-tuning (tier 2) | Outcome |
|---|---|---|---|---|---|
| Forecasting, fev-bench SQL Elo | 1707 | 1535 | 1535: today's recipe. No arm is better | **1582**: the pilot's weights, today's recipe. SQL 5.0% lower | below the bar: stops at R2 |
| Outlier detection, ADBench mean AUROC | 0.77 | 0.663 | **0.756**: the density under 32 random contexts of 64 rows | **0.759** with trained reference rows; **0.774** as the mean rank of the two scores | at the bar with a score chosen after the pilot: to the user |
| Imputation | none | D4 measured | — | — | to the user |
| Treatment effects | none | D5 measured | — | — | to the user |

Tier 1, without training:

- **Outlier detection gains 0.092 AUROC from inference alone.** Scoring each row under many small random contexts and averaging the log densities raises ADBench from 0.663 to 0.756, above IsolationForest (0.722) and the TabPFN v2 extension (0.704), and it removes D3's failure on a dense group at 2% (0.425 to 0.987).
- **No forecasting arm is better than today's recipe.** The parts of the recipe that hold a forecast inside the context's range are confirmed on D1, but removing them costs 2.2% to 2.6% of fev-bench's SQL. A straight-line trend fitted outside the model reaches the least-squares fit on D1's linear trends (MASE 0.37 instead of 11.75) and costs 3.5% on fev-bench: one task improves by 31% and another gets 41% worse.
- **The v2 candidate's weights are v1.1 on every task suite**, as predicted, with one exception that D4 found: they use a missing cell as information. Where a mask tells about the cell to impute they are 14% better than v1.1, and where it only looks as if it did they are 9% worse.
- **Three predictions of the derivations failed** and are corrected in [v2-task-formulations.md](v2-task-formulations.md): reference rows do not fail on a small dense group, the rare-class output is the class's share of the context only for two classes, and the context-row target scale may raise the trend capture by itself (section "Derivations against the measurements").

Tier 2, a fine-tuning of the candidate on task-shaped tables alone:

- **2.3 H100-h on one GPU are enough to teach both tasks.** After 2,000 steps at batch 32 the model continues a trend (D1's linear trend: MASE 12.5 to 0.66, where the least-squares fit has 0.37 and seasonal naive 3.67) and ranks a dense group in the context above the inliers at every contamination up to 20% (D3: 1.00, where the starting weights fall from 0.88 to 0.24). On the benchmarks that is 5.0% of fev-bench's SQL (Elo 1538 to 1582) and 0.088 AUROC on ADBench with reference rows (0.664 to 0.752).
- **Eight times the tables add little, and for forecasting nothing.** The probe (8 H100, batch 256) has 0.759 on ADBench and a forecast that overshoots: on fev-bench it is behind the pilot (SQL 3.8% lower than the start, Elo 1519).
- **Neither bar is reached by a planned score.** Forecasting is 125 Elo short at best. Outlier detection is 0.011 short with the trained reference rows, and 0.004 above with the mean rank of the trained reference rows and the small-context density, a score that was added after the pilot's results were seen.
- **On ordinary tables the fine-tuned weights are much worse.** TabArena-Lite falls from Elo 1808 to 1252 (pilot) and 1234 (probe), with classification errors 30% higher. They saw task-shaped tables only and are not candidates for anything; what a mix of ordinary and task-shaped tables does to TabArena-Lite, which is the user's question, has not been run ("TabArena-Lite of the fine-tuned weights").

**Cost:** 40.0 H100-h (tier 1 11.6 in 35 jobs, tier 2 27.6 in 18 jobs, three TabArena-Lite runs 0.9), against estimates of about 20 and 22. The cap is 500.

## Reading point R1

The rule is the user's (plan, section 5): the best arm at or above the bar ends the task without training; a distance to the bar of at most the largest gain others report from such training (180 SQL Elo, 0.13 AUROC) sends the task to the probe; a larger distance stops it.

- **Forecasting: in the probe.** The best is today's recipe at 1535, 172 below the bar.
- **Outlier detection: in the probe.** The best arm is 0.756, 0.014 below the bar. The arm is not at the bar, so the rule raises no release question; that it is an inference change worth 0.092 AUROC is put to the user in "For the user to decide".
- **Next by the plan:** the pilot of item 2.1 on one GPU, then the probe if the pilot moves a task's diagnostic.

Two things the rule does not ask, stated because they bear on what the probe can be expected to give:

- For forecasting, tier 1 lowers the expectation. What fails on D1 is the trend, and neither of the two inference changes that repair D1 helps fev-bench: with a line one task of ten improves by more than 2% (rohlik_orders_1D, by 31%) and six get worse by 2% to 41%. The probe trains the model on the same thing, a continuation beyond the context. The bar is the rating of TimesFM-2.5, which has a lower SQL than TabSil on 7 of the 10 tasks.
- For outlier detection, the probe trains the reference-row form, whose value without training is 0.665. The best score without training, 0.756, is another form (the density) and is not what the probe trains.

## Reading point R2

The rule (plan, section 5): a task whose probe is at or above the bar is in the branch of tier 3; a task below stops; if no task is left, the plan ends without tier 3.

- **Forecasting: stops.** The best is the pilot's weights with today's recipe, SQL Elo 1582, 125 below the bar; the probe's weights have 1519. The gain that is there is 4% to 5% of the SQL, on four of the ten tasks.
- **Outlier detection: not settled by the rule.** With the form the probe trains, reference rows, the probe's weights have 0.759, and with the best wrapper of tier 1 0.757: both below 0.77. The mean rank of the two has 0.774. It was not among the scores the plan named for this reading, it was added after the pilot showed the two scores strong on different tables, and its margin over the bar (0.004) is smaller than its standard error against either score alone (0.011 and 0.015).
- **Tier 3 is not started.** Forecasting has stopped; whether outlier detection is "at the bar" depends on a score chosen after the fact; and the GPU setup of tier 3 is the user's decision in any case (plan, section 12).

What tier 2 says about tier 3, beyond the rule:

- The plan's branch (10,000 steps of stage 2 with 10% task-shaped tables, 223 H100-h) was sized before it was known how quickly these tables teach. The pilot got its results in 2,000 steps at batch 32: 32,000 tables per task. The branch would show the model 128,000 per task, the probe showed 256,000 and was not better.
- For outlier detection, the branch's possible gain over the inference change of tier 1 is small: 0.756 (no training) against 0.759 to 0.774 (trained), except under heavy contamination with a dense group, where reference rows hold at 20% (D3: 1.00) and the small-context density does not (0.78).
- A fine-tuning of the candidate on a mix of ordinary and task-shaped tables, a few thousand steps, would test the same question as the branch (the task gains in one model, under the TabArena guard) at a tenth to a fifth of its cost, with the guard read against the weights it starts from. It is not in the plan and was not run.

## Tier 1

### 1.1 The v2 candidate's two seeds

Predicted: no change against v1.1 on forecasting and outlier detection. The difference between the two seeds is the noise of a task metric between two trainings.

| Suite | Metric | v1.1 | #14's run | N10 |
|---|---|---|---|---|
| fev-bench (10 tasks, 129 windows) | SQL Elo | 1535 (+220/−129) | 1538 (+219/−135) | 1538 (+219/−135) |
| | MASE Elo | 1460 | 1473 | 1464 |
| | task-mean SQL, geometric mean of the ratio to v1.1 | 1 | 1.007 | 0.998 |
| | the same ratio per task, smallest to largest | — | 0.971 to 1.045 | 0.983 to 1.019 |
| ADBench (12 tables × 3 seeds) | mean AUROC | 0.6633 | 0.6641 | 0.6642 |
| | mean AUROC with a clean context (uses labels) | 0.8203 | 0.8214 | 0.8210 |
| Imputation (10 tables × 3 seeds) | NRMSE at 10 / 30 / 50% | 0.546 / 0.646 / 0.745 | 0.544 / 0.647 / 0.746 | 0.544 / 0.647 / 0.744 |
| | accuracy at 10 / 30 / 50% | 0.652 / 0.632 / 0.590 | 0.651 / 0.631 / 0.588 | 0.651 / 0.632 / 0.589 |
| Treatment effects, S-learner | IHDP √PEHE out of sample | 0.553 | 0.557 | 0.554 |
| | ACIC 2016 √PEHE out of sample | 0.540 | 0.566 | 0.563 |
| D1, linear trend 15, default arm | trend capture / MASE | −1.31 / 11.75 | −1.50 / 12.76 | −1.34 / 11.90 |
| D1, linear trend 15, `plain` arm | trend capture / MASE | −0.71 / 9.03 | −0.91 / 9.95 | −0.77 / 9.21 |
| D2, y independent of x | excess NLL, normal target, 500 rows | 0.052 | 0.052 | 0.051 |
| D3, dense group at 2% | AUROC | 0.425 | 0.437 | 0.430 |

- The prediction holds: the candidate is v1.1 on all four suites and on D1 to D3.
- **Noise between two trainings:** the task-mean SQL of fev-bench moves by about 0.5% and a single task by up to 6%; the SQL Elo is the same to the digit because no ranking against another method changes. ADBench's mean AUROC moves by 0.0001. D1's linear-15 MASE moves by 0.9, and all four series seeds are worse in #14's run, so this is a difference between the weights and not between series.

### 1.2 Forecasting recipe

Arms on v1.1 (`arms.py`): (b) `no-target-power`, the default members without the target power transform; (c) `standard`, 8 members of the `standard` feature kind; (d) `plain8`, both; (e) the best of them with a straight line fitted to the context by least squares, its continuation added to the forecast of the residual (`+trend`). D1's earlier `plain` arm is (d) with 2 members.

**D1** (synthetic daily series, 365 context steps, horizon 56, 4 seeds; trend strength is the rise over the horizon in noise standard deviations):

| Arm | Linear 15: trend capture | Linear 15: MASE | Largest forecast minus largest context value | Exponential 15: MASE | Piecewise 15: MASE |
|---|---|---|---|---|---|
| default | −1.31 | 11.75 | 0.03 | 4.72 | 4.90 |
| (b) `no-target-power` | −1.22 | 11.39 | 5.28 | 4.77 | 4.73 |
| (c) `standard` | −0.97 | 10.06 | 0.04 | 4.23 | 4.46 |
| (d) `plain8` | −0.82 | 9.57 | 8.29 | 4.16 | 4.06 |
| `plain` (2 members) | −0.71 | 9.03 | 8.50 | 4.19 | 4.16 |
| default `+trend` | 0.99 | 0.37 | 13.57 | 4.64 | 4.95 |
| `plain8+trend` | 0.99 | 0.37 | 13.50 | 4.40 | 4.57 |
| seasonal naive | — | 3.67 | — | 4.36 | 4.59 |
| least squares on index and seasonal terms | — | 0.37 | — | 7.43 | 11.53 |

- The cap is where the derivation put it: gone in (b) and (d), still there in (c), where half of the members keep the transform (formulations, section 3).
- Without the trend fit the MASE stays between 9.0 and 11.8, as predicted: the model's forecast falls back toward the context whatever the recipe. With the line it is the least-squares fit's 0.37 (estimated: about 0.4).
- A line over the whole context does nothing for a trend that is not a line. On the exponential and the piecewise series every arm stays near seasonal naive.
- On the series without a trend the arms agree (MASE 0.51 to 0.52).

**fev-bench** (the suite: 10 tasks, every window, 8 members, seed 0). The SQL ratio is the geometric mean over tasks of the arm's task-mean SQL over the default's; below 1 is better.

| Arm (v1.1) | SQL Elo | MASE Elo | SQL ratio | Tasks better, of 10 | Largest changes |
|---|---|---|---|---|---|
| default (the suite's run) | 1535 (+220/−129) | 1460 | 1 | — | — |
| (b) `no-target-power` | 1486 (+192/−145) | 1389 | 1.022 | 3 | ecdc_ili 1.146, redset_1H 1.060, us_consumption_1M 0.948 |
| (c) `standard` | 1526 (+202/−132) | 1460 | 0.999 | 4 | us_consumption_1M 0.958, ecdc_ili 1.021 |
| (d) `plain8` | 1476 (+193/−139) | 1377 | 1.026 | 3 | ecdc_ili 1.186, redset_1H 1.067, us_consumption_1M 0.938 |
| (e) default `+trend` | 1453 (+133/−175) | 1342 | 1.035 | 3 | rohlik_orders_1D 0.694, us_consumption_1M 1.409, ecdc_ili 1.183, solar_1D 1.086, hospital 1.081 |

- **No arm is better than the default**, so (e) was run on the default recipe.
- The target power transform helps on fev-bench: without it ecdc_ili loses 15% to 19% and redset_1H 6% to 7%, and one task gains, us_consumption_1M by 5% to 6%. The feature kind alone changes nothing.
- The line helps one task, rohlik_orders_1D (SQL 1.47 to 1.02, where TabPFN-TS (TabPFN v2) has 1.18, TimesFM-2.5 1.01 and Chronos-2 0.96). It hurts us_consumption_1M (41%), ecdc_ili (18%), solar_1D (9%) and hospital (8%). Why these and not the others was not examined.
- The Elo has a 95% interval of about +200/−130 over tasks, so only the SQL ratios separate the arms.

### 1.3 Outlier score

Arms on v1.1 (`arms.py`): (b) `ctx<m>`, the log density of a row under 32 random subsets of m context rows, each with its own column order and 2 members, averaged; (c) `ref-uniform` and `ref-marginals`, the probability of the class "reference row" when as many reference rows as context rows are appended, uniform in the context's bounding box widened by 10%, or the context's columns shuffled independently; (d) `refine`, the density after the 10% of context rows with the highest score among themselves are dropped. The plan's sizes for (b) were 64 and 256. 16, 32 and 128 were added after 64 came out ahead of 256.

**ADBench** (the suite's 12 tables × 3 seeds, the contaminated training rows as the context). The difference is the mean over tables of the arm's AUROC minus the suite's score, with its standard error over tables.

| Score | Mean AUROC | Difference to the suite's score | Tables better, of 12 |
|---|---|---|---|
| the suite's score: density, whole context, 10 column orders, 8 members | 0.663 | — | — |
| (b) `ctx256` | 0.737 | +0.074 ± 0.046 | 9 |
| (b) `ctx64` | **0.756** | +0.092 ± 0.047 | 8 |
| (b) `ctx32` | 0.754 | +0.091 ± 0.052 | 9 |
| (b) `ctx16` | 0.742 | +0.079 ± 0.054 | 7 |
| (c) `ref-uniform` | 0.665 | +0.002 ± 0.031 | 6 |
| (c) `ref-marginals` | 0.523 | −0.141 ± 0.078 | 4 |
| IsolationForest | 0.722 | +0.059 ± 0.054 | 7 |
| TabPFN v2 extension | 0.704 | +0.041 ± 0.012 | 10 |
| the suite's score with a clean context (uses labels) | 0.820 | +0.157 ± 0.036 | 12 |

- Our IsolationForest equals the suite's reference on all 36 cells, so the splits are the suite's.
- **The smaller the context, the better, down to 32 to 64 rows.** `ctx64` is ahead of IsolationForest and closes 59% of the gap between the contaminated and the clean context.
- The gain is where contamination hurt most: cover 0.50 to 0.98, cardio 0.53 to 0.82, PageBlocks 0.68 to 0.93. Four tables are behind the suite's score with `ctx64`: landsat 0.57 to 0.50, magic.gamma 0.78 to 0.75, Waveform 0.70 to 0.68, and annthyroid by 0.001.
- The same evaluations read as the median over the subsets give 0.750 (64 rows), and as the log of the mean density 0.736: the mean of the log densities is the plan's arm and the best reading. With the mean of densities, one subset that holds the group is enough to make it dense.
- A uniform reference gives the density score's mean in about a twentieth of its time (133 s against 41 minutes on one H100 for the 36 cells), with other tables ahead: cardio 0.74 and cover 0.65, but shuttle 0.73 against 0.91. Shuffled columns as the reference do badly where the suite's score does well (annthyroid 0.39, shuttle 0.29, mammography 0.32).
- `ctx64` takes 33 minutes on one H100 for the 36 cells, against the suite's score's 41.

**D3** (600 context rows, 6 columns, 3 seeds; AUROC):

| Outliers | Share in the context | density | (d) `refine` | `ctx256` | `ctx64` | `ctx16` | (c) `ref-uniform` | `ref-marginals` |
|---|---|---|---|---|---|---|---|---|
| dense group | 2% | 0.425 | 0.427 | 0.836 | 0.987 | 0.994 | 0.853 | 0.393 |
| | 5% | 0.427 | 0.380 | 0.660 | 0.887 | 0.972 | 0.661 | 0.368 |
| | 10% | 0.440 | 0.382 | 0.695 | 0.743 | 0.897 | 0.436 | 0.250 |
| | 20% | 0.453 | 0.467 | 0.792 | 0.778 | 0.719 | 0.258 | 0.365 |
| curve | 2% | 0.679 | 0.706 | 0.984 | 1.000 | 0.998 | 0.976 | 0.550 |
| | 5% | 0.218 | 0.264 | 0.661 | 0.998 | 0.992 | 0.924 | 0.333 |
| | 10% | 0.064 | 0.105 | 0.229 | 0.871 | 0.993 | 0.847 | 0.267 |
| | 20% | 0.050 | 0.068 | 0.143 | 0.531 | 0.926 | 0.688 | 0.165 |
| scattered (global, local) | 0 to 20% | 0.975 to 1 | 0.994 to 1 | 0.998 to 1 | 0.998 to 1 | 0.989 to 1 | 0.998 to 1 | 0.984 to 1 |

- (b) holds: the group at 2% goes from 0.425 to 0.987 with 64 rows, and up to a contamination of 10% each smaller size is better on the group and on the curve. At 20% the sizes are within 0.07 of each other on the group. The three seeds of the group at 2% are 0.998, 0.965 and 1.000 with 64 rows, against 0.494, 0.255 and 0.527.
- (d) holds: dropping the highest-scoring context rows changes nothing, since the group's rows are not among them.
- **(c) does not hold.** Reference rows were predicted to fail on the group as the density does. They rank it above the inliers at 2% (0.853) and at 5% (0.661), and fail from 10% on.
- The score with the whole context read the way the small contexts are (8 column orders, 2 members, mean of the log densities) gives 0.613 on the group at 2%: part of the step from 0.425 is the mean of the logs over column orders, and most of it is the size of the context.

### 1.4 D4: imputation when cells are not missing completely at random

Five mechanisms mask a target column at 30%; each target column has one driver column (`d4_missing.py`). The plan's third mechanism is split in two, because the derivation's prediction for it holds only for one of the forms.

- `mcar`: completely at random.
- `by_other`: the target is missing where the driver is large.
- `other_obs`: the driver is missing where the target is large, in the rows that observe the target. The rows to impute all show the driver. This is missing at random in Rubin's sense and the counter-example of formulations, section 5.
- `other_all`: the same in every row, so that a hidden driver tells about the target in the rows to impute too. This is the mechanism of `mask_by_target` in the candidate's prior.
- `self`: the target is missing where it is large.

**Gaussian tables** (2,000 rows; a standard normal target, a driver with correlation 0.6, two noise columns; 3 seeds). The bias is the mean of imputed minus true over the missing target cells. "Complete cases" is a least-squares fit on the rows that observe the target, with the driver's missing indicator as a feature: the linear version of what the wrapper converges to.

| Mechanism | Bias: v1.1 | Bias: #14's run | Bias: N10 | Bias: complete cases | RMSE: v1.1 | RMSE: #14's run | RMSE: N10 | RMSE: complete cases |
|---|---|---|---|---|---|---|---|---|
| `mcar` | 0.001 | 0.002 | 0.003 | 0.007 | 0.808 | 0.808 | 0.808 | 0.806 |
| `by_other` | 0.025 | 0.024 | 0.004 | 0.031 | 0.802 | 0.801 | 0.801 | 0.791 |
| `other_obs` | −0.209 | −0.293 | −0.281 | −0.284 | 0.855 | 0.879 | 0.876 | 0.871 |
| `other_all`, rows that hide the driver | −0.707 | −0.027 | −0.045 | −0.010 | — | — | — | — |
| `other_all`, all rows | −0.139 | 0.006 | 0.010 | 0.012 | 0.807 | 0.702 | 0.703 | 0.702 |
| `self` | −0.798 | −0.796 | −0.803 | −0.783 | 1.065 | 1.064 | 1.068 | 1.054 |

- `by_other` is like `mcar`: the same conditional, as derived.
- `self`: the bias is the closed form's −0.83 for the complete cases' conditional mean, for all three sets of weights.
- **`other_all`: v1.1 does not use a hidden driver as information and the candidate does.** In the rows that hide the driver v1.1 is off by −0.71, where ignoring the mask entirely gives −0.96 and using it gives 0. Both seeds of the candidate are at the level of the fit that has the indicator (RMSE 0.70 against v1.1's 0.81).
- **`other_obs`: the candidate is more biased than v1.1**, at the complete cases' −0.28. It applies in the rows to impute the tilt that the context shows, and here the tilt does not hold for them. The mean under the full distribution, which ignores the masks, has no bias in this case (RMSE 0.807).

**The imputation suite's tables** (10 tables, at most 2,000 rows, 3 seeds; the numeric columns are paired by their rank correlation, one of a pair is the target and the other its driver; NRMSE on the missing target cells):

| Mechanism | v1.1 | #14's run | N10 | MissForest | Mean | v1.1 over MissForest |
|---|---|---|---|---|---|---|
| `mcar` | 0.490 | 0.490 | 0.490 | 0.583 | 0.999 | 0.84 |
| `by_other` | 0.640 | 0.639 | 0.639 | 0.706 | 1.250 | 0.91 |
| `other_obs` | 0.530 | 0.577 | 0.577 | 0.632 | 0.999 | 0.84 |
| `other_all` | 0.559 | **0.481** | **0.481** | 0.643 | 0.999 | 0.87 |
| `self` | 0.836 | 0.831 | 0.834 | 0.974 | 1.536 | 0.86 |

- TabSil stays ahead of MissForest under every mechanism, by 9% to 16%.
- `by_other` raises every method's error (v1.1 by 31%, MissForest by 21%, the mean by 25%): the rows to impute lie where the context is thin. `self` raises it by 71%, 67% and 54%.
- **On real tables the candidate's weights are 14% better than v1.1 under `other_all` and 9% worse under `other_obs`.** The two seeds agree to the third digit.

### 1.5 D5: treatment effects as overlap shrinks

Tables with a known effect (`d5_overlap.py`): 5 features, outcome and effect from random tanh networks, 1,000 observed rows and 1,000 test rows, 10 tables. The propensity is a steep function of two features clipped to [c, 1 − c]: c = 0.5 is a randomized experiment, and at 0.02 almost every row has a propensity of 0.02 or 0.98.

| Clip c | √PEHE, S-learner | √PEHE, T-learner | S over T |
|---|---|---|---|
| 0.5 | 0.241 | 0.282 | 0.86 |
| 0.3 | 0.250 | 0.301 | 0.84 |
| 0.2 | 0.260 | 0.316 | 0.83 |
| 0.1 | 0.282 | 0.344 | 0.83 |
| 0.05 | 0.293 | 0.355 | 0.84 |
| 0.02 | 0.304 | 0.371 | 0.83 |

- The error grows as overlap shrinks, as derived: by 26% for the S-learner and 32% for the T-learner from a randomized experiment to a clip of 0.02 (ratios per table 1.26 ± 0.05 and 1.32 ± 0.05).
- The S-learner is ahead at every clip and does not grow faster: the T-learner's growth minus the S-learner's is 0.05 ± 0.02.
- The candidate's two seeds give the same numbers to within 0.003.
- The standard errors over the 10 tables are 0.011 to 0.021.

### 1.6 Rare class

Tables whose labels do not depend on the features (`d2_rare_class.py`): the rarest class has 1 to 10 of n context rows, the other classes share the rest, 5 seeds, 1 and 5 noise features. The table gives the predicted probability of the rarest class over its share of the context, n_c / n. The posterior under a uniform prior on the class probabilities, (n_c + 1) / (n + K), is about 2 times the share for one row and about 1.33 times for three, a little less when n is small against K.

| Classes | Context rows | 1 row | 2 rows | 3 rows |
|---|---|---|---|---|
| 2 | 50 | 1.73 | 1.40 | 1.38 |
| 2 | 200 | 1.02 | 1.38 | 1.32 |
| 2 | 1,000 | 1.06 | 1.15 | 1.28 |
| 3 | 50 | 1.33 | 1.25 | 1.19 |
| 3 | 200 | 0.60 | 0.78 | 0.94 |
| 3 | 1,000 | 0.55 | 0.62 | 0.73 |
| 10 | 50 | 2.15 | 1.76 | 1.45 |
| 10 | 200 | 2.05 | 1.83 | 1.69 |
| 10 | 1,000 | 1.51 | 1.42 | 1.31 |

- With 2 classes and 200 rows or more, a class with one context row gets its share of the context (1.02 and 1.06), half of the posterior, as derived.
- With 3 classes and 200 rows or more it gets less than its share (0.55 to 0.60). With 10 classes it gets 1.5 to 2.2 times its share: above the posterior at 50 and 200 rows, between the share and the posterior at 1,000. **The derivation holds for two classes only.**

### 1.7 D1 on phase 1B's weights

Two control seeds (C) and the two seeds trained with the context-row target scale (YCTX), all stage-1 runs of 30,000 steps ([v2-experiment-results.md](v2-experiment-results.md), Phase 1B). Predicted: no difference in trend capture.

| D1, linear trend 15 | C seed 0 | C seed 1 | YCTX seed 0 | YCTX seed 1 |
|---|---|---|---|---|
| trend capture, `plain` arm | −0.21 | 0.26 | 0.41 | 0.55 |
| MASE, `plain` arm | 6.38 | 3.79 | 3.49 | 3.13 |
| trend capture, default arm | −0.75 | −0.06 | −0.13 | −0.01 |
| MASE, default arm | 8.93 | 5.41 | 5.78 | 5.15 |
| mean trend capture over the 9 trend cells, `plain` arm | −0.14 | −0.05 | 0.08 | 0.03 |

- The plan's rule asks whether the scaled seeds are above the controls by more than the spread between seeds. In the `plain` arm their mean is 0.45 above the controls' and the two control seeds are 0.48 apart: **not resolved**, and by the rule the design of the forecast tables stays.
- The direction is against the prediction: both scaled seeds are above both control seeds in the `plain` arm, on the linear trend and on the mean over all trend cells. The sorted-split tables of tier 2 carry the context-row scale, so the probe cannot tell the scale from the sorting. If the probe's trend capture rises, a run with the scale alone is the control it lacks.
- These stage-1 weights of 30,000 steps capture a trend better than v1.1 (−0.71 in the `plain` arm), and two seeds of one recipe differ by 0.5: D1's trend capture is a noisy reading of a training.

## Tier 2

The probe is a short fine-tuning of the v2 candidate (#14's run) on task-shaped tables alone. Its weights are not for release: with every table task-shaped it is the best case for a prior that mixes such tables in at a small share.

### The tables

`graph_scm_task_probe` (`priors/transforms/tasks.py`) is `graph_scm_hicard_missing`, the prior of the weights it starts from, with two options at 1.

**Sorted split** (every regression table). The rows are ordered by one column before the cut, ascending or descending, so the eval rows lie beyond the train rows in that column, and the target is standardized with the train rows as inference does (clipped at 20). Two things were added to the plan's smallest form after a sample of the tables showed what it gives:

- With a random ordering column the eval targets rarely leave the train targets' range: in 39 sampled tables the median share of eval targets beyond it was 0, and the eval targets' mean lay 0.23 train standard deviations from the train mean. Such a table asks for no continuation. So half of the tables are ordered by the column with the largest rank correlation to the target (median over tables 0.50); there the eval mean lies 0.65 standard deviations off, 1.6 in a quarter of the tables.
- In half of the tables the ordering column is replaced by the row index, which is what a forecaster's running index is.
- A third of the prior's regression tables have no column with enough distinct values to order by and stay as they are.

**Outlier table** (every classification table). The train rows are data rows labeled 0 and reference rows labeled 1 (20% to 50% of them; uniform in the data rows' padded bounding box, or the data rows' columns drawn independently, half each). A share of the data rows are hidden anomalies: none with probability 0.1, else U(0, 0.4). The eval rows are held-out data rows with their true label, half of them anomalies. The anomalies of a table are of one kind: the rows of whole classes of the SCM table, rows with a few columns moved, a tight group off the bulk, uniform rows, or rows whose columns are taken from different rows.

- The contaminated context and the balanced queries follow TACTIC (arXiv 2603.14171, Section 3: "the number of anomalies is sampled uniformly from [0.05,0.3] times the number of nominal points", or none; "equal numbers of nominal and anomalous samples") and ICLAD (arXiv 2603.19497, Section 4.1: "The anomaly ratio is sampled uniformly from [0,0.4]"). Whole classes as anomalies are TACTIC's first prior and ICLAD's structural anomalies. Both were read for this in the papers' text. Neither has reference rows: both end in an output of their own.
- Drawing a table takes 0.09 s against 0.07 s for the base prior (median of 24 tables of each task at the pilot's sizes).

### 2.1 Pilot

One H100, 2,000 steps at batch 32 with stage 2's settings (lr 1e-4, 1% warmup, micro batch 1), tables of 400 to 2,048 rows, from #14's weights and borders. 2.26 H100-h (estimate 3).

- **The table code does not stall:** waiting for tables took 0.2% to 2.0% of a 50-step window, and a step took 3.8 s.
- The loss on the outlier tables fell from 1.50 in the first 50 steps to 0.21. The loss on the sorted tables stayed at 0.10 to 0.13 from the first window to the last.

**D1**, #14's run against the pilot's weights (mean of 4 series seeds; MASE):

| Arm | Weights | Linear 15: trend capture | Linear 15 | Exponential 15 | Piecewise 15 | Linear 5 | No trend | Mean of the 9 trend cells |
|---|---|---|---|---|---|---|---|---|
| default | #14's run | −1.50 | 12.76 | 5.00 | 4.93 | 3.75 | 0.511 | 3.84 |
| default | pilot | 0.42 | 2.97 | 3.55 | 3.77 | 0.90 | 0.513 | 1.77 |
| `no-target-power` | #14's run | −1.44 | 12.46 | 4.97 | 4.68 | 3.71 | 0.510 | 3.77 |
| `no-target-power` | pilot | 0.93 | **0.66** | **2.21** | **1.53** | 0.71 | 0.513 | **1.07** |
| `plain8` | #14's run | −0.97 | 10.25 | 4.31 | 3.96 | 2.99 | 0.514 | 3.19 |
| `plain8` | pilot | 1.05 | 0.71 | 2.00 | 2.58 | 0.95 | 0.516 | 1.16 |
| seasonal naive | — | — | 3.67 | 4.36 | 4.59 | 1.62 | 0.858 | 2.25 |
| least squares on index and seasonal terms | — | — | 0.37 | 7.43 | 11.53 | 0.49 | 0.513 | 3.34 |

- **The model now continues a trend.** The trend capture on the linear trend goes from −1.4 to 0.9 without the cap, on every one of the four series seeds (−1.68 to −1.24 before, 0.82 to 1.05 after; the seeds' spread is 0.4). The MASE falls from 12.5 to 0.66. At strength 15 it is below seasonal naive on all three trend kinds, and on the exponential and the piecewise trend below the least-squares fit too, which has only a line for the trend.
- With the default recipe the cap still holds the forecast at the context's range (largest forecast minus largest context value 0.03), and the gain is smaller: 12.76 to 2.97.
- The series without a trend is unchanged (0.51).
- With `standard` members only the forecast overshoots a change of slope: on the piecewise trend the largest forecast is 45 above the context's largest value where the truth is 14, and the MASE is 2.58 against 1.53 with the default members.

**D3**, reference rows as the score (AUROC, mean of 3 seeds):

| Outliers | Share in the context | `ref-uniform`: #14's run | `ref-uniform`: pilot | `ref-marginals`: #14's run | `ref-marginals`: pilot | density: pilot |
|---|---|---|---|---|---|---|
| dense group | 2% | 0.875 | 1.000 | 0.404 | 0.997 | 0.460 |
| | 5% | 0.698 | 1.000 | 0.381 | 0.998 | 0.423 |
| | 10% | 0.479 | 1.000 | 0.264 | 0.998 | 0.432 |
| | 20% | 0.242 | 0.999 | 0.364 | 0.999 | 0.450 |
| curve | 2% | 0.985 | 1.000 | 0.571 | 0.978 | 0.791 |
| | 5% | 0.954 | 0.999 | 0.372 | 0.985 | 0.351 |
| | 10% | 0.881 | 0.996 | 0.314 | 0.963 | 0.116 |
| | 20% | 0.742 | 0.876 | 0.158 | 0.796 | 0.085 |

- **A dense group in the context no longer hides.** With reference rows the group is ranked above the inliers at every contamination up to 20%, where the starting weights fall from 0.88 to 0.24. The three seeds at 2% are 1.000 each (0.865, 0.785 and 0.974 before).
- The density score of the same weights still fails on the group (0.42 to 0.46) and is a little higher on the curve: the fine-tuning does not ask for it.

**By the plan's rule both tasks are in the probe:** each diagnostic moved by several times its seeds' spread.

The pilot's weights on the two benchmarks, which the plan did not ask for at this point (1.1 H100-h, as an early reading for the probe):

| Benchmark | Wrapper | #14's run | Pilot |
|---|---|---|---|
| ADBench, mean AUROC | `ref-uniform` | 0.664 | **0.752** (+0.088 ± 0.036 over tables, 10 of 12 better) |
| | `ref-marginals` | 0.526 | 0.687 |
| fev-bench, SQL Elo | default recipe | 1538 | **1582** (+193/−100) |
| | `plain8` | — | 1562 (+187/−113) |
| fev-bench, task-mean SQL against #14's run with the default recipe | default recipe | 1 | **0.950** |
| | `plain8` | — | 0.965 |
| fev-bench, MASE Elo | default recipe | 1473 | 1557 |

- On fev-bench the default recipe is again the better one: the tasks that gain are the same under both (rohlik_orders_1D 0.72 and 0.67, us_consumption_1M 0.87 and 0.86, ercot_1D 0.95 and 0.94), and without the target transform ecdc_ili loses 20% as it did before the fine-tuning.
- On ADBench two tables lose with the trained reference rows: annthyroid 0.87 to 0.78 and mammography 0.82 to 0.68.

### 2.2 Probe

One node of 8 H100, 2,000 steps at batch 256, otherwise as the pilot (seed 12): 256,000 tables per task, eight times the pilot's. 21.4 H100-h (estimate 16; the plan's check was 21).

- A step took 4.43 s, against the pilot's 3.8 s for the same 32 tables per GPU: every step waits for the slowest of the 8 GPUs.
- The loss on the outlier tables ended at 0.20 (pilot 0.21), and the loss on the sorted tables at 0.11, as in the pilot.

### 2.3 The probe's weights

**D1** (MASE; mean of 4 series seeds):

| Arm | Weights | Linear 15: trend capture | Linear 15 | Exponential 15 | Piecewise 15 | Linear 5 | No trend | Mean of the 9 trend cells | Piecewise 15: largest forecast minus largest context value (truth 14.1) |
|---|---|---|---|---|---|---|---|---|---|
| default | #14's run | −1.50 | 12.76 | 5.00 | 4.93 | 3.75 | 0.511 | 3.84 | 0.1 |
| default | pilot | 0.42 | 2.97 | 3.55 | 3.77 | 0.90 | 0.513 | 1.77 | 0.1 |
| default | probe | 0.42 | 2.96 | 3.48 | 3.72 | 0.92 | 0.513 | 1.74 | 0.1 |
| `no-target-power` | #14's run | −1.44 | 12.46 | 4.97 | 4.68 | 3.71 | 0.510 | 3.77 | 4.4 |
| `no-target-power` | pilot | 0.93 | 0.66 | 2.21 | 1.53 | 0.71 | 0.513 | 1.07 | 12.8 |
| `no-target-power` | probe | 1.10 | 0.79 | 1.86 | 1.82 | 1.22 | 0.512 | 1.11 | 32.8 |
| `plain8` | #14's run | −0.97 | 10.25 | 4.31 | 3.96 | 2.99 | 0.514 | 3.19 | 5.8 |
| `plain8` | pilot | 1.05 | 0.71 | 2.00 | 2.58 | 0.95 | 0.516 | 1.16 | 45.0 |
| `plain8` | probe | 1.26 | 1.43 | 4.00 | 5.87 | 1.56 | 0.517 | 1.88 | 62.2 |

- **Eight times the tables do not improve D1, and they make the forecast overshoot.** With the default members and no target transform the probe is where the pilot is (mean of the trend cells 1.11 against 1.07). Its trend capture is above 1, and with `standard` members only it is 1.26 on the linear trend and 1.8 on the piecewise one, where the MASE (5.87) is back above seasonal naive (4.59).
- With the default recipe the cap hides the difference: pilot and probe agree to 0.07.

**D3:** as the pilot. Reference rows rank the dense group above the inliers at every contamination (uniform 1.000, 1.000, 1.000, 0.998 at 2%, 5%, 10% and 20%; shuffled columns 0.997 to 0.999), and the curve at 1.000, 0.998, 0.995 and 0.857.

**fev-bench** (the ratio is the geometric mean over tasks of the task-mean SQL over that of #14's run with the default recipe, SQL Elo 1538):

| Weights | Arm | SQL Elo | MASE Elo | SQL ratio | Tasks better, of 10 | Changes above 3% |
|---|---|---|---|---|---|---|
| pilot | default | **1582** (+193/−100) | 1557 | **0.950** | 5 | rohlik_orders_1D 0.72, us_consumption_1M 0.87, ercot_1D 0.94, redset_1H 0.96 |
| pilot | `plain8` | 1562 (+187/−113) | 1496 | 0.965 | 4 | rohlik_orders_1D 0.67, us_consumption_1M 0.86, ercot_1D 0.94, epf_be 1.05, ecdc_ili 1.20 |
| probe | default | 1519 (+198/−106) | 1532 | 0.962 | 6 | rohlik_orders_1D 0.75, us_consumption_1M 0.93, ercot_1D 0.94, redset_1H 0.96, solar_with_weather_1H 0.97, ecdc_ili 1.09 |
| probe | `no-target-power` | 1460 (+163/−100) | 1519 | 0.997 | 4 | rohlik_orders_1D 0.77, us_consumption_1M 0.94, ercot_1D 0.94, epf_be 1.09, ecdc_ili 1.26 |
| probe | `plain8` | 1507 (+151/−82) | 1525 | 0.979 | 5 | rohlik_orders_1D 0.71, us_consumption_1M 0.85, ercot_1D 0.93, hospital 1.03, epf_be 1.06, ecdc_ili 1.34 |

- **The fine-tuning is worth 4% to 5% of fev-bench's SQL with the default recipe**, eight to ten times the noise between two trainings (0.5%, item 1.1). The tasks that gain are the same in every run: rohlik_orders_1D by a quarter, us_consumption_1M, ercot_1D and redset_1H.
- **The probe is not ahead of the pilot.** Its SQL ratio is 0.962 against 0.950, and ecdc_ili, which the pilot left where it was, is 9% worse.
- The Elo follows the count of methods passed per task, not the size of the gains: against the seven other methods on ten tasks TabSil has the lower SQL in 41 of 70 comparisons with #14's weights, 44 with the pilot's and 39 with the probe's.
- The default recipe stays the best arm for the fine-tuned weights. Without the target transform ecdc_ili loses 26% to 34%.
- **The bar of 1707 is not reached:** the best is 1582.

**ADBench** (mean AUROC over the 36 cells; the difference to #14's run is the mean over tables with its standard error):

| Score | #14's run | Pilot | Probe | Probe minus #14's run | Tables better, of 12 |
|---|---|---|---|---|---|
| `ref-uniform`, the form the probe trains | 0.664 | 0.752 | **0.759** | +0.096 ± 0.036 | 11 |
| `ref-marginals` | 0.526 | 0.687 | 0.733 | +0.207 ± 0.053 | 12 |
| `ctx64`, the best score of tier 1 | 0.756 | — | 0.757 | +0.001 ± 0.005 | 6 |
| mean rank of `ctx64` and `ref-uniform` | 0.733 | — | **0.774** | +0.042 ± 0.014 | 10 |

- **The trained reference rows reach 0.759**, from 0.664, and are 0.011 below the bar. Eight times the pilot's tables add 0.007 to the uniform reference and 0.046 to the shuffled-column one.
- The density under small contexts does not change with the fine-tuning, as expected: 0.757.
- The two scores are strong on different tables (reference rows on shuttle 0.99, Wilt 0.82, fault 0.67 and landsat 0.60; the density on cover 0.97 and annthyroid 0.87), and the mean of their ranks gives **0.774**: above the bar by 0.004, with 0.781, 0.773 and 0.769 on the three splits. It is ahead of the density alone by 0.017 ± 0.011 over tables and of the reference rows alone by 0.015 ± 0.015.
- **This arm was added after the pilot's results had shown the two scores to differ by table**, so it was chosen on the tables it is scored on. With the starting weights the same mean of ranks is below the density alone (0.733 against 0.756): the gain comes with the trained reference rows.

## TabArena-Lite of the fine-tuned weights

Measured on 2026-11-28 after the user's correction of the goal: the two fine-tunings of tier 2 on the benchmark the task tables were meant to serve. TabArena-Lite (51 datasets, one split each) with the settings of the candidate's reference run (`--member-kinds numeric-cycle --categories counts`, seed 0), one H100 each, 0.31 H100-h per run. G is the paired error ratio against #14's run (above 1 is worse), from `training/scripts/paired_report.py`.

| Weights | Elo | Classification Elo | Regression Elo | G, all 51 (95% CI) | G, classification (38) | G, regression (13) | Better / worse |
|---|---|---|---|---|---|---|---|
| #14's run, the start (run again with this code) | 1807.6 | 1793.0 | 2113.8 | 1 | 1 | 1 | — |
| pilot: task tables only, 2,000 steps at batch 32 | **1252.4** | 1147.0 | 1823.2 | 1.220 (1.149 to 1.299) | 1.295 (1.209 to 1.392) | 1.024 (1.012 to 1.041) | 6 / 45 |
| probe: task tables only, 2,000 steps at batch 256 | **1234.1** | 1118.8 | 1844.5 | 1.228 (1.157 to 1.306) | 1.309 (1.220 to 1.409) | 1.020 (1.007 to 1.035) | 5 / 46 |

- The start reproduces its record to the digit (1807.6), so the three runs are comparable.
- **Training on task-shaped tables alone breaks TabArena-Lite**: 555 and 574 Elo lower, errors 22% to 23% higher, 45 and 46 of 51 datasets worse. Classification carries it (errors 30% higher, every multiclass dataset worse): its output was retrained to tell data rows from reference rows. Regression, whose task tables are still regression tables, loses 2%.
- **This does not answer what mixing does.** These weights saw no ordinary table for 2,000 steps. [v2-task-formulations.md](v2-task-formulations.md), section 10, derives the mixture: in the limit of capacity and data it cannot lower the loss on ordinary synthetic tables, it keeps the ordinary answer as far as the task tables are recognizable (the two built here are), and on real tables it can gain only where a table resembles the task tables or where a task gives more targets per table.
- **The test that is missing:** the same fine-tuning on a mix of ordinary and task-shaped tables against a control of the same steps on ordinary tables alone, both read here. `graph_scm_hicard_task_mix` (a quarter of the tables of each task task-shaped) is registered for it and no run has used it. Expected effect, from the derivation: near zero, sign unknown, against a resolution of about 20 Elo for one pair of runs on TabArena-Lite.

## Derivations against the measurements

| Prediction (formulations) | Measured | Verdict |
|---|---|---|
| The candidate's weights are v1.1 on forecasting and outlier detection (plan, 1.1) | SQL ratio 1.007 and 0.998; AUROC 0.664 against 0.663 | holds |
| The cap on the median is gone without the target transform, and stays while half of the members have it (section 3) | largest forecast minus largest context value 5.3 and 8.3 against 0.03 and 0.04 | holds |
| D1's linear-15 MASE between 9.03 and 11.75 without the trend fit, about 0.4 with it (section 3) | 9.6 to 11.4; 0.37 | holds |
| Small random contexts raise D3's group at 2% above 0.425 (section 4, a hypothesis) | 0.987 with 64 rows; ADBench +0.092 | holds |
| Dropping the highest-scoring context rows does not change D3's group (section 4) | 0.427 | holds |
| Reference rows fail on D3's group as the density does (section 4) | 0.853 at 2%, 0.661 at 5%, 0.436 at 10% | **fails** below 10% |
| A mask of the column by another column leaves the error at the MCAR error with a ratio like the baselines' (section 5) | bias 0.03; error up 31% against the baselines' 21% and 25% | holds |
| Under a mask of another column by the cell to impute, v1.1 is biased and the candidate less (section 5) | every-row form: −0.71 against −0.03 and −0.05. Form of the counter-example: −0.21 against −0.29 and −0.28 | holds for one form, **reversed** for the other |
| Under self-masking the bias is the closed form's (section 5) | −0.80 against −0.83 | holds |
| The effect error grows as overlap shrinks (section 6) | +26% (S), +32% (T) | holds |
| A class with 1 to 3 context rows gets its share of the context, below the posterior (section 2) | one context row among 200 or more: 1.02 to 1.06 of the share with 2 classes, 0.55 to 0.60 with 3, 1.5 to 2.1 with 10 | holds for 2 classes, **fails** for 10 |
| The context-row target scale alone does not change the trend capture (plan, 1.7) | scaled seeds 0.41 and 0.55, controls −0.21 and 0.26 | not resolved; the direction is against it |
| Forecasting needs a prior whose queries lie beyond the context, with the context-row scale; no change of loss or structure (section 3) | after 2,000 steps on sorted tables the trend capture on D1 is 0.93 and the MASE 0.66 (12.5 before) | holds |
| Reference rows with hidden outliers in the training tables make the output the outlier probability; the decoder can express it (section 4, "in principle, not demonstrated") | D3's group 1.00 at every contamination to 20%; ADBench 0.664 to 0.759 | holds on D3; on ADBench the gain is the overlap of the prior's anomaly kinds with the tables', as the section says |

What the three failures change:

- **Reference rows.** The derivation took the classifier for the exact ratio of the contaminated density to the reference density. With 600 data rows and 600 reference rows it resolves less than the chain-rule density does: a group of 2% of the rows is not yet a region of class 0. So the reference-row form is not as blind to a small dense group as stated, and the tables of the probe train a form whose starting point on D3 is 0.85 and not 0.43.
- **Rare class.** The covering split's factor was derived for labels without features under a uniform prior on the class probabilities, and it matches the output for two classes. With more classes the output departs from it in both directions, so the prior's own distribution of class sizes, which the derivation left out, matters as much. A correction of the output by that factor is not supported beyond two classes.
- **Target scale.** That the scale "puts no query beyond the context" is right and does not settle the question: under the all-row scale the context's mean tells where the queries' mean lies, which pulls a forecast toward the context. This is the scale term of section 2, estimated there at 0.07% to 0.5% of the loss on ordinary tables. On a table with a trend it may be larger.

## Cost

| Item | Jobs | H100-h | Estimate |
|---|---|---|---|
| 1.1 the candidate's two seeds on four suites and D1 to D3 | 10 | 5.03 | 5.6 |
| 1.2 forecasting recipe: D1 with 9 arms, fev-bench with 4 arms, one check run | 6 | 1.96 | 3 |
| 1.3 outlier score: ADBench with 4 context sizes and 2 references, D3 with every arm | 8 | 3.97 | 6.5 |
| 1.4 D4, three sets of weights, MissForest and the mean | 3 | 0.37 | 3 |
| 1.5 D5, three sets of weights | 3 | 0.10 | 1.5 |
| 1.6 rare class | 1 | 0.03 | 0.1 |
| 1.7 D1 on four sets of phase 1B's weights | 4 | 0.11 | 0.4 |
| **Tier 1** | **35** | **11.57** | **about 20** |

| 2.1 pilot | 1 | 2.26 | 3 |
| 2.1 the pilot's weights on D1, D3, ADBench with reference rows and fev-bench with two arms | 5 | 1.07 | — (D1 and D3 were planned; the benchmarks were added) |
| 2.2 probe (8 H100) | 1 | 21.40 | 16; the plan's check was 21 |
| 2.3 the probe's weights on D1, D3, fev-bench with three arms, ADBench with three scores | 7 | 2.12 | 2.5 |
| controls: #14's run on the arms that tier 1 had only for v1.1 | 4 | 0.71 | — |
| **Tier 2** | **18** | **27.55** | **about 22** |
| TabArena-Lite of the start, the pilot and the probe (after the correction of the goal) | 3 | 0.92 | — |
| **Total** | **56** | **40.04** | **about 42** |

Counted from the job records (`duration_seconds` of every job, times its GPUs). 15 more jobs failed within a second of being submitted, because the workspace allows 8 `GPU_1xH100` nodes at a time; they used no GPU time and were submitted again one by one.

The probe cost 34% more than its estimate. Its first 200 steps projected 21.3 H100-h against the plan's check of 21, and the run was left to finish: the pilot had by then raised what the probe could be expected to show, and a job in progress can only be cancelled by the user. The user was told when the projection was made.

## Differences from the plan

- **Where the code is.** The plan put the inference changes on `main`. The arms are wrappers in `training/scripts/v2_prior_diag/arms.py` and nothing in `tabsil` changes: an arm is moved into the package only if the user releases it.
- **1.2:** (e) was started on the default recipe together with (b) to (d), since it turned out the best of them.
- **1.3:** context sizes of 16, 32 and 128 rows were added after the first results. The best size, 64, is one of the plan's two. Each small-context run also records the median and the mean-density readings of the same evaluations, and D3 has the whole context read the same way as a control.
- **1.4:** Gaussian tables with known values replace the prior tables, since on a prior table the value a fit converges to is not known. The third mechanism is measured in two forms.
- **1.5:** the tables come from a random-network generator in the script, where the effect is known, not from the training prior. All three sets of weights were run.
- **2.x, the sorted split:** half of the tables are ordered by the column the target follows most, and in half the ordering column is the row index. Both were added to the plan's smallest form before the pilot, after a sample of the tables showed that a random ordering column rarely puts an eval target beyond the train targets.
- **2.1:** the pilot's weights were also run on fev-bench and on ADBench with reference rows (0.96 H100-h), as an early reading for the probe.
- **2.3:** the mean rank of two outlier scores was added as an arm after the pilot's results; "Reading point R2" says what that means for its number. The arms of tier 1 were run on #14's weights as the fine-tuning's control, since tier 1 had them for v1.1 only.

## Runs

MLflow experiment `tabsil`; every job ran on one H100. The v1.1 references are the suites' runs of 2026-11-26: fev-bench `4782073097a84405961e17cfc5a0d2a4`, ADBench `17363fac14c8419eaee5294048e393b6`, imputation `48624ad4c33b4b90a4aa39777a5144d1`, treatment effects `934715cb29d0478ba0e77981702a5fba`.

| Item | Run name | MLflow run |
|---|---|---|
| 1.1 | `tabsil-pt1-prior14-fev`, `tabsil-pt1-n10-fev` | `c4fc4e51e6ef47a0a3e0826577dae489`, `05c7dd33481a4acf9b4b00141775d1c1` |
| 1.1 | `tabsil-pt1-prior14-adbench`, `tabsil-pt1-n10-adbench` | `96af09f2ecb8442fb7ba78dcc98db87d`, `9be0ac4415994abf85c9ff12280c71bc` |
| 1.1 | `tabsil-pt1-prior14-impute`, `tabsil-pt1-n10-impute` | `2697e7bccc084519b2816ef1c3bdc24f`, `a2674b7c019d42f58ef1a0dfd70c94d8` |
| 1.1 | `tabsil-pt1-prior14-causal`, `tabsil-pt1-n10-causal` | `5b7f17be8045486a81c639e6e5d262c7`, `697c6d157302433d80c16513b701a562` |
| 1.1 | `tabsil-pt1-prior14-diag`, `tabsil-pt1-n10-diag` | `766ec9d918464a67bcbd885a1744314a`, `d2b515386ba1472b95524fd1f94c4cb7` |
| 1.2 | `tabsil-pt1-v11-d1-arms` | `6021729817874fccbb33499f1864c116` |
| 1.2 | `tabsil-pt1-v11-fev-no-target-power`, `-standard`, `-plain8`, `-default-trend` | `eacd18d04b27446d9ce22305d733d54a`, `0282b7d3088d4fcc8700395dfa4af312`, `52af07e400664012ba28772b2e9cafcf`, `fce96e4637a94bf994051baea7b69ae9` |
| 1.2 | `tabsil-pt1-check-fev-arm` (one task, two windows) | `50be60c2680e4c4abd8982068ed91c89` |
| 1.3 | `tabsil-pt1-v11-adbench-ctx16`, `-ctx32`, `-ctx64`, `-ctx256` | `de5ce9fe67e046bbb169e0f4152308b5`, `9aee35ed1c034fac890cb0bfb138155c`, `2495095edf854d03a7edfbfb75e51c99`, `fd4b915b02bf4a62924c92c200c2086f` |
| 1.3 | `tabsil-pt1-v11-adbench-ref` | `3b702542030743b79c3aed8bc66a8904` |
| 1.3 | `tabsil-pt1-v11-d3-arms-a`, `-b`, `-c` | `e23558fa8ea34db5a42fd1a1797dcfff`, `bdf4e43b35b54884bfd19674ed26c967`, `f5d708315fb24b80aeba2ce7590645ef` |
| 1.4 | `tabsil-pt1-v11-d4`, `tabsil-pt1-prior14-d4`, `tabsil-pt1-n10-d4` | `bb70f46c4ef648379a8323573445157e`, `08acf60afcf34d5c8860e0374cddc65e`, `2dbbc57e140447c5bc30a35ef36523f9` |
| 1.5 | `tabsil-pt1-v11-d5`, `tabsil-pt1-prior14-d5`, `tabsil-pt1-n10-d5` | `4a4b09b36da04a709264ce46dfafbd01`, `626ba53bf0fb4f2c9ca0b533d9c390f4`, `95172672ccd449bdb635af19a088e6c0` |
| 1.6 | `tabsil-pt1-v11-d2-rare` | `5282cbcfc3814c14bd8205277506c4e2` |
| 1.7 | `tabsil-pt1-1b-c-s0-d1`, `-c-s1-d1`, `-yctx-s0-d1`, `-yctx-s1-d1` | `7849691f36364b409eb60172fb0681f5`, `f3de079cb0664b7a8d0c86f6480fcc8f`, `19a14157344045a5baf35fd1980bc7b3`, `304df0fb49d841d6b29d1141c78326f0` |

| 2.1 | `tabsil-pt2-pilot` (training) | `485af41601ff457088ddd86fb4156574` |
| 2.1 | `tabsil-pt2-pilot-d1`, `-d3`, `-adbench-ref`, `-fev-default`, `-fev-plain8` | `2e47e2b9a92b4d9ca57bfc168e144f53`, `347ebf763fdd4f179511efd71ad7e435`, `363fdffcbde848d2b0b66a324670f996`, `35c34c3b793946359fc1d80b8ec8cac2`, `29566bb8a11044198d9f6534ecb5cc48` |
| 2.2 | `tabsil-pt2-probe` (training) | `615704d802054102a3e2351d79cfad69` |
| 2.3 | `tabsil-pt2-probe-d1`, `-d3`, `-adbench-ref`, `-adbench-ctx64` | `f820ae141839441f808704cf80fe6fa3`, `84a42ca73d314aa3bed6fb437c35ec01`, `f440f4f06c464f9da5cfa35369736103`, `a050b61d4b9b43d8a4a5ae963c5f3290` |
| 2.3 | `tabsil-pt2-probe-fev-default`, `-fev-no-target-power`, `-fev-plain8` | `152a3f8b735f4204827a0fb5a133c913`, `743fc16736044a3994f066336cd73ded`, `578d02694c874b26b451e1492ac7f876` |
| TabArena-Lite | `tabsil-pt2-prior14-lite`, `tabsil-pt2-pilot-lite`, `tabsil-pt2-probe-lite` | `572c86865a2d45049af4cd9e861a57d8`, `e4176a0d9e8e41f99d888d2f90428ed0`, `68bc32b020ee415dacb4b61ba96afc6b` |
| controls | `tabsil-pt2-prior14-d1-arms`, `-d3-ref`, `-adbench-ref`, `-adbench-ctx64-ref` | `7bbb939b78a8452eae867ea90bdae7a9`, `26f1e9b60f4648eeb169663052ddf75d`, `20076a89b35043d89580ba00e94ac098`, `4a2e14fe5ad84df0a2734e93b9054c7d` |

The code of the tier-1 runs is the tier-1 branch at `df0da5b`, and at `dd718b9` for the 16- and 32-row contexts on ADBench and the 16-, 32- and 128-row contexts on D3. The runs of 1.1 on the four suites used the evaluation code of `0fd0cf6`. The pilot, the probe and their evaluations ran the tier-2 branch at `819d66c`; the two ADBench runs with the mean rank ran `b85bf10`.

## For the user to decide

Nothing here is adopted or released, and no run is waiting. The first item is the one that bears on the user's goal.

1. **Whether to run the test of the user's question: task-shaped tables mixed into ordinary training, read on TabArena-Lite.**
   - The run: three fine-tunings of the candidate with the same steps and seed, on ordinary tables alone (the control), on `graph_scm_hicard_task_mix` (a quarter task-shaped) and on task tables alone, each followed by TabArena-Lite. About 14 H100-h in all: three runs of 3 to 5 H100-h on one 8xH100 node each, less than an hour each, and their evaluations. All three would use the pilot's tables of 400 to 2,048 rows; the control shows what a fine-tuning on such short tables does by itself.
   - Expected effect on TabArena-Lite: near zero, sign unknown (formulations, section 10). One pair of runs resolves about 20 Elo; a difference below that needs TabArena with 3 splits and the development suite, 2.5 H100-h per arm more.
   - Not running it leaves the question answered by the derivation alone.
2. **Whether to look for task shapes that the derivation leaves room for.** Those are tasks that give several targets per table (several columns, masked cells), which need an output used in training only. That is a change of structure, undecided since the masked-cell proposal, and without a controlled result in the literature.
3. **The small-context outlier score.** An inference change with the v1.1 weights: ADBench 0.663 to 0.756, D3's group 0.43 to 0.99, in less time than today's score; behind today's score on 4 of the 12 tables, by 0.001 to 0.07. It does not touch TabArena-Lite. Releasing it is the user's decision.
4. **What follows D4 and D5,** as measurements of the candidate's weights: they read a missing cell as information (14% better where the mask is informative, 9% worse where it is not), and the S-learner's error grows by a quarter as overlap shrinks.
