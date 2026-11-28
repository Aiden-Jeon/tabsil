# v2 plan for priors and tasks: results

**Status:** tier 1 (no training) is measured and reading point R1 is read, on 2026-11-28. Both tasks that have a bar go on to tier 2. Tiers 2 and 3 are not run.

The plan is [v2-prior-task-plan.md](v2-prior-task-plan.md) and the derivations it tests are in [v2-task-formulations.md](v2-task-formulations.md). The code of every measurement is in [`training/scripts/v2_prior_diag`](../scripts/v2_prior_diag/README.md). "v1.1" is the released weights with tabsil 0.1.1's inference settings (`aiden-jeon/tabsil@v1.1`). "#14's run" and "N10" are the two seeds of the v2 candidate, stage-2 runs on `graph_scm_hicard_missing` (MLflow `277916ec05a3457aa6d07f2bc08a60c4` and `520d88d73cdc4e5ea90cfd27425be00d`, `model/final.pt`).

## Summary

| Task | Bar | v1.1 today | Best without training | Left to the bar | R1 |
|---|---|---|---|---|---|
| Forecasting, fev-bench SQL Elo | 1707 | 1535 | 1535: today's recipe. No arm is better | 172 | in the probe (the rule's limit is 180) |
| Outlier detection, ADBench mean AUROC | 0.77 | 0.663 | **0.756**: the density under 32 random contexts of 64 rows | 0.014 | in the probe (the rule's limit is 0.13) |
| Imputation | none | D4 measured | — | — | to the user (section 12 of the plan) |
| Treatment effects | none | D5 measured | — | — | to the user (section 12 of the plan) |

- **Outlier detection gains 0.092 AUROC from inference alone.** Scoring each row under many small random contexts and averaging the log densities raises ADBench from 0.663 to 0.756, above IsolationForest (0.722) and the TabPFN v2 extension (0.704), and it removes D3's failure on a dense group at 2% (0.425 to 0.987). It stops 0.014 short of the bar.
- **No forecasting arm is better than today's recipe.** The two parts of the recipe that cap a forecast at the context's range are confirmed on D1, but removing them costs 2.2% to 2.6% of fev-bench's SQL. A straight-line trend fitted outside the model reaches the least-squares fit on D1's linear trends (MASE 0.37 instead of 11.75) and costs 3.5% on fev-bench: one task improves by 31% and another gets 41% worse.
- **The v2 candidate's weights are v1.1 on every task suite**, as predicted, with one exception that D4 found: they use a missing cell as information. Where a mask tells about the cell to impute they are 14% better than v1.1, and where it only looks as if it did they are 9% worse.
- **Three predictions of the derivations failed** and are corrected in [v2-task-formulations.md](v2-task-formulations.md): reference rows do not fail on a small dense group, the rare-class output is the class's share of the context only for two classes, and the context-row target scale may raise the trend capture by itself (section "Derivations against the measurements").
- **Cost:** 11.6 H100-h in 35 jobs, against an estimate of about 20.

## Reading point R1

The rule is the user's (plan, section 5): the best arm at or above the bar ends the task without training; a distance to the bar of at most the largest gain others report from such training (180 SQL Elo, 0.13 AUROC) sends the task to the probe; a larger distance stops it.

- **Forecasting: in the probe.** The best is today's recipe at 1535, 172 below the bar.
- **Outlier detection: in the probe.** The best arm is 0.756, 0.014 below the bar. The arm is not at the bar, so the rule raises no release question; that it is an inference change worth 0.092 AUROC is put to the user in "For the user to decide".
- **Next by the plan:** the pilot of item 2.1 on one GPU, then the probe if the pilot moves a task's diagnostic.

Two things the rule does not ask, stated because they bear on what the probe can be expected to give:

- For forecasting, tier 1 lowers the expectation. What fails on D1 is the trend, and neither of the two inference changes that repair D1 helps fev-bench: with a line one task of ten improves by more than 2% (rohlik_orders_1D, by 31%) and six get worse by 2% to 41%. The probe trains the model on the same thing, a continuation beyond the context. The bar is the rating of TimesFM-2.5, which has a lower SQL than TabSil on 7 of the 10 tasks.
- For outlier detection, the probe trains the reference-row form, whose value without training is 0.665. The best score without training, 0.756, is another form (the density) and is not what the probe trains.

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

Counted from the job records (`duration_seconds` of every job, one H100 each). 15 more jobs failed within a second of being submitted, because the workspace allows 8 `GPU_1xH100` nodes at a time; they used no GPU time and were submitted again one by one.

## Differences from the plan

- **Where the code is.** The plan put the inference changes on `main`. The arms are wrappers in `training/scripts/v2_prior_diag/arms.py` and nothing in `tabsil` changes: an arm is moved into the package only if the user releases it.
- **1.2:** (e) was started on the default recipe together with (b) to (d), since it turned out the best of them.
- **1.3:** context sizes of 16, 32 and 128 rows were added after the first results. The best size, 64, is one of the plan's two. Each small-context run also records the median and the mean-density readings of the same evaluations, and D3 has the whole context read the same way as a control.
- **1.4:** Gaussian tables with known values replace the prior tables, since on a prior table the value a fit converges to is not known. The third mechanism is measured in two forms.
- **1.5:** the tables come from a random-network generator in the script, where the effect is known, not from the training prior. All three sets of weights were run.

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

The code of the runs is this branch at `df0da5b`, and at `dd718b9` for the 16- and 32-row contexts on ADBench and the 16-, 32- and 128-row contexts on D3. The runs of 1.1 on the four suites used the evaluation code of `0fd0cf6`.

## For the user to decide

Nothing here is adopted or released by the plan.

1. **The small-context outlier score.** It is an inference change with the v1.1 weights: ADBench 0.663 to 0.756, D3's group 0.43 to 0.99, in less time than today's score. It is behind today's score on 4 of the 12 tables, by 0.001 to 0.07. Releasing it as `TabSilUnsupervised`'s outlier score, as an option, or not at all is the user's decision.
2. **Whether the probe should keep outlier detection.** The rule sends it on. What the probe would add is the last 0.014 to the bar, by a form (reference rows) that starts at 0.665.
3. **What follows D4.** The candidate's prior teaches the model to read a missing cell as information. That is worth 14% where the mask is informative and costs 9% where it is not, and a table does not say which it is. The choice between the two priors for imputation is a choice of what to assume about real masks.
4. **What follows D5.** The S-learner's error grows by a quarter between a randomized experiment and a clip of 0.02 and stays below the T-learner's. The plan trains nothing for effects.
5. **A trend rule for forecasting.** One line over the whole context helps one task of ten and costs 3.5% overall. A rule that decides per series, or fits the recent part of the context only, is not in the plan and was not run.
