# v2 task-formulation checks

Scripts that reproduce the numbers stated as "checked" in
[`training/docs/v2-task-formulations.md`](../../docs/v2-task-formulations.md): sections 1 to 10 as
they were at commit `9933bd0` (closed PR #52), and section 11, added for
[`v2-top5-plan.md`](../../docs/v2-top5-plan.md).
No model weights, no inference, no GPU. NumPy and SciPy only, except `s02c` (imports the repository's
`train_rows_cover_classes`). Each script is deterministic (fixed seeds), prints its numbers and ends
with assertions against the stated tolerance. Three numbers of the document were not reproduced when
the scripts were written; the document was corrected to the scripts' numbers on 2026-11-28 and says
so in place ("Corrections" below). No tolerance was loosened to make a number pass.

Run all (from the repository root; the `s02c` check needs the workspace environment):

```
for f in training/scripts/v2_task_checks/s*.py; do case $f in *s02c*) uv run --frozen --package tabsil-train python -I $f;; *) uv run --no-project --with numpy --with scipy python -I $f;; esac || echo "FAILED $f"; done
```

Every script exits 0. Runtimes below are wall-clock on a laptop, including `uv` start-up. `s02c` is about 11 s
(60,000 tables per configuration instead of 300,000, to stay near the 10 s budget).

## Group 1: claims of the existing document

| Script | Section | Statement | Number stated | Number reproduced | Result | Runtime |
|---|---|---|---|---|---|---|
| s01_pinball_crps.py | 1 | mean pinball over k/1000 against CRPS/2, normal F | 1.0010 (to 1e-4) | 1.000999 | pass | 3 s |
| s01_pinball_crps.py | 1 | integral identity int rho_tau = CRPS/2 | 1e-8 | 3e-15 | pass | |
| s01_pinball_crps.py | 1 | pinball minimizer is the quantile (mixture of N(2,0.5^2) and t4) | argmin = quantile | max error 8e-7 | pass | |
| s02a_conjugate_t.py | 2 (step 7) | posterior predictive t against 400,000 draws, quantiles 0.01 to 0.99 | within 3 Monte Carlo standard errors, largest difference 0.017 (corrected; was "within 0.007") | 0.0170 at the 0.99 level, 2.6 standard errors; other seeds 0.0042 to 0.0082 | pass (asserts 4 standard errors) | 1 s |
| s02b_flat_limit.py | 2 (step 7) | flat-prior limit equals the textbook prediction interval | identity | 3e-11 | pass | 1 s |
| s02c_covering_split.py | 2 (step 7) | Beta(1,1), 20 context rows with one positive, covering split, 1 query | 0.049 | 0.0532 (s.e. 0.004) | pass | 11 s (all of s02c) |
| s02c_covering_split.py | 2 (step 7) | same, 10 queries | 0.050 | 0.0470 (s.e. 0.001) | pass | |
| s02c_covering_split.py | 2 (step 7) | same, random split | 0.094 | 0.0987 | pass | |
| s02c_covering_split.py | 2 (step 7) | covering formula and random formula | 0.050 and 0.091 | 0.0500 and 0.0909 | pass | |
| s02c_covering_split.py | 2 (step 7) | Beta(2,5), 10 context rows with one positive, 1 query | 0.107 (formula 0.106) | 0.1075 (formula 0.1064) | pass | |
| s02c_covering_split.py | 2 (step 7) | Beta(2,5), 6 queries | 0.111 | 0.1102 | pass | |
| s03a_median_cap.py | 3 (step 7) | median of the half-clipped mixture is at most m | F(m) >= 1/2 | min F(m) = 0.5154 over 20,000 cases; 0 violations | pass | 3 s |
| s03a_median_cap.py | 3 (step 7) | quantile at u > 1/2 equals the unclipped members' quantile at 2u-1 when above m | identity | 0 violations in 49,589 (u, case) pairs | pass | |
| s03b_beyond_probs.py | 3 (step 5) | query beyond the context maximum, D1 (n = 365) | 0.3% | 1/366 = 0.27%; simulation matches 1/(n+1) | pass | 1 s |
| s03b_beyond_probs.py | 3 (step 5) | beyond by a share delta, D1 last step (delta = 56/365) | 1e-29 | 10^-29.04 | pass | |
| s03b_beyond_probs.py | 3 (step 5) | every query beyond the context, D1 (N = 421, n_Q = 56) | below 1e-70 | 10^-70.4 | pass | |
| s03c_trend_scale.py | 3 (step 6) | context targets, all-row units: mean; sd | -0.23; 0.87 | -0.2304; 0.8669 (seed average) | pass | 0.5 s |
| s03c_trend_scale.py | 3 (step 6) | N / n_Q | 7.5 | 7.518 | pass | |
| s03c_trend_scale.py | 3 (step 6) | sqrt(12)(1/2 + h/T), h = 1..56 | 1.74 to 2.26 | 1.7415 to 2.2635 (discrete standardization 1.7368 to 2.2588) | pass | |
| s03d_d1_forecast.py | 3 (step 7) | seasonal naive MASE, D1 linear trend 15 | 3.674 | 3.6736 (trend capture 0.279 against 0.28) | pass | 0.5 s |
| s03d_d1_forecast.py | 3 (step 7) | least-squares MASE, D1 linear trend 15 | 0.368 | 0.3680 (trend capture 0.993 against 0.99) | pass | |
| s04a_density_auroc.py | 4 (step 7) | density-score AUROC at shares 0, 1e-6, 1e-5, 1e-4, 0.02 | 0.9995, 0.857, 0.527, 0.168, 0.003 | 0.9994, 0.8574, 0.5271, 0.1685, 0.0037 | pass (tolerance 0.01) | 1 s |
| s04a_density_auroc.py | 4 (step 7) | likelihood-ratio AUROC | 1.000 | 1.0000 at every share | pass | |
| s04a_density_auroc.py | 4 (step 7) | threshold sigma^d/(1 + sigma^d), sigma = 0.15, d = 6 | 1.1e-5 | 1.139e-5 | pass | |
| s04b_uniform_binomial.py | 4 (step 7) | uniform outliers on [-6,6]^6: AUROC of both scores | 0.9992 | 0.9992 both; density order non-decreasing in -log p0 (5,355 float ties in the far tail) | pass | 1 s |
| s04b_uniform_binomial.py | 4 (step 7) | 64 rows, share 0.02: no group member; at most one | 0.27; 0.63 | 0.2745; 0.6329 (simulated 0.2738; 0.6337) | pass | |
| s05a_selfmask_closed.py | 5 (step 7) | self-masking gap of means, b = 1 | -1.17 | -1.1706 (Monte Carlo -1.1705) | pass | 2 s |
| s05a_selfmask_closed.py | 5 (step 7) | same, b = 0.5 and b = 2 | (not stated) | -0.7563 and -1.4487 (Monte Carlo agrees to 0.001) | no stated value; closed form verified | |
| s05a_selfmask_closed.py | 5 (step 7) | RMSE of complete-case-mean imputation; of E[x given M = 1] | 1.42; 0.80 | 1.4153; 0.7955 | pass | |
| s05a_selfmask_closed.py | 5 (step 7) | gap with a predictive column, rho = 0, 0.5, 0.9, 0.99 (x1 = 0) | -1.17, -0.94, -0.30, -0.03 | -1.1706, -0.9434, -0.2954, -0.0337 (Monte Carlo -1.177, -0.943, -0.299, -0.034) | pass | |
| s05b_mask_by_other_column.py | 5 (step 7) | complete-case regression of x on x1, mask Phi(-0.74 + 1.5 x1), rho = 0.6 | slope 0.600, intercept 0.001 | 0.6000, -0.0003 | pass | 2 s |
| s05b_mask_by_other_column.py | 5 (step 7) | mean x1 of rows to impute against context | 0.90; -0.46 | 0.8953; -0.4627 | pass | |
| s05b_mask_by_other_column.py | 5 (step 7) | rows to impute beyond the context's 99th percentile of x1 | 32% | 31.6% | pass | |
| s05c_mar_counterexample.py | 5 (step 7) | truth E[x1 given x2 = 1] (rho = 0.6) | 0.60 | 0.600 | pass | 1 s |
| s05c_mar_counterexample.py | 5 (step 7) | mean x1 of context rows at x2 = 1 | 0.17 | 0.1761 (quadrature; rounds to 0.18), 0.1832 (window) | pass (tolerance 0.01 on quadrature) | |
| s05c_mar_counterexample.py | 5 (step 7) | context regression of x1 on x2: slope; intercept | 0.48; -0.30 | 0.4827; -0.2952 | pass | |
| s06a_overlap_sd.py | 6 (step 7) | sd of the least-squares effect, clip 0.5 to 0.02 (simulated) | 0.046 to 0.066 (corrected; was 0.041 to 0.062) | 0.0456 to 0.0660 (400 replications) | pass (tolerance 0.003) | 3 s |
| s06a_overlap_sd.py | 6 (step 7) | formula at clip 0.5 and 0.02 | 0.045 and 0.065 (corrected; was 0.063) | 0.0447 and 0.0646 | pass | |
| s06a_overlap_sd.py | 6 (step 7) | T-learner sd, clip 0.5 to 0.02 | 0.046 to 0.066 (corrected; was 0.060 to 0.088) | 0.0456 to 0.0661 (ATE reading) | pass | |
| s06b_density_ratio_interval.py | 6 (step 6) | density ratio, clip 0.1 and 0.02 | 18; 98 | 18.00; 98.00 | pass | 1 s |
| s06b_density_ratio_interval.py | 6 (step 6) | randomized queries, clip 0.1 and 0.02 | 3.6; 3.9 | 3.60; 3.92 | pass | |
| s06b_density_ratio_interval.py | 6 (step 7) | posterior sd of tau(x) at x = 0; at x = 1.5 | 0.19; 0.34 | 0.1918; 0.3248 (tolerance 0.02) | pass | |
| s06b_density_ratio_interval.py | 6 (step 7) | sd of the difference of two predictive distributions at x = 0; 1.5 | 1.43; 1.45 | 1.4272; 1.4510 | pass | |
| s07a_chain_rule.py | 7 | chain rule along three random orders equals the joint log density (5-column Gaussian) | identity | max error 1.8e-14 | pass | 1 s |
| s07b_ranking_conformal.py | 8 | expected gains of the sure relevance 1 and of the 50/50 relevance 0 or 2 | 1 and 1.5 | 1.0 and 1.5 | pass | 1 s |
| s07b_ranking_conformal.py | 8 | split conformal coverage at least 1 - alpha = 0.9 (m = 19, 99, 499) | >= 0.9 | 0.8979 (s.e. 0.0034), 0.9009, 0.9004 | pass (exact bound 18/20 = 0.9 for m = 19; the simulated value is within one s.e.) | |
| s08_hazards.py | 8 | person-period frequencies recover the hazards | 0.002 | 0.0018 | pass | 0.5 s |
| s08_hazards.py | 8 | product of person-period hazards is the Kaplan-Meier curve | identical | identical (0) | pass | |
| s08_hazards.py | 8 | truth at the fourth bin | 0.69 | 0.6857 (Kaplan-Meier 0.6840) | pass | |
| s08_hazards.py | 8 | uncensored subjects alone: survival at the fourth bin, censoring uniform on bins 1 to 8 | 0.21 (corrected; was 0.37 with an unrecorded censoring range) | 0.2122; 0.350 with censoring on 1..12 | pass (tolerance 0.01) | |
| s10a_mixture_bound.py | 10 (step 3) | 0 <= E_0 KL(p_0 \|\| q*) <= -log(1 - w), 20,000 random cases | holds | 0 violations; largest ratio to the bound 0.75 | pass | 2 s |
| s10a_mixture_bound.py | 10 (step 3) | both extremes reached; -log(1 - 0.25) | 0 and -log(1-w); 0.29 | 0 and 0.287682; 0.2877 | pass | |
| s10b_sorted_split.py | 10 (step 3) | query beyond the context in some column, probability at most 2d/(n+1), n = 1000, d = 20 | 4% | bound 0.0400; exact (independent columns) 0.0392; simulated 0.0382 and 0.0340 (correlated 0.5) | pass | 2 s |

## Group 2: section 11

| Script (section) | Claim | Number reproduced | Pass or result | Runtime |
|---|---|---|---|---|
| s11a_mixture_bma.py (11.2) | (i) on A-tables, mixture excess over A is >= 0 and <= E[-log P(A given D)] <= -log(1 - w). Excess at w = 0.1, n = 8, 16, 32, 64, 256: +0.0033, +0.0008, +0.0005, +0.0005, 0.0000; E[-log P(A given D)] = 0.077, 0.054, 0.028, 0.018, 0.000; -log(0.9) = 0.105 | bounds hold within 3 s.e. | pass | 5 s |
| s11a_mixture_bma.py (11.2) | (ii) coverage gain on B-tables: RMSE of A minus RMSE of B, by n = 8, 16, 32, 64, 256 | +0.738, +0.484, +0.281, +0.183, +0.086 (printed) | reported | |
| s11a_mixture_bma.py (11.2) | (iii) mixture excess over the matching component falls with n (both components) | n = 8 to 256 falls on A and B tables | pass | |
| s11b_dataset_share.py (11.2) | G = (1 - x)^(m/51) equals the direct geometric mean over a simulated 51-vector | max difference 1.1e-16 | pass | 0.4 s |
| s11c_queries_per_table.py (11.3) | (a) variance (sb2 + sw2/m)/B | simulated / formula 0.9974 | pass | 1.5 s |
| s11c_queries_per_table.py (11.3) | (b) grid optimum equals m* = sqrt(c_ctx n sw2/(c_q sb2)) for three settings | 32 vs 31.62; 100 vs 100.00; 5 vs 5.00 | pass | |
| s11c_queries_per_table.py (11.3) | (c) final SGD excess / noise variance constant across (B, m) | relative spread 1.6% | pass | |
| s11c_queries_per_table.py (11.3) | (d) variance factor when m doubles, (1 + r)/(1 + r/2), r = sw2/(m sb2) | simulated 1.3350 against 1.3333 (r = 1) | pass | |
| s11d_shared_representation.py (11.4) | OLS risk d/(n - d - 1) | 20,000 reps: 2.2083 (2.2222), 0.5119 (0.5128), 0.1120 (0.1117) for n = 30, 60, 200 | pass | 5 s |
| s11d_shared_representation.py (11.4) | oracle U, theta = 0: r/(n - r - 1) | 0.1154 (0.1154), 0.0534 (0.0536), 0.0155 (0.0153) | pass | |
| s11d_shared_representation.py (11.4) | oracle U, any theta: (1 + sin^2 ‖beta‖^2) r/(n - r - 1) + sin^2 ‖beta‖^2 (the bias term alone is not the whole excess; see notes) | e.g. n = 200, theta = pi/8: 0.3931 against 0.3929 | pass | |
| s11d_shared_representation.py (11.4) | aux estimator risk (500 reps; theta = 0, pi/8, pi/4, pi/2) | n = 30, K = 10: 1.12, 1.40, 2.09, 3.06 (OLS 2.08); n = 60, K = 10: 0.17, 0.54, 1.44, 2.73 (OLS 0.51); n = 200, K = 10: 0.039, 0.413, 1.314, 2.588 (OLS 0.112); K = 50 at n = 200: 0.023, 0.399, 1.306, 2.591 | reported | |
| s11e_compute_equivalence.py (11.5) | G = (1 + kappa)^(ln G0 / ln R) for three anchors; reproduces each anchor at 1 + kappa = R. Its Elo column is the rule of 40 per 1%; the section's live Elo per doubling (+46, +72) comes from `gap_report.py`'s `whatif_share.csv` | anchors reproduced; table in the output (e.g. kappa = 1: G 0.9930, 0.9773, 0.9872; kappa = 3.6, Kumo S->M: 0.9508) | pass | 0.3 s |
| s11f_pairwise_auc.py (11.6) | well-specified: CE and pairwise AUC equal, and equal the Bayes AUC | 0.8447, 0.8447, 0.8447 | pass | 2 s |
| s11f_pairwise_auc.py (11.6) | misspecified: AUC difference (pairwise minus CE) and sign (finding, not asserted) | CE 0.7618, pairwise 0.7615, Bayes 0.9229; difference -0.0003 (CE higher; within noise) | reported | |
| s11g_truncated_mean.py (11.6) | mean of the 999-quantile output against the true mean, and the trapezoid reading | normal: step bias +0.0031 (= Phi^-1(0.999)/1000), trapezoid 0; t3 +0.0059; t5 +0.0046; lognormal sd-unit bias: sigma 0.5 +0.0026 (trapezoid -0.0011), sigma 1 +0.0015 (-0.0035), sigma 2 -0.0050 (-0.0095); MSE share (b^2/(1 + b^2)) at most 4e-5 | reported | 1 s |

## Corrections

When the scripts were written, three stated numbers were not reproduced. The document now carries the
scripts' numbers, each corrected in place with the date (its "Checks as scripts"):

- **s02a** (section 2, step 7): "within 0.007" was the largest difference of one run whose seed and
  design were not recorded. In this design seed 0 gives 0.017 at the 0.99 level and seeds 1 to 4 give
  0.004 to 0.008, all within 2.6 Monte Carlo standard errors of a sample quantile. The check asserts
  4 standard errors at every level.
- **s06a** (section 6, step 7): the T-learner's 0.060 to 0.088 is not reproduced by any reading of the
  stated design (the mean effect over the rows gives 0.0456 to 0.0661; the effect at x = 0 the same; at
  x = 1.5 0.084 and 0.118; the root mean squared error over the rows 0.057 and 0.081), and the stated
  least-squares 0.041 lies 3 standard errors from the formula. The corrected sentence has 0.046 to
  0.066 for both and 0.045 and 0.065 for the formula; both grow by the same factor, 1.45.
- **s08** (section 8): the uncensored-only 0.37 depends on a censoring range that was not recorded;
  with the censoring bin uniform on 1 to 8 it is 0.21 (0.35 on 1 to 12). The corrected sentence states
  the range.

## Design readings (where the document was not complete)

- s01: CRPS in closed form; the mixture for the minimizer is 50/50 N(2, 0.5^2) and unscaled t4.
- s02a: n = 40, d = 3, X and beta drawn with seed 0 as stated in the docstring (the document gives no seed).
- s02c: a table is N = n + n_Q rows with theta ~ Beta(a, b) and Bernoulli labels; the repository function
  orders the rows; the context count is conditioned on. 60,000 tables per configuration; the standard
  error of the 1-query figures is about 0.004.
- s03b: "beyond by a share delta" means beyond the context maximum by delta times the feature's range
  (range 1 for U(0, 1)); the D1 row uses delta = 56/365.
- s03c: population sd (ddof 0) for the all-row standardization; mean and sd averaged over seeds 0..3.
- s04a/b: test rows are 20,000 inliers against 20,000 group (or uniform outlier) members.
- s05a: a = -0.74 as the document states (the missing rate is then 30.0%); the predictive-column bias is the
  closed form at x1 = 0 and is also checked by a window of |x1| < 0.05.
- s05c: (x1, x2) standard bivariate normal with correlation 0.6 (this is the reading that gives the stated
  truth 0.60); the context is the rows that observe both columns.
- s06a: LS estimator is the t coefficient of an OLS of y on (1, x, t); clip c gives e in [c, 1 - c].
- s06b: the posterior of the four coefficients with precision X'X + I/100 (sigma = 1), design by seed 0.
- s07b: the predictor is a linear fit on an independent training set of 100 rows; 400 replications.
- s08: 200,000 subjects; hazard 0.09 in every bin; censoring uniform on 1..8 bins.
- s11a: 2,000 tables per component, 20 queries per table, seeds 0 (A) and 1 (B); RMSE against x'beta.
- s11d: n = 30, 60, 200; 500 replications for the table; the approximations use 20,000 replications (the risk is
  heavy-tailed at n = 30). The oracle risk includes the omitted component's effect on the estimated
  coefficients: its exact value is (1 + ‖v‖^2) r/(n - r - 1) + ‖v‖^2 with v the component of beta
  orthogonal to U, which is what the simulations match. With K < r only K directions exist, so the
  estimator uses min(r, K) singular vectors.
- s11f: 200,000 draws for the CE fit and the AUC; 1,500 x 1,500 pairs for the pairwise fit.
- s11g: "standardized" lognormal means scaled to sd 1 without centring, so the true mean is E[X]/sd(X) (sd
  units). The step reading puts each 0.001 bucket at its upper quantile and the top tail at q_999.
  Bias b is the output mean minus the true mean; the MSE share is b^2/(1 + b^2) for an otherwise exact
  prediction.
- s11e and s11b are arithmetic; their Elo is 4000 (1 - G) (40 Elo per 1% of error), not live Elo.

## Not checked

- Nothing here uses the TabSil weights; D1, D3, D4 and D5 measurements are not reproduced (only their
  generator arithmetic in s03d).
- The s11a and s11d results are for their stated designs; they say nothing about the TabSil model.
