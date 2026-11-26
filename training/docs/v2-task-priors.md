# v2 training prior: task priors, trained from scratch

Status: design. No v2 training job has run. Diagnostics D1-D3 ran on the v1 weights; D4 and D5 are planned.

## Decision

v2 is trained from scratch with the v1 architecture and losses (`tabsil.recipe.STRUCTURE`, the retrieval decoder for classification, 999 quantiles with the pinball loss for regression). Only the training data changes: the v2 prior adds task priors to v1's classification and regression tables. The three-stage curriculum keeps v1's lengths (150K / 10K / 2K steps, about 1,550 H100-h). So v1 is the control of the final run: same structure, recipe and budget, and only the prior differs.

TabSil 0.1.x already handles four tasks zero-shot with the v1 weights. Each wrapper turns its task into the regression or classification the model was trained on:
- forecasting: `tabsil.forecast`, #17
- outlier scores: `tabsil.unsupervised`, #19
- imputation: `TabSilUnsupervised.impute`
- treatment effects: `TabSilCATE`

The last two are on their branches, waiting for #21. v2 keeps these wrappers and trains on tables shaped like the ones they send.

Keeping the architecture limits what a task prior can do:
- Every task must be a classification or regression table that the current model reads. There is no task token, no column role and no new head.
- So a task prior can change only three things: which rows are the eval rows (the split), what the context holds, and how y is scaled.
- Two ideas need a new architecture and are left out of v2: a separate outlier head (FoMo-0D, OutFormer) and a treatment-column indicator (Do-PFN). They are v3 candidates if v2 falls short on those tasks.

The release is named v2 (Hub tag `v2`, tabsil 0.2.0), because the weights are retrained from scratch on a different prior. The model card states that the structure is v1's.

| Task | Zero-shot gap of the v1 weights | v2 task prior | Status |
|---|---|---|---|
| Forecasting | D1: with the 0.1.1 recipe no forecast goes more than 0.14 past the largest context value; without its clipping the forecasts go higher but still miss the trend | temporal split, y scaled with the train rows, and a forecasting recipe that does not clip | design (§1) |
| Outlier detection | D3: a small dense group of anomalies scores as inliers | reference rows labeled 1 in the context | design (§2) |
| Imputation | not measured yet (D4) | eval rows chosen by a MAR/MNAR mechanism | proposal, only if D4 shows a gap (§3) |
| Treatment effects | not measured yet (D5) | eval rows with an intervened binary column | proposal, only if D5 shows a gap (§4) |

## Diagnostics (v1.1 weights, tabsil 0.1.1 defaults)

AI Runtime, 1x H100, MLflow run `tabsil-v2-prior-diag` `27795403ab8d427988f65ab575662179` (2026-11-26). The code is main at f5b26a7, which includes release/0.1.1. The weights are pinned to `aiden-jeon/tabsil@v1.1` (`DIAG_MODEL`), and their config sets the inference recipe. Nothing failed and nothing is imputed.

Earlier runs (`edee9216…`, `ffc589d4…`) had no `plain` arm and gave D1 `default` and D3 identical tables. D2 moved by at most 0.01 once the outer buckets were scored as half-normal tails.

### D1: forecasts stay at the largest context value

Synthetic daily series: 365 context steps and a 56-step horizon. Each is level + trend + weekly and monthly seasonality + noise (sd 1). Strength is the trend's rise over the horizon in noise sds. 4 seeds.

The two arms separate the model from the 0.1.1 inference recipe:
- `default`: the weights' recipe, which is 8 numeric-cycle members, half with the target power transform.
- `plain`: 2 `standard` members without the target power transform. Nothing in this recipe bounds a forecast; why the default one does is explained below.

The forecast-max column is the largest forecast minus the largest context value, as the mean and the maximum over seeds.

| Trend | Strength | True max − context max | Forecast max − context max, default (mean / max) | plain (mean / max) | MASE default | MASE plain | MASE seasonal naive | MASE OLS (index + sin/cos) |
|---|---|---|---|---|---|---|---|---|
| linear | 0 | −1.03 | −2.16 / −1.15 | −2.19 / −1.13 | **0.51** | 0.51 | 0.86 | 0.51 |
| linear | 2 | 1.67 | −0.47 / −0.07 | −0.09 / 0.19 | 1.53 | 1.24 | 0.95 | 0.51 |
| linear | 5 | 5.12 | 0.01 / 0.03 | 2.56 / 3.13 | 3.42 | 2.73 | 1.62 | 0.49 |
| linear | 15 | 14.11 | 0.03 / **0.05** | 8.50 / 12.98 | **11.75** | **9.03** | 3.67 | 0.37 |
| exponential | 15 | 13.52 | 0.07 / 0.09 | 3.56 / 4.69 | 4.72 | 4.19 | 4.36 | 7.43 |
| piecewise | 15 | 14.11 | 0.07 / 0.14 | 5.37 / 5.68 | 4.90 | 4.16 | 4.59 | 11.53 |

- Without a trend both arms equal OLS and beat seasonal naive, so the periodic part is learned.
- With the default recipe no forecast, over every seed, goes more than 0.14 above the largest context value. The linear trend's forecast even falls back (trend capture −1.31). This is the limitation the TabPFN-TS paper reports for TabPFN (Hoo et al. 2025, arXiv 2501.02945, Section 6.1).
- **Part of that is the inference recipe.**
  - The target-power members map their distribution back through `_clipped_inverse`, which clips it to the train target range (`tabsil/inference/ensemble.py`). Half of the 8 members can put no mass above the context maximum.
  - The `quantile` feature kind maps every running-index value beyond the train rows to the top quantile, and `robust` squashes it into ±3.
  - Without these (`plain`), forecasts go up to 13 above the context maximum.
- **The other part is the model.** Even `plain` misses the trend: on linear 15 its MASE is 9.03, worse than seasonal naive (3.67), and its trend capture is −0.71, so the mean forecast still falls back.

The v1 prior explains the model's part. The graph SCM's regression target is passed through TabICL's `outlier_removing(threshold=4)` and then `standard_scaling` over all rows of the table, train and eval together (`tabicl/prior/_graph_scm.py`, `_reg2cls.py`). Eval rows are a random subset of the same rows. So the model has practically never been asked for a held-out target far outside the train targets' range.

Both parts need a fix: the forecasting prior (§1), and a forecasting recipe that does not clip.

### D2: unpredictable targets are calibrated about as well as predictable ones

NLL on 500 test rows. The model's density is `predict_distribution`'s: uniform within the inner buckets, half-normal in the two outer ones. The excess is the model's NLL minus the true distribution's NLL, averaged over 100, 500 and 2,000 train rows, 1 and 5 features, and 3 seeds.

| Target | Excess over the true density | Excess over a model that ignores x |
|---|---|---|
| classification, random labels (0.6/0.3/0.1) | 0.001 | 0.004 (the train label frequencies) |
| y independent of x: normal | 0.087 | 0.077 (a KDE of the train targets, as below) |
| y independent of x: lognormal | 0.142 | −0.156 |
| y independent of x: Student t3 | 0.113 | −0.138 |
| y independent of x: bimodal | 0.268 | 0.173 |
| control, y = sin(2 x0) + N(0, 0.3²) | 0.412 | — |

The v1 prior drops unpredictable datasets (`filter_unpredictable`, `tabsil_train/recipe.py`). That did not leave the model overconfident on independent targets: it is no worse there than on the predictable control. Turning the filter off is not part of this design. The weak spot is the shape: a two-mode target is 0.17 nats worse than a KDE.

### D3: clustered anomalies in the context break density-based scores

Inliers lie on a nonlinear 6-dimensional manifold. The context has 600 rows, and outliers make up 0-20% of them. The test set is 300 inliers and 30 outliers of the same kind. Scores are TabSil −log density (3 column orders) and IsolationForest. The table shows AUROC, 3 seeds.

| Outliers | 0% | 2% | 5% | 10% | 20% | IsolationForest at 10% |
|---|---|---|---|---|---|---|
| global (uniform in 1.2x the box) | 1.000 | 0.993 | 0.998 | 0.998 | 0.992 | 0.987 |
| local (2 columns shifted 3 sd) | 0.999 | 1.000 | 0.997 | 0.998 | 0.975 | 0.996 |
| **cluster** (a 0.15 sd blob 2 sd off) | 0.999 | **0.425** | 0.427 | 0.440 | 0.453 | 0.853 |
| **curve** (a 1-d curve 1.5 sd off) | 1.000 | **0.679** | 0.218 | **0.064** | 0.050 | 0.961 |

- Scattered outliers do no harm even at 20%.
- Twelve similar anomalies (2%) are enough to make the cluster region dense, and the score fails. On the curve the anomalies end up with a higher log density than the inliers (+7.1 against −2.3 at 10%).
- The density is not wrong. A density score cannot flag a dense group by construction. That fits #19's gap between contaminated and clean context on ADBench (AUROC 0.662 against 0.820). Several tables there have anomalies that form a class of their own (cover, PageBlocks, cardio).
- So more prior variety for the per-column conditionals would not fix it. The model needs a task in which a small dense group in the context is still an outlier.

## What the code allows today

Checked on `dev-v2` (de8225c) and main (f5b26a7).

- **Regression loss.** The regression output is 999 quantiles with the pinball loss, and a missing (NaN) target contributes 0 (`tabsil/models/quantile_distribution.py`). Eval rows can therefore be left out of the loss without a code change. The regression borders are only used at inference to turn quantiles into bucket probabilities.
- **Out-of-range targets at inference.** The borders span ±22.6 train-target sds (v1), so the model's output can put mass far outside the context range. The 0.1.1 recipe still keeps it in, as D1 shows:
  - The target-power members clip their distribution to [min, max] of the train targets (`_clipped_inverse`).
  - The `quantile` and `robust` feature kinds saturate inputs beyond the train range.
- **Classification decoder.** It is a retrieval decoder (`tabsil/models/column_row_transformer.py`, `RetrievalDecoder`). A test row's probabilities are the attention-weighted average of the train rows' one-hot labels. **A class that no train row carries gets probability 0.** A one-class setup, with every context row labeled 0 and the model asked for P(outlier), is therefore impossible. Eval rows can be left out of the classification loss with `IGNORE_INDEX` (−100).
- **Scaling at inference.** Inference standardizes x and the regression y with the train rows only (`standardize_cells`; `EnsembleRegressor.fit`, `inference/ensemble.py`). The v1 prior standardizes y over all rows and compresses it beyond 4 sd, so the two disagree whenever eval targets leave the train range.
- **Batches.** A `PriorBatch` has one `n_train`, and the eval rows are the last rows of every dataset. A temporal split only has to put the time steps in order before that cut.

## Base prior and guards

- **Base prior.** #14's `graph_scm_hicard_missing` (merged into dev-v2, de8225c). Its stage 2 run passed every #14 criterion (TabArena-Lite Elo 1769 against 1727 for `stage2-10k`). That run had one seed and no control with the v1 prior (A0). The screening control below uses this prior, so the v2 decisions do not depend on A0.
- **Guards for every run.** Each run is compared with its reference: C in the screening, v1 in the final run, both at the same tabsil inference version. A run fails if either holds:
  - TabArena-Lite overall, classification or regression Elo is lower beyond its 95% CI.
  - AMLB ROC AUC or the regression RMSE ratio is worse by more than 0.005.

## 1. Forecasting prior

Goal: tables in which the eval rows are the future of the train rows, built from the same features that `TabSilForecaster` builds. The eval targets must be allowed to lie outside the train targets' range.

**Series (one per dataset).** `y(t) = level + trend(t) + seasonal(t) + covariate effect + noise` for t = 0..n−1, with components from ForecastPFN (Dooley et al. 2023, 2311.01933), Mamba4Cast (2410.09385) and KernelSynth (Chronos, 2403.07815):
- trend: none, linear, exponential or damped, with 0-3 change points (piecewise linear, as in D1). The slope is set so that the rise over the horizon ranges from 0 to about 20 train-target sds.
- seasonal: sums of sines at calendar periods of a random frequency (7 and 30.5 for daily data, 24 and 168 for hourly, 12 for monthly, …) and at random periods. Also some periodic GP samples (KernelSynth's periodic and RBF kernels).
- noise: iid Gaussian or Student t, AR(1), or multiplicative. Some series get level shifts, spikes or zero inflation (redset_1H in fev-bench is 95% zeros).
- covariates: in about 30% of the series, 1-3 graph SCM columns over t that also enter y. They are passed as known covariates, as `TabSilForecaster` does with columns present in both the context and the future frame.

**Features.** `tabsil.forecast.time_features` is called on a synthetic timestamp index at the drawn frequency. The prior then uses the inference code itself: the running index, the calendar sin/cos and the FFT seasonal periods found in the context part only. This is the same principle #14 follows for categorical codes.

**Split and scaling.**
- The train rows are the first n_train time steps and the eval rows the rest, so the split is temporal. The horizon fraction comes from the existing table size rule.
- y is standardized with the train rows' mean and sd, as at inference.
- TabICL's 4 sd outlier compression is not applied. The horizon is clipped at ±20 sd to stay inside the borders.

**Mixture.**
- Start with a share of 0.1 of the regression datasets, in all three stages. Chronos found about 10% synthetic data best when mixed with real series, and more synthetic data hurt there.
- Classification batches are unchanged.

**Inference recipe (main, before the runs are judged).**
- `TabSilForecaster` needs members that can leave the context range: no target power transform, or an inverse that is not clipped to the train targets.
- The running index also needs a kind that does not saturate (`standard` or `power`, not `quantile` or `robust`).
- D1's `plain` arm is the first version of this recipe. The forecasting recipe change ships on main with the v1 weights, and fev-bench is measured again with it before phase 1. Every forecasting comparison below then uses that recipe, so a prior gain is not confused with a recipe gain.

**Success criteria (final run).**
- fev-bench task-mean SQL Elo above v1's with the same forecasting recipe. v1 scored 1526 with the 0.1.1 recipe.
- The D1 linear-15 MASE below seasonal naive in the `plain` arm.

## 2. Outlier detection prior

Goal: a task in which a context with hidden outliers, including small dense groups, still scores those outliers high. It must work with the retrieval decoder, so the context has to contain both classes.

**Recommended: reference rows ("unsupervised as supervised", Hastie, Tibshirani & Friedman, ESL §14.2.4).**

The context holds:
- the data rows, all labeled 0 (inlier). They are an SCM table whose rows include hidden outliers at a contamination of U(0, 0.15).
- reference rows labeled 1, drawn from a broad distribution derived from the data rows.

The eval rows are held-out data rows with their true label (0 inlier, 1 outlier). About half are outliers, as in FoMo-0D's queries. All eval rows count in the loss.

The model has to learn that rows in a small group far from the bulk belong with the reference class, even though the context labels them 0. Contaminated context is the point here. FoMo-0D (2409.05672) and OutFormer (2602.03018) train only with clean context and contaminate only the queries, which does not cover the D3 failure.

At inference a new `TabSilOutlier` appends reference rows built by the same code and returns P(class 1) per test row. No change to the architecture or the loss is needed.

- **Outlier kinds.** These are the D3 kinds plus OutFormer's SCM-based ones:
  - global (a padded box)
  - local (a few columns shifted)
  - cluster (a tight blob of 1-10% of the rows)
  - dependency (a node of the SCM resampled from its marginal, which breaks its edges)
  - a second small SCM's rows (structured off-manifold groups, the D3 curve case)
- **Reference distribution.**
  - Half the time it is a product of the data rows' marginals (each column shuffled independently), which flags dependency anomalies.
  - Half the time it is uniform in the padded bounding box, which flags global ones.
  - The reference rows make up 20-50% of the context. Inference uses one fixed recipe, and the prior draws around it.
- **Mixture.** Start with a share of 0.1 of the classification datasets, in all three stages. Every other classification dataset is unchanged, and `CoverClasses` still holds (both classes appear in the context).
- **Success criteria (final run).**
  - ADBench (12 tables × 3 seeds, #19 protocol) mean AUROC above IsolationForest's 0.722, and contaminated AUROC within 0.05 of v1's clean-context 0.820.
  - The D3 cluster at 2% at or above 0.9.

**Out of scope: a separate outlier head** as in FoMo-0D and OutFormer, trained with a binary loss on contaminated queries. It changes the architecture, so it is not part of v2. If the reference rows fail, outlier detection stays zero-shot in v2.

**Not recommended: contamination with an inlier-only loss on the per-column conditionals.** The loss would ignore the outlier eval rows, but D3 shows that the context's outlier group is learned as density. The model would stay unconstrained exactly where it fails.

## 3. Imputation prior (only if D4 shows a gap)

**Where the v1 prior differs.** `TabSilUnsupervised.impute` fits one model per column. That column is the target, and the rows missing it are the eval rows. In real tables the missingness mechanism selects those rows: under MAR it depends on the other columns, and under MNAR on the missing value itself. In the v1 prior the eval rows are a random subset, which is MCAR on the target. #14 already puts MAR/MNAR missing cells into x (`StructuredMissing`), so the input side is covered and the target side is not.

**Prior.** Base-prior tables, including #14's missing cells in x, whose eval rows come from a missingness score instead of a random draw:
- MAR: the score is logistic in 1-3 x columns.
- MNAR: the score is logistic in y (self-masking).
- The batch keeps its single n_train. The n_eval rows are drawn without replacement with probabilities from the score and moved to the end.
- Start with a share of 0.1 of the classification and of the regression datasets.

TabImpute (2510.02625) found that mixing missingness patterns lowered its default model, which is MCAR-only. So the first arm uses MAR only, and an MNAR arm follows only if the MAR arm passes.

**D4 (to run before the arm).** Imputation error of the v1.1 weights under MCAR, MAR and MNAR masks of the same rate:
- on SCM tables with a known mechanism, and on the impute suite (`--suite impute`)
- against the suite's baselines

The arm runs only if TabSil's error grows from MCAR to MAR/MNAR clearly more than the baselines' does.

**Success criteria (final run).** The impute suite error is lower than v1 under MAR and MNAR masks and not higher under MCAR.

## 4. Treatment effect prior (only if D5 shows a gap)

**What fits the structure.** In the S-learner, the default of `TabSilCATE`, the treatment is an ordinary feature column, and the model predicts y for each row with t set to 0 and to 1. With every confounder observed, E[y | x, t] is the interventional mean, so the regression prior already trains the right target. What v1 never sees is the query itself: the S-learner asks for (x, t) pairs that are rare in the context, such as treated rows where treatment was unlikely. In the v1 prior the eval rows follow the train rows' distribution.

**Prior.**
- Base-prior regression tables in which one binary node T is a feature. T's parents are pre-treatment columns, and T enters y.
- No post-treatment column is passed as a feature, so the conditional mean stays the interventional mean and the target stays correct.
- Train rows are observational: T comes from its structural equation, with propensities pushed toward 0 and 1 by a random amount to vary overlap.
- Eval rows draw T ~ Bernoulli(0.5) independently of x, as do(T), and take y from the SCM under that T.
- Start with a share of 0.05 of the regression datasets.

This cannot cover hidden confounding or mediators, nor tell the model which column is T. Those need the Do-PFN indicator, which is v3.

**D5 (to run before the arm).** On the causal suite the S-learner is already strong (IHDP √PEHE out of sample 0.53, against 2.08 for the EconML T-learner with GBM). D5 measures the v1.1 S-learner's √PEHE on SCM tables while the propensity clip shrinks from 0.5 to 0.02, against the true CATE and the EconML T-learner. The arm runs only if TabSil's error grows clearly faster than the T-learner's as overlap shrinks.

**Success criteria (final run).** IHDP and ACIC √PEHE not higher than v1, and lower on D5's low-overlap tables.

## Open items before implementation

1. **Sync.** `tabsil.forecast` and `tabsil.unsupervised` are not on dev-v2. Merge main into dev-v2 with a PR first. Merge again after #21 brings imputation and `TabSilCATE` to main.
2. **Prior options.** Each task prior is an option of the base prior, as in #14 (`--prior-type`), and is off by default.
3. **Shared code.** The forecast features (`time_features`, `seasonal_periods`) and the reference rows come from `src/tabsil/` and are imported by the prior, so the training and inference recipes cannot drift apart. The reference rows are new code, shipped with `TabSilOutlier`.
4. **Forecast covariates.** The SCM covariate path needs a time-indexed root. Either add a generator that wraps the graph SCM with t as an extra root, or use the series generator alone and draw the covariates separately. Decide in the implementation PR.
5. **Forecast horizon in stage 1.** Stage 1 eval shares go up to 70% of the rows. That is longer than most fev-bench horizons relative to their contexts. The batch has one n_train, so capping the horizon of the forecast datasets means a separate n_train for them. Decide in the implementation PR.
6. **Borders.** Keep drawing the regression borders from the base regression prior (`make_borders`), not from the mixture. #14's options change only x, so with v1's sizes they stay at v1's ±22.6 sd, and the forecast clip at ±20 sd stays inside them.
7. **Forecasting recipe.** The inference change of §1 lands on main with its own fev-bench run. It uses the v1 weights, does not retrain, and is a 0.1.x release. Phase 1 starts after it.
8. **Screening length.** It is not known whether 30K steps separate the priors. The 250K stage-1 run was measured on TabArena-Lite up to 225K; check its MLflow evals for early checkpoints before fixing the length.

## Experiment plan (each run needs approval before it starts)

The earlier plan screened each task prior with a stage 2 run from v1's `stage1-150k` (≈ 307 H100-h each). Training from scratch makes that the wrong test, because the task priors would never be in stage 1. Short from-scratch runs cost less and match the final run.

GPU costs scale from v1: stage 1 ran 150K steps in about 30 h on 32 H100, so 30K steps take about 6 h, or 190 H100-h.

| Phase | Run | What | GPU time | Compared with |
|---|---|---|---|---|
| 0 | D4, D5 | the diagnostics of §3 and §4 on the v1.1 weights | minutes on 1 H100 | — |
| 1 | C (2 seeds) | from scratch, stage 1 only, 30K steps, base prior | 2 × 190 H100-h | each other (seed spread) |
| 1 | TS | C + forecasting share 0.1 | 190 H100-h | C |
| 1 | OD | C + outlier share 0.1 | 190 H100-h | C |
| 1 | IMP, CA | C + imputation or treatment prior, only if D4 or D5 shows a gap | 190 H100-h each | C |
| 2 | ALL | C + every task prior that passed phase 1 | 190 H100-h | C and each single arm |
| 3 | v2 | the ALL prior, v1's curriculum 150K / 10K / 2K | ≈ 1,550 H100-h | v1 |

- **Evaluation of every run:** TabArena-Lite, AMLB, the task suites (fev-bench, ADBench, impute, causal) and the diagnostics D1-D5 (`training/scripts/v2_prior_diag/`).
- **Phase 1 pass rule.** An arm passes if its task metric beats both C seeds by more than their spread, and the guards hold against C. The second C seed gives the spread that #14's single run lacked.
- **Phase 2 pass rule.**
  - Mixing has to keep each task's gain: every task metric of ALL stays within the C seed spread of its single arm.
  - TabImpute found that mixing training patterns lowered its default model, so this is checked before the full run.
  - A task that drops gets a lower share or leaves the mixture.
- **Phase 3.** v2 must meet the success criteria of §1-§4 for the task priors it contains, and the guards against v1, at the same tabsil inference version.
- **Total:** about 0.95-1.33K H100-h for phases 1-2 (5-7 short runs, depending on D4 and D5) and 1.55K for v2.
- **Risk.** Phase 1-2 runs see only 1,024-row tables, and a gain at 30K may not hold at 150K. Phase 3 checks every criterion again on the full run.
