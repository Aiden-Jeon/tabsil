# v2 training prior: forecasting and outlier detection

Status: design. No training job has run for it. Two of the four problem types planned for v2 are covered here: forecasting and outlier detection. Imputation and causal effect estimation are being evaluated zero-shot on main first.

TabSil 0.1.x forecasts (`tabsil.forecast`, #17) and scores outliers (`tabsil.unsupervised`, #19) zero-shot with the v1 weights: it turns each task into the regression and classification it was trained on. The fev-bench and ADBench suites measure that. This document asks which training data the v2 prior needs so that these uses improve. It relies on three diagnostics of the v1 weights (`training/scripts/v2_prior_diag/`) and on a check of the code paths involved.

## Diagnostics (v1.1 weights, tabsil 0.1.1 defaults)

AI Runtime, 1x H100, MLflow run `tabsil-v2-prior-diag` `edee9216d2a44f4c9484ae6a91578fa6` (2026-11-26, with the scripts as committed; the earlier run `ffc589d4…` printed identical tables). The code is main (#17, #19) merged with release/0.1.1. Nothing failed and nothing is imputed.

### D1: forecasts stop at the largest context value

Synthetic daily series: 365 context steps and a 56-step horizon. Each is level + trend + weekly and monthly seasonality + noise (sd 1). Strength is the trend's rise over the horizon in noise sds. 4 seeds.

| Trend | Strength | True max − context max | Forecast max − context max | MASE TabSil | MASE seasonal naive | MASE OLS (index + sin/cos) |
|---|---|---|---|---|---|---|
| linear | 0 | −1.03 | −2.16 | **0.51** | 0.86 | 0.51 |
| linear | 2 | 1.67 | −0.47 | 1.53 | 0.95 | 0.51 |
| linear | 5 | 5.12 | **0.01** | 3.42 | 1.62 | 0.49 |
| linear | 15 | 14.11 | **0.03** | **11.75** | 3.67 | 0.37 |
| exponential | 15 | 13.52 | 0.07 | 4.72 | 4.36 | 7.43 |
| piecewise | 15 | 14.11 | 0.07 | 4.90 | 4.59 | 11.53 |

- Without a trend TabSil equals OLS and beats seasonal naive, so the periodic part is learned.
- With a trend, no forecast goes more than 0.07 above the largest context value. The linear trend's forecast even falls back (the trend capture is −1.31). This is the limitation the TabPFN-TS paper reports for TabPFN (Hoo et al. 2025, arXiv 2501.02945, Section 6.1).

The v1 prior explains it. The graph SCM's regression target is passed through TabICL's `outlier_removing(threshold=4)` and then `standard_scaling` over all rows of the table, train and eval together (`tabicl/prior/_graph_scm.py`, `_reg2cls.py`). Eval rows are a random subset of the same rows. So the model has practically never been asked for a held-out target far outside the train targets' range.

### D2: unpredictable targets are calibrated about as well as predictable ones

NLL on 500 test rows. The excess is the model's NLL minus the true distribution's NLL, averaged over 100, 500 and 2,000 train rows, 1 and 5 features, and 3 seeds.

| Target | Excess over the true density | Excess over a KDE of the train targets |
|---|---|---|
| classification, random labels (0.6/0.3/0.1) | 0.001 | 0.004 |
| y independent of x: normal | 0.087 | 0.077 |
| y independent of x: lognormal | 0.136 | −0.162 |
| y independent of x: Student t3 | 0.108 | −0.143 |
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

Checked on `dev-v2` (8e5007d) and main.

- **Regression loss.** The regression output is 999 quantiles with the pinball loss, and a missing (NaN) target contributes 0 (`tabsil/models/quantile_distribution.py`). Eval rows can therefore be left out of the loss without a code change. The regression borders are only used at inference to turn quantiles into bucket probabilities. They span ±22.6 train-target sds (v1), so targets far outside the context range can be represented.
- **Classification decoder.** It is a retrieval decoder (`tabsil/models/column_row_transformer.py`, `RetrievalDecoder`). A test row's probabilities are the attention-weighted average of the train rows' one-hot labels. **A class that no train row carries gets probability 0.** A one-class setup, with every context row labeled 0 and the model asked for P(outlier), is therefore impossible. Eval rows can be left out of the classification loss with `IGNORE_INDEX` (−100).
- **Scaling at inference.** Inference standardizes x and the regression y with the train rows only (`standardize_cells`; `EnsembleRegressor.fit`, `inference/ensemble.py`). The v1 prior standardizes y over all rows and compresses it beyond 4 sd, so the two disagree whenever eval targets leave the train range.
- **Batches.** A `PriorBatch` has one `n_train`, and the eval rows are the last rows of every dataset. A temporal split only has to put the time steps in order before that cut.

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
- Start with a share of 0.1 of the regression datasets. Chronos found about 10% synthetic data best when mixed with real series, and more synthetic data hurt there.
- Classification batches are unchanged.
- Stage 2 only, as in #14's option A. The quantile head does not depend on the borders, so the stage 1 weights and borders carry over.

**Success criteria.** These are fixed before a run:
- fev-bench task-mean SQL Elo above 1526, the v1 zero-shot result, and the D1 linear-15 MASE below seasonal naive.
- TabArena-Lite overall and regression Elo not lower than the baseline run beyond its 95% CI.
- AMLB regression RMSE ratio within 0.005.

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
- **Mixture.** Start with a share of 0.1 of the classification datasets. Every other classification dataset is unchanged, and `CoverClasses` still holds (both classes appear in the context).
- **Success criteria.**
  - ADBench (12 tables × 3 seeds, #19 protocol) mean AUROC above IsolationForest's 0.722, and contaminated AUROC within 0.05 of TabSil's clean-context 0.820.
  - The D3 cluster at 2% at or above 0.9.
  - TabArena-Lite classification Elo not lower than the baseline beyond its 95% CI.

**Fallback.** A separate outlier head as in FoMo-0D and OutFormer, trained with a binary loss on contaminated queries. This changes the architecture. Use it only if the reference-row run fails its criteria.

**Not recommended: contamination with an inlier-only loss on the per-column conditionals.** The loss would ignore the outlier eval rows, but D3 shows that the context's outlier group is learned as density. The model would stay unconstrained exactly where it fails.

## Open items before implementation

1. **Sync.** `tabsil.forecast` and `tabsil.unsupervised` are not on dev-v2. Merge main (after release/0.1.1 is merged back into main) into dev-v2 with a PR first.
2. **Overlap with #14.** #14 changes the same files (`priors/graph_scm.py`, `priors/transforms/`). Build on #14 once it is merged. The baseline prior of the runs below is then #14's `graph_scm_hicard_missing` if it is adopted, and `graph_scm` otherwise.
3. **Forecast row count.** The SCM covariate path needs a time-indexed root. Either add a generator that wraps the graph SCM with t as an extra root, or use the series generator alone and draw the covariates separately. Decide in the implementation PR.
4. **Reference rows at inference.** The reference rows must come from shared code in `src/tabsil/` that the prior imports, so the training and inference recipes cannot drift apart.

## Experiment plan (each run needs approval before it starts)

| Run | What | GPU time | Compared with |
|---|---|---|---|
| TS | stage 2, 10K steps from `stage1-150k`, baseline prior + forecasting share 0.1 | ≈ 307 H100-h (as #14 A) | the baseline stage 2 run (#14 A, `277916ec…`, or `stage2-10k`) |
| OD | the same with the outlier share 0.1 instead | ≈ 307 H100-h | the same |

One variable per run, all other settings as #14 A. Evaluate each on TabArena-Lite, AMLB, fev-bench, ADBench and D1/D3 (`training/scripts/v2_prior_diag/`).

The two are only combined after each one passes on its own. TabImpute (2510.02625) found that mixing training patterns lowered its default model. That is a reason not to stack priors untested.
