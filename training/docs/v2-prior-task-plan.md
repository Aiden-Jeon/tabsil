# v2 plan: priors and tasks

Status: **plan, approved by the user on 2026-11-28 after the review of section 11. Tiers 1 and 2 are measured and reading points R1 and R2 are read ([v2-prior-task-results.md](v2-prior-task-results.md)): forecasting stops at R2, outlier detection is at its bar only with a score added after the pilot, and tier 3 is not started. What follows is the user's decision.** Nothing was trained or evaluated for writing this document. It follows the stop of the v2 experiment plan ([v2-experiment-plan.md](v2-experiment-plan.md) §7, decisions 12 and 13) and is built to the twelve rules of [v2-retrospective.md](v2-retrospective.md) ("Rules for the next plan"; section 9 lists where each is kept). The derivations it rests on are in [v2-task-formulations.md](v2-task-formulations.md). It was read by an agent that did not write it (section 11).

The request it answers, as the user made it on 2026-11-28 after reading the retrospective: plan again in line with the retrospective, with more diverse priors and more diverse tasks, and with the tasks derived in formulas first.

> **Correction, 2026-11-28, after tiers 1 and 2 were run.** The user's goal is the accuracy on TabArena-Lite, and more diverse priors and tasks were asked for as the means to raise it. This plan took the task metrics themselves as the goal. That was a misreading on the planner's side: the goal question put to the user offered only readings in which the tasks' own metrics were the target, so the answer recorded below as decision (a) could not say otherwise. The measurements made under this plan stand as measurements. Its bars and reading points were set against the wrong target, and its TabArena guard was never read, because tier 3 was not reached. What task-shaped tables do to TabArena-Lite is read in the results document of the runs (`v2-prior-task-results.md`).

How to read:

- Every number has its source and its kind. **Measured**: from a run in the records. **Derived**: from the formulations document. **Estimated**: computed from measured numbers by a stated step. **Reported**: another group's number for another model, read in its paper. **Unknown**: no basis.
- G is the paired error ratio of [v2-experiment-plan.md](v2-experiment-plan.md) §3 (below 1 is better). On TabArena-Lite, 1% of error is about 40 Elo at TabSil's place in the pool (its §1.1).
- GPU costs are H100-hours.

## Summary

**The goal this plan was built to (the answer recorded in the planning dialog of 2026-11-28; see the correction above):** the task metrics, in one model. Forecasting has to reach a fev-bench SQL Elo of 1707 (it is 1535), and outlier detection an ADBench mean AUROC of 0.77 (it is 0.663). TabArena only has to hold: within 0.6% of its control. Imputation and treatment effects are in scope and are ahead of their baselines; they get their first measurements against the cases the derivations name. The cap is 500 H100-h.

**The outlook.** No shipped gain is expected from this plan with evidence. Whether the bars can be reached is unknown: they are at the size of the largest gains that others report for models trained on one task's tables, and those reports are not controlled tests of a prior alone. The plan is ordered so that this is found out cheaply: about 20 H100-h without training, then about 22 for a pilot and one short probe, and the one expensive run, about 232, only if the probe reaches a bar. That run exists because the goal is one model: with a checkpoint per task the plan would end with the probe and need no guard.

| Tier | What is done | What it adds, in the user's unit | Kind of evidence | H100-h | Runs if |
|---|---|---|---|---|---|
| 0 | Nothing. What is in hand | TabArena-Lite +20 to +40 Elo over v1.1, as the weights of the v2 candidate. Forecasting 1535 and outlier detection 0.663, as v1.1 has them | measured | 0 | — |
| 1 | No training: inference changes and diagnostics, on existing weights | forecasting and outlier detection: unknown on the benchmarks (the bars need +172 Elo and +0.107). On D1's linear trend the forecast error would fall from 11.75 to about 0.4 if the model forecasts the detrended series as well as a trend-free one (an estimate from a least-squares fit, not a run of the model). Imputation and effects: the first measurement of whether a gap exists; no gain | measured for part of D1; derived; unknown | about 20 | the user approves the plan |
| 2 | One probe: 2,000 steps on forecast- and outlier-shaped tables alone, after a one-GPU pilot. Its weights are not shipped | no shipped gain. It tells whether training reaches a bar at all: others report up to +180 SQL Elo and +0.13 AUROC | reported for other models; unknown for TabSil | about 22 | after tier 1 a bar is not reached and the distance left is within the reported gain. With today's numbers that holds for both tasks whatever tier 1 gives |
| 3 | One branch: 10,000 steps of stage 2 with 10% task tables, read against the TabArena guard | the bars in the one model with TabArena held: at least +172 SQL Elo or +0.107 AUROC if the branch keeps what the probe showed | unknown | about 232 | the probe reaches the bar for at least one task |

- **If every tier runs:** about 275 H100-h. About 225 of the cap are not planned, and using them is a new decision.
- **If no bar is reached:** the plan ends after about 42 H100-h, with the inference changes that helped and no new weights. The v2 candidate stays what it is.
- **No run aims at TabArena accuracy.** No prior candidate of section 3 has evidence for more than the +20 to +40 Elo that the user judged not worth further runs.
- **For the user to decide** (section 12): whether to run the plan, the GPU setup of tier 3, the series generator, what follows the imputation and effect diagnostics, and five smaller points.

## 1. Premise check

Rule 4 of the retrospective: a premise of the request that the evidence speaks against goes back to the user before the plan is built on it. The request has three premises.

### Premise 1: a more diverse prior makes the model better

**The request assumes** that adding variety to the training prior raises accuracy.

**The evidence says:**

- **Measured here, one prior change has a gain, and it is the size the user declined.** Stage 2 on `graph_scm_hicard_missing` against the v1 prior: pooled G 0.992 (0.980 to 1.000) over two seeds, TabArena-Lite Elo 1768 → 1788 and 1808. Without polish_companies_bankruptcy it is 0.997, and on the development suite 0.996 (v2-experiment-results, phase 1A). The other parts taken from Kumo-Tabular add nothing to it: 1.003 (0.999 to 1.008). On 2026-11-28 the user judged +20 to +40 Elo not worth further runs.
- **A gain of a prior part sits in the datasets that have the property it adds.** G is a geometric mean over datasets. One dataset of 51 improved by 40% is 1%: that is the hicard result (polish 0.60, kddcup 0.96, the rest level). A part that should give 2.5%, about 100 Elo, has to improve 13 datasets by 10% each, or 3 by 35%.
- **Controlled results in the literature are of that size or come from adding a function family.** Mitra reports +63 Elo, on its own pool, for adding one tree-ensemble prior to an SCM prior, and nothing more for four further priors (1062 for six priors against 1063 for two; arXiv 2510.21204, Table 6; seeds not stated). Mitra-v2 states that varying its mixture weights "did not produce a large change in downstream accuracy" (arXiv 2609.04540, §2.2). Kumo-Tabular's ablation: "The prior change marginally improves performance", at 100K steps with three seeds (arXiv 2611.23001, §7). TabICLv2 developed its prior "mostly without experimental feedback". An earlier note of this project held that no controlled ablation of a prior exists. That is wrong: Mitra's is one.
- **The distance between a prior and real tables can be measured without training, and it does not predict accuracy.** The TabICL prior, the family of TabSil's, is told apart from real tables by a classifier with an AUC of 0.9997 and covers 9 to 16% of them, yet a dataset's distance to the prior has a correlation below 0.14 with the model's accuracy on it (Davies et al. 2026, arXiv 2605.06343, Table 2).

**Options.**

| Option | What it buys | Evidence |
|---|---|---|
| Prior variety aimed at the tasks: tables shaped like the tasks, with TabArena as a guard | the task metrics of premise 2 | derived (formulations); gains reported for models built for one task |
| Prior variety aimed at TabArena | +0 to +40 Elo for a part that adds a property some benchmark datasets have; unknown otherwise | measured for one part; no candidate of section 3 has evidence above +40 |
| Neither | 0 | — |

### Premise 2: more tasks need training

**The request assumes** that the model has to be trained for the tasks.

**The evidence says** that the gap of each task is of a different kind (formulations, summary):

| Task | Gap of the v1.1 weights today (measured) | What the derivation says is needed | Kind |
|---|---|---|---|
| Forecasting | fev-bench SQL Elo 1535: 49 below TabPFN-TS on TabPFN v2, 229 below TabPFN-TS-3, 307 below Chronos-2. D1, linear trend 15: MASE 11.75 with the default recipe, 9.03 with the `plain` one, 3.67 for seasonal naive | the default recipe caps the median at the context maximum: inference. The model's forecast falls back toward the context mean, because training never asks for a query far beyond the context: the trend fitted outside the model (inference, not measured), or a prior with temporal splits | inference, then prior |
| Outlier detection | ADBench mean AUROC 0.663: 0.059 below IsolationForest, 0.157 below the same model with a clean context. D3, a dense group at 2%: 0.425 | the score is the density of the contaminated context, and a dense group has a high density. A better density does not repair it. Small random contexts (inference, not measured) or a prior with labeled reference rows and hidden outliers | inference, then prior |
| Imputation | ahead of every baseline under MCAR in the mean (NRMSE 0.546 against 0.616 for MissForest at 10%). Other mechanisms not measured (D4) | MCAR is the v1 prior's own split. A mask that depends on other columns keeps the conditional. A mask of another column that depends on the cell tilts the context, and `graph_scm_hicard_missing` has such tables. Self-masking is not identified: a prior teaches its own mechanism's tilt | none; the v2 candidate's weights for the third case; prior for self-masking, with a limit |
| Treatment effects | ahead of the baselines on IHDP (root PEHE 0.55 against 2.08 for the EconML T-learner; CausalPFN reports 0.58). Low overlap not measured (D5) | the point estimate is already the posterior mean. An interval for the effect needs a third task value, and hidden confounding a prior with a marked treatment column | none; structure for intervals |

So two tasks have a measured gap, and for both a part of it may close without training. Two are ahead of their baselines and have nothing measured against them.

The derivation of the reference case found one more thing, on classification itself: the v1 prior puts one row of every class into the context first, so its split depends on the labels, and the classifier is trained toward class odds that are lower than the posterior's by about $n_c/(n_c + 1)$ for a class with $n_c$ context rows (formulations §2; checked on the prior's own function, not measured on the model). It can matter for the log loss of tables with rare classes and cannot change a binary AUC.

**Options.** Inference changes first and training only for the gap that remains; training on task priors directly; or no task work.

### Premise 3: one set of weights for every task

**The request assumes** (as [v2-task-priors.md](v2-task-priors.md) did) that the tasks are trained into the one model.

**The evidence says** that no source shows this with numbers. The models that report gains from training on task-shaped tables are separate models or separate checkpoints: TabPFN-TS-3 is a TabPFN-3 checkpoint "fine-tuned on synthetic time-series data" (arXiv 2605.13986, §3.3); TACTIC, TabImpute and CausalPFN are trained for one task each; Kumo-Tabular's paper covers classification and regression only, and Kumo's forecasting and anomaly models are separate time-series models (NVIDIA SDM documentation, read by a subagent). The one report of a tabular model trained on a shifted task says it lost in distribution (Drift-Resilient TabPFN, "lower predictive performance on ID tasks"). Whether a task prior at a small share costs TabArena accuracy in TabSil is unknown.

**Options.** One model with a TabArena guard, which needs a branch that can resolve the guard (0.6%, about 180 to 305 H100-h per branch). Or a checkpoint per task, fine-tuned from the released weights, which needs no guard and can be read on the task's own metric after a short run.

### Not in the request: the lever with support for TabArena

Kumo-Tabular-M has 4.6 times the training compute of its S model and is 111 Elo above it on the public leaderboard (1915 against 1804); L has 14.7 times and is 175 above. Kumo-S is level with v1.1 at v1's compute (G 0.996; v2-experiment-plan §1.3). Size and compute change together along that line, so this is a reported association for another model. For TabSil, 100K more stage-1 steps gave 1.7% on small tables with p 0.11 and nothing on the full readout (§1.2). 4.6 times v1's compute is about 5,600 H100-h more. It is listed so that it is not left out again; nothing in this plan proposes it.

### Decisions asked before the plan

The plan's tiers, runs and thresholds depend on these, and the user sets them (rules 1, 3 and 10). They were put to the user on 2026-11-28 in two dialogs, each option with its expected gain and cost and none marked as recommended. The user's answers of 2026-11-28:

| # | Decision | The user's answer |
|---|---|---|
| a | The goal | **The task metrics, in one model.** TabArena is a guard: a model fails if it is worse than its control by more than 0.6%, about 25 Elo, which is what a branch resolves. The other options were a checkpoint per task, task metrics and TabArena Elo together, and no plan |
| b | The tasks | **All four:** forecasting, outlier detection, imputation, treatment effects. The last two start with the diagnostics D4 and D5. No new task was added |
| c | The stop thresholds | **Forecasting: fev-bench SQL Elo 1707** (the place of TimesFM-2.5 in this repository's pool, +172 from today). **Outlier detection: ADBench mean AUROC 0.77** (about what IsolationForest has with a clean context, +0.107 from today). The other options were the nearest named model (1584 and 0.722), the synthetic diagnostics only, and no threshold |
| c′ | How the thresholds apply | **After the tier without training, a task's probe runs only if the distance left to its bar is within the largest gain reported for training on that task's tables:** 180 SQL Elo (TabPFN-TS to TabPFN-TS-3) and 0.13 AUROC (TACTIC over the density score). **After the probe, the branch runs only for a task whose probe reaches the bar.** The user chose this over stopping every task that misses its bar without training, and over always running the probe, knowing that today's distances (172 and 0.107) pass the first rule even with no gain from tier 1 |
| d | The budget cap | **500 H100-h** (the options were 0, 50, 150 and 500) |
| e | A change of structure | Not asked: none of the four tasks needs one for what was chosen (formulations §9). The structure stays v1's |

Not set by the user, and therefore not filled in here: a bar for imputation and for treatment effects. This plan measures them and trains nothing for them (section 12).

## 2. What exists already

Rule 2: the first tier trains nothing, so what is in hand comes first. The artifact lists were read from MLflow on 2026-11-28.

### Weights

| Weights | What they are | Where |
|---|---|---|
| v1.1 | the released model: stage 1 150K, stage 2 10K, stage 3 2K steps on `graph_scm` | Hub tag `v1.1` |
| `stage1-150k`, `stage2-10k` | v1's earlier stages | Hub tags |
| #14's run, N10 | the v2 candidate: v1's stage 1 with 10K steps of stage 2 on `graph_scm_hicard_missing`, seeds 0 and 1 | MLflow `277916ec05a3457aa6d07f2bc08a60c4`, `520d88d73cdc4e5ea90cfd27425be00d` |
| R10 | the same on `graph_scm_realism`, seed 0 | MLflow `b9396104c0664d4e859318f0518a6052` |
| The end of the 250K-step stage 1 | stage 1 only, on `graph_scm` | MLflow `32b8195f4f794208a3425878e5c3fcf3` (the earlier experiment) |
| Phase 1B, C seeds 0 to 2 | stage 1 from scratch, 30K steps, v1's recipe | MLflow `2c0aaf05b27846fa944051ea0be59d53`, `7f1e2cea52724763b0a1e5619fd03464`, `1089b4f1ab6a4a1bbee2e19f409aa00c` |
| Phase 1B, YCTX seeds 0 and 1 | the same with `target_train_rows` | MLflow `74f6ab8c8f7a4ae89135be011810c109`, `bc2c2e476e3142b1a01eade86ef490ad` |
| Phase 1B, OPTK seeds 0 and 1 | the same with Kumo's optimizer split | MLflow `e3d013f9bbd145b7a9afe64da95010a2`, `64aae055e7f14ddaa052729d395a58cd` |

Stopped and not used: L20's checkpoint of step 5,000 (`893293ed7d8d419780be2d178b8cc713`) and PS1's of step 10,000 (`8d64487b2d2a42b5805fe640f62d7e82`). The user stopped both runs. This plan does not resume them.

### Code

- The four task wrappers and their suites: `TabSilForecaster` (`--suite fev`), `TabSilUnsupervised` (`--suite adbench`, `--suite impute`), `TabSilCATE` (`--suite causal`).
- The diagnostics D1 to D3 (`training/scripts/v2_prior_diag/`). D4 and D5 are not written.
- The paired report on TabArena with 3 splits and on the development suite, and the seed report (`paired_report.py`, `seed_arm_report.py`).
- The prior options of section 3, the context-row target scale (`target_train_rows`), the fixed reference validation set, and the timing probes (`training/scripts/v2_eff/`).
- Not written: any task prior (a temporal split or series generator, reference rows, a mechanism-driven split, randomized-treatment queries), the wrapper that would go with reference rows, a forecasting recipe without the clip on main, the trend fitted outside the model, and scoring with small random contexts.

### Measurements

| What | Value | Source |
|---|---|---|
| TabArena-Lite Elo | v1.1 1768; N10 1788; #14's run 1808 | results 0.1, phase 1A |
| Noise between two stage-2 branches of one checkpoint | 0.3% pooled; a branch resolves about 0.6% | results, "ν" |
| Noise between two stage-1 runs of 30K steps | 0.1 to 0.5% pooled; an arm of two seeds against three control seeds has a CI of about ±0.5% | results, "Phase 1B" |
| Stage 2, 10K steps, v1's size | 305 H100-h on 32 H100 at micro batch 1 (N10); about 223 on 8 H100 at micro batch 1 and about 178 at micro batch 2 (timing runs `bf797ac1…`, `3bcffc89…`); 162 by L20's 5,200 steps on 8 H100 at micro batch 2 | results, "GPU cost", "L20" |
| Stage 1, v1's size, 8 H100 | 3.74 H100-h per 1K steps (`7266d800…`); a 30K-step run 117 to 123 | results, "GPU cost", "Phase 1B" |
| A smoke run of 300 steps | about 0.5 H100-h on one GPU, about 3 on 8 H100 | results, "Why R10 was slow", "Phase 1B" |
| Evaluation of one set of weights | fev 0.5 H100-h; ADBench 1.7; causal 0.3; imputation 0.2 for TabSil alone, 9 with the baselines; D1 to D3 0.1; TabArena-Lite, 3 splits and the development suite about 2.5 together | docs/task-results; v2_prior_diag README; results, phase 1A |
| The task suites at v1.1 | section 1, premise 2 | docs/task-results |

## 3. Prior candidates

For each candidate: the property of real tables it adds, the evidence for a gain (whether the comparison was controlled, whether compute was matched, the size), the expected gain for TabSil with its kind, and the cost with the risk for the generation speed. Expected gains are in TabArena-Lite Elo unless a task metric is named.

**No candidate below has evidence for more than +40 Elo on TabArena-Lite.** The candidates of (d) are aimed at task metrics.

### (a) In the code

| Candidate | Adds | Evidence | Expected gain | Cost and speed |
|---|---|---|---|---|
| `graph_scm` (the v1 prior) | the base: the TabICLv2 graph SCM, 2 to 10 classes with half the tables binary, unpredictable tables redrawn, random missing cells in 30% of the tables | — | the control | measured: 3.30 s per step in stage 2 on 32 H100 |
| `graph_scm_hicard_missing` | categorical columns with 21 or more categories in 10% of the tables; missing values that depend on values and come in column groups | measured, two seeds against the v1 prior at equal steps: pooled G 0.992 (0.980 to 1.000); 0.997 without one dataset | **+20 to +40, measured, and in hand as #14's run and N10** | none: the weights exist |
| `graph_scm_realism` = the above plus warped columns, up to 200 categories, correlated column blocks, relabeled targets, coarse values | skewed and bounded marginals, collinear columns, targets that depend on few columns, low-resolution values | measured, one seed against the two hicard seeds: 1.003 (0.999 to 1.008) at 10K stage-2 steps | 0, measured; a longer stage 2 is not measured | 450 H100-h for 10K steps where 305 were planned: rare tables take 30 to 100 s to draw. `TABSIL_PREFETCH_FACTOR=16` removes the stalls in a simulation and in the stage-1 smoke runs |
| `graph_scm_base_realism` (the parts Kumo has in every stage, from step 0) | the same properties from the start of stage 1, and too-easy tables redrawn | not measured: the two runs were stopped at steps 10,500 and 5,000 by the user's decision of 2026-11-27. The plan expected it to be neutral at 30K steps | unknown | 9% slower per step than the control with the prefetch setting |
| `target_train_rows` | the target scaled with the context rows, as at inference | measured, two seeds against three controls at 30K stage-1 steps: 0.995 (0.990 to 1.000), inside the seed difference | 0 to +20, measured and not resolved. A condition of every prior of (d) whose split is not random (formulations §3, §5) | none on speed |
| `noise_prob` (no registered prior turns it on) | noisy copies of about 10% of the columns | none. Four earlier 20K-step runs with one such option each scored 18 to 60 Elo below their control, with one seed and a readout that is not trusted (v2-experiment-plan §1.2) | unknown | under 1% of the generation time (results, "Why R10 was slow") |
| The class-covering split of every prior above (`CoverClasses`) | — (it keeps every query's class in the context, which the decoder needs) | derived: it lowers the odds of a class with few context rows (formulations §2). Two ways to undo it: at inference, by the factor $(n_c + 1)/n_c$, or in the prior, by a random split with the queries of absent classes left out of the loss, which needs training | unknown; only in the log loss of tables with rare classes | the inference form costs an evaluation; the prior form a branch |

### (b) In Kumo-Tabular and not ported

Read in `NVIDIA/structured-data-models` at commit `924e992` (`train/kumo/tabular/prior.py`, `loader.py`, `README.md`).

| Part | Stage in Kumo | Adds | TabSil today |
|---|---|---|---|
| Count and distributional numeric columns (10% of numeric columns built from a signal, a noise scale and a subgroup probability; 30% of those rounded) | every stage | counts, heteroscedastic columns | not ported; `corrected_num_converters` ports the warp only |
| Argmax categories (probability 0.15 per categorical converter) | stages 2 and 3 | classes and categories that are a hard function of the causes | not ported |
| Heavy-tailed regression targets (a power or exponential warp of 20% of the targets) | stage 3 | skewed and heavy-tailed targets | not ported |
| Small tables replayed (5% of the batches at 200 to 1,000 rows, 25% at 1,000 to 10,000) | stage 3 | keeps small tables in late training | not ported |
| A wider column ladder for long tables (M and L: 100 features up to 32K and 25K rows, 53 and 40 at 60K) | stage 3 | wide and long tables together | TabICLv2's ladder: 80 at 20K rows down to 20 at 60K |
| Hyperparameters shared within a table (the `Correlated` draws) | every stage | tables whose mechanisms are alike | not ported |
| Classes uniform in 2 to 10; one model per task; FP16 for the regressor's stage 1 | every stage | — | half the tables binary; one model; BF16, with no temperature growth found (results 0.4) |

- **Evidence.** Kumo's own ablation of its prior changes, quoted in section 1: marginal at 100K steps. "During development, we observed that the prior changes matter more with longer training" is its only statement on longer runs, without a number. For the stage-2 parts that are ported, R10 measured 1.003.
- **Expected gain.** Unknown for the parts not ported; 0 measured for those that are. The three stage-3 parts have no place while stage 3 itself shows 1.000 (results 0.6).
- **Cost.** New generator code for the first two rows, inside the SCM draw, which is 82 to 86% of the generation time. A smoke run first.

### (c) Other generator families

| Family | Adds | Evidence | Expected gain for TabSil | Cost and speed |
|---|---|---|---|---|
| Tables labeled by fitted tree ensembles (Mitra's ET, GB, RF, DT priors) | axis-aligned, piecewise-constant targets | controlled within one study, seeds not stated: SCM alone 1000 Elo, with the ET prior 1063, with all six 1062 (Mitra Table 6, TabRepo classification, own pool) | unknown. TabSil's SCM already has tree mechanisms among its functions (the `tabicl` prior's default function set; Kumo's source has `RandomTree`), where Mitra's SCM prior is TabPFN's kind (its v2 report calls it "Base TabPFN/SCM") | a new generator that fits an ensemble per table; generation time unknown; smoke run first |
| A hybrid SCM and "realism" modules (Mitra-v2; arXiv 2605.18971) | mixed mechanisms, irregular marginals | the second is controlled at a small scale (a small model and a fixed number of synthetic tables; its numbers were read by a subagent and are not verified) | unknown | a new generator |
| Continued pretraining on real tables (Real-TabPFN, arXiv 2507.03971) | real marginals and dependencies | not a prior ablation: TabPFN v2 after 20K steps on curated OpenML and Kaggle tables has a normalized AUC of 0.976 against 0.954 on 29 small classification datasets; web-scraped tables lowered it | unknown. It ends "trained only on synthetic tables", and it needs a corpus that shares no table with TabArena, the development suite, AMLB, BeyondArena and the task suites | the corpus and its leakage check are the cost; the training is short |
| An adversarially trained generator (APT, arXiv 2502.04573) | tables the current model finds hard | reported: 10% lower MSE on 35 regression datasets against its TabPFN baseline (read by a subagent, not verified) | unknown | a second network trained with the model |
| A prior with drift over time (Drift-Resilient TabPFN; the dynamic SCM of TabPFN-3) | mechanisms that move with a time index | reported: better out of distribution and "lower predictive performance on ID tasks"; TabPFN-3 reports no ablation | unknown on TabArena, with a reported loss in distribution; see (d) for the task side | a second-order generator |

### (d) Task-shaped priors

These change the split, the context or the scale, as the derivations require. The gain is in the task's metric. On TabArena the expected effect is zero or negative, which is why a guard or a separate checkpoint is needed (section 1, premise 3).

| Candidate | What changes | Derivation | Evidence for a gain | Expected gain | Cost and speed |
|---|---|---|---|---|---|
| A sorted split: the query rows are those with the largest values of one feature, on ordinary prior tables | split and scale | formulations §3: a random split never puts a query far beyond the context; the SCM already has linear and quadratic mechanisms | none: not found in the literature in this form | unknown, on D1 and fev-bench | the smallest change: no new generator and no effect on generation time |
| A series generator with trend, seasonality and noise, split in time | generator, split, scale | formulations §3: the trained forecast is the prior's, so the prior needs functions of the index that continue | reported: TabPFN-TS-3, fine-tuned on synthetic series, has an SQL skill of 43.1% on fev-bench against 39.6% for TabPFN-TS on TabPFN v2, with the base model changed as well. In this repository's pool the two are at 1764 and 1584 | unknown; the reported pair spans +180 Elo of SQL in this repository's pool | a new generator: smoke run first |
| Labeled reference rows with outliers hidden among the data rows | context and labels | formulations §4: the classifier converges to the posterior outlier probability | reported for a model built for the task: TACTIC, trained on contaminated contexts, has a mean AUROC of 80.6 on 38 ADBench tables with a contaminated context against 67.6 for the density score of the TabPFN-2.5 extension and 77.7 for IsolationForest | unknown: +0.03 to +0.13 AUROC is what that model has over IsolationForest and over the density score | a new generator part and a wrapper; whether the retrieval decoder learns it is not known |
| Query rows chosen by a missingness mechanism | split and scale | formulations §5: a mask that depends on other columns keeps the conditional; self-masking changes it and is not identified | reported: TabImpute's specialists, 0.4% on its column-MAR pattern; 21% on self-masking and a loss of 39% on MCAR | about 0 for the first; for self-masking positive on a benchmark masked by the prior's mechanism and negative elsewhere | a split by a score; no generation cost |
| Query rows with a randomized treatment | split | formulations §6: against training's queries a counterfactual query has the density ratio $(1 - e)/(e P(t = 0))$, unbounded at low overlap; with randomized queries it is at most $2/P(t = 0)$ | none | unknown; D5 is not run | a marked column in the generator's graph |
| Masked cells with a reconstruction output | structure | formulations §7: all conditionals in one pass; no change for classification and regression at the optimum | no controlled evidence | unknown | a new output: a change of structure |

Tiers 2 and 3 use the first and the third row of this table. The series generator of the second row is built only by a later decision of the user (section 12). The last three rows are not in the plan.

### Measuring a prior against real tables without training

- **It can be done.** Summary statistics per table, a classifier that tells synthetic tables from real ones, and the share of real tables with a synthetic neighbor (arXiv 2605.06343). Drawing a few thousand prior tables takes CPU minutes.
- **It did not predict accuracy** in the one study that tested it (section 1). So it does not rank the candidates above.
- **What did work in this repository** is narrower: #14 named, before its run, the four datasets with the property its prior adds. With two seeds the gain is on two of them (polish 0.60, kddcup 0.96), and the other 47 TabArena datasets are level (0.999 to 1.000). A candidate of (a) to (c) is worth a run only if benchmark datasets can be named that have its property and on which TabSil is behind. The per-dataset gaps to named models are in the existing evaluation runs and cost no GPU to read.

## 4. Tiers and items

Every item has its expected effect, the resolution of its design and the decision each outcome leads to (rule 5). "R1" to "R3" are the reading points of section 5. No item adopts anything on "not worse", and a result without an effect starts no run.

### Tier 1: no training (about 20 H100-h)

All items are evaluations of weights that exist (section 2). The code they need is in section 8.

| # | What | Expected effect | Resolution | Decision by outcome | H100-h |
|---|---|---|---|---|---|
| 1.1 | The v2 candidate's two seeds (#14's run and N10) on the four task suites and on D1 to D3 | no change against v1.1 on forecasting and outlier detection: their prior differs from v1's in missing values and categories only (derived; unknown) | the difference between the two seeds is the measurement: the noise of a task metric between two trainings | gives the control values and the seed noise that R2 and R3 are read against. Changes no decision by itself | 5.6 |
| 1.2 | Forecasting recipe on v1.1, four arms: (b) no target transform, (c) the `standard` feature kind only, (d) both, (e) the best of them with a straight-line trend fitted on the context outside the model | D1, linear trend 15: the cap on the median is gone in (b) and (d) (derived); MASE between 9.03 and 11.75 without the trend fit and about 0.4 with it (estimate). fev-bench: unknown; the bar needs +172 Elo | the suite and the weights are fixed, so the difference between two arms is exact for the suite. Over tasks the suite's Elo has a CI of +220/−129 | R1, forecasting | 3 |
| 1.3 | Outlier score on v1.1, three arms: (b) the mean over many small random contexts, of 64 and of 256 rows, (c) reference rows as a second class, which is also the control of a later prior, (d) on D3 only: the highest-scoring context rows dropped | (b) D3's group at 2% above today's 0.425 if the hypothesis holds; ADBench unknown; the bar needs +0.107. (c) fails on D3's group as the density score does (derived). (d) no change on D3's group (derived) | as 1.2; ADBench has three seeds per table | R1, outlier detection. If (c) or (d) contradicts its prediction, formulations §4 is revised before a probe is designed on it | 6.5 |
| 1.4 | D4: imputation error of v1.1, #14's run and N10 under four masks of one rate: completely at random; by another column; another column's mask by the cell to impute; self-masking. On prior tables and on the suite's tables, against MissForest and the mean | the second like the first; under the third v1.1 biased and the candidate's seeds less; under the fourth the closed form of formulations §5 (all derived) | three seeds of masks, as the suite has | goes to the user with the measured gaps (section 12). This plan trains nothing for imputation: the item is a measurement, not a step toward a bar | 3 (estimate) |
| 1.5 | D5: root PEHE of the S-learner on prior tables as the propensity's clip shrinks from 0.5 to 0.02, against the true effect and a T-learner | an error that grows as overlap shrinks; whether faster than the T-learner's is unknown | realizations, as the causal suite has | goes to the user (section 12). This plan trains nothing for effects: a measurement, not a step toward a bar | 1.5 |
| 1.6 | D2 with one to three context rows of the rarest class | that class's probability near $n_c / n$, below the posterior's (derived for labels without features) | the true probabilities are known | confirms or refutes formulations §2. The correction's effect on TabArena is not measured in this plan (section 12) | 0.1 |
| 1.7 | D1 on phase 1B's weights: two control seeds and the two seeds with the context-row target scale | no difference in trend capture: the scale alone puts no query beyond the context (derived) | two seeds per arm | if the scaled seeds' trend capture is above the controls' by more than the spread between seeds, the forecast tables' design is revisited before the probe | 0.4 |

### Tier 2: one probe (about 22 H100-h)

The probe is a short fine-tuning of the v2 candidate on task-shaped tables alone. Its weights are not shipped. It is the best case for the branch of tier 3: with every table of a task type task-shaped it sees 256,000 tables per task in 2,000 steps, twice what the branch's 10% share gives in 10,000 steps. If the probe does not reach a bar, the branch would not.

The forecast-shaped tables are the smallest form first: ordinary prior tables whose rows are sorted by one feature before the cut, with the context-row target scale (section 3, (d)). They need no new generator and cannot slow the generation. The outlier-shaped tables are data rows with hidden outliers and labeled reference rows; they have no smaller form.

| # | What | Expected effect | Resolution | Decision by outcome | H100-h |
|---|---|---|---|---|---|
| 2.1 | Pilot on one GPU: 2,000 steps at batch 32 from #14's weights, the regression steps on sorted-split tables and the classification steps on outlier-shaped tables. It is also the smoke run of the new table code (rule 6) | waiting for tables below 2% of the time: expected for the sorted split, unknown for the outlier tables. D1's trend capture rising from today's −0.71 and D3's group rising from 0.425: unknown | D1 has 4 seeds; D3 has 3 seeds, about 0.15 apart | table code that stalls is fixed before 2.2, as R10's was. A task whose diagnostic moves by more than its seeds' spread is in the probe. If D1 does not move, the series generator goes to the user as a question, with its code and a second pilot of 3 H100-h (section 12). If D3 does not move, outlier detection goes back to the user with dropping it as an option (rule 9) | 3 |
| 2.2 | Probe on one 8xH100 node: 2,000 steps at batch 256 with stage 2's learning rate, tables of 400 to 2,048 rows, from #14's weights, with the tables of 2.1. A task that R1 has finished or stopped, or that 2.1 has sent back, keeps the ordinary tables | reaching the bar needs +172 SQL Elo and +0.107 AUROC from today. Others' largest gains from such training are +180 and +0.13 (reported); for TabSil unknown | fev-bench and ADBench as in tier 1, with the seed noise of 1.1 | R2 | 16 (12 to 24; estimate) |
| 2.3 | The probe's weights on D1 and D3 first, then on fev-bench and ADBench with the tier-1 wrapper that did best | — | — | — | 2.5 |

### Tier 3: one branch (about 232 H100-h)

| # | What | Expected effect | Resolution | Decision by outcome | H100-h |
|---|---|---|---|---|---|
| 3.0 | Timing job: 110 steps of 3.1's setup with its prior | a step of at most 9.69 s, the time of this setup with the hicard prior | — | a projected cost more than 30% above 223 sends 3.1 back to the user (rule 9) | 2.5 |
| 3.1 | Branch: stage 2 from `stage1-150k`, 10,000 steps on `graph_scm_hicard_missing` with 10% of the regression tables forecast-shaped and 10% of the classification tables outlier-shaped, in the probe's form (only the tasks that passed R2; the context-row target scale on the forecast-shaped tables only), one 8xH100 node at micro batch 1 with `TABSIL_PREFETCH_FACTOR=16`, a new seed | the task metrics of the probe with half its task tables: unknown. TabArena: no change or a small loss; Drift-Resilient TabPFN reports 0.4 to 0.8 points of accuracy lost in distribution for a prior of its kind | TabArena: 0.6% pooled against the mean of #14's run and N10 (ν 0.003). Task metrics: the seed noise of 1.1 | R3 | 223 |
| 3.2 | Evaluation of the branch: TabArena with 3 splits and the development suite for the guard, the four task suites, D1 to D3 | — | — | — | 6 |

The branch's control is the two existing runs on the same prior without task tables. They ran on 32 H100 at micro batch 1. The branch has the same micro batch, so a step has the same number of table sizes, and its GPU count changes only which tables it draws, as a seed does (results, "What a larger micro batch changes"). That its noise against the controls is the measured ν is an assumption. The exact setup of the controls costs 305 instead of 223 (section 12).

## 5. Stop thresholds and reading points

The thresholds are the user's (section 1): forecasting at a fev-bench SQL Elo of **1707**, outlier detection at an ADBench mean AUROC of **0.77**, and the TabArena guard at **0.6%**: the pooled G over TabArena with 3 splits and the development suite, against the mean of #14's run and N10, must not exceed 1.006.

| Point | When | Rule, per task |
|---|---|---|
| R1 | tier 1 is evaluated | Take the best arm, or today's wrapper if no arm is better. **At or above the bar:** the task is done without training, its arm goes to the user as a release question, and the task is in no later tier. **Below the bar by at most the reported gain** (180 SQL Elo; 0.13 AUROC): the task is in the probe. **Below by more:** the task stops |
| R2 | the probe is evaluated | **At or above the bar:** the task is in the branch. **Below:** the task stops. If no task is left, the plan ends without tier 3 |
| R3 | the branch is evaluated | **The guard holds and a task's bar is reached:** the branch's weights go to the user as the v2 candidate for that task. **Otherwise:** they are not adopted, and nothing further is run |

- With today's numbers R1 lets both tasks through even if tier 1 gains nothing (172 against 180, 0.107 against 0.13). The user chose the rule knowing that. What R1 decides is whether a task needs training at all.
- **Between the points** the plan goes back to the user, with dropping the item as an option (rule 9), when an item's cost is projected more than 30% above its estimate, or when its expected gain falls: the pilot leaves a task's diagnostic unmoved, or an arm of 1.3 contradicts the derivation the probe rests on.
- **Each status report begins with four lines** (rule 11): the task metrics so far against the bars and the TabArena guard; where the plan is expected to end, with the kind of evidence; the H100-h spent, counted from the job records; and what is left of the plan and of the cap. A lower estimate of the plan's own budget is not reported as a result.
- Nothing in this plan is adopted or released by the plan. Every adoption is put to the user.

## 6. Costs and their bases

| Item | H100-h | Basis |
|---|---|---|
| fev-bench, ADBench, the causal suite, the imputation suite for TabSil alone, D1 to D3, per set of weights | 0.5, 1.7, 0.3, 0.2, 0.1 | measured (docs/task-results; v2_prior_diag README) |
| The imputation suite with all its baselines | 9 | measured (docs/task-results); most of it is the TabPFN extension's `impute` on 8 H100 |
| 1.4 | 3 | estimate: three sets of weights on four masks at 0.2 each, and MissForest and the mean on the node's CPUs (MissForest takes 11.6 s per table, rate and seed) |
| TabArena-Lite, 3 splits and the development suite, per set of weights | 2.5 | measured (results, phase 1A) |
| 2.1, one GPU, 2,000 steps at batch 32 | 3 | estimate from the time of a stage-2 table by its size, about 0.1 GPU-seconds for tables of at most 2,048 rows (`tabsil-v2-eff-s2-m-pertable-1gpu`, `f4c22b5c5627474bb763934e5b9cbb14`) |
| 2.2, 8 H100, 2,000 steps at batch 256 | 16 | the same estimate with the waiting of 8 GPUs. Its own first 100 steps are the check: a projected cost above 21 goes back to the user |
| 3.1, 8 H100, micro batch 1, 10,000 steps | 223 | measured on this setup with `graph_scm_hicard_missing`: 9.69 s per step, 21.5 H100-h per 1,000 steps (`tabsil-v2-eff-s2-m-mb1`, `bf797ac1f97047ec8a76849336322c02`: stage 2, v1's size, 8 H100, micro batch 1), with 3.5% for the start, validation and checkpoints. The prior of 3.1 is new, so the timing job 3.0 comes first (rule 6) |

Checks that cost under 1% of what they inform (rule 7): 3.0 before 3.1 (1.1%); the probe's D1 and D3 (0.2) before its benchmark evaluations; 1.6 and 1.7 before the designs that rest on their derivations. None is skipped.

## 7. Order

1. **Code** (section 8). No GPU.
2. **Tier 1.** Its items are independent evaluations and run together. Then the report of R1.
3. **Tier 2**, only after R1: the pilot, then the probe, then its evaluation and the report of R2.
4. **Tier 3**, only after R2: the timing job, then the branch, then its evaluation and the report of R3.

No run starts beside the step whose result bears on it (rule 8). The only items that run side by side are those of tier 1, and none of them can cancel another.

## 8. Code needed before the runs

Not part of the pull request of this plan, which is documents only. Prior code goes to `dev-v2`. The inference arms of tier 1 were written as wrappers in `training/scripts/v2_prior_diag/arms.py`, so nothing in `tabsil` changed for measuring them; an arm goes into the package on `main` only if the user releases it (this replaces the first version of this paragraph, which put the inference changes on `main` before the measurements).

| Change | For |
|---|---|
| `TabSilForecaster`: a recipe without the clipped target transform and with the feature kind as an option; a straight-line trend fitted on the context and added back | 1.2 |
| `TabSilUnsupervised`: the score as a mean over small random contexts; a wrapper that appends reference rows and returns the probability of their class | 1.3, 2.3, 3.2 |
| The diagnostics D4 and D5, D2 with a rare class, D1 for given weights | 1.4 to 1.7 |
| Forecast-shaped tables as an option of the registered priors, off by default: a split that sorts the rows of an ordinary table by one feature, with the context-row target scale. The series generator of v2-task-priors §1 is not built unless the user decides so (section 12) | 2.x, 3.1 |
| Outlier-shaped tables as an option, off by default: data rows with hidden outliers and the label 0, reference rows with the label 1, query rows with their true label (v2-task-priors §2, with the contamination ranges of TACTIC and ICLAD read first) | 2.x, 3.1 |
| A registered prior for the probe and one for the branch, so that the `prior_type` of a run names its data | 2.x, 3.1 |

## 9. The twelve rules

| Rule of the retrospective | Where this plan keeps it |
|---|---|
| 1. The approval question holds the expected gain in the user's unit next to the cost | The two dialogs of 2026-11-28 (section 1); the tier table of the summary; the questions of section 12 |
| 2. A tier that trains nothing; every tier states what it adds | Tiers 0 and 1 train nothing; the summary's column "what it adds" with its kind of evidence. Gains that are unknown are written as unknown, with the budget next to them |
| 3. The gain below which the rest is not run, and the reading points, fixed before the first run | Section 5: the user's bars, R1 to R3, and the four lines of each status report |
| 4. A premise that the evidence speaks against goes back to the user first | Section 1: three premises with options, put to the user before sections 4 to 8 were written |
| 5. Expected effect, resolution and decision by outcome for every run; nothing adopted on "not worse" | The tables of section 4; R3 adopts nothing by itself |
| 6. Costs measured on the setup that runs; a timing run above 100 H100-h; a smoke run for a new prior | Section 6: 3.1 cites `bf797ac1…` and is preceded by 3.0; 2.1 is the smoke run of the new table code |
| 7. A check under 1% of the run it informs comes before the decision | Section 6, last paragraph |
| 8. No run beside the step whose result bears on it | Section 7 |
| 9. A cost above 130% or a falling expected gain reopens the decision, with "drop" as an option | Section 5, "Between the points"; 2.1, 2.2 and 3.0 name their triggers |
| 10. "Recommended" only with an expected gain and not below the user's bar; every question has an option to stop | No tier and no option of this plan is marked as recommended: no gain above the bars has evidence. Both dialogs had an option not to do it |
| 11. A status report begins with the outlook | Section 5, the four lines; the summary of this document |
| 12. A plan above 500 H100-h is read by an agent that did not write it | This plan has about 275 H100-h under a cap of 500, below the rule's limit. It was read all the same, before any run (section 11) |

## 10. Relation to the earlier documents

| Document | Used as it is | Replaced by this plan | Not taken up |
|---|---|---|---|
| [v2-experiment-plan.md](v2-experiment-plan.md) | its evaluation protocol (§3: the paired ratio, the pooled suites), its cost table, the noise ν; its decisions 12 and 13 stand: L20, PS1 and phases 3 to 5 are not resumed | its phase 6 ("after the core": task priors as stage-2 branches at 182 to 307 H100-h each, judged by the branch rule) is replaced by tiers 1 to 3; its clause that adopts an unresolved arm for copying a published configuration is not used | the candidates of its §2 for TabArena accuracy; the optimizer split, which phase 1B left unresolved at 1.003 |
| [v2-task-priors.md](v2-task-priors.md) | D1 to D3 and their numbers; the design of its §2 as the outlier-shaped tables of tiers 2 and 3; D4 and D5 as tier-1 items | its success criteria, by the user's bars; "the model's part needs a prior" for forecasting, since the trend can be fitted outside the model (not measured); "a MAR arm is expected to gain little" is kept and made precise (formulations §5); its decision to keep the v1 structure was asked again where a derivation needs structure, and no chosen task does; its §1's series generator is held back for the sorted split (section 12) | its §3 and §4 as training arms: this plan trains nothing for imputation or effects |
| [v2-experiment-results.md](v2-experiment-results.md) | every measurement cited here, the phase-1B verdict among them (`target_train_rows` 0.995, 0.990 to 1.000, not resolved) | — | — |
| [v2-scaling.md](v2-scaling.md) | — | — | not used |
| [v2-retrospective.md](v2-retrospective.md) | its twelve rules (section 9) | — | — |

## 11. Review

Rule 12 asks for a reading by an agent that did not write the plan, with one question: what is the expected gain, and is the cheapest way to it in the plan? The agent was given the paths of the two documents and of the retrospective, the question, and nothing else. It read the plan on 2026-11-28, before any run of the plan exists. Its answer, as it gave it:

> **Expected gain.** The plan gives no expected gain in the user's unit for any tier that trains. The one measured gain is already in hand: the v2 candidate (v1's stage 1 plus 10K stage-2 steps on graph_scm_hicard_missing) scores 1808 against 1768 for v1.1 on TabArena-Lite, which is +20 to +40 Elo at a pooled error ratio of 0.992. That is a TabArena gain, not a task gain, and the user has declined further runs for it. The task bars are +172 SQL Elo on fev-bench (1535 to 1707) and +0.107 AUROC on ADBench (0.663 to 0.77). For both, the plan says the gain is unknown. Its one number, a MASE of about 0.4 on the D1 linear trend (from 11.75), comes from a least-squares fit on the same synthetic series. It is not a run of the model and not a fev-bench result. The reported comparators for the bars are not controlled tests of a prior alone: the +180 SQL Elo is TabPFN-TS to TabPFN-TS-3, where the base model also changed, and the +0.13 AUROC belongs to a different model (TACTIC). So the honest expected gain of the whole plan is zero shipped, unknown on the task metrics, and a sequence of probes that will say whether the bars are reachable at all.
>
> **Is the cheapest way in the plan?** Tier 1 is the right start. It costs about 30 H100-h, trains nothing, and its forecasting arms are the cheapest test of a gap that the derivation places in inference. Four points remain.
>
> - The one-model goal is the expensive part. The user chose one model for all tasks. The plan's own premise 3 says that a checkpoint per task, fine-tuned from the released weights, needs no TabArena guard and can be read on its own metric after a short run. Under the chosen goal, tier 3 (about 232 H100-h) plus its guard evaluation costs about 210 H100-h more than the probe (about 22). The plan should state that cost next to the choice, so the user sees what the guard buys.
> - The gate does not filter. Today's distances (172 SQL Elo and 0.107 AUROC) are already inside the reported gains (180 and 0.13). Both tasks therefore enter the probe even if tier 1 gains nothing. Only the 3 H100-h pilot (item 2.1) tests anything before the 22 H100-h probe.
> - The probe's table form is not fixed. Section 8 offers two forms: a new series generator (which needs a smoke run and has a generation cost) and a sorted split of ordinary tables. Section 3(d) calls the sorted split the smallest change, with no effect on generation time, and the derivation says that for forecasting only the split and the scale must change. The probe should try the sorted split first and add the generator only if the sorted split does not move D1 or D3.
> - Items 1.4 and 1.5 (the imputation and effect diagnostics D4 and D5, about 11.5 H100-h together) produce measurements with no bar and no training behind them. They are not a route to any gain in the plan. They are worth running only if the user wants those measurements; otherwise they should be dropped, since the plan trains nothing for those two tasks.
>
> **Bottom line.** The cheapest route to a task number in the plan is tier 1's forecasting arms (about 3 H100-h), then the sorted-split pilot (about 3 H100-h), then a 22 H100-h probe. Only if the probe reaches a bar does a larger run make sense, and the plan should say that the expected gain of that run is unknown. The question to the user should carry three things: the +20 to +40 Elo already in hand, the unknown gain on the two task bars, and the cost of the one-model choice relative to per-task checkpoints.

What was changed after it (the review read the plan as it stood before these changes, so its numbers are the earlier ones):

| Point of the review | Change |
|---|---|
| No expected gain in the user's unit for a tier that trains | The summary now says so in its first lines: nothing shipped is expected, the task gains are unknown, and tiers 1 and 2 are what finds out. The D1 number is marked as an estimate from a least-squares fit, not a run of the model |
| The one-model goal is the expensive part | Stated in the summary next to the tiers and in section 12, item 1: tier 3 exists only because the goal is one model, and costs about 232 H100-h where a checkpoint per task would end with the probe |
| The probe's table form is not fixed | Changed. The pilot and the probe use the sorted split for forecasting. The series generator is not built unless the pilot leaves D1 unmoved, and then it goes to the user as a question with its cost (sections 4, 8 and 12) |
| D4 and D5 cost 11.5 H100-h and lead to no gain in the plan | D4 was cut from 10 to 3 H100-h: MissForest and the mean are the only baselines. The dialog of 2026-11-28 had put each diagnostic at 1 to 2 H100-h, and the plan's 10 was above that. Tier 1 is now about 20 H100-h. Both are stated to be measurements and not steps toward a bar |

What was not changed:

| Point of the review | Why not |
|---|---|
| The gate R1 does not filter | It is the user's rule (section 1, c′), chosen with this consequence stated in the dialog. Section 5 says that R1 only decides whether a task needs training at all. The summary now says it too. The pilot is the first test before the probe, as the review notes |
| Drop D4 and D5 unless the user wants them | The user chose all four tasks on 2026-11-28, with these diagnostics named as the first step for the two tasks that are ahead of their baselines |
| "Does not move D1 or D3" as the condition for the generator | The sorted split bears on forecasting only. The condition is D1. The outlier-shaped tables have no smaller form |

## 12. For the user to decide

Where a number is the user's to set, none is filled in. Item 1 is decided; the others are open.

1. **Whether to run the plan, starting with tier 1. Decided 2026-11-28: run it.** Three things belonged to that question. In hand without any run: +20 to +40 Elo on TabArena-Lite, as the v2 candidate's weights. The gain on the two task bars: unknown; they need +172 SQL Elo and +0.107 AUROC, and the only number for tier 1 is an estimate of about 0.4 for 11.75 on D1's linear trend. The cost of the one-model goal: tier 3, about 232 H100-h, which a checkpoint per task would not need. Tier 1 costs about 20 H100-h, shows whether a task needs training at all, and gives imputation and effects their first measurement. Not running it leaves every gain of this plan unknown and costs nothing.
2. **The GPU setup of tier 3**, if the plan gets there: one 8xH100 node at micro batch 1 (223 H100-h, 28 h), which differs from the controls in the GPU count, or 32 H100 as the controls (305 H100-h, 9.5 h; measured, N10). The plan's table has the first, as the user chose one node for L20 and for phase 3 on 2026-11-27.
3. **The series generator for forecasting.** It is built only if the pilot leaves D1 unmoved with the sorted split, or if the user wants it after a probe that moves D1 and misses the bar. Its cost is the generator's code, a pilot of 3 H100-h and a probe of about 16. Expected gain: unknown.
4. **Imputation and treatment effects after D4 and D5.** No bar was set for them and this plan trains nothing for them. Their diagnostics come back with the measured gaps and the question whether a gap is worth a prior, and at what bar.
5. **The inference changes that help.** If an arm of 1.2 or 1.3 is better on its benchmark, whether it becomes the default of a tabsil release for the v1.1 weights.
6. **The classification correction on TabArena.** Formulations §2 derives a factor for rare classes. Measuring its effect costs about 2.5 H100-h. Expected gain: unknown, and only in the log loss of tables with rare classes; it is not in the plan because nothing says it exceeds the +20 to +40 Elo that the user declined to chase.
7. **The check scripts of the formulations.** The reviewers asked for them in the repository. This pull request is documents only.
8. **The 225 H100-h of the cap that the plan does not use.** They are not planned.
9. **Open from before:** whether and how the v2 candidate (#14's run and N10) is released (v2-experiment-plan, phase 5).

## Sources

Read on 2026-11-28. The task literature is listed in [v2-task-formulations.md](v2-task-formulations.md).

| Source | Used for | Status |
|---|---|---|
| Mitra, [arXiv 2510.21204](https://arxiv.org/abs/2510.21204), §4.7 and Table 6 | the prior mixture ablation | opened |
| Mitra-v2, [arXiv 2609.04540](https://arxiv.org/abs/2609.04540), §2.2 | the mixture weights | opened; its appendix C was not read |
| Kumo-Tabular, [arXiv 2611.23001](https://arxiv.org/abs/2611.23001), §7 and Table 2; `NVIDIA/structured-data-models` at `924e992` | the prior ablation, the training budget, the prior's parts | opened |
| Mind the Gap?, [arXiv 2605.06343](https://arxiv.org/abs/2605.06343) | prior against real tables | opened |
| Real-TabPFN, [arXiv 2507.03971](https://arxiv.org/abs/2507.03971) | continued pretraining on real tables | opened |
| TabPFN-3 report, [arXiv 2605.13986](https://arxiv.org/abs/2605.13986); TACTIC, [arXiv 2603.14171](https://arxiv.org/abs/2603.14171); TabImpute, [arXiv 2510.02625](https://arxiv.org/abs/2510.02625) | the gains reported for task-shaped training | opened |
| Shaping the Prior, [arXiv 2605.18971](https://arxiv.org/abs/2605.18971) | a small-scale controlled prior ablation | abstract opened; numbers by a subagent |
| APT, [arXiv 2502.04573](https://arxiv.org/abs/2502.04573); Drift-Resilient TabPFN, [arXiv 2411.10634](https://arxiv.org/abs/2411.10634); TabICLv2, [arXiv 2602.11139](https://arxiv.org/abs/2602.11139) | as cited | subagent; the quotations from the last two are those checked in v2-experiment-plan §1 |
