# v2 plan: priors and tasks

Status: **draft, 2026-11-28. Sections 1 to 3 are written; the plan itself (tiers, runs, stop thresholds) is not, because it needs five decisions of the user first (section 1, "Decisions asked before the plan").** Nothing was trained or evaluated for this document and no GPU job was run. It follows the stop of the v2 experiment plan ([v2-experiment-plan.md](v2-experiment-plan.md) §7, decisions 12 and 13) and is built to the twelve rules of [v2-retrospective.md](v2-retrospective.md) ("Rules for the next plan"). The derivations it rests on are in [v2-task-formulations.md](v2-task-formulations.md).

The request it answers, as the user made it on 2026-11-28 after reading the retrospective: plan again in line with the retrospective, with more diverse priors and more diverse tasks, and with the tasks derived in formulas first.

How to read:

- Every number has its source and its kind. **Measured**: from a run in the records. **Derived**: from the formulations document. **Estimated**: computed from measured numbers by a stated step. **Reported**: another group's number for another model, read in its paper. **Unknown**: no basis.
- G is the paired error ratio of [v2-experiment-plan.md](v2-experiment-plan.md) §3 (below 1 is better). On TabArena-Lite, 1% of error is about 40 Elo at TabSil's place in the pool (its §1.1).
- GPU costs are H100-hours.

## 1. Premise check

Rule 4 of the retrospective: a premise of the request that the evidence speaks against goes back to the user before the plan is built on it. The request has three premises.

### Premise 1: a more diverse prior makes the model better

**The request assumes** that adding variety to the training prior raises accuracy.

**The evidence says:**

- **Measured here, one prior change has a gain, and it is the size the user declined.** Stage 2 on `graph_scm_hicard_missing` against the v1 prior: pooled G 0.992 (0.980 to 1.000) over two seeds, TabArena-Lite Elo 1768 → 1788 and 1808. Without polish_companies_bankruptcy it is 0.997, and on the development suite 0.996 (v2-experiment-results, phase 1A). The other parts taken from Kumo-Tabular add nothing to it: 1.003 (0.999 to 1.008). On 2026-11-28 the user judged +20 to +40 Elo not worth further runs.
- **A gain of a prior part sits in the datasets that have the property it adds.** G is a geometric mean over datasets. One dataset of 51 improved by 40% is 1%: that is the hicard result (polish 0.60, kddcup 0.96, the rest level). A part that should give 2.5%, about 100 Elo, has to improve 13 datasets by 10% each, or 3 by 35%.
- **Controlled results in the literature are of that size or come from adding a function family.** Mitra reports +63 Elo, on its own pool, for adding one tree-ensemble prior to an SCM prior, and nothing more for four further priors (1062 for six priors against 1063 for two; arXiv 2510.21204, Table 6; seeds not stated). Mitra-v2 states that varying its mixture weights "did not produce a large change in downstream accuracy" (arXiv 2609.04540, §2.2). Kumo-Tabular's ablation: "The prior change marginally improves performance", at 100K steps with three seeds (arXiv 2611.23001, §7). TabICLv2 developed its prior "mostly without experimental feedback". The memo this project held, that no controlled ablation of a prior exists, is wrong: Mitra's is one.
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
| Forecasting | fev-bench SQL Elo 1535: 49 below TabPFN-TS on TabPFN v2, 229 below TabPFN-TS-3, 307 below Chronos-2. D1, linear trend 15: MASE 11.75 with the default recipe, 9.03 without its clip, 3.67 for seasonal naive | the recipe caps the median at the context maximum: inference. The model returns toward the context mean, as a prior without trends does: the trend fitted outside the model (inference, not measured) or a prior with temporal splits and trends | inference, then prior |
| Outlier detection | ADBench mean AUROC 0.663: 0.059 below IsolationForest, 0.157 below the same model with a clean context. D3, a dense group at 2%: 0.425 | the score is the density of the contaminated context, and a dense group has a high density. A better density does not repair it. Small random contexts (inference, not measured) or a prior with labeled reference rows and hidden outliers | inference, then prior |
| Imputation | ahead of every baseline under MCAR (NRMSE 0.546 against 0.616 for MissForest at 10%). MAR and MNAR not measured (D4) | MCAR is the v1 prior's own split. MAR keeps the conditional. MNAR is not identified: a prior teaches its own mechanism's tilt | none; prior for MNAR, with a limit |
| Treatment effects | ahead of the baselines on IHDP (root PEHE 0.55 against 2.08 for the EconML T-learner; CausalPFN reports 0.58). Low overlap not measured (D5) | the point estimate is already the posterior mean. Intervals and hidden confounding need structure | none; structure for intervals |

So two tasks have a measured gap, and for both a part of it may close without training. Two are ahead of their baselines and have nothing measured against them.

**Options.** Inference changes first and training only for the gap that remains; training on task priors directly; or no task work.

### Premise 3: one set of weights for every task

**The request assumes** (as [v2-task-priors.md](v2-task-priors.md) did) that the tasks are trained into the one model.

**The evidence says** that no source shows this with numbers. The models that report gains from training on task-shaped tables are separate models or separate checkpoints: TabPFN-TS-3 is a TabPFN-3 checkpoint "fine-tuned on synthetic time-series data" (arXiv 2605.13986, §3.3); TACTIC, TabImpute and CausalPFN are trained for one task each; Kumo-Tabular's paper covers classification and regression only, and Kumo's forecasting and anomaly models are separate time-series models (NVIDIA SDM documentation, read by a subagent). The one report of a tabular model trained on a shifted task says it lost in distribution (Drift-Resilient TabPFN, "lower predictive performance on ID tasks"). Whether a task prior at a small share costs TabArena accuracy in TabSil is unknown.

**Options.** One model with a TabArena guard, which needs a branch that can resolve the guard (0.6%, about 180 to 305 H100-h per branch). Or a checkpoint per task, fine-tuned from the released weights, which needs no guard and can be read on the task's own metric after a short run.

### Not in the request: the lever with support for TabArena

Kumo-Tabular-M has 4.6 times the training compute of its S model and is 111 Elo above it on the public leaderboard (1915 against 1804); L has 14.7 times and is 175 above. Kumo-S is level with v1.1 at v1's compute (G 0.996; v2-experiment-plan §1.3). Size and compute change together along that line, so this is a reported association for another model. For TabSil, 100K more stage-1 steps gave 1.7% on small tables with p 0.11 and nothing on the full readout (§1.2). 4.6 times v1's compute is about 5,600 H100-h more. It is listed so that it is not left out again; nothing in this plan proposes it.

### Decisions asked before the plan

The plan's tiers, runs and thresholds depend on these, and the user sets them (rules 1, 3 and 10). They were put to the user on 2026-11-28. The answers are recorded here when they arrive.

| # | Decision | Answer |
|---|---|---|
| a | The goal: which metric has to move, and whether TabArena must rise or only hold; one model or a checkpoint per task | pending |
| b | Which tasks are in the plan | pending |
| c | The gain below which the rest is not run, per metric | pending |
| d | The budget cap in H100-h | pending |
| e | Whether the structure may change (asked only if a chosen task needs it) | pending |

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
| A sorted split: the query rows are those with the largest values of one feature, on ordinary prior tables | split and scale | formulations §3: the queries of forecasting never occur under a random split; the SCM already has linear and quadratic mechanisms | none: not found in the literature in this form | unknown, on D1 and fev-bench | the smallest change: no new generator and no effect on generation time |
| A series generator with trend, seasonality and noise, split in time | generator, split, scale | formulations §3: the forecast of a trend-free prior returns to the context mean | reported: TabPFN-TS-3, fine-tuned on synthetic series, has an SQL skill of 43.1% on fev-bench against 39.6% for TabPFN-TS on TabPFN v2, with the base model changed as well. In this repository's pool the two are at 1764 and 1584 | unknown; the reported pair spans +180 Elo of SQL in this repository's pool | a new generator: smoke run first |
| Labeled reference rows with outliers hidden among the data rows | context and labels | formulations §4: the classifier converges to the posterior outlier probability | reported for a model built for the task: TACTIC, trained on contaminated contexts, has a mean AUROC of 80.6 on 38 ADBench tables with a contaminated context against 67.6 for the density score of the TabPFN-2.5 extension and 77.7 for IsolationForest | unknown: +0.03 to +0.13 AUROC is what that model has over IsolationForest and over the density score | a new generator part and a wrapper; whether the retrieval decoder learns it is not known |
| Query rows chosen by a missingness mechanism | split and scale | formulations §5: MAR keeps the conditional; MNAR changes it and is not identified | reported: TabImpute's specialists, 0.4% on MAR; 21% on self-masking and a loss of 39% on MCAR | MAR about 0; MNAR positive on a benchmark masked by the prior's mechanism and negative elsewhere | a split by a score; no generation cost |
| Query rows with a randomized treatment | split | formulations §6: counterfactual queries have an unbounded density ratio at low overlap | none | unknown; D5 is not run | a marked column in the generator's graph |
| Masked cells with a reconstruction output | structure | formulations §7: all conditionals in one pass; no change for classification and regression at the optimum | no controlled evidence | unknown | a new output: a change of structure |

### Measuring a prior against real tables without training

- **It can be done.** Summary statistics per table, a classifier that tells synthetic tables from real ones, and the share of real tables with a synthetic neighbor (arXiv 2605.06343). Drawing a few thousand prior tables takes CPU minutes.
- **It did not predict accuracy** in the one study that tested it (section 1). So it does not rank the candidates above.
- **What did work in this repository** is narrower: #14 named, before its run, the four datasets with the property its prior adds. With two seeds the gain is on two of them (polish 0.60, kddcup 0.96), and the other 47 TabArena datasets are level (0.999 to 1.000). A candidate of (a) to (c) is worth a run only if benchmark datasets can be named that have its property and on which TabSil is behind. The per-dataset gaps to named models are in the existing evaluation runs and cost no GPU to read.

## 4 to 9. The plan

Not written yet: the tier table with a first tier that trains nothing, the run table with the expected effect, the resolution and the decision of each run, the stop thresholds and the points at which they are read, the table of the twelve rules, the relation to the earlier documents, the outside review and the list of the user's decisions. They follow the answers to the five decisions of section 1.

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
