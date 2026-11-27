# v2 experiment plan

Status: plan, in progress. Results are in [v2-experiment-results.md](v2-experiment-results.md) (phase 0 done 2026-11-26; N10 and R10 of phase 1A done 2026-11-27, L20 proposed and not launched). Nothing was trained or evaluated for this document itself. It re-checks [v2-task-priors.md](v2-task-priors.md) and [v2-scaling.md](v2-scaling.md) against MLflow, the code and the primary sources (2026-11-26), and replaces their experiment plans. Every run below needs approval before it starts.

## Summary

The goal of v2 is higher TabArena classification and regression accuracy, with a model trained from scratch. Forecasting, outlier detection, imputation and treatment effects must not get worse than v1.

The check changed four things:

1. **Stage-1 TabArena-Lite Elo is not a usable readout.** A stage-1 model has only seen 1,024-row tables. On real tables of that size it keeps improving with training. On longer tables its regression error rises, and stage 2 repairs that. The flat stage-1 Elo is this effect, not evidence of a prior mismatch (§1.2).
2. **One prior change is worth about 1% of error, and so is the difference between two runs.** #14's gain is a geometric-mean error ratio of 0.991. Two stage-1 runs with the same seed and prior differ by 0.990 at 25K steps. One-variable screening with one or two seeds cannot separate the two (§1.1).
3. **v1 has Kumo-Tabular-L's size and 7% of its training compute.** At v1's compute, Kumo trains its 27M model, and that model is level with v1 (§1.3). Size and compute have to be chosen together.
4. **The measured gains are in stage 2.** Stage 2 gives all of its gain on tables above 2.5K rows, #14's prior was a stage-2 change, and stage 2 was still improving at 10K steps (§1.1, §1.2).

The plan therefore:
- judges every arm by the paired per-dataset error ratio on more data than TabArena-Lite's 51 splits, and evaluates stage-1 models with the context capped at 1,024 rows (§3)
- tests stage-2 changes as branches from one stage-1 checkpoint, which is the low-noise design (phase 1A)
- tests stage-1 changes as a few bundles with 2-3 seeds on a small model (phase 1B)
- picks the model size at matched compute (phase 2), then trains v2 (phases 3-5)
- moves the task priors after the core, as stage-2 branches with guards (phase 6)

The recommended budget is about 4,350 H100-h. A lean version is about 2,690 (§5).

**Update 2026-11-27.** Phases 2 to 5 were redesigned from measured GPU costs (§4, "Cost basis for phases 2 to 5"; [v2-experiment-results.md](v2-experiment-results.md), "GPU cost"). A smaller model saves little compute, a run costs less per step on fewer GPUs, and stage 3 has shown no effect that the suites can measure. The recommended options are no size experiment, v2's stage 1 on 16 H100, its stage 2 on one node at micro batch 2, and no stage 3. They cost about 930-1,110 H100-h where the plan had 2,115-2,422. The whole plan then comes to about 3,190-3,370, against about 4,380-4,680 with phases 2 to 5 as planned (§5). They are proposals until the user decides (§7).

## 1. What was checked

Verdicts: confirmed, partly, wrong, not verifiable. "Records" are MLflow runs read on 2026-11-26. The paired numbers in §1.1 and §1.2 were computed for this document from the runs' `eval/tabarena-lite/results_per_split.csv`.

| # | Claim | Verdict | Evidence | Effect on the plan |
|---|---|---|---|---|
| 1 | The v2 plan serves the goal ([v2-task-priors.md](v2-task-priors.md), Decision) | Wrong for the current goal | Its success criteria are task-suite metrics, and TabArena is only a guard. No source shows that such task priors raise i.i.d. accuracy. Drift-Resilient TabPFN reports "lower predictive performance on ID tasks" with a temporal-shift prior. TabPFN-3 has temporal and out-of-distribution priors, without an ablation | Task priors move to phase 6 |
| 2 | "Prior mismatch; capacity is not tested" ([v2-scaling.md](v2-scaling.md), Verdict) | Numbers confirmed, reading partly wrong | Every value of the checkpoint table matches MLflow. The flat Elo is a context-length effect (§1.2). On tables within the training size, more stage-1 steps lower the real error | Stage-1 models are evaluated with a capped context. "More stage-1 steps do not help" is not shown |
| 3 | "v1 is as large as the top models, with TabPFN-3.5's backbone, so the gap is not size" | Partly | Counts confirmed: TabPFN-3.5 219.0M (24 layers × 16 heads at width 1024, from its checkpoint), Kumo-Tabular-L 214M, Seldon 213M. The leaderboard Elo values match TabArena v0.1.9.23. But size helps where training grows with it: LimiX-2 gains 45.7 Elo per doubling from 12.5M to 406M with a fixed recipe, and Kumo's L beats its S on 34 of 36 datasets. v1 has L's size at 7% of its compute (§1.3). TabPFN-3.5's training recipe is not published | Size is chosen at matched compute (phase 2) |
| 4 | The Kumo prior comparison (#14, #26) | Partly, with errors | See §1.4. "Kumo turns all of these off in stage 1" is wrong: structured missing values, correlated blocks and the column warp are on in every stage. Several parts are not ported | The v2 prior is split by stage as in Kumo |
| 5 | #14's prior gives +42 Elo | Partly | Error ratio 0.991 (95% CI 0.982-0.998, 32 datasets better and 19 worse). The gain sits in the datasets named beforehand: polish 0.84, anneal 0.94, kddcup 0.95, Amazon 0.99. Without those four it is 0.996. No second seed exists. The gain overlaps the 0.1.1 inference gain (§1.1) | Phase 1A repeats the run with another seed |
| 6 | A 30K-step stage-1 run screens a prior change ([v2-task-priors.md](v2-task-priors.md), phase 1) | Wrong as designed | The readout is the context-length effect (§1.2). Run-to-run noise is as large as the expected effect (§1.1). Kumo's own ablation: the prior "marginally improves" at 100K steps and matters "more with longer training". TabICLv2 developed its prior "mostly without experimental feedback, since fine-grained ablations are impractical" | Replaced by phases 1A and 1B |
| 7 | TabArena-Lite can resolve effects of a few tens of Elo | Wrong | 51 datasets with one split each. The Elo CI of ±100 is for one method, not for a difference. With the paired ratio at 80% power, 51 splits resolve about 1.2% for two branches of one checkpoint (per-dataset sd 0.03), and 1.6-3.9% for two independent stage-1 runs (sd 0.04-0.10), before run-to-run noise (§1.1) | §3: paired ratio, more datasets and splits, seeds |
| 8 | The task-prior citations | Mostly confirmed, three errors | All arXiv ids resolve. Wrong: "no work trains on a contaminated context" (TACTIC 2603.14171 and ICLAD 2603.19497 do). Wrong condition: TabImpute's 2.455 against 1.966 are all-pattern numbers, and under MCAR MissForest is 1.018. Left out: TabImpute's Self-MNAR model is worse overall (1.695 against 1.585). LimiX v1 has one masked objective, not separate heads | Corrected when phase 6 is designed |
| 9 | No controlled evidence for a masked auxiliary loss | Confirmed | LimiX and LimiX-2 have no ablation of it. LimiX v1 also uses it, so "all supervised except LimiX-2" was wrong | Stays deferred |
| 10 | Training settings not examined so far | Found | The regression target is standardized over all rows in training and over the train rows at inference. Muon runs on every parameter, also 1-d ones. One model learns both tasks. The validation tables depend on the run seed and on the arm's own prior (§2, §8) | Phase 1B arms and the code list |
| 11 | The cost estimates | Confirmed | 0.71 s per step on 32 H100 is 6.3 H100-h per 1K stage-1 steps. Stage 2 is 30.7 and stage 3 is 144 per 1K steps (9.58 h and 9.0 h on 32 H100). The costs of other sizes are FLOPs estimates | The timing runs stay first |
| 12 | The size experiment ([v2-scaling.md](v2-scaling.md), steps A, B, G) | Question right, design changed | Step A compares sizes at equal steps and reads the prior loss on seed-dependent tables. The sources pair size with compute | Phase 2 compares sizes at equal compute |

### 1.1 Effect sizes and noise in the existing runs

G is the geometric mean over the 51 TabArena-Lite datasets of the error ratio b / a. The error is TabArena's: 1 − ROC AUC for binary, log loss for multiclass, RMSE for regression. Below 1 means b is better. The CI is a bootstrap over datasets. All runs are the repository harness without bagging, with 0% imputed.

| Comparison (b against a) | G (95% CI) | Classification / regression | Better / worse | Elo change |
|---|---|---|---|---|
| **Noise.** Two stage-1 runs with the same seed and prior, 150K and 250K schedules, both at 25K steps (`2769dcc2…`, `40f5801a…`) | 0.990 (0.978-1.001) | 0.991 / 0.989 | 34 / 17 | +36 |
| The same pair at 50K, 75K, 100K, 125K | 1.014, 1.010, 0.992, 0.990 | | | |
| Stage 2 10K → stage 3 2K (`81a9845f…`, `775c4d65…`) | 1.000 (0.998-1.002) | 1.000 / 1.000 | 23 / 28 | +2 |
| **Effects.** #14 prior against v1 prior, stage 2 10K (`f2ca29bf…`, `a0a2e4e5…`) | 0.991 (0.982-0.998) | 0.988 / 0.998 | 32 / 19 | +42 |
| Inference 0.1.0 → 0.1.1 on `stage2-10k` (`f2ca29bf…`, `a8b69941…`) | 0.994 (0.985-1.002) | 0.993 / 0.998 | 31 / 20 | +35 |
| 1 → 8 ensemble members, v1 (`c9970716…`, `8c9c085d…`) | 0.986 (0.976-0.994) | 0.986 / 0.985 | 35 / 16 | +80 |
| Stage 2: 5K → 10K (`2889d6e9…`, `81a9845f…`) | 0.993 (0.987-0.998) | 0.991 / 0.998 | 35 / 16 | +32 |
| Stage 1 150K → stage 2 10K (`6f629c89…`, `f2ca29bf…`) | 0.894 | 0.966 / 0.712 | 38 / 13 | +296 |

- **1% of error is about 40 Elo** at this place in the pool.
- **Two nominally equal stage-1 runs differ by about 1%.** The 25K pair even passes a Wilcoxon test (p 0.02), as #14's gain does. A single stage-1 run per arm cannot show a 1% effect.
- **Branches of one checkpoint are much closer.** The end of stage 2 and the end of stage 3 have a G of 1.000, with a per-dataset sd of 0.007. The noise between two stage-2 runs from the same stage-1 weights is not known, because none was repeated.
- **#14's largest gains and the 0.1.1 inference gains fall on the same datasets.** Under 0.1.1 alone, polish is 0.89 and kddcup 0.87. The per-dataset log ratios of the two changes correlate at 0.62 (0.19 by rank, so the overlap is in the few large gains). #14's weights were never evaluated with 0.1.1 inference, so the combined gain is unknown.
- The 5K point of stage 2 is a checkpoint in the middle of a 10K schedule. A 5K schedule would end higher, so 0.993 overstates the effect of the length.

### 1.2 Stage 1 and table length

TabArena-Lite has 19 datasets with at most 2,500 training rows, and 24 classification and 8 regression datasets above that.

| Comparison | ≤ 2.5K rows (19) | > 2.5K rows, classification (24) | > 2.5K rows, regression (8) |
|---|---|---|---|
| Stage 1, 25K → 250K steps (`40f5801a…`, `42419873…`) | 0.922 (0.835-0.978) | 0.916 | 1.676 |
| Stage 1, 75K → 250K (`1d2ed9db…`, `42419873…`) | 0.949 (0.893-0.985) | 0.958 | 1.482 |
| Stage 1, end of the 150K run → end of the 250K run (`b3c0f0dd…`, `42419873…`) | 0.983 (0.959-1.000) | 0.999 | 1.106 |
| Stage 1 150K → stage 2 10K | 0.999 (0.991-1.004) | 0.948 | 0.577 |

- Within stage 1, regression improves on tables of at most 1.1K rows (0.96 from 25K to 250K) and gets worse above 2.5K rows (1.55 to 1.76). The rank correlation between the number of rows and the change is 0.78 over the 13 regression datasets.
- Stage 2 changes nothing on small tables and gives all of its gain on long ones.
- Kumo reports the same for its models: the stage-1 checkpoint "keeps pace with the released model up to about 1K training rows", and stage 2 "adds 65 to 170 points" at every size.
- So the prior loss and the real error do move together on tables of the training size. From 75K to 250K steps the error on those tables falls by 5%. The last 100K steps give 1.7%, which 19 datasets cannot confirm (p 0.11).
- A second cause is possible and not checked. Kumo trains its stage-1 regressors in FP16, because BF16 let the first in-context layer's attention temperature grow about 30-fold. TabSil trains both tasks in BF16. Phase 0 reads the temperatures from the v1 checkpoints.
- Earlier short runs do not support realism options in a short stage 1. Four 20K-step runs on an older layout each turned one option on at 0.3: coarse values, high-cardinality codes, long-tailed targets, class imbalance (`prior103-C…F` in the minipfn experiment, as far as their logged parameters show). They scored 18 to 60 Elo below their control `prior103-B`, with one seed each and the full-length readout that this plan does not trust.

### 1.3 Size and compute

Tables per task are steps × batch. v1 alternates the two tasks, so each task gets half of its steps. H100-h cover both tasks. Elo is the public leaderboard (full TabArena, v0.1.9.23).

| Model | Parameters | Tables per task, stage 1 / 2 / 3 | H100-h | Elo |
|---|---|---|---|---|
| TabSil v1 (one model) | 221.5M | 19.2M / 1.28M / 0.26M | 1,555 | 1728 |
| Kumo-Tabular-S (two models) | 27M | 32M / 5.1M / 1.5M | 1,573 | 1804 |
| Kumo-Tabular-M (two models) | 61M | 64M / 5.1M / 1.5M | 7,186 | 1915 |
| Kumo-Tabular-L (two models) | 214M | 128M / 7.7M / 2.3M | 23,148 | 1979 |
| TabICLv2 (two models) | 28M | 32M / 2.6M / 0.64M | 1,176 | — |

- v1 saw 15%, 17% and 11% of Kumo-L's tables in the three stages, and 40-60% of TabICLv2's.
- Kumo raises the training with the size: M doubles S's batch, and L doubles the batch and the steps, "to account for the increase in parameter count".
- On the 51 TabArena-Lite splits, v1 with 0.1.1 inference (`138151a2…`) has these error ratios G to the public results of other models. The references are 8-fold bagged and TabSil is fit once.

  | Reference | G, v1 / reference | Classification / regression | v1 better on |
  |---|---|---|---|
  | Kumo-Tabular-S | 0.996 | 0.988 / 1.020 | 26 of 51 |
  | Kumo-Tabular-M | 1.047 | 1.052 / 1.032 | 14 |
  | Kumo-Tabular-L | 1.070 | 1.079 / 1.044 | 10 |
  | TabPFN-3.5 (default) | 1.020 (1.004 without polish_companies_bankruptcy) | 1.023 / 1.010 | 19 |

- So Kumo-S reaches v1's accuracy with an eighth of the parameters and the same compute. The leaderboard's 76 Elo between them is with tabsil 0.1.0's inference.
- Kumo's sizes also price the compute: M has 4.6 times S's compute and 4.9% less error, and L has 14.7 times and 6.9% less. The size and the training change together along that line, and the sources do not say how much each gives.

### 1.4 The Kumo prior, corrected

Checked in `NVIDIA/structured-data-models` (`train/kumo/tabular/`) and the paper (Table 2, Table A.1).

| Part | Kumo | TabSil today |
|---|---|---|
| Structured missing values (0.2), random missing cells | every stage. About 21% of tables have missing values in stage 1 and 27% later | `structured_missing_prob` is on only in the #14 and #26 priors. Random cells are in 30% of tables at rates up to 30%; Kumo has 1% of tables at 1-8% |
| Correlated blocks (0.3), warp of the column itself | every stage | options of `graph_scm_realism` |
| Categories per column | 2-200 for independent columns | `corrected_cat_sizes` gives 2-200. The docstrings say 100 |
| Near-empty columns, latent-factor missingness, relabel, coarsen, high-cardinality codes, argmax classes | stages 2 and 3 | all but argmax classes are options. High-cardinality is 10% of tables from 21 categories; Kumo has 5% from 201 |
| Too-easy tables | rejected in stage 1, half kept later | not ported |
| Noisy-measurement columns, per-table function preferences, heavy-tailed targets (stage 3), row replay (stage 3) | yes | not ported |
| Regression target | standardized with the context rows (`loader.py`) | standardized over all rows. Inference uses the train rows |
| Optimizer | Muon on weight matrices, AdamW on the rest, peak lr 3e-4 / 4e-5 / 1e-5 | Muon on every parameter, 8e-4 / 1e-4 / 2e-5 |
| Tasks | one model per task. Classes uniform in 2-10 | one model. Half of the classification tables are binary |

## 2. What could raise accuracy

| Lever | Evidence | Strength | Tested in |
|---|---|---|---|
| Stage-2 prior realism | #14: 0.991, on the named datasets. Kumo starts its realism parts in stage 2 and reports that its prior changes "matter more with longer training" | medium | 1A |
| Stage-2 length | 5K → 10K: 0.993. Kumo-L uses 6 times v1's stage-2 tables | medium | 1A |
| Target standardized with the train rows | Kumo does it in training and at inference. TabSil's training and inference disagree. D1 showed the consequence for forecasts | medium, not measured here | 1B |
| Optimizer split and lr | Kumo, Mitra-v2 and EXAONE-Tabular use Muon with AdamW. TabICLv2: Muon against AdamW is about 100 Elo | weak | 1B |
| Every-stage prior parts | Kumo ships them. TabICLv2: "the prior yields the largest effect". Kumo's ablation: marginal at 100K steps | weak at screening length | 1B, as a guard |
| Model size for the budget | §1.3: Kumo-S equals v1 at the same compute with 27M parameters; larger sizes win only with several times the compute | medium that it matters, direction unknown | 2 |
| Stage-1 length | §1.2: 1.7% on small tables for 100K more steps, not significant | weak | budget tiers |
| One model per task | most families do it. TabPFN-3.5, TabFM and LimiX train one. No ablation exists | none | optional |
| Class mix, share of random missing cells | they differ from Kumo. No evidence | none | inside the prior bundle only |
| Task priors | no evidence for i.i.d. accuracy, one negative report | for the task metrics only | 6 |
| Masked auxiliary loss | no controlled evidence | none | deferred |
| Stage 3 | v1: 1.000. Kumo: at most 23 Elo, on tables above 10K rows | small | kept for long tables |
| Inference recipe (16 members, sign flips) | Kumo's ablation. It needs no retraining | — | main, not v2 |

## 3. Evaluation protocol

- **Statistic.** G, as in §1.1, between an arm and its control, on the same splits and the same evaluation seed. Splits are averaged within a dataset first. It is reported overall, for classification and for regression, with a bootstrap CI over datasets (2,000 rounds), the Wilcoxon signed-rank p and the count of datasets better and worse. Elo is reported but decides nothing.
- **Data.**
  - TabArena-Lite with 3 splits per dataset (153 splits).
  - A development suite outside TabArena (`--suite dev`): the AMLB and CTR23 datasets of `--suite all` (29 + 28, 10 folds), without those that are also among TabArena's 51. Of these 57, 9 are in TabArena (6 classification and 3 regression), which leaves 48: 23 classification and 25 regression. An earlier name match over the full AMLB lists had found 10 classification and 3 regression overlaps. The list for these 57 and its check are in [v2-experiment-results.md](v2-experiment-results.md) (0.5).
  - Decisions use both, pooled and apart. The final claim is made on the full TabArena with the official 8-fold protocol, and on BeyondArena, which no decision has used.
- **Stage-1 checkpoints** are evaluated with `--subsample-train-rows 1024`. Stage 1 trains on contexts of 307 to 922 rows. TabICLv2 ran its ablations the same way, on 60 datasets cut to 2,048 training rows.
- **Checkpoints after stage 2** use the full context.
- **Inference** is tabsil 0.1.1's defaults (8 members) for every run and every control.
- **Prior validation loss.** Today each run draws 64 validation tables per task from its own prior and seed, so the loss cannot be compared across priors or seeds. A fixed reference set makes it a second readout for arms that share a prior (§8).
- **Seeds.** Arms trained from scratch get 2-3 seeds. Branches from one checkpoint get one run each, and one of them is repeated to measure the noise.

Decision rules, fixed here:

- **Branch from a shared checkpoint (phases 1A, 4, 6).** Let ν be |ln G| between the two seeds of the repeated branch. An arm is adopted if its G is at most exp(−max(0.005, 2ν)), its CI lies below 1, and neither task is worse by more than 0.5%.
- **Arm trained from scratch (phase 1B).** Start with 3 control seeds and 2 arm seeds. Each run gets one score, its mean log error over the datasets.
  - If both arm seeds score below all three control seeds, run the third arm seed. The arm is adopted on the evidence if that seed is also below all three (a one-sided permutation p of 0.05) and G of the seed means is at most 0.98.
  - Otherwise the effect is below what this scale resolves, about 2%. The arm is then adopted only if it removes a train/inference mismatch or copies a published configuration, and G is at most 1.01.
  - An arm with G above 1.02 and a CI above 1 is dropped.
- **Guards for every run** are those of [v2-task-priors.md](v2-task-priors.md), read as ratios: no task type worse than its control by more than 0.5% beyond the noise, and 0% imputed.

## 4. Experiments

Costs are H100-h on 32 H100 from v1's throughput (§1, row 11). Costs at other sizes use the FLOPs ratios of [v2-scaling.md](v2-scaling.md) (XS 0.29, S 0.60 in stage 1) until the timing runs replace them. Evaluations cost about 1-3 H100-h per checkpoint and are left out of the totals. Phases 2 to 5 use the costs measured on 2026-11-27 ("Cost basis for phases 2 to 5").

Order: phase 0, then 1A and 1B side by side, then 2, 3, 4 and 5. Each phase is approved on the results of the one before it.

### Phase 0: measurement (about 30 H100-h)

| Step | What | Cost |
|---|---|---|
| 0.1 | #14's weights (`277916ec…`) with 0.1.1 inference, on the 51 splits. Runs with today's code | about 1 |
| 0.2 | `stage1-150k` and v1 with the context capped at 1,024 rows, on the 51 splits. Runs with today's code | about 2 |
| 0.3 | Timing runs of 300 steps for XS and S on 8 H100, as [v2-scaling.md](v2-scaling.md) step 0 | under 5 |
| 0.4 | The learned attention temperatures of `stage1-150k`, `stage2-10k` and v1, per layer. No forward pass | minutes |
| 0.5 | The evaluation code of §8: 3 splits per dataset, the development suite, the paired report | — |
| 0.6 | v1, `stage2-10k`, `stage1-150k` and #14's weights on the 3 splits and the development suite, the stage-1 weights also with the cap. These are the baselines of every later comparison | about 20 |

- 0.2 tests §1.2. If the reading is right, the capped `stage1-150k` is within about 1% of the capped v1, in regression too. If it is not, phase 1B's readout has to be rethought before any run.
- 0.4 decides whether phase 1B gets an FP16 arm for the regression steps.

```bash
# 0.2 for stage1-150k (v1: @v1.1; 0.1: runs:/277916ec05a3457aa6d07f2bc08a60c4/model/final.pt without the cap)
databricks air run -p e2-demo-field-eng -f training/scripts/eval_arena.yaml --watch \
  --override mlflow_experiment_directory=$MLFLOW_EXPERIMENT_DIR \
  --override mlflow_run_name=tabsil-stage1-arena-cap1024 \
  --override compute.accelerator_type=GPU_8xH100 --override compute.num_accelerators=8 \
  --override env_variables.TABSIL_SHARDS=8 \
  --override "env_variables.TABSIL_CHECKPOINTS=--classifier-path hf://aiden-jeon/tabsil@stage1-150k --regressor-path hf://aiden-jeon/tabsil@stage1-150k" \
  --override "env_variables.TABSIL_EXTRA_ARGS=--method-name tabsil-stage1-cap1024 --subsample-train-rows 1024"
```

### Phase 1A: stage-2 pilot on the v1 stage-1 weights (1,228 H100-h)

Stage 2 from `aiden-jeon/tabsil@stage1-150k`, as #14's experiment A. It needs no new training code and can run beside phase 1B. Its controls exist: `stage2-10k` and #14's run.

| Run | Prior | Steps | Seed | Question | Cost |
|---|---|---|---|---|---|
| N10 | `graph_scm_hicard_missing` | 10K | 1 | Does #14's gain repeat, and how far apart are two stage-2 runs (ν)? | 307 |
| R10 | `graph_scm_realism` | 10K | 0 | Do Kumo's other stage-2 parts add to #14? | 307 |
| L20 | the better of the two | 20K | 0 | Does a longer stage 2 help? | 614 |

- N10 against `stage2-10k`: #14's gain repeats if G is at most 0.995 and the four named datasets have a mean ratio of at most 0.97, the criterion #14 used.
- R10 against the mean of #14's run and N10, and L20 against the 10K run of its prior, use the branch rule of §3.
- The result fixes the stage-2 prior and length of phase 4. The best run is also a release candidate that does not depend on phases 1B to 3.

```bash
# N10 (R10: TABSIL_PRIOR_TYPE=graph_scm_realism, TABSIL_SEED=0; L20: TABSIL_STEPS=20000, about 19 h, so timeout 1440 or two chunks)
databricks air run -p e2-demo-field-eng -f training/scripts/train.yaml --watch \
  --override mlflow_experiment_directory=$MLFLOW_EXPERIMENT_DIR \
  --override mlflow_run_name=tabsil-v2-s2-n10 \
  --override compute.num_accelerators=32 --override timeout_minutes=720 \
  --override env_variables.TABSIL_CONFIG_NAME=stage2 \
  --override env_variables.TABSIL_STEPS=10000 \
  --override env_variables.TABSIL_INIT_HUB=aiden-jeon/tabsil@stage1-150k \
  --override env_variables.TABSIL_PRIOR_TYPE=graph_scm_hicard_missing \
  --override env_variables.TABSIL_SEED=1 \
  --override env_variables.TABSIL_NUM_WORKERS=16 \
  --override env_variables.TABSIL_COMPILE=1 --override env_variables.TABSIL_TF32=1 \
  --override env_variables.TABSIL_WEIGHT_AVERAGE=10 --override env_variables.TABSIL_WEIGHT_AVERAGE_EVERY=100 \
  --override env_variables.TABSIL_MUON_DISTRIBUTED=all
```

### Phase 1B: stage-1 recipe on a small model (500-670 H100-h)

From scratch, stage 1 only, 30K steps, batch 256, `--emsize 128` (XS, 56M). Each run is about 56 H100-h. The readout is the capped evaluation of §3 and the reference validation loss.

| Arm | Change against C | Seeds | Needs code |
|---|---|---|---|
| C | v1 recipe, prior `graph_scm` | 3 | no |
| YCTX | the regression target is standardized again with the train rows, after the prior's clip | 2 (+1) | a prior option; the borders are drawn with it |
| OPTK | Kumo's optimizer: Muon on weight matrices, AdamW on 1-d parameters, target embeddings, heads and attention scales; peak lr 3e-4 | 2 (+1) | optimizer groups |
| PS1 | the every-stage prior parts at Kumo's values: column warp, 2-200 categories, correlated blocks 0.3, structured missing 0.2, random missing cells in 1% of tables at 1-8%, too-easy tables rejected | 2 (+1) | a registered prior; the too-easy filter |

- YCTX, OPTK and PS1 are bundles that copy a published configuration, not single variables. This scale cannot attribute a 1% effect to one option (§1.1).
- PS1 is expected to be neutral at 30K steps (§1, row 6). It is run to catch harm. The stage-2 parts of the prior are tested in 1A, where they act.
- If two or more arms are adopted, one run with all of them checks that they combine (56-168 H100-h).
- Kumo ran its ablation in the same way: 100K-step runs of its small model, three seeds, every outer split.

### Cost basis for phases 2 to 5 (2026-11-27)

Phases 2 to 5 below were redesigned on 2026-11-27, after a phase-1B run turned out to cost 124 H100-h where the plan had 56. The costs were measured with short timing jobs ([v2-experiment-results.md](v2-experiment-results.md), "GPU cost"). The option marked "recommended" in each phase is a proposal. The user decides (§7), and each phase is still approved on the results of the one before it.

Cost per 1K steps at batch 256, with the code as it is. A whole run costs 3-4% more for its start, validation and checkpoints.

| Stage and size | GPUs | Micro batch | s / step | H100-h per 1K steps | Basis |
|---|---|---|---|---|---|
| Stage 1, M (v1's size) | 8 | 4 | 1.79 | 3.98 | measured |
| Stage 1, M | 16 | 4 | about 1.08 | about 4.8 | estimate |
| Stage 1, M | 32 | 4 | 0.71 | 6.31 | v1's record, older code; the estimate for today's code is the same |
| Stage 1, S | 8 | 4 | 1.74 | 3.87 | measured (results 0.3) |
| Stage 1, XS | 8 | 8, `TABSIL_PREFETCH_FACTOR=16` | 1.18 | 2.62 | measured |
| Stage 2, M | 8 | 1 | 9.69 | 21.5 | measured |
| Stage 2, M | 8 | 2 | 7.93 | 17.6 | measured |
| Stage 2, M | 8 | 2, `TABSIL_PREFETCH_FACTOR=16`, collector frozen (§8) | 7.36 | 16.3 | measured |
| Stage 2, M | 16 | 1 | about 5.5 | about 24 | estimate |
| Stage 2, M | 32 | 1 | 3.30 | 29.3 | measured (N10) |
| Stage 3, M | 8 | 1 | — | about 85-100 | estimate from a one-GPU run |
| Stage 3, M | 32 | 1 | 16.2 | 144 | v1's record, older code |

What the measurements change:

- **A smaller model saves little.** XS costs 0.66 of M per step in stage 1 at its cheapest setting, and S 0.97. The plan's FLOPs ratios were 0.29 and 0.60. The forward, the optimizer step and the waiting take the same time at every size, and only the backward shrinks.
- **A run costs less per step on fewer GPUs.** Every GPU waits for the slowest one at each step, and on more GPUs each has fewer tables to even its work out. Stage 2 costs 21.5 H100-h per 1K steps on 8 H100 and 29.3 on 32. Stage 1 costs 3.98 and 6.31. Fewer GPUs take longer, so each phase below states both.
- **The micro batch is at its limit for M and S in stage 1** (4: 8 runs out of memory). XS takes 8. Stage 2 takes 2, which saves 18%.
- **Two changes to the training code would lower every cost** (§8): freezing the garbage collector's long-lived objects (4.6% in stage 1 and 9-10% in stage 2 on 8 H100, measured; more on more GPUs) and giving every GPU of a step the same work (estimated). The costs below are without them, unless a row says so.

How runs are set up from here on:

- One node (8 H100) per run where the wall time allows it, and 16 H100 where a run is on the critical path. 32 H100 costs 50-70% more than one node for the same steps (phases 3 and 4), so it is for runs whose wall time matters more than that.
- Stage 1 at micro batch 4 (M) or 8 with `TABSIL_PREFETCH_FACTOR=16` (XS). Stage 2 on one node at micro batch 2 with `TABSIL_PREFETCH_FACTOR=16`.
- A control and its arm use the same GPU count and micro batch: both change which tables a run sees, and the micro batch also the number of table sizes per step (results, "What a larger micro batch changes").

### Phase 2: model size at equal compute (redesigned 2026-11-27: skip recommended; was 570 H100-h)

The original design (2026-11-26) gave each size about 190 H100-h of stage 1 with its own cosine schedule and priced the sizes by FLOPs, so that XS would get 3.3 times M's steps. Measured, equal compute is M 48K, S 49K and XS 72K steps: XS gets 1.5 times M's steps and S the same as M.

| Option | Runs | H100-h | Wall time | Gives | Gives up |
|---|---|---|---|---|---|
| A. As planned, each size at its cheapest setting | M 48K, S 49K and XS 72K steps, one seed each | 570, and 380 more for second seeds | 24 h on 3 nodes | the three sizes at one compute point | one seed per size cannot separate two sizes closer than about 2% (§1.1) |
| B. XS against phase 1B's runs | 2 XS seeds of 45K steps with the recipe 1B adopts, 119 H100-h each, which is the cost of one 1B run. The M side is 1B's 30K-step seeds of that recipe | 240 | 15 h on 2 nodes | XS against M at equal compute, judged by the 1B rule (§3) with M's seed noise measured | S. It costs what M costs, so at equal compute it has M's steps and fewer parameters |
| C. Skip | none | 0 | — | 570 H100-h and a day | a measured size curve. v2 keeps v1's size |

**Recommended: C.** B if a smaller v2 is wanted for its inference cost.

- **The reason for the phase is mostly gone.** A smaller model was to buy more training at the same compute (§1.3). It buys 1.5 times the steps in stage 1, not 3.3, and in stage 2 XS costs 0.79-0.87 of M.
- **The rule can hardly be met.** A smaller size is chosen only at G ≤ 0.98 against M (kept from the original design). 3.3 times the steps were worth 5% on small tables (§1.2), so 1.5 times is worth 1-2%. Against that stands what the size itself costs: LimiX-2 loses 45.7 Elo per halving at a fixed recipe (§1, row 3), about 2% for XS.
- **A tie keeps M**, so only an unlikely outcome would change the plan.
- **What B answers is a product question**: how much accuracy a model with a quarter of the parameters gives up for its speed at inference. It can run at any time after 1B and does not block phase 3.
- **Condition for B.** Two XS seeds can show 2% only if 1B's seeds of the chosen recipe lie within about 1% of each other (G between two seeds). If they do not, B is not run.

### Phase 3: v2 stage 1 (redesigned 2026-11-27: 620-745 H100-h; was 950)

From scratch with the recipe and stage-1 prior that 1B adopts, at v1's size, for 150K steps. The H100-h below hold the 3-4% for the start, validation and checkpoints.

| Option | GPUs | s / step | H100-h | Wall time | Basis |
|---|---|---|---|---|---|
| A | 8 (one node) | 1.79 | 620 | 77 h, in 4 chunks | measured |
| B | 16 (two nodes) | about 1.08 | about 745 | 47 h, in 3 chunks | estimate: the simulation gives 1.79 and 0.71 for A and C |
| C | 32 (four nodes), as v1 | 0.71 | 950 | 30 h, in 2 chunks | v1's record |

With the collector change of §8: A 592 H100-h in 74 h (measured), B about 680 in 42 h and C about 830 in 26 h (estimates).

**Recommended: B, if a 300-step timing run on 16 H100 (5 H100-h) gives at most 1.15 s per step. Otherwise A.**

- **Phase 4 needs L20's verdict as well as these weights.** L20 runs on one node until about 2026-11-29 09:00 UTC and is evaluated after that. If phase 3 starts on 1B's verdict (2026-11-28 00:10 UTC), C ends about 11-29 06:00 and then waits for L20, B ends about 11-30 00:00 and A about 12-01 05:00.
- **B buys 30 h for 125 H100-h against A.** C's further 205 H100-h buy 12 h at most.
- **A is the choice if the cost comes first**, as it did for L20 (one node instead of four, decided 2026-11-27).
- **Chunks.** `TABSIL_STOP_STEP` and `TABSIL_RESUME_RUN_ID`, each chunk within the job timeout used so far (22-24 h).
- **A first check after 5K steps (about 25 H100-h).** The reference validation loss is set against 1B's runs of the same recipe at 5K steps, where both schedules are still near their peak lr. A loss outside 1B's range by more than a few percent stops the run before the rest is spent.
- **Gate, unchanged.** On the capped evaluation, G against `stage1-150k` is at most 1.00. A v2 stage 1 that is worse than v1's stops here, and the best run of 1A stays the candidate.
- **Not proposed: a longer stage 1.** The 950 H100-h of the original estimate would now pay for 230K steps on one node. The evidence for more stage-1 steps is weak (§2), so the saving is kept.
- **If phase 2 runs and picks XS**, 150K steps cost about 410 H100-h on one node (49 h), and equal compute to A is 228K steps.

### Phase 4: v2 stage 2 (redesigned 2026-11-27: 182 H100-h for 10K steps, 364 for 20K; was 307-614)

Stage 2 from phase 3's weights. Its prior is the stage-1 prior plus the stage-2 parts that 1A chose: the high-cardinality and structured-missing parts of `graph_scm_hicard_missing`, without the parts `graph_scm_realism` adds. Its length is 20K steps if L20 passes the branch rule and 10K if it does not. It is compared with the best run of 1A and with v1 on the full context.

Per 10K steps (twice for 20K), with the 3-4% for the start, validation and checkpoints:

| Option | GPUs | Micro batch | H100-h | Wall time | Basis |
|---|---|---|---|---|---|
| A | 8 | 2, `TABSIL_PREFETCH_FACTOR=16` | 182 | 23 h | measured without the prefetch setting, which removes stalls of 0.2 s per step |
| B | 8 | 1, as N10 and #14's run | 223 | 28 h | measured |
| C | 16 | 2 | about 205 | about 13 h | estimate |
| D | 32 | 1, as N10 | 305 | 9.5 h | measured (N10) |

With the collector change of §8, A is 169 H100-h in 21 h (measured).

**Recommended: A.** C if 20K steps are chosen and 46 h are too long.

- **A costs 60% of D.** On one node a GPU has 32 tables per step instead of 8, so the GPUs' work evens out, and micro batch 2 spreads the fixed cost of a forward over two tables.
- **Micro batch 2 is a change to the recipe.** A step has 128 table sizes instead of 256. The tables' distribution and every table's weight are the same (results, "What a larger micro batch changes"). 1A's runs and the noise ν were measured at micro batch 1 on 32 H100. Option B keeps micro batch 1 for 22% more.
- **Memory.** Micro batch 2 peaks at 58.7 GB on the largest stage-2 table, of 85 GB. Micro batch 4 does not fit.
- **L20 runs option A's setup for 20K steps** (8 H100 at micro batch 2, decided 2026-11-27 from these measurements, about 350 H100-h in 44 h). Its step time and memory over a whole run are the check of this cost.
- **Phase 6's branches** follow the same setup, at about 182 H100-h per 10K steps instead of 307. A control and its arms use the same GPU count and micro batch.

### Phase 5: stage 3 and release (redesigned 2026-11-27: no stage 3 by default; was 288 H100-h)

- **Stage 3 has shown no effect that the suites can measure.** v1 against its own stage 2 is 1.000 on TabArena and 0.999 on the development suite (results 0.6). By table size (results, "Stage 3 by table size"): 0.999 (0.997-1.001) on the 15 TabArena datasets above 10K training rows and 1.000 on the 4 above 50K. The only CI below 1 is that of the 3 regression datasets above 20K rows (0.996). The noise between two stage-2 runs is 0.003.
- **It costs 144 H100-h per 1K steps** on 32 H100 (v1's record), 288 for v1's 2K steps. On one GPU a stage-3 table takes 0.91 GPU-seconds against 2.03 in v1's run, so more than half of that cost is waiting for the slowest GPU. One node is estimated at 85-100 H100-h per 1K steps. That was not measured.

| Option | H100-h | Gives | Gives up |
|---|---|---|---|
| A. No stage 3. The stage-2 model is evaluated and released | 0 | 288 H100-h and 9 h | at most what stage 3 gave v1: 0.1% on tables above 10K rows, within the noise |
| B. 1K steps on one node, judged on TabArena's 15 datasets above 10K rows and on BeyondArena | about 95 (estimate), 12 h | a direct test on v2's weights | — |
| C. As planned: 2K steps on 32 H100 | 288 | v1's recipe unchanged | — |

**Recommended: A, with B kept open.** Stage 3 continues from the stage-2 weights, so it can be added after the release evaluation without repeating anything. B is run if that evaluation shows v2 behind v1 on the tables above 10K rows by more than the noise.

- **Release evaluation, unchanged**, against v1 at the same tabsil version: the full TabArena with the official protocol, AMLB, and the four task suites as guards (fev-bench, ADBench, the imputation suite, IHDP and ACIC). As before, evaluations are not in the totals.

### Later screens (2026-11-27)

Phase 1B resolves about 2% with 7 runs of 30K steps at v1's size (9 before PS1 was dropped), at about 124 H100-h each. The same resolution for less, for a control and one arm:

| Design | H100-h | Resolves | Holds if |
|---|---|---|---|
| As 1B: M, 30K steps, 3 control and 2 (+1) arm seeds, one node per run | 620-745, of which 372 are the control that all arms share | about 2% | — |
| The same at 20K steps | 415-495 | about 2% | 1B's own logs show it: the arm − control difference of the reference loss at 20K has the sign and about the size of the one at 30K, and the C seeds are as close at 20K. Read when 1B ends; no run |
| The same on XS at micro batch 8 | 405-490 | about 2%, for XS | XS stands for M (§7, first risk). 1B was moved to M to avoid that |
| 2 control and 2 arm seeds first, the third of each only if both arm seeds beat both controls | 495, up to 745 | about 2% | — (the rule's p of 0.05 needs all six) |
| Stage-2 branches from one checkpoint on one node: control, its repeat and the arm | 545 for 10K steps each; 182 with a control and a repeat that exist | about 0.6% (ν 0.003, phase 1A) | the change can be switched on in stage 2. ν was measured at micro batch 1 on 32 H100 and is measured again by the repeat |

**Recommended: stage-2 branches for every question they can answer** (prior parts, table sizes, stage-2 length and lr, the task priors of phase 6). They resolve three times finer for the same cost. A screen from scratch is kept for what a branch cannot test (the optimizer, the size): on one node per run, with 2 + 2 seeds first, and at 20K steps if 1B's logs allow it.

- **What sets the cost of a screen from scratch** is the spread between seeds, about 1% (§1.1), not the evaluation. For a fixed resolution the cost is proportional to the cost of a run times the squared spread, so a design has to lower one of the two.
- **Not known: the noise between two continuations of one stage-1 checkpoint.** If it is as small as in stage 2, stage-1 questions about the prior could be branched too, at about 40 H100-h per 10K steps. Two such runs (80 H100-h) would measure it. Not proposed for now.

### Phase 6: after the core

Each of these is a stage-2 branch from the v2 stage-1 weights, at 307 H100-h per 10K steps at v1's size on 32 H100 (about 182 on one node at micro batch 2, phase 4), judged by the branch rule plus its own metric.

- **Task priors** (forecasting, outlier detection, imputation, treatment effects), designed as in [v2-task-priors.md](v2-task-priors.md) with the corrections of §1, row 8. YCTX already gives every regression table the train-row scaling that the forecasting prior asked for. D4 and D5 still decide the imputation and treatment arms.
- **One model per task**, a longer stage 1, the cell-stage growth arm CS, and the masked auxiliary loss. None has evidence that would put it ahead of phases 1-5.

## 5. Budget

| Tier | Contents | H100-h |
|---|---|---|
| Stage 2 only | phase 0 and 1A. No training from scratch; the result is new stages 2 and 3 on v1's stage 1 | 1,258 (+288 for stage 3) |
| Lean | phase 0; 1A without L20; 1B with 9 runs; no phase 2 (v1's size); phase 3; phase 4 at 10K; phase 5 | 2,689 |
| Recommended | phases 0 to 5 as written, phase 4 at 20K | 4,350 |

Budget after the redesign of 2026-11-27, in H100-h. The tiers above are the plan of 2026-11-26.

| Phase | Plan (recommended tier) | Spent or committed by 2026-11-27 | Redesign, recommended options |
|---|---|---|---|
| 0 | 30 | about 30, done | — |
| 1A | 1,228 | N10 305 and R10 450, done. L20 about 360 on 8 H100 at micro batch 2 (decided 2026-11-27; 614 on 32 H100 before) | — |
| 1B | 500-670 | about 940: 7 runs at v1's size at about 124 each and the two PS1 runs that were stopped. Running | — |
| GPU cost measurements | — | 29, done | — |
| Phases 0 to 1B | 1,928 | about 2,230: about 1,250 spent by 2026-11-27 08:00 UTC, with the evaluations, smoke runs and diagnostics that the rows above leave out, about 620 left of 1B and about 360 for L20 | — |
| 2 | 570 | — | 0 (skip). 240 for option B |
| 3 | 950 | — | 745 on 16 H100. 620 on 8 H100 |
| 4 | 307 (10K steps) to 614 (20K) | — | 182 to 364 on 8 H100 at micro batch 2 |
| 5 | 288 | — | 0 (no stage 3). About 95 for option B |
| Phases 2 to 5 | 2,115-2,422 | — | 930-1,110 |
| Total | 4,350 | | about 3,190-3,370 |

- With phases 2 to 5 as planned, the total would be about 4,380-4,680.
- The cheapest set, with phase 3 on one node, is 800-985 for phases 2 to 5.
- With the collector change of §8 the recommended set is about 850-1,020.
- What the redesign gives up: a measured size curve (phase 2), whatever stage 3 adds on long tables, which was 0.1% for v1 and within the noise (phase 5), and wall time against runs on 32 H100: 17 h in phase 3 and 13 h per 10K steps in phase 4.

- The earlier plans added up to 2.7-4.3K H100-h: phases 1 and 2 of [v2-task-priors.md](v2-task-priors.md) at 0.95-1.52K, the final run at 1.55K, and steps A and B of [v2-scaling.md](v2-scaling.md) at 0.17-1.19K.
- What the measured levers support: #14's stage-2 prior gave about 1%, and the second half of stage 2 gave less than 1%. Together that is about the distance to TabPFN-3.5 (default) in §1.3, which is 2%, most of it on one dataset.
- Kumo-M is 4.7% ahead of v1 and Kumo-L 7%, at 4.6 and 15 times v1's compute. Nothing measured here closes that at these budgets.
- The gain from the stage-1 changes of 1B and from the size is not known before those phases run.

## 6. What changes in the earlier plans

| Plan | Kept | Changed | Deferred |
|---|---|---|---|
| [v2-task-priors.md](v2-task-priors.md) | D1-D3 and their readings. The forecasting recipe on main (open item 7). Guards and approval for every run. Prior code that calls the inference code | The base prior is no longer `graph_scm_hicard_missing` from step 0: its parts are split by stage (§1.4). Phase 1's 30K screening is replaced by 1A and 1B. "Only the prior changes" no longer holds: the recipe and perhaps the size change, and each change is measured against its own control | The TS, OD, IMP and CA arms, the ALL mix, D4 and D5: phase 6 |
| [v2-scaling.md](v2-scaling.md) | The records, which match MLflow. The parameter counts per stage. The timing runs. `--emsize` and `--n-cls` (#27). CS stays gated | The verdict: the flat stage-1 Elo is a context-length effect (§1.2). Step A becomes phase 2, at equal compute with the capped readout | Steps B and G |
| #26's "Next" | `graph_scm_realism` and its options | It is tested in stage 2 (R10), not from step 0 of a from-scratch run | — |

## 7. Risks and decisions

Risks:
- **The small model may not stand for the large one.** TabICLv2 reports that the prior and the architecture interact, and that more depth showed no gain at 280K steps. Phase 3's gate is the check.
- **ν comes from one pair of runs.** It is a lower bound on the stage-2 noise.
- **Decisions still read TabArena-Lite.** The development suite and the untouched BeyondArena limit the overfitting, but the 51 datasets have been used for every choice so far. Kumo states the same about its own development.
- **PS1 changes several things at once.** If it does harm, v2 falls back to the v1 stage-1 prior with the stage-2 parts from 1A.
- **The size choice rests on one compute point.**
- **(2026-11-27) The costs on 16 H100 are estimates.** The simulation behind them gives the measured step on 8 H100 and v1's record on 32 H100 within 3% in stage 1, and is 4-8% low in stage 2. A 300-step timing run comes before a run on 16 H100 is committed.
- **(2026-11-27) Without phase 2, v2's size rests on other models' evidence** (§1.3), not on a TabSil run.
- **(2026-11-27) Micro batch 2 in stage 2 halves the table sizes per step.** No run has compared it with micro batch 1.
- **(2026-11-27) Without stage 3, v2 has seen no table above 10,240 rows.** v1's stage 3 did not change the error on such tables in TabArena. BeyondArena has not been looked at.
- **(2026-11-27) L20 now differs from its controls in two things.** It runs 20K steps at micro batch 2 on 8 H100. #14's run and N10 ran 10K steps at micro batch 1 on 32 H100. Its G against them holds the length and the micro batch together. If L20 passes the branch rule, phase 4 takes both. If it does not, a 10K-step run at micro batch 2 (182 H100-h) is what separates the two.

Decisions for the user:
1. The budget tier (§5).
2. Whether phase 1A may run on v1's stage-1 weights although v2 is trained from scratch. It is recommended: it measures the two best-supported levers and the noise, and it is the fallback.
3. Whether v2 may change the training recipe and the size. v1 is then not a one-variable control of the final run.
4. Whether TabSil stays one model for both tasks. The plan keeps it and lists the test as optional.
5. Whether the task priors wait for phase 6.

Decisions from the redesign of 2026-11-27, open. The recommended option is first.
6. Phase 2: skip it (0 H100-h), run XS against 1B's seeds (240), or run it as planned (570).
7. Phase 3: 16 H100 after a timing check (about 745 H100-h, 47 h), 8 H100 (620, 77 h), or 32 H100 as v1 (950, 30 h).
8. Phase 4: one node at micro batch 2 (182 H100-h and 23 h per 10K steps), one node at micro batch 1 (223, 28 h), or 32 H100 as N10 (305, 9.5 h). The length follows L20.
9. Phase 5: no stage 3 (0), 1K steps on one node (about 95), or 2K steps on 32 H100 as v1 (288).
10. The code changes of §8: the collector change before phase 3 (about 3 lines, the training result unchanged), and equal work per GPU (it changes which tables a run sees, and its effect is an estimate).
11. Later screens: stage-2 branches on one node wherever they can answer the question, and screens from scratch at 20K steps if 1B's logs allow it.

## 8. Code needed first

| Change | For | Base |
|---|---|---|
| `eval.py`: more than one split per dataset; the development suite without TabArena's datasets; a paired report of two runs (G, CI, Wilcoxon, per task) | every phase | dev-v2 |
| The arena cache key includes the weights, not only the checkpoint path | every phase | dev-v2 |
| A fixed reference validation set, independent of the run seed and of the arm's prior, with more than 64 tables | 1B, 2 | dev-v2 |
| A prior option that standardizes the regression target with the train rows | YCTX | dev-v2 |
| Optimizer groups: AdamW for the parameters Muon does not suit | OPTK | dev-v2 |
| A registered stage-1 prior with the every-stage parts; a lower bound on the random-missing rate; the too-easy-table filter | PS1 | dev-v2 |
| Docstrings that say 100 categories where the code gives 200 (`generators/graph_scm.py`, `transforms/dependence.py`) | — | dev-v2 |
| The regression loss averages over rows with a missing target as zeros; exclude them, as the classification loss does | phase 6 | dev-v2 |

### Changes proposed by the cost measurements (2026-11-27)

Not made. The evidence for each is in [v2-experiment-results.md](v2-experiment-results.md) ("Changes that need code").

| Change | For | Effect |
|---|---|---|
| `gc.collect()` and `gc.freeze()` on every GPU after the first log window, and again at every log step (`tabsil_train.training.loop`) | every run | stage 1 −4.6% and stage 2 −9 to −10% on 8 H100, measured with one freeze. About −15% and −12% on 32 H100, estimated. The training result is unchanged |
| The table sizes of a step drawn from one stream that all GPUs share, and dealt out so that every GPU has the same predicted work | stages 2 and 3, and every run on more than one node | stage 2 on 32 H100 from 29.3 to about 17-20 H100-h per 1K steps, together with the collector change (simulation). It changes which tables a run sees |
| The largest data wait over the GPUs as a logged metric | every run | none on speed. A stall of 1 s per step shows as 0% today |
| A profile of the forward | stage 1 above all | unknown. The forward is 28-32% of a stage-1 step and takes the same time at every width and feature count |

Not ported from Kumo and left for later: the noisy-measurement columns, per-table function preferences, argmax classes, heavy-tailed targets and row replay.

## Sources

Read on 2026-11-26. Where a paper and its code disagree, the code is quoted.

- Kumo-Tabular: [arXiv 2611.23001](https://arxiv.org/abs/2611.23001) (Table 2, §7, Table A.1, §9) and [NVIDIA/structured-data-models](https://github.com/NVIDIA/structured-data-models) (`train/kumo/tabular/`)
- TabICLv2: [arXiv 2602.11139](https://arxiv.org/abs/2602.11139) (§4.1, §7, appendices B and C) and [soda-inria/tabicl](https://github.com/soda-inria/tabicl)
- TabPFN-3.5: the [technical report](https://storage.googleapis.com/prior-labs-tabpfn-public/reports/tabpfn-v3.5-report.pdf), the [checkpoint](https://huggingface.co/Prior-Labs/tabpfn_3_5) and [PriorLabs/TabPFN](https://github.com/PriorLabs/TabPFN). TabPFN-3: the [technical report](https://storage.googleapis.com/prior-labs-tabpfn-public/reports/TabPFN_3_model_report.pdf)
- LimiX-2: [arXiv 2609.17488](https://arxiv.org/abs/2609.17488) (§6, Table 8). LimiX: [arXiv 2509.03505](https://arxiv.org/abs/2509.03505)
- Mitra-v2: [arXiv 2609.04540](https://arxiv.org/abs/2609.04540). Seldon: [arXiv 2611.32257](https://arxiv.org/abs/2611.32257). EXAONE-Tabular: [arXiv 2608.25774](https://arxiv.org/abs/2608.25774)
- TabArena: [arXiv 2506.16791](https://arxiv.org/abs/2506.16791), [autogluon/tabarena](https://github.com/autogluon/tabarena), and the [leaderboard](https://huggingface.co/spaces/TabArena/leaderboard) at v0.1.9.23
- Task priors: Drift-Resilient TabPFN [arXiv 2411.10634](https://arxiv.org/abs/2411.10634), TACTIC [arXiv 2603.14171](https://arxiv.org/abs/2603.14171), ICLAD [arXiv 2603.19497](https://arxiv.org/abs/2603.19497), TabImpute [arXiv 2510.02625](https://arxiv.org/abs/2510.02625)
- Paired comparison over datasets: Demšar, JMLR 2006

Not verifiable from public sources: TabPFN-3.5's training recipe and prior details; the training budgets of LimiX-2, Seldon, Causilo, Inpriora and Lattice; the run-to-run variance of any published model on TabArena.
