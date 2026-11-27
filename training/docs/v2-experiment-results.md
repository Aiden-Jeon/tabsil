# v2 experiment results

Results of the runs in [v2-experiment-plan.md](v2-experiment-plan.md), by phase. Every run is in the MLflow experiment `/Users/jongseob.jeon@databricks.com/pfn-cookbook/tabsil`. Paired numbers come from `training/scripts/paired_report.py` (plan §3): G is the geometric mean over datasets of the error ratio b / a, after the splits of each dataset are averaged. The 95% CI is a bootstrap over datasets with 2,000 rounds, p is the two-sided Wilcoxon signed-rank test, and the last count is datasets better / worse. Every evaluation uses tabsil 0.1.1's inference defaults with 8 members, and every evaluation listed here has 0% imputed splits.

## Phase 0

Started and finished on 2026-11-26.

### Inference settings

The Hub tags `stage1-150k` and `stage2-10k` save 0.1.0's member kinds (`["standard"]`) in their `config.json`. Only `v1.1` saves 0.1.1's settings. So every run below passes `--member-kinds numeric-cycle --categories counts`, which are 0.1.1's defaults. The earlier 0.1.1 run on `stage2-10k` (`a8b69941…`) was run the same way. #14's weights (`runs:/277916ec…/model/final.pt`) have no saved settings and get 0.1.1's defaults anyway. The flags are logged as the `member_kinds` and `categories` params. Since 0.5, the settings each task actually ran with are also logged, as the `inference_classification` and `inference_regression` params.

### 0.1: #14's weights with 0.1.1 inference

`tabsil-p0-prior14-v011-arena` (`7cc31c4845c442079d998a1c02839533`), TabArena-Lite, 51 splits, 0% imputed. Elo 1807.6 (classification 1793.0, regression 2113.8).

| b against a | G (95% CI) | p | Classification / regression | Better / worse | Elo a → b |
|---|---|---|---|---|---|
| #14 0.1.1 against v1.1 (`138151a2…`) | 0.991 (0.974-1.000) | 0.32 | 0.987 / 1.001 | 28 / 23 | 1768 → 1808 |
| #14 0.1.1 against `stage2-10k` 0.1.1 (`a8b69941…`) | 0.990 (0.975-0.999) | 0.13 | 0.986 / 1.001 | 30 / 21 | 1762 → 1808 |
| #14 0.1.1 against #14 0.1.0 (`a0a2e4e5…`) | 0.994 (0.979-1.004) | 0.38 | 0.991 / 1.001 | 27 / 24 | 1769 → 1808 |
| #14 0.1.1 against `stage2-10k` 0.1.0 (`f2ca29bf…`) | 0.985 (0.963-0.999) | 0.007 | 0.980 / 0.999 | 33 / 18 | 1727 → 1808 |

- #14's prior and 0.1.1's inference add up almost fully: 0.991 (prior, §1.1) × 0.994 (inference on `stage2-10k`, §1.1) = 0.985, and the combined change measures 0.985. The plan's concern was that the two gains might overlap because they fall on the same few datasets. They do not cancel.
- Under 0.1.1 inference, #14's prior is still worth about 1% (0.990 against `stage2-10k` 0.1.1), all of it in classification. Regression is level (1.001).
- On 51 single splits this is not significant (p 0.13, CI upper end 0.999). Phase 1A's N10 repeats the prior on another seed, and the 3-split runs below give the same comparison on 153 splits.

### 0.2: stage 1 with the context capped at 1,024 rows

`tabsil-p0-stage1-cap1024-arena` (`41e98b3cfc864355a98b564a6beb3613`) and `tabsil-p0-v1-cap1024-arena` (`6bd8bb032b88475889f6f937c29e9224`), TabArena-Lite, `--subsample-train-rows 1024`, the same rows for both (seed 0), 0% imputed.

| b against a | G (95% CI) | p | Classification / regression | Better / worse | Elo a → b |
|---|---|---|---|---|---|
| `stage1-150k` capped against v1.1 capped | 1.003 (1.000-1.006) | 0.17 | 1.002 / 1.005 | 22 / 29 | 1226 → 1219 |

By table size (training rows of split 0): ≤ 2.5K rows (19 datasets) 1.004, > 2.5K classification (24) 1.003, > 2.5K regression (8) 1.003. The largest single ratios are airfoil_self_noise (1.043) and APSFailure (1.034).

**Verdict: §1.2's reading holds.** With the context capped at 1,024 rows, the v1 stage-1 checkpoint is within 0.3% of the finished v1, and regression is within 0.5% (1.005, CI 0.999-1.012). Without the cap, stage 2 gave 0.577 on long regression tables (§1.2). So the gap between stage 1 and v1 is the context length, not what stage 1 learned. Stages 2 and 3 add nothing on 1,024-row contexts. The capped readout of phase 1B and phase 2 measures what stage 1 learns, and phase 1B can start with that readout.

### 0.3: timing runs

300 steps of stage 1 on 8 H100: batch 256, micro batch 4 (8 accumulation steps), prior `graph_scm`, `TABSIL_COMPILE=1 TABSIL_TF32=1 TABSIL_MUON_DISTRIBUTED=all TABSIL_NUM_WORKERS=16` (the speed options of the stage-2 runs). Seconds per step is the logged `sec_per_step` at steps 200 and 300. Step 100 includes compilation. `data_wait_frac` is at most 0.002 in every run.

| Size | Run | Parameters | s / step (steps 200, 300) | GPU memory | H100-h per 1K steps | 30K steps | Ratio to M | FLOPs ratio ([v2-scaling.md](v2-scaling.md)) |
|---|---|---|---|---|---|---|---|---|
| XS (`TABSIL_EMSIZE=128`) | `tabsil-p0-timing-xs` (`881c60c29ebe4f02b074fcd58dc45c46`) | 56.0M | 1.497, 1.471 | 30.8 GB | 3.27 | 98 | 0.81 | 0.29 |
| S (`TABSIL_EMSIZE=192`) | `tabsil-p0-timing-s` (`3a846b768b1a4d85a32d273f64847c55`) | 125.1M | 1.753, 1.730 | 45.2 GB | 3.84 | 115 | 0.95 | 0.60 |
| M (v1 size) | `tabsil-p0-timing-m` (`e9d825c1749e47e099d78c0dbc18317b`) | 221.5M | 1.808, 1.820 | 60.9 GB | 4.04 | 121 | 1 | 1 |

The S and M runs were submitted again because the first submissions (`ba180929…`, `c76a7346…`) failed at once on the workspace's GPU_8xH100 quota (8 nodes) and never ran.

- **A smaller model is barely cheaper to train.** XS costs 81% of M per step and S 95%, against FLOPs ratios of 0.29 and 0.60. At stage-1 table sizes (at most 1,024 rows, 100 features) the work that does not scale with the width dominates: the per-cell and per-row attention over short sequences, the prior's tables and the optimizer. This was the risk named in [v2-scaling.md](v2-scaling.md) (Cost).
- **M on 8 H100 is cheaper per step than v1's record on 32.** v1's stage 1 ran 0.71 s per step on 32 H100, which is 6.31 H100-h per 1K steps and was written by minipfn. Today's code takes 4.04 on 8 H100, which is 0.64 times as much. The cost per GPU on 32 H100 with today's code is not measured.

Corrected costs (8 H100, today's code):

| Item | Plan's estimate | Measured basis |
|---|---|---|
| One XS 30K-step run (phase 1B) | 56 H100-h | **98 H100-h** (12.3 h wall on 8 H100) |
| Phase 1B, 9-12 XS runs | 500-670 | **880-1,180** |
| Phase 1B at M size instead, 9-12 runs | — | 1,090-1,460 |
| Phase 2, 190 H100-h per size: steps | M 30K, S ≈ 50K, XS ≈ 100K | **M 47K, S 49K, XS 58K** |

- Phase 2 still compares the sizes at equal compute. With these costs, equal compute is close to equal steps: XS gets 1.24 times M's steps, not 3.3 times. A smaller size can then win only through its own capacity, not through more training.
- Phase 1B is a decision for the user, not a fixed correction. On XS it costs 1.8 times the plan's estimate. At M size it costs only 1.24 times as much as on XS, and it removes the plan's first risk (§7: "the small model may not stand for the large one"). Before choosing, one XS timing run at micro batch 8 (XS uses 30.8 GB at micro batch 4, so 8 fits) would show whether the accumulation overhead accounts for part of XS's cost. That run would take about 1.5 H100-h.

### 0.4: attention temperatures

Every attention that reads a variable number of keys multiplies q by a_h · log n, after an RMSNorm without gain on q and k (`LogScale` in `tabsil.models.column_row_transformer`). Here a_h is a learned scalar per head, initialized from U(0, 1) with mean 0.5. `training/scripts/v2_prior_diag/attention_temperatures.py` reads it from the weight files, with no forward pass.

Mean of a_h over heads, and the largest a_h, per layer group (`stage1-150k` / `stage2-10k` / `v1.1`):

| Group | Layers × heads | Mean a_h | Largest a_h |
|---|---|---|---|
| column summary (inducing points → train rows) | 6 × 4 | 0.310 / 0.309 / 0.309 | 0.718 / 0.717 / 0.717 |
| row stage (cells of a row, gated) | 6 × 4 | 0.225 / 0.225 / 0.225 | 0.386 / 0.387 / 0.387 |
| row ICL (rows → train rows) | 24 × 16 | 0.263 / 0.262 / 0.261 | 0.762 / 0.764 / 0.764 |
| classification retrieval decoder | 1 × 6 | 0.115 / 0.115 / 0.115 | 0.207 / 0.208 / 0.208 |

Per layer:

| Layer | `stage1-150k` mean (max) | `stage2-10k` mean (max) | `v1.1` mean (max) |
|---|---|---|---|
| row ICL 0 (the first in-context layer) | 0.191 (0.445) | 0.189 (0.449) | 0.189 (0.449) |
| row ICL 1 | 0.184 (0.404) | 0.182 (0.406) | 0.182 (0.406) |
| row ICL 2 | 0.195 (0.647) | 0.194 (0.645) | 0.194 (0.645) |
| row ICL 3 | 0.247 (0.507) | 0.244 (0.504) | 0.244 (0.505) |
| row ICL 4-11 | 0.211-0.277 (0.362-0.583) | 0.210-0.275 (0.358-0.583) | 0.210-0.275 (0.358-0.583) |
| row ICL 12-23 | 0.235-0.331 (0.316-0.762) | 0.234-0.330 (0.318-0.764) | 0.234-0.330 (0.319-0.764) |
| column summary 0-5 | 0.168-0.432 (0.225-0.718) | 0.166-0.432 (0.223-0.717) | 0.166-0.432 (0.223-0.717) |
| row stage 0-5 | 0.180-0.286 (0.236-0.386) | 0.181-0.286 (0.240-0.387) | 0.181-0.286 (0.240-0.387) |

The script's output holds every layer and head.

- **No temperature grew.** All 438 heads end with a_h ≤ 0.764, and the mean (0.26) is half the mean of the initial draw. In the first in-context layer, the mean is 0.19 and the largest head is 0.45. The growth Kumo-Tabular reports for BF16 (about 30-fold, which here would be a_h of about 15) is absent from all three checkpoints. Stages 2 and 3 move any a_h by at most 0.012.
- One model trains both tasks, so the temperatures are shared. A regression-only growth would still show here, and none does.
- **A cause to keep in mind.** TabSil trains `head_scale` with Muon and weight decay 0.01 (Muon on every parameter, plan §1.4). Kumo trains these scales with AdamW. Muon's update on a 1-d parameter has a fixed size per step, so it may hold the temperatures down. Phase 1B's OPTK arm moves the attention scales to AdamW, and that could let them grow.

**Recommendation: phase 1B gets no FP16 arm.** The BF16 failure Kumo guards against is not present in v1's training. In the OPTK runs, the first row-ICL layer's `head_scale` should be watched: the training runs already log `weight_norm/row_icl_log_scale/muon` per group, and the AdamW group would need the same. An FP16 regression arm is added only if those scales grow several-fold.

### 0.5: evaluation code

PR #31 adds `--suite tabarena-3split`, `--suite dev` and `training/scripts/paired_report.py`, and puts the weights and the applied inference settings in the arena cache key. The checks below were made on the runs of 0.1, 0.2 and 0.6.

**The development suite holds no TabArena data.** Of the 57 AMLB / CTR23 tasks, 9 have data that is also among TabArena's 51 datasets (`amlb.TABARENA_OVERLAP`). That leaves 48: 23 classification (13 binary and 10 multiclass) and 25 regression. The 57 were compared with the curated TabArena metadata of the `tabarena` package in three ways:

- Names, in lower case with only letters and digits (`training/scripts/dev_suite_overlap.py`). This gives 7 matches: blood-transfusion-service-center, qsar-biodeg, credit-g, churn, airfoil_self_noise, concrete_compressive_strength and QSAR_fish_toxicity.
- Sources, read from both lists. This gives 2 more. AMLB's dna is the UCI splice-junction data that TabArena has as splice. AMLB's wine-quality-white is the white-wine half of TabArena's wine_quality.
- Row counts of the remaining 48 against the 51. Two pairs have the same count and are different data: energy_efficiency and diabetes (768 rows), and grid_stability and Bank_Customer_Churn (10,000 rows). The next closest pair is kr-vs-kp (3,196 rows, chess endgames) and splice (3,190 rows, DNA).

Related tables with other rows stay in the suite. kc1 and pc4 are other NASA projects than TabArena's jm1. Australian is another credit table than credit-g. student_performance_por is another UCI table than students_dropout_and_academic_success. The plan's earlier count of 10 classification and 3 regression overlaps was a name match over the full AMLB lists, made before the list for these 57.

**Split 0 of the 3-split suite is the TabArena-Lite split.** For v1.1, `stage2-10k` and #14, the error on split 0 of the 3-split run equals the error of the TabArena-Lite run with the same weights and settings, on all 51 datasets and to the last digit. Splits 1 and 2 are the other two folds of repeat 0.

**The error has the same definition in both suites.** It is 1 − ROC AUC for binary tasks, the log loss for multiclass tasks and the RMSE for regression. In `results_per_split.csv` the `metric` column is `roc_auc`, `log_loss` and `rmse` for the three task types. One detail differs. The development suite clips probabilities at 1e-15 in the log loss. TabArena's scorer (`sklearn.metrics.log_loss`) clips at the machine epsilon of the predictions' type, 1.2e-7 for float32. The two differ only where a test row's true class gets a probability below these values.

**The paired report matches a second computation.** A separate script that uses only pandas and scipy gives the same G, CI, p and counts for every comparison of 0.1, 0.2 and 0.6. No dataset of any run is near the report's error floor of 1e-6: the smallest dataset error is 1.8e-4 (kr-vs-kp).

The report stops if two pooled suites share a dataset, because that dataset would count twice. For MLflow runs it prints a warning when a and b differ in the evaluation seed, the fold count, the context cap or the applied inference settings.

### 0.6: baselines

Every later comparison uses these runs as its control. The 3-split runs are TabArena with splits 0-2 of each dataset (`--suite tabarena-3split`, 153 splits, 8 H100) and have 0% imputed splits. The dev runs are the development suite (`--suite dev`: 23 classification and 25 regression tasks × 10 folds, 1 H100). A failed fold stops a dev run, and all of them finished. Every run logs `inference_classification` and `inference_regression` with 8 members and the numeric-cycle kinds.

| Weights | Context | TabArena 3-split run | Elo | Development suite run |
|---|---|---|---|---|
| v1 (`hf://aiden-jeon/tabsil@v1.1`) | full | `tabsil-p0-v1-3split` (`98c91ea0fe5342a4927698f4da85ca4b`) | 1781.9 | `tabsil-p0-v1-dev` (`02a24d08adc24dc89b9196561948024e`) |
| `stage2-10k` | full | `tabsil-p0-stage2-3split` (`fa88e6df7c9149b785ce72cc540a084c`) | 1769.4 | `tabsil-p0-stage2-dev` (`799d9184c15e4d94ba9596f2c0e1c769`) |
| #14 (`runs:/277916ec…/model/final.pt`) | full | `tabsil-p0-prior14-3split` (`facabb77551d4d16832f464df491125f`) | 1804.2 | `tabsil-p0-prior14-dev` (`f12907e66ccf45f0a14894b271edcad7`) |
| `stage1-150k` | full | `tabsil-p0-stage1-3split` (`3654b35b686b4bafba84747aeb2b189d`) | 1443.4 | `tabsil-p0-stage1-dev` (`687401da634643f2aa77492dacd02070`) |
| `stage1-150k` | 1,024 rows | `tabsil-p0-stage1-cap1024-3split` (`151d71dd3efd4cb18bb6b7b6e7ff145a`) | 1222.5 | `tabsil-p0-stage1-cap1024-dev` (`089d674c6cb24c4ba19f6f29becd3939`) |

Paired comparisons (G (95% CI), p, better / worse). "Pooled" puts the 51 + 48 = 99 datasets together.

| b against a | TabArena 3-split | Development suite | Pooled | Pooled classification / regression |
|---|---|---|---|---|
| #14 against `stage2-10k` | 0.986 (0.963-0.999), p 0.07, 29 / 22 | 1.000 (0.994-1.006), p 0.26, 19 / 29 | 0.993 (0.980-1.001), p 0.66 | 0.991 / 0.997 |
| v1 against `stage2-10k` (stage 3) | 1.000 (0.999-1.001), 27 / 24 | 0.999 (0.997-1.001), 25 / 23 | 1.000 (0.998-1.001) | 1.000 / 0.999 |
| `stage2-10k` against `stage1-150k` | 0.895 (0.835-0.949), 46 / 5 | 0.912 (0.861-0.956), 36 / 12 | 0.903 (0.864-0.937) | 0.970 / 0.805 |
| `stage1-150k` capped against full | 1.112 (1.015-1.229) | 1.182 (1.038-1.376) | 1.146 (1.055-1.251) | 1.357 / 0.873 |

- **#14's prior gain does not reach the development suite.** On TabArena's 153 splits it is 0.986. Polish alone is 0.58, and without the four datasets named in advance G is 0.998. On the 48 held-out datasets it is 1.000, and classification there is slightly worse (1.005, 7 better and 16 worse). Pooled it is 0.993 with p 0.66. Under the plan's branch rule (§3: CI below 1), #14's prior would not be adopted on this evidence. Phase 1A's N10 criterion reads TabArena only (G ≤ 0.995 and the four named datasets ≤ 0.97). It should be read next to the development suite, which says the gain sits in a few TabArena datasets.
- **Stage 3 changes nothing** on either suite (1.000 and 0.999), as on TabArena-Lite (§1.1).
- **Stage 2's gain is real on held-out data**: 0.912 on the development suite against 0.895 on TabArena, mostly in regression (0.861 and 0.707).
- **The cap helps stage-1 regression and hurts classification.** Capped against full context, stage-1 regression error is 0.873 (pooled), because stage 1 has not learned long contexts. Classification error is 1.357, because the capped model loses the extra rows. The 12 + 11 equal datasets have at most 1,024 training rows. So capped runs are compared only with capped runs.

## Phase 0 summary

| Step | Result |
|---|---|
| 0.1 | #14 + 0.1.1 is 1808 Elo on TabArena-Lite, and the prior and the inference gains add up (0.985 against `stage2-10k` 0.1.0). On TabArena with 3 splits the prior gives 0.986, and on the development suite 1.000 |
| 0.2 | §1.2's reading holds: capped `stage1-150k` against capped v1 is 1.003 overall and 1.005 in regression. Phase 1B's capped readout stands |
| 0.3 | XS costs 98 H100-h per 30K steps on 8 H100 (the plan said 56), S 115 and M 121. Phase 1B costs 880-1,180 H100-h on XS. Phase 2 at 190 H100-h per size is M 47K, S 49K and XS 58K steps |
| 0.4 | No attention temperature grew (largest a_h 0.76, first in-context layer mean 0.19). Phase 1B gets no FP16 arm; watch the scales in OPTK |
| 0.5 | PR #31: `--suite tabarena-3split`, `--suite dev`, `training/scripts/paired_report.py`, weights and applied inference in the arena cache key. The development suite's 48 datasets hold no TabArena data |
| 0.6 | The baselines above |

**Phase 1B.** The readout, the controls and the FP16 question are settled, so phase 1B can start as designed except for its cost. On XS it costs 1.8 times the plan's estimate, and at M size only 1.24 times the XS cost. The size of the 1B model (XS or M) is a decision to make before the runs. One XS timing run at micro batch 8 (about 1.5 H100-h) would make the XS number firmer.

**Phase 1A.** The development suite does not confirm #14's prior gain. N10 and R10 should be judged on the pooled report, not on TabArena alone.

## GPU cost

Measured on 2026-11-27 with short timing jobs, to redesign phases 2 to 5 of the plan (plan §4, "Cost basis for phases 2 to 5"). No model was trained for a result. Every job is 300 steps of stage 1 or 110 steps of stage 2 on 8 H100, or a one-GPU job of 40 to 2,000 steps, and only its timing is read. Together they used 28.9 H100-h ("Cost of the measurements").

- **Code and settings.** The training code of dev-v2 67c0c94, unchanged, at batch 256 with the speed options of the 0.3 timing runs (`TABSIL_COMPILE=1 TABSIL_TF32=1 TABSIL_MUON_DISTRIBUTED=all TABSIL_NUM_WORKERS=16`) and `TABSIL_RANK_TIMING=1`. Stage 2 starts from `aiden-jeon/tabsil@stage1-150k` on `graph_scm_hicard_missing`, as N10.
- **Scripts.** `training/scripts/v2_eff/`. `step_probe.py` runs `train.py` in its own process and also times the optimizer step, the clipping, each GPU's data wait and the pauses of Python's garbage collector. `step_breakdown.py` prints the breakdowns below from a run's records.
- **What is read.** Stage 1: steps 100 to 299, because the first 100 hold the compilation. Stage 2 on 8 H100: steps 20 to 109. A one-GPU run uses batch 32 (stage 1) or batch 8 (stage 2), which gives its GPU the 8 micro batches per step that a GPU has at batch 256 on 8 H100 (stage 1) or on 32 H100 (stage 2).
- **Sizes.** M is v1's size (221.5M parameters), S is `TABSIL_EMSIZE=192` (125.1M) and XS is `TABSIL_EMSIZE=128` (56.0M).
- **Units.** GPU-seconds per table is the step time × GPUs / 256. H100-h per 1K steps is the step time × GPUs × 1000 / 3600. Neither holds the start of a run, validation or checkpoints, which add 3-4% in the phase 1B runs (1.85-1.88 s per step over the whole run at a logged 1.80).

### Runs

Stage 1, prior `graph_scm`. Utilization and power are means over the GPUs during the timed steps. An H100 draws up to 700 W.

| Run | GPUs | Size | Micro batch × accumulation | s / step | GPU-s per table | H100-h per 1K steps | GPU memory | Utilization | Power |
|---|---|---|---|---|---|---|---|---|---|
| `tabsil-v2-eff-s1-m-mb4` (`7f3d817711f8474f96b9a69900bd903b`) | 8 | M | 4 × 8 | 1.783 | 0.0557 | 3.96 | 60.9 GB | 85% | 408 W |
| `tabsil-v2-eff-s1-m-mb4-probe` (`39dbb037c0fc4a86be4d7d37f4354046`), steps 100-199 | 8 | M | 4 × 8 | 1.798 | 0.0562 | 4.00 | 60.9 GB | 85% | 431 W |
| the same run, steps 200-299, after `gc.freeze()` | 8 | M | 4 × 8 | 1.716 | 0.0536 | 3.81 | | | |
| `tabsil-v2-eff-s1-m-mb8` (`f360ede4bdb948d2becb1d4c70accad4`) | 8 | M | 8 × 4 | out of memory | | | 76.6 GiB allocated of 79.2 | | |
| `tabsil-p0-timing-s` (0.3, `3a846b768b1a4d85a32d273f64847c55`) | 8 | S | 4 × 8 | 1.742 | 0.0544 | 3.87 | 45.2 GB | | |
| `tabsil-v2-eff-s1-s-mb8` (`9a73d59260054e9d896cd5c94829017d`) | 8 | S | 8 × 4 | out of memory | | | 76.9 GiB allocated of 79.2 | | |
| `tabsil-v2-eff-s1-xs-mb4` (`f89228d9f1784d2fb24d4d0ea80d8c51`) | 8 | XS | 4 × 8 | 1.485 | 0.0464 | 3.30 | 30.8 GB | 69% | 298 W |
| `tabsil-v2-eff-s1-xs-mb8` (`60e1cd2dec0140ae81ea56f48fffebbf`) | 8 | XS | 8 × 4 | 1.286 | 0.0402 | 2.86 | 60.3 GB | 80% | 317 W |
| `tabsil-v2-eff-s1-xs-mb8-pf16` (`2d73b26d771346da8ad742dea737baed`), `TABSIL_PREFETCH_FACTOR=16`, steps 100-199 | 8 | XS | 8 × 4 | 1.179 | 0.0369 | 2.62 | 60.3 GB | 76% | 357 W |
| the same run, steps 200-299, after `gc.freeze()` | 8 | XS | 8 × 4 | 1.147 | 0.0359 | 2.55 | | | |
| `tabsil-v2-eff-s1-m-mb4-1gpu` (`a56e5525096d41a88bb34527d9c9516f`) | 1 | M | 4 × 8 (batch 32) | 1.688 | 0.0527 | | 57.5 GB | 59% | 408 W |

Stage 2, size M, prior `graph_scm_hicard_missing`. "Frozen" is the part of a run after `gc.freeze()` (see "Collector pauses").

| Run | GPUs | Micro batch × accumulation | s / step | GPU-s per table | H100-h per 1K steps | GPU memory | Utilization | Power |
|---|---|---|---|---|---|---|---|---|
| N10 (phase 1A, `520d88d73cdc4e5ea90cfd27425be00d`) | 32 | 1 × 8 | 3.30 | 0.413 | 29.3 | 31.6 GB | 84% | 314 W |
| `tabsil-v2-eff-s2-m-mb1` (`bf797ac1f97047ec8a76849336322c02`), steps 20-59 | 8 | 1 × 32 | 9.69 | 0.303 | 21.5 | 30.1 GB | 75% | 346 W |
| the same run, steps 60-109, frozen | 8 | 1 × 32 | 8.72 | 0.272 | 19.4 | | | |
| `tabsil-v2-eff-s2-m-mb2` (`8b9545538dbd4bf5b08f1ff4f88072e8`), steps 20-59 | 8 | 2 × 16 | 7.93 | 0.248 | 17.6 | 56.4 GB | 90% | 452 W |
| the same run, steps 60-109, frozen | 8 | 2 × 16 | 7.25 | 0.227 | 16.1 | | | |
| `tabsil-v2-eff-s2-m-mb2-pf16` (`adfefd0fcbf54ec9b4bf869a377617ad`), `TABSIL_PREFETCH_FACTOR=16`, frozen from step 20 | 8 | 2 × 16 | 7.36 | 0.230 | 16.3 | 56.4 GB | 94% | 491 W |
| `tabsil-v2-eff-s2-m-mb1-1gpu` (`b1272880043645f0a3432e0fbc91accc`) | 1 | 1 × 8 (batch 8) | 1.879 | 0.235 | | 28.1 GB | 82% | 458 W |
| `tabsil-v2-eff-s2-m-mb2-1gpu` (`92290b3eea294ee6bc89e28df0c924bc`) | 1 | 2 × 4 (batch 8) | 1.484 | 0.186 | | 58.4 GB | 85% | 536 W |
| `tabsil-v2-eff-s2-xs-mb1-1gpu` (`1d43de23aed3403fa58e8493c5457176`), size XS | 1 | 1 × 8 (batch 8) | 1.632 | 0.204 | | 14.3 GB | 47% | 277 W |
| `tabsil-v2-eff-s2-xs-mb2-1gpu` (`0d142cbfab9a4eeca93531f39433eb8e`), size XS | 1 | 2 × 4 (batch 8) | 1.180 | 0.147 | | 30.5 GB | 58% | 353 W |

The XS stage-2 runs take their weights from the 300-step run `tabsil-p0-timing-xs`, so only their timing means anything.

Stage 3, size M, one GPU, 150 steps at batch 8 from `aiden-jeon/tabsil@stage2-10k` on `graph_scm_hicard_missing` (steps 40 to 149):

| Run | GPUs | Micro batch × accumulation | s / step | GPU-s per table | GPU memory | Utilization | Power |
|---|---|---|---|---|---|---|---|
| `tabsil-v2-eff-s3-m-mb1-1gpu` (`5df481cb6c75457e8de45e82f09d0024`) | 1 | 1 × 8 (batch 8) | 7.25 | 0.906 | 50.1 GB | 78% | 561 W |
| v1's stage 3 (plan §1, row 11: 2K steps in 9.0 h, older code) | 32 | 1 × 8 | 16.2 | 2.03 | | | |

Memory checks of stage 2 at micro batch 2, one GPU, every table at one row count (`TABSIL_MIN_ROWS` = `TABSIL_MAX_ROWS`), 40 steps:

| Run | Rows | Activation checkpointing | Largest allocation | Largest use seen by the driver |
|---|---|---|---|---|
| `tabsil-v2-eff-s2-m-mb2-rows10240-1gpu` (`fb21384235a845ed894a38674aed16f6`) | 10,240 (the largest stage-2 table) | on | 58.7 GB | 68.1 GB |
| `tabsil-v2-eff-s2-m-mb2-rows2048-1gpu` (`2734a67c0c954e40b146652abbb22ccf`) | 2,048 (the largest table without it) | off | 54.0 GB | 54.7 GB |

`tabsil-v2-eff-s2-m-pertable-1gpu` (`f4c22b5c5627474bb763934e5b9cbb14`) runs stage 2 at batch 1 for 2,000 steps, so that every step is one table. It gives the time of a table by its size ("GPUs per run").

### Where a stage-1 step goes

M on 8 H100 at micro batch 4, in milliseconds per step. Forward is the mean over the GPUs. Backward is that of the GPU that did not wait in the gradient all-reduce, and "waiting" is what the other GPUs' backward holds beyond it.

| Part | M (`39dbb037…`, steps 100-199) | M after `gc.freeze()` (steps 200-299) | XS, micro batch 4 (`f89228d9…`) | XS, micro batch 8 (`60e1cd2d…`) | M on one GPU (`a56e5525…`) |
|---|---|---|---|---|---|
| Forward | 500 (27.8%) | 489 | 475 (32.0%) | 295 (22.9%) | 477 |
| Backward | 1,014 (56.4%) | 1,034 | 705 (47.4%) | 595 (46.3%) | 897 |
| Waiting for the slowest GPU | 147 (8.2%) | 55 | 179 (12.0%) | 229 (17.8%) | 0 |
| Optimizer step (Muon, split over the GPUs) | 122 (6.8%) | 115 | | | |
| Gradient clipping | 6.7 (0.4%) | 5.4 | | | |
| Per-group norms (rank 0) | 3.0 (0.2%) | 3.5 | | | |
| Data wait | 1.3 (0.1%) | 1.4 | | | |
| Other (loss all-reduce, logging) | 5.3 (0.3%) | 12.9 | | | |
| Optimizer step to other, together | 138 (7.7%) | 139 | 127 (8.5%) | 167 (13.0%) | 314 |
| Step | 1,798 | 1,716 | 1,486 | 1,286 | 1,688 |

- **Forward and backward are 84% of the step**, the optimizer step 7% and waiting 8%. The data wait is nothing at this size (1.3 ms).
- **The first M run agrees**: `7f3d8177…` has 495, 1,013, 147 and 128 ms for 1,783.
- **Muon costs 122 ms a step on 8 H100.** On one GPU, where nothing is split, the optimizer step is 243 ms, and it is 243 ms for XS too (the one-GPU stage-2 runs `92290b3e…` and `1d43de23…`). On 8 H100 XS's optimizer step is 131-137 ms (`2d73b26d…`). So its time is not the Newton-Schulz arithmetic, which shrinks with the size.

### Why a quarter of the parameters costs 83%

XS has 0.25 times M's parameters and 0.29 times its FLOPs, and its step takes 0.83 times as long at the same micro batch (1,486 against 1,783-1,798 ms).

- **The forward takes the same time at both sizes**: 475 against 500 ms, about 60 ms per micro batch.
- **The forward does not depend on the table either.** On one GPU, where no waiting hides it, the forward of a step is 409 ms + 13 ms per 1M cells + 58 ms per 1K train rows, and the two explain 2% of its variance, over steps of 0.6M to 2.5M cells. The backward is 447 ms + 231 ms per 1M cells + 108 ms per 1K train rows (R² 0.81).
- So the forward is bound by the cost of issuing its operations, not by the GPU's arithmetic, and a narrower model does not shorten it. This was not profiled: which part of the forward holds the time is not known.
- **The optimizer step and the rest are also the same** (127 against 128-138 ms).
- **Only the backward shrinks**, to 0.70 (705 against 1,014 ms).
- **A larger micro batch spreads the forward's fixed cost.** At micro batch 8, XS's forward is 74 ms per micro batch of 8 tables, against 59 ms for 4 tables. Its step drops by 13% (1,286 ms).
- **Utilization overstates how full the GPU is.** M shows 85% utilization at 408-431 W of 700, XS 69% at 298 W. The GPU is busy with many short operations.

### Micro batch: what fits and what it saves

Stage 1 (8 H100, batch 256, so the micro batch can be 4, 8, 16 or 32):

| Size | Micro batch 4 | Micro batch 8 | Cheapest setting |
|---|---|---|---|
| M | 1.78-1.80 s, 60.9 GB | out of memory | micro batch 4: 3.96-4.00 H100-h per 1K steps |
| S | 1.74 s, 45.2 GB | out of memory | micro batch 4: 3.87 |
| XS | 1.49 s, 30.8 GB | 1.29 s, 60.3 GB (16 cannot fit); 1.18 s with `TABSIL_PREFETCH_FACTOR=16` | micro batch 8 with prefetch 16: 2.62 |

- **M and S stay at micro batch 4.** At 8, both fail in the first steps with 76.6-76.9 GiB allocated.
- **XS at its cheapest setting is 0.66 of M's cost per step** (2.62 against 3.98), against 0.83 at micro batch 4 and a FLOPs ratio of 0.29.
- **XS at micro batch 8 outruns the data loader.** In 14 of 100 classification steps one GPU waits for its tables (1.6 s on average, up to 4.0 s) and the other seven wait for it. Rank 0's `data_wait_frac` shows 0.1%. A micro batch of 8 tables takes a prior worker twice as long to build, and the loader hands micro batches over in order. With `TABSIL_PREFETCH_FACTOR=16` the stalls are gone (the longest data wait among the GPUs is 1 ms per step) and the step is 1.18 s instead of 1.29.

Stage 2 (M, micro batch 1 against 2):

| | Micro batch 1 | Micro batch 2 | Ratio |
|---|---|---|---|
| One GPU, forward + backward per table | 203 ms | 152 ms | 0.75 |
| One GPU, GPU-seconds per table | 0.235 | 0.186 | 0.79 |
| 8 H100, H100-h per 1K steps | 21.5 | 17.6 | 0.82 |
| 8 H100, after `gc.freeze()` | 19.4 | 16.1 | 0.83 |
| GPU memory | 30.1-31.6 GB | 56.4-58.7 GB | |

- **Micro batch 2 fits and saves 17-21%.** Its largest allocation is 58.7 GB with every table at 10,240 rows, and 54.0 GB at 2,048 rows, the largest table without activation checkpointing. The feature count of these tables is drawn from 1-100 and the 160 micro batches of each check include draws near 100. N10's largest over 10K steps was 31.6 GB at micro batch 1, so about 60 GB is expected over a full run. An H100 has 85 GB (79.2 GiB).
- **Micro batch 4 does not fit** at v1's size (twice 58.7 GB).
- **The cheapest stage-2 setting measured** is micro batch 2 with `TABSIL_PREFETCH_FACTOR=16` and the collector frozen: 7.36 s per step, 16.3 H100-h per 1K steps (`adfefd0f…`), against 29.3 for N10's setup. 12% of that step is still waiting for the slowest GPU (889 ms), now from the table sizes alone.
- **XS in stage 2 costs 0.87 of M at micro batch 1 and 0.79 at micro batch 2** (0.204 against 0.235 and 0.147 against 0.186 GPU-seconds per table on one GPU), with a FLOPs ratio of 0.37 at these table sizes. At micro batch 1 its forward is as long as M's (501 against 535 ms per 8 tables) and its backward 0.76 (827 against 1,092 ms). XS uses 30.5 GB at micro batch 2, so micro batch 4 would fit. It was not run.

### Collector pauses

Python's garbage collector stops each GPU's process for a full collection at intervals. The other GPUs then wait in the gradient all-reduce.

| | Stage 1, M, 8 H100 | Stage 2, M, micro batch 1, 8 H100 | Stage 2, M, micro batch 2, 8 H100 |
|---|---|---|---|
| Length of a full collection, mean (longest) | 0.69 s (0.80) | 1.16 s (1.33) | 0.94 s (1.25) |
| Micro batches of one GPU between two | 267 | 177 | 135 |
| Tracked objects per process | 1.2-1.4M | 2.0-2.1M | 1.6-1.7M |
| Steps in which a GPU pauses for more than 0.3 s | 18% | 82% | 68% |
| s / step as the code is → after `gc.freeze()` | 1.798 → 1.716 (−4.6%) | 9.69 → 8.72 (−10.0%) | 7.93 → 7.25 (−8.6%) |

- **What is collected is not garbage.** The collector walks the objects that live for the whole run: the model, the optimizer state and the compiled graphs. `gc.freeze()` moves the objects that exist at that point out of its reach. The probe calls it once, at step 200 (stage 1) or 60 (stage 2), and the two halves of the run are compared.
- **The stage-1 pauses are regular**: each GPU pauses about every 34 steps, and the forward of the paused GPU is 0.65 s longer in that step (`7f3d8177…`, steps 100-299).
- **The stage-2 numbers are less exact.** The two halves of a run see different tables, and a 10-step window varies by ±1 s. The micro-batch-1 run also has data stalls in both halves (next section). The waiting itself falls from 2,016 to 1,054 ms (micro batch 1) and from 1,602 to 911 ms (micro batch 2).
- **Stage 2 needs more than one freeze.** After the freeze, full collections go on over the objects made since (0.17-0.23 s each, 4% of steps above 0.3 s), because stage 2 keeps compiling for new table sizes. Freezing again at every log step was not measured.
- **The training result does not depend on the collector.**
- **A run on more GPUs loses more**, because a pause on any GPU stops all of them and each GPU has fewer micro batches per step. The estimates are in "GPUs per run".

### Data stalls on one node in stage 2

At micro batch 1 on 8 H100, the longest wait of a GPU for its tables is 1.14 s per step on average (0.89 s after the freeze), and 17% of steps have a wait above 0.3 s. The other GPUs wait for that GPU unless it would have finished early anyway. Rank 0 logs a wait of 0-5%. At micro batch 2 the longest wait is 0.17-0.28 s.

The cause is the one of R10's slowness (phase 1A in PR #33, "Why R10 was slow"): the loader hands tables over in request order, and a table is requested 32 micro batches ahead per task. At micro batch 1 on 8 H100 that head start is one step of the task, about 19 s. `graph_scm_hicard_missing` has tables of 10 to 23 s (0.33% of them over 10 s), and a worker builds its four in a row. On 32 H100 the same 32 tables are 4 steps of the task (26 s), and N10 shows little of it (0.3% on rank 0). Micro batch 2 doubles the head start. With `TABSIL_PREFETCH_FACTOR=16` at micro batch 2 the longest data wait among the GPUs is 3 ms per step (`adfefd0f…`).

### GPUs per run

The same work costs more GPU time on more GPUs.

| Stage 2, M, micro batch 1 | GPU-seconds per table | H100-h per 1K steps |
|---|---|---|
| 1 GPU | 0.235 | 16.7 |
| 8 H100 | 0.303 | 21.5 |
| 32 H100 (N10) | 0.413 | 29.3 |

- **Every GPU waits for the slowest one at every step.** A step ends in the gradient all-reduce, when the GPU with the most work is done. On 32 H100 a GPU has 8 tables per step, and their sizes differ widely between GPUs. On 8 H100 it has 32, which evens out.
- **A table's time follows from its size.** In the batch-1 run, a table takes 73 ms + 498 ms per 1M cells + 78 ms per 10K rows + 122 ms per (10K rows)², with R² 0.98 and a residual sd of 18 ms over 1,700 tables. The mean is 196 ms, the 99th percentile 626 ms. Tables above 2,048 rows are 51% of the tables and 74% of the time.
- **Stage 1 loses less to sizes**, since every table has 1,024 rows: after the freeze, waiting is 55 ms of 1,716 (3.2%) on 8 H100.
- **Stage 3 loses the most.** Its tables have 400 to 60,000 rows. One GPU needs 0.906 GPU-seconds per table, and v1's run on 32 H100 needed 2.03. Stage 3 was not timed on 8 H100. If one node costs 1.3 to 1.5 times the one-GPU time, as in stage 2 (1.29), it is 85 to 100 H100-h per 1K steps against 144.

Estimates for other GPU counts. The simulation draws each GPU's tables (stage 2) or micro batches (stage 1) from the one-GPU records, ends a step with the slowest GPU, and adds the collector pauses at their measured rate and the measured rest of the step. It is scaled to the 8-H100 runs. Against the other measurements it gives 8.21 s for 8.55 s (stage 2 on 8 H100 without its data stalls, −4%), 3.03 s for N10's 3.30 s (−8%) and, for stage 1, 0.71 s for v1's 0.71 s on 32 H100 (older code). These are estimates, not measurements, and probably a few percent low for more than one node.

| | 8 H100 | 16 H100 | 32 H100 |
|---|---|---|---|
| Stage 1, M: s / step, as the code is | 1.79 (measured) | about 1.08 | about 0.71 (v1: 0.71) |
| Stage 1, M: H100-h per 1K steps | 3.98 | about 4.8 | about 6.3 |
| Stage 1, M, collector frozen: s / step | 1.72 (measured) | about 0.98 | about 0.60 |
| Stage 1, M, collector frozen: H100-h per 1K steps | 3.81 | about 4.4 | about 5.3 |
| Stage 2, M, micro batch 1: s / step, as the code is | 9.69 (measured, with 1.1 s of data stalls) | about 5.5 | 3.30 (measured) |
| Stage 2, M, micro batch 1: H100-h per 1K steps | 21.5 | about 24 | 29.3 |
| Stage 2, simulated, collector frozen: H100-h per 1K steps | 17.2 | 19.5 | 23.6 |
| Stage 2, simulated, collector frozen and equal work per GPU: H100-h per 1K steps | 15.1 | 15.6 | 16.6 |

"Equal work per GPU" gives every GPU of a step the same predicted time, as a change to the prior's table sizes would (see "Changes that need code"). It was not run.

### What a larger micro batch changes

From the code (`tabsil_train.priors.graph_scm`, `priors.pipeline`, `training.loop`, `training.data`, `tabsil.models.column_row_transformer`). The batch stays 256, so a larger micro batch means fewer accumulation steps.

- **Fewer table sizes per step.** The prior draws the rows, the train-row share and the feature count once per micro batch (`GraphSCMPrior.sample_batch`), and then each table on its own (its graph, class count and missing values). At batch 256 a step has 64 sizes at micro batch 4 and 32 at 8 (stage 1), 256 at micro batch 1 and 128 at 2 (stage 2). The size distribution of a table is unchanged. The gradient of a step averages over fewer sizes, which is the one statistical difference.
- **No padding and no shared statistics.** The tables of a micro batch have the same size, so nothing is padded. The model standardizes each table with its own train rows, attention stays inside a table, and there is no dropout or batch norm. A table's loss does not depend on the other tables of its micro batch.
- **The same weights.** The loss is the mean over the eval rows of a micro batch, whose tables have equal row counts. Each micro batch's loss is divided by the accumulation count and DDP averages the GPUs, so every table has weight 1/256 at any micro batch. Clipping, the non-finite check and the optimizer see the gradient of the whole batch.
- **Other tables.** The prior's random stream is used per micro batch, and its seed holds the GPU's rank. A run at another micro batch or GPU count sees other tables at the same seed. Such runs compare as draws from one prior, not table by table.
- **The data loader's head start** is counted in micro batches, and a worker builds a micro batch in one piece. See the stalls above.
- **Memory** grows with the micro batch. Activation checkpointing depends on the row count only.

### Stage 3 by table size

No run. v1 (stage 3, 2K steps) against `stage2-10k` on the 0.6 baselines, by the training rows of a dataset:

| Training rows | TabArena 3-split | Development suite |
|---|---|---|
| all | 1.0000 (0.9988-1.0013), 51 datasets | 0.9991 (0.9969-1.0010), 48 |
| at most 2,500 | 0.9998 (0.9982-1.0014), 19 | 0.9973 (0.9940-1.0000), 26 |
| 2,500 to 10,000 | 1.0011 (0.9990-1.0037), 17 | 1.0011 (0.9989-1.0037), 22 |
| above 10,000 | 0.9990 (0.9970-1.0011), 15 | none: the largest has 9,000 |
| above 10,000, regression | 0.9977 (0.9949-1.0001), 5 | |
| above 20,000, regression | 0.9961 (0.9927-0.9998), 3 | |
| above 50,000 | 1.0003 (0.9965-1.0058), 4 | |

Stage 3 trains on tables of up to 60,000 rows. On the 15 TabArena datasets above 10K training rows it changes the error by 0.1%, within the CI. The only CI below 1 is that of 3 regression datasets (diamonds 0.993, physiochemical_protein 0.996, Food_Delivery_Time 1.000). This is below the noise between two stage-2 runs (ν 0.003, phase 1A).

### Changes that need code

None was made. Each is a proposal with what the measurements above say it would give.

| Change | Where | Expected effect | Evidence |
|---|---|---|---|
| `gc.collect()` and `gc.freeze()` on every GPU once the first log window is done, and again at every log step | `tabsil_train.training.loop`, about 3 lines | stage 1 on 8 H100 −4.6%, stage 2 on 8 H100 −9 to −10%. On 32 H100 about −15% (stage 1) and −12% (stage 2), estimated. Results are unchanged | measured with `TABSIL_EFF_GC=freeze:<step>`; a single freeze, not the repeated one |
| Equal work per GPU in a step: the step's table sizes drawn from one stream that all GPUs share and dealt out by predicted time | the prior's size draw (`GraphSCMPrior.sample_shape`) and `tabsil_train.training.data` | stage 2: the waiting that is left after the freeze, 12-13% of the step on 8 H100 and about 30% on 32 H100. With the freeze, 32 H100 would cost about 17-20 H100-h per 1K steps instead of 29.3 | simulation only. It changes which tables a run sees, not their distribution |
| The largest data wait over the GPUs as a logged metric | `tabsil_train.training.loop` | none on speed. A stall of 1 s per step shows as 0% today | "Data stalls" above |
| A profile of the forward | — | unknown, at most the forward's share (28-32% of a stage-1 step) | the forward takes 60 ms per micro batch at any width and feature count |

Two settings need no code: one node per run where the wall time allows it, and `TABSIL_PREFETCH_FACTOR=16` on one node in stage 2 and for XS at micro batch 8.

### Cost of the measurements

28.9 H100-h in 21 jobs, within the 40 that were set: 25.9 on one 8xH100 node (11 jobs, one at a time) and 3.0 on one-GPU nodes (10 jobs).

- 1.6 H100-h gave nothing: a one-GPU job that hung (0.4, see below) and the first 8-H100 run with the probe, which was stopped after 9 minutes because the probe did not yet upload its records (1.2). The two out-of-memory checks (2.0) were meant to fail.
- **A one-GPU stage-1 run from scratch needs `OMP_NUM_THREADS=1`.** torchrun sets it only for more than one process. Without it the process pool that draws the regression borders hangs (`tabsil-v2-eff-s1-m-mb4-1gpu`, first try, stopped after 23 minutes at 4% CPU). `timing.yaml` notes it.
- **Not measured**: any run on 16 or 32 H100 (one node at a time was allowed), stage 3 on 8 H100, S with the probe, and micro batch 4 for XS in stage 2.
