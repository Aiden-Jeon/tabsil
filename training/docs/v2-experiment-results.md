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
| 0.5 | PR #31: `--suite tabarena-3split`, `--suite dev`, `training/scripts/paired_report.py`, weights and applied inference in the arena cache key |
| 0.6 | The baselines above |

**Phase 1B.** The readout, the controls and the FP16 question are settled, so phase 1B can start as designed except for its cost. On XS it costs 1.8 times the plan's estimate, and at M size only 1.24 times the XS cost. The size of the 1B model (XS or M) is a decision to make before the runs. One XS timing run at micro batch 8 (about 1.5 H100-h) would make the XS number firmer.

**Phase 1A.** The development suite does not confirm #14's prior gain. N10 and R10 should be judged on the pooled report, not on TabArena alone.
