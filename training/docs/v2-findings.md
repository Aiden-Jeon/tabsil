# v2: what is established

Status: 2026-11-28. v2 work is paused. This page lists only what the v2 runs measured, with the source of each number, and what they left open. It has no plan and no recommendation. How the plans came about and what went wrong is in [v2-retrospective.md](v2-retrospective.md).

- **Elo** is TabArena-Lite as `training/scripts/eval.py` runs it: 51 datasets, one split each, each split fit once without bagging. It is not the official leaderboard's number.
- **G** is the paired error ratio of [v2-experiment-plan.md](v2-experiment-plan.md) §3: the geometric mean over datasets of one model's error over another's. Below 1 is better. Brackets hold the 95% bootstrap CI.
- **Pooled** is TabArena with 3 splits (51 datasets) and the development suite (48 datasets that are not in TabArena) together: 99 datasets.
- Every evaluation below had 0% imputed splits.

## Summary

- **No run of the two v2 plans made weights better than #14's run, which finished before either plan was written** (2026-11-25). It is v1's stage 1 with 10K steps of stage 2 on `graph_scm_hicard_missing`: Elo 1808 against 1768 for v1.1. Its repeat with another seed, N10, has 1788.
- **That gain sits in a few TabArena datasets.** The two seeds together are G 0.992 (0.980-1.000) pooled, 0.996 (0.988-1.002) on the development suite, and 0.997 pooled without polish_companies_bankruptcy.
- **Every other change that was measured is within the noise of its control**: Kumo's other stage-2 prior parts, stage 3, and two stage-1 recipe changes (YCTX, OPTK).
- **Training on task-shaped tables alone breaks TabArena-Lite**: 1808 falls to 1252 and 1234. Mixing them into ordinary training was not run.
- **GPU spent**: 1,870 H100-h on the v2 experiment plan and 40.0 on the plan for priors and tasks.

## 1. The weights

| Weights | What they are | Elo | Evaluation run |
|---|---|---|---|
| v1.1 (`aiden-jeon/tabsil@v1.1`) | the released v1 with tabsil 0.1.1's inference | 1768 | `138151a2…` |
| `stage2-10k` | v1 before stage 3, with 0.1.1's inference | 1762.4 | `a8b69941…` |
| #14's run (`277916ec05a3457aa6d07f2bc08a60c4`) | `stage1-150k` + 10K steps of stage 2 on `graph_scm_hicard_missing`, seed 0 | 1807.6 | `7cc31c48…` |
| N10 (`520d88d73cdc4e5ea90cfd27425be00d`) | the same, seed 1 | 1788.3 | `e2ff88f0…` |

Neither #14's run nor N10 is released.

## 2. The stage-2 prior `graph_scm_hicard_missing`

| Comparison | G | Datasets better / worse | Source ([results](v2-experiment-results.md)) |
|---|---|---|---|
| #14's run against v1.1, TabArena-Lite | 0.991 (0.974-1.000) | 28 / 23 | 0.1 |
| N10 against `stage2-10k`, TabArena-Lite | 0.992 (0.977-1.000) | 29 / 22 | N10 |
| both seeds against `stage2-10k`, TabArena 3-split | 0.988 (0.966-1.000) | | ν |
| both seeds against `stage2-10k`, development suite | 0.996 (0.988-1.002) | | ν |
| both seeds against `stage2-10k`, pooled | 0.992 (0.980-1.000) | 49 / 50 | ν |
| the same without polish_companies_bankruptcy | 0.997 | | ν |

- The three stage-2 runs from `stage1-150k` on the newer priors (#14's run, N10, R10) lower the error on polish_companies_bankruptcy by about 40% (0.581, 0.611, 0.591 on 3 splits) and on kddcup09_appetency by 3 to 4%. Without the four datasets named in advance, the ones the prior was made for, G on TabArena is 0.996 to 1.000 for the two hicard seeds.
- #14's run alone is 1.000 (0.994-1.006) on the development suite.
- The prior's gain and 0.1.1's inference gain add up: #14's run with 0.1.1's inference against `stage2-10k` with 0.1.0's is 0.985 (Elo 1727 to 1808).
- At TabSil's place in the pool, 1% of error is about 40 Elo (plan §1.1).

## 3. Changes that did not move the error

| Change | Runs | G against the control | In the results |
|---|---|---|---|
| `graph_scm_realism` in stage 2: Kumo's other prior parts on top of the hicard prior | R10, 10K steps | 1.003 (0.999-1.008) pooled, against the two hicard seeds | not adopted |
| Stage 3 (2K steps) | v1 against `stage2-10k` | 1.000 (0.998-1.001) pooled; 0.999 (0.997-1.001) on the 15 TabArena datasets above 10K training rows | no effect |
| YCTX: stage-1 targets standardized with the train rows only | 2 seeds of 30K steps from scratch, against 3 control seeds, context capped at 1,024 rows | 0.995 (0.990-1.000) pooled; against the control seed of the same number 0.993 and 1.001 | not resolved |
| OPTK: Kumo-Tabular's optimizer split | 2 seeds of 30K steps, as YCTX | 1.003 (0.999-1.008) pooled; its reference classification loss is above all three control seeds at every validation step | not resolved |

Stopped before they had a result:

- PS1 (prior parts for every stage, in stage 1): cancelled at steps 10,500 and 5,000.
- L20 (stage 2 for 20K steps on the hicard prior): cancelled at step 5,200 and not evaluated. Its `checkpoints/step_0005000.pt` is in MLflow run `893293ed7d8d419780be2d178b8cc713`.
- YCTX and OPTK together: cancelled at step 700.
- Not run: the model size at equal compute (phase 2), a new stage 1 (phase 3) and the stages after it (phases 4 and 5).

## 4. What a comparison can resolve

| Noise | Pooled | TabArena | Development suite | Source ([results](v2-experiment-results.md)) |
|---|---|---|---|---|
| two stage-2 runs from the same weights, seeds 0 and 1 (#14's run, N10) | 0.3% | 0.2% (Lite), 0.3% (3 splits) | 0.9% | ν |
| two stage-1 runs of one recipe, 30K steps, capped readout (C seeds 0 to 2) | 0.1% to 0.5% | 0.0% to 0.2% (3 splits) | 0.5% to 1.1% | "The noise of a stage-1 rerun" |

- On TabArena-Lite the two hicard seeds are 1808 and 1788.
- TabArena-Lite's 51 splits are split 0 of the 3-split suite, with identical errors, so the two are not independent evidence.

## 5. Stage 1 and the context length

- **The flat stage-1 Elo is a context-length effect.** Stage 1 has TabArena-Lite Elo 1429.2 after 150K steps and 1428.0 after 250K (two runs, [v2-scaling.md](v2-scaling.md)). With the context capped at 1,024 rows, `stage1-150k` is within 0.3% of the finished v1: G 1.003 (1.000-1.006), regression 1.005 (results 0.2). So stages 2 and 3 add nothing on contexts of 1,024 rows, and their gain is on longer contexts. The reading in [v2-scaling.md](v2-scaling.md), a mismatch between the prior and real data, does not hold.
- **Stage 2's gain over stage 1 holds on held-out data**: `stage2-10k` against `stage1-150k` is 0.903 (0.864-0.937) pooled, 0.912 on the development suite, and 0.805 for regression pooled (results 0.6).

## 6. v1 against other models

On the 51 TabArena-Lite splits, v1.1 against the public results of other models ([v2-experiment-plan.md](v2-experiment-plan.md) §1.3; compute from the Kumo-Tabular paper, Table 2). The references are 8-fold bagged and TabSil is fit once.

| Model | Parameters | Training H100-h | G, v1.1 / model | v1.1 better on |
|---|---|---|---|---|
| TabSil v1 | 221.5M | 1,555 | — | — |
| Kumo-Tabular-S | 27M | 1,573 | 0.996 | 26 of 51 |
| Kumo-Tabular-M | 61M | 7,186 | 1.047 | 14 |
| Kumo-Tabular-L | 214M | 23,148 | 1.070 | 10 |
| TabPFN-3.5 (default) | 219.0M | not published | 1.020 (1.004 without polish_companies_bankruptcy) | 19 |

At the same compute, Kumo-S with an eighth of the parameters is level with v1. The Kumo models ahead of v1 trained with 4.6 and 14.7 times Kumo-S's compute.

## 7. Task-shaped tables

Measured under the plan for priors and tasks, PRs #50, #51 and #52, which are not merged. The results are in [v2-prior-task-results.md](https://github.com/Aiden-Jeon/tabsil/blob/9933bd05df52127c3cb03293057b2a2d361200d3/training/docs/v2-prior-task-results.md). That plan took the task benchmarks (fev-bench, ADBench) as its goal by a misreading. The goal of v2 is TabArena-Lite, and it was measured on the fine-tuned weights only after the goal was corrected:

| Weights, after 2,000 steps on task-shaped tables alone (sorted-split regression tables, outlier tables) | Elo | G against the start, all 51 | Classification (38) | Regression (13) | Datasets better / worse |
|---|---|---|---|---|---|
| #14's run, the start, evaluated again | 1807.6 | 1 | 1 | 1 | — |
| pilot: 1 H100, batch 32 | 1252.4 | 1.220 (1.149-1.299) | 1.295 | 1.024 | 6 / 45 |
| probe: 8 H100, batch 256 | 1234.1 | 1.228 (1.157-1.306) | 1.309 | 1.020 | 5 / 46 |

- **A fine-tuning on a mix of ordinary and task-shaped tables was not run.** Its prior, `graph_scm_hicard_task_mix`, is registered on the branch of PR #52 and no run has used it.
- **On the task benchmarks** the same fine-tunings give fev-bench SQL Elo 1538 to 1582 (pilot) and 1519 (probe), and ADBench mean AUROC with reference rows 0.664 to 0.752 and 0.759. On the synthetic checks the pilot continues a linear trend (D1: MASE 12.5 to 0.66) and reference rows rank a dense group of outliers above the inliers at every contamination up to 20% (D3: AUROC 1.00).
- **Without training**, on v1.1's weights: the mean log density over 32 random contexts of 64 rows raises ADBench from 0.663 to 0.756 (IsolationForest 0.722). It is a wrapper in `training/scripts/v2_prior_diag` on that branch and not part of tabsil. No change of the forecasting recipe beat today's on fev-bench (SQL Elo 1535).
- **#14's run and N10 equal v1.1 on fev-bench, ADBench, imputation and treatment effects**, except that they read a missing cell as information. On the imputation suite's tables they are 14% better than v1.1 where another column's mask depends on the cell to impute in every row (`other_all`), and 9% worse where it does so only in the rows that observe the cell (`other_obs`).

## 8. Cost

- **Stage 2 at v1's size, batch 256, per 1K steps:** 29.3 H100-h on 32 H100 (N10), 21.5 on 8 H100 at micro batch 1, 16.2 on 8 H100 at micro batch 2 (L20's 5,200 steps). The same work costs more GPU time on more GPUs, because at every step each GPU waits for the one with the most work.
- **Stage 1, 30K steps on 8 H100:** 121 H100-h at v1's size (M), 115 for S (125M) and 98 for XS (56M) at micro batch 4. At its cheapest setting XS is 0.66 of M's cost per step. A smaller model is not much cheaper to train.
- **`gc.freeze`** (PR #40, on by default): stage 1 −6.7% and stage 2 at micro batch 2 −2.3% per step on 8 H100, with unchanged results.
- **`corrected_cat_sizes` slows the prior.** Categorical columns with up to 200 categories make rare tables that take tens of seconds to draw, and the in-order loader holds every GPU while it waits. R10 ran at 4.95 s per step against N10's 3.30 and cost 450 H100-h against 305. In stage 1, `TABSIL_PREFETCH_FACTOR=16` removed the stalls (phase 1B). No stage-2 run used it.
- **The workspace runs at most 8 `GPU_8xH100` nodes and 8 `GPU_1xH100` nodes at a time.** A request above that fails instead of waiting.

## 9. GPU spent

Counted from the job records.

| Work | H100-h |
|---|---|
| v2 experiment plan, 108 jobs from 2026-11-26 13:45 UTC on: phase 0 22, phase 1A 848 (N10 305, R10 450, L20 85, evaluations and a diagnosis 8), phase 1B 965, cost measurements 35 ([v2-retrospective.md](v2-retrospective.md)) | 1,870 |
| Plan for priors and tasks, 56 jobs on 2026-11-28: tier 1 11.6, tier 2 27.6, TabArena-Lite of the fine-tunings 0.9 | 40.0 |
| **Total** | **1,910** |

#14's run (306 H100-h, 2026-11-25) came before both and is not counted here.

## 10. Open

- Whether a longer stage 2 helps (L20).
- What PS1 does, whether YCTX and OPTK combine, and whether OPTK helps a longer stage 1. Kumo-Tabular trains stage 1 for 500K to 1M steps, and OPTK was tested at 30K.
- The model size at equal compute.
- What task-shaped tables mixed into ordinary training do to TabArena-Lite.
- Whether #14's weights or the small-context outlier score are released.

## Sources

- On dev-v2: [v2-experiment-plan.md](v2-experiment-plan.md), [v2-experiment-results.md](v2-experiment-results.md), [v2-retrospective.md](v2-retrospective.md), [v2-scaling.md](v2-scaling.md).
- PRs #50, #51 and #52, at `9933bd0`: [v2-prior-task-results.md](https://github.com/Aiden-Jeon/tabsil/blob/9933bd05df52127c3cb03293057b2a2d361200d3/training/docs/v2-prior-task-results.md), [v2-task-formulations.md](https://github.com/Aiden-Jeon/tabsil/blob/9933bd05df52127c3cb03293057b2a2d361200d3/training/docs/v2-task-formulations.md) (section 10 and "Checks and corrections"), [v2-prior-task-plan.md](https://github.com/Aiden-Jeon/tabsil/blob/9933bd05df52127c3cb03293057b2a2d361200d3/training/docs/v2-prior-task-plan.md).
