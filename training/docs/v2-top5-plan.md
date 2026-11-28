# v2 plan: TabArena-Lite top 5

Status: **plan, not approved.** Written on 2026-11-28 on the branch `Aiden-Jeon/v2-top5-plan`. Nothing was trained or evaluated for it and no GPU job was run: the per-dataset gap reads MLflow artifacts and the public leaderboard on a CPU, and the derivations are checked with NumPy. It is built to the twelve rules of [v2-retrospective.md](v2-retrospective.md) (§11 lists where each is met). Its evidence from the v2 runs is [v2-findings.md](https://github.com/Aiden-Jeon/tabsil/blob/3c97ccc5f408e350e542c4803bc71e7b7935c765/training/docs/v2-findings.md) (PR #53, open) with the results documents it cites. The derivations for the tasks are §11 of [v2-task-formulations.md](v2-task-formulations.md). An agent that did not write the plan read it before any run (§14). **No run of this plan starts before the user approves it and sets the budget, the stop threshold and whether the compute lever is in scope (§15).**

The request, 2026-11-28 (translated): "Plan again in line with the retrospective. More diverse priors, more diverse tasks. For the task part, derive in formulas how it would be done, as with the regression analysis. Your goal is to make the Tablite Elo top 5." The user approved this reading before the plan was written: the twelve rules, and only the facts of the findings as evidence; priors and tasks as means to raise TabArena-Lite, every run read on TabArena-Lite against a same-steps control; derivations in the style of formulations §2 step 7 and §10 for every task; rank 5 or better on the public TabArena-Lite board, models only.

How to read:

- **Measured**: a number from a run in the records. **Derived**, **checked**: from the formulations document; checked means a NumPy script reproduces it. **Estimated**: computed from measured numbers by a stated step; an estimate **by analogy** applies another model's measured change to TabSil's errors (§4.3). **Reported**: another group's number for another model. **Unknown**: no basis. **Not verified**: could not be checked.
- G is the paired error ratio of [v2-experiment-plan.md](v2-experiment-plan.md) §3 (below 1 is better). Pooled is TabArena with 3 splits plus the development suite (99 datasets).
- Elo has two units (§1.2). **Repo** is `eval.py`'s TabArena-Lite in the pool of the pinned `tabarena` package, the unit of the findings. **Live** is the public board, or a run's split-0 errors inserted into it and refit with TabArena's Elo method (`training/scripts/gap_report.py`, §4). This plan decides in the live unit.
- Costs are H100-hours (H100-h). A node is one `GPU_8xH100`; the workspace runs at most 8 such nodes and 8 `GPU_1xH100` nodes at a time, and a request above that fails instead of waiting.

## Summary

**The outlook.**

- **The bar.** Rank 5 on the live board is Kumo-Tabular-Medium at 1921 (§1). #14's weights, inserted into the live pool, are 1816.9 at rank 11. The distance is **+107 live Elo** (1924.3 in the same fit), and +144 from v1.1.
- **What the distance is in error.** Elo on this board counts wins per dataset, and TabSil sits in a crowd: on 35% of the (dataset, top-15 model) pairs the two errors are within 1% of each other. So the distance is small in size and large in breadth. Cutting every one of #14's 51 errors by 0.73% reaches rank 5. A change that lowers the error on four named datasets does almost nothing: of #14's +37.6 over v1.1, those four give 4.5 (§4.2).
- **What priors and tasks can give, by their evidence.** One prior change has a measured gain, and it is in hand (#14). Closing the whole gap to Kumo-Tabular-Medium on the datasets of one property, such as missing values or 21 to 100 features, gives at most +27 to +58; half of it gives +14 to +33 (§4.3). Kumo-Tabular's other prior parts moved none of these properties by 0.5% (R10, §4.4). The task-shaped tables built so far change nothing on TabArena in the limit, and the tasks that add targets per table are bounded by what more training gives (formulations §11). **On this evidence, priors and tasks alone do not reach the bar.**
- **What reaches it, by analogy.** A change as broad as one of Kumo-Tabular's compute steps: Kumo's S → M change applied to #14's errors gives +96 (rank 6), its M → L change +115 (rank 5) (§4.3). For TabSil that is Kumo-Tabular-Medium's tables per task at TabSil's size: 4.4 to 4.7 times #14's compute, about 3,640 to 3,830 H100-h and 19 to 20 days on one node (tier 3), +94 to +147 by the same analogy. Whether TabSil follows Kumo's slope is not measured. Item 2.1 measures it for 223 H100-h from weights that exist.
- **For the user to decide (§15):** whether the compute lever is in scope; the budget cap; the stop threshold. None is filled in.

| Tier | What is done | Adds, in live Elo over the tier below; evidence | H100-h | Runs if |
|---|---|---|---|---|
| 0 | Nothing. #14's weights | in hand: 1816.9, rank 11; +37 over v1.1 (the second seed of the same prior, N10, +21). Measured on split 0 and inserted into the live pool | 0 | — |
| 1 | No training: the gradient-noise statistics of #14's weights; the 250K-step stage 1 made loadable and read with the capped context | +0: measurements. They decide items 2.1 and 2.3 | about 4 | the user approves the plan |
| 2 | Stage 2 from the 250K-step stage 1 (2.1); by the user's choice, one prior-and-task branch (2.2); gated, a masked-cell branch (2.3) | 2.1: +0 to about +43 (estimated by analogy: a share of Kumo's compute step in proportion to the compute, §4.3). 2.2: +0 to about +30 (estimated). 2.3: unknown | 223 for 2.1; +223 each for 2.2 and 2.3 | 1.2 shows the 250K stage 1 ahead (2.1); the user asks for 2.2; 1.1 opens 2.3 |
| 3 | The compute lever: Kumo-Tabular-Medium's tables per task at TabSil's size, 500K / 40K / 12K steps against v1's 150K / 10K / 2K, one node per stage | +94 to +147 over #14 (estimated by analogy: one S → M step of Kumo, +94 to +97, rank 6; 1.3 of its M → L step, +141 to +147, rank 4), so about the bar. Without stage 3, up to 23 less (reported by Kumo) | 3,640 to 3,830; 2,630 without stage 3 | the user puts it in scope, and 2.1 reads a slope at reading point R2 |

- **Reaching the bar on the evidence needs tier 3**, and tier 3 lands about at the bar: rank 4 to 6 by analogy, inside the board's CIs of about ±100 to ±200 (§1.4). The plan's margin is 25 Elo above Kumo-Tabular-Medium (§1.4).
- **The priors and tasks the request asked for are planned in full** (§5, §6): every candidate with its property, evidence, expected Elo and cost. The ones whose expected effect is below what a run resolves are listed and not run (rule 5); item 2.2 bundles the best of them for the user to run by choice.

## 1. The bar

### 1.1 The live board

TabArena-Lite, models only, imputation yes, read on 2026-11-28 from the Space `TabArena/leaderboard` at commit `a525fe89b5824325055ad58ff3de5590047a2877` (its current main, last modified 2026-11-27 23:36 UTC). The all-splits column is the same pool on every split.

| Rank | Model | Lite Elo (95% CI) | All splits |
|---|---|---|---|
| 1 | Causilo 1.5 | 2137 (+224/−135) | 2026 |
| 2 | Kumo-Tabular (L) | 2064 (+181/−111) | 1973 |
| 3 | LimiX-2-1016 | 2052 (+184/−110) | 2004 |
| 4 | Seldon | 1947 (+192/−135) | 1908 |
| 5 | **Kumo-Tabular-Medium** | **1921 (+148/−97)** | 1910 |
| 6 | TabPFN-3.5 | 1874 (+146/−96) | 1868 |
| 7 | Lattice | 1842 | 1798 |
| 16 | TabSil (default): v1 with tabsil 0.1.0's inference | 1745 (+91/−72) | 1726 |

Every one of these entries is a "(default)" entry, verified, with 0% imputed. The models-only table also holds "(tuned)" and "(tuned + ensembled)" rows as separate entries, such as RealTabPFN-2.5 at 1519 (default) and 1586 (tuned + ensembled).

### 1.2 The two units

`eval.py` scores a run in the pool of the pinned `tabarena` package. That pool lacks seven newer entrants: Causilo 1.5, LimiX-2-1016, Seldon, Lattice, Seldon-Mini, Inpriora and TabAbax. **In it, #14's run is 5th of 99** (`7cc31c48…`'s `leaderboard.csv`: Kumo-Tabular (L) 2048.6, LimiX-2 1927.2, Kumo-Tabular-Medium 1907.5, TabPFN-3.5 1858.3, #14 1807.6). So the eval's `position` is not the public rank.

`training/scripts/gap_report.py` reads the live board's per-dataset errors from the Space's pages database and refits its Elo with TabArena's method (Bradley–Terry on per-dataset wins, ties as half, RandomForest (default) at 1000). It reproduces the published Lite Elo of all 114 entrants to within 1.1 Elo. With the two TabSil runs inserted (116 entrants):

| Entrant | Repo Elo | Live Elo | Live rank |
|---|---|---|---|
| Kumo-Tabular-Medium | 1907.5 | 1924.3 (published 1921) | 5 |
| TabPFN-3.5 | 1858.3 | 1873.4 (published 1874) | 6 |
| #14's run (`7cc31c48…`) | 1807.6 | **1816.9** | **11** |
| v1.1 (`138151a2…`) | 1767.6 | 1780.0 | 15 |
| TabSil (default), the published entry | 1725.6 for the same weights and inference (`483b3b18…`) | 1744.5 (published 1745) | 18 |

**The two units agree to 5 to 20 Elo**: live minus repo is +9 for #14, +12 for v1.1, +13 to +17 for Kumo-M, Kumo-L and TabPFN-3.5 (published or refit), +5 and +6 for Causilo and Kumo-S, +19 for the published TabSil entry. The dispatch's distances (+113 and +153) subtracted a repo Elo from a live one; in one unit they are +107 and +144.

### 1.3 The distance

| From | Live Elo | To rank 5 | Uniform cut of every error that reaches rank 5 |
|---|---|---|---|
| #14's weights | 1816.9 (rank 11) | +107 | 0.73% (`reduction.csv`) |
| v1.1 | 1780.0 (rank 15) | +144 | not computed |

The cut can be on part of the datasets: 1.26% on the 30 binary datasets alone, or 0.94% on the 38 classification datasets, reaches rank 5. On the 8 multiclass or the 13 regression datasets alone it does not, even at 99% (§4.2).

### 1.4 The CI, the pool, and how the rank is checked

- **The CI.** Kumo-Tabular-Medium's own interval is +148/−97, and Seldon (4th) is 26 above it, TabPFN-3.5 (6th) 47 below. A run that ends a few Elo above Kumo-Tabular-Medium is inside every neighbour's interval. The plan's bar is therefore **25 live Elo above Kumo-Tabular-Medium in the same fit**, about the difference between two stage-2 seeds of one prior (1817 and 1800 live, 1808 and 1788 repo). The user may set another margin (§15).
- **The pool moves.** When #14's run was evaluated (2026-11-26 13:53 UTC), the public columns of its `comparison.csv` had no Causilo 1.5 and Kumo-Tabular-Medium at 1926; on 2026-11-28 Causilo 1.5 is first and Kumo-Tabular-Medium is at 1921. A new entrant above TabSil pushes it down one place. The board is re-read at every reading point (§8), the gap script is run again on the new revision, and the bar is restated before any decision.
- **The final check is a submission, not this eval.** The leaderboard runs its own protocol. Whether it bags foundation models is not settled in its sources: the TabArena paper (§2.1) says foundation models are refit on train and validation data, its repository speaks of 8-fold bagging (read by a subagent, not verified). For TabSil, 8-fold bagging was measured once without a gain (Elo 1711 against 1727, p 0.26, at 6 to 7 times the time; the study of 2026-11-25). The rank is read on the board after the submission and the maintainers' run, with the time columns it reports.

## 2. What exists

Rule 2: the first tier trains nothing, so what is in hand comes first.

### 2.1 Weights

| Weights | What they are | Live Elo (rank) | Repo Elo | Where |
|---|---|---|---|---|
| v1.1 | the released v1 (stage 1 150K, stage 2 10K, stage 3 2K steps on `graph_scm`) with tabsil 0.1.1's inference | 1779.7 (14) | 1767.6 | Hub tag `v1.1` |
| `stage1-150k`, `stage2-10k` | v1's earlier stages | — | 1443.7, 1762.4 | Hub tags |
| The end of the 250K-step stage 1 | stage 1 only, on `graph_scm`, v1's recipe for 250K steps. 1.7% better than the 150K run on the 19 tables of at most 2,500 training rows (G 0.983, p 0.11), level on longer classification tables (v2-experiment-plan §1.2) | — | — | MLflow `32b8195f4f794208a3425878e5c3fcf3` in `tabpfn-v3-reproduction`, written by minipfn. `train.py --init-hub` takes a Hub repo or a `save_pretrained` folder; minipfn runs no longer load in `train.py`, so the checkpoint has to be converted as `stage1-150k` was (item 1.2) |
| **#14's run** | `stage1-150k` plus 10K steps of stage 2 on `graph_scm_hicard_missing`, seed 0, 32 H100, micro batch 1 | **1817.4 (11)** | **1807.6** | MLflow `277916ec05a3457aa6d07f2bc08a60c4`, `model/final.pt`; not released |
| N10 | the same, seed 1 | 1800.3 (14) | 1788.3 | MLflow `520d88d73cdc4e5ea90cfd27425be00d` |
| R10 | the same on `graph_scm_realism`, seed 0 | 1758.8 (15) | 1747.4 | MLflow `b9396104c0664d4e859318f0518a6052` |
| L20 at step 5,000 | stage 2 on the hicard prior in a 20K-step schedule, stopped by the user; a point in the middle of its schedule | not evaluated | — | MLflow `893293ed7d8d419780be2d178b8cc713` |
| Phase 1B | stage 1 from scratch, 30K steps: C seeds 0 to 2, YCTX seeds 0 and 1, OPTK seeds 0 and 1 | capped readout only | — | v2-experiment-results, "Phase 1B" |

The live Elo of each row is its split-0 errors inserted alone into the live board (a separate fit from the table of §1.2, which has two TabSil runs inserted, hence 1817.4 and 1816.9 for #14). The user stopped L20 and PS1; this plan does not resume either.

### 2.2 Code

- The priors: `graph_scm` (v1), `graph_scm_hicard_missing` (#14), `graph_scm_realism` (Kumo's other stage-2 parts), `graph_scm_base_realism` (Kumo's every-stage parts), and their options in `training/src/tabsil_train/priors/`. Features per table are uniform between 1 and 100 for tables of at most 10,240 rows, fewer for longer ones (`table_size.py`).
- Training options from phase 1B: `target_train_rows` (YCTX), the optimizer split (OPTK), the fixed reference validation set, `gc_freeze`, `TABSIL_PREFETCH_FACTOR`, `TABSIL_STOP_STEP`, `TABSIL_RESUME_RUN_ID`. The schedule is a 1% warmup and a cosine to 0 per stage (`recipe.py`).
- Evaluation: `eval.py` with TabArena-Lite, TabArena 3-split, the development suite, BeyondArena and the task suites; `paired_report.py`, `seed_arm_report.py`; timing probes in `training/scripts/v2_eff/`.
- **New in this pull request, CPU only:** `training/scripts/gap_report.py` with its tests (§4), and `training/scripts/v2_task_checks/` (formulations §11 and the checks of sections 1 to 10).
- On the closed branch `Aiden-Jeon/v2-tier2` (`9933bd0`), not on dev-v2: the task-shaped tables (`priors/transforms/tasks.py`: sorted-split and outlier tables), the prior `graph_scm_hicard_task_mix`, their tests, and the tier-1 inference arms. This plan uses none of them (§6).

### 2.3 Measurements

| What | Value | Source |
|---|---|---|
| The prior gain | pooled G 0.992 (0.980 to 1.000), two seeds; 0.997 without polish_companies_bankruptcy; polish 0.596 and kddcup09_appetency 0.964 on 3 splits | v2-findings §2 |
| Changes without an effect | `graph_scm_realism` 1.003; stage 3 1.000; YCTX 0.995 and OPTK 1.003 at 30K stage-1 steps, not resolved | v2-findings §3 |
| Noise | two stage-2 seeds 0.3% pooled, 1808 and 1788 repo, 1817 and 1800 live; two stage-1 seeds 0.1 to 0.5% pooled | v2-findings §4; §2.1 |
| Stage 1 and the context | the flat stage-1 Elo is a context-length effect; stage 2's gain holds on held-out data (0.903 pooled) | v2-findings §5 |
| v1 against other models | Kumo-S 0.996 at v1's compute, Kumo-M 1.047, Kumo-L 1.070, TabPFN-3.5 1.020 | v2-findings §6 |
| Task-shaped tables alone | 1808 → 1252 and 1234 repo | v2-findings §7 |
| Cost | stage 2 per 1K steps: 29.3 H100-h on 32 H100, 21.5 on 8 H100 at micro batch 1, 16.2 at micro batch 2; stage 1, 30K steps at v1's size: 121 H100-h on 8 H100 (117 to 123 as run); `gc_freeze` −6.7% in stage 1; `corrected_cat_sizes` slows the prior, `TABSIL_PREFETCH_FACTOR=16` removed the stalls in stage 1 | v2-findings §8 |

### 2.4 Inference changes already measured

None is adopted. They bound what an inference change can still give.

| Change | Result | Source |
|---|---|---|
| 0.1.0 → 0.1.1 inference | +35 repo Elo (G 0.994), in v1.1 | v2-experiment-plan §1.1 |
| 8 ensemble members against 1 | +80 repo Elo (0.986), in v1.1; 4 members −11 against 8; more than 8 not tested | v2-experiment-plan §1.1; the member study of 2026-11-25 |
| 8-fold bagging | no gain (1711 against 1727, p 0.26) at 6 to 7 times the time; removed in PR #6 | the bagging study of 2026-11-25 |
| Square-root regression members | geometric RMSE ratio 0.971, 0.992, then 1.000 on a third panel; not adopted | Codex line, `dev-v2-codex-2`, `training/scripts/top5_research/fixed-weight-followup.md` (CPU; not re-checked here) |
| Balanced binary class order over the members | ΔAUC −0.0002 on the independent panel; not adopted | the same |
| Frequency-rank codes for nominal columns of 21 or more levels | ΔAUC −0.0035, 1 of 3 panels positive; not adopted | Codex line, `Aiden-Jeon/v2-prior-oracles`, `5cf036d` and `9fe8fbc` |
| 2,048 → 4,096 context rows on three binary sources | AUC −0.006 for TabSil and −0.005 for Kumo-M | Codex line, `top5_research/top5-research-handoff.md` |
| A tuned + ensembled entry | by analogy +16 to +31 for TabSil (RealTabPFN-2.5's and TabPFNv2's changes applied to #14's errors, §4.3) | `gap_report.py`, `whatif_transplant.csv` |

## 3. Premise check

Rule 4: a premise of the request that the evidence speaks against goes back to the user before the plan is built on it.

**The request assumes** that more diverse priors and more diverse tasks can take TabSil from rank 11 to rank 5: +107 live Elo over #14's weights.

**The evidence says:**

1. **The one measured prior gain is in hand, and in Elo it is mostly not the prior's target.** Stage 2 on `graph_scm_hicard_missing`: pooled G 0.992 over two seeds (v2-findings §2), +37.6 live Elo for #14 over v1.1. The four datasets the prior was made for give 4.5 of the 37.6; the rest is small changes on the other 47 datasets (median ratio 0.9997), the size of the difference between two seeds (§4.2).
2. **A prior part aimed at one property cannot carry the distance.** Closing the whole gap to Kumo-Tabular-Medium on the datasets of one weak property gives +27 (minority class below 10%) to +58 (above 10K rows); half the gap gives +14 to +33 (§4.3). Kumo-Tabular's other prior parts, measured as R10, moved no property by more than 0.5% (§4.4). Reported elsewhere: Kumo's prior ablation "marginally improves" at 100K steps (arXiv 2611.23001 §7); Mitra adds +63 in its own pool for one tree-ensemble prior and nothing for four more (arXiv 2510.21204, Table 6); Mitra-v2's mixture weights changed little (arXiv 2609.04540 §2.2).
3. **Tasks act on TabArena only through two channels, and both are small or bounded** (formulations §11). The task tables built so far are recognizable and TabArena's splits are random, so in the limit they change nothing there; alone they cost 555 and 574 repo Elo (v2-findings §7). The tasks that add targets per table act through the training term, and cannot give more than the extra training they stand for: about 28 to 91 Elo per doubling of effective data (formulations §11.5), with the doubling not measured for any task.
4. **The lever with a reported association of the needed size is training compute, and it is broad.** Kumo-Tabular's S → M step (4.6 times the compute, 2.3 times the parameters, twice the stage-1 tables) improves 42 of 51 datasets; applied to #14's errors it gives +96 (rank 6). Its M → L step improves 45 of 51 and gives +115 (rank 5) (§4.3). Kumo-S at v1's compute is level with v1 (G 0.996, v2-findings §6). This is an association for another model, with the size changed along with the compute. TabSil's own records of more training are small and on one axis each: 100K more stage-1 steps gave 1.7% on 19 small tables (p 0.11), the second half of stage 2 gave 0.993 within one schedule, stage 3 gave 1.000.
5. **The leader's jump is not explained.** Causilo 1.5 is 256 Elo above Causilo 1.0 on the full board two months later (its blog's table), and the blog states no change of architecture, data, prior, objective or compute (nums.world/blog/causilo-1-5, opened). Applied to #14, Causilo's per-dataset change gives +239 (rank 2). Gains of that size exist that compute does not explain; what made this one is not published.

**Options**, each with its expected gain over #14's weights in live Elo:

| Option | What | Expected gain | Kind of evidence | H100-h | Calendar |
|---|---|---|---|---|---|
| A. Stop | #14's weights stay the candidate; its release and submission are separate decisions | 0; rank 11 | measured | 0 | — |
| B. Priors and tasks | tiers 1 and 2 without tier 3: the measurements, the compute-slope reading 2.1, and by choice the prior-and-task branch 2.2 and the masked-cell branch 2.3 | +0 to about +43 from 2.1, +0 to about +30 from 2.2, unknown from 2.3; not additive with certainty | estimated (§7) | about 230; up to about 680 with 2.2 and 2.3 | about 2 days per branch on one node |
| C. Compute | tier 3: Kumo-Tabular-Medium's tables per task at TabSil's size, with whatever of B passes | +94 to +147 by analogy, rank 4 to 6; range 0 (TabSil's slope flat) to about +200 (Kumo's S → L change gives +199) | estimated by analogy for another model; TabSil's own slope read first at 2.1 | 3,640 to 3,830 on one node per stage (stage 3 estimated); 2,630 without stage 3; 4,850 to 5,030 with stage 1 on 32 H100 | 19 to 20 days on one node; 13 to 14 with stage 1 on 32 H100; 13.5 without stage 3 |
| D. A tuned + ensembled entry | a search space over TabSil's inference settings and weights, run under TabArena's tuning protocol (200 random configurations and post-hoc ensembling; read by a subagent, not verified) | +16 to +31 by analogy (rank 7 to 8) | estimated by analogy; a different kind of entry | about 65 for 25 configurations to about 500 for 200 on Lite (estimated: 0.31 H100-h per Lite run, times 8 folds per configuration) | days |

- **Only C has an association that reaches the bar**, and it lands about at the bar (rank 4 to 6), not clearly above it. B's best item, 2.1, is also the cheapest reading of whether C would work for TabSil.
- **D** was the option with a reported +67 to +89 on the board (RealTabPFN-2.5, TabPFNv2). Applied to TabSil's errors in the crowd at rank 11, the same changes give +16 to +31. It is listed because the earlier plans did not price it.
- **B is what the request asked for**, and it is planned in full in §5 to §7. On its evidence it does not reach the bar by itself.

## 4. Where TabSil is behind: the per-dataset gap

`training/scripts/gap_report.py` (this pull request) joins the split-0 errors of #14's run and v1.1 with the live board's errors on the 51 TabArena-Lite datasets and their properties (TabArena's task metadata at `autogluon/tabarena` `8e55fb9`, OpenML's feature lists). It runs in seconds on a CPU. Every number of this section comes from its run at Space revision `a525fe89…`:

```bash
DATABRICKS_CONFIG_PROFILE=e2-demo-field-eng MLFLOW_TRACKING_URI=databricks \
  uv run --frozen --package tabsil-train python training/scripts/gap_report.py \
  --cache-dir /tmp/top5-gap-data --out-dir /tmp/top5-gap-out \
  --space-revision a525fe89b5824325055ad58ff3de5590047a2877 \
  --compare "R10=987892ced69c4889b4f063c329d11958:7cc31c4845c442079d998a1c02839533+e2ff88f0c3644705a5a57b73f73cd35c" \
  --compare "N10=e2ff88f0c3644705a5a57b73f73cd35c:7cc31c4845c442079d998a1c02839533"
```

### 4.1 By task and by property

#14 against the references, all 51 datasets (G, 95% CI, datasets where #14 is better / worse):

| Reference | G | Better / worse |
|---|---|---|
| Causilo 1.5 | 1.067 (1.033 to 1.120) | 4 / 47 |
| Kumo-Tabular (L) | 1.060 (1.025 to 1.115) | 12 / 39 |
| LimiX-2-1016 | 1.067 (1.024 to 1.133) | 15 / 36 |
| Seldon | 1.046 (1.017 to 1.090) | 11 / 40 |
| **Kumo-Tabular-Medium** | **1.037 (1.012 to 1.076)** | 14 / 37 |
| TabPFN-3.5 | 1.010 (0.986 to 1.038) | 22 / 29 |

Against Kumo-Tabular-Medium by task: classification 1.038 (38), regression 1.033 (13), binary 1.043 (30), multiclass 1.020 (8). By property:

| Property | Bucket | Datasets | G against Kumo-Tabular-Medium | G against the top 5's geometric mean |
|---|---|---|---|---|
| Missing values | any | 11 | **1.088** | 1.122 |
| | none | 40 | 1.023 | 1.037 |
| Features | 21 to 100 | 17 | **1.087** | 1.120 |
| | at most 20 | 28 | 1.017 | 1.029 |
| | above 100 | 6 | 0.989 | 1.003 |
| Minority class share | below 10% | 13 | **1.069** | 1.098 |
| | 10% or more | 25 | 1.022 | 1.041 |
| Categorical columns | none | 17 | 1.060 | 1.093 |
| | any | 34 | 1.025 | 1.036 |
| Largest number of levels | 21 or more | 12 | 1.025 | 1.042 |
| Training rows | below 2K | 18 | 1.044 | 1.059 |
| | 2K to 10K | 18 | 1.043 | 1.066 |
| | above 10K | 15 | 1.020 | 1.038 |

- **TabSil is furthest behind on wide numeric tables, tables with missing values, and binary tables with a rare class.** These overlap: polish_companies_bankruptcy is in all four of the weakest buckets.
- **It is not behind on long tables or on very wide ones.** Above 10K rows it is 2.0% behind and above 100 features 1.1% ahead.
- The ten datasets furthest behind Kumo-Tabular-Medium: polish_companies_bankruptcy 1.99, hazelnut-spread-contaminant-detection 1.35, airfoil_self_noise 1.22, customer_satisfaction_in_airline 1.20, anneal 1.13, miami_housing 1.08, website_phishing 1.07, coil2000_insurance_policies 1.04, Marketing_Campaign 1.04, physiochemical_protein 1.04. The 51 datasets are in the appendix.

### 4.2 What moves the rank

| What-if on #14's errors (live pool) | Elo | Rank |
|---|---|---|
| as it is | 1816.9 | 11 |
| one dataset at Kumo-Tabular-Medium's error, the best single one (website_phishing) | +9.2 | 10 |
| one dataset at the best top-5 error, the best single one (E-CommereShippingData) | +16.4 | 9 |
| polish_companies_bankruptcy at Kumo-Tabular-Medium's error | +2.0 | 11 |
| every binary dataset at Kumo-Tabular-Medium's error | 1857.7 | 7 |
| every regression dataset at Kumo-Tabular-Medium's error | 1865.7 | 7 |
| every multiclass dataset at Kumo-Tabular-Medium's error | 1824.4 | 10 |
| every error cut by 0.73% | about 1924 | 5 |
| every binary error cut by 1.26% | about 1924 | 5 |

The v1.1 → #14 change split by datasets (`whatif_split.csv`): v1.1 is 1779.7; with #14's errors on polish_companies_bankruptcy only, +2.8; on anneal +1.7; on kddcup09_appetency +0.7; on Amazon_employee_access −0.7; on all four, +4.5; #14 itself, +37.6.

- **Elo counts wins, and TabSil sits in a crowd.** Matching Kumo-Tabular-Medium exactly on every binary dataset gives rank 7, because on the 9 binary datasets where #14 is better it loses that lead, while a cut of 1.26% everywhere wins narrow battles on every dataset.
- **So a change is worth what its breadth is worth.** A gain concentrated on a few datasets, as #14's prior is in G, buys a few Elo. A small gain on most datasets buys many.

Realistic changes are not uniform. With a per-dataset spread like that between two seeds (sd of the log ratio 0.01, v2-findings §4), the cut needed is larger (`whatif_noise.csv`, 10 draws per cell):

| G of the change | sd 0: Elo gain (rank) | sd 0.01: mean Elo gain, share of draws at rank ≤ 5 | sd 0.02: the same |
|---|---|---|---|
| 0.9975 | +41 (7) | +10, 0.0 | −38, 0.0 |
| 0.995 | +77 (6) | +50, 0.0 | −11, 0.0 |
| 0.9925 | +108 (5) | +69, 0.2 | +8, 0.0 |
| 0.990 | +147 (4) | +105, 0.4 | +51, 0.1 |
| 0.985 | +194 (4) | +175, 1.0 | +98, 0.4 |
| 0.980 | +230 (4) | +203, 1.0 | +139, 0.8 |

**A broad change of about 1.5% (G 0.985) with seed-sized noise reaches rank 5 in every draw.** Per-dataset noise costs Elo on average here, because #14 wins more of its narrow battles than it loses.

### 4.3 Elo bounds and analogies

**Gap closed on the datasets of a property** where #14 is behind Kumo-Tabular-Medium (a share of the log gap; `whatif_buckets.csv`). This bounds a prior part aimed at that property:

| Datasets | Number | Behind Kumo-M | 25% of the gap | 50% | 100% | At the best top-5 error |
|---|---|---|---|---|---|---|
| all | 51 | 37 | +40 (7) | +75 (6) | +150 (4) | +482 (1) |
| missing values | 11 | 9 | +11 (9) | +20 (8) | +35 (7) | +70 (6) |
| 21 to 100 features | 17 | 14 | +14 (9) | +26 (7) | +43 (7) | +91 (6) |
| minority class below 10% | 13 | 9 | +8 (10) | +14 (9) | +27 (7) | +59 (6) |
| no categorical column | 17 | 11 | +9 (9) | +17 (8) | +33 (7) | +78 (6) |
| above 10K training rows | 15 | 13 | +19 (8) | +33 (7) | +58 (6) | +98 (6) |
| binary (2 classes) | 30 | 21 | +18 (8) | +32 (7) | +61 (6) | +187 (4) |

The four weakest properties together (missing values, 21 to 100 features, a minority class below 10%, or no categorical column) cover 36 datasets, 26 of them behind Kumo-Tabular-Medium: closing 25% of their gap gives +27 (rank 7), 50% gives +47 (7), all of it +93 (6), and the best top-5 error +223 (4). #14's prior closed about a third of the log gap on polish_companies_bankruptcy (v1.1 2.82 times Kumo-M's error, #14 1.99) and left the other datasets of its buckets level.

**Another entrant's change, dataset by dataset, applied to #14's errors** (`whatif_transplant.csv`). This is an analogy: it assumes TabSil would move as that entrant moved.

| Change | G | Datasets better | Elo gain | Rank |
|---|---|---|---|---|
| Kumo-Tabular S → M (4.6 times the compute, 2.3 times the parameters) | 0.951 | 42 | +96 | 6 |
| Kumo-Tabular M → L (3.2 times the compute, 3.5 times the parameters) | 0.978 | 45 | +115 | 5 |
| Kumo-Tabular S → L (14.7 times the compute) | 0.931 | 47 | +199 | 4 |
| TabPFN-3.5-Fast → TabPFN-3.5 (84M → 220M parameters) | 0.972 | 41 | +78 | 6 |
| Seldon-Mini → Seldon (27M → 213M) | 0.948 | 40 | +56 | 6 |
| Lattice-tiny → Lattice | 0.975 | 44 | +95 | 6 |
| TabPFN-3 → TabPFN-3.5 | 0.941 | 43 | +151 | 4 |
| Causilo → Causilo 1.5 (not disclosed) | 0.926 | 47 | +239 | 2 |
| RealTabPFN-2.5 default → tuned | 0.993 | 30 | −32 | 14 |
| RealTabPFN-2.5 default → tuned + ensembled | 0.968 | 36 | +31 | 7 |
| TabPFNv2 default → tuned + ensembled | 0.966 | 25 | +16 | 8 |

A share $f$ of a Kumo step, applied as $f$ times its per-dataset log ratios (the power law in compute behind §11.5 of the formulations): S → M at $f$ = 0.25, 0.5, 0.75 gives +30, +50, +75; M → L gives +32, +60, +94. **These shares price item 2.1 and tier 3** (§7), at today's measured costs (§9; #14's recipe is 150K stage-1 and 10K stage-2 steps, about 820 H100-h):

| Run | Compute against #14's recipe | Share of Kumo's S → M step (2.19 doublings) | Elo | Share of its M → L step (1.69 doublings) | Elo |
|---|---|---|---|---|---|
| 2.1: 250K instead of 150K stage-1 steps, the same stage 2 | ×1.49 (0.57 doublings) | 0.26 | +31 (rank 7) | 0.34 | +43 (rank 7) |
| Tier 3: 500K / 40K / 12K steps | ×4.4 to ×4.7 (2.1 to 2.2 doublings) | 0.97 to 1.02 | +94 to +97 (rank 6) | 1.27 to 1.33 | +141 to +147 (rank 4) |

The M → L column for tier 3 extrapolates beyond Kumo's own step.

### 4.4 What the existing runs say about the weak properties

R10 against the mean of #14's run and N10, by property (`whatif_compare.csv`; all three from `stage1-150k`, 10K steps of stage 2): 21 to 100 features 1.001, missing values 0.998, minority class below 10% 0.999, no categorical column 1.002; below 2K rows 1.010 (worse). **Kumo-Tabular's warp, 200 categories, correlated blocks, relabel and coarsen do not move the properties where TabSil is behind.** N10 against #14, the noise of a seed, by the same buckets: 0.995 to 1.008.

## 5. Prior diversity

For each candidate: the property of real tables it adds, the evidence (controlled? compute matched? seeds? size?), the datasets of §4 on which TabSil is behind and which have the property, the expected effect in live Elo with its kind, the cost, and the generation-speed risk. **A candidate earns a run only if it names such datasets** (the method that worked for #14). §5.6 ranks them.

### 5.1 In the code

| Candidate | Adds | Evidence | Named datasets behind | Expected live Elo; kind | Cost; speed |
|---|---|---|---|---|---|
| `graph_scm_hicard_missing` | 21 or more categories in 10% of the tables; missing values that depend on values, in column groups | measured, two seeds against the v1 prior at equal steps: pooled 0.992 | polish (64 features, missing, rare class), kddcup09 | **+37 in hand** (+21 for the second seed); prior | 0 |
| `graph_scm_realism` | warped columns, up to 200 categories, correlated blocks, relabel, coarsen | measured, one seed against the two hicard seeds: 1.003; no weak property moved by 0.5% (§4.4) | none moved | 0 (measured; R10 is 1758.8 live against 1817.4 for #14); prior | 450 H100-h for its run (stalls) |
| `graph_scm_base_realism` (Kumo's every-stage parts from step 0) | the same parts from the start of stage 1, too-easy tables redrawn | not measured (PS1 stopped at 10,500 and 5,000 steps) | — | 0 expected by R10's result for the same parts in stage 2; prior | 9% slower per step |
| `target_train_rows` (YCTX) | the regression target scaled with the context rows, as at inference | measured, 2 seeds against 3 at 30K stage-1 steps: 0.995 (0.990 to 1.000), 63 of 99 better, seeds disagree (0.993, 1.001). Derived: a scale term of 0.07 to 0.5% of the regression loss (formulations §2) | all 13 regression datasets; the change is broad, not aimed | +0 to about +30 (estimated: the regression-only cut cannot reach rank 5; the 100% regression bound is +52); prior | a flag; no speed risk |
| `noise_prob` | noisy copies of 10% of the columns | none; four 20K-step runs with one option each scored 18 to 60 below their control (untrusted readout) | — | unknown, sign against; prior | under 1% of the generation time |
| The covering split (`CoverClasses`) | — (it keeps every query's class in the context) | derived (formulations §2); measured in tier 1 of the closed plan: holds for 2 classes only | multiclass: anneal (5 classes, rarest 0.9%) | about 0: binary AUC cannot change; multiclass bound at 100% of the gap +19; prior or inference | one branch |

### 5.2 In Kumo-Tabular and not ported

Read in `NVIDIA/structured-data-models` at `924e992` by the closed plan (v2-prior-task-plan §3 (b)).

| Part | Stage in Kumo | Adds | Expected; kind | Cost; speed |
|---|---|---|---|---|
| Count and distributional numeric columns (10% of numeric columns, 30% of those rounded) | every stage | counts, heteroscedastic columns | 0 by R10's analogy for Kumo's other numeric parts; no named dataset; prior | inside the SCM draw: generation-speed risk, smoke run needed |
| Argmax categories | 2 and 3 | classes that are hard functions of the causes | 0, unknown; prior | inside the SCM draw |
| Heavy-tailed regression targets (20% of targets) | 3 | skewed targets | 0: stage 3 itself measured 1.000; prior | a transform |
| Small tables replayed in stage 3 | 3 | small tables in late training | 0: stage 3 1.000 | a loader change |
| A wider column ladder for long tables | 3 | wide and long tables together | 0: TabSil is not behind above 10K rows by more than 2% or above 100 features at all (§4.1) | the table-size rule |
| Shared hyperparameters within a table, noisy-measurement columns, per-table function preferences, too-easy tables rejected | every stage | — | 0, unknown | inside the SCM draw for the first three |

Kumo's whole prior, against TabICLv2's, "marginally improves" at 100K steps with three seeds (arXiv 2611.23001 §7). **What separates Kumo-Tabular from TabSil is not a prior part; it is 3 to 9 times the tables per task** (§7, tier 3).

### 5.3 Other families

| Family | Adds | Evidence | Expected for TabSil; kind | Cost; speed |
|---|---|---|---|---|
| Tables labeled by fitted tree ensembles (Mitra) | axis-aligned, piecewise-constant targets | controlled within one study, seeds not stated: SCM alone 1000, +ET prior 1063, all six priors 1062, Mitra's own pool (arXiv 2510.21204, Table 6) | unknown, smaller than Mitra's: TabSil's SCM already has tree mechanisms in its function set; no named dataset; prior | a generator that fits an ensemble per table; generation time unknown |
| A hybrid SCM and realism modules (Mitra-v2; arXiv 2605.18971) | mixed mechanisms, irregular marginals | Mitra-v2: mixture weights changed little, the hybrid SCM not isolated (§2.2, appendix C, read by a subagent); arXiv 2605.18971: controlled at small scale, numbers not opened | 0, unknown; prior | a new generator |
| **Continued training on real tables (Real-TabPFN, arXiv 2507.03971)** | real marginals and dependencies | TabPFN v2 after 20,000 steps at batch 1 on OpenML and Kaggle tables: normalized AUC 0.954 → 0.976, raw mean AUC 0.929 → 0.932, on 29 AMLB classification datasets; seeds not stated; web-scraped tables lowered it (read by a subagent) | unknown; reported positive for a weaker model; the raw gain is about 4% of the binary error. Kind: prior (real tables) | GPU: under about 20 H100-h. The cost is the corpus and its leakage check against TabArena, the development suite, AMLB, BeyondArena and the task suites, which share OpenML as a source, and the leaderboard's leakage column |
| An adversarially trained generator (APT, arXiv 2502.04573) | tables the current model finds hard | regression MSE 0.340 → 0.306 on 35 datasets against its TabPFN baseline, classification within one standard deviation (read by a subagent) | unknown, reported for TabPFN v1's prior; prior | a second network trained with the model |
| Drift priors (Drift-Resilient TabPFN, TabPFN-3's dynamic SCM) | mechanisms that move with time | "lower predictive performance on ID tasks" (v2-experiment-plan §1) | 0 or negative; prior | a second-order generator |

### 5.4 What the 2026 top five disclose

From the papers, model cards and repositories (literature report of 2026-11-28; read by a subagent except where marked opened):

| Model | Size | Training | Disclosed ablations |
|---|---|---|---|
| Causilo 1.5 | not stated (Causilo 1.0: 36M per task, arXiv 2609.22866) | not stated; 1.0 saw about 36M synthetic tables | none on accuracy. The 1.5 blog (opened) names no change but "handles high-dimensional data without feature sampling" |
| Kumo-Tabular (L), Kumo-Tabular-Medium | 214M, 61M per task, two models | synthetic only, Kumo's prior; per task 1M / 60K / 18K steps at 128 tables (L), 500K / 40K / 12K (M); 23,148 and 7,186 H100-h (Table 2) | prior: "marginal" at 100K steps, three seeds; stage 2 +65 to +170 Elo, stage 3 at most +23 |
| LimiX-2-1016 | about 400M (model card) | a masked joint objective (CCMM); compute and steps not published; what "1016" changes not published | size at a fixed recipe: +45.7 Elo per doubling from 12.5M to 406M (arXiv 2609.17488 §6) |
| Seldon | 213.5M and 215.5M, a checkpoint per task | "No real-world data was used in pretraining"; nothing else | none |
| TabPFN-3.5 (6th) | about 220M, one checkpoint for both tasks | not published | none |

- **The only top-5 recipe that can be copied is Kumo-Tabular's**, and against TabSil it differs mainly in the tables per task: Kumo-M has 3.3, 4 and 6 times v1's per-task tables in stages 1, 2 and 3 (v2-experiment-plan §1.3).
- **Nothing disclosed supports a prior part or an objective at the size of the distance.** LimiX-2's masked objective has no ablation; Causilo 1.5's change is not described.

### 5.5 Real tables against the prior, without training

The distance between a prior and real tables can be measured on a CPU, and it does not predict accuracy: a classifier tells the TabICL prior from real tables with an AUC of 0.9997, and a dataset's distance to the prior correlates below 0.14 with the model's accuracy on it (arXiv 2605.06343, Table 2; opened by the closed plan). So this plan does not rank priors by such a distance. It ranks them by the datasets they name and the Elo those datasets can carry (§4.3).

### 5.6 Ranked by expected Elo per H100-h

| Rank | Candidate | Expected live Elo | H100-h | In the plan |
|---|---|---|---|---|
| 1 | `graph_scm_hicard_missing` | +37, in hand | 0 | tier 0 |
| 2 | `target_train_rows` in stage 2 | +0 to about +30 | 223 (one branch) | item 2.2, by the user's choice |
| 3 | Continued training on real tables | unknown, reported positive for another model | under 20 on the GPU; the corpus and the leakage check first | listed, not planned: its cost is the corpus work, and its effect has no estimate for TabSil |
| 4 | Kumo's parts not ported, Mitra's trees, APT | 0 or unknown; no named dataset | code plus one branch each | not planned |
| 5 | `graph_scm_realism`, `graph_scm_base_realism`, `noise_prob`, the covering split, drift priors | 0 measured, 0 expected, or negative | — | not planned |

**No prior candidate has an expected effect that a branch can resolve and that reaches the bar.** The best one, YCTX in stage 2, is at the resolution of one pair (§7).

## 6. Task diversity

The derivations are §11 of [v2-task-formulations.md](v2-task-formulations.md): for every task, (a) what training on the mixture converges to, (b) the three channels with their sign and size, (c) the check, (d) the expected effect and kind. Its ranked table, in short:

| Rank | Task or signal | Channel | Expected live Elo | Kind | Cost | In the plan |
|---|---|---|---|---|---|---|
| 1 | Missing values that depend on the target and other columns (the hicard prior) | coverage | +37, in hand | prior | 0 | tier 0 |
| 2 | Further missingness mechanisms | coverage of the 11 missing-value datasets | +0 to about +20 (estimated; the 100% bound for the missing-value datasets is +35) | prior | one branch | not planned: no part named beyond the hicard prior's |
| 3 | Masked-cell reconstruction | supervision per table; shared representation | unknown sign; at most about 28 to 91 per doubling of effective data | structure and loss | code, then one branch | item 2.3, gated by item 1.1 |
| 4 | Several targets per table | supervision per table | unknown; at most the compute the shared stages save | structure | code, then one branch | not planned |
| 5 | Unlabeled rows as context | coverage on every table | unknown; one controlled positive report (abstract) | structure, prior, inference | the largest code change | not planned: whether the leaderboard counts it is not verified |
| 6 | More query rows per table | supervision per table | about 0 unless item 1.1 finds $\sigma_w^2/(m \sigma_b^2) \ge 1$ | prior | one branch | gated by item 1.1 |
| 7 | AUC-aligned pairwise loss; a mean output for RMSE | finite capacity; loss | about 0 (checked) | loss | — | not planned |
| 8 | Forecasting, outlier, treatment-effect, survival, drift and clustering tables | none on TabArena | 0, sign unknown; task tables alone cost 555 to 574 | prior | — | not planned |
| 9 | Multi-output, ranking, density, time-series anomalies, effect intervals, conformal intervals, Bayesian optimization | none: the weights do not change | exactly 0 | inference | 0 | not planned |

**Which tasks enter which tier at what share.** None enters as task-shaped tables: at any share $w$ the limit is the ordinary answer on recognizable tables, and the only channel that can lower the TabArena error is supervision per table (formulations §10, §11). The masked-cell head (rank 3) enters tier 2 as item 2.3 at a loss weight to be fixed in its code review, only if item 1.1 finds training noise-limited (§7), since otherwise more targets per pass have nothing to act on. The task tables of the closed branch are not used.

## 7. Tiers

Every run has its expected effect, the resolution of its design, and the decision each outcome leads to (rule 5). The resolution of one branch against the two existing seeds is **0.6% pooled** (G on TabArena 3-split plus the development suite; the branch rule's threshold is 0.9944, from ν 0.003) and **about 20 Elo on TabArena-Lite** (the two hicard seeds are 1817 and 1800 live). No rule adopts on "not worse". A result without an effect starts no run.

### Tier 0: nothing trained (0 H100-h)

| What | Adds | Evidence |
|---|---|---|
| #14's weights (`277916ec…`, not released) | the base: 1816.9 live, rank 11; +37 over v1.1 | measured; one seed (the second seed of the prior is 1800.3) |
| This pull request: the gap analysis, the derivations and their checks | no Elo; the expected effects of §4 to §6 | CPU |

Whether #14's weights are released or submitted is an open decision from before (v2-experiment-plan, phase 5) and is not part of this plan.

### Tier 1: no training (about 4 H100-h)

| # | What | Expected | Resolution | Decision by outcome | H100-h |
|---|---|---|---|---|---|
| 1.1 | Gradient-noise statistics of #14's weights on stage-2 prior tables: the per-query gradient's variance within and between tables ($\sigma_w^2$, $\sigma_b^2$) and the noise scale $\mathrm{tr}\Sigma / \lVert G \rVert^2$ at batch 256 (formulations §11.3) | a measurement; no Elo | the ratio $r = \sigma_w^2/(m \sigma_b^2)$ at stage 2's query counts, and the noise scale against 256 tables | $r \ge 1$ or a noise scale above 256 tables: gradient noise limits training, and item 2.3 goes to the user with its code. Otherwise the supervision items of §6 (ranks 3, 4 and 6) are dropped | about 1 (one GPU, a few hundred tables; a probe script like `v2_eff/step_probe.py`, code needed) |
| 1.2 | The 250K-step stage 1 converted to a `save_pretrained` folder or a Hub tag `stage1-250k` (CPU, as `stage1-150k` was), then read with the context capped at 1,024 rows on TabArena 3-split and the development suite, against `stage1-150k`'s capped runs | G about 0.98 to 0.99 on small tables, level elsewhere (estimated from v2-experiment-plan §1.2) | the noise of two stage-1 runs, 0.1 to 0.5% pooled | G pooled at most 0.995: item 2.1 runs. Above: 2.1 does not run, and tier 3 goes to the user with TabSil's stage-1 slope read as flat | about 3 |
| 1.3 | The live board re-read and `gap_report.py` run on its current revision | — | — | the bar restated | 0 |

Item 1.2 costs 1.3% of item 2.1, and item 1.1 under 1% of item 2.3 (rule 7).

### Tier 2: stage-2 branches from stage-1 weights that exist

Setup for all three: 10K steps of stage 2 at batch 256 on one 8xH100 node at **micro batch 1**, the setup of #14's run and N10 in all but the GPU count, so that every step holds the same table sizes as the controls; the GPU count changes which tables a step draws, as a seed does (v2-experiment-results, "What a larger micro batch changes"). Cost: 21.5 H100-h per 1K steps measured (`bf797ac1…`), 223 with the start, validation and checkpoints. The control is the mean of #14's run and N10 (`paired_report.py --a runs:/<#14>+runs:/<N10>`). Evaluation: TabArena-Lite, TabArena 3-split and the development suite, about 2.5 H100-h, then `gap_report.py` for the live Elo.

| # | What | Expected effect | Resolution | Decision by outcome | H100-h |
|---|---|---|---|---|---|
| 2.1 | **Stage 2 from the 250K-step stage 1**, `graph_scm_hicard_missing`, seed 0. The question: does more stage-1 training carry through stage 2, and at what slope? | G about 0.985 to 1.000 pooled; +0 to about +43 live Elo. Estimated: a flat slope gives 0; Kumo's slope gives 0.26 of its S → M step (+31) to 0.34 of its M → L step (+43) (§4.3) | 0.6% pooled; about 20 Elo per pair. A Kumo-like slope (G 0.987 by the S → M share, 0.993 by the M → L share; its Elo +31 to +43) is above both | **G ≤ 0.990 with its CI below 1**: TabSil's stage-1 compute carries through stage 2 at about three quarters of the slope of Kumo's S → M step or more; tier 3 goes to the user priced with the measured slope, and 2.1's weights become the candidate. **0.990 < G ≤ 0.9944**: a smaller slope, at most about that of Kumo's M → L step; tier 3 is priced again with it and goes back to the user (rule 9). **Otherwise**: no slope shown; tier 3's stage-1 part has no TabSil evidence, and tier 3 goes back to the user with dropping it as the first option | 223 + 2.5 |
| 2.2 | **By the user's choice only:** the prior-and-task branch: `graph_scm_hicard_missing` with `target_train_rows`, from `stage1-150k`, seed 2 | +0 to about +30 live Elo (estimated; §5.1). By rule 5 it is not in the plan by default: its expected effect is at the resolution of one pair | as 2.1 | G ≤ 0.9944 pooled with its CI below 1, and at least +20 live Elo over the controls' mean: the option goes into tier 3's stage 2. Otherwise: priors and tasks stop | 223 + 2.5 |
| 2.3 | **Gated by item 1.1:** a masked-cell head with its loss, from `stage1-150k` on the hicard prior: hidden cells of the query rows enter as missing cells, which the cell encoder already marks, and a head predicts them from the row's in-context output and the column's summary | unknown sign; at most about 28 to 91 Elo per doubling of effective data (formulations §11.5); the doubling it buys is read first on the reference validation loss at equal steps | as 2.1 | as 2.2 | code (a head, its loss, the masking), a smoke run of 0.5, then 223 + 2.5 |

- 2.1 and 2.2 do not bear on each other and may run side by side on two nodes. Neither starts beside tier 1, whose items decide them (rule 8).
- A prior or setup used at scale for the first time gets a smoke run first (rule 6): 2.2 and 2.3 change the prior options and the model, so each has a 300-step smoke run on one GPU (about 0.5 H100-h) that checks the step time and the data wait before the branch.

### Tier 3: the compute lever (only if the user puts it in scope)

Kumo-Tabular-Medium's tables per task at TabSil's size: 500K / 40K / 12K steps against v1's 150K / 10K / 2K (per task 64M, 5.1M and 1.5M tables; TabSil alternates the tasks, so N steps at batch 256 are N × 128 tables per task, as Kumo's per-task models at batch 128). From scratch, v1's stage-1 recipe and prior; stage 2 on `graph_scm_hicard_missing` with whatever of tier 2 passed. One node per stage.

| # | What | Expected effect | Resolution | Decision by outcome | H100-h; wall time |
|---|---|---|---|---|---|
| 3.0 | Timing jobs for the setups not yet measured: stage 3 on one node (110 steps); stage 1 on 32 H100 with today's code if the user wants the shorter wall time | — | — | a projected cost more than 30% above this table goes back to the user (rule 9) | about 10; and 5 |
| 3.1 | Stage 1, 500K steps, one node, micro batch 4, `gc_freeze` on | the capped readout better than the 250K run's by the slope 2.1 measured | the stage-1 seed noise, 0.1 to 0.5% pooled | **R3a**: not better than the 250K run by more than 0.5% pooled at the capped readout: stop before stage 2 (saves about 1,700 to 1,850) | 1,950 (3.74 measured, `7266d800…`, plus 3 to 4% as in phase 1B); about 10 days |
| 3.2 | Stage 2, 40K steps, one node, micro batch 2 | +94 to +147 over #14 with 3.3, by analogy (§4.3) | 0.6% pooled; about 20 Elo per pair | **R3b**: the live Elo of the result (`gap_report.py`) at most at rank 7 or G pooled above 0.985 against #14: stop before stage 3 (saves about 1,000 to 1,200). Otherwise stage 3 | 648 (16.2 measured, L20); about 3.4 days |
| 3.3 | Stage 3, 12K steps, one node | at most +23 (Kumo's report); TabSil's 2K-step stage 3 measured 1.000 | as 3.2 | **R3c**: the final candidate is the better of 3.2 and 3.3 by the pooled rule; then the submission | 1,020 to 1,200 (estimated: 85 to 100 per 1K steps from a one-GPU run, `5df481cb…`; not measured on one node); 5 to 6 days |
| 3.4 | Release evaluation and the public submission (§1.4) | the rank on the board | the board's CI | the user's release decision | evaluations about 5 |

- **Total:** 3,640 to 3,830 H100-h, 19 to 20 days on one node (the stages run in sequence). Without stage 3: about 2,630, 13.5 days. With stage 1 on 32 H100 (6.31 H100-h per 1K steps, v1's record with older code; today's code not measured there): 4,850 to 5,030, 13 to 14 days.
- **A cheaper start, not measured for this model:** stage 1 continued from the 250K-step checkpoint for 250K more steps with a new warmup, about 975 H100-h less and 5 days shorter. Whether a re-warmed continuation equals one schedule of 500K steps is not known for TabSil.
- **Reading points inside a stage need another schedule.** With a cosine to 0, a checkpoint in the middle of a stage is not an end point (as L20's step 5,000 is not). A warmup-stable-decay schedule would give end points on the way (code needed, not measured for this model). The plan reads at the end of each stage only.
- **Kumo-Tabular-L's tables** (1M / 60K / 18K) are a further step: about 6,430 to 6,700 H100-h and 33 to 35 days on one node; +199 by Kumo's S → L analogy. Not planned.

## 8. Reading points and the stop threshold

| Point | When | What is read |
|---|---|---|
| R0 | the user approves | the board re-read (1.3); the budget, threshold and scope set |
| R1 | tier 1 is done | 1.1 and 1.2: whether 2.1 and 2.3 run |
| R2 | tier 2's branches are evaluated | 2.1's slope; 2.2's and 2.3's gains; the live Elo of the best weights; whether tier 3 runs |
| R3a, R3b, R3c | the end of each stage of tier 3 | §7 |
| R4 | the submission is on the board | the rank |

**The threshold is the user's** (rule 3). Three candidates, with what each implies today:

| Threshold | Rule at every reading point | What it implies with today's numbers |
|---|---|---|
| (a) The bar | stop when the measured live Elo plus the expected gain of what is left, with its evidence, is below Kumo-Tabular-Medium + 25 | B alone ends at R2 whatever 2.1 shows, since its best case is about +43 (+107 needed). Tier 3 runs only if 2.1 reads a slope that puts tier 3's forecast at the bar |
| (b) +40 per tier | stop when a tier adds less than +40 live Elo measured (the size the user judged not worth runs on 2026-11-28) | 2.1 must show about the slope of Kumo's M → L step or more (+43 by that analogy); at the slope of its S → M step 2.1 shows +31, and the plan ends at R2 with about 230 H100-h spent |
| (c) The resolution | stop when a tier's gain cannot be told from the noise: less than 20 live Elo, or a pooled G above 0.9944 | the most permissive: tier 3 runs if 2.1 shows anything resolvable |

**Chosen threshold: the user sets** (§15).

**Between the points** the plan goes back to the user with dropping the item as an option (rule 9) when an item's projected cost is more than 30% above this plan's, or when its expected gain falls: 1.2 reads the 250K stage 1 as flat, 2.1 reads no slope, or a re-read of the board moves the bar by more than the margin.

**Every status report begins with four lines** (rule 11): the live Elo and rank of the best weights so far against the bar; where the plan is expected to end, with the kind of evidence; the H100-h spent, counted from the job records; what is left of the plan and of the cap. A lower estimate of the plan's own budget is not reported as a result.

## 9. Costs and their bases

| Item | H100-h | Basis |
|---|---|---|
| Stage 1, v1's size, 8 H100, micro batch 4, per 1K steps | 3.74 (logged steps) and 3.9 to 4.1 with the start, validation and checkpoints | measured: `tabsil-v2-gcfreeze-s1-m-mb4` (`7266d800…`); phase 1B's 30K-step runs, 117 to 123 |
| Stage 1, 32 H100, per 1K steps | 6.31 | v1's record, older code; not measured with today's code |
| Stage 2, 8 H100, micro batch 1, per 1K steps | 21.5 | measured: `tabsil-v2-eff-s2-m-mb1` (`bf797ac1…`) |
| Stage 2, 8 H100, micro batch 2, per 1K steps | 16.2 for a whole job | measured: L20's 5,200 steps (`893293ed…`) |
| Stage 3, 8 H100, per 1K steps | 85 to 100 | estimated from a one-GPU run (`5df481cb…`, 0.906 GPU-seconds per table); not measured on one node: item 3.0 |
| Stage 3, 32 H100, per 1K steps | 144 | v1's record, older code |
| A smoke run of 300 steps | about 0.5 on one GPU, about 3 on 8 H100 | measured (v2-experiment-results, phase 1B) |
| TabArena-Lite, 3-split and the development suite, one set of weights | about 2.5 | measured (phase 1A) |
| One TabArena-Lite run, one H100 | 0.31 | measured (v2-prior-task-results, "TabArena-Lite of the fine-tuned weights") |

Checks that cost under 1% of what they inform (rule 7): 1.1 before 2.3 (under 1%), 1.2 before 2.1 (1.3%, the nearest the check can be made), the smoke runs before 2.2 and 2.3, 3.0 before tier 3 (0.4%). None is skipped.

## 10. Order and code

1. **Code, no GPU:** the conversion of the 250K checkpoint (1.2); the gradient-noise probe (1.1). 2.2 needs no code (a registered prior and an existing training option). 2.3 needs a head that predicts hidden cells from the row's in-context output and the column's summary, its loss and the masking, with tests and a review, before its smoke run.
2. **Tier 1.** Its items are independent and run together. Then R1.
3. **Tier 2**, after R1: 2.1, and 2.2 if the user asked for it, side by side; 2.3 if R1 opened it. Then R2.
4. **Tier 3**, after R2 and only if the user put it in scope: 3.0, then 3.1 to 3.3 in sequence, each read at its end. Then the submission.

No run starts beside the step whose result bears on it (rule 8). The prior code of 2.2 and 2.3 goes to dev-v2. Nothing is adopted or released by the plan; every adoption is put to the user.

## 11. The twelve rules

| Rule of the retrospective | Where this plan meets it |
|---|---|
| 1. The approval question holds the expected gain in the user's unit next to the cost | §15: each question has the expected live Elo next to the H100-h; the tier table of the summary; §3's options |
| 2. A tier that trains nothing; every tier states what it adds | tiers 0 and 1 train nothing; each row of the summary's tier table has "adds +X, evidence: …" |
| 3. The gain below which the rest is not run, and the reading points, fixed before the first run | §8: the reading points and three candidate thresholds; the number is left to the user (§15) |
| 4. A premise that the evidence speaks against goes back to the user first | §3: "the request assumes …; the evidence says …; options", with the compute lever priced |
| 5. Expected effect, resolution and decision by outcome for every run; nothing adopted on "not worse" | §7: the three entries in every row. Candidates below the resolution of their run are listed and not run (§5.6, §6); 2.2 runs only by the user's choice, with the reason stated |
| 6. Costs measured on the setup that runs; a timing job above 100 H100-h; a smoke run for a new prior or setup | §9: every cost names its run or is marked an estimate; 3.0 times stage 3 on one node before it is used; 2.2 and 2.3 have smoke runs |
| 7. A check under 1% of the run it informs comes before the decision | §9, last paragraph |
| 8. No run beside the phase whose result bears on it | §10. 2.1 bears on tier 3, so tier 3 waits for R2 |
| 9. A cost above 130% or a falling expected gain reopens the decision, with "drop" as an option | §8, "Between the points"; 2.1's and 3.0's decision entries |
| 10. "Recommended" only with its expected gain and not below the user's bar; every question has an option to stop | no option of this plan is marked recommended; every question of §15 has "stop" or "skip" |
| 11. A status report begins with the outlook | §8, the four lines; this document starts with the outlook |
| 12. A plan above 500 H100-h is read by an agent that did not write it | §14, dated before any run |

## 12. Relation to the earlier documents

| Document | Used as it is | Replaced or corrected here | Not taken up |
|---|---|---|---|
| [v2-retrospective.md](v2-retrospective.md) | the twelve rules (§11) | — | — |
| [v2-findings.md](https://github.com/Aiden-Jeon/tabsil/blob/3c97ccc5f408e350e542c4803bc71e7b7935c765/training/docs/v2-findings.md) (PR #53) | every measured number of §2 | "1% of error is about 40 Elo" holds for a change of #14's kind. For Elo on the live board the breadth of a change matters more than its G (§4.2): a uniform 0.75% reaches rank 5, while #14's prior gives 4.5 of its 37.6 Elo on its four named datasets | — |
| [v2-experiment-plan.md](v2-experiment-plan.md) | the evaluation protocol (§3), the cost table, the branch rule | its decisions 12 and 13 stand; this plan reopens no phase of it. Its §1.3 compute association is used by analogy (§4.3) | phases 1B to 5 as designed |
| [v2-prior-task-plan.md](https://github.com/Aiden-Jeon/tabsil/blob/9933bd05df52127c3cb03293057b2a2d361200d3/training/docs/v2-prior-task-plan.md) (closed) | its prior candidates (§3), with their expected effects restated in live Elo | its goal (task metrics) is replaced by TabArena-Lite | its tiers 1 to 3 |
| [v2-task-formulations.md](v2-task-formulations.md) | sections 1 to 10, with three checked numbers corrected | section 11 is new | — |
| Codex line (`dev-v2-codex-2`, `Aiden-Jeon/v2-rank-gap`, `v2-prior-oracles`, `v2-task-risks`) | its CPU results on inference changes (§2.4), cited and not re-checked | — | its plan (`top5_research/tabsil-top5-plan.md`) is another agent's and is not merged into this one |

## 13. Sources

Read on 2026-11-28. "Opened" means the cited passage was read for this plan; "subagent" means a subagent read it and the number used was not re-opened; "earlier document" means the number is cited from the document named.

| Source | Used for | Status |
|---|---|---|
| TabArena leaderboard Space, `TabArena/leaderboard` at `a525fe89…`: `website_leaderboard.csv` (Lite and all splits), `data/pages.sqlite` | §1, §4 | opened; decoded by `gap_report.py` |
| TabArena task metadata, `autogluon/tabarena` at `8e55fb9`; OpenML feature lists | §4 properties | read by the script |
| MLflow runs `7cc31c48…`, `138151a2…`, `e2ff88f0…`, `987892ce…` (eval artifacts) and their `leaderboard.csv` | §1.2, §2.1, §4 | opened |
| Kumo-Tabular, [arXiv 2611.23001](https://arxiv.org/abs/2611.23001), Table 2 and §7 | compute, tables, the prior ablation | subagent (Table 2 matches v2-experiment-plan §1.3) |
| LimiX-2, [arXiv 2609.17488](https://arxiv.org/abs/2609.17488) §6; the LimiX-2-1016 model card | the size curve, the objective | subagent |
| Causilo 1.5 blog (nums.world/blog/causilo-1-5); Causilo, [arXiv 2609.22866](https://arxiv.org/abs/2609.22866) | §3 item 5, §5.4 | blog opened; the paper by a subagent |
| Seldon model card (Hugging Face `Neuralk-AI/seldon`) | §5.4 | subagent |
| TabPFN-3.5, [arXiv 2609.17895](https://arxiv.org/abs/2609.17895) | §5.4 | subagent |
| Mitra, [arXiv 2510.21204](https://arxiv.org/abs/2510.21204), Table 6; Mitra-v2, [arXiv 2609.04540](https://arxiv.org/abs/2609.04540) | §5.3 | earlier document (the closed plan opened them) and subagent |
| Real-TabPFN, [arXiv 2507.03971](https://arxiv.org/abs/2507.03971); APT, [arXiv 2502.04573](https://arxiv.org/abs/2502.04573) | §5.3 | subagent |
| Mind the Gap?, [arXiv 2605.06343](https://arxiv.org/abs/2605.06343) | §5.5 | earlier document |
| TabArena, [arXiv 2506.16791](https://arxiv.org/abs/2506.16791), §2.1, §3.1; `autogluon/tabarena` README | the protocol, tuning | subagent; the conflict on bagging is not resolved |

Not published, and so not used: the training compute of Causilo, LimiX-2, Seldon, Lattice, Inpriora, TabFM and TabPFN-3.5; what changed from Causilo 1.0 to 1.5 and from LimiX-2 to LimiX-2-1016; any controlled ablation of a masked-cell objective, of the number of query rows, of several targets per table, or of a ranking loss for tabular in-context models.

## 14. Review

Pending: rule 12's reading by an agent that did not write the plan is run on the draft pull request before it is marked ready, and recorded here with its date and what changed.

## 15. For the user to decide

Every question has its expected gain next to its cost, an option to stop, and no option marked recommended (rules 1 and 10).

1. **Whether the compute lever is in scope.** In scope: tier 3 runs after R2 if 2.1 reads a slope: +94 to +147 live Elo by analogy (rank 4 to 6) for 3,640 to 3,830 H100-h and 19 to 20 days (2,630 and 13.5 days without stage 3). Not in scope: the plan ends at R2, with at most about +43 from 2.1 for about 230 H100-h. Or stop now: #14's weights at rank 11, 0 H100-h. **The user sets.**
2. **The budget cap.** About 4 (tier 1 only), about 230 (tiers 1 and 2.1), about 680 (with 2.2 and 2.3), or about 3,900 to 4,300 (with tier 3), or 0. **The user sets.**
3. **The stop threshold** of §8: the bar, +40 per tier, the resolution, or another number. **The user sets.**
4. **The margin above Kumo-Tabular-Medium** for the bar: 25 live Elo in this plan, about the difference between two seeds. **The user sets.**
5. **Item 2.2**, the prior-and-task branch the request asked for: +0 to about +30 for 223 H100-h, at the resolution of one pair, so outside the tiers unless the user asks for it.
6. **Option D**, a tuned + ensembled entry: +16 to +31 by analogy for about 65 to 500 H100-h, a different kind of entry. Not planned unless the user asks.
7. **Open from before:** whether #14's weights are released or submitted (v2-experiment-plan, phase 5).

## Appendix: the 51 datasets

#14's run against Kumo-Tabular-Medium, the top five's geometric mean, TabPFN-3.5 and v1.1 (error ratios, below 1 is better), with TabSil's rank among the six references and itself. Sorted by the ratio to Kumo-Tabular-Medium. From `per_dataset.csv` of the run in §4.

| Dataset | Task | Train rows | Features | Categorical | Max levels | Missing | Minority | #14 / Kumo-M | #14 / top-5 | #14 / TabPFN-3.5 | #14 / v1.1 | Rank of 7 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| polish_companies_bankruptcy | binary | 3,940 | 64 | 0% | — | 1.2% | 6.9% | 1.989 | 2.458 | 1.562 | 0.706 | 7 |
| hazelnut-spread-contaminant-detection | binary | 1,600 | 30 | 0% | — | — | 50.0% | 1.349 | 1.566 | 1.134 | 0.990 | 7 |
| airfoil_self_noise | regression | 1,002 | 5 | 20% | 27 | — | — | 1.217 | 1.246 | 1.049 | 1.009 | 7 |
| customer_satisfaction_in_airline | binary | 86,587 | 21 | 76% | 6 | 0.0% | 45.3% | 1.200 | 1.252 | 1.232 | 1.012 | 7 |
| anneal | multiclass | 599 | 38 | 84% | 14 | — | 0.9% | 1.129 | 1.101 | 0.892 | 0.956 | 5 |
| miami_housing | regression | 9,184 | 15 | 7% | 2 | — | — | 1.075 | 1.081 | 1.025 | 0.985 | 7 |
| website_phishing | multiclass | 902 | 9 | 100% | 3 | — | 7.6% | 1.069 | 1.076 | 1.075 | 1.003 | 7 |
| coil2000_insurance_policies | binary | 6,548 | 85 | 4% | 39 | — | 6.0% | 1.043 | 1.044 | 0.999 | 0.999 | 6 |
| Marketing_Campaign | binary | 1,493 | 25 | 32% | 8 | 0.0% | 14.9% | 1.038 | 1.026 | 0.992 | 0.980 | 4 |
| physiochemical_protein | regression | 30,487 | 9 | 0% | — | — | — | 1.035 | 1.046 | 1.045 | 1.007 | 7 |
| diamonds | regression | 35,960 | 9 | 33% | 8 | — | — | 1.035 | 1.043 | 1.018 | 1.004 | 7 |
| Diabetes130US | binary | 47,679 | 47 | 83% | 759 | 4.1% | 8.8% | 1.030 | 1.045 | 1.042 | 0.999 | 7 |
| qsar-biodeg | binary | 703 | 41 | 12% | 11 | — | 33.7% | 1.026 | 1.033 | 0.961 | 0.999 | 5 |
| online_shoppers_intention | binary | 8,220 | 17 | 41% | 20 | — | 15.5% | 1.023 | 1.023 | 1.010 | 1.002 | 7 |
| jm1 | binary | 7,257 | 21 | 0% | — | 0.0% | 19.3% | 1.023 | 1.037 | 1.031 | 0.996 | 7 |
| Amazon_employee_access | binary | 21,846 | 9 | 100% | 7518 | — | 5.8% | 1.022 | 1.090 | 1.086 | 1.019 | 7 |
| superconductivity | regression | 14,175 | 81 | 0% | — | — | — | 1.021 | 1.029 | 0.992 | 0.996 | 6 |
| Food_Delivery_Time | regression | 30,301 | 9 | 33% | 1320 | — | — | 1.020 | 1.020 | 1.018 | 1.003 | 7 |
| maternal_health_risk | multiclass | 676 | 6 | 0% | — | — | 26.8% | 1.020 | 1.050 | 0.979 | 0.988 | 6 |
| houses | regression | 13,760 | 8 | 0% | — | — | — | 1.017 | 1.028 | 1.023 | 0.999 | 7 |
| blood-transfusion-service-center | binary | 499 | 4 | 0% | — | — | 23.8% | 1.014 | 1.007 | 1.007 | 1.002 | 5 |
| concrete_compressive_strength | regression | 687 | 8 | 0% | — | — | — | 1.014 | 1.028 | 0.974 | 1.012 | 6 |
| QSAR-TID-11 | regression | 3,828 | 1024 | 0% | — | — | — | 1.012 | 1.023 | 0.995 | 1.000 | 5 |
| MIC | multiclass | 1,133 | 111 | 85% | 10 | 3.5% | 0.7% | 1.012 | 1.039 | 1.049 | 0.994 | 7 |
| HR_Analytics_Job_Change_of_Data_Scientists | binary | 12,772 | 12 | 75% | 123 | 0.0% | 24.9% | 1.011 | 1.014 | 1.005 | 0.994 | 7 |
| NATICUSdroid | binary | 4,994 | 86 | 100% | 2 | — | 35.0% | 1.010 | 1.019 | 0.998 | 0.992 | 6 |
| bank-marketing | binary | 30,141 | 13 | 54% | 12 | — | 11.7% | 1.008 | 1.036 | 1.003 | 0.985 | 7 |
| students_dropout_and_academic_success | multiclass | 2,949 | 36 | 47% | 46 | — | 17.9% | 1.008 | 1.009 | 0.994 | 1.003 | 6 |
| Another-Dataset-on-used-Fiat-500 | regression | 1,025 | 7 | 14% | 3 | — | — | 1.008 | 1.017 | 1.021 | 1.001 | 7 |
| in_vehicle_coupon_recommendation | binary | 8,456 | 24 | 88% | 25 | — | 43.2% | 1.007 | 1.010 | 1.057 | 0.983 | 6 |
| credit_card_clients_default | binary | 20,000 | 23 | 13% | 7 | — | 22.1% | 1.005 | 1.009 | 1.006 | 1.000 | 7 |
| kddcup09_appetency | binary | 33,333 | 212 | 18% | 15416 | 63.5% | 1.8% | 1.004 | 1.013 | 0.941 | 0.983 | 5 |
| E-CommereShippingData | binary | 7,333 | 10 | 40% | 5 | — | 40.3% | 1.001 | 1.006 | 1.006 | 0.994 | 7 |
| wine_quality | regression | 4,331 | 12 | 8% | 2 | — | — | 1.001 | 1.001 | 1.002 | 1.000 | 6 |
| credit-g | binary | 667 | 20 | 65% | 10 | — | 30.0% | 1.000 | 1.012 | 1.007 | 0.994 | 7 |
| GiveMeSomeCredit | binary | 100,000 | 10 | 0% | — | 2.2% | 6.7% | 1.000 | 1.008 | 1.012 | 1.004 | 7 |
| Is-this-a-good-customer | binary | 1,149 | 13 | 62% | 22 | — | 11.4% | 1.000 | 1.015 | 1.007 | 1.004 | 6 |
| heloc | binary | 6,973 | 23 | 0% | — | — | 47.8% | 0.999 | 1.004 | 1.006 | 1.002 | 6 |
| Bank_Customer_Churn | binary | 6,667 | 10 | 40% | 3 | — | 20.4% | 0.998 | 1.002 | 0.997 | 1.004 | 4 |
| hiva_agnostic | multiclass | 2,563 | 1617 | 100% | 2 | — | 1.1% | 0.997 | 0.995 | 0.997 | 0.999 | 2 |
| QSAR_fish_toxicity | regression | 605 | 6 | 0% | — | — | — | 0.997 | 0.998 | 0.999 | 1.002 | 2 |
| healthcare_insurance_expenses | regression | 892 | 6 | 50% | 4 | — | — | 0.997 | 0.997 | 0.994 | 1.000 | 1 |
| Fitness_Club | binary | 1,000 | 6 | 50% | 7 | 0.2% | 30.3% | 0.996 | 0.999 | 0.994 | 1.003 | 3 |
| diabetes | binary | 512 | 8 | 0% | — | — | 34.9% | 0.994 | 0.995 | 1.045 | 0.986 | 3 |
| churn | binary | 3,333 | 19 | 21% | 51 | — | 14.1% | 0.992 | 1.021 | 0.974 | 0.956 | 5 |
| seismic-bumps | binary | 1,723 | 15 | 20% | 3 | — | 6.6% | 0.984 | 0.981 | 0.947 | 1.001 | 1 |
| Bioresponse | binary | 2,501 | 1776 | 0% | — | — | 45.8% | 0.967 | 0.982 | 0.956 | 0.998 | 2 |
| splice | multiclass | 2,127 | 60 | 100% | 6 | — | 24.0% | 0.966 | 1.022 | 0.982 | 1.009 | 5 |
| SDSS17 | multiclass | 52,035 | 11 | 18% | 6264 | — | 17.9% | 0.965 | 0.996 | 1.011 | 1.001 | 3 |
| taiwanese_bankruptcy_prediction | binary | 4,546 | 94 | 0% | — | — | 3.2% | 0.955 | 0.973 | 0.907 | 0.999 | 2 |
| APSFailure | binary | 50,667 | 170 | 0% | — | 8.3% | 1.8% | 0.946 | 0.969 | 0.702 | 1.015 | 3 |
