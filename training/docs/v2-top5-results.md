# v2 top-5 plan: results

Results of the runs of [v2-top5-plan.md](https://github.com/Aiden-Jeon/tabsil/blob/Aiden-Jeon/v2-top5-plan/training/docs/v2-top5-plan.md) (PR #54), in the order of its §10. The user approved the plan on 2026-11-29 ("실행") with the compute lever in scope by path 3A, a budget cap of about 2,150 H100-h and the stop threshold at the bar (Kumo-Tabular-Medium + 25 live Elo). Every run is in the MLflow experiment `/Users/jongseob.jeon@databricks.com/pfn-cookbook/tabsil`. GPU time is counted from the job records (run time × GPUs).

## Outlook

| | |
|---|---|
| Best weights so far | #14's run: 1816.9 live, rank 11 (the plan's §1.2) |
| The bar | Kumo-Tabular-Medium + 25 = 1949.3 live, +132.4 over #14 |
| Where the plan is expected to end | tier 3's scenario +75 to +117 by analogy if R2 passes; R2 passes only if 2.1 measures +49 or more |
| Spent | 0.9 H100-h, tier 1 (job records); item 2.1 running since 2026-11-28 15:51 UTC |
| Left of the cap | about 2,149 of 2,150 H100-h; item 2.1 is planned at 225 |

## Code before the runs

- **The 250K-step stage 1 as a start for stage 2** (item 1.1; PR #55). `training/hub/convert_minipfn.py` applies the config mapping PR #3 removed and checks every tensor bit for bit. Converting the 150K run's `model/final.pt` (`064d9d34…`) gives exactly the weights of `hf://aiden-jeon/tabsil@stage1-150k` (weights digest `50105695…`), so the conversion is the one that produced the published branch. The 250K run's `model/final.pt` (`32b8195f…`, MLflow experiment `tabpfn-v3-reproduction`) is converted and logged as `tabsil-stage1-250k`, `runs:/1c6ff838472b49cfa15263e7f84282d1/model` (digest `04d8b862…`); the weights are not on the Hub. `train.py --init-hub` takes such a `runs:/` folder.

## Tier 1

### 1.1: the 250K-step stage 1, capped at 1,024 rows

The converted 250K-step stage 1 against `stage1-150k`, both read with the training rows capped at 1,024 (`--subsample-train-rows 1024`) and the inference of #14's runs (`--member-kinds numeric-cycle --categories counts`, 8 members), on TabArena 3-split (51 datasets, 3 splits each) and the development suite (48 datasets, 10 splits each). The 150K runs are phase 0.6's (`151d71dd…` and `089d674c…`), the 250K runs `581c5064…` and `35e8771d…`. Paired with `training/scripts/paired_report.py`; no split imputed.

| Suite | Datasets (splits) | G (95% CI) | Wilcoxon p | Better / worse | Repo Elo, capped |
|---|---|---|---|---|---|
| TabArena 3-split | 51 (153) | 0.987 (0.980 to 0.994) | 1.1e-05 | 41 / 10 | 1222 → 1239 |
| Development suite | 48 (480) | 0.986 (0.973 to 0.996) | 0.005 | 33 / 15 | — |
| **Pooled** | 99 (633) | **0.987 (0.980 to 0.993)** | 4.0e-07 | 74 / 25 | — |

Pooled by task: classification 0.987 (0.976 to 0.995), regression 0.986 (0.978 to 0.993); binary 0.989 (0.978 to 1.000), multiclass 0.982 (0.959 to 0.999).

**Reading.** G is 0.987 pooled with its CI below 1, so by the plan's rule item 2.1 runs. The difference is 2.6 to 13 times the noise of two stage-1 seeds (0.1 to 0.5% pooled, v2-findings §4). This reads stage 1 alone, on tables of at most 1,024 training rows: the 100K more steps of stage 1 are not flat there. How much of it carries through stage 2 to the full context is item 2.1's question; the capped Elo is not a forecast of it.

GPU time: 312 s on 8 H100 (3-split) and 764 s on one H100 (development suite), 0.90 H100-h.

### 1.2: the real-table corpus and its leakage check

«pending»

### 1.3: the board

Read on 2026-11-29: the Space `TabArena/leaderboard` is at revision `a525fe89…` (last modified 2026-11-27 23:36 UTC), the revision the plan's numbers come from. The plan's §1 stands: #14's weights at 1816.9 live, rank 11; Kumo-Tabular-Medium at 1924.3 in the same fit; the bar at 1949.3.

## Tier 2

### 2.1: stage 2 from the 250K-step stage 1

Running. Job `770158738861775`, MLflow run `tabsil-top5-2-1-s2-from-250k`, started 2026-11-28 15:51 UTC: stage 2 on `graph_scm_hicard_missing`, 10K steps, seed 0, from `runs:/1c6ff838…/model`, on one 8xH100 node with batch 256 at micro batch 1 (32 accumulation steps; #14's run had 8 on 32 GPUs) and #14's weight average (10 snapshots, every 100 steps). Everything else is the stage-2 preset, as in #14's run and N10. It runs in two chunks of 5,000 steps (`TABSIL_STOP_STEP`), each with a timeout of 18 hours; the second resumes the first's run.
