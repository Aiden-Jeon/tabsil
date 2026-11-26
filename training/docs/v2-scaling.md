# v2 scaling: model size or prior

Status: design. No run of this plan has started. Every number below was read from MLflow on 2026-11-26 or computed from the model config; nothing was trained or evaluated for this document.

## Question

Which axis would raise TabSil's TabArena classification and regression scores more: a larger model, or the prior? The v2 plan keeps v1's size ([v2-task-priors.md](v2-task-priors.md), open item 9). This document checks that decision against the existing training records and designs a size experiment with a rule for when to change it.

Criteria, fixed before reading the records:
- **Capacity limit:** the prior validation loss is also flat between 150K and 225K/250K steps. The next experiment is a larger model.
- **Prior mismatch:** the prior validation loss keeps falling while TabArena-Lite stays flat. The model learns more of the prior, and it does not carry over to real tables. Fixing the prior comes first.
- If the records show both, say so, and claim only what the numbers show.

## What the training records show

### Runs

All runs are in the MLflow experiment `/Users/jongseob.jeon@databricks.com/pfn-cookbook/tabpfn-v3-reproduction` (id 2345355994784185), written by minipfn. Both training runs have the v1 structure (`emsize` 256, `nlayers` 24, 6 column/row block pairs, `n_cls` 4; logged `n_parameters` 221,548,733), the `graph_scm` prior, batch 256 on 32 H100, lr 8e-4 with 1% warmup and cosine decay to 0 at the run's last step.

| Run | MLflow runs (chunks) | Steps |
|---|---|---|
| 150K (the v1 stage 1, Hub `stage1-150k`) | `dc047c3d2d304b1ea0bbdbea2a75bf31` (0-75K), `064d9d34abee45e79c978e55de42f8b0` (75K-150K) | 150,000 |
| 250K | `4f23ca16371f4032a4442d4ad4228884` (0-76K), `a50110762f3e4e4c9b594763e2157780` (76K-151K), `72c0697c213345bfbfe1fdb2e4d3483d` (151K-227K), `32b8195f4f794208a3425878e5c3fcf3` (227K-250K) | 250,000 |
| v1 stage 2 (from the 150K run) | `bd74e8010dcf4e62bc2be166cc27424f` | 10,000 |

Metrics:
- `val_loss_classification` (CE) and `val_loss_regression` (pinball loss): fixed prior validation tables, every 5K steps.
- `train_loss_classification` and `train_loss_regression`: every 100 steps. Every step draws new prior tables, so the train loss is also a held-out loss on the training distribution. Its level differs from the validation loss and is not compared with it.
- TabArena-Lite: `tabarena_lite_elo`, `tabarena_lite_classification_elo` and `tabarena_lite_regression_elo` of the eval runs `*-eval-tabarena-lite`. These use the repository harness without bagging (`bagging` 0), and 0% of splits are imputed in every run below. The 95% CIs come from the runs' `eval/tabarena-lite/comparison.csv`.
- AMLB: `classification_mean_roc_auc` and `regression_mean_neg_rmse` of the eval runs `*-eval-benchmark`. The second metric is a plain mean over datasets, so the datasets with large targets dominate it. The per-dataset RMSE ratio quoted on the model card (1.136 at 150K, 1.137 at 225K) is not logged as a metric.

### Prior loss, train loss and benchmarks per checkpoint

Validation loss is classification / regression. lr is the scheduled lr at that step. TabArena-Lite is overall (classification, regression).

| Step | 150K run: lr | val loss | TabArena-Lite | 250K run: lr | val loss | TabArena-Lite | AMLB ROC AUC / mean −RMSE (250K run) |
|---|---|---|---|---|---|---|---|
| 25K | 7.5e-4 | 0.2144 / 0.05480 | 1357.0 (1399.0, 1326.1) | 7.8e-4 | 0.2135 / 0.05459 | 1392.6 (1431.8, 1374.6) | 0.9335 / −1697.7 |
| 50K | 6.1e-4 | 0.2091 / 0.05247 | 1399.1 (1477.1, 1267.7) | 7.3e-4 | 0.2098 / 0.05319 | 1390.5 (1482.3, 1220.6) | 0.9351 / −1711.7 |
| 75K | 4.1e-4 | 0.2038 / 0.05132 | 1421.1 (1525.2, 1234.7) | 6.4e-4 | 0.2056 / 0.05204 | 1411.6 (1508.6, 1241.0) | 0.9354 / −1684.2 |
| 100K | 2.0e-4 | 0.2020 / 0.05030 | 1428.6 (1539.3, 1228.9) | 5.3e-4 | 0.2029 / 0.05118 | 1419.4 (1524.8, 1227.9) | 0.9371 / −1700.3 |
| 125K | 5.5e-5 | 0.1999 / 0.04961 | 1438.2 (1563.7, 1205.3) | 4.1e-4 | 0.2039 / 0.05069 | 1413.4 (1516.0, 1224.9) | 0.9369 / −1669.8 |
| 150K | 0 | **0.1999 / 0.04945** | **1429.2 (1554.7, 1187.3)**, CI +104/−107 | 2.8e-4 | 0.2021 / 0.05023 | 1419.8 (1522.9, 1234.9) | 0.9376 / −1677.1 |
| 175K | | | | 1.7e-4 | 0.2001 / 0.04974 | 1430.7 (1553.1, 1199.0) | 0.9382 / −1658.5 |
| 200K | | | | 7.8e-5 | 0.1989 / 0.04916 | 1432.1 (1552.2, 1208.1) | 0.9380 / −1662.8 |
| 225K | | | | 2.0e-5 | 0.1985 / 0.04880 | 1422.7 (1541.2, 1195.0) | 0.9383 / −1651.6 |
| 250K | | | | 0 | **0.1984 / 0.04874** | **1428.0 (1549.0, 1197.9)**, CI +94/−110 | 0.9383 / −1667.7 |

AMLB of the 150K run at 150K (`3f7135791fc342598ad389722020843f`): 0.9383 / −1658.5. Its earlier checkpoints have no AMLB run (not found).

Train loss, mean of the logged values over the last 10K steps (classification / regression): 0.2956 / 0.05685 for the 150K run (140K-150K) and 0.2929 / 0.05624 for the 250K run (240K-250K).

Eval run ids, in step order:
- 150K run, TabArena-Lite: `2769dcc2…`, `7f2f4ad0…`, `e34c205e…`, `8688a464…`, `d065b226…`, `b3c0f0dd978941428d7a6ed465dca7cb`
- 250K run, TabArena-Lite: `40f5801a…`, `8d82ca93…`, `1d2ed9db…`, `9d41cde2…`, `b9fe36bc…`, `47f6ad51…`, `800ad487…`, `fd67045b…`, `88e30010df9f43cc899233289e0689f6`, `4241987375b04fa9bcfb1f70e905a6ca`
- 250K run, AMLB: `89403cde…`, `9331d03d…`, `54aa02a0…`, `9d166aaa…`, `809f395b…`, `9b07a02f…`, `5a780392…`, `64a32039…`, `535f7a13…`, `af3ae175764b4b5aab146b6f6af695b3`

For scale, the stage 2 records:
- The v1 stage 2 reached TabArena-Lite 1693.1 at 5K (`2889d6e9…`) and 1724.9 at 10K (`81a9845f…`), about 300 above its stage 1.
- #14's `graph_scm_hicard_missing` stage 2 reached 1769.1 against 1727.1 for `stage2-10k`, +42 (`a0a2e4e5…` against `f2ca29bf…`, experiment `pfn-cookbook/tabsil`).

### Runs at other sizes

No run of this structure at another size has an evaluation (not found).
- In the minipfn experiment every run past 300 steps has 212.9M-225.6M parameters. These are layout variants (`kumo110-*`, the v3.5 layout with `emsize` 128 and `n_cls` 8), not size steps.
- The only smaller runs are the 300-step timing runs of the older v3 layout (`7bcc2943…` and `6b948fb5…`, 52.1M and 57.3M) and have no eval.
- The TabPFN, v2 and v2.5 reproductions (`pfn-cookbook/tabpfn-*-reproduction`, 7.2M-25.8M) are another architecture, with separate classification and regression models. They cannot place a size curve for this one.

## Verdict: prior mismatch; capacity is not tested

**The prior loss is not flat.** Both runs end with lr 0. The 250K run ends 0.0015 lower in classification (0.1984 against 0.1999, −0.8%) and 0.0007 lower in regression (0.04874 against 0.04945, −1.4%). The train loss falls about as much (−0.9% and −1.1%).
- Within one run the evaluation noise is small: the last five validations of the 250K run (230K-250K, lr below 1.3e-5) vary by at most 0.0001 and 0.00006.
- Between the two runs it is larger. At 5K-20K steps, where the two lrs differ by less than 3%, their validation losses differ by up to 0.0025 (classification) and 0.00045 (regression). So the classification gain lies within run-to-run differences, and the regression gain is just above them.
- 1.67× the steps bought about 1% of prior loss. More steps at this size are close to what they can give.

**TabArena-Lite and AMLB do not move.**
- TabArena-Lite is 1428.0 against 1429.2, with CIs of about ±100. Classification is 1549.0 against 1554.7, regression 1197.9 against 1187.3.
- AMLB ROC AUC is 0.9383 in both.

**Within a run, prior loss and the real-table score part ways.**
- From 75K to 250K in the 250K run, the validation loss falls 3.5% (classification) and 6.3% (regression). TabArena-Lite rises by 16 Elo, which is within the CI: classification +40, regression −43.
- From 25K to 250K the regression validation loss falls 10.7% while regression Elo falls from 1374.6 to 1197.9 (−177). The 150K run shows the same: from 1326.1 at 25K to 1187.3 at 150K.
- So in stage 1, the better the model fits the regression prior, the worse it does on real regression tables. That is the prior-mismatch case of the criteria.
- Part of it is stage 1's 1,024-row tables: the v1 stage 2 brings regression Elo to 2074.2 (`81a9845f…`).

**The data changes are what moved TabArena at a fixed size.**
- Stage 2's longer tables: about +300 Elo.
- #14's prior: +42 in 10K steps.
- 100K more stage-1 steps: −1.

**What the records cannot show.** No run varies the size, so it is unknown whether a larger model would lower the prior loss. It is also unknown whether a lower loss would then reach TabArena.
- The small gain from 1.67× the steps fits a model near its capacity at this size. A capacity limit is therefore not ruled out.
- But at stage 1 a lower prior loss carried over weakly (classification) or negatively (regression). So a lower prior loss from size alone is not expected to move TabArena much.
- LimiX-2 leads the TabArena leaderboard with 24 blocks at width 256, close to v1's numbers. Its log-linear size curve (12.5M-406M) also changes structure and prior, so it does not say which of the two TabSil lacks.

**Consequence.** Open item 9 stands for now: the prior and objective axes come first. The size experiment below is built as a cheap check first, and its expensive step runs only if the cheap one shows that size moves the prior loss.

The same records answer part of open item 8 (screening length):
- The two runs have the same structure and prior, and at 25K nearly the same lr. Their TabArena-Lite still differs by 36 at 25K (1357.0 against 1392.6), 9 at 50K and 10 at 75K.
- From 25K to 75K they rise by 64 (150K run) and 19 (250K run). From 75K on, both stay within 1411-1438 up to 250K.
- So a 30K stage-1 screen sits on the rising part of the curve and has a run-to-run gap of about 35 Elo. Arm differences smaller than that need the per-dataset paired comparison or the prior loss. The prior loss differed between the runs by at most 0.0025 at the same early step.

## Scaling experiment

### Sizes

The grid scales the width `emsize` with everything else at `STRUCTURE`:
- 24 row ICL layers, 16 heads, 6 column/row block pairs, 4 CLS tokens, so the row ICL width is 4 × `emsize`.
- Width scales every stage: the cell stages, which do most of the stage-1 compute, and the row ICL, which holds most of the parameters.
- All heads divide their widths, and the row-stage head dim stays even (`ColumnRowTransformerConfig.__post_init__`).

Parameters come from `ColumnRowTransformer(config).num_parameters_by_stage()` on the meta device. The v1 count equals the logged `n_parameters`.

The FLOPs ratio is the analytic forward matmul and attention FLOPs per table, relative to v1:
- Blocks have 8·dim² parameters, which the counts confirm.
- "Stage 1" is 1,024 rows with 50 and with 100 features and a train share of 0.6.
- "Inference" is 10,000 rows, 50 features and a train share of 0.8.
- Backward is a fixed multiple of forward, so the training ratio is the same.

| Arm | `emsize` | Row ICL width | Parameters | of which row ICL | Cell stages (column + row) | FLOPs ratio, stage 1 | FLOPs ratio, inference |
|---|---|---|---|---|---|---|---|
| XS | 128 | 512 | 56.0M | 50.4M | 2.6M | 0.29 | 0.37 |
| S | 192 | 768 | 125.1M | 113.3M | 5.6M | 0.59-0.60 | 0.65 |
| M (v1, = C) | 256 | 1,024 | 221.5M | 201.4M | 9.9M | 1.00 | 1.00 |
| L | 384 | 1,536 | 496.7M | 453.1M | 21.9M | 2.12-2.13 | 1.89 |
| L-depth (optional) | 256, `nlayers` 48 | 1,024 | 422.9M | 402.8M | 9.9M | 1.21-1.35 | 1.60 |

- The four width points are about 2× apart in parameters, so three intervals fit a log-linear curve.
- 91% of v1's parameters are in the row ICL. Yet the cell stages do 65-79% of the stage-1 FLOPs, because they run on every cell and not on every row. Depth therefore adds parameters cheaply in stage 1.
- L-depth runs only if L passes step A, to find out whether depth buys the same for less (see the rules).

### Training lengths

Three levels. Each runs only if the previous one passes.

| Step | Arms | Training | Readout |
|---|---|---|---|
| A | XS, S, M, L | stage 1 from scratch, 30K steps (the phase 1 length) | prior validation loss at the end (lr 0); TabArena-Lite and AMLB as guards |
| B | M, L | stage 1 75K steps, then stage 2 5K steps from it | TabArena-Lite and AMLB after stage 2; prior loss of both stages |
| C | the chosen size | v1's curriculum 150K / 10K / 2K | the v2 run itself (phase 3 of v2-task-priors.md) |

Why these lengths:
- Step A is stage 1 only, as phase 1 is, so M is phase 1's C (see sharing below).
- Its readout is the prior loss, not TabArena. In stage 1 the regression Elo moved against the prior loss, and the whole stage-1 Elo range from 25K to 250K (1357-1438) is smaller than one CI.
- Step B reads TabArena after a short stage 2. Stage 2 is where the real-table scores are made (+300 for v1), and v1's 5K checkpoint already had 1693 of its 1725.
- 75K is where v1's stage-1 TabArena-Lite reached its plateau (1421 and 1412 in the two runs, against a maximum of 1438).

All arms use the cosine schedule of their own length. A 30K point cannot be read from a longer run, because its lr has not decayed.

### Prior and recipe

- Prior `graph_scm_hicard_missing` (#14, on dev-v2), the same as phase 1's C, with no task prior. Size is the only difference from C.
- The recipe is `curriculum(1, steps=30_000)` in step A, and `curriculum(1, 75_000)` then `curriculum(2, 5_000)` in step B. Batch 256, lr 8e-4 / 1e-4, `filter_unpredictable` on.
- Muon scales each matrix's lr by 0.2·sqrt(max(rows, cols)) (`match_rms_adamw`), so the update RMS stays the same across widths, and lr 8e-4 is kept for every size. This was not tested at another width; see the risks.
- Micro batch: v1's stage 1 used 62-63 GB per H100 at micro batch 4 (`gpu_mem_gb`, `064d9d34…`). L's cell activations are 1.5× as large, so L uses micro batch 2. The batch stays 256, so the math is unchanged. XS and S keep 4.

### Evaluation

For every run, at its last step:
- **Prior validation loss**, classification and regression, logged by the training run on the same fixed tables for every size.
- **TabArena-Lite**: overall, classification and regression Elo, plus the per-dataset error ratio against M (the #14 comparison).
- **AMLB**: ROC AUC and the regression RMSE ratio.
- **Guards** as in [v2-task-priors.md](v2-task-priors.md) (Base prior and guards), against M at the same tabsil inference version. A run fails if either holds:
  - TabArena-Lite overall, classification or regression Elo is lower beyond its 95% CI.
  - AMLB ROC AUC or the regression RMSE ratio is worse by more than 0.005.
- The inference time logged by TabArena-Lite (`tabarena_lite_*_time_infer_s_mean`), for the trade-off below.

### Cost

The basis is v1's throughput: stage 1 ran 150K steps in about 30 h on 32 H100 (0.71 s per step, `064d9d34…`), so 30K steps take about 192 H100-h and 75K about 480. Stage 2 ran 10K steps in 9.58 h on 32 H100 (`bd74e801…`), so 5K steps take about 153 H100-h.

Each arm scales these by its FLOPs ratio. For L's stage 2 the ratio is taken as 2.0, between its stage-1 and 10,000-row values.

| Step | Run | H100-h |
|---|---|---|
| 0 | a 300-step timing run per size on 8 H100, as `7bcc2943…` | under 5 in total |
| A | XS 30K | ≈ 56 |
| A | S 30K | ≈ 115 |
| A | M 30K | 0 extra: phase 1's two C seeds |
| A | L 30K | ≈ 409 |
| A | L-depth 30K (optional) | ≈ 232-259 |
| | **Step A total** | **≈ 580** (≈ 830 with L-depth) |
| B | M 75K + stage 2 5K | ≈ 480 + 153 = 633 |
| B | L 75K + stage 2 5K | ≈ 1,022 + 306 = 1,328 |
| | **Step B total** | **≈ 1,960** |
| C | v2 at L instead of M | ≈ 3,200 instead of 1,550 |

- The data loader is not the limit (`data_wait_frac` 0.001 in stage 1), so the FLOPs ratio is the right first guess for L.
- XS and S are more likely to be held back by non-matmul work and may not be as cheap as their ratio. Step 0 replaces every estimate with a measured seconds per step before step A is approved.
- Each TabArena-Lite and AMLB evaluation took 0.09 h (8 shards) and 0.31 h of wall time for #14 (`a0a2e4e5…`, `c1cd42f9…`). That is a few H100-h at most, and it is left out of the totals.

### Decision rules

**Step A** (read at 30K, lr 0). Let s be the spread of the two C seeds' final prior validation loss, for classification and for regression separately.
- **Steep: run step B.** L's loss is below both C seeds by more than s, in classification and in regression. XS, S and M fall in order, so size acts on the prior loss at all. L's guards hold against C.
- **Flat: stop.** L is within s of C. Then capacity does not limit the prior loss at this length; open item 9 stays as it is, and the prior and objective axes remain.
- **Mixed** (one task gains and the other does not): report it. Step B runs only on a separate decision.
- In every case, record whether S is within s of C on the prior loss and within the guards on TabArena-Lite. A model with 0.56× the parameters and 0.65× the inference FLOPs at the same accuracy is worth a separate decision. It does not change v2 by itself.
- If L passes, L-depth (optional) shows whether depth buys the same for less (1.2-1.35× the stage-1 cost against 2.1×).

**Step B** (read after the 5K stage 2). Propose lifting open item 9 only if all hold:
- L's TabArena-Lite overall Elo is at least 40 above M's. 40 is what #14's prior change bought at a fixed size for about 306 H100-h. A size change costs about twice as much for every later run and for every user's inference, so it has to buy at least as much as one prior change did.
- L's classification and regression Elo are not lower than M's, and the per-dataset geometric-mean error ratio of L to M is below 1 for binary, multiclass and regression.
- The guards hold.

Otherwise keep v1's size. The prior and objective axes remain the v2 axes, and a larger model stays a v3 candidate (open item 9).

**If item 9 is lifted:**
- v1 is no longer the clean control of the final run, because the size and the prior would both differ.
- The size effect is then the step B pair (M and L on the same prior), and the prior effect is phases 1-2 at M.
- Phase 3 would still be compared with v1 on the success criteria and the guards. The model card would state both changes.

### Trade-off: inference cost

A larger model makes every user's `fit`/`predict` slower, including plain classification and regression, not just v2's new tasks.
- **Compute.** L is about 1.9× v1's FLOPs per table at 10,000 rows (2.1× at 1,024). On tables where inference is compute-bound, `fit`/`predict` takes about twice as long. For reference, v1's stage-1 checkpoint took a mean 7.2 s per classification split on TabArena-Lite (`tabarena_lite_classification_time_infer_s_mean`, `b3c0f0dd…`). Step A's L run logs the same metric and replaces this estimate.
- **Memory.** The cell state (batch × rows × columns × `emsize`) is 1.5× larger. Chunked inference starts moving it to the CPU (`interleaved_state_device_bytes`, 16 GiB) at about two-thirds of today's table size.
- **Weights.** The download is 2.2× larger: 496.7M against 221.5M parameters.
- **Smaller models.** S and XS go the other way (0.65× and 0.37× the inference FLOPs). They matter only if they hold accuracy.

### Sharing with v2-task-priors phase 1

- Step A's M is phase 1's C: both are v1 size, from scratch, stage 1 only, 30K steps, prior `graph_scm_hicard_missing` and no task prior. Both C seeds serve both plans, and their spread is the noise for step A.
- This holds only if every step A arm uses C's config except `emsize` (and L's micro batch), and the same evaluation at the same tabsil inference version.
- Step A can run beside phase 1 or after C. Its arms do not depend on the task priors.
- Step B's M run (75K + 5K stage 2 on the base prior) is not part of phase 1. It also gives open item 8 a measured gap between a 30K and a 75K stage 1 on the new prior.

### Before step A (code)

- `training/scripts/train.py` always builds `STRUCTURE` ("The model is always the TabSil structure"). A resume or `--init-run-id` must have the preset's structure.
- Step A needs structure overrides (for example `--emsize`, `--nlayers` and `TABSIL_EMSIZE`, `TABSIL_NLAYERS`). They are already `TrainConfig` fields, so `run_architecture` reads them back from the run params.
- Step B needs the stage 2 source check to compare against the source run's own structure.
- Checkpoints store the model config (`tabsil.models.save_checkpoint`), so `training/scripts/eval.py` loads any size without a change.
- This is a small PR to dev-v2, before step A.

### Risks

- **lr.** lr 8e-4 was tuned at v1's size. Suppose L comes out flat with a train loss above M's at the same steps. Then lr cannot be told apart from capacity. One L run at lr 4e-4 (30K, ≈ 409 H100-h, separate approval) separates the two.
- **Training length.** A 30K-step model is limited by training length before capacity (open item 9). A flat step A can miss a limit that appears at 150K. Larger models usually learn faster per step, so a size gain that is real at 150K should show as a lower prior loss at 30K, but that is not guaranteed. Step C checks again at full length if item 9 is lifted.
- **One seed per arm.** Except for M, each arm has one seed, and step B uses step A's C spread as its noise. The +40 Elo threshold of step B is below the CI of a single comparison. The paired per-dataset ratio carries the decision.
- **Prior loss and TabArena.** If L lowers the prior loss without moving TabArena in step B, that repeats the prior mismatch found above. The result then points to the prior, not to size.
