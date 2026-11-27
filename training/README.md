# tabsil-train

Training, the synthetic-data prior, and the benchmark evaluation of [TabSil](../README.md). This is a uv workspace member of the TabSil repository: it depends on the `tabsil` inference package (`src/tabsil/`) and is imported as `tabsil_train`.

| Path | Contents |
|---|---|
| `src/tabsil_train/priors/` | the TabICLv2 graph SCM prior (`tabicl` package), table size rules, transforms; `graph_scm_hicard_missing` adds high-cardinality categorical columns and structured missing values, and `graph_scm_realism` adds Kumo-Tabular's other realism parts on top; `graph_scm_base_realism` has only Kumo-Tabular's every-stage parts (`--prior-type`) |
| `src/tabsil_train/training/` | training config (`TrainConfig`), the training loop, Muon (and Muon + AdamW), the reference validation set, losses, stability, diagnostics |
| `src/tabsil_train/benchmarks/` | AMLB / CTR23, TabArena, BeyondArena, uncertainty metrics |
| `src/tabsil_train/recipe.py` | the TabICLv2 three-stage curriculum (`curriculum(stage)`) |
| `src/tabsil_train/hub.py` | MLflow run checkpoints (`runs:/...`) and the model structure in run params |
| `scripts/` | `train.py`, `eval.py` and their Databricks AI Runtime job specs (`*.yaml`); `paired_report.py`: the paired per-dataset error ratio of two eval runs (v2 plan §3); `dev_suite_overlap.py`: the AMLB / TabArena overlap behind `--suite dev` |
| `scripts/v2_prior_diag/` | diagnostics of the v1 weights for the v2 prior (forecast extrapolation, calibration, contaminated context); `attention_temperatures.py`: the learned attention temperatures of saved weights |
| `scripts/v2_eff/` | step-time measurements for the v2 plan's GPU cost: `step_probe.py` (runs `train.py` and times the optimizer step, each GPU's data wait and the garbage collector), `timing.yaml` (its job spec), `step_breakdown.py` (the step time by part from a run's records) |
| `docs/` | design notes (`v2-experiment-plan.md`: the v2 experiment plan, with the check of the two notes it replaces; `v2-experiment-results.md`: the results of its runs, by phase; `v2-task-priors.md`: task priors and the diagnostics of the v1 weights; `v2-scaling.md`: model size and the stage-1 records) |
| `hub/` | `push_to_hub.py`, `sync_hub_card.py`: upload weights from a training run and the model card |
| `tests/` | training, prior, evaluation and upload tests |

From the repository root:

```bash
uv sync --package tabsil-train
uv run --package tabsil-train pytest training/tests
PYTHONHASHSEED=0 uv run --package tabsil-train python training/scripts/train.py --config-name smoke

# TabArena / BeyondArena (pandas 2.3, without the dev group)
uv sync --package tabsil-train --no-dev --extra arena --group arena-dev
```

The Databricks job specs upload `src/` and `training/src/` and put them on `PYTHONPATH`; their `environment.dependencies` add what the AI v6 base environment does not have (`tabicl` for training, `openml` for AMLB, `tabarena` for the arenas).

## v2 recipe options

Options for the stage-1 recipe arms of [`docs/v2-experiment-plan.md`](docs/v2-experiment-plan.md) (phase 1B, and phase 2 with the adopted ones). All are off by default, so the default config and the existing priors train as before (`tests/test_default_behavior.py` checks this against values recorded before the options existed). Each is a `TrainConfig` field, so the MLflow run params name the arm. A resumed run keeps its own values.

| Arm | `train.py` flag | Variable (`train.yaml` override) | What it does |
|---|---|---|---|
| all | `--reference-val-tables 512` | `TABSIL_REFERENCE_VAL_TABLES=512` | a second validation set (`ref_*` metrics) drawn from `graph_scm` with the v1 prior settings and fixed seeds, at the run's table sizes, independent of the run's seed and prior (`training/reference.py`, run tag `reference_val_digest`) |
| YCTX | `--target-train-rows` | `TABSIL_TARGET_TRAIN_ROWS=1` | the regression target standardized again with the train rows after the prior's clip, as inference does; tables with a train-row sd of at most 0.1 drawn again; the borders follow (`priors/transforms/target.py`) |
| OPTK | `--optimizer muon_adamw --lr 3e-4 --muon-momentum 0.95` | `TABSIL_OPTIMIZER=muon_adamw`, `TABSIL_LR=3e-4`, `TABSIL_MUON_MOMENTUM=0.95` | Kumo-Tabular's optimizer: Muon on the Linear weights, AdamW on the target encoders, the output, the attention logit scales and the other parameters (`training/muon.py`) |
| PS1 | `--prior-type graph_scm_base_realism` | `TABSIL_PRIOR_TYPE=graph_scm_base_realism` | warped columns, 2-200 categories, correlated blocks 0.3, structured missing values 0.2, random missing cells in 1% of the tables at 1-8%, too-easy tables drawn again (`priors/graph_scm_base_realism.py`) |

What the reference loss compares: one task's loss on the same tables across seeds, priors and optimizers, a second readout for arms that share a prior. Across priors it also measures each prior's distance to the reference prior. With YCTX the reference regression targets are scaled with the train rows as in that run's training and the loss is mapped back to the set's units, so `ref_loss_regression` compares YCTX with C in those units (not in the train-row units of inference); its `val_*` regression loss is in train-row units on its own prior and does not compare with C's. The classification metrics are not affected by YCTX.

`ref_loss_regression` gives each run the targets in the form it trained on. A run without YCTX sees targets standardized over all rows, which tell it a little about the eval rows. So every run also logs `ref_loss_regression_train_rows`: the same tables with each target standardized with its train rows, as inference gives it, and the loss mapped back to the set's units. That is the number to compare YCTX with C on. For a YCTX run the two metrics are equal.

Each run also logs what the arms need for monitoring and bookkeeping:
- `head_scale_mean/<group>` and `head_scale_max/<group>`: the attention temperatures a_h per stage and for the first row ICL layer (`row_icl_first`). OPTK moves them from Muon to AdamW
- the `prior_options` param: the values the prior module fixes. `graph_scm_base_realism` replaces `missing_dataset_prob`, so that config param does not describe a PS1 run
- the `tabicl_version` and `reference_val_digest` tags. Compare `ref_*` only between runs with the same digest

### Phase 1B launch

From scratch, stage 1, 30K steps (warmup 1% = 300), batch 256, v1's size (no `TABSIL_EMSIZE`), one `GPU_8xH100` node per run. A run takes about 15 hours and 121 H100-h (1.81 s per step, phase 0.3 of [`docs/v2-experiment-results.md`](docs/v2-experiment-results.md)). The speed options are those of the timing run `tabsil-p0-timing-m`.

```bash
COMMON="--override mlflow_experiment_directory=$MLFLOW_EXPERIMENT_DIR \
  --override timeout_minutes=1320 \
  --override env_variables.TABSIL_CONFIG_NAME=stage1 \
  --override env_variables.TABSIL_STEPS=30000 \
  --override env_variables.TABSIL_BATCH_SIZE=256 \
  --override env_variables.TABSIL_REFERENCE_VAL_TABLES=512 \
  --override env_variables.TABSIL_NUM_WORKERS=16 \
  --override env_variables.TABSIL_COMPILE=1 --override env_variables.TABSIL_TF32=1 \
  --override env_variables.TABSIL_MUON_DISTRIBUTED=all"

# C: v1 recipe, graph_scm (seeds 0, 1, 2)
databricks air run -p e2-demo-field-eng -f training/scripts/train.yaml $COMMON \
  --override mlflow_run_name=tabsil-v2-1b-c-s0 --override env_variables.TABSIL_SEED=0

# YCTX (seeds 0, 1)
databricks air run -p e2-demo-field-eng -f training/scripts/train.yaml $COMMON \
  --override mlflow_run_name=tabsil-v2-1b-yctx-s0 --override env_variables.TABSIL_SEED=0 \
  --override env_variables.TABSIL_TARGET_TRAIN_ROWS=1

# OPTK (seeds 0, 1)
databricks air run -p e2-demo-field-eng -f training/scripts/train.yaml $COMMON \
  --override mlflow_run_name=tabsil-v2-1b-optk-s0 --override env_variables.TABSIL_SEED=0 \
  --override env_variables.TABSIL_OPTIMIZER=muon_adamw \
  --override env_variables.TABSIL_LR=3e-4 \
  --override env_variables.TABSIL_MUON_MOMENTUM=0.95

# PS1 (seeds 0, 1)
databricks air run -p e2-demo-field-eng -f training/scripts/train.yaml $COMMON \
  --override mlflow_run_name=tabsil-v2-1b-ps1-s0 --override env_variables.TABSIL_SEED=0 \
  --override env_variables.TABSIL_PRIOR_TYPE=graph_scm_base_realism \
  --override env_variables.TABSIL_PREFETCH_FACTOR=16
```

PS1 needs `TABSIL_PREFETCH_FACTOR=16`. Its prior (`corrected_cat_sizes`, `corrected_num_converters`) has a few micro batches that take 20 to 45 s to make. The loader hands micro batches over in request order with a head start of workers per task × prefetch, about 14 s at the default 4, so such a micro batch stops its GPU and the other seven wait in the all-reduce. `data_wait_frac` is measured on rank 0 and shows about an eighth of it. In 300-step runs on 8 H100, PS1 took 2.02 and 2.49 s per step (steps 101-200 and 201-300) at prefetch 4 and 1.95 and 1.98 at 16, against 1.80 for C. The tables and their order are the same at any prefetch. The remaining 9% is in the forward and backward time at equal table sizes and was not traced further.

The workspace's `GPU_8xH100` quota is 8 nodes, and a request above it fails at once instead of waiting. A resumed run (`TABSIL_RESUME_RUN_ID`) keeps its seed, steps, warmup, batch, prior and recipe options, and the table sizes and prior settings it was trained with (`DATA_OPTIONS` in `training/scripts/train.py`), whatever preset the resume command names. A table size or prior setting given on resume that differs from the run's stops the script.
