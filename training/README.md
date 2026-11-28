# tabsil-train

Training, the synthetic-data prior, and the benchmark evaluation of [TabSil](../README.md). This is a uv workspace member of the TabSil repository: it depends on the `tabsil` inference package (`src/tabsil/`) and is imported as `tabsil_train`.

| Path | Contents |
|---|---|
| `src/tabsil_train/priors/` | the TabICLv2 graph SCM prior (`tabicl` package), table size rules, transforms; `graph_scm_hicard_missing` adds high-cardinality categorical columns and structured missing values, and `graph_scm_realism` adds Kumo-Tabular's other realism parts on top; `graph_scm_base_realism` has only Kumo-Tabular's every-stage parts; `graph_scm_task_probe` makes every table task-shaped (rows ordered by one column before the cut; data rows with hidden anomalies against reference rows) for the short fine-tuning of `docs/v2-prior-task-results.md`, not for a model to release (`--prior-type`) |
| `src/tabsil_train/training/` | training config (`TrainConfig`), the training loop, Muon (and Muon + AdamW), the reference validation set, losses, stability, diagnostics |
| `src/tabsil_train/benchmarks/` | the benchmarks, one folder per task: `tabular/` (AMLB / CTR23, TabArena, BeyondArena, uncertainty metrics), `forecast/` (fev-bench), `unsupervised/` (ADBench, imputation), `causal/` (IHDP, ACIC 2016) |
| `src/tabsil_train/recipe.py` | the TabICLv2 three-stage curriculum (`curriculum(stage)`) |
| `src/tabsil_train/hub.py` | `load_model` for the scripts (a Hub repo `hf://...`, a `save_pretrained` folder, a checkpoint file or an MLflow run checkpoint `runs:/...`), the model structure in run params, and Hub weights as the checkpoint a training run starts from (`--init-hub`) |
| `src/tabsil_train/benchmarks/tabular/subsample.py` | `SubsampledEstimator`: at most N training rows per fit, for context-length experiments (`eval.py --subsample-train-rows`) |
| `scripts/` | `train.py`, `eval.py` and their Databricks AI Runtime job specs (`*.yaml`); `paired_report.py`: the paired per-dataset error ratio of two eval runs (v2 plan §3); `seed_arm_report.py`: the plan's rule for arms trained from scratch with several seeds (scores per run, G of the seed means); `dev_suite_overlap.py`: the AMLB / TabArena overlap behind `--suite dev` |
| `scripts/v2_prior_diag/` | diagnostics of the v1 weights for the v2 prior (forecast extrapolation, calibration, contaminated context); `attention_temperatures.py`: the learned attention temperatures of saved weights |
| `scripts/v2_eff/` | step-time measurements for the v2 plan's GPU cost: `step_probe.py` (runs `train.py` and times the optimizer step, each GPU's data wait and the garbage collector), `timing.yaml` (its job spec), `step_breakdown.py` (the step time by part from a run's records) |
| `docs/` | design notes (`v2-experiment-plan.md`: the v2 experiment plan, with the check of the two notes it replaces; `v2-experiment-results.md`: the results of its runs, by phase; `v2-task-priors.md`: task priors and the diagnostics of the v1 weights; `v2-scaling.md`: model size and the stage-1 records) |
| `hub/` | `push_to_hub.py`, `sync_hub_card.py`: upload weights from a training run and the model card |
| `tests/` | training, prior, evaluation and upload tests |

The Databricks job specs upload `src/` and `training/src/` and put them on `PYTHONPATH`; their `environment.dependencies` add what the AI v6 base environment does not have (`tabicl` for training, `openml` for AMLB, `tabarena` for the arenas).

## Environments

From the repository root:

```bash
uv sync --all-packages                                # tabsil + tabsil-train + dev tools
uv run pytest tests                                   # inference tests
uv run --package tabsil-train pytest training/tests   # training, prior and evaluation tests
```

A suite with an extra in the [Evaluation](#evaluation) table needs it: `uv sync --package tabsil-train --extra causal`, `uv sync --all-packages --extra impute`. The `arena` and `fev` extras need pandas 2.3 and `huggingface-hub<2`, which the dev tools do not allow, so they go without them: `uv sync --package tabsil-train --no-dev --extra arena --group arena-dev` (or `--extra fev`). Each `uv sync` removes what that command does not install.

## Training

Synthetic tables come from the TabICLv2 graph SCM prior (`tabicl` package) with at most 100 features. Training follows the TabICLv2 three-stage curriculum, with Muon on every parameter, batch 256 on 32 GPUs, and 1% warmup then cosine decay in each stage:

| Stage | Rows per table (train share) | Steps | Peak lr | Starts from |
|---|---|---|---|---|
| 1 | 1,024 (30–90%) | 250K | 8e-4 | scratch |
| 2 | 400–10,240 (80%) | 20K | 1e-4 | stage 1 weights |
| 3 | 400–60,000 (80%) | 5K | 2e-5 | stage 2 weights |

The published v1 weights were trained with shorter stages, 150K / 10K / 2K steps (`--steps`; warmup and cosine follow the stage length): a separate 250K stage-1 run, measured up to 225K, did no better than 150K on TabArena-Lite (measured with `training/scripts/eval.py`, without bagging). Their intermediate stages are branches of the Hub repo (`stage1-150k`, `stage2-10k`).

How v1 was trained (the stage table it used with grad clip, what differs from the TabICLv2 scripts, compute, validation loss, weight hashes and the scores of each stage) is in the model card's [Training](https://huggingface.co/aiden-jeon/tabsil#training) and [Intermediate checkpoints](https://huggingface.co/aiden-jeon/tabsil#intermediate-checkpoints) sections (source: [`hub/tabsil/README.md`](hub/tabsil/README.md)).

```bash
# local check (CPU, about a minute)
PYTHONHASHSEED=0 uv run --package tabsil-train python training/scripts/train.py --config-name smoke

# full training: Databricks AI Runtime job specs in training/scripts/train.yaml
uv run --package tabsil-train python training/scripts/train.py --config-name stage1
uv run --package tabsil-train python training/scripts/train.py --config-name stage2 --init-run-id <stage 1 run>
uv run --package tabsil-train python training/scripts/train.py --config-name stage3 --init-run-id <stage 2 run>

# the v1 weights: the same stages with --steps 150000 / 10000 / 2000
```

Set `PYTHONHASHSEED=0` for reproducible data: the `tabicl` prior iterates over a set of strings, so its tables depend on the hash seed.

## Evaluation

`scripts/eval.py` evaluates a checkpoint, MLflow run, or Hub model; its module docstring describes every suite, its options and what it needs. The arena suites fit each split once without bagging, a quick check for comparing checkpoints; the official TabArena-Lite result is reproduced with TabArena's `TabSil` model ([autogluon/tabarena#753](https://github.com/autogluon/tabarena/pull/753)). The suites and their Databricks AI Runtime job specs (in `scripts/`):

| Suite (`--suite`) | Benchmark | Job spec |
|---|---|---|
| `all`, `classification`, `regression` | AMLB (classification / CTR23 regression); `--uncertainty` adds CRPS, coverage, PIT and calibration metrics | `eval.yaml` |
| `dev` | the development suite of the v2 plan: the AMLB tasks without the 9 whose data is also in TabArena-Lite (23 classification, 25 regression) | `eval.yaml` |
| `tabarena-lite`, `tabarena-3split`, `beyondarena`, `beyondarena-lite` | TabArena-Lite (`tabarena-3split`: the first 3 splits of its 51 datasets) and BeyondArena (the `arena` extra) | `eval_arena.yaml` |
| `fev` | forecasting on 10 fev-bench tasks (the `fev` extra) | `eval_fev.yaml` |
| `adbench` | unsupervised outlier detection on 12 ADBench tables | `eval.yaml` |
| `causal` | treatment effects on IHDP and ACIC 2016 (the `causal` extra, for the EconML baselines) | `eval_causal.yaml` |
| `impute` | missing value imputation on 10 OpenML tables (the `impute` extra) | `eval_impute.yaml` |

## v2 recipe options

Options for the stage-1 recipe arms of [`docs/v2-experiment-plan.md`](docs/v2-experiment-plan.md) (phase 1B, and phase 2 with the adopted ones). All are off by default, so the default config and the existing priors train as before (`tests/test_default_behavior.py` checks this against values recorded before the options existed). Each is a `TrainConfig` field, so the MLflow run params name the arm. A resumed run keeps its own values.

| Arm | `train.py` flag | Variable (`train.yaml` override) | What it does |
|---|---|---|---|
| all | `--reference-val-tables 512` | `TABSIL_REFERENCE_VAL_TABLES=512` | a second validation set (`ref_*` metrics) drawn from `graph_scm` with the v1 prior settings and fixed seeds, at the run's table sizes, independent of the run's seed and prior (`training/reference.py`, run tag `reference_val_digest`) |
| YCTX | `--target-train-rows` | `TABSIL_TARGET_TRAIN_ROWS=1` | the regression target standardized again with the train rows after the prior's clip, as inference does; tables with a train-row sd of at most 0.1 drawn again; the borders follow (`priors/transforms/target.py`) |
| OPTK | `--optimizer muon_adamw --lr 3e-4 --muon-momentum 0.95` | `TABSIL_OPTIMIZER=muon_adamw`, `TABSIL_LR=3e-4`, `TABSIL_MUON_MOMENTUM=0.95` | Kumo-Tabular's optimizer: Muon on the Linear weights, AdamW on the target encoders, the output, the attention logit scales and the other parameters (`training/muon.py`) |
| PS1 | `--prior-type graph_scm_base_realism` | `TABSIL_PRIOR_TYPE=graph_scm_base_realism` | warped columns, 2-200 categories, correlated blocks 0.3, structured missing values 0.2, random missing cells in 1% of the tables at 1-8%, too-easy tables drawn again (`priors/graph_scm/variants.py`) |

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

The workspace's `GPU_8xH100` quota is 8 nodes, and a request above it fails at once instead of waiting. A resumed run (`TABSIL_RESUME_RUN_ID`) keeps its seed, steps, warmup, batch, prior and recipe options, the table sizes and prior settings it was trained with (`DATA_OPTIONS` in `training/scripts/train.py`), its activation checkpointing threshold and, unless `TABSIL_MICRO_BATCH` is given, its micro batch. Resume it with its own `TABSIL_CONFIG_NAME`: another preset stops the script, and so does a table size or prior setting that differs from the run's.
