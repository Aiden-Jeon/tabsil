# tabsil-train

Training, the synthetic-data prior, and the benchmark evaluation of [TabSil](../README.md). This is a uv workspace member of the TabSil repository: it depends on the `tabsil` inference package (`src/tabsil/`) and is imported as `tabsil_train`.

| Path | Contents |
|---|---|
| `src/tabsil_train/priors/` | the TabICLv2 graph SCM prior (`tabicl` package), table size rules, transforms |
| `src/tabsil_train/training/` | training config (`TrainConfig`), the training loop, Muon, losses, stability, diagnostics |
| `src/tabsil_train/benchmarks/` | AMLB / CTR23, TabArena, BeyondArena, uncertainty metrics |
| `src/tabsil_train/recipe.py` | the TabICLv2 three-stage curriculum (`curriculum(stage)`) |
| `src/tabsil_train/hub.py` | MLflow run checkpoints (`runs:/...`) and the model structure in run params |
| `scripts/` | `train.py`, `eval.py` and their Databricks AI Runtime job specs (`*.yaml`) |
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
| `tabarena-lite`, `beyondarena`, `beyondarena-lite` | TabArena-Lite and BeyondArena (the `arena` extra) | `eval_arena.yaml` |
| `fev` | forecasting on 10 fev-bench tasks (the `fev` extra) | `eval_fev.yaml` |
| `adbench` | unsupervised outlier detection on 12 ADBench tables | `eval.yaml` |
| `causal` | treatment effects on IHDP and ACIC 2016 (the `causal` extra, for the EconML baselines) | `eval_causal.yaml` |
| `impute` | missing value imputation on 10 OpenML tables (the `impute` extra) | `eval_impute.yaml` |
