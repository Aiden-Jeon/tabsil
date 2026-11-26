# tabsil-train

Training, the synthetic-data prior, and the benchmark evaluation of [TabSil](../README.md). This is a uv workspace member of the TabSil repository: it depends on the `tabsil` inference package (`src/tabsil/`) and is imported as `tabsil_train`.

| Path | Contents |
|---|---|
| `src/tabsil_train/priors/` | the TabICLv2 graph SCM prior (`tabicl` package), table size rules, transforms; `graph_scm_hicard_missing` adds high-cardinality categorical columns and structured missing values, and `graph_scm_realism` adds Kumo-Tabular's other realism parts on top; `graph_scm_base_realism` has only Kumo-Tabular's every-stage parts (`--prior-type`) |
| `src/tabsil_train/training/` | training config (`TrainConfig`), the training loop, Muon (and Muon + AdamW), the reference validation set, losses, stability, diagnostics |
| `src/tabsil_train/benchmarks/` | AMLB / CTR23, TabArena, BeyondArena, uncertainty metrics |
| `src/tabsil_train/recipe.py` | the TabICLv2 three-stage curriculum (`curriculum(stage)`) |
| `src/tabsil_train/hub.py` | MLflow run checkpoints (`runs:/...`) and the model structure in run params |
| `scripts/` | `train.py`, `eval.py` and their Databricks AI Runtime job specs (`*.yaml`) |
| `scripts/v2_prior_diag/` | diagnostics of the v1 weights for the v2 prior (forecast extrapolation, calibration, contaminated context) |
| `docs/` | design notes (`v2-experiment-plan.md`: the v2 experiment plan, with the check of the two notes it replaces; `v2-task-priors.md`: task priors and the diagnostics of the v1 weights; `v2-scaling.md`: model size and the stage-1 records) |
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
