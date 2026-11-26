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
| `scripts/v2_prior_diag/` | diagnostics of the v1 weights for the v2 prior (forecast extrapolation, calibration, contaminated context) |
| `docs/` | design notes (`v2-task-priors.md`: the v2 prior, task priors trained from scratch with the v1 architecture) |
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
