# v2 prior diagnostics

Small synthetic checks of the v1 weights that decide what the v2 training prior adds for forecasting and outlier detection. The design they feed is [`training/docs/v2-task-priors.md`](../../docs/v2-task-priors.md).

| Script | Question |
|---|---|
| `d1_forecast.py` | Does `TabSilForecaster` continue a trend past the largest context value? Synthetic daily series (linear, exponential, piecewise trend, 4 strengths) against seasonal naive and OLS on index + sin/cos, with the weights' inference recipe (`default`) and with 2 plain members that nothing clips (`plain`) |
| `d2_calibration.py` | Is the predictive distribution calibrated when y is independent of x (the v1 prior drops unpredictable datasets)? NLL against the true density, a KDE of the train targets and a predictable control |
| `d3_contamination.py` | How does `TabSilUnsupervised`'s outlier score change with 0-20% outliers in the context, per outlier kind (global, local, cluster, curve), against IsolationForest? |

They need `tabsil.tasks.forecast` and `tabsil.tasks.unsupervised` (main, #17 and #19) and the 0.1.1 inference code. On dev-v2 they run only after main is merged in. The weights are pinned by `DIAG_MODEL` (default `aiden-jeon/tabsil@v1.1`), and the model is loaded once per diagnostic. The reported numbers used main f5b26a7. Run them on AI Runtime (1x H100, about 4 minutes), not locally:

```bash
databricks air run -f training/scripts/v2_prior_diag/diag.yaml \
  --override mlflow_experiment_directory=$MLFLOW_EXPERIMENT_DIR
```

`run_all.py` runs the three in turn, each within `DIAG_TIMEOUT` seconds. It logs each one's CSV files and outputs under `diag/` of the MLflow run as soon as that diagnostic ends. A failed diagnostic does not stop the others, but the job ends with an error.

`DIAG_MODEL` also takes a file that a training run logged to MLflow (`runs:/<run id>/model/final.pt`), a Hub repo as `hf://owner/name@revision` and a `.pt` file.

## Task diagnostics and inference arms

The checks of tier 1 of [`training/docs/v2-prior-task-plan.md`](../../docs/v2-prior-task-plan.md), which trains nothing. Their derivations are in [`training/docs/v2-task-formulations.md`](../../docs/v2-task-formulations.md) and their results in [`training/docs/v2-prior-task-results.md`](../../docs/v2-prior-task-results.md). None of them runs by default: name them in `DIAG_ARGS`.

| Script | Question | Settings |
|---|---|---|
| `arms.py` | Not a diagnostic: the forecast recipes and outlier scores the others compare, as wrappers around the released inference classes. Forecast arms: `default`, `no-target-power`, `standard`, `plain8`, `plain`, each also with `+trend` (a straight line fitted to the context outside the model). Outlier scores: `density`, `ctx16` to `ctx256`, `ctxfull` (the density under small random contexts of that many rows, read as mean, median and mixture), `ref-uniform`, `ref-marginals` (the probability of a reference-row class), `refine` (the density after the highest-scoring context rows are dropped) | — |
| `d1_forecast.py` | D1 with any forecast arm | `D1_ARMS` |
| `d3_contamination.py` | D3 with any outlier score | `D3_ARMS`, `D3_KINDS` |
| `d2_rare_class.py` | What probability does a class with 1 to 3 context rows get when the labels do not depend on the features: its share of the context rows, or the posterior's? | — |
| `d4_missing.py` | How does the imputation error change when cells are not missing completely at random? Five mechanisms at one rate (completely at random, by another column, another column's mask by the cell to impute in the rows that observe it and in every row, self-masking), on Gaussian tables where the value a fit on the complete cases converges to is known, and on the imputation suite's tables against MissForest and the mean | `D4_PARTS`, `D4_TABLES`, `D4_BASELINES` |
| `d5_overlap.py` | How does the error of the S-learner's and the T-learner's effect estimates grow as the propensity's clip shrinks from 0.5 to 0.02, on tables with a known effect? | — |
| `fev_arms.py` | The forecasting suite (fev-bench) with one forecast arm | `FEV_ARM`, `FEV_TASKS`, `FEV_WINDOWS` |
| `adbench_arms.py` | The outlier suite's tables and splits (ADBench) with outlier scores | `ADBENCH_ARMS`, `ADBENCH_DATASETS` |

`d2_rare_class.py`, `d5_overlap.py` and the arms of D1 and D3 run with `diag.yaml`. `d4_missing.py`, `fev_arms.py` and `adbench_arms.py` read benchmark data and run with `arms.yaml`, which has their dependencies and a longer time limit:

```bash
databricks air run -f training/scripts/v2_prior_diag/diag.yaml \
  --override mlflow_experiment_directory=$MLFLOW_EXPERIMENT_DIR \
  --override "env_variables.DIAG_ARGS=d1_forecast" \
  --override "env_variables.D1_ARMS=default,plain8,plain8+trend"
databricks air run -f training/scripts/v2_prior_diag/arms.yaml \
  --override mlflow_experiment_directory=$MLFLOW_EXPERIMENT_DIR \
  --override "env_variables.DIAG_ARGS=fev_arms" --override "env_variables.FEV_ARM=plain8"
```

The workspace has a quota of 8 `GPU_1xH100` nodes, and a job submitted beyond it fails at once instead of waiting.

## Prior speed

Two scripts for the time the training prior takes to make tables. They do not use the weights. The numbers and the reading are in [`training/docs/v2-experiment-results.md`](../../docs/v2-experiment-results.md) (Phase 1A, "Why R10 was slow").

| Script | Question |
|---|---|
| `prior_timing.py` | How long does one training table of a prior take, and where does the time go (SCM draws, the dataset filter, the transforms)? One process per CPU pair generates tables as the training loop's prior workers do, and every table is one CSV row. Run it on AI Runtime with `prior_timing.yaml` |
| `queue_sim.py` | What step time do those generation times give in the training loop? It replays the measured times through the loop's DataLoader queue (workers per task, prefetch per worker, tables handed over in request order or in any order) for 32 GPUs. Nothing is generated and no model runs, so it runs anywhere |

```bash
databricks air run -f training/scripts/v2_prior_diag/prior_timing.yaml \
  --override mlflow_experiment_directory=$MLFLOW_EXPERIMENT_DIR \
  --override "env_variables.TIMING_ARGS=--priors graph_scm_realism --minutes 45"
# with diag/prior_timing.csv.gz of that run:
python training/scripts/v2_prior_diag/queue_sim.py prior_timing.csv.gz --prior graph_scm_realism \
  --compute 0.215 --overhead 1.58 --workers 8 16 --prefetch 4 16 --any-order
```
