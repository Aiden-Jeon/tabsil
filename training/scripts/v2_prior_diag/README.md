# v2 prior diagnostics

Small synthetic checks of the v1 weights that decide what the v2 training prior adds for forecasting and outlier detection. The design they feed is [`training/docs/v2-task-priors.md`](../../docs/v2-task-priors.md).

| Script | Question |
|---|---|
| `d1_forecast.py` | Does `TabSilForecaster` continue a trend past the largest context value? Synthetic daily series (linear, exponential, piecewise trend, 4 strengths) against seasonal naive and OLS on index + sin/cos, with the weights' inference recipe (`default`) and with 2 plain members that nothing clips (`plain`) |
| `d2_calibration.py` | Is the predictive distribution calibrated when y is independent of x (the v1 prior drops unpredictable datasets)? NLL against the true density, a KDE of the train targets and a predictable control |
| `d3_contamination.py` | How does `TabSilUnsupervised`'s outlier score change with 0-20% outliers in the context, per outlier kind (global, local, cluster, curve), against IsolationForest? |

They need `tabsil.forecast` and `tabsil.unsupervised` (main, #17 and #19) and the 0.1.1 inference code. On dev-v2 they run only after main is merged in. The weights are pinned by `DIAG_MODEL` (default `aiden-jeon/tabsil@v1.1`), and the model is loaded once per diagnostic. The reported numbers used main f5b26a7. Run them on AI Runtime (1x H100, about 4 minutes), not locally:

```bash
databricks air run -f training/scripts/v2_prior_diag/diag.yaml \
  --override mlflow_experiment_directory=$MLFLOW_EXPERIMENT_DIR
```

`run_all.py` runs the three in turn, each within `DIAG_TIMEOUT` seconds. It logs each one's CSV files and outputs under `diag/` of the MLflow run as soon as that diagnostic ends. A failed diagnostic does not stop the others, but the job ends with an error.

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
