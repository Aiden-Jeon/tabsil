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
