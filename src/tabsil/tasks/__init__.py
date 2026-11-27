"""The estimators users call, one module per task.

Classification and regression are the two tasks the model itself outputs:

- `classification`: `TabSilClassifier`
- `regression`: `TabSilRegressor`

Both are the preprocessing-member ensembles of `tabsil.inference` with TabSil's inference defaults
(`tabsil.recipe`). The other three are built by calling those two; they have no weights of their
own:

- `forecast`: `TabSilForecaster`, time series forecasting with the regressor
- `unsupervised`: `TabSilUnsupervised`, per-column conditional distributions from the regressor
  (numeric columns) and the classifier (categorical columns), and from them row log densities,
  outlier scores and imputation
- `causal`: `TabSilCATE`, treatment effects with the regressor (outcomes) and the classifier
  (propensity scores) as the base learners of S/T/X/DR meta-learners

Nothing is imported here: `tabsil` loads each estimator on first use.

    from tabsil import TabSilClassifier, TabSilForecaster
"""
