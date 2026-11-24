"""Inference: use a trained model in sklearn style (preprocessing members, ensembles).

- `preprocessing`: preprocessing of one ensemble member (outlier clipping, one-hot, column
  subsets)
- `ensemble`: preprocessing-member ensembles `EnsembleClassifier` and `EnsembleRegressor`. They
  work with any model class that has `forward(x, y_train, task)` (`tabsil.recipe.TabSilClassifier`,
  `TabSilRegressor`)
- `ensemble_selection`: `GreedyEnsembleClassifier`, which greedily selects from a member pool by
  out-of-fold loss within the train rows (aggregation that spends more inference compute,
  classification only)
- `diag_scale`: `DiagScale`, test-time adaptation that learns only a diagonal attention scale per
  dataset (wraps an ensemble estimator, classification only)
- `subsample`: `SubsampledEstimator`, which reduces the rows passed to fit to N (class-stratified
  for classification; for context-length experiments)

The classes with TabSil's default inference settings are in `tabsil.recipe`.
"""
