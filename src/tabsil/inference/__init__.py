"""Inference: use a trained model in sklearn style (preprocessing members, ensembles).

- `preprocessing`: preprocessing of one ensemble member (outlier clipping, one-hot, column
  subsets)
- `ensemble`: preprocessing-member ensembles `EnsembleClassifier` and `EnsembleRegressor`. They
  work with any model class that has `forward(x, y_train, task)` (`tabsil.recipe.TabSilClassifier`,
  `TabSilRegressor`)
- `subsample`: `SubsampledEstimator`, which reduces the rows passed to fit to N (class-stratified
  for classification; for context-length experiments)

The classes with TabSil's default inference settings are in `tabsil.recipe`.
"""
