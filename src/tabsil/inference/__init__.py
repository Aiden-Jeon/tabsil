"""Inference: use a trained model in sklearn style (preprocessing members, ensembles). This is the
engine under the estimators of `tabsil.tasks`.

- `preprocessing`: preprocessing of one ensemble member (outlier clipping, one-hot, column
  subsets)
- `ensemble`: preprocessing-member ensembles `EnsembleClassifier` and `EnsembleRegressor`. They
  work with any model class that has `forward(x, y_train, task)`

The classes with TabSil's default inference settings are `TabSilClassifier` and `TabSilRegressor`
(`tabsil.tasks.classification`, `tabsil.tasks.regression`); the settings are in `tabsil.recipe`.
"""
