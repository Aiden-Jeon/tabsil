# TabSil

TabSil is a tabular foundation model. One model handles both classification and regression: give it a training table and it predicts test rows in a single forward pass (in-context learning), with no gradient steps on your data. It is trained only on synthetic tables drawn from a structural causal model prior.

*Sil* (실) is Korean for *thread*: the model reads a table by weaving through its columns and rows.

This package is inference only: the model, the scikit-learn style estimators and loading the weights from the Hugging Face Hub ([aiden-jeon/tabsil](https://huggingface.co/aiden-jeon/tabsil)). Training, the synthetic-data prior and the benchmarks are in the [GitHub repository](https://github.com/Aiden-Jeon/tabsil).

## Install

```bash
uv add tabsil                     # or: pip install tabsil
```

Python ≥ 3.11, PyTorch ≥ 2.11.

## Use

Without a model, the estimators download the v1 weights from the Hub at the tag `v1.1` (`tabsil.recipe.HUB_REVISION`), so a release keeps the same weights when the repo's `main` changes:

```python
from tabsil import TabSilClassifier, TabSilRegressor

clf = TabSilClassifier(device="cuda").fit(X_train, y_train)
proba = clf.predict_proba(X_test)

reg = TabSilRegressor(device="cuda").fit(X_train, y_train)
y_pred = reg.predict(X_test)
```

Since tabsil 0.1.2 the same weights also run zero-shot forecasting (`TabSilForecaster`), outlier detection and imputation (`TabSilUnsupervised`) and treatment effects (`TabSilCATE`). Other weights, the inference settings and these estimators are described in the [repository README](https://github.com/Aiden-Jeon/tabsil#use).

## Results

On [TabArena-Lite](https://huggingface.co/spaces/TabArena/leaderboard) with TabArena's official pipeline, TabSil places 18th of 121 methods (Elo 1755, tabsil 0.1.0). The charts, the other benchmarks and how they were run are in the [repository README](https://github.com/Aiden-Jeon/tabsil#results) and on the [model card](https://huggingface.co/aiden-jeon/tabsil).

## License

Apache License 2.0. See `LICENSE` and `NOTICE`.
