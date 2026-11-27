<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/Aiden-Jeon/tabsil/main/assets/logo/tabsil-logo-dark.svg">
    <img alt="TabSil" src="https://raw.githubusercontent.com/Aiden-Jeon/tabsil/main/assets/logo/tabsil-logo.svg" width="320">
  </picture>
</p>

# TabSil

TabSil is a tabular foundation model. One model handles both classification and regression: give it a training table and it predicts test rows in a single forward pass (in-context learning), with no gradient steps on your data. It is trained only on synthetic tables drawn from a structural causal model prior.

*Sil* (실) is Korean for *thread*: the model reads a table by weaving through its columns and rows.

This repository has everything needed to reproduce TabSil. It holds two packages:

- `tabsil` (`src/tabsil/`): inference only. The model, the scikit-learn style estimators (classification and regression; forecasting, outlier detection, imputation and treatment effects since tabsil 0.1.2), and loading weights from the Hugging Face Hub.
- `tabsil-train` (`training/`, imported as `tabsil_train`): the synthetic-data prior, the training curriculum, the benchmark harness, and the Hub upload scripts. It is a uv workspace member that depends on `tabsil`.

> Status: the TabSil v1 weights are on the Hugging Face Hub: [aiden-jeon/tabsil](https://huggingface.co/aiden-jeon/tabsil) (model card with the benchmark results).

## Results

[TabArena-Lite](https://huggingface.co/spaces/TabArena/leaderboard) with TabArena's official pipeline: every method, TabSil included, is 8-fold bagged, and the leaderboard is computed as on the public one. TabSil places 18th of 121 methods. The chart and the table below it are a hand-picked subset: TabSil, 4 of the 17 methods above it, TabPFN-3 and TabICLv2 (which it takes ideas from) below it, and RandomForest, the Elo anchor; the [full table](https://huggingface.co/datasets/aiden-jeon/tabsil-tabarena-lite/blob/1c4beda3ed9683fa5bd15f4cc9569a8180b04ec4/leaderboard_website.md) has all 121.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/Aiden-Jeon/tabsil/main/assets/charts/tabarena-lite-dark.svg">
    <img alt="Dot plot of TabArena-Lite Elo with 95% confidence intervals for TabSil and the compared methods; the table below it has the numbers." src="https://raw.githubusercontent.com/Aiden-Jeon/tabsil/main/assets/charts/tabarena-lite-light.svg" width="840">
  </picture>
</p>

<details>
<summary>Table: Elo, improvability, fit and infer time</summary>

| Method | Place | Elo (+CI) | Improvability (%) | Fit (s/1K) | Infer (s/1K) |
| --- | --- | --- | --- | --- | --- |
| Kumo-Tabular (default) | 1 | 2090 (+159/−110) | 3.7 | 2.33 | 0.254 |
| TabPFN-3.5 (default) | 7 | 1882 (+139/−91) | 7.5 | 1.85 | 0.473 |
| TabPFN-3.5-Fast (default) | 16 | 1806 (+91/−59) | 9.1 | 0.73 | 0.153 |
| Mitra-v2 (default) | 17 | 1774 (+101/−68) | 8.5 | 59.03 | 0.886 |
| **TabSil** | **18** | **1755** (+88/−73) | 9.0 | 2.53 | 0.812 |
| TabPFN-3 (default) | 25 | 1626 (+73/−48) | 11.6 | 1.29 | 0.383 |
| TabICLv2 (default) | 29 | 1567 (+61/−57) | 13.1 | 0.73 | 0.137 |
| RandomForest (default) | 112 | 1000 (+74/−75) | 28.4 | 0.43 | 0.053 |

</details>

- TabArena-Lite has 51 datasets, one split each. TabSil ran as the `TabSil` model of TabArena ([autogluon/tabarena#753](https://github.com/autogluon/tabarena/pull/753)) at its default configuration: tabsil 0.1.0 from PyPI, the weights at the tag `v1`, 8×H100 with one process per GPU. No split is imputed. Elo puts RandomForest (default) at 1000, with a 95% CI. Lower is better for improvability, fit and infer.
- All methods use the same protocol, so the Fit and Infer columns are like-for-like; the other methods' times were measured by TabArena on its own hardware.
- The pool is the TabArena results available when the run was made (2026-11-25, TabArena at the submission commit `1360e0e8`): 120 methods plus TabSil, 121 methods. It is not a released leaderboard version, so the live leaderboard's numbers differ. These are the results submitted with the pull request; the maintainers' own run of the full TabArena may change them. The raw results (`results.pkl` of all 51 splits), the hardware and the logs are in [aiden-jeon/tabsil-tabarena-lite](https://huggingface.co/datasets/aiden-jeon/tabsil-tabarena-lite).
- tabsil 0.1.1 changed the inference defaults (see [Use](#use)); these results are tabsil 0.1.0's, and the official-protocol run of 0.1.1 follows its release.
  - In the repository's harness (each split fit once, no bagging; see [Evaluation](#evaluation)) the same weights score 1768 with 0.1.1's inference against 1726 with 0.1.0's.
  - Median fit time is 0.23 s per 1K training rows, against 0.014; 0.1.1's fit runs the training rows through the model.
  - Median predict time is 0.16 s per 1K test rows, against 0.81.
- Fit and infer time are defined in [Evaluation](#evaluation). More results (AMLB, per-problem-type and per-stage Elo) are on the [model card](https://huggingface.co/aiden-jeon/tabsil).

### Other tasks

The same weights also run four more tasks zero-shot (tabsil ≥ 0.1.2); TabSil is measured at the Hub tag `v1.1`. The protocols, comparison methods, caveats and the tables behind this figure are in [docs/task-results.md](https://github.com/Aiden-Jeon/tabsil/blob/main/docs/task-results.md).

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/Aiden-Jeon/tabsil/main/assets/charts/other-tasks-dark.svg">
    <img alt="Four small charts comparing TabSil with the baselines of each benchmark: forecasting (fev-bench), outlier detection (ADBench), treatment effects (IHDP, ACIC 2016) and imputation (OpenML, MCAR). The tables in docs/task-results.md have the numbers." src="https://raw.githubusercontent.com/Aiden-Jeon/tabsil/main/assets/charts/other-tasks-light.svg" width="840">
  </picture>
</p>

## Install

Inference only, from PyPI:

```bash
uv add tabsil                     # or: pip install tabsil
uv add "tabsil @ git+https://github.com/Aiden-Jeon/tabsil"   # the latest main instead of a release
```

Training and evaluation, from a clone:

```bash
git clone https://github.com/Aiden-Jeon/tabsil
cd tabsil
uv sync --all-packages            # tabsil + tabsil-train (training, priors, AMLB evaluation) + dev tools, Jupyter
```

The other environments (inference tests only, `tabsil-train` only, and the `arena`, `fev`, `causal` and `impute` extras of the evaluation suites) are listed in [`training/README.md`](https://github.com/Aiden-Jeon/tabsil/blob/main/training/README.md#environments).

Python ≥ 3.11, PyTorch ≥ 2.11 (the benchmark results above were measured with PyTorch 2.11).

## Use

Without a model, `TabSilClassifier()` and `TabSilRegressor()` download the v1 weights from the default Hugging Face Hub repo (`tabsil.recipe.HUB_REPO`, [aiden-jeon/tabsil](https://huggingface.co/aiden-jeon/tabsil)) at the tag `v1.1` (`tabsil.recipe.HUB_REVISION`): the v1 weights with tabsil 0.1.1's inference settings. A release keeps downloading the same weights when the repo's `main` changes; tabsil 0.1.0 downloads the tag `v1`, the same weights with its own settings:

```python
from tabsil import TabSilClassifier, TabSilRegressor

clf = TabSilClassifier(device="cuda").fit(X_train, y_train)
proba = clf.predict_proba(X_test)

reg = TabSilRegressor(device="cuda").fit(X_train, y_train)
y_pred = reg.predict(X_test)
```

Other weights are given as the first argument: a Hub repo id (`"owner/name"` for its `main`, or `"aiden-jeon/tabsil@stage2-10k"` for a branch, tag or commit), a folder written by `tabsil.hub.save_pretrained` (`model.safetensors` + `config.json`), or a model:

```python
from tabsil.models import load_checkpoint

model, borders = load_checkpoint("tabsil_run/final.pt")  # a training checkpoint
reg = TabSilRegressor(model, borders).fit(X_train, y_train)
```

Both are scikit-learn style estimators over an ensemble of 8 preprocessing members:

- each member sees at most 768 of the original columns
- members cycle through four numeric transforms: the values as they are, Yeo-Johnson, robust scaling, and rank-Gauss
- categorical columns with more than 20 categories also get a log count column
- classification averages the members' softmax at temperature 1.0

`fit` runs each member's training rows through the model once and keeps what the test rows read from them, so `predict` runs only the test rows (`tabsil.inference.ensemble`). The settings come with the weights (`config.json` `inference`, `tabsil.recipe.INFERENCE`); tabsil 0.1.0 used standard members only, no count columns and temperature 0.9.

### Forecasting, outlier detection, imputation and treatment effects

(tabsil ≥ 0.1.2) Zero-shot with the same weights ([results](https://github.com/Aiden-Jeon/tabsil/blob/main/docs/task-results.md)). Defaults, missing value handling and the differences from TabPFN-TS, the official TabPFN extension and EconML are in the module docstrings: `src/tabsil/forecast.py`, `unsupervised.py` and `causal.py`.

```python
from tabsil import TabSilCATE, TabSilForecaster, TabSilUnsupervised

# Forecasting, as TabPFN-TS: time features in (index, calendar, FFT-detected seasonality; no lags),
# quantiles out. Each series is fit on its last 4,096 rows (`max_context_length`).
# context: one row per (item_id, timestamp) with a target column; other columns are covariates
fc = TabSilForecaster(device="cuda").fit(context)
pred = fc.predict(24)  # item_id, timestamp, predictions, "0.1" ... "0.9"
pred = fc.predict(future=future, quantile_levels=[0.05, 0.5, 0.95])  # known covariates in `future`

# Outlier detection without labels: a log density per row from column conditionals chained along
# 10 random column orders (`n_permutations`; one fit and prediction per column and order), as the
# official TabPFN extension does.
model = TabSilUnsupervised(device="cuda").fit(X_train)  # unlabeled rows: the context
scores = model.outlier_score(X_test)  # = -model.log_density(X_test); higher = more anomalous

# Imputation from the same conditionals, one fit per column with a missing value: the predictive
# mean (`strategy="median"` for the median) for numeric cells, the most probable class for
# categorical ones. Without `categorical_features`, columns with at most 40 distinct values and
# more than 10 rows per value are treated as categorical.
model = TabSilUnsupervised(categorical_features=[3, 5], device="cuda").fit(X)  # X with NaN
X_filled = model.impute(X)  # observed cells unchanged
draws = model.impute(X, n_samples=5)  # multiple imputation: (5, n, d)

# Treatment effects of a binary treatment on a continuous outcome: a meta-learner with TabSil as
# the base learner (`learner="s" | "t" | "x" | "dr"`; `regressor=` and `classifier=` take any
# scikit-learn estimator instead of TabSil).
est = TabSilCATE(device="cuda").fit(X, treatment, y)  # the S-learner
tau = est.effect(X_test)  # CATE per row
ate = est.ate()  # mean CATE over the training rows
```

## Model

| Part | Choice | From |
|---|---|---|
| Row in-context learning | 24 layers × 16 heads, width 1024 | TabPFN-3.5 report |
| Cell encoder | Fourier features + within-column ECDF, q/k RMSNorm, target LayerNorm, pre-head MLP | TabPFN-3.5 report |
| Column / row stages | 6 interleaved pairs, cell width 256, 4 heads, 256 inducing points, 4 CLS tokens | Kumo-Tabular |
| Attention temperature | gain-free q/k RMSNorm × learned log-scale (`logscale`, `gated_logscale`), no soft-cap | Kumo-Tabular |
| Classification head | many-class retrieval decoder | TabPFN-3.5 |
| Regression head | 999 quantiles, pinball loss | Kumo-Tabular |
| Tasks | one multitask model for classification and regression | TabPFN-3.5 |

Code: `src/tabsil/models/column_row_transformer.py`; the structure and inference numbers are in `src/tabsil/recipe.py`, the training curriculum in `training/src/tabsil_train/recipe.py`.

## Training

TabSil is trained on synthetic tables from the TabICLv2 graph SCM prior, in the TabICLv2 three-stage curriculum. The stage table, the commands to train it and to reproduce the published v1 weights, and the Hub branches of the intermediate stages are in [`training/README.md`](https://github.com/Aiden-Jeon/tabsil/blob/main/training/README.md#training).

## Evaluation

`training/scripts/eval.py` evaluates a checkpoint, MLflow run, or Hub model on AMLB (classification / CTR23 regression), TabArena(-Lite) and BeyondArena, plus uncertainty metrics (CRPS, coverage, PIT, calibration) and the task suites `--suite fev | adbench | causal | impute`. The suites, their job specs and environments are in [`training/README.md`](https://github.com/Aiden-Jeon/tabsil/blob/main/training/README.md#evaluation).

Train and inference time are reported as on the TabArena leaderboard: seconds per 1K train rows (fit) and per 1K test rows (predict), averaged over each dataset's splits, then the median over datasets. In the arenas they appear next to the baselines on the same splits (`summary.json`, `comparison.csv`).

The official-protocol TabArena-Lite result in [Results](#results) is reproduced with TabArena's `TabSil` model ([autogluon/tabarena#753](https://github.com/autogluon/tabarena/pull/753)). The arena suites of `training/scripts/eval.py` fit each split once without bagging, a quicker check used to compare checkpoints; their Elo is not the official one.

## Acknowledgements

TabSil puts together ideas from published work:

- **TabPFN** (Prior Labs): the prior-fitted network approach; the TabPFN-3 / 3.5 technical reports for the column → row → row-ICL backbone. No weights are used. A few inference routines (balanced feature subsampling, chunked class attention) are adapted from the Apache-2.0 TabPFN inference code; see `NOTICE`.
- **TabICL / TabICLv2** (Inria SODA): the graph SCM prior (used through the `tabicl` package), the three-stage training curriculum, and the QASSMax query gate of the `gated_logscale` attention temperature (BSD-3-Clause, `LICENSES/TabICL-BSD-3-Clause.txt`; see `NOTICE`).
- **Kumo-Tabular** (NVIDIA, `NVIDIA/structured-data-models`): interleaved column/row stages, log-scale attention temperatures, and the quantile regression head, re-implemented from the blog post and public source. The per-member numeric transforms and the category count columns of the inference ensemble follow its `numerical_processor` and `AddCategoryCounts` (re-implemented, not copied). The log-scale attention temperature (`LogScale`) follows the Apache-2.0 NVIDIA code (`sdm/nn/scaling.py`); see `NOTICE`.
- **TabPFN-TS** (Prior Labs): forecasting with a tabular model; `TabSilForecaster`'s time features are adapted from the Apache-2.0 `tabpfn-time-series` package (see `NOTICE`). The forecasting evaluation uses **fev** / **fev-bench** (AutoGluon) and the comparison results of fm-agent-bench.
- **Treatment effects**: the meta-learners of Künzel et al. (S/T/X) and Kennedy (DR), with EconML as the baseline. The benchmarks are IHDP (Hill 2011, in the 100-realization files of Shalit et al. 2017) and ACIC 2016 (Dorie et al. 2019, from causallib, CDLA-Sharing-1.0), with the ACIC split of CausalPFN (see `NOTICE`). The data is downloaded at run time.
- **Muon** optimizer, and the **nanoTabPFN speedrun** for the training speed options.

TabSil grew out of [minipfn](https://github.com/Aiden-Jeon/minipfn), a hands-on guide that reimplements PFN and TabPFN from scratch. TabSil is now trained in this repository; minipfn no longer holds code for it.

## License

Apache License 2.0. See `LICENSE` and `NOTICE`.
