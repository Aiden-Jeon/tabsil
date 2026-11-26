<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/tabsil-logo-dark.svg">
    <img alt="TabSil" src="assets/tabsil-logo.svg" width="320">
  </picture>
</p>

# TabSil

TabSil is a tabular foundation model. One model handles both classification and regression: give it a training table and it predicts test rows in a single forward pass (in-context learning), with no gradient steps on your data. It is trained only on synthetic tables drawn from a structural causal model prior.

*Sil* (실) is Korean for *thread*: the model reads a table by weaving through its columns and rows.

This repository has everything needed to reproduce TabSil. It holds two packages:

- `tabsil` (`src/tabsil/`): inference only. The model, the scikit-learn style estimators, and loading weights from the Hugging Face Hub.
- `tabsil-train` (`training/`, imported as `tabsil_train`): the synthetic-data prior, the training curriculum, the benchmark harness, and the Hub upload scripts. It is a uv workspace member that depends on `tabsil`.

> Status: the TabSil v1 weights are on the Hugging Face Hub: [aiden-jeon/tabsil](https://huggingface.co/aiden-jeon/tabsil) (model card with the benchmark results).

## Results

[TabArena-Lite](https://huggingface.co/spaces/TabArena/leaderboard) with TabArena's official pipeline: every method, TabSil included, is 8-fold bagged, and the leaderboard is computed as on the public one. TabSil places 18th of 121 methods. The table lists TabSil, the foundation models near it, the strongest tuned baselines and the RandomForest anchor; the [full table](https://huggingface.co/datasets/aiden-jeon/tabsil-tabarena-lite/blob/1c4beda3ed9683fa5bd15f4cc9569a8180b04ec4/leaderboard_website.md) has all 121.

| Method | Place | Elo (+CI) | Improvability (%) | Fit (s/1K) | Infer (s/1K) |
| --- | --- | --- | --- | --- | --- |
| Kumo-Tabular (default) | 1 | 2090 (+159/−110) | 3.7 | 2.33 | 0.254 |
| TabPFN-3.5 (default) | 7 | 1882 (+139/−91) | 7.5 | 1.85 | 0.473 |
| TabPFN-3.5-Fast (default) | 16 | 1806 (+91/−59) | 9.1 | 0.73 | 0.153 |
| Mitra-v2 (default) | 17 | 1774 (+101/−68) | 8.5 | 59.03 | 0.886 |
| **TabSil** | **18** | **1755** (+88/−73) | 9.0 | 2.53 | 0.812 |
| EXAONE-Tabular (default) | 20 | 1753 (+66/−52) | 9.9 | 1.98 | 0.582 |
| TabPFN-3 (default) | 25 | 1626 (+73/−48) | 11.6 | 1.29 | 0.383 |
| TabICLv2 (default) | 29 | 1567 (+61/−57) | 13.1 | 0.73 | 0.137 |
| RealTabPFN-2.5 (default) | 33 | 1519 (+59/−44) | 13.8 | 5.72 | 0.611 |
| RealMLP (tuned + ensembled) | 36 | 1489 (+45/−44) | 15.2 | 2791.97 | 13.886 |
| CatBoost (tuned) | 45 | 1394 (+46/−39) | 16.6 | 1347.32 | 0.037 |
| LightGBM (tuned + ensembled) | 46 | 1387 (+36/−37) | 17.6 | 416.63 | 2.236 |
| XGBoost (tuned + ensembled) | 55 | 1349 (+39/−42) | 18.3 | 700.96 | 1.438 |
| RandomForest (default) | 112 | 1000 (+74/−75) | 28.4 | 0.43 | 0.053 |

- TabArena-Lite has 51 datasets, one split each. TabSil ran as the `TabSil` model of TabArena ([autogluon/tabarena#753](https://github.com/autogluon/tabarena/pull/753)) at its default configuration: tabsil 0.1.0 from PyPI, the weights at the tag `v1`, 8×H100 with one process per GPU. No split is imputed. Elo puts RandomForest (default) at 1000, with a 95% CI. Lower is better for improvability, fit and infer.
- All methods use the same protocol, so the Fit and Infer columns are like-for-like; the other methods' times were measured by TabArena on its own hardware.
- The pool is the TabArena results available when the run was made (2026-11-25, TabArena at the submission commit `1360e0e8`): 120 methods plus TabSil, 121 methods. It is not a released leaderboard version, so the live leaderboard's numbers differ. These are the results submitted with the pull request; the maintainers' own run of the full TabArena may change them. The raw results (`results.pkl` of all 51 splits), the hardware and the logs are in [aiden-jeon/tabsil-tabarena-lite](https://huggingface.co/datasets/aiden-jeon/tabsil-tabarena-lite).
- tabsil 0.1.1 changed the inference defaults (see [Use](#use)); the table above is tabsil 0.1.0's, and the official-protocol run of 0.1.1 follows its release.
  - In the repository's harness (each split fit once, no bagging; see [Evaluation](#evaluation)) the same weights score 1768 with 0.1.1's inference against 1726 with 0.1.0's.
  - Median fit time is 0.23 s per 1K training rows, against 0.014; 0.1.1's fit runs the training rows through the model.
  - Median predict time is 0.16 s per 1K test rows, against 0.81.
- Fit and infer time are defined in [Evaluation](#evaluation). More results (AMLB, per-problem-type and per-stage Elo) are on the [model card](https://huggingface.co/aiden-jeon/tabsil).

### Forecasting (fev-bench)

Zero-shot forecasting with the v1 weights (`TabSilForecaster`, no retraining) on 10 univariate [fev-bench](https://github.com/autogluon/fev) tasks, every rolling window (129), scored with fev's SQL (scaled quantile loss, the main metric) and MASE. The tasks, windows and comparison results are those of [fm-agent-bench](https://github.com/Aiden-Jeon/fm-agent-bench) P7 (commit `b1b0101`), which ran the other methods with the same protocol. Elo is computed over the task means with Seasonal Naive at 1000, with a 95% CI from 200 bootstrap rounds over the tasks.

| Method | SQL Elo | MASE Elo |
| --- | --- | --- |
| Chronos-2 | 1843 (+192/−85) | 1689 (+209/−98) |
| TabPFN-TS-3 (public result) | 1765 (+306/−173) | 1652 (+309/−146) |
| TimesFM-2.5 | 1708 (+235/−133) | 1628 (+252/−143) |
| TabPFN-TS (TabPFN v2) | 1594 (+237/−143) | 1490 (+179/−118) |
| **TabSil** | **1526** (+202/−132) | **1460** (+192/−130) |
| Chronos-Bolt | 1438 (+125/−148) | 1332 (+144/−174) |
| TabICL-TS | 1438 (+126/−84) | 1200 (+109/−114) |
| AutoARIMA (public result) | 1335 (+125/−263) | 1178 (+158/−216) |
| AutoETS | 1230 (+245/−478) | 1167 (+243/−299) |
| Seasonal Naive | 1000 (+78/−271) | 1000 (+122/−332) |

- Snapshot 2026-11-26: fev 0.10.0, data `autogluon/fev_datasets` at `d3e7e72`, 1×H100 (42 min for TabSil). Our Seasonal Naive reproduces fm-agent-bench's on every window. No window is imputed. Reproduce with `training/scripts/eval.py --suite fev` (job spec `training/scripts/eval_fev.yaml`).
- The tasks are redset_1H, solar_with_weather_1H, ercot_1D, epf_be, rohlik_orders_1D, solar_1D, ecdc_ili, hospital, us_consumption_1M and hospital_admissions_1D. Only known covariates are used, and only by TabSil and TabPFN-TS. The pool differs from the fev-bench leaderboard's, so the numbers are not comparable to it.

### Outlier detection (ADBench)

Unsupervised outlier detection on 12 [ADBench](https://github.com/Minqi824/ADBench) tables × 3 seeds, zero-shot with the v1 weights (`TabSilUnsupervised`, 10 column orders). Each model is fitted on the unlabeled training split, which keeps its anomalies (a contaminated context), and scores the test split. The split, the tables and the comparison numbers come from the fm-agent-bench anomaly track: the official TabPFN extension (`TabPFNUnsupervisedModel.outliers`, TabPFN v2 weights) and pyod's default methods. Our IsolationForest equals ADBench Table D4 on 12/12 tables.

| Method | Mean AUROC | Mean AUPRC | Elo (IsolationForest = 1000) |
| --- | --- | --- | --- |
| IsolationForest | 0.722 | 0.358 | 1000 (+89/−82) |
| TabPFN v2 extension | 0.704 | 0.308 | 991 (+198/−156) |
| ECOD | 0.695 | 0.358 | 971 (+192/−128) |
| **TabSil** | **0.662** | **0.264** | **896** (+149/−143) |

- Elo pool: TabSil, the TabPFN extension and pyod's 10 default methods; TabSil places 9th of 12. TabSil is above the TabPFN extension on 1 of 12 tables, above IsolationForest on 5.
- With a clean context (only the training rows labeled normal, a diagnostic that uses labels and stays out of the Elo) TabSil's mean AUROC is 0.820 (TabPFN extension 0.847, IsolationForest 0.767), and it is within 0.01 of the best pyod method or above it on 8 of 12 tables. Like the TabPFN extension, the density model is much stronger than the unsupervised score suggests; the anomalies in the context hurt it most (cover +0.46, cardio +0.31).
- Run with `training/scripts/eval.py --suite adbench` (8×H100, 14 min). Per-table numbers, the differences from the official extension and what is not verified are in [#19](https://github.com/Aiden-Jeon/tabsil/pull/19).

### Treatment effects (IHDP, ACIC 2016)

CATE estimation zero-shot with the v1 weights at the Hub tag `v1`, i.e. tabsil 0.1.0's inference settings (`standard` members, temperature 0.9; `TabSilCATE()` without a model now loads `v1.1`, which was not measured here). TabSil is the base learner of S/T/X/DR meta-learners, on IHDP (100 realizations: the files of Shalit et al. 2017, 672 training and 75 test rows each) and ACIC 2016 (the 10 simulations shipped with causallib, 4,802 rows × 58 covariates, CausalPFN's 90/10 split). √PEHE is against the noiseless effect mu1 − mu0, within-sample (training rows) and out-of-sample (test rows); ε_ATE is the absolute ATE error. Mean ± standard error over realizations, lower is better. The EconML baselines (0.17.1, default settings, scikit-learn gradient boosting as every base model) ran on the same splits.

| Method | IHDP √PEHE in | IHDP √PEHE out | IHDP ε_ATE out | ACIC √PEHE in | ACIC √PEHE out | ACIC ε_ATE out |
| --- | --- | --- | --- | --- | --- | --- |
| **TabSil S-learner** (default) | **0.50 ± 0.07** | **0.53 ± 0.09** | **0.12 ± 0.01** | 1.09 ± 0.38 | 0.55 ± 0.11 | 0.05 ± 0.02 |
| TabSil T-learner | 0.60 ± 0.06 | 0.62 ± 0.09 | 0.13 ± 0.01 | 1.05 ± 0.24 | 0.73 ± 0.13 | 0.05 ± 0.01 |
| TabSil X-learner | 0.89 ± 0.09 | 0.97 ± 0.16 | 0.18 ± 0.02 | **0.87 ± 0.28** | **0.38 ± 0.04** | **0.05 ± 0.01** |
| TabSil DR-learner | 0.82 ± 0.07 | 0.75 ± 0.07 | 0.24 ± 0.02 | 1.21 ± 0.30 | 0.68 ± 0.19 | 0.05 ± 0.02 |
| EconML T-learner (GBM) | 1.43 ± 0.17 | 2.08 ± 0.33 | 0.25 ± 0.04 | 1.22 ± 0.19 | 0.86 ± 0.03 | 0.13 ± 0.03 |
| EconML X-learner (GBM) | 2.58 ± 0.36 | 2.91 ± 0.48 | 0.32 ± 0.04 | 1.04 ± 0.24 | 0.69 ± 0.09 | 0.08 ± 0.02 |
| EconML S-learner (GBM) | 2.99 ± 0.48 | 3.24 ± 0.58 | 0.49 ± 0.11 | 1.54 ± 0.30 | 1.16 ± 0.15 | 0.21 ± 0.05 |
| EconML DR-learner (GBM) | 19.49 ± 2.75 | 11.38 ± 1.20 | 1.84 ± 0.26 | 2.06 ± 0.44 | 1.32 ± 0.24 | 0.16 ± 0.04 |
| EconML CausalForestDML | 3.60 ± 0.53 | 3.85 ± 0.61 | 0.60 ± 0.07 | 1.65 ± 0.37 | 1.23 ± 0.24 | 0.15 ± 0.04 |

- On IHDP, every TabSil learner has a lower mean √PEHE, within- and out-of-sample, than every EconML baseline (the ATE errors are closer: EconML's T-learner has the lowest within-sample ε_ATE after TabSil's S and T). The S-learner's out-of-sample √PEHE is below the best EconML method's on 100 of 100 realizations; the means are pulled up by a few realizations with large effects (realization 9: 7.4 for TabSil S vs 17.6 for EconML T). On ACIC 2016 the X-learner is best out-of-sample, TabSil's T-learner is behind EconML's X-learner, and within-sample only TabSil's X-learner is ahead of EconML's X-learner. The default is the S-learner: the best average rank of the four and one fit.
- Published numbers on the same files, with protocol differences (`training/src/tabsil_train/benchmarks/causal_reference.csv`): CausalPFN (Balazadeh et al. 2025) reports out-of-sample √PEHE 0.58 ± 0.07 on IHDP and 0.92 ± 0.11 on ACIC 2016, TabPFN as the outcome model 0.95 ± 0.20 and 0.54 ± 0.08, with baselines tuned per dataset (their Table 1, IHDP T-learner 1.73, BART 2.50). Shalit et al. 2017 report CFR-Wass 0.71 / 0.76 (in / out) on IHDP, averaged over 1,000 realizations instead of these 100.
- Snapshot 2026-11-26: TabSil on 1×H100 (9 min job, 440 fits of a learner on a realization, 0 failed, nothing imputed); EconML on a laptop CPU. Reproduce with `training/scripts/eval.py --suite causal --regressor-path hf://aiden-jeon/tabsil@v1` (job spec `training/scripts/eval_causal.yaml`; the EconML methods need the `causal` extra).

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
uv sync                           # tabsil + dev tools, Jupyter (inference tests: uv run pytest tests)
uv sync --all-packages            # tabsil + tabsil-train (training, priors, AMLB evaluation) + dev tools, Jupyter
uv sync --package tabsil-train    # tabsil-train and its test tools only (tests: uv run --package tabsil-train pytest training/tests)
uv sync --package tabsil-train --no-dev --extra arena --group arena-dev   # TabArena / BeyondArena evaluation
uv sync --package tabsil-train --no-dev --extra fev --group arena-dev     # fev-bench forecasting evaluation
uv sync --package tabsil-train --extra causal                             # treatment effect evaluation (EconML baselines)
```

Each `uv sync` makes the environment match that command exactly (it removes what the command does not install): `--package tabsil-train` leaves out the root dev tools such as Jupyter, so use `--all-packages` to have both. The `arena` extra needs pandas 2.3, so it is installed without the dev groups (pandas 3). The `fev` extra (fev 0.10.0) needs `huggingface-hub<2`, so it is installed without the dev groups too.

Python ≥ 3.11, PyTorch ≥ 2.11 (the benchmark results below were measured with PyTorch 2.11).

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

### Forecasting

`TabSilForecaster` forecasts time series zero-shot with the regression model, as TabPFN-TS does: each time step becomes a table row of time features (running index, calendar sin/cos, FFT-detected seasonal sin/cos; no lags), and the horizon is predicted in one pass from the series' own past. The quantiles come from the regressor's predictive distribution; the point forecast is its median.

```python
from tabsil import TabSilForecaster

# context: one row per (item_id, timestamp) with a target column; other columns are covariates
fc = TabSilForecaster(device="cuda").fit(context)
pred = fc.predict(24)  # item_id, timestamp, predictions, "0.1" ... "0.9"
pred = fc.predict(future=future, quantile_levels=[0.05, 0.5, 0.95])  # known covariates in `future`
```

Each series is fit on its last 4,096 rows (`max_context_length`; `None` allows up to the supported 1,000,000 rows). See `src/tabsil/forecast.py` for the missing-value and constant-series handling.

### Outlier detection

Without labels, `TabSilUnsupervised` gives each row a log density and an outlier score (higher = more anomalous). It predicts each column from the columns before it with the same model (regression buckets for numeric columns, class probabilities for categorical ones), chains the conditionals along 10 random column orders, and averages them, as the official TabPFN extension does ([results](#outlier-detection-adbench)):

```python
from tabsil import TabSilUnsupervised

model = TabSilUnsupervised(device="cuda").fit(X_train)  # unlabeled rows: the context
scores = model.outlier_score(X_test)  # = -model.log_density(X_test)
```

The cost is one fit and prediction per column and order (`n_permutations`, default 10). `model.conditional(X, target, given)` returns the distribution of one column given others, the part imputation and synthetic data can later build on.

### Treatment effects

`TabSilCATE` estimates conditional average treatment effects of a binary treatment on a continuous outcome from observational data, with TabSil as the base learner of a meta-learner (zero-shot, the v1 weights): S-, T-, X- or DR-learner (`learner="s" | "t" | "x" | "dr"`). Outcome models are `TabSilRegressor`, propensity scores `TabSilClassifier`.

```python
from tabsil import TabSilCATE

est = TabSilCATE(device="cuda").fit(X, treatment, y)  # the S-learner
tau = est.effect(X_test)  # CATE per row
ate = est.ate()  # mean CATE over the training rows
```

`regressor=` / `classifier=` take any scikit-learn estimator instead of TabSil. See `src/tabsil/causal.py` for the learners and where they differ from EconML's.

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

Synthetic tables come from the TabICLv2 graph SCM prior (`tabicl` package) with at most 100 features. Training follows the TabICLv2 three-stage curriculum, with Muon on every parameter, batch 256 on 32 GPUs, and 1% warmup then cosine decay in each stage:

| Stage | Rows per table (train share) | Steps | Peak lr | Starts from |
|---|---|---|---|---|
| 1 | 1,024 (30–90%) | 250K | 8e-4 | scratch |
| 2 | 400–10,240 (80%) | 20K | 1e-4 | stage 1 weights |
| 3 | 400–60,000 (80%) | 5K | 2e-5 | stage 2 weights |

The published v1 weights were trained with shorter stages, 150K / 10K / 2K steps (`--steps`; warmup and cosine follow the stage length): a separate 250K stage-1 run, measured up to 225K, did no better than 150K on TabArena-Lite (measured with `training/scripts/eval.py`, without bagging). Their intermediate stages are branches of the Hub repo (`stage1-150k`, `stage2-10k`); see the [model card](https://huggingface.co/aiden-jeon/tabsil).

```bash
# local check (CPU, about a minute)
uv run --package tabsil-train python training/scripts/train.py --config-name smoke

# full training: Databricks AI Runtime job specs in training/scripts/train.yaml
uv run --package tabsil-train python training/scripts/train.py --config-name stage1
uv run --package tabsil-train python training/scripts/train.py --config-name stage2 --init-run-id <stage 1 run>
uv run --package tabsil-train python training/scripts/train.py --config-name stage3 --init-run-id <stage 2 run>

# the v1 weights: the same stages with --steps 150000 / 10000 / 2000
```

Set `PYTHONHASHSEED=0` for reproducible data: the `tabicl` prior iterates over a set of strings, so its tables depend on the hash seed.

## Evaluation

`training/scripts/eval.py` evaluates a checkpoint, MLflow run, or Hub model on AMLB (classification / CTR23 regression), TabArena(-Lite), and BeyondArena, plus uncertainty metrics (CRPS, coverage, PIT, calibration), forecasting on 10 fev-bench tasks (`--suite fev`, job spec `training/scripts/eval_fev.yaml`), unsupervised outlier detection on 12 ADBench tables (`--suite adbench`, job spec in `training/scripts/eval.yaml`), and treatment effects on IHDP and ACIC 2016 (`--suite causal`, job spec `training/scripts/eval_causal.yaml`).
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
