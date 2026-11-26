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
```

Each `uv sync` makes the environment match that command exactly (it removes what the command does not install): `--package tabsil-train` leaves out the root dev tools such as Jupyter, so use `--all-packages` to have both. The `arena` extra needs pandas 2.3, so it is installed without the dev groups (pandas 3). The `fev` extra (fev 0.10.0) needs `huggingface-hub<2`, so it is installed without the dev groups too.

Python ≥ 3.11, PyTorch ≥ 2.11 (the benchmark results below were measured with PyTorch 2.11).

## Use

Without a model, `TabSilClassifier()` and `TabSilRegressor()` download the v1 weights from the default Hugging Face Hub repo (`tabsil.recipe.HUB_REPO`, [aiden-jeon/tabsil](https://huggingface.co/aiden-jeon/tabsil)) at the tag `v1` (`tabsil.recipe.HUB_REVISION`), so a release keeps downloading the same weights when the repo's `main` changes:

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

Both are scikit-learn style estimators over an ensemble of 8 preprocessing members; each member sees at most 768 of the original columns.

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

`training/scripts/eval.py` evaluates a checkpoint, MLflow run, or Hub model on AMLB (classification / CTR23 regression), TabArena(-Lite), and BeyondArena, plus uncertainty metrics (CRPS, coverage, PIT, calibration), and forecasting on 10 fev-bench tasks (`--suite fev`, job spec `training/scripts/eval_fev.yaml`).
Train and inference time are reported as on the TabArena leaderboard: seconds per 1K train rows (fit) and per 1K test rows (predict), averaged over each dataset's splits, then the median over datasets. In the arenas they appear next to the baselines on the same splits (`summary.json`, `comparison.csv`).

The official-protocol TabArena-Lite result in [Results](#results) is reproduced with TabArena's `TabSil` model ([autogluon/tabarena#753](https://github.com/autogluon/tabarena/pull/753)). The arena suites of `training/scripts/eval.py` fit each split once without bagging, a quicker check used to compare checkpoints; their Elo is not the official one.

## Acknowledgements

TabSil puts together ideas from published work:

- **TabPFN** (Prior Labs): the prior-fitted network approach; the TabPFN-3 / 3.5 technical reports for the column → row → row-ICL backbone. No weights are used. A few inference routines (balanced feature subsampling, chunked class attention) are adapted from the Apache-2.0 TabPFN inference code; see `NOTICE`.
- **TabICL / TabICLv2** (Inria SODA): the graph SCM prior (used through the `tabicl` package), the three-stage training curriculum, and the QASSMax query gate of the `gated_logscale` attention temperature (BSD-3-Clause, `LICENSES/TabICL-BSD-3-Clause.txt`; see `NOTICE`).
- **Kumo-Tabular** (NVIDIA, `NVIDIA/structured-data-models`): interleaved column/row stages, log-scale attention temperatures, and the quantile regression head, re-implemented from the blog post and public source. The log-scale attention temperature (`LogScale`) follows the Apache-2.0 NVIDIA code (`sdm/nn/scaling.py`); see `NOTICE`.
- **TabPFN-TS** (Prior Labs): forecasting with a tabular model; `TabSilForecaster`'s time features are adapted from the Apache-2.0 `tabpfn-time-series` package (see `NOTICE`). The forecasting evaluation uses **fev** / **fev-bench** (AutoGluon) and the comparison results of fm-agent-bench.
- **Muon** optimizer, and the **nanoTabPFN speedrun** for the training speed options.

TabSil grew out of [minipfn](https://github.com/Aiden-Jeon/minipfn), a hands-on guide that reimplements PFN and TabPFN from scratch. TabSil is now trained in this repository; minipfn no longer holds code for it.

## License

Apache License 2.0. See `LICENSE` and `NOTICE`.
