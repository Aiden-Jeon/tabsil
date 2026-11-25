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

[TabArena-Lite](https://huggingface.co/spaces/TabArena/leaderboard), scored the way the public leaderboard is. TabSil places 11th of 99 methods. The table lists every method above it and five baselines below it.

| Method | Elo (+CI) | Improvability (%) | Fit (s/1K) | Infer (s/1K) |
| --- | --- | --- | --- | --- |
| Kumo-Tabular (default) | 2046 (+179/−113) | 2.9 | 2.33 | 0.254 |
| LimiX-2 (default) | 1924 (+167/−117) | 3.2 | 31.16 | 8.935 |
| Kumo-Tabular-Medium (default) | 1902 (+140/−100) | 4.9 | 1.36 | 0.150 |
| TabPFN-3.5 (default) | 1858 (+147/−96) | 6.9 | 1.85 | 0.473 |
| Kumo-Tabular-Small (default) | 1801 (+90/−63) | 8.2 | 0.79 | 0.092 |
| Causilo (default) | 1798 (+95/−71) | 8.3 | 0.67 | 0.104 |
| TabFM (default) | 1797 (+131/−108) | 7.1 | 16.13 | 6.663 |
| TabPFN-3.5-Fast (default) | 1786 (+90/−57) | 8.6 | 0.73 | 0.153 |
| Mitra-v2 (default) | 1751 (+107/−75) | 7.8 | 59.03 | 0.886 |
| EXAONE-Tabular (default) | 1741 (+72/−50) | 9.3 | 1.98 | 0.582 |
| **TabSil** | **1726** (+104/−77) | 8.2 | 0.0141 | 0.805 |
| TabPFN-3 (default) | 1619 (+74/−52) | 11.1 | 1.29 | 0.383 |
| TabICLv2 (default) | 1557 (+68/−58) | 12.6 | 0.73 | 0.137 |
| RealTabPFN-2.5 (default) | 1514 (+57/−41) | 13.3 | 5.72 | 0.611 |
| CatBoost (default) | 1339 (+35/−40) | 17.8 | 5.81 | 0.025 |
| XGBoost (default) | 1182 (+50/−53) | 21.0 | 2.06 | 0.122 |

- TabArena-Lite has 51 datasets, one split each. TabSil ran on 1×H100 with the default inference (8 preprocessing members). No split is imputed. Elo puts RandomForest (default) at 1000, with a 95% CI. Lower is better for improvability, fit and infer.
- Every method on the leaderboard is 8-fold bagged; TabSil is fit once per split, and its fit only stores the context. So the Fit column is not like-for-like. The Infer column is comparable.
- All values come from the same run. The pool is the public leaderboard [v0.1.9.12](https://huggingface.co/spaces/TabArena/leaderboard/blob/3cffdcceb64f9cd6c6cbb5aa852a4f16296749bd/data/entrants_models/imputation_yes/splits_lite/tasks_all/datasets_all/website_leaderboard.csv) (98 methods) plus TabSil, 99 methods. The run recomputes every method's Elo, CI and improvability from the leaderboard's public results with TabSil in the pool, so they differ slightly from v0.1.9.12 (Elo by at most 4 for the methods shown). Fit and infer times are the same as on the leaderboard. The live leaderboard has added methods since, so its numbers differ too.
- Fit and infer time are defined in [Evaluation](#evaluation). More results (AMLB, per-stage, per-task Elo) are on the [model card](https://huggingface.co/aiden-jeon/tabsil).

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
```

Each `uv sync` makes the environment match that command exactly (it removes what the command does not install): `--package tabsil-train` leaves out the root dev tools such as Jupyter, so use `--all-packages` to have both. The `arena` extra needs pandas 2.3, so it is installed without the dev groups (pandas 3).

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

The published v1 weights were trained with shorter stages, 150K / 10K / 2K steps (`--steps`; warmup and cosine follow the stage length): a separate 250K stage-1 run, measured up to 225K, did no better than 150K on TabArena-Lite. Their intermediate stages are branches of the Hub repo (`stage1-150k`, `stage2-10k`); see the [model card](https://huggingface.co/aiden-jeon/tabsil).

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

`training/scripts/eval.py` evaluates a checkpoint, MLflow run, or Hub model on AMLB (classification / CTR23 regression), TabArena(-Lite), and BeyondArena, plus uncertainty metrics (CRPS, coverage, PIT, calibration).
Train and inference time are reported as on the TabArena leaderboard: seconds per 1K train rows (fit) and per 1K test rows (predict), averaged over each dataset's splits, then the median over datasets. In the arenas they appear next to the baselines on the same splits (`summary.json`, `comparison.csv`).

## Acknowledgements

TabSil puts together ideas from published work:

- **TabPFN** (Prior Labs): the prior-fitted network approach; the TabPFN-3 / 3.5 technical reports for the column → row → row-ICL backbone. No weights are used. A few inference routines (balanced feature subsampling, chunked class attention) are adapted from the Apache-2.0 TabPFN inference code; see `NOTICE`.
- **TabICL / TabICLv2** (Inria SODA): the graph SCM prior (used through the `tabicl` package), the three-stage training curriculum, and the QASSMax query gate of the `gated_logscale` attention temperature (BSD-3-Clause, `LICENSES/TabICL-BSD-3-Clause.txt`; see `NOTICE`).
- **Kumo-Tabular** (NVIDIA, `NVIDIA/structured-data-models`): interleaved column/row stages, log-scale attention temperatures, and the quantile regression head, re-implemented from the blog post and public source. The log-scale attention temperature (`LogScale`) follows the Apache-2.0 NVIDIA code (`sdm/nn/scaling.py`); see `NOTICE`.
- **Muon** optimizer, and the **nanoTabPFN speedrun** for the training speed options.

TabSil grew out of [minipfn](https://github.com/Aiden-Jeon/minipfn), a hands-on guide that reimplements PFN and TabPFN from scratch. TabSil is now trained in this repository; minipfn no longer holds code for it.

## License

Apache License 2.0. See `LICENSE` and `NOTICE`.
