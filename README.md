# TabSil

TabSil is a tabular foundation model. One model handles both classification and regression: give it a training table and it predicts test rows in a single forward pass (in-context learning), with no gradient steps on your data. It is trained only on synthetic tables drawn from a structural causal model prior.

*Sil* (실) is Korean for *thread*: the model reads a table by weaving through its columns and rows.

This repository has everything needed to reproduce TabSil: the model, the synthetic-data prior, the training curriculum, inference, and the benchmark harness.

> Status: the code is complete; trained weights are not published yet.

## Install

```bash
git clone https://github.com/Aiden-Jeon/tabsil
cd tabsil
uv sync                  # inference and training
uv sync --extra arena    # + TabArena / BeyondArena evaluation
```

Python ≥ 3.12, PyTorch 2.11.

## Use

```python
from tabsil import TabSilClassifier, TabSilRegressor

clf = TabSilClassifier().fit(X_train, y_train)  # downloads the weights from the Hugging Face Hub
proba = clf.predict_proba(X_test)

reg = TabSilRegressor().fit(X_train, y_train)
y_pred = reg.predict(X_test)
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

Code: `src/tabsil/models/column_row_transformer.py`; all numbers are in `src/tabsil/recipe.py`.

## Training

Synthetic tables come from the TabICLv2 graph SCM prior (`tabicl` package) with at most 100 features. Training follows the TabICLv2 three-stage curriculum, with Muon on every parameter, batch 256 on 32 GPUs, and 1% warmup then cosine decay in each stage:

| Stage | Rows per table (train share) | Steps | Peak lr | Starts from |
|---|---|---|---|---|
| 1 | 1,024 (30–90%) | 250K | 8e-4 | scratch |
| 2 | 400–10,240 (80%) | 20K | 1e-4 | stage 1 weights |
| 3 | 400–60,000 (80%) | 5K | 2e-5 | stage 2 weights |

```bash
# local check (CPU, about a minute)
uv run python scripts/train.py --config-name smoke

# full training: Databricks AI Runtime job specs in scripts/train.yaml
uv run python scripts/train.py --config-name stage1
uv run python scripts/train.py --config-name stage2 --init-run-id <stage 1 run>
uv run python scripts/train.py --config-name stage3 --init-run-id <stage 2 run>
```

Set `PYTHONHASHSEED=0` for reproducible data: the `tabicl` prior iterates over a set of strings, so its tables depend on the hash seed.

## Evaluation

`scripts/eval.py` evaluates a checkpoint, MLflow run, or Hub model on AMLB (classification / CTR23 regression), TabArena(-Lite), and BeyondArena, plus uncertainty metrics (CRPS, coverage, PIT, calibration) and test-time compute options (greedy ensemble selection, DiagScale).

## Acknowledgements

TabSil puts together ideas from published work:

- **TabPFN** (Prior Labs): the prior-fitted network approach; the TabPFN-3 / 3.5 technical reports for the column → row → row-ICL backbone. The public inference code was read for design details; no code or weights are used.
- **TabICL / TabICLv2** (Inria SODA): the graph SCM prior (used through the `tabicl` package) and the three-stage training curriculum.
- **Kumo-Tabular** (NVIDIA, `NVIDIA/structured-data-models`): interleaved column/row stages, log-scale attention temperatures, and the quantile regression head, re-implemented from the blog post and public source.
- **Muon** optimizer, and the **nanoTabPFN speedrun** for the training speed options.

TabSil grew out of [minipfn](https://github.com/Aiden-Jeon/minipfn), a hands-on guide that reimplements PFN and TabPFN from scratch. Checkpoints that minipfn trained for this architecture (`minipfn-v1`) load in TabSil.

## License

Apache License 2.0. See `LICENSE` and `NOTICE`.
