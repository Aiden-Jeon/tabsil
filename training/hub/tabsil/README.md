---
license: apache-2.0
library_name: pytorch
pipeline_tag: tabular-classification
tags:
  - tabular-foundation-model
  - prior-data-fitted-networks
  - in-context-learning
  - tabular-classification
  - tabular-regression
---

![TabSil](images/tabsil-banner.png)

# TabSil

TabSil is a tabular foundation model. One set of weights handles both classification and regression: give it a training table and it predicts the test rows in a single forward pass (in-context learning), with no gradient steps on your data. It was trained only on synthetic tables from the TabICLv2 graph SCM prior. Code: [Aiden-Jeon/tabsil](https://github.com/Aiden-Jeon/tabsil) (model `src/tabsil/models/column_row_transformer.py`, structure and inference `src/tabsil/recipe.py`, training `training/src/tabsil_train/recipe.py`).

*Sil* (실) is Korean for *thread*: the model reads a table by weaving through its columns and rows.

TabSil combines published designs (TabPFN-3.5, Kumo-Tabular, TabICLv2; see Acknowledgements). It is not a checkpoint of any of them, and it includes none of their weights.

## Model

- Architecture: column/row stages → in-context stage (`column_row_transformer`), 221,548,733 parameters.
  - **Cell encoder** (TabPFN-3.5): each feature is z-normalised over the training rows, and missing cells get an indicator. A token holds 3 features. Learned Fourier features and within-column ECDF ranks are added. Width 256.
  - **Column / row stages** (Kumo-Tabular): 6 interleaved pairs of a column block (256 inducing points that summarise the training rows) and a row block (4 CLS tokens per row), 4 heads each. The 4 CLS outputs are joined into one 1,024-wide row vector. Test rows read the training-row summaries but never each other.
  - **In-context stage** (TabPFN-3.5): 24 layers, 16 heads, width 1,024. Training rows attend to each other; test rows attend to training rows only.
  - **Attention temperature** (Kumo-Tabular): gain-free q/k RMSNorm × a learned log-scale (`logscale`; the row stage uses `gated_logscale`, whose query gate follows TabICLv2's QASSMax). No logit soft-cap.
  - **Outputs:** classification uses a many-class retrieval decoder (TabPFN-3.5). Regression predicts 999 quantiles trained with the pinball loss (Kumo-Tabular), turned into a distribution over 5,000 buckets for prediction. The target encoders and a pre-head MLP are per task; everything else is shared.
- Inputs: up to 160 classes, 1,000,000 training rows and 6,000 features (the limits TabSil is evaluated with, the same as TabPFN-3.5's). Training covered up to 60,000 rows and 100 features.
- Inference: an ensemble of 8 members with standard scaling. Each member shuffles features and classes and adds a random row-identifier column. Categorical columns with at most 20 categories are one-hot encoded, as long as the table stays within 100 columns. Tables with more than 768 columns give each member 768 of them, split so that every column is used about equally often (`balanced`). Classification averages the members' softmax at temperature 0.9; in regression, half of the members also power-transform the target. The member count, kinds and column cap ship in `config.json` (`inference`).

## Training

The TabICLv2 three-stage curriculum ([arXiv 2602.11139](https://arxiv.org/abs/2602.11139), public scripts `scripts/train_v2_clf_stage{1,2,3}.sh`), shortened (see below). Each stage starts from the previous stage's weights with a fresh optimizer: 1% warmup, then cosine decay. All stages use batches of 256 tables, Muon on every parameter (momentum 0.9, weight decay 0.01), bf16 and 32×H100. Each step trains one task, and the tasks alternate.

| Stage | Steps | Tables | Peak lr | Grad clip |
| --- | --- | --- | --- | --- |
| 1 | 150k | 1,024 rows, 30–90% of them training rows | 8e-4 | 10 |
| 2 | 10k | 400–10,240 rows (log-uniform), 79–81% training rows | 1e-4 | 10 |
| 3 | 2k | 400–60,000 rows (log-uniform), 79–81% training rows | 2e-5 | 1 |

In every stage, tables have at most 100 features (fewer for the longest tables). Half of the classification tables are binary. Graphs and datasets that the TabICL filter marks as unpredictable are drawn again.

What differs from the TabICLv2 scripts (our choices):

- **One multitask model, batch 256.** TabICLv2 trains separate classification and regression models with batch 64.
- **Shorter stages.** The recipe in `tabsil_train.recipe` keeps the per-task table count of the scripts (250k / 20k / 5k steps). These weights were trained for 150k / 10k / 2k steps (`--steps`). A separate 250k stage-1 run did not do better than the 150k run: TabArena-Lite 1423 at 225k vs 1429 at 150k, and AMLB classification 0.9383 in both. In both runs the AMLB regression ratio rose during stage 1 (to 1.137 at 225k and 1.136 at 150k); stage 2 brings it back below 1.
- **Activation checkpointing** for tables longer than 2,048 rows in stages 2 and 3, to fit long tables in GPU memory. It changes memory and speed only.

Compute: 32×H100 for about 30 hours (stage 1), 9.6 hours (stage 2) and 9.0 hours (stage 3), about 1,550 H100-hours. This does not count the 250k stage-1 run or shorter experiments.

The weights were trained with the same model code in [minipfn](https://github.com/Aiden-Jeon/minipfn), before TabSil moved to its own repository, and converted to the TabSil format.

The weights in this repo (`sha256:e03c8c32849a`, the first 12 hex digits of the SHA-256 of `model.safetensors`) are the raw weights at the end of stage 3. They are not a weight average. Every number on this card was measured with these weights, or with the stage-1 and stage-2 branch weights where noted.

Validation uses fixed prior tables at each stage's sizes. Final values (classification CE / regression pinball loss): 0.200 / 0.0495 after stage 1, 0.248 / 0.0492 after stage 2 and 0.320 / 0.0548 after stage 3. Later stages have longer tables, so these values are not comparable across stages.

## Results

### TabPFN v2 benchmark

The 29 AMLB classification and 28 regression datasets from the TabPFN v2 paper, 10 OpenML folds each. Classification is scored by mean ROC AUC. For regression, each dataset's RMSE is divided by the paper's, and the table gives the geometric mean of these ratios (below 1 is better than the paper). The reference values are the TabPFN v2 paper's (Supplementary Tables 1 and 2, default settings). "minipfn TabPFN-3.5" is a from-scratch TabPFN-3.5 reimplementation trained on the same prior ([aiden-jeon/minipfn-tabpfn-v3-5](https://huggingface.co/aiden-jeon/minipfn-tabpfn-v3-5)), scored with the same code.

| Checkpoint | Mean ROC AUC | RMSE ratio (geo. mean) |
| --- | --- | --- |
| Stage 1 (150k steps) | 0.9383 | 1.136 |
| Stage 2 (+10k steps) | 0.9390 | 0.997 |
| **Stage 3 (+2k steps, this model)** | **0.9390** | **0.996** |
| minipfn TabPFN-3.5 | 0.9371 | 1.005 |
| TabPFN v2 paper | 0.9323 | 1 |

Classification: this model beats the paper on 25 of 29 datasets, ties on 1 (car, 1.000) and is below on 3: credit-g (−0.006), cmc (−0.002) and Australian (−0.001).

Regression: the median RMSE ratio is 0.983. 21 of 28 datasets are better than the paper and 25 of 28 are within 1.1×. The worst are socmob (1.22), auction_verification (1.21) and energy_efficiency (1.18); the best are SAT11-HAND-runtime-regression (0.87), geographical_origin_of_music (0.92) and cpu_activity (0.92).

Most of the regression gain came from stage 2, the first stage with long tables: the ratio fell from 1.136 to 0.997. Stage 3 changed little on this suite.

### TabArena-Lite

[TabArena-Lite](https://huggingface.co/spaces/TabArena/leaderboard) (51 datasets, one split each), scored the way the public leaderboard is: Elo against the leaderboard's method pool, with RandomForest (default) at 1000. The same weights serve the classification and regression splits, with the default inference. No split exceeds the size limits, so none is imputed. Every method on the public leaderboard is 8-fold bagged; this model is fit once per split.

| Method | Elo | Imputed splits |
| --- | --- | --- |
| TabPFN-3.5 (default), public | 1858 | |
| Mitra v2 (default), public | 1751 | |
| EXAONE-Tabular (default), public | 1741 | |
| **TabSil (this model)** | **1727** (+106/−78) | 0.0% |
| TabPFN-3 (default), public | 1619 | |
| TabICLv2 (default), public | 1557 | |
| RealTabPFN-2.5 (default), public | 1514 | |

This model places 11th of 99 methods. By problem type its Elo is 1699 (classification) and 2078 (regression). By stage, the Elo was 1429 after stage 1, 1725 after stage 2 and 1727 after stage 3; almost all of the stage-2 gain is regression (1187 → 2074). minipfn TabPFN-3.5 scored 1563 with the same protocol (measured in a separate run, so against a slightly different pool).

### BeyondArena

TBD.

## Limitations

- **Synthetic prior only.** The model never saw real data during training.
- **Large and wide tables.** Training stopped at 60,000 rows and 100 features. Wider tables rely on the 768-column cap per member.
- **Shortened curriculum.** The stages are shorter than the TabICLv2 scripts (150k / 10k / 2k instead of 250k / 20k / 5k steps). Stage 3 changed the scores by less than the confidence intervals.
- **Weakest regression datasets.** socmob, auction_verification and energy_efficiency are 1.18–1.22× the paper's RMSE.

## Usage

```bash
uv add "tabsil @ git+https://github.com/Aiden-Jeon/tabsil"
```

```python
from tabsil import TabSilClassifier, TabSilRegressor

# both download aiden-jeon/tabsil (one multitask model)
# x_train can be a pandas DataFrame with categorical columns
clf = TabSilClassifier(device="cuda").fit(x_train, y_train)
proba = clf.predict_proba(x_test)  # (n_test, n_classes), columns in clf.classes_ order

reg = TabSilRegressor(device="cuda").fit(x_train, y_train)
pred = reg.predict(x_test)  # mean of the predictive distribution
```

Each class above loads its own copy of the 222M weights. To use both tasks with one copy, load the model once and pass it in (the inference settings come with it):

```python
from tabsil import TabSilClassifier, TabSilRegressor, from_pretrained

model, borders = from_pretrained("aiden-jeon/tabsil")
clf = TabSilClassifier(model, device="cuda")
reg = TabSilRegressor(model, borders, device="cuda")
```

`fit` does not train weights. It fits each ensemble member's preprocessing on the training rows, which then serve as context at prediction time. To reproduce the benchmark (from a clone, `uv sync --all-packages`; downloads the OpenML tasks):

```bash
uv run --package tabsil-train python training/scripts/eval.py --suite all \
  --classifier-path hf://aiden-jeon/tabsil \
  --regressor-path hf://aiden-jeon/tabsil
```

`--suite tabarena-lite` and `--suite beyondarena` run the arena suites (they need the `arena` extra; see the repository README).

## Intermediate checkpoints

The weights at the end of stages 1 and 2 are kept as branches of this repo. Each branch holds the weights, config and a short README that points back here; this card on `main` describes all of them.

| Branch | Weights | Mean ROC AUC | RMSE ratio | TabArena-Lite Elo |
| --- | --- | --- | --- | --- |
| `stage1-150k` | `sha256:46ff015e71f8` | 0.9383 | 1.136 | 1429 |
| `stage2-10k` | `sha256:f7d460f62732` | 0.9390 | 0.997 | 1725 |
| `main` (stage 3, +2k) | `sha256:e03c8c32849a` | 0.9390 | 0.996 | 1727 |

```python
clf = TabSilClassifier("aiden-jeon/tabsil@stage2-10k")
```

In the benchmark script, write the branch after `@`: `--classifier-path hf://aiden-jeon/tabsil@stage2-10k`.

## Files

- `model.safetensors`: weights, plus the regression bucket borders (`borders`)
- `config.json`: model size (`ColumnRowTransformerConfig` fields), task (`multitask`) and inference settings

## Acknowledgements

- **TabPFN** (Prior Labs): the prior-fitted network approach, and the TabPFN-3 / 3.5 technical reports for the column → row → in-context backbone, the cell encoder and the retrieval decoder. No TabPFN weights are used.
- **TabICL / TabICLv2** (Inria SODA): the graph SCM prior (used through the `tabicl` package), the three-stage curriculum, and the QASSMax query gate of the row-stage temperature.
- **Kumo-Tabular** (NVIDIA): interleaved column/row stages, log-scale attention temperatures and the quantile regression head.
- **Muon** optimizer.

## License

Apache-2.0. The training data were generated with the TabICL prior (BSD-3-Clause). A few inference routines are adapted from Apache-2.0 TabPFN and NVIDIA code, and the query gate from BSD-3-Clause TabICLv2 code; see `NOTICE` in the repository. No TabPFN weights are included.

## Citation

```bibtex
@misc{tabsil2026,
  title        = {{TabSil}: a tabular foundation model},
  author       = {Jeon, Jongseob},
  year         = {2026},
  howpublished = {\url{https://github.com/Aiden-Jeon/tabsil}}
}

@article{priorlabs2026tabpfn35,
  title   = {{TabPFN-3.5}: Technical Report},
  author  = {{Prior Labs}},
  journal = {arXiv preprint arXiv:2609.17895},
  year    = {2026}
}

@inproceedings{qu2025tabicl,
  title     = {{TabICL}: A Tabular Foundation Model for In-Context Learning on Large Data},
  author    = {Qu, Jingang and Holzm{\"u}ller, David and Varoquaux, Ga{\"e}l and Le Morvan, Marine},
  booktitle = {International Conference on Machine Learning},
  year      = {2025}
}
```
