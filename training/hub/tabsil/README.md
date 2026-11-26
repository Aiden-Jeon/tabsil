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
- Inference (tabsil 0.1.1): an ensemble of 8 members. Each member shuffles features and classes and adds a row-identifier column: random for the training rows, a hash of the row's values for test rows (so a test row's prediction does not depend on the other test rows).
  - Numeric columns: the members cycle through four transforms, following Kumo-Tabular: the values as they are, Yeo-Johnson, robust scaling (median and interquartile range, squashed smoothly into ±3) and rank-Gauss.
  - Categorical columns: those with at most 20 categories are one-hot encoded, as long as the table stays within 100 columns. Those with more also get a column of log(1 + training-row count of the category).
  - Wide tables: tables with more than 768 columns give each member 768 of them, split so that every column is used about equally often (`balanced`).
  - Outputs: classification averages the members' softmax at temperature 1.0. In regression, half of the members also power-transform the target, and each numeric transform meets both target forms.
  - Settings: the member count, kinds, column cap, category counts and temperature ship in `config.json` (`inference`). The tag `v1.1` has these settings; `v1` has tabsil 0.1.0's: standard members only, no count columns, temperature 0.9.
  - Speed: `fit` runs the training rows through the model once and keeps what the test rows read from them, so `predict` runs only the test rows.

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
- **Shorter stages.** The recipe in `tabsil_train.recipe` keeps the per-task table count of the scripts (250k / 20k / 5k steps). These weights were trained for 150k / 10k / 2k steps (`--steps`). A separate 250k stage-1 run did not do better than the 150k run: TabArena-Lite 1423 at 225k vs 1429 at 150k (the repository's harness without bagging; see TabArena-Lite), and AMLB classification 0.9383 in both (measured before the tabsil 0.1.0 inference change, which moved the 150k Elo to 1431; with tabsil 0.1.1's inference it is 1444). In both runs the AMLB regression ratio rose during stage 1 (to 1.137 at 225k and 1.136 at 150k); stage 2 brings it back below 1.
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
| Stage 1 (150k steps) | 0.9380 | 1.138 |
| Stage 2 (+10k steps) | 0.9387 | 0.994 |
| **Stage 3 (+2k steps, this model)** | **0.9387** | **0.992** |
| minipfn TabPFN-3.5 | 0.9371 | 1.005 |
| TabPFN v2 paper | 0.9323 | 1 |

The TabSil rows use tabsil 0.1.1's inference. Classification: this model beats the paper on 25 of 29 datasets, ties on 1 (car, 1.000) and is below on 3: credit-g (−0.006), cmc (−0.003) and Australian (−0.001).

Regression: the median RMSE ratio is 0.982. 21 of 28 datasets are better than the paper and 25 of 28 are within 1.1×. The worst are auction_verification (1.22), energy_efficiency (1.18) and socmob (1.17); the best are SAT11-HAND-runtime-regression (0.87), cpu_activity (0.92) and geographical_origin_of_music (0.93).

Most of the regression gain came from stage 2, the first stage with long tables: the ratio fell from 1.138 to 0.994. Stage 3 changed little on this suite.

With tabsil 0.1.0's inference (standard members only, no category counts, temperature 0.9) the same weights scored 0.9390 / 0.996 (stage 3), 0.9390 / 0.997 (stage 2) and 0.9383 / 1.136 (stage 1). 0.1.1's inference is 0.0003 lower in mean ROC AUC on every stage and better in regression on stages 2 and 3.

### TabArena-Lite

[TabArena-Lite](https://huggingface.co/spaces/TabArena/leaderboard) (51 datasets, one split each) with TabArena's official pipeline: every method, this model included, is 8-fold bagged, and Elo is computed against the method pool with RandomForest (default) at 1000. This model ran as the `TabSil` model of TabArena ([autogluon/tabarena#753](https://github.com/autogluon/tabarena/pull/753)) at its default configuration (tabsil 0.1.0, the weights at the tag `v1`), on 8×H100 with one process per GPU. The same weights serve the classification and regression splits. No split is imputed (0.0%). This model places 18th of 121 methods. The table lists this model, the foundation models near it, the strongest tuned baselines and the RandomForest anchor; the [full table](https://huggingface.co/datasets/aiden-jeon/tabsil-tabarena-lite/blob/1c4beda3ed9683fa5bd15f4cc9569a8180b04ec4/leaderboard_website.md) has all 121.

| Method | Place | Elo (+CI) | Improvability (%) | Fit (s/1K) | Infer (s/1K) |
| --- | --- | --- | --- | --- | --- |
| Kumo-Tabular (default) | 1 | 2090 (+159/−110) | 3.7 | 2.33 | 0.254 |
| TabPFN-3.5 (default) | 7 | 1882 (+139/−91) | 7.5 | 1.85 | 0.473 |
| TabPFN-3.5-Fast (default) | 16 | 1806 (+91/−59) | 9.1 | 0.73 | 0.153 |
| Mitra-v2 (default) | 17 | 1774 (+101/−68) | 8.5 | 59.03 | 0.886 |
| **TabSil (this model)** | **18** | **1755** (+88/−73) | 9.0 | 2.53 | 0.812 |
| EXAONE-Tabular (default) | 20 | 1753 (+66/−52) | 9.9 | 1.98 | 0.582 |
| TabPFN-3 (default) | 25 | 1626 (+73/−48) | 11.6 | 1.29 | 0.383 |
| TabICLv2 (default) | 29 | 1567 (+61/−57) | 13.1 | 0.73 | 0.137 |
| RealTabPFN-2.5 (default) | 33 | 1519 (+59/−44) | 13.8 | 5.72 | 0.611 |
| RealMLP (tuned + ensembled) | 36 | 1489 (+45/−44) | 15.2 | 2791.97 | 13.886 |
| CatBoost (tuned) | 45 | 1394 (+46/−39) | 16.6 | 1347.32 | 0.037 |
| LightGBM (tuned + ensembled) | 46 | 1387 (+36/−37) | 17.6 | 416.63 | 2.236 |
| XGBoost (tuned + ensembled) | 55 | 1349 (+39/−42) | 18.3 | 700.96 | 1.438 |
| RandomForest (default) | 112 | 1000 (+74/−75) | 28.4 | 0.43 | 0.053 |

- The pool is the TabArena results available when the run was made (2026-11-25, TabArena at the submission commit `1360e0e8`): 120 methods plus this model, 121 methods. It is not a released leaderboard version, so the live leaderboard's numbers differ. These are the results submitted with the pull request; the maintainers' own run of the full TabArena may change them. The raw results (`results.pkl` of all 51 splits), the hardware and the logs are in [aiden-jeon/tabsil-tabarena-lite](https://huggingface.co/datasets/aiden-jeon/tabsil-tabarena-lite).
- All methods use the same protocol, so the Fit and Infer columns are like-for-like; the other methods' times were measured by TabArena on its own hardware.
- The Elo CI is 95%. Fit and infer are median seconds per 1K train and test rows. Lower is better for improvability, fit and infer.

By problem type, from the same results scored on each subset (TabArena's `compare` with `subset="classification"` or `"regression"`; methods without results on a subset drop out), this model's Elo is 1722 (+118/−86, 19th of 119) on the 38 classification datasets and 2123 (+216/−100, 18th of 116) on the 13 regression datasets.

The official-protocol results above are tabsil 0.1.0's. tabsil 0.1.1 changed the inference defaults (see Inference); its official-protocol run follows the release.

The numbers below come from the repository's own harness (`training/scripts/eval.py --suite tabarena-lite`), not the official protocol: each split is fit once without bagging, and Elo is computed against the public leaderboard v0.1.9.12 pool plus the checkpoint. It is a quicker check for comparing checkpoints and inference settings.

- **Inference settings:** with tabsil 0.1.1's inference this model scores 1768 (+94/−86, 9th of 99), against 1726 (11th of 99) with 0.1.0's and 1755 with 0.1.0 under the official protocol above. By problem type that is classification 1741 (0.1.0: 1698) and regression 2130 (2080). Per split, 31 of 51 errors are lower and 20 higher. Median times are 0.23 s per 1K training rows to fit and 0.16 s per 1K test rows to predict, against 0.014 and 0.81 with 0.1.0; 0.1.1's fit runs the training rows through the model, so its predictions skip them.
- **Stages (0.1.1's inference):** the Elo was 1444 after stage 1, 1762 after stage 2 and 1768 after stage 3. Almost all of the stage-2 gain is regression (1189 → 2122). With 0.1.0's inference the stages scored 1431, 1727 and 1726. minipfn TabPFN-3.5 scored 1563 with the same harness (measured in a separate run, so against a slightly different pool).

### BeyondArena

TBD.

## Limitations

- **Synthetic prior only.** The model never saw real data during training.
- **Large and wide tables.** Training stopped at 60,000 rows and 100 features. Wider tables rely on the 768-column cap per member.
- **Shortened curriculum.** The stages are shorter than the TabICLv2 scripts (150k / 10k / 2k instead of 250k / 20k / 5k steps). Stage 3 changed the scores by less than the confidence intervals.
- **Weakest regression datasets.** socmob, auction_verification and energy_efficiency are 1.18–1.22× the paper's RMSE.

## Usage

```bash
uv add tabsil   # or: pip install tabsil
```

```python
from tabsil import TabSilClassifier, TabSilRegressor

# both download aiden-jeon/tabsil at the tag v1.1 (one multitask model)
# x_train can be a pandas DataFrame with categorical columns
clf = TabSilClassifier(device="cuda").fit(x_train, y_train)
proba = clf.predict_proba(x_test)  # (n_test, n_classes), columns in clf.classes_ order

reg = TabSilRegressor(device="cuda").fit(x_train, y_train)
pred = reg.predict(x_test)  # mean of the predictive distribution
```

Each class above loads its own copy of the 222M weights. To use both tasks with one copy, load the model once and pass it in (the inference settings come with it):

```python
from tabsil import TabSilClassifier, TabSilRegressor, from_pretrained

model, borders = from_pretrained("aiden-jeon/tabsil@v1.1")
clf = TabSilClassifier(model, device="cuda")
reg = TabSilRegressor(model, borders, device="cuda")
```

`fit` does not train weights. It fits each ensemble member's preprocessing on the training rows and runs them through the model once; prediction reads that context and runs only the test rows. To reproduce the benchmark (from a clone, `uv sync --all-packages`; downloads the OpenML tasks):

```bash
uv run --package tabsil-train python training/scripts/eval.py --suite all \
  --classifier-path hf://aiden-jeon/tabsil@v1.1 \
  --regressor-path hf://aiden-jeon/tabsil@v1.1
```

`--suite tabarena-lite` and `--suite beyondarena` run the arena suites (they need the `arena` extra; see the repository README). They fit each split once without bagging, a quicker check; the official-protocol TabArena-Lite result above is reproduced with TabArena's `TabSil` model ([autogluon/tabarena#753](https://github.com/autogluon/tabarena/pull/753)).

## Intermediate checkpoints

The v1 weights on `main` are also tagged `v1`. tabsil 0.1.0's `TabSilClassifier()` and `TabSilRegressor()` download that tag; tabsil 0.1.1's download `v1.1`, the same weights with 0.1.1's inference settings in `config.json`. So both keep the same weights if `main` changes. The weights at the end of stages 1 and 2 are kept as branches of this repo. Each branch holds the weights, config and a short README that points back here; this card on `main` describes all of them.

| Branch | Weights | Mean ROC AUC | RMSE ratio | TabArena-Lite Elo (harness) |
| --- | --- | --- | --- | --- |
| `stage1-150k` | `sha256:46ff015e71f8` | 0.9380 | 1.138 | 1444 |
| `stage2-10k` | `sha256:f7d460f62732` | 0.9387 | 0.994 | 1762 |
| `main` (stage 3, +2k) | `sha256:e03c8c32849a` | 0.9387 | 0.992 | 1768 |

The scores use tabsil 0.1.1's inference (see TabArena-Lite and the TabPFN v2 benchmark for 0.1.0's). The Elo column is the repository's harness without bagging, for comparing the stages; with the official protocol and tabsil 0.1.0, `main` scores 1755.

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
