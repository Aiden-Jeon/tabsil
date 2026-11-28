# v2 task formulations

Status: written on 2026-11-28 for the re-plan that follows the stop of the v2 plan ([v2-retrospective.md](v2-retrospective.md), [v2-prior-task-plan.md](v2-prior-task-plan.md)). Nothing was trained or evaluated for it, and no GPU job was run. It derives, task by task, what the current model estimates when a task is put to it as a table, and what would have to change for the task to be done well. The designs it starts from are in [v2-task-priors.md](v2-task-priors.md). Numbers of the v1.1 weights come from that document and from [docs/task-results.md](../../docs/task-results.md). The closed forms were reproduced numerically with NumPy (no model weights), and each derivation was read once by an agent that did not write it ("Checks and corrections").

How to read:

- Every task has the same nine steps: problem, assumptions, estimand, identification, where training converges, table construction, closed-form check, required change and limits, measurement. Classification and regression come first and are the reference for the rest.
- **Measured** is a number from a run in the records. **Derived** follows from the stated assumptions by the steps shown. **Checked** means the closed form was reproduced numerically. **Estimated**, **unknown** and **not verified** are marked as such.
- A required change is one of four kinds. **Inference**: the wrapper or the inference recipe, no training. **Prior**: which tables are drawn (the generator, the split, the context, the scale). **Loss**. **Structure**: an input or an output that the architecture does not have.

## Summary

| Task | Estimand | What the v1 weights compute today | What the derivation says is missing | Kind of change | Basis |
|---|---|---|---|---|---|
| Classification, regression | the posterior predictive distribution of the target | the same, within the context's classes and between the 0.001 and 0.999 quantiles | nothing. The target scale of training and inference differ by a term of order $1/n_Q$ | none (a prior option removes the scale difference) | derived; checked; the scale term is measured at 0.3% of the loss |
| Forecasting | the predictive distribution of each future step | the extrapolation of a prior without trends, which returns toward the context mean; with the default recipe the median cannot pass the largest context value | queries beyond the context's range never occur in training, and the prior has no trend | inference first (a recipe without the clip; the trend fitted outside the model), then prior (temporal split, trend, train-row scale) | D1 measured; the cap and the return to the mean are derived and checked |
| Outlier detection | the probability that a row is an outlier | the density of the contaminated context | a dense group of outliers has a high density: the score ranks it as inliers once its share passes $\sigma_c^{d}$ | inference (small random contexts: untested) or prior (labeled reference rows with hidden contamination) | D3 measured (0.425); the exact density fails on the same generator too (0.42, checked) |
| Imputation | the distribution of a missing cell given the observed cells and the fact that it is missing | the distribution among rows that observe the cell | MCAR: nothing. MAR: the same conditional, asked where the context is thin. MNAR: a different conditional that the data do not identify | MCAR none; MAR none expected; MNAR prior, and it only moves the answer toward the prior's average mechanism | derived; checked; TabImpute's specialists agree (MAR 1.076 against 1.072, self-masking 2.184 against 1.716) |
| Treatment effects | the effect $\tau(x)$ and, if wanted, its uncertainty | the posterior mean of $\tau(x)$ under no hidden confounding | the point estimate: nothing. An interval for $\tau(x)$ does not follow from two predictive distributions. Hidden confounding needs the model to know which column is the treatment | none for the point estimate; structure for intervals and for hidden confounding | derived; checked; IHDP and ACIC measured |
| Masked-cell loss | every cell's distribution given any visible cells | not in the model | for classification and regression: no change at the optimum. For imputation and density: all conditionals in one pass, and rows with missing cells can serve as context | structure (a head on the cells) | derived; no controlled evidence in the literature |

Three results carry most of the document (section 1): training makes the network the conditional distribution of the hidden target under the distribution of training tables; a split that depends only on the features leaves that conditional's form unchanged and a split that depends on the target changes it; and outside the training tables' support the loss constrains nothing.

## 1. Setup

### Notation

| Symbol | Meaning |
|---|---|
| $\phi$, $\pi$ | one data-generating process (in the code, one graph SCM draw with its functions, noise and column types) and the prior over such processes |
| $(x_i, y_i)$ | a row: its features and its target. Rows are independent given $\phi$ unless a task says otherwise |
| $C$, $Q$ | the index sets of the context rows and of the query rows, with $n = \lvert C \rvert$, $n_Q = \lvert Q \rvert$ and $N = n + n_Q$ |
| $D$ | the context table, the rows $(x_i, y_i)$ with $i \in C$ |
| $(x_\star, y_\star)$ | one query row and its hidden target |
| $P$ | a task distribution: the joint distribution of $(D, x_\star, y_\star)$ that a prior, a split rule and a scaling rule produce |
| $p_P(y_\star \mid x_\star, D)$ | the conditional distribution of the hidden target under $P$ |
| $q_\theta(\cdot \mid x_\star, D)$ | the network's output |
| $\tau_k = k/1000$, $\hat q_k$ | the quantile levels and the regression outputs, $k = 1, \dots, 999$ |
| $\rho_\tau(u) = \max(\tau u, (\tau - 1) u)$ | the pinball loss at level $\tau$ |

### What the model is

Read from the code on `dev-v2` (77e4ecf).

- **M1, regression output.** 999 values $\hat q_k$ trained with the mean over $k$ of $\rho_{\tau_k}(y_\star - \hat q_k)$ (`tabsil.models.quantile_distribution.pinball_loss`). Inference sorts them and turns them into bucket probabilities on borders that span ±22.6 standard deviations of the context target. The mass below $\hat q_1$ and above $\hat q_{999}$, 0.001 each, is put into the buckets that hold those quantiles, so the output has no tail beyond them.
- **M2, classification output.** A retrieval decoder (`RetrievalDecoder`): the query attends to the context rows with weights $a_i^h$ in head $h$, and the class probability is the attention mass on the rows of that class,

$$
p_c = \frac{1}{H} \sum_{h=1}^{H} \sum_{i \in C} a_i^h \mathbf 1[y_i = c]
$$

  The logit is $\log(\max(p_c, 10^{-5}) + 3 \cdot 10^{-5})$ and the loss is the cross-entropy of its softmax. A class that no context row carries has $p_c = 0$ and ends at a probability of $4 \cdot 10^{-5}$ whatever the query is.
- **M3, scale.** Features are standardized with the context rows and clipped at ±100; a rank feature of each cell saturates at 0 or 1 outside the context's range (`standardize_cells`, `ecdf_midrank`). The v1 prior standardizes a regression target over all rows of the table after clipping it at 4 standard deviations. Inference standardizes it with the context rows. The prior option `target_train_rows` makes training do the same.
- **M4, rows.** A batch has one context size, and the query rows are the last rows. Query rows attend to context rows only, never to each other, and all statistics come from the context rows. A prediction therefore depends on the context and on its own query row, and the outputs are per row: a joint distribution over several queries is not available.
- **M5, context targets.** Every context row carries a target (it is embedded and added to the row). There is no value for "target unknown", so an unlabeled row cannot be a context row.
- **M6, roles.** No input tells the model what the task is or what role a column plays. A column that is constant on the context rows is dropped.
- **M7, loss mask.** A query row with a missing target contributes zero to the pinball loss, and the classification loss ignores the index −100.

### Proposition 1: what training converges to

Training minimizes the expected loss over tables drawn from a task distribution $P$ (Müller et al. 2022, arXiv 2112.10510, Insight 1; Nagler 2023, arXiv 2305.11097, Theorem 2.2).

For the cross-entropy,

$$
\mathbb E_P \left[ -\log q_\theta(y_\star \mid x_\star, D) \right] = \mathbb E_P \left[ \mathrm{KL}\left( p_P(\cdot \mid x_\star, D) \Vert q_\theta(\cdot \mid x_\star, D) \right) \right] + \mathbb E_P \left[ \mathrm H\left( p_P(\cdot \mid x_\star, D) \right) \right]
$$

The second term does not depend on $\theta$, so the minimum is at $q_\theta = p_P(\cdot \mid x_\star, D)$ for almost every $(x_\star, D)$ under $P$.

For the pinball loss at level $\tau$, write $F$ for the distribution function of $p_P(\cdot \mid x_\star, D)$. Then

$$
\frac{\partial}{\partial q} \mathbb E \left[ \rho_\tau(y_\star - q) \mid x_\star, D \right] = F(q) - \tau
$$

which is zero at $q = F^{-1}(\tau)$. So $\hat q_k$ converges to the $\tau_k$ quantile of the same conditional distribution. The mean over the 999 levels is a quadrature of the continuous ranked probability score,

$$
\frac{1}{999} \sum_{k=1}^{999} \rho_{\tau_k}\left( y - F^{-1}(\tau_k) \right) \approx \frac{1000}{999} \int_0^1 \rho_\tau\left( y - F^{-1}(\tau) \right) d\tau = \frac{1000}{999} \cdot \frac{\mathrm{CRPS}(F, y)}{2}
$$

so the regression loss is a proper score of the whole distribution between the 0.001 and 0.999 quantiles (checked for a normal $F$: the ratio of the two sides is 1.0010 at every $y$ tried).

**The estimand of training is therefore always the same object: the conditional distribution of what is hidden given what is shown, under the distribution of training tables.** A task is a choice of what is shown (the context, the features, the scale), what is hidden (the target) and which rows are queries (the split).

### Proposition 2: a split on the features and a split on the target

Let all $N$ rows be drawn first and a rule then assign each row to the context or to the queries.

**(a) The rule depends on the features only** (and on independent randomness). The assignment is then independent of the targets given the features, so given $\phi$ and the features the targets keep their distribution:

$$
p_P(y_\star \mid x_\star, D) = \int p(y_\star \mid x_\star, \phi) \pi_S(\phi \mid D, x_\star) d\phi
$$

The outcome terms $p(y_i \mid x_i, \phi)$ enter the posterior $\pi_S$ unweighted. The rule can add evidence about the distribution of the features only. In an SCM prior the features and the outcome mechanism share causes, so that evidence is not empty (Kobialka et al. 2026, arXiv 2610.14989), but the form of the estimand is that of the random split.

**(b) The rule depends on the target.** If each row with features $x$ and target $y$ becomes a query with probability $s(x, y)$, independently of the other rows, then

$$
p(y_\star \mid x_\star, \phi, \star \in Q) = \frac{s(x_\star, y_\star) p(y_\star \mid x_\star, \phi)}{\int s(x_\star, y) p(y \mid x_\star, \phi) dy}
$$

and the context rows are tilted by $1 - s$ in the same way. The conditional distribution itself changes. (A rule that draws a fixed number of query rows with probabilities proportional to a score, as a batch with one context size needs, tilts in the same direction; the formula is then approximate.)

This is the ignorability argument of missing-data theory (Rubin 1976) applied to the split. It sorts task-shaped priors into two kinds: those that keep the estimand and move the queries (a temporal split, missingness that depends on other columns, a randomized treatment in the queries), and those that change the estimand (missingness that depends on the missing value, censoring, an outlier label).

### Proposition 3: outside the training tables the loss constrains nothing

Let $P$ be the distribution of training tables and $P'$ the distribution of the tables of a task, with the same conditional $p(y_\star \mid x_\star, D)$. If $P'$ has a density $w$ with respect to $P$ on $(x_\star, D)$, then

$$
\mathbb E_{P'} \left[ \mathrm{KL}(p \Vert q_\theta) \right] = \mathbb E_P \left[ w \cdot \mathrm{KL}(p \Vert q_\theta) \right] \le \lVert w \rVert_\infty \mathbb E_P \left[ \mathrm{KL}(p \Vert q_\theta) \right]
$$

A small training loss bounds the task's error only through the largest density ratio. Where $P'$ puts mass that $P$ does not, there is no bound at all: what the network returns there is whatever its architecture extrapolates to. A task-shaped prior of the first kind is a way to make $\lVert w \rVert_\infty$ finite and small.

## 2. Classification and regression (the reference case)

**1. Problem.** Given a table $D$ of labeled rows and the features $x_\star$ of a new row, predict its target: class probabilities, or a predictive distribution with its mean and quantiles.

**2. Assumptions.**

- A1. The rows are independent draws from one process $\phi$.
- A2. $\phi$ is a draw from the prior $\pi$ (the graph SCM prior).
- A3. The context is a uniformly random subset of the rows.
- A4. The target is on the scale the model is shown: class indices, or a standardized value.

**3. Estimand.** The posterior predictive distribution

$$
p_\pi(y_\star \mid x_\star, D) = \int p(y_\star \mid x_\star, \phi) \pi(\phi \mid D) d\phi
$$

and its functionals: the class probabilities, the mean $\mathbb E[y_\star \mid x_\star, D]$, the quantiles.

**4. Identification.** Under the prior the estimand is a fixed function of $(x_\star, D)$. On a real table it equals the true conditional only as far as the posterior concentrates on the truth, which needs the truth to be in the prior's support and the context to be informative. Nagler (2023, Theorems 5.2 and 5.4, Section 6.3) shows that the variance of such a predictor vanishes with $n$, and that its bias vanishes only if the prediction comes to depend on nearby rows alone, which a transformer's attention does not guarantee.

**5. Where training converges.** By Proposition 1 with A3: the classification output converges to $p_\pi(y_\star = c \mid x_\star, D)$ and $\hat q_k$ to the $\tau_k$ quantile of $p_\pi(y_\star \mid x_\star, D)$. Two limits come from the outputs, not from the loss:

- M2 restricts the classifier to distributions supported on the context's classes. The prior makes this harmless: its tables are arranged so that every class of the query rows also appears in the context (`CoverClasses`).
- M1 fixes the levels at 0.001 to 0.999, so the distribution beyond them is not learned.

**6. Table construction.** The table is the table. The estimand of step 3 is reached if training and inference show the target on the same scale, and they do not quite. With the v1 prior the model is trained on $z = (y - m_{\mathrm{all}})/s_{\mathrm{all}}$, computed over all $N$ rows, and at inference it is given $(y - m_C)/s_C$ from the context rows. The all-row scale carries information about the query targets: if $\bar z_C$ is the mean of the context targets in all-row units, the means of the two groups differ by

$$
\bar y_Q - \bar y_C = -\frac{N}{n_Q} s_{\mathrm{all}} \bar z_C
$$

So a model trained this way can read the mean of the hidden targets off its context. Knowing a mean over $n_Q$ rows lowers the variance of each of them by a share $1/n_Q$, and the standard deviation $s_{\mathrm{all}}$ carries about as much again. The effect on the loss is of order $1/n_Q$: about 0.2 to 0.3% for stage-1 tables, where $n_Q$ is between 102 and 717. The 30K-step control runs of phase 1B measure 0.3% between the two scalings (reference loss 0.0633 against 0.0631; v2-experiment-results, "Phase 1B", "Reference validation loss"). Under A3 the mismatch is this small. Under a split that is not random it is not (section 3).

**7. Closed-form check: regression analysis.** Let the prior be the conjugate linear model: $y = x^\top \beta + \varepsilon$ with $\varepsilon \sim \mathcal N(0, \sigma^2)$, $\beta \mid \sigma^2 \sim \mathcal N(m_0, \sigma^2 \Lambda_0^{-1})$ and $\sigma^2 \sim \mathrm{IG}(a_0, b_0)$. With $X$ and $y$ the context,

$$
\Lambda_n = \Lambda_0 + X^\top X, \quad m_n = \Lambda_n^{-1}\left( \Lambda_0 m_0 + X^\top y \right), \quad a_n = a_0 + \frac n2, \quad b_n = b_0 + \frac12 \left( y^\top y + m_0^\top \Lambda_0 m_0 - m_n^\top \Lambda_n m_n \right)
$$

and the posterior predictive distribution is a Student t distribution with $2 a_n$ degrees of freedom,

$$
y_\star \mid x_\star, D \sim t_{2 a_n}\left( x_\star^\top m_n, \frac{b_n}{a_n} \left( 1 + x_\star^\top \Lambda_n^{-1} x_\star \right) \right)
$$

A model trained to convergence on tables from this prior would therefore return $\hat q_k = x_\star^\top m_n + s \cdot t_{2 a_n}^{-1}(\tau_k)$, with $s^2$ the scale above. Its mean is the ridge estimate, and in the flat-prior limit ($\Lambda_0 \to 0$, $a_0 \to -d/2$, $b_0 \to 0$) its mean is the least-squares fit and its central 95% interval is the textbook prediction interval $x_\star^\top \hat\beta \pm t_{n-d}^{-1}(0.975) \hat\sigma \sqrt{1 + x_\star^\top (X^\top X)^{-1} x_\star}$. In this sense the model does regression analysis: for a prior that is a regression model, the trained output is that model's analysis, estimate and interval together. (Checked: 400,000 posterior draws against the formula, quantiles at 0.01 to 0.99 within 0.007.)

For classification the corresponding case is a Bernoulli target without features and a $\mathrm{Beta}(a, b)$ prior: $p(y_\star = 1 \mid D) = (a + n_1)/(a + b + n)$, which the decoder can express as attention mass on the $n_1$ positive rows whenever both classes are in the context. If $n_1 = 0$ the answer is $a/(a + b + n)$ and the decoder returns $4 \cdot 10^{-5}$ (M2; a loss of 10.1 nats on such a row).

Diagnostic D2 measured the v1.1 weights against known distributions: an excess of 0.001 nats over the true density for random class labels, and 0.09 to 0.27 nats for targets independent of the features, with the two-mode target the worst (v2-task-priors, D2).

**8. Required change and limits.** None for the task. `target_train_rows` removes the scale difference of step 6. Its measured effect on real tables is a pooled error ratio of 0.995 (0.990 to 1.000) at 30K steps, inside the difference between two control seeds (v2-experiment-results, "Phase 1B").

**9. Measurement.** TabArena for accuracy; D2 for calibration. A scale effect of 0.3% is below what any design of this repository resolves (0.6% for a stage-2 branch).

## 3. Forecasting

**1. Problem.** A series $y_1, \dots, y_T$ with timestamps, and possibly covariates that are known for the future. Wanted for each step $h = 1, \dots, H$: the predictive distribution of $y_{T+h}$, reported as quantiles. fev-bench scores the scaled quantile loss (SQL) and MASE (arXiv 2509.26468, Eq. 1 and 2): the quantile loss over the levels 0.1 to 0.9, and the absolute error of the point forecast, each divided by the in-sample error of the seasonal naive forecast.

**2. Assumptions.**

- F1. The series is a regression on time features: $y_t = g(\varphi(t)) + \varepsilon_t$, where $\varphi(t)$ holds the running index, calendar sines and cosines, the seasonal periods found in the context, and the known covariates. This is the form of TabPFN-TS (Hoo et al. 2025, arXiv 2501.02945), which `TabSilForecaster` follows. No lag of the target is a feature.
- F2. $g$ is a draw from the prior. For the v1 weights that is the graph SCM prior, which has no notion of time.
- F3. The context is the first $T$ rows and the queries are the next $H$: a split on the running index.
- F4. The target is standardized with the context rows.

**3. Estimand.** For each $h$, the distribution $p(y_{T+h} \mid y_{1:T})$ given the features of all $T + H$ rows. These are the marginals. The joint distribution over the horizon is not available (M4), so sample paths and sums over the horizon are out of reach.

**4. Identification.** The values of $g$ on the horizon are not determined by its values on the context: any continuation is compatible with the data. What makes a forecast is the prior over continuations. So the question for the v1 weights is what continuation their training implies.

**5. Where training converges.** The v1 prior draws the split at random (A3). The tables of forecasting have every query beyond the context's range of one feature. Under a random split that configuration has probability

$$
\binom{N}{n_Q}^{-1}
$$

for a given feature, below $10^{-70}$ for D1's 365 and 56 rows. By Proposition 3 the training loss says nothing about such tables. The split depends on a feature only, so by Proposition 2(a) a prior that draws temporal splits would train the network toward $p(y_{T+h} \mid y_{1:T})$ under its own distribution over series. The pinball loss it minimizes is, up to a constant factor, the quantile loss in SQL's numerator, at 999 levels instead of nine.

**6. Table construction.** Rows are time steps, the features are $\varphi(t)$, the target is $y_t$, the context is the past. The output equals the estimand of step 3 under three conditions, and the v1 setup fails each of them.

- **Support.** The training tables must contain temporal splits (step 5).
- **Prior content.** The prior over $g$ must contain functions of the index that keep going. Under a prior of stationary functions the Bayes forecast returns to the context mean (step 7).
- **Scale.** With an all-row standardization and a temporal split, the context of a trending series is shifted: for a linear trend and D1's sizes the context targets have mean −0.23 and standard deviation 0.87 in all-row units (checked), which tells the model where the future lies. Inference always shows mean 0 and standard deviation 1. A temporal-split prior therefore needs `target_train_rows`. The eval targets then sit at

$$
z_{T+h} = \sqrt{12} \left( \frac12 + \frac hT \right)
$$

  for a linear trend, between 1.73 and 2.26 for D1, well inside the output's range. The size of the extrapolation is not the obstacle.

A fourth condition is on the wrapper, not on the model: the inference recipe must let a forecast leave the context's range.

**7. Closed-form check.**

*The default recipe caps the median.* Half of the default members transform the target and map their distribution back with an inverse that is clipped to the context's range (`_clipped_inverse`). All of their mass lies at or below the largest context value, up to one bucket. If $F_u$ is the distribution of the other half, the mixture has

$$
F(m) = \frac12 + \frac12 F_u(m) \ge \frac12 \quad \text{at the context maximum } m
$$

so its median, the point forecast, is at most $m$ plus one bucket, whatever the weights are. Above the median the mixture's quantile at level $u$ is the unclipped members' quantile at level $2u - 1$, so the upper quantiles are too low as well (checked). D1 measured exactly this: with the default recipe no forecast exceeds the context maximum by more than 0.14, and on the linear trend of strength 15 by 0.05, where the true maximum is 14.1 above it. This part of the gap is the recipe's and needs no training.

*A prior without trends returns to the mean.* Take a zero-mean Gaussian process with a stationary kernel $k$ as the stand-in for a prior without trends. On the standardized target its forecast is

$$
\hat z_{T+h} = k_\star^\top \left( K + \sigma^2 I \right)^{-1} z
$$

and $k_\star \to 0$ as the horizon moves away from the data by more than the kernel's length scale: the forecast returns to 0, the context mean. On D1's series (4 seeds, strength 15, the periodic part fitted first), this forecaster and the measured arms give:

| Forecaster on the linear trend of strength 15 | MASE | Trend capture |
|---|---|---|
| Least squares on the index and the periodic columns (the Bayes forecast under a linear-trend prior) | 0.37 | 0.99 |
| Flat at the mean of the last 28 context values | 5.10 | 0.00 |
| Flat at the context mean | 25.5 | −4.0 |
| Stationary kernel, length scale 0.3 context standard deviations of the index | 4.78 | 0.06 |
| Stationary kernel, length scale 0.1 | 19.9 | −2.9 |
| Seasonal naive | 3.67 | 0.28 |
| **TabSil v1.1, recipe without the clip (measured, D1 `plain`)** | **9.03** | **−0.71** |
| **TabSil v1.1, default recipe (measured, D1 `default`)** | **11.75** | **−1.31** |

Seasonal naive and least squares reproduce D1's columns (3.674 and 0.368), so the series are the same. The measured forecasts lie between the two stationary rows: they fall back toward the context mean by about a fifth (recipe without the clip) to a third (default) of the distance. Without a trend the same stand-in gives 0.53 and the model 0.51. The model's part of the gap is what a prior without trends does when asked beyond the data. It is not a defect of the fit.

*A trend prior is a regression plus a stationary remainder.* If $y_t = h(t)^\top \beta + r_t$ with a vague prior on $\beta$ and a stationary process for $r$, the Bayes forecast is (Rasmussen and Williams 2006, Section 2.7, Eq. 2.42)

$$
\hat y_{T+h} = h(T+h)^\top \bar\beta + k_\star^\top K_y^{-1} \left( y - H^\top \bar\beta \right), \quad \bar\beta = \left( H K_y^{-1} H^\top \right)^{-1} H K_y^{-1} y
$$

That is: estimate the trend by generalized least squares, forecast the remainder with the stationary forecaster, add the trend back. With ordinary least squares in place of the generalized fit this can be done in the wrapper, around the model as it is. On the stand-in it brings the linear trend of strength 15 from a MASE of 9.6 to 0.38. It does not solve trends that are not linear: 5.9 on the exponential and 7.8 on the piecewise trend with a line through the whole context, 1.8 and 6.1 with a line through the last 120 steps, against 4.4 and 4.6 for seasonal naive. These are numbers of the stand-in. They are estimates for the model, not measurements.

**8. Required change and limits.**

- **Inference, no training.** (a) A forecasting recipe whose members do not clip, as D1's `plain` arm: measured on D1, not on fev-bench. (b) The trend fitted outside the model as above: derived, not measured with the model.
- **Prior.** Temporal splits, functions of the index that continue (trend, level shift), and the context-row scale. A split that sorts the rows of an ordinary prior table by one feature is the smallest form: the graph SCM already has linear and quadratic mechanisms, and only the split and the scale change. The series generator of v2-task-priors §1 is the larger form.
- **Loss.** None: the pinball loss is the benchmark's loss.
- **Structure.** None for marginal forecasts of one series.
- **Limits that stay.** No joint distribution over the horizon (M4). No use of the residual's autocorrelation without lag features. One series per fit: other series help only if their rows are put into the context.

**9. Measurement.**

- D1 (v2_prior_diag): MASE and trend capture against seasonal naive and least squares. A change works on D1 if the linear trend of strength 15 falls below seasonal naive's 3.67.
- fev-bench, 10 tasks (docs/task-results): SQL Elo 1535 for the v1.1 weights. TabPFN-TS on TabPFN v2 is at 1584, TimesFM-2.5 at 1707, TabPFN-TS-3 at 1764 and Chronos-2 at 1842.
- What others gained by training on series: TabPFN-TS-3 is a TabPFN-3 checkpoint "fine-tuned on synthetic time-series data" and has an SQL skill of 43.1% on fev-bench against 39.6% for TabPFN-TS on TabPFN v2 (TabPFN-3 report, arXiv 2605.13986, Table 1). The base model changed too, so the gain of the series training alone is not known. TabICLv2 with the same kind of wrapper is at 30.8% in that table: the tabular prior alone moves forecasting by as much.
- Predictions that would show this section wrong: a model trained with `target_train_rows` and nothing else (the YCTX runs of phase 1B) extrapolates D1's trend better than its control; or the trend fitted outside the model leaves the linear trend's MASE far above 0.5.

## 4. Outlier detection

**1. Problem.** A table of unlabeled rows, some of which may be outliers. Wanted: a score per row that ranks outliers above inliers. ADBench scores the AUROC.

**2. Assumptions.**

- O1. Rows are draws from the mixture $p_c = (1 - \varepsilon) p_0 + \varepsilon p_1$ of an inlier distribution $p_0$ and an outlier distribution $p_1$ (Huber's contamination model).
- O2. The rows to score come from the same mixture as the context.
- O3. $p_0$, $p_1$ and $\varepsilon$ are draws from a prior.

**3. Estimand.** The posterior probability that a row is an outlier,

$$
\eta(x) = \frac{\varepsilon p_1(x)}{(1 - \varepsilon) p_0(x) + \varepsilon p_1(x)}
$$

or any increasing function of the likelihood ratio $\Lambda(x) = p_1(x)/p_0(x)$: a ranking by $\Lambda$ has the largest AUROC (Neyman and Pearson).

**4. Identification.** The mixture $p_c$ is identified by the data. Its split into $\varepsilon$, $p_0$ and $p_1$ is not: what counts as an outlier is an assumption. The usual one is that outliers are diffuse. Ranking by low density is the optimal ranking if and only if $\Lambda$ is a decreasing function of $p_c$. If $p_1$ is constant, $c$, on the region of the data, then $p_c = (1 - \varepsilon) p_0 + \varepsilon c$ is increasing in $p_0$ and $\Lambda = c/p_0$ is decreasing in it, so the condition holds: **the density score is the right score for diffuse outliers** (checked: for uniform outliers the density score and the likelihood ratio have the same AUROC, 0.9992). If outliers concentrate, $p_1$ is large where $p_c$ is large, and the two orders part.

**5. Where training converges.**

*The wrapper of today.* `TabSilUnsupervised` scores a row by $-\log \hat p(x)$ with the chain rule over the columns, $\log \hat p(x) = \sum_j \log \hat p(x_{\pi_j} \mid x_{\pi_{\lt j}})$, each factor a regression or classification fit on the context. By section 2 each factor estimates a conditional of the context's distribution. The score therefore estimates $-\log p_c(x)$, the density of the contaminated context.

*Reference rows.* Let the context hold the data rows with label 0 and reference rows with label 1 drawn from a known density $g$, in shares $1 - \rho$ and $\rho$. A classifier then converges to

$$
P(\text{reference} \mid x) = \frac{\rho g(x)}{\rho g(x) + (1 - \rho) p_c(x)}
$$

an increasing function of $g(x)/p_c(x)$ (Hastie, Tibshirani and Friedman 2009, Section 14.2.4; the formula is Bayes' rule and is derived here, the book's equation number was not re-read). With the v1 weights this is density estimation by another route: for a uniform $g$ it ranks rows as the density score does.

*Reference rows with hidden outliers.* Now let the training tables hide outliers among the data rows, keep the label 0 for all of them, and give the query rows their true label. By Proposition 1 the classifier converges to

$$
P_\pi(o_\star = 1 \mid x_\star, D)
$$

the posterior probability that the query is an outlier given a contaminated, unlabeled context, under the prior over outlier kinds. This is the estimand of step 3, with the identification of step 4 supplied by the prior. The decoder can express it, since both labels are in the context (M2): the output is the attention mass on the reference rows.

**6. Table construction.** For the trained form: the context is the data rows (label 0) followed by reference rows (label 1), and the queries are held-out data rows. The output equals $\eta$ to the extent that the prior's outlier kinds are those of the tables it is used on. That cannot be read from an unlabeled table (step 4): the gain on a benchmark is the overlap between the prior's notion of an outlier and the benchmark's labels.

A one-class form, with every context row labeled 0 and no reference rows, is impossible with this decoder: the probability of the absent class is the constant $4 \cdot 10^{-5}$ for every query (M2).

**7. Closed-form check.**

*A dense group passes for inliers.* Let the inliers be $\mathcal N(0, I_d)$ and the outliers a group $\mathcal N(\mu, \sigma_c^2 I_d)$ far from them. A typical inlier has $\log p_c \approx \log(1 - \varepsilon) - \frac d2 \log 2\pi - \frac12 \chi^2_d$, and a typical group member $\log \varepsilon - \frac d2 \log 2\pi - d \log \sigma_c - \frac12 \chi^2_d$. The density score ranks the group as more normal than the inliers when

$$
\log \frac{\varepsilon}{1 - \varepsilon} - d \log \sigma_c \gt 0, \quad \text{that is} \quad \varepsilon \gt \frac{\sigma_c^{d}}{1 + \sigma_c^{d}}
$$

For D3's group, $\sigma_c = 0.15$ and $d = 6$, the threshold is a share of $1.1 \cdot 10^{-5}$. Checked by simulation with the exact density:

| Share $\varepsilon$ of the group | 0 | $10^{-6}$ | $10^{-5}$ | $10^{-4}$ | 0.02 |
|---|---|---|---|---|---|
| AUROC of the density score $-\log p_c$ | 0.9995 | 0.857 | 0.527 | 0.168 | 0.003 |
| AUROC of the likelihood ratio | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

*D3 itself.* The same computation on D3's generator (the inlier manifold's density by Monte Carlo over its two factors, the group's density exact; D3's three seeds for the manifold and the group, fresh test rows):

| Share of the group in the context | 0 | 2% | 20% |
|---|---|---|---|
| Exact contaminated density, mean AUROC (by seed) | 1.000 | **0.42** (0.29, 0.07, 0.91) | 0.22 (0.04, 0.01, 0.60) |
| v1.1 weights, measured in D3 (by seed) | 0.999 | **0.425** (0.49, 0.26, 0.53) | 0.453 (0.49, 0.36, 0.51) |

The exact density fails as the model does. The seeds do not agree one by one: the model's values stay nearer to 0.5, as a density estimated from 12 rows is flatter than the exact one, and the equal means at 2% are in part chance. What the table shows is that the failure belongs to the estimand. A better density estimate, from more prior variety or more training, moves toward the first row, which is no better.

*What does not help.* Dropping the highest-scoring rows from the context and scoring again removes rows of low density. The group has a high density, so it stays (derived; not run).

**8. Required change and limits.**

- **Inference, no training.** The score must stop seeing the group as dense. One way follows from the threshold: a context of $m$ random rows holds about $\varepsilon m$ group members, and an estimate from one or two points cannot be as sharp as the group. At $\varepsilon = 0.02$ a context of 64 rows has no group member with probability 0.27 and at most one with 0.63 (checked). Scoring with many small random contexts and averaging is what IsolationForest does with its subsamples of 256. Whether the model's density from 64 rows is still good enough on the inliers is not known. **Unknown gain; not in the literature for this kind of model as far as was found.**
- **Inference, of small expected effect.** The output has no mass beyond its extreme quantiles (M1), so a cell there gets a floor value. At most 0.2% of a calibrated model's cells are affected per conditional. Unknown effect on the AUROC.
- **Prior.** The reference-row task with hidden outliers of step 5: a prior change, with the wrapper that appends the same reference rows at inference. No change of the loss or of the structure.
- **Structure.** Not needed by the derivation. The models built for this task end in a two-way output on the query, trained with the cross-entropy against the hidden label (TACTIC, arXiv 2603.14171, Section 3; FoMo-0D, arXiv 2409.05672, and ICLAD, arXiv 2603.19497, are reported to do the like, read by a subagent). This model would replace that output by the attention mass on reference rows. Whether a retrieval decoder learns that as well is not known.

**9. Measurement.**

- D3: the group at 2% (0.425 today; 0.9 is the design's criterion) and the curve at 10% (0.064).
- ADBench, 12 tables (docs/task-results): mean AUROC 0.663 against 0.722 for IsolationForest and 0.704 for the TabPFN v2 extension; 0.820 with a clean context, which uses labels and is the upper reference.
- What others gained by training on contaminated contexts: on 38 ADBench tables TACTIC reports a mean AUROC of 80.6 with a contaminated context, against 67.6 for the TabPFN-2.5 unsupervised extension (the method of today's wrapper) and 77.7 for IsolationForest; with a clean context 90.4, 88.9 and 82.3 (TACTIC, Tables 5 and 7). That is a model built for the task alone, on other tables than the 12 here.
- Predictions that would show this section wrong: context refinement raises D3's group at 2%; or small random contexts do not.

## 5. Imputation

**1. Problem.** A table with missing cells. Wanted for each missing cell: its distribution given the row's observed cells, a point value (the mean for a numeric column, the mode for a categorical one), or draws.

**2. Assumptions.**

- I1. Rows are independent draws of a full row $x$ and a mask $M$.
- I2. The mechanism is one of: MCAR, $P(M \mid x) = P(M)$; MAR, $P(M \mid x) = P(M \mid x_{\mathrm{obs}})$; MNAR, anything else (Rubin 1976).
- I3. The wrapper works column by column: for column $j$ the context is the rows that observe it, the features are the other columns (missing ones stay missing), and the queries are the rows that miss it (`TabSilUnsupervised.impute`).

**3. Estimand.** For a row that misses column $j$: $p(x_j \mid x_{-j}, M_j = 1)$, where $x_{-j}$ stands for the row's other observed cells.

**4. Identification.** The context estimates the distribution among the rows that observe the column, $p(x_j \mid x_{-j}, M_j = 0)$. By Bayes' rule

$$
p(x_j \mid x_{-j}, M_j = 1) = \frac{P(M_j = 1 \mid x_j, x_{-j})}{P(M_j = 1 \mid x_{-j})} p(x_j \mid x_{-j})
$$

- MCAR and MAR: the ratio is 1. Both conditionals equal $p(x_j \mid x_{-j})$: the estimand is identified and is what the context estimates. What the step needs is that, given the cells of the row that are shown, whether $x_j$ is missing does not depend on $x_j$. With several incomplete columns this is a condition per column and is not implied by Rubin's MAR for the whole mask.
- MNAR: the ratio depends on the missing value. The observed data are compatible with many mechanisms, and the estimand is not identified without an assumption on the mechanism.

**5. Where training converges.** For one column the wrapper's table is a regression or classification table, and its split is the mask.

- MCAR is a random split: the v1 prior is already an MCAR imputation prior for one column, and the output converges to the estimand (section 2).
- MAR is a split on the features (Proposition 2(a)): the same conditional, but the queries sit where the mechanism sends them. With missingness that rises with a column $x_1$, the rows to impute have large $x_1$ and the context has few such rows. This is Proposition 3 with a finite density ratio: the estimand is right, the accuracy is asked where the context is thin.
- MNAR is a split on the target (Proposition 2(b)). A prior that draws its query rows by a mechanism trains the network toward $p_\pi(x_j \mid x_{-j}, M_j = 1, D)$, which averages over the prior's mechanisms given a context of complete cases. The rows that miss the column are not in the context (M5) and the queries do not see each other (M4), so the one piece of evidence on the mechanism that the table holds, the distribution of the incomplete rows, never reaches the model.

**6. Table construction.** As I3. For an MNAR prior the scale matters again: under self-masking the context's mean is shifted, and inference standardizes with the context. The prior must do the same (`target_train_rows`), or the tilt it teaches is on another scale than the one it is asked on.

**7. Closed-form check.**

*Self-masking.* Let $x \sim \mathcal N(0, 1)$ be missing with probability $\Phi(a + b x)$. With $\alpha = a/\sqrt{1 + b^2}$ and $\kappa = b/\sqrt{1 + b^2}$,

$$
\mathbb E[x \mid M = 1] = \kappa \frac{\varphi(\alpha)}{\Phi(\alpha)}, \quad \mathbb E[x \mid M = 0] = -\kappa \frac{\varphi(\alpha)}{1 - \Phi(\alpha)}
$$

so the mean of the complete cases misses the mean of the missing values by

$$
\mathbb E[x \mid M = 0] - \mathbb E[x \mid M = 1] = -\kappa \frac{\varphi(\alpha)}{\Phi(\alpha) (1 - \Phi(\alpha))}
$$

At a missing rate of 30% and $b = 1$ this is −1.17 standard deviations, and the error of imputing the complete-case mean is 1.42 where the best possible is 0.80 (checked by simulation to three digits, also at $b = 0.5$ and $b = 2$).

*With a predictive column.* If another column $x_1$ has correlation $\rho$ with $x$ and $s^2 = 1 - \rho^2$ is the variance it leaves, the bias of the complete-case regression at a given $x_1$ is the same expression with $b s$ in place of $b$, times $s$:

$$
-s \frac{b s}{\sqrt{1 + b^2 s^2}} \frac{\varphi(\alpha_1)}{\Phi(\alpha_1) (1 - \Phi(\alpha_1))}, \quad \alpha_1 = \frac{a + b \rho x_1}{\sqrt{1 + b^2 s^2}}
$$

It is −1.17 at $\rho = 0$, −0.94 at 0.5, −0.30 at 0.9 and −0.03 at 0.99 (checked at $x_1 = 0$). **The MNAR error scales with what the other columns leave unexplained.**

*MAR.* With missingness $\Phi(-0.74 + 1.5 x_1)$ and $\rho = 0.6$, the regression on the complete cases has slope 0.600 and intercept 0.001, the truth. The rows to impute have a mean $x_1$ of 0.90 against −0.46 in the context, and 32% of them lie beyond the context's 99th percentile of $x_1$ (checked).

*Against the literature.* TabImpute trains one model per pattern (arXiv 2510.02625, Table 2, normalized RMSE):

| Evaluated on | Trained on MCAR (its default) | Trained on that pattern | Trained on self-masking |
|---|---|---|---|
| MCAR | 0.819 | 0.819 | 1.135 |
| Column MAR | 1.076 | 1.072 | 1.770 |
| Self-masking MNAR | 2.184 | 1.716 | 1.716 |

The MAR specialist gains 0.4% on MAR, as "the same conditional" predicts. The self-masking specialist gains 21% on its own pattern and loses 39% on MCAR: it has learned the tilt of its prior and applies it where there is none. Its paper states that a mix of patterns "degraded overall because it focused only on Self-MNAR".

**8. Required change and limits.**

- MCAR: none. The v1.1 weights are ahead of every baseline of the suite (NRMSE 0.546 at 10% against 0.616 for MissForest).
- MAR: none expected. A split that follows the mechanism would train the thin region (prior). TabImpute's 0.4% is the only number.
- MNAR: prior, with a limit that a prior cannot remove. The model can learn the average tilt of the mechanisms it is trained on. On a table with another mechanism, or none, that tilt is an error of the size in the table above. Whether real missing values follow a given mechanism cannot be measured: their true values are unknown. A benchmark that masks cells by a mechanism measures the match between that mechanism and the prior's.
- Using the incomplete rows as evidence, or as context for other columns, needs a context row without a target: structure (M5), or the masked-cell output of section 7.
- Draws that respect the dependence between several missing cells of a row need the cells imputed one after the other, each conditioned on the ones before: inference only, at one fit per cell pattern.

**9. Measurement.** D4 (not run): the same mask rate under MCAR, MAR and self-masking, on prior tables with a known mechanism and on the imputation suite, against the suite's baselines. Predictions: the MAR error is the MCAR error with a ratio like the baselines'; the self-masking error follows the closed form and shrinks with the columns' predictability.

## 6. Treatment effects

**1. Problem.** Rows $(x, t, y)$ with a binary treatment $t$ and an outcome $y$. Wanted: the conditional average treatment effect $\tau(x) = \mathbb E[Y(1) - Y(0) \mid x]$, scored by the root of the PEHE, and the average effect.

**2. Assumptions.**

- T1. Consistency: the observed outcome is $Y(t)$ for the treatment received.
- T2. No hidden confounding: $(Y(0), Y(1))$ is independent of $t$ given $x$.
- T3. Overlap: $0 \lt e(x) \lt 1$ for the propensity $e(x) = P(t = 1 \mid x)$.
- T4. The outcome surfaces $\mu_t(x) = \mathbb E[y \mid x, t]$ are draws from the regression prior, with the treatment as one more feature (the S-learner of `TabSilCATE`).

**3. Estimand.** $\tau(x) = \mu_1(x) - \mu_0(x)$; for uncertainty, its posterior distribution given the data.

**4. Identification.** Under T1 to T3, $\mathbb E[Y(t) \mid x] = \mathbb E[y \mid x, t]$: the effect is a difference of two regression functions. Without T2 it is not identified by the data, and an answer exists only through a prior over hidden structure (the approach of Do-PFN, arXiv 2506.06039, whose prior does not impose T2; read by a subagent, not re-checked).

**5. Where training converges.** With T4 the S-learner's table is an ordinary regression table, and the model's mean converges to $\mathbb E_\pi[\mu(x, t) \mid D]$ (section 2). The mean is linear, so

$$
\mathbb E_\pi[\tau(x) \mid D] = \mathbb E_\pi[\mu(x, 1) \mid D] - \mathbb E_\pi[\mu(x, 0) \mid D]
$$

**The difference of the two predicted means is the posterior mean of the effect.** No training on effects is needed for the point estimate.

The uncertainty is another matter. The posterior variance of the effect is

$$
\mathrm{Var}(\tau(x) \mid D) = \mathrm{Var}(\mu_1(x) \mid D) + \mathrm{Var}(\mu_0(x) \mid D) - 2 \mathrm{Cov}(\mu_1(x), \mu_0(x) \mid D)
$$

and the model returns two predictive distributions of $y$, each with variance $\mathrm{Var}(\mu_t(x) \mid D) + \sigma^2$, as separate outputs (M4). Neither the outcome noise $\sigma^2$ can be taken out nor the covariance read off. An interval for the effect is not available from the outputs.

**6. Table construction.** The context is the observed rows with the features $(x, t)$ and the target $y$. The queries are $(x, 1)$ and $(x, 0)$. The split is on the features only (Proposition 2(a)), but the counterfactual query $(x, 1 - t)$ of a row lies where its treatment is rare. The density of such queries relative to the context is

$$
\frac{p(x \mid t = 0)}{p(x \mid t = 1)} = \frac{1 - e(x)}{e(x)} \cdot \frac{P(t = 1)}{P(t = 0)}
$$

for treated-arm queries at control rows, which is unbounded as $e(x) \to 0$. Low overlap is the case of Proposition 3 in which the bound is lost: the prediction there is the prior's extrapolation.

Two things cannot be put into the table. The model is not told which column is the treatment (M6), so a prior with hidden confounding cannot teach it to adjust "the treatment column". And a query cannot ask for the noiseless $\mu_t(x)$ in place of $y$: the same table would have to mean two tasks.

**7. Closed-form check.**

*Linear outcome.* For $y = \alpha + \beta^\top x + \tau t + \varepsilon$ fitted by least squares, the Frisch–Waugh–Lovell theorem gives

$$
\mathrm{Var}(\hat\tau) = \frac{\sigma^2}{\sum_i (t_i - \hat t_i)^2}
$$

with $\hat t_i$ the linear projection of the treatment on the features. The sum is about $n \mathbb E[e(x)(1 - e(x))]$ or more, so the error grows as overlap shrinks, like $(\mathbb E[e(1 - e)])^{-1/2}$. Checked with $n = 2{,}000$: as the propensity's clip goes from 0.5 to 0.02, the standard deviation of $\hat\tau$ rises from 0.041 to 0.062 (the formula gives 0.045 and 0.063), and a T-learner of two linear fits rises from 0.060 to 0.088. Both grow at the same rate when both are correctly specified. D5's criterion, that TabSil's error grows faster than a T-learner's, therefore tests the prior's extrapolation and not overlap as such.

*Interval.* In a Bayesian linear model with an interaction, $n = 200$ and $\sigma = 1$: the posterior standard deviation of $\tau(x)$ is 0.19 at $x = 0$ and 0.34 at $x = 1.5$. Two predictive distributions treated as independent give 1.43 and 1.45 (checked). The outputs overstate the effect's uncertainty by the outcome noise.

**8. Required change and limits.**

- The point estimate under T2: none. The S-learner's root PEHE on IHDP is 0.55 out of sample, where CausalPFN, a model trained for the task, reports 0.58 and the EconML T-learner has 2.08 (docs/task-results; protocols differ).
- Low overlap: prior. Query rows whose treatment is drawn independently of the features put the S-learner's queries into the training tables with a bounded ratio (v2-task-priors §4). Unknown gain; D5 is not run.
- An interval for the effect: structure. A model that returns the posterior of $\mu_t(x)$ must be trained on noiseless targets (CausalPFN, arXiv 2506.07918, trains on the conditional expected potential outcome under T2). Without an input that says which task is asked (M6), one set of weights cannot serve both that and ordinary regression. Sampling schemes on top of the outputs (the martingale posterior, arXiv 2505.11325, not re-read) are inference only and cost many passes.
- Hidden confounding: structure (a marked treatment column) and a prior. Do-PFN marks the treatment by its position.

**9. Measurement.** IHDP and ACIC 2016 for the point estimate. D5 (not run): the root PEHE on prior tables as the propensity's clip shrinks from 0.5 to 0.02, against the true effect and a T-learner.

## 7. The masked-cell reconstruction loss

The idea on record (2026-11-26, undecided): hide random cells of the training tables and train an extra output, on the cell states, to reconstruct them. This section derives what such a loss estimates.

**1. Setup.** Let $S$ be the set of hidden cells of a table and let the head return, for a hidden cell $(i, j)$, a distribution of its value: 999 quantiles for a numeric column, class probabilities for a categorical one, with the losses of M1 and M2.

**2. What it converges to.** By Proposition 1, for each hidden cell

$$
q_\theta(\cdot) \to p_\pi\left( x_{ij} \mid x_{i, \mathrm{vis}}, X_{\mathrm{vis}}, S \right)
$$

the posterior predictive distribution of the cell given the visible cells of its own row and of all other rows. The mask is drawn independently of the values, so it is ignorable (Proposition 2(a)): this is the MCAR conditional of section 5, for every column and every set of visible columns at once.

**3. What it gives each task.**

- **Imputation.** The estimand of section 5 under MCAR, in one pass for all missing cells where the wrapper needs one fit per column. Rows with missing cells are themselves context for other cells, which M5 forbids today. The estimand does not change, so no gain in accuracy follows from the derivation except through that extra context.
- **Density.** The chain rule needs $p(x_{\pi_j} \mid x_{\pi_{\lt j}})$ for every prefix of a column order. Those are masked-cell conditionals with between none and all but one of the other columns visible. If the mask rate is drawn over the whole range from 0 to 1, the masked loss is the order-agnostic autoregressive objective, a lower bound on the log-likelihood that is tight when the conditionals are consistent (Uria et al. 2014; Hoogeboom et al. 2022, arXiv 2110.02037; neither re-read here). For a Gaussian table the sum of the conditionals along any order equals the joint log density exactly (checked). A mask rate that stays well below 1 never trains the conditionals with few visible columns, which are the ones the chain rule starts with: a loss meant for imputation at ordinary missing rates does not by itself serve density.
- **Outlier detection.** The density it yields is again the density of the contaminated context. Section 4's result is unchanged.
- **Classification and regression.** The main loss's minimizer is the posterior predictive distribution of the target with or without the extra loss. At the optimum the auxiliary loss changes nothing. Any effect is one of finite training, and its sign is not derivable. No controlled evidence exists: LimiX and LimiX-2 train with a masked objective and report no ablation of it (v2-experiment-plan §1, row 9).

**4. Required change.** Structure: an output on the cell states, used in training only if it serves as an auxiliary loss, kept if it serves imputation and density.

## 8. New task candidates

A candidate is listed with what it is, whether it can be put as classification or regression tables, a public benchmark, and earlier work with prior-fitted networks. The last column is the kind of change by the short derivations below. Sources marked † were read for this document; the others were read by a subagent and are not verified.

| Candidate | As classification or regression tables | Benchmark | Earlier work | Kind of change |
|---|---|---|---|---|
| Survival analysis with right censoring | yes: one binary row per subject and time bin | SurvSet collection; C-index, integrated Brier score | Off-the-shelf tabular models through classification reductions, 60 datasets (arXiv 2601.22259†, abstract). Models trained on a survival prior: arXiv 2603.29475, 2605.15488, 2606.04564 | inference; a trained censoring prior needs structure |
| Multi-output and multi-label prediction | yes: one table per output, earlier outputs as features | none standard found | wrappers only | inference |
| Ranking | yes: regression on the relevance | MSLR-WEB30K, Yahoo LTRC, Istella; NDCG | a repository and a 2026 conference paper, not read | inference; the context size is the limit |
| Density estimation and synthetic rows | yes: the chain rule of section 4, sampled | no agreed benchmark; TabPFGen (arXiv 2406.05216) scores downstream AUC | TabPFGen; the TabPFN unsupervised extension | inference; no tail beyond the extreme quantiles (M1) |
| Prediction under drift over time | yes: time as a feature and a temporal split | TabReD (arXiv 2406.19380) | Drift-Resilient TabPFN (arXiv 2411.10634): better out of distribution, "lower predictive performance on ID tasks" | prior: the split of section 3, plus drift |
| Time-series anomaly detection | yes: the forecast's distribution as the score | TSB-AD; older benchmarks are criticized (arXiv 2009.13807) | none found | inference, on top of forecasting; inherits sections 3 and 4 |
| Clustering | no | the 44 datasets of TabClustPFN | TabClustPFN (arXiv 2601.21656†, abstract): a loss that is invariant to label permutation and a network for the number of clusters | loss and structure |
| Using unlabeled rows | no | none specific | SemiTabICL (arXiv 2610.14989†, abstract): a model pretrained to use them beats a matched supervised one | structure (M5) and prior |
| Intervals for effects | not from two predictive distributions (section 6) | coverage on IHDP, ACIC | CausalPFN; the martingale posterior | structure, or many passes at inference |
| Prediction intervals with guaranteed coverage | yes: conformal calibration of the quantile outputs | any regression benchmark | conformal methods on tabular models (arXiv 2609.34887) | inference |
| Bayesian optimization | yes: an acquisition function from the predictive distribution | HPO-B | PFNs4BO (arXiv 2305.17535) | inference |

**Survival.** Time is cut into bins $k = 1, \dots, K$. A subject has an event bin $T$ and a censoring bin $C$; observed are $t = \min(T, C)$ and $\delta = \mathbf 1[T \le C]$. The hazard is $h_k(x) = P(T = k \mid T \ge k, x)$ and the survival function $S(k \mid x) = \prod_{j \le k} (1 - h_j(x))$. If censoring is independent of the event time given $x$, the likelihood of a subject is

$$
\prod_{k=1}^{t} h_k(x)^{\mathbf 1[k = t, \delta = 1]} \left( 1 - h_k(x) \right)^{1 - \mathbf 1[k = t, \delta = 1]}
$$

which is the likelihood of $t$ independent binary rows with features $(x, k)$ and the label "the event happened in this bin". The estimand, $h_k(x)$, is the class probability of an ordinary binary table, and the censoring is handled by which rows exist. The rows of a subject are dependent as a set, but the likelihood is this product, so the posterior over hazards is the one of an exchangeable table. Checked: on simulated data the row frequencies recover the hazards to 0.002 and their running product is the Kaplan–Meier curve; using the uncensored subjects alone is biased (a survival of 0.37 where the truth is 0.69 at the fourth bin). The cost is a context of $\sum_i t_i$ rows. A prior that shows censored targets in the context and true event times in the queries would train the direct answer, but the model would have to know which column is the event indicator (M6).

**Multi-output.** $p(y_1, \dots, y_m \mid x) = \prod_k p(y_k \mid x, y_{\lt k})$. Each factor is an ordinary table with the earlier outputs as features, known in the context and filled with draws or predictions for the queries. Exact by the chain rule, at $m$ fits; errors of early outputs propagate.

**Ranking.** For one query group, the expected discounted gain of an order is a sum over positions of a decreasing weight times $\mathbb E[\mathrm{gain}(y) \mid x]$ of the item placed there. It is largest when the items are sorted by that expectation (rearrangement inequality). The regression mean is the estimand, and the table is the pairs of query and item. Benchmarks have millions of rows, far beyond a context.

**Density and synthetic rows.** For the exact posterior predictive distribution of a new row the chain rule holds along every column order, so drawing the columns in turn from the model's conditionals draws a row. The model's conditionals are approximations, so orders disagree and are averaged (section 4). M1 bounds every drawn value by the extreme quantiles.

**Drift over time.** With the time as a feature, the estimand is $p(y \mid x, t_\star)$ at a time beyond the context's. It is the construction of section 3 with features: a split on a feature (Proposition 2(a)) and a prior over how the mechanism moves with that feature. Not identified without the prior, as in section 3.

**Time-series anomaly detection.** A score of the form $-\log \hat p(y_t \mid \text{past})$ from the forecaster. It needs section 3's forecast to be right and has section 4's problem: anomalous stretches in the context are a dense group.

**Clustering.** The labels are defined up to a permutation, so the loss must be invariant to it, and the decoder returns only classes that the context carries (M2). Neither holds here.

**Unlabeled rows.** The estimand is $p(y_\star \mid x_\star, D, X_U)$ with the unlabeled features $X_U$. If the prior draws the feature distribution and the outcome mechanism independently, $X_U$ changes nothing. In SCM priors they are not independent, so it does (arXiv 2610.14989). M5 keeps such rows out of the context.

**Intervals with coverage.** Conformal calibration of two quantile outputs on held-out context rows gives intervals with coverage of at least $1 - \alpha$ under exchangeability, whatever the model. Inference only; it costs the rows held out.

## 9. What the derivations say together

| Constraint | What it blocks | Tasks |
|---|---|---|
| M1: no mass beyond the 0.001 and 0.999 quantiles | tail densities, extreme draws | outlier scores, density, synthetic rows |
| M2: classes come from the context | a class without a labeled row | one-class outlier detection, clustering |
| M4: queries are independent given the context; outputs are per row | joint distributions; evidence from the set of queries | forecast paths, effect intervals, MNAR, joint draws of several cells |
| M5: a context row needs a target | unlabeled context | MNAR, unlabeled rows, clustering |
| M6: no task input and no column role | two meanings of one table | hidden confounding, noiseless targets, a censoring indicator |

By the kind of change:

- **Inference only**: the forecasting recipe without the clip and the trend outside the model; small random contexts for the outlier score; survival through binary rows; chains for several outputs; conformal intervals. None of these is measured except the first, on D1.
- **Prior only**: temporal splits with trends and the context-row scale (forecasting, drift); reference rows with hidden outliers; query rows with a randomized treatment; MNAR query rows, with the limit of section 5.
- **Structure**: the masked-cell output; a value for "no target" in the context; a task input or column roles.
- No task here needs a new loss with the outputs as they are, except clustering.

`target_train_rows` is a condition of every prior whose split is not random (sections 3 and 5). That is a reason from the derivation. Its measured effect on ordinary tables is not resolved (section 2).

## Checks and corrections

Numerical checks, NumPy and SciPy only, no model weights:

| Section | What was reproduced |
|---|---|
| 1 | the pinball minimizer is the quantile (five levels of a skewed mixture); the mean pinball loss over 999 levels is 1.0010 times half the CRPS |
| 2 | the Student t predictive of the conjugate linear model against 400,000 posterior draws; the decoder's floor |
| 3 | D1's seasonal naive and least-squares columns; the stand-in forecasters of the table; the scale of a linear trend; the median of a half-clipped mixture; the trend fitted outside a stationary forecaster |
| 4 | the density score's AUROC against the share of a dense group, by simulation and by the $\chi^2$ form; uniform outliers; D3's generator with the exact contaminated density; the binomial counts of small contexts |
| 5 | the self-masking means, variance and bias at three settings; the bias with a predictive column at four correlations; the MAR regression |
| 6 | the variance of the least-squares effect against the Frisch–Waugh–Lovell form at four overlap levels; the posterior of the effect against two predictive distributions |
| 7, 8 | the chain rule along three random orders of a Gaussian table; person-period frequencies against the hazards and the Kaplan–Meier product |

Review by agents that did not write the derivations: pending in this draft. Their findings and what was changed are listed here before the pull request is marked ready.

## Sources

Read on 2026-11-28. "Opened" means the cited passage was read for this document; "subagent" means a subagent read it and it was not re-checked; "not opened" means the statement is a standard result that is derived in the text, and no equation number is given for it.

| Source | Used for | Status |
|---|---|---|
| Nagler, Statistical Foundations of Prior-Data Fitted Networks, [arXiv 2305.11097](https://arxiv.org/abs/2305.11097) | Proposition 1 (Theorem 2.2), bias and variance | opened |
| Müller et al., Transformers Can Do Bayesian Inference, [arXiv 2112.10510](https://arxiv.org/abs/2112.10510) | Proposition 1 (Insight 1) | subagent |
| Kobialka et al., Characterizing the Separability Gap in Prior-Data Fitted Networks, [arXiv 2610.14989](https://arxiv.org/abs/2610.14989) | Proposition 2, unlabeled rows | opened (abstract) |
| Rasmussen and Williams, Gaussian Processes for Machine Learning, 2006, Section 2.7 | the trend plus a stationary remainder (Eq. 2.39 to 2.42) | opened |
| fev-bench, [arXiv 2509.26468](https://arxiv.org/abs/2509.26468) | SQL and MASE (Eq. 1 to 4) | opened |
| TabPFN-TS, [arXiv 2501.02945](https://arxiv.org/abs/2501.02945) | the series-to-table form; its Section 6.1 on extrapolation | subagent; cited in v2-task-priors |
| TabPFN-3 technical report, [arXiv 2605.13986](https://arxiv.org/abs/2605.13986) | TabPFN-TS-3 and the fev-bench table | opened |
| TACTIC, [arXiv 2603.14171](https://arxiv.org/abs/2603.14171) | contaminated contexts in training; Tables 5 to 8 | opened |
| FoMo-0D, [arXiv 2409.05672](https://arxiv.org/abs/2409.05672); ICLAD, [arXiv 2603.19497](https://arxiv.org/abs/2603.19497) | the output of dedicated outlier models | subagent |
| TabImpute, [arXiv 2510.02625](https://arxiv.org/abs/2510.02625) | Tables 2 and 3, the default's training pattern | opened |
| Do-PFN, [arXiv 2506.06039](https://arxiv.org/abs/2506.06039); CausalPFN, [arXiv 2506.07918](https://arxiv.org/abs/2506.07918) | what dedicated causal models are trained on | subagent; CausalPFN's IHDP and ACIC numbers are those of docs/task-results |
| Martingale posteriors for prior-fitted networks, [arXiv 2505.11325](https://arxiv.org/abs/2505.11325) | sampling-based uncertainty | subagent |
| Tabular Foundation Models Can Do Survival Analysis, [arXiv 2601.22259](https://arxiv.org/abs/2601.22259) | survival through classification | opened (abstract) |
| arXiv 2603.29475, 2605.15488, 2606.04564 | survival models trained on a prior | subagent |
| TabClustPFN, [arXiv 2601.21656](https://arxiv.org/abs/2601.21656) | clustering | opened (abstract) |
| TabPFGen, [arXiv 2406.05216](https://arxiv.org/abs/2406.05216); Drift-Resilient TabPFN, [arXiv 2411.10634](https://arxiv.org/abs/2411.10634); TabReD, [arXiv 2406.19380](https://arxiv.org/abs/2406.19380); arXiv 2009.13807; arXiv 2609.34887; PFNs4BO, [arXiv 2305.17535](https://arxiv.org/abs/2305.17535) | the candidate table | subagent; the quotation from Drift-Resilient TabPFN is the one checked in v2-experiment-plan §1 |
| Rubin 1976 (Biometrika 63); Hastie, Tibshirani and Friedman 2009, Section 14.2.4; Uria et al. 2014; Hoogeboom et al. 2022, [arXiv 2110.02037](https://arxiv.org/abs/2110.02037); the CRPS as an integral of quantile losses | ignorability; the density ratio by classification; order-agnostic autoregression; Proposition 1's quadrature | not opened |
