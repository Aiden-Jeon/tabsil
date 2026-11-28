# v2 task formulations

Status: written on 2026-11-28 for the re-plan that follows the stop of the v2 plan ([v2-retrospective.md](v2-retrospective.md), [v2-prior-task-plan.md](v2-prior-task-plan.md)). Nothing was trained or evaluated for it, and no GPU job was run. It derives, task by task, what the current model estimates when a task is put to it as a table, and what would have to change for the task to be done well. The designs it starts from are in [v2-task-priors.md](v2-task-priors.md). Numbers of the v1.1 weights come from that document and from [docs/task-results.md](../../docs/task-results.md). The closed forms were reproduced numerically with NumPy (no model weights). Each derivation was then read once by an agent that had not written it; the review found errors, and what was changed is listed in "Checks and corrections".

How to read:

- Every task has the same nine steps: problem, assumptions, estimand, identification, where training converges, table construction, closed-form check, required change and limits, measurement. Classification and regression come first and are the reference for the rest.
- **Measured** is a number from a run in the records. **Derived** follows from the stated assumptions by the steps shown. **Checked** means the closed form was reproduced numerically. **Estimated**, **unknown** and **not verified** are marked as such.
- A required change is one of four kinds. **Inference**: the wrapper or the inference recipe, no training. **Prior**: which tables are drawn (the generator, the split, the context, the scale). **Loss**. **Structure**: an input or an output that the architecture does not have.

## Summary

| Task | Estimand | What the v1 weights compute today | What the derivation says is missing | Kind of change | Basis |
|---|---|---|---|---|---|
| Regression | the posterior predictive distribution of the target | the same between the 0.001 and 0.999 quantiles, as a mixture over 8 preprocessing members | nothing. The target scale of training and inference differ by a term of order $1/n_Q$ | none (a prior option removes most of the scale difference) | derived; checked; a difference of 0.3% is observed once and does not isolate the term |
| Classification | the posterior predictive class probabilities | those probabilities with each class's odds lowered by about $n_c/(n_c + 1)$, within the context's classes | the prior's split puts one row of every class into the context first, so it depends on the labels: a class with few context rows is under-predicted | inference (multiply the factor back): derived, not measured | derived; checked on the prior's own split function |
| Forecasting | the predictive distribution of each future step | a forecast that falls back toward the context mean, as a stationary smoother does; with the default recipe the median exceeds the largest context value by at most one bucket | training never asks for a query far beyond the context: the split is random, so the prior's linear mechanisms are never continued | inference first (a recipe without the clip; the trend fitted outside the model), then prior (temporal split, trend, context-row scale) | D1 measured (trend capture −0.71); the cap is derived from the code and agrees with D1; the trend outside the model is derived, not measured |
| Outlier detection | the probability that a row is an outlier | the density of the contaminated context | a dense group of outliers has a high density: the score ranks it as inliers once its share times its density passes the inliers' | inference (small random contexts: a hypothesis, untested) or prior (labeled reference rows with hidden contamination) | D3 measured (0.425); the exact density fails on the same tables too (0.455, checked) |
| Imputation | the distribution of a missing cell given what the row shows and the fact that the cell is missing | the distribution among rows that observe the cell | MCAR: nothing. A mask that depends on other columns: the same conditional, asked where the context is thin. A mask of another column that depends on the cell: the context is tilted, also under MAR. Self-masking: a different conditional that the data do not identify | MCAR none; the second none expected; the third prior, which `graph_scm_hicard_missing` already has; self-masking prior, and it only moves the answer toward the prior's average mechanism | derived; checked; consistent with TabImpute's specialists (MAR 1.076 against 1.072, self-masking 2.184 against 1.716) |
| Treatment effects | the effect $\tau(x)$ and, if wanted, its uncertainty | the posterior mean of $\tau(x)$ under no hidden confounding | the point estimate: nothing. An interval for $\tau(x)$ does not follow from two predictive distributions. Hidden confounding needs a prior with hidden causes and a marked treatment column | none for the point estimate; a third task value (structure) for intervals; prior with a position convention, or structure, for hidden confounding | derived; checked; IHDP and ACIC measured |
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

Read from the code on `dev-v2` (449c38d).

- **M1, regression output.** 999 values $\hat q_k$ trained with the mean over $k$ of $\rho_{\tau_k}(y_\star - \hat q_k)$ (`tabsil.models.quantile_distribution.pinball_loss`). Inference sorts them and turns them into bucket probabilities on borders that span ±22.6 standard deviations of the context target. The mass below $\hat q_1$ and above $\hat q_{999}$, 0.001 each, is put into the buckets that hold those quantiles, so the output has no tail beyond those buckets.
- **M2, classification output.** A retrieval decoder (`RetrievalDecoder`): the query attends to the context rows with weights $a_i^h$ in head $h$, and the class probability is the attention mass on the rows of that class,

$$
p_c = \frac{1}{H} \sum_{h=1}^{H} \sum_{i \in C} a_i^h \mathbf 1[y_i = c]
$$

  The logit is $\log(\max(p_c, 10^{-5}) + 3 \cdot 10^{-5})$ and the loss is the cross-entropy of its softmax. A class that no context row carries has $p_c = 0$ and ends at a probability of $4 \cdot 10^{-5}$ whatever the query is.
- **M3, scale.** Features are standardized with the context rows and clipped at ±100; a rank feature of each cell saturates at 0 or 1 outside the context's range (`standardize_cells`, `ecdf_midrank`). The v1 prior clips a regression target at 4 standard deviations and standardizes it, both over all rows of the table. Inference standardizes it with the context rows. The prior option `target_train_rows` standardizes again with the context rows; the clip stays an all-row one.
- **M4, rows.** A batch has one context size, and the query rows are the last rows. Query rows attend to context rows only, never to each other, and all statistics come from the context rows. A prediction therefore depends on the context and on its own query row, and the outputs are per row: a joint distribution over several queries is not available.
- **M5, context targets.** Every context row carries a target (it is embedded and added to the row). The model has no value for "target unknown", so an unlabeled row cannot serve as a context row.
- **M6, roles.** The only task input is the choice between classification and regression, which selects the target embedding and the output. No input names a further task or the role of a column. The model does see column positions (the row stage rotates the feature tokens by position), so a role could be tied to a position by a prior and a wrapper; inference shuffles the column order per member today. A column that is constant on the context rows carries no value: its cells are set to zero, and only its missing indicator is left.
- **M7, loss mask.** A query row with a missing target contributes zero to the pinball loss (and still counts in the mean's denominator), and the classification loss ignores the index −100. No prior of today produces a missing target, so neither is used in training.

### Proposition 1: what training converges to

Training minimizes the expected loss over tables drawn from a task distribution $P$ (Müller et al. 2022, arXiv 2112.10510, Insight 1; Nagler 2023, arXiv 2305.11097, Theorem 2.2).

For the cross-entropy,

$$
\mathbb E_P \left[ -\log q_\theta(y_\star \mid x_\star, D) \right] = \mathbb E_P \left[ \mathrm{KL}\left( p_P(\cdot \mid x_\star, D) \Vert q_\theta(\cdot \mid x_\star, D) \right) \right] + \mathbb E_P \left[ \mathrm H\left( p_P(\cdot \mid x_\star, D) \right) \right]
$$

The second term does not depend on $\theta$, so the minimum is at $q_\theta = p_P(\cdot \mid x_\star, D)$ for almost every $(x_\star, D)$ under $P$. "Converges to" means this minimizer throughout: it assumes that the network can represent it and that training reaches it.

For the pinball loss at level $\tau$, write $F$ for the distribution function of $p_P(\cdot \mid x_\star, D)$. Then

$$
\frac{\partial}{\partial q} \mathbb E \left[ \rho_\tau(y_\star - q) \mid x_\star, D \right] = F(q) - \tau
$$

which is zero at $q = F^{-1}(\tau)$. So $\hat q_k$ converges to the $\tau_k$ quantile of the same conditional distribution. The continuous ranked probability score is twice the integral of the pinball loss over the levels, and the mean over the 999 levels is a quadrature of it,

$$
\frac{1}{999} \sum_{k=1}^{999} \rho_{\tau_k}\left( y - F^{-1}(\tau_k) \right) \approx \frac{1000}{999} \int_0^1 \rho_\tau\left( y - F^{-1}(\tau) \right) d\tau = \frac{1000}{999} \cdot \frac{\mathrm{CRPS}(F, y)}{2}
$$

so the regression loss is a proper score over the levels 0.001 to 0.999 (checked for a normal $F$: the integral identity holds to $10^{-8}$, and the quadrature's ratio to $\mathrm{CRPS}/2$ is 1.0010 to within $10^{-4}$, the error of the two end intervals).

**The estimand of training is therefore always the same object: the conditional distribution of what is hidden given what is shown, under the distribution of training tables.** A task is a choice of what is shown (the context, the features, the scale), what is hidden (the target) and which rows are queries (the split).

### Proposition 2: a split on the features and a split on the target

Let all $N$ rows be drawn first and a rule then assign each row to the context or to the queries.

**(a) The rule depends on the features and on independent randomness only.** The assignment is then independent of the targets given the features, so given $\phi$ and the features the targets keep their distribution:

$$
p_P(y_\star \mid x_\star, D) = \int p(y_\star \mid x_\star, \phi) \pi_S(\phi \mid D, x_\star) d\phi
$$

The outcome terms $p(y_i \mid x_i, \phi)$ enter the posterior $\pi_S$ unweighted. The rule can add evidence about the distribution of the features only. In an SCM prior the features and the outcome mechanism share causes, so that evidence is not empty (Kobialka et al. 2026, arXiv 2610.14989), but the form of the estimand is that of the random split. The features of the other query rows are not shown to the network (M4), so $\pi_S$ is an average over them: the network cannot use the query set as evidence.

**(b) The rule depends on the target.** If each row with features $x$ and target $y$ becomes a query with probability $s(x, y)$, independently of the other rows, then

$$
p(y_\star \mid x_\star, \phi, \star \in Q) = \frac{s(x_\star, y_\star) p(y_\star \mid x_\star, \phi)}{\int s(x_\star, y) p(y \mid x_\star, \phi) dy}
$$

and the context rows are tilted by $1 - s$ in the same way. The conditional distribution itself changes. The formula is exact for this rule, also when the number of selected rows is conditioned on (the count then adds a factor to the posterior over $\phi$). A rule that draws a fixed number of rows with probabilities proportional to a score tilts in the same direction, but not by this formula: with 3 rows and one query the two differ by up to 0.2 in probability (a reviewer's enumeration).

This is the ignorability argument of missing-data theory (Rubin 1976) applied to the split. It sorts splits into two kinds: those that keep the estimand and move the queries (a temporal split, missingness that depends on other columns, a randomized treatment in the queries), and those that change the estimand (missingness that depends on the missing value, censoring, an outlier label, and the class-covering rule of today's prior, section 2).

### Proposition 3: outside the training tables the loss constrains little or nothing

Let $P$ be the distribution of training tables and $P'$ the distribution of the tables of a task, with the same conditional $p(y_\star \mid x_\star, D)$. If $P'$ has a density $w$ with respect to $P$ on $(x_\star, D)$, then

$$
\mathbb E_{P'} \left[ \mathrm{KL}(p \Vert q_\theta) \right] = \mathbb E_P \left[ w \cdot \mathrm{KL}(p \Vert q_\theta) \right] \le \lVert w \rVert_\infty \mathbb E_P \left[ \mathrm{KL}(p \Vert q_\theta) \right]
$$

A small excess of the training loss over its entropy term bounds the task's divergence only through the density ratio. Where the ratio is astronomically large the bound is useless, and where $P'$ puts mass that $P$ does not there is none: what the network returns there is whatever the trained function does beyond its data. A task-shaped prior of the first kind is a way to make the ratio small.

## 2. Classification and regression (the reference case)

**1. Problem.** Given a table $D$ of labeled rows and the features $x_\star$ of a new row, predict its target: class probabilities, or a predictive distribution with its mean and quantiles.

**2. Assumptions.**

- A1. The rows are independent draws from one process $\phi$.
- A2. $\phi$ is a draw from the prior $\pi$ (the graph SCM prior).
- A3. The context is a uniformly random subset of the rows. This holds for regression. For classification the prior's split depends on the labels (step 5).
- A4. The target is on the scale the model is shown: class indices, or a standardized value.

**3. Estimand.** The posterior predictive distribution

$$
p_\pi(y_\star \mid x_\star, D) = \int p(y_\star \mid x_\star, \phi) \pi(\phi \mid D) d\phi
$$

and its functionals: the class probabilities, the mean $\mathbb E[y_\star \mid x_\star, D]$, the quantiles.

**4. Identification.** Under the prior the estimand is a fixed function of $(x_\star, D)$. On a real table it equals the true conditional only as far as the posterior concentrates on the truth, which needs the truth to be in the prior's support and the context to be informative. Nagler (2023, Theorems 5.2 and 5.4, Section 6.3) shows that the variance of such a predictor vanishes with $n$, and that its bias vanishes only if the prediction comes to depend on nearby rows alone, which a transformer's attention does not guarantee.

**5. Where training converges.**

*Regression.* By Proposition 1 with A3, $\hat q_k$ converges to the $\tau_k$ quantile of $p_\pi(y_\star \mid x_\star, D)$. M1 fixes the levels at 0.001 to 0.999, so the distribution beyond them is not learned.

*Classification.* M2 restricts the classifier to the context's classes. The prior deals with that by its split: one row of every class that occurs in the table is put into the context first, the rest of the context is filled at random, and a table with a single class is drawn again (`CoverClasses`, `train_rows_cover_classes`). So A3 fails: the split depends on the labels, and by Proposition 2(b) the classifier is trained toward

$$
p_\pi\left( y_\star \mid x_\star, D, \text{every class of the table is in } D \right)
$$

which is not the posterior predictive distribution. A class with few context rows is less likely under it: with one row of a class in the context, a query can be of that class only if the table holds a second one, and the tables in which the context's single row is the only one never offer such a query. The size is derived in step 7. A context with a single class never occurs in training.

**6. Table construction.** The table is the table. For regression the estimand of step 3 is reached if training and inference show the target on the same scale, and they do not quite. With the v1 prior the model is trained on $z = (y - m_{\mathrm{all}})/s_{\mathrm{all}}$, computed over all $N$ rows, and at inference it is given $(y - m_C)/s_C$ from the context rows. The all-row scale carries information about the query targets: if $\bar z_C$ is the mean of the context targets in all-row units, the means of the two groups differ by

$$
\bar y_Q - \bar y_C = -\frac{N}{n_Q} s_{\mathrm{all}} \bar z_C
$$

So a model trained this way can read the mean of the hidden targets off its context. Knowing a mean over $n_Q$ rows lowers the variance of each of them by a share $1/n_Q$, and the pinball loss, which scales with the standard deviation, by about $1/(2 n_Q)$: between 0.07% and 0.5% for stage-1 tables, where $n_Q$ is between 717 and 102, and 0.16% on average. The standard deviation $s_{\mathrm{all}}$ and the all-row clip are further channels of the same kind, whose size is not derived. The one observation of that order: the three control runs of phase 1B, trained on all-row targets, have a reference loss that is 0.3% higher when the reference targets are scaled with the context rows than when they are scaled over all rows (0.0633 against 0.0631; v2-experiment-results, "Phase 1B", "Reference validation loss"). That difference also holds the runs' unfamiliarity with the other scale, so it does not isolate the term. Under A3 the mismatch is this small. Under a split that is not random it is not (section 3).

What the wrapper returns is not one such output but a mixture over 8 members that see the table through different preprocessing, and 4 of them are given a power-transformed target and are mapped back (`EnsembleRegressor`). Each member's output is the model's answer for its own view, on a target form that the prior does not generate. The mixture is the inference recipe's choice and is outside what this derivation covers.

**7. Closed-form check.**

*Regression analysis.* Let the prior be the conjugate linear model: $y = x^\top \beta + \varepsilon$ with $\varepsilon \sim \mathcal N(0, \sigma^2)$, $\beta \mid \sigma^2 \sim \mathcal N(m_0, \sigma^2 \Lambda_0^{-1})$ and $\sigma^2 \sim \mathrm{IG}(a_0, b_0)$. With $X$ and $y$ the context,

$$
\Lambda_n = \Lambda_0 + X^\top X, \quad m_n = \Lambda_n^{-1}\left( \Lambda_0 m_0 + X^\top y \right), \quad a_n = a_0 + \frac n2, \quad b_n = b_0 + \frac12 \left( y^\top y + m_0^\top \Lambda_0 m_0 - m_n^\top \Lambda_n m_n \right)
$$

and the posterior predictive distribution is a Student t distribution with $2 a_n$ degrees of freedom, location $x_\star^\top m_n$ and squared scale $s^2$:

$$
y_\star \mid x_\star, D \sim t_{2 a_n}\left( x_\star^\top m_n, s^2 \right), \quad s^2 = \frac{b_n}{a_n} \left( 1 + x_\star^\top \Lambda_n^{-1} x_\star \right)
$$

A model trained to convergence on tables from this prior would therefore return $\hat q_k = x_\star^\top m_n + s \cdot t_{2 a_n}^{-1}(\tau_k)$. Its mean is the ridge estimate. In the flat-prior limit ($\Lambda_0 \to 0$, $a_0 \to -d/2$, $b_0 \to 0$, the improper prior proportional to $1/\sigma^2$) its mean is the least-squares fit and its central 95% interval is the textbook prediction interval $x_\star^\top \hat\beta \pm t_{n-d}^{-1}(0.975) \hat\sigma \sqrt{1 + x_\star^\top (X^\top X)^{-1} x_\star}$. So for a prior that is a regression model, the trained output would be that model's analysis, estimate and interval together. (Checked: 400,000 posterior draws against the formula, quantiles at 0.01 to 0.99 within 0.007; the flat limit against the textbook interval.)

*Classification with the covering split.* Let the labels be Bernoulli($\theta$) without features, $\theta \sim \mathrm{Beta}(a, b)$, and the context hold $n_1$ positive and $n_0$ negative rows. Under a random split the classifier would be trained toward the posterior predictive probability $(a + n_1)/(a + b + n)$. Under the covering split with one query, the query is positive only if the table has $n_1 + 1$ positives, and it is then one of the $n_1$ positives that were not put into the context first. Counting both cases gives the odds

$$
\frac{P(y_\star = 1)}{P(y_\star = 0)} = \frac{a + n_1}{b + n_0} \cdot \frac{n_1 / (n_1 + 1)}{n_0 / (n_0 + 1)}
$$

The posterior odds are multiplied by a factor that depends on the counts alone and not on the prior. For a class with one context row among many the factor is $\tfrac12$, with two rows $\tfrac23$, with ten $\tfrac{10}{11}$. For the uniform prior the result is the empirical frequency $n_1/n$ where the posterior predictive probability is $(n_1 + 1)/(n + 2)$. Checked by running the prior's own function on simulated tables: with one positive row among 20 the covering split gives 0.049 and 0.050 (1 and 10 queries), the random split 0.094 and the formulas 0.050 and 0.091; for $\mathrm{Beta}(2, 5)$ and 10 context rows, 0.107 against the formula's 0.106. With several queries and small tables the factor is approximate (0.111 with 6 queries in that case). With features the direction is the same, and the size is not derived.

The decoder can express any probability between 0 and 1 for a class that is in the context, since the context rows' keys carry their labels and the query's vector, after the in-context layers, may depend on the counts. That it does so exactly is not demonstrated. For a class that is not in the context it returns $4 \cdot 10^{-5}$ (M2), a loss of 10.1 nats on a row of that class; such a row cannot occur in training.

*Measured.* Diagnostic D2 measured the v1.1 weights against known distributions: an excess of 0.001 nats over the true density for random class labels with frequencies 0.6, 0.3 and 0.1 and 100 to 2,000 context rows, and 0.09 to 0.27 nats for targets independent of the features, with the two-mode target the worst (v2-task-priors, D2). With at least ten rows of the rarest class the covering factor is too small for D2 to show.

**8. Required change and limits.**

- Regression: none for the task. `target_train_rows` removes the scale difference of step 6 except for the clip. Its measured effect on real tables is a pooled error ratio of 0.995 (0.990 to 1.000) at 30K steps, inside the difference between two control seeds (v2-experiment-results, "Phase 1B").
- Classification: the covering split makes rare classes less likely than the posterior predictive distribution has them. If the model has learned the factor of step 7, multiplying each class's probability by $(n_c + 1)/n_c$ and normalizing again undoes it: inference, no training. Derived for labels without features, not measured. It cannot change a binary AUC, since the factor is the same for every query of a table. It can change the log loss where a class has few context rows. The same can be done in the prior: a random split, with the queries whose class is absent from the context left out of the loss (M7), trains the classifier toward the posterior predictive distribution renormalized over the context's classes. That needs training.
- A class that is absent from the context cannot be predicted (M2).

**9. Measurement.** TabArena for accuracy; D2 for calibration, and a variant of D2 with one to three context rows of the rarest class for the covering factor. A scale effect of 0.3% is below what a stage-2 branch resolves (0.6%).

## 3. Forecasting

**1. Problem.** A series $y_1, \dots, y_T$ with timestamps, and possibly covariates that are known for the future. Wanted for each step $h = 1, \dots, H$: the predictive distribution of $y_{T+h}$, reported as quantiles. fev-bench scores the scaled quantile loss (SQL) and MASE (arXiv 2509.26468, Eq. 1 and 2): the quantile loss over the levels 0.1 to 0.9, and the absolute error of the point forecast, each divided by the in-sample error of the seasonal naive forecast.

**2. Assumptions.**

- F1. The series is a regression on time features: $y_t = g(\varphi(t)) + \varepsilon_t$, where $\varphi(t)$ holds the running index, calendar sines and cosines, the seasonal periods found in the context, and the known covariates. This is the form of TabPFN-TS (Hoo et al. 2025, arXiv 2501.02945), which `TabSilForecaster` follows. No lag of the target is a feature.
- F2. $g$ is a draw from the prior. For the v1 weights that is the graph SCM prior. It has no notion of time. It does have linear and other monotone mechanisms, but its split is random.
- F3. The context is the first $T$ rows and the queries are the next $H$: a split on the running index.
- F4. The target is standardized with the context rows.

**3. Estimand.** For each $h$, the distribution $p(y_{T+h} \mid y_{1:T})$ given the features of all $T + H$ rows. These are the marginals. The joint distribution over the horizon is not available (M4), so sample paths and sums over the horizon are out of reach.

**4. Identification.** The values of $g$ on the horizon are not determined by its values on the context: any continuation is compatible with the data. What makes a forecast is the prior over continuations, if the model has been trained to return it.

**5. Where training converges.** The v1 prior draws the split at random (A3). How far beyond the context such a split sends a query, for a feature that is uniform like the running index:

| Event under a random split | Probability | For D1 (365 context rows, 56 queries) |
|---|---|---|
| A given query lies beyond the context's largest value of a given feature | $1/(n + 1)$ | 0.3% |
| It lies beyond by a share $\delta$ of the feature's range | $(1 - \delta)^{n+1}/(n + 1)$ | $10^{-29}$ at D1's last step, $\delta = 0.15$ |
| Every query lies beyond the context's largest value | $\binom{N}{n_Q}^{-1}$ | below $10^{-70}$ |

A training table does hold queries just past the edge of a feature. It never holds a query far past it, nor a table whose queries are all past it. By Proposition 3 the training loss says nothing about forecasting's tables: what the network returns there is not the prior's forecast but whatever the trained function does beyond its data.

The split depends on a feature only, so by Proposition 2(a) a prior that draws temporal splits would train the network toward $p(y_{T+h} \mid y_{1:T})$ under its own distribution over series. SQL's numerator is twice the sum of the same pinball loss at nine of the 999 levels.

**6. Table construction.** Rows are time steps, the features are $\varphi(t)$, the target is $y_t$, the context is the past. The output equals the estimand of step 3 under three conditions, and the v1 setup meets none of them.

- **Support.** The training tables must contain temporal splits (step 5).
- **Prior content.** What the trained forecast then is depends on the prior's functions of the index: for stationary functions it returns to the context mean, for a prior with a linear part it continues the trend (step 7). The graph SCM has linear and quadratic mechanisms, so with temporal splits its own forecast would continue a trend that the context makes plain. That is derived and not measured.
- **Scale.** With an all-row standardization and a temporal split, the context gives the future away. By the identity of section 2, step 6, the mean of the query targets is $-(N/n_Q) s_{\mathrm{all}} \bar z_C$ above the context's, exactly, and for a trend $\bar z_C$ is far from zero: for a linear trend and D1's sizes the context targets have mean −0.23 and standard deviation 0.87 in all-row units (checked), and $N/n_Q = 7.5$. A model trained that way would learn to read the future's level off its context. Inference always shows mean 0 and standard deviation 1. A temporal-split prior therefore needs `target_train_rows`: here the term of section 2 is not small. In context units a linear trend puts the query targets at about

$$
z_{T+h} = \sqrt{12} \left( \frac12 + \frac hT \right)
$$

  between 1.74 and 2.26 for D1, well inside the output's range. The size of the extrapolation is not the obstacle.

A fourth condition is on the wrapper, not on the model: the inference recipe must let a forecast leave the context's range.

**7. Closed-form check.**

*The default recipe caps the median.* The default recipe has 8 members, and 4 of them transform the target and map their distribution back with an inverse that is clipped to the context's range (`_clipped_inverse`). All of their mass lies at or below the largest context value $m$, up to one bucket. The members are averaged with equal weight. If $F_u$ is the distribution of the other four,

$$
F(m) = \frac12 + \frac12 F_u(m) \ge \frac12
$$

so the mixture's median, which is the point forecast, is at most $m$ plus one bucket, whatever the model predicts. This holds whenever at least half of the members transform the target, which is the case for member counts that are multiples of 4. Where the other members' quantile at level $2u - 1$ is above $m$, the mixture's quantile at level $u$ is that quantile, so the upper quantiles are too low as well (checked). D1 measured the cap: with the default recipe no forecast exceeds the context maximum by more than 0.14, and on the linear trend of strength 15 by 0.05, where the true maximum is 14.1 above it and the `plain` recipe reaches 13.0. This part of the gap is the recipe's and needs no training.

D1's `plain` arm differs from the default in three things at once: 2 members instead of 8, the `standard` feature kind only, and no target transform. The fall of the MASE from 11.75 to 9.03 is that of the three together.

*What the model does beyond the context.* D1's trend capture is negative in both arms: on the linear trend of strength 15 the mean forecast lies below the level of the last 28 context values, by 0.71 (`plain`) and 1.31 (default) times the true rise. A forecaster that continued the trend would have a capture of 1, and one that held the last level 0.

A stationary smoother behaves as the model does. As a stand-in, take a zero-mean Gaussian process on the standardized index $u$ with the kernel $\exp(-(u - u')^2 / (2 \ell^2))$, fitted to the standardized remainder of the context after a least-squares fit of a constant and the periodic columns, with the series' true noise (standard deviation 1) as its noise. Its forecast is

$$
\hat z_{T+h} = k_\star^\top \left( K + \sigma^2 I \right)^{-1} z
$$

and $k_\star \to 0$ once the horizon is farther from the data than the length scale $\ell$: the forecast returns to 0, the context mean. On D1's series (4 seeds, linear trend of strength 15; $\ell$ in units of the standard deviation of the context's index, 105 steps):

| Forecaster | MASE | Trend capture |
|---|---|---|
| Least squares on the index and the periodic columns (the forecast of a linear-trend prior) | 0.37 | 0.99 |
| Flat at the mean of the last 28 context values | 5.10 | 0.00 |
| Flat at the context mean | 25.5 | −4.0 |
| Stationary stand-in, $\ell = 0.1$ | 19.9 | −2.9 |
| Stationary stand-in, $\ell = 0.3$ | 4.78 | 0.06 |
| Stationary stand-in, $\ell = 1$ | 3.54 | 0.30 |
| Stationary stand-in, $\ell = 3$ | 0.96 | 0.86 |
| Seasonal naive | 3.67 | 0.28 |
| **TabSil v1.1, `plain` recipe (measured, D1)** | **9.03** | **−0.71** |
| **TabSil v1.1, default recipe (measured, D1)** | **11.75** | **−1.31** |

Seasonal naive and least squares reproduce D1's columns (3.674 and 0.368), so the series are the same. Without a trend the stand-in gives 0.53 at every $\ell$ and the model 0.51.

The reading has limits. The stand-in has a free length scale and covers everything from a return to the mean to a continued trend, so its agreement with the model in size is a fit and not a test. What D1 shows is the sign: the model falls back toward the context mean, by about a fifth (`plain`) to a third (default) of the distance, which a short stationary smoother does and a trend-continuing forecaster does not. Two causes are possible and D1's arms do not separate them: training never asked for a forecast beyond the data (step 5), and one of the model's inputs is the same for every horizon step, since the rank feature of the index saturates past the context (M3) while the standardized value does not.

*A trend prior is a regression plus a stationary remainder.* If $y_t = h(t)^\top \beta + r_t$ with a vague prior on $\beta$ and a stationary process for $r$, the Bayes forecast is (Rasmussen and Williams 2006, Section 2.7, Eq. 2.42)

$$
\hat y_{T+h} = h(T+h)^\top \bar\beta + k_\star^\top K_y^{-1} \left( y - H^\top \bar\beta \right), \quad \bar\beta = \left( H K_y^{-1} H^\top \right)^{-1} H K_y^{-1} y
$$

That is: estimate the trend by generalized least squares, forecast the remainder with the stationary forecaster, add the trend back. With ordinary least squares in place of the generalized fit this can be done in the wrapper, around the model as it is. After a straight line is taken out of D1's linear series, what is left is the series without a trend, which the model forecasts as well as least squares does (0.51 at strength 0). With the same absolute error at strength 15 the MASE would be about 0.4 (least squares has 0.37; the stand-in goes from 9.6 to 0.38 at $\ell = 0.2$). **This is an estimate, not a measurement.** A straight line removes only part of a trend that is not linear, and how much depends on the window it is fitted on. That case is not quantified here.

**8. Required change and limits.**

- **Inference, no training.** (a) A forecasting recipe that lets a forecast leave the context's range: no clipped target transform. Measured on D1 together with two other changes (`plain`), not on fev-bench. (b) The trend fitted outside the model as above: derived, not measured with the model.
- **Prior.** Temporal splits, functions of the index that continue (trend, level shift), and the context-row scale. A split that sorts the rows of an ordinary prior table by one feature is the smallest form: the graph SCM already has linear and quadratic mechanisms, and only the split and the scale change. The series generator of v2-task-priors §1 is the larger form.
- **Loss.** None: the pinball loss is the loss of the benchmark's metric.
- **Structure.** None is required for marginal forecasts of one series. The saturating rank feature (M3) is an input that cannot tell horizon steps apart. Whether a model trained on temporal splits learns to rely on the standardized value instead is not known.
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

**4. Identification.** The mixture $p_c$ is identified by the data. Its split into $\varepsilon$, $p_0$ and $p_1$ is not: what counts as an outlier is an assumption. The usual one is that outliers are diffuse. Ranking by low density is the optimal ranking if and only if $\Lambda$ is a decreasing function of $p_c$. If $p_1$ is constant, $c$, on the region of the data, then $p_c = (1 - \varepsilon) p_0 + \varepsilon c$ is increasing in $p_0$ and $\Lambda = c/p_0$ is decreasing in it, so the condition holds: **the density score is the right score for diffuse outliers** (checked with $\mathcal N(0, I_6)$ inliers and outliers uniform on the box of half-width 6: inside the box the two scores order the rows identically, AUROC 0.9992 for both). If outliers concentrate, $p_1$ is large where $p_c$ is large, and the two orders part.

**5. Where training converges.**

*The wrapper of today.* `TabSilUnsupervised` scores a row by $-\log \hat p(x)$ with the chain rule over the columns, $\log \hat p(x) = \sum_j \log \hat p(x_{\pi_j} \mid x_{\pi_{\lt j}})$, each factor a regression or classification fit on the context. By section 2 each factor estimates a conditional of the context's distribution. The score therefore estimates $-\log p_c(x)$, the density of the contaminated context.

*Reference rows, with the weights of today.* Let the context hold the data rows with label 0 and reference rows with label 1 drawn from a known density $g$, in shares $1 - \rho$ and $\rho$. A classifier then converges to

$$
P(\text{reference} \mid x) = \frac{\rho g(x)}{\rho g(x) + (1 - \rho) p_c(x)}
$$

an increasing function of $g(x)/p_c(x)$ (Hastie, Tibshirani and Friedman 2009, Section 14.2.4; the formula is Bayes' rule and is derived here, the book's equation number was not re-read). This is density estimation by another route: for a uniform $g$ it ranks rows as the density score does, and fails where the density score fails.

*Reference rows in training tables with hidden outliers.* Now let the training tables hide outliers among the data rows, keep the label 0 for all of them, and give the query rows their true label. By Proposition 1 the classifier is trained toward

$$
P_\pi(o_\star = 1 \mid x_\star, D)
$$

the posterior probability that the query is an outlier given a contaminated, unlabeled context, under the prior over outlier kinds. This is the estimand of step 3, with the identification of step 4 supplied by the prior. Two remarks on what makes it possible:

- The reference rows carry no information about the outliers. They are in the context because the decoder returns only classes that the context holds (M2). What they are drawn from is a design choice.
- The output is the share of the query's attention that falls on the reference rows. The context rows' keys carry their labels, so that share can in principle take any value between 0 and 1 whatever the reference rows' features are. That training finds such a solution is not demonstrated: the rows nearest to an outlier in a dense group are its fellow group members, which carry the label 0.

**6. Table construction.** For the trained form: the context is the data rows (label 0) followed by reference rows (label 1), and the queries are held-out data rows. After training, the output is the posterior probability under the prior's outlier kinds. It is the $\eta$ of a real table only as far as that table's outliers are of those kinds, and an unlabeled table cannot say (step 4): the gain on a benchmark is the overlap between the prior's notion of an outlier and the benchmark's labels.

A one-class form, with every context row labeled 0 and no reference rows, is impossible with this decoder: the probability of the absent class is the constant $4 \cdot 10^{-5}$ for every query (M2).

**7. Closed-form check.**

*A dense group passes for inliers.* Let the inliers be $\mathcal N(0, I_d)$ and the outliers a group $\mathcal N(\mu, \sigma_c^2 I_d)$ far from them. A typical inlier has $\log p_c \approx \log(1 - \varepsilon) - \frac d2 \log 2\pi - \frac12 \chi^2_d$, and a typical group member $\log \varepsilon - \frac d2 \log 2\pi - d \log \sigma_c - \frac12 \chi^2_d$. The density score ranks the group as more normal than the inliers when

$$
\log \frac{\varepsilon}{1 - \varepsilon} - d \log \sigma_c \gt 0, \quad \text{that is} \quad \varepsilon \gt \frac{\sigma_c^{d}}{1 + \sigma_c^{d}}
$$

In general the group overtakes the inliers when $\varepsilon p_1$ at the group exceeds $(1 - \varepsilon) p_0$ at a typical inlier. For unit Gaussian inliers and a group as tight as D3's, $\sigma_c = 0.15$ and $d = 6$, the threshold is a share of $1.1 \cdot 10^{-5}$. Checked by simulation with the exact density (group centre at 2 in every coordinate):

| Share $\varepsilon$ of the group | 0 | $10^{-6}$ | $10^{-5}$ | $10^{-4}$ | 0.02 |
|---|---|---|---|---|---|
| AUROC of the density score $-\log p_c$ | 0.9995 | 0.857 | 0.527 | 0.168 | 0.003 |
| AUROC of the likelihood ratio | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

*D3 itself.* D3's inliers lie on a thin two-dimensional surface in six dimensions and are far denser than a unit Gaussian, so the share at which the group overtakes them is much larger and depends on the geometry. The exact contaminated density was computed on D3's own tables (the script's draw order, so the same context and test rows; the inlier density by Monte Carlo over its two factors, the group's density exact):

| Share of the group in the context | 0 | 2% | 20% |
|---|---|---|---|
| Exact contaminated density, mean AUROC (seeds 0, 1, 2) | 1.000 | **0.455** (0.376, 0.148, 0.841) | 0.210 (0.077, 0.000, 0.553) |
| v1.1 weights, measured in D3 (seeds 0, 1, 2) | 0.999 | **0.425** (0.494, 0.255, 0.527) | 0.453 (0.486, 0.360, 0.512) |

The exact density passes an AUROC of 0.5 at a share of about 1% in seed 0, between 0.1% and 0.3% in seed 1, and only near 20% in seed 2. So at 2% it fails in two of the three geometries and on average, as the model does. The seeds do not agree one by one. With 30 test outliers a single AUROC is noisy, and the model's density is flatter than the exact one: its mean log density of the inliers is −1.3, −1.3 and −3.7 where the exact values are 1.2, 0.3 and −0.3. What the table shows is that the failure belongs to the estimand. A better density estimate, from more prior variety or more training, moves toward the first row, which is no better.

*What does not help.* Dropping the highest-scoring rows from the context and scoring again removes rows of low density. The group has a high density, so it stays (derived; not run).

**8. Required change and limits.**

- **Inference, no training: a hypothesis.** The score must stop seeing the group as dense. A context of $m$ random rows holds about $\varepsilon m$ group members: at $\varepsilon = 0.02$ a context of 64 rows has no group member with probability 0.27 and at most one with 0.63 (checked). If a density estimated from so few rows cannot be as sharp at one or two points as the group is, scoring with many small random contexts and averaging would not see the group as dense. IsolationForest works on subsamples of 256 rows for a like reason. This does not follow from the threshold above, which a single group member in 64 rows already exceeds; it depends on how the model smooths, and on whether its density from 64 rows is still good enough on the inliers. **Unknown gain; not found in the literature for this kind of model.**
- **Inference, of small expected effect.** The output has no mass beyond its extreme quantiles (M1), so a cell there gets a floor value. At most 0.2% of a calibrated model's cells are affected per conditional. Unknown effect on the AUROC.
- **Prior.** The reference-row task with hidden outliers of step 5: a prior change, with the wrapper that appends the same reference rows at inference. No change of the loss or of the structure.
- **Structure.** Not needed by the derivation. The models built for this task end in a two-way output on the query, trained with the cross-entropy against the hidden label (TACTIC, arXiv 2603.14171, Section 3; FoMo-0D, arXiv 2409.05672, and ICLAD, arXiv 2603.19497, are reported to do the like, read by a subagent). This model would replace that output by the attention share on reference rows, with the open question of step 5.

**9. Measurement.**

- D3: the group at 2% (0.425 today; 0.9 is the design's criterion) and the curve at 10% (0.064).
- ADBench, 12 tables (docs/task-results): mean AUROC 0.663 against 0.722 for IsolationForest and 0.704 for the TabPFN v2 extension; 0.820 with a clean context, which uses labels and is the upper reference.
- What others gained by training on contaminated contexts: on 38 ADBench tables TACTIC reports a mean AUROC of 80.6 with a contaminated context, against 67.6 for the TabPFN-2.5 unsupervised extension (the method of today's wrapper) and 77.7 for IsolationForest; with a clean context 90.4, 88.9 and 82.3 (TACTIC, Tables 5 and 7). That is a model built for the task alone, on other tables than the 12 here.
- Predictions that would show this section wrong: context refinement raises D3's group at 2%; or reference rows with today's weights rank the group above the inliers.

## 5. Imputation

**1. Problem.** A table with missing cells. Wanted for each missing cell: its distribution given the row's observed cells, a point value (the mean for a numeric column, the mode for a categorical one), or draws.

**2. Assumptions.**

- I1. Rows are independent draws of a full row $x$ and a mask $M$.
- I2. The mechanism is one of: MCAR, $P(M \mid x) = P(M)$; MAR, $P(M \mid x) = P(M \mid x_{\mathrm{obs}})$; MNAR, anything else (Rubin 1976).
- I3. The wrapper works column by column: for column $j$ the context is the rows that observe it, the features are the other columns (missing ones stay missing), and the queries are the rows that miss it (`TabSilUnsupervised.impute`).

**3. Estimand.** For a row that misses column $j$: $p(x_j \mid o, M_j = 1)$, where $o$ is what the row shows: the values of its other observed cells and which of the other cells are missing.

**4. Identification.** The context estimates the distribution among the rows that observe the column, $p(x_j \mid o, M_j = 0)$. By Bayes' rule

$$
p(x_j \mid o, M_j = 1) = \frac{P(M_j = 1 \mid x_j, o)}{P(M_j = 1 \mid o)} p(x_j \mid o)
$$

and the context's conditional is the same expression with $1 - P$ in both places.

- The two are equal if and only if $P(M_j = 1 \mid x_j, o)$ does not depend on $x_j$: given what the row shows, whether $x_j$ is missing must not depend on $x_j$. The estimand is then identified and is what the context estimates.
- MCAR satisfies this. Rubin's MAR satisfies it when column $j$ is the only incomplete column. With several incomplete columns it does not follow from MAR: if another column's mask depends on $x_j$ in the rows that observe $x_j$, which MAR allows, then what a row shows depends on the target (example in step 7).
- If the condition fails, the ratio depends on the missing value. The observed data are compatible with many mechanisms, and the estimand is not identified without an assumption on the mechanism.

**5. Where training converges.** For one column the wrapper's table is a regression or classification table, and its split is the mask.

- **MCAR** is a random split: the v1 prior is already an MCAR imputation prior for one column, and the output converges to the estimand (section 2).
- **A mask of column $j$ that depends on the other columns' shown values** is a split on the features (Proposition 2(a)): the same conditional, but the queries sit where the mechanism sends them. With missingness that rises with a column $x_1$, the rows to impute have large $x_1$ and the context has few such rows. This is Proposition 3 with a finite density ratio: the estimand is right, and the accuracy is asked where the context is thin.
- **A mask of another column that depends on $x_j$** selects on the target without removing a row: the rows that show that column are a tilted sample. The v1 prior has no such tables, since its missing cells are completely at random. `graph_scm_hicard_missing` has them: its structured missing values include columns that are missing in one tail of the target and in one tail of another column (`mask_by_target`, `mask_censoring`).
- **A mask of column $j$ that depends on $x_j$ itself** (self-masking MNAR) is a split on the target (Proposition 2(b)). A prior that draws its query rows by a mechanism trains the network toward $p_\pi(x_j \mid o, M_j = 1, D)$, which averages over the prior's mechanisms given a context of complete cases. The incomplete rows reach the model one at a time, as queries: they are not context (M5) and do not see each other (M4). What they could add, how their shown cells differ from the complete rows', is evidence under a prior and not identification. No input removes the dependence on the prior's mechanisms.

**6. Table construction.** As I3. For an MNAR prior the scale matters again: under self-masking the context's mean is shifted, and inference standardizes with the context. The prior must do the same (`target_train_rows`), or the tilt it teaches is on another scale than the one it is asked on.

**7. Closed-form check.**

*Self-masking.* Let $x \sim \mathcal N(0, 1)$ be missing with probability $\Phi(a + b x)$. With $\alpha = a/\sqrt{1 + b^2}$, $\kappa = b/\sqrt{1 + b^2}$ and $\lambda = \varphi(\alpha)/\Phi(\alpha)$,

$$
\mathbb E[x \mid M = 1] = \kappa \lambda, \quad \mathbb E[x \mid M = 0] = -\kappa \frac{\varphi(\alpha)}{1 - \Phi(\alpha)}, \quad \mathrm{Var}(x \mid M = 1) = 1 - \kappa^2 \lambda (\lambda + \alpha)
$$

so the mean of the complete cases misses the mean of the missing values by

$$
\mathbb E[x \mid M = 0] - \mathbb E[x \mid M = 1] = -\kappa \frac{\varphi(\alpha)}{\Phi(\alpha) (1 - \Phi(\alpha))}
$$

At a missing rate of 30% and $b = 1$ this is −1.17 standard deviations. The error of imputing the complete-case mean is then 1.42, where 0.80 is possible for someone who knows the mechanism (checked by simulation to three digits, also at $b = 0.5$ and $b = 2$).

*With a predictive column.* If another column $x_1$ has correlation $\rho$ with $x$ and $s^2 = 1 - \rho^2$ is the variance it leaves, the bias of the complete-case regression at a given $x_1$ is the same expression with $b s$ in place of $b$, times $s$:

$$
-s \frac{b s}{\sqrt{1 + b^2 s^2}} \frac{\varphi(\alpha_1)}{\Phi(\alpha_1) (1 - \Phi(\alpha_1))}, \quad \alpha_1 = \frac{a + b \rho x_1}{\sqrt{1 + b^2 s^2}}
$$

It is −1.17 at $\rho = 0$, −0.94 at 0.5, −0.30 at 0.9 and −0.03 at 0.99 (checked at $x_1 = 0$). **The self-masking bias falls with the variance that the other columns leave unexplained**, a little slower than in proportion.

*A mask of the column that depends on another column.* With missingness $\Phi(-0.74 + 1.5 x_1)$ and $\rho = 0.6$, the regression on the complete cases has slope 0.600 and intercept 0.001, the truth. The rows to impute have a mean $x_1$ of 0.90 against −0.46 in the context, and 32% of them lie beyond the context's 99th percentile of $x_1$ (checked).

*A mask of another column that depends on the cell to impute.* Let column 1 be missing completely at random in 30% of the rows, and column 2 be missing with probability $\Phi(-0.5 + x_1)$ in the rows that observe column 1. This is MAR in Rubin's sense. To impute column 1 at $x_2 = 1$ the truth is $\mathbb E[x_1 \mid x_2 = 1] = 0.60$. The context rows that show column 2 have a mean $x_1$ of 0.17 there, and their regression of $x_1$ on $x_2$ has slope 0.48 and intercept −0.30 (checked; the example is the reviewer's, "Checks and corrections").

*Against the literature.* TabImpute trains one model per pattern (arXiv 2510.02625, Table 2, normalized RMSE):

| Evaluated on | Trained on MCAR (its default) | Trained on that pattern | Trained on self-masking |
|---|---|---|---|
| MCAR | 0.819 | 0.819 | 1.135 |
| Column MAR | 1.076 | 1.072 | 1.770 |
| Self-masking MNAR | 2.184 | 1.716 | 1.716 |

The MAR specialist gains 0.4% on its pattern, which is consistent with a conditional that does not change. The self-masking specialist gains 21% on its own pattern and loses 39% on MCAR: it has learned the tilt of its prior and applies it where there is none. Its paper states that a mix of patterns "degraded overall because it focused only on Self-MNAR".

**8. Required change and limits.**

- MCAR: none. The v1.1 weights are ahead of every baseline of the suite in the mean (NRMSE 0.546 at 10% against 0.616 for MissForest; lower than every baseline on 25 of the 30 table and rate cells).
- A mask of column $j$ that depends on other columns: none expected. A split that follows the mechanism would train the thin region (prior). TabImpute's 0.4% is the only number.
- A mask of another column that depends on $x_j$: prior, and the prior exists. The weights of the v2 candidate were trained on such tables, and whether they impute better under such masks than v1.1 can be measured without training. Not measured.
- Self-masking: prior, with a limit that a prior cannot remove. The model can learn the average tilt of the mechanisms it is trained on. On a table with another mechanism, or none, that tilt is an error of the size in the table above. Whether real missing values follow a given mechanism cannot be measured: their true values are unknown. A benchmark that masks cells by a mechanism measures the match between that mechanism and the prior's.
- Using the incomplete rows as context for other columns needs a context row without a target: structure (M5), or the masked-cell output of section 7.
- Draws that respect the dependence between several missing cells of a row need the cells imputed one after the other, each conditioned on the ones before: inference only, at one fit per cell pattern.

**9. Measurement.** D4 (not run): the same mask rate under MCAR, a mask that depends on another column, a mask of another column that depends on the target, and self-masking, on prior tables with a known mechanism and on the imputation suite, against the suite's baselines. Predictions: under the second mechanism the error is the MCAR error with a ratio like the baselines'; under the third, v1.1 is biased and the weights trained on `graph_scm_hicard_missing` less so; the self-masking error follows the closed form and shrinks with the columns' predictability.

## 6. Treatment effects

**1. Problem.** Rows $(x, t, y)$ with a binary treatment $t$ and an outcome $y$. Wanted: the conditional average treatment effect $\tau(x) = \mathbb E[Y(1) - Y(0) \mid x]$, scored by the root of the PEHE, and the average effect.

**2. Assumptions.**

- T1. Consistency: the observed outcome is $Y(t)$ for the treatment received.
- T2. No hidden confounding: $(Y(0), Y(1))$ is independent of $t$ given $x$.
- T3. Overlap: $0 \lt e(x) \lt 1$ for the propensity $e(x) = P(t = 1 \mid x)$.
- T4. The outcome surfaces $\mu_t(x) = \mathbb E[y \mid x, t]$ are draws from the regression prior, with the treatment as one more feature (the S-learner of `TabSilCATE`).

**3. Estimand.** $\tau(x) = \mu_1(x) - \mu_0(x)$; for uncertainty, its posterior distribution given the data.

**4. Identification.** T1 and T2 give $\mathbb E[Y(t) \mid x] = \mathbb E[y \mid x, t]$, and T3 makes both sides defined at every $x$: the effect is a difference of two regression functions. Without T2 it is not identified by the data, and an answer exists only through a prior over hidden structure (the approach of Do-PFN, arXiv 2506.06039, whose prior does not impose T2; read by a subagent, not re-checked).

**5. Where training converges.** With T4 the S-learner's table is an ordinary regression table, and the model's mean converges to $\mathbb E_\pi[\mu(x, t) \mid D]$ (section 2). The mean is linear, so

$$
\mathbb E_\pi[\tau(x) \mid D] = \mathbb E_\pi[\mu(x, 1) \mid D] - \mathbb E_\pi[\mu(x, 0) \mid D]
$$

**The difference of the two predicted means is the posterior mean of the effect under the regression prior.** No training on effects is needed for the point estimate. (The mean that `predict` returns values the two tail buckets at the ends of the context's target range, so it is the output's mean up to its tails.)

The uncertainty is another matter. The posterior variance of the effect is

$$
\mathrm{Var}(\tau(x) \mid D) = \mathrm{Var}(\mu_1(x) \mid D) + \mathrm{Var}(\mu_0(x) \mid D) - 2 \mathrm{Cov}(\mu_1(x), \mu_0(x) \mid D)
$$

and the model returns two predictive distributions of $y$, each with variance $\mathrm{Var}(\mu_t(x) \mid D) + \sigma^2$, as separate outputs (M4). Neither the outcome noise $\sigma^2$ can be taken out nor the covariance read off. An interval for the effect is not available from the outputs.

**6. Table construction.** The context is the observed rows with the features $(x, t)$ and the target $y$. The queries are $(x, 1)$ and $(x, 0)$. The split is on the features only (Proposition 2(a)), but the counterfactual query $(x, 1 - t)$ of a row lies where its treatment is rare. In training, the queries of a table follow that table's own distribution of $(x, t)$. Against those, the density of a treated query at a control row's $x$ is

$$
w(x) = \frac{p(x \mid t = 0)}{p(x, t = 1)} = \frac{1 - e(x)}{e(x) P(t = 0)}
$$

which is unbounded as $e(x) \to 0$ (18 at a propensity clip of 0.1 and 98 at 0.02 in the design of step 7). Low overlap is the case of Proposition 3 in which the bound is lost: the prediction there is whatever the trained function returns where its context has no rows. If the training tables' queries had a treatment drawn with probability $\tfrac12$ independently of the features, the ratio would be $(1 - e(x)) / (\tfrac12 P(t = 0))$, at most $2 / P(t = 0)$ (3.6 and 3.9 in the same design).

Two things are not in the table. No input marks the treatment column: a prior with hidden confounding could mark it only by its position, with a wrapper that puts the treatment there and members that do not shuffle it (M6). And a query cannot ask for the noiseless $\mu_t(x)$ in place of $y$: with one task value for regression, the same table would have to mean two tasks.

**7. Closed-form check.**

*Linear outcome.* For $y = \alpha + \beta^\top x + \tau t + \varepsilon$ fitted by least squares, the Frisch–Waugh–Lovell theorem gives

$$
\mathrm{Var}(\hat\tau) = \frac{\sigma^2}{\sum_i (t_i - \hat t_i)^2}
$$

with $\hat t_i$ the linear projection of the treatment on the features. The sum is at least about $n \mathbb E[e(x)(1 - e(x))]$, so the standard deviation is at most about $\sigma / \sqrt{n \mathbb E[e (1 - e)]}$, a bound that grows as overlap shrinks; the actual rate is set by the residual of the linear projection. Checked with $x \sim \mathcal N(0, 1)$, $e(x) = \Phi(2.5 x)$ clipped, $y = 0.5 x + t + \varepsilon$, $n = 2{,}000$ and 400 replications: as the clip goes from 0.5 to 0.02, the standard deviation of $\hat\tau$ rises from 0.041 to 0.062 (the formula gives 0.045 and 0.063), and a T-learner of two linear fits rises from 0.060 to 0.088. In this design both grow at the same rate, since both are correctly specified. If TabSil's error grew faster than a T-learner's in D5, that would point at how the model answers where its context is empty and not at overlap alone. One linear design does not settle that reading.

*Interval.* In a Bayesian linear model with an interaction ($x \sim \mathcal N(0, 1)$, $t \sim \mathrm{Bernoulli}(\Phi(1.5 x))$, a $\mathcal N(0, 10^2)$ prior on the four coefficients, $n = 200$, $\sigma = 1$), the posterior standard deviation of $\tau(x)$ is 0.19 at $x = 0$ and 0.34 at $x = 1.5$. Two predictive distributions treated as independent give 1.43 and 1.45 (checked). The outputs overstate the effect's uncertainty by the outcome noise, $2 \sigma^2$ in variance.

**8. Required change and limits.**

- The point estimate under T2: none. The S-learner's root PEHE on IHDP is 0.55 out of sample, where CausalPFN, a model trained for the task, reports 0.58 and the EconML T-learner has 2.08 (docs/task-results; protocols differ). The posterior mean under the regression prior is not always the best estimate on a real table: on ACIC 2016 the X-learner with the same base model has 0.37 out of sample against the S-learner's 0.54.
- Low overlap: prior. Query rows whose treatment is drawn independently of the features put the S-learner's queries into the training tables with a bounded ratio (step 6; v2-task-priors §4). Unknown gain; D5 is not run.
- An interval for the effect: a further task. A model that returns the posterior of $\mu_t(x)$ must be trained on noiseless targets (CausalPFN, arXiv 2506.07918, trains on the conditional expected potential outcome under T2). One set of weights can serve both that and ordinary regression only with a third task value and its target embedding and output: a small change of structure. Sampling schemes on top of the outputs (the martingale posterior, arXiv 2505.11325, not re-read) are inference only and cost many passes.
- Hidden confounding: a prior with hidden causes and a marked treatment. The mark can be a position, which needs a wrapper and no new input (untested here; Do-PFN marks the treatment by its position), or an input of its own (structure).

**9. Measurement.** IHDP and ACIC 2016 for the point estimate. D5 (not run): the root PEHE on prior tables as the propensity's clip shrinks from 0.5 to 0.02, against the true effect and a T-learner.

## 7. The masked-cell reconstruction loss

The idea on record (2026-11-26, undecided): hide random cells of the training tables and train an extra output, on the cell states, to reconstruct them. This section derives what such a loss estimates.

**1. Setup.** Let $S$ be the set of hidden cells of a table and let the head return, for a hidden cell $(i, j)$, a distribution of its value: 999 quantiles for a numeric column, class probabilities for a categorical one, with the losses of M1 and M2.

**2. What it converges to.** By Proposition 1, for each hidden cell

$$
q_\theta(\cdot) \to p_\pi\left( x_{ij} \mid x_{i, \mathrm{vis}}, X_{\mathrm{vis}}, S \right)
$$

the posterior predictive distribution of the cell given the visible cells of its own row and of all other rows. The mask is drawn independently of the values, so it is ignorable (Proposition 2(a)): this is the MCAR conditional of section 5, for every column and every set of visible columns at once. This holds for every weighting of the cells in the loss, as long as each number of visible columns occurs in training.

**3. What it gives each task.**

- **Imputation.** The estimand of section 5 under MCAR, in one pass for all missing cells where the wrapper needs one fit per column. Rows with missing cells are themselves context for other cells, which M5 rules out today. The estimand does not change, so no gain in accuracy follows from the derivation except through that extra context.
- **Density.** The chain rule needs $p(x_{\pi_j} \mid x_{\pi_{\lt j}})$ for every prefix of a column order. Those are masked-cell conditionals with between none and all but one of the other columns visible. If the number of hidden cells of a row is uniform between 1 and the number of columns (a mask rate drawn uniformly per row gives that) and the loss is averaged over the hidden cells, the masked loss equals the mean, over column orders and positions, of the chain-rule term $-\log q(x_{\pi_j} \mid x_{\pi_{\lt j}})$: the order-agnostic autoregressive objective (Uria et al. 2014; Hoogeboom et al. 2022, arXiv 2110.02037; neither re-read here). With the loss summed over the hidden cells the positions get unequal weights; the minimizers are the same. For a Gaussian table the sum of the exact conditionals along any order equals the joint log density (checked). A mask rate that stays well below 1 never trains the conditionals with few visible columns, which are the ones the chain rule starts with: a loss meant for imputation at ordinary missing rates does not by itself serve density.
- **Outlier detection.** The density it yields is again the density of the contaminated context. Section 4's result is unchanged.
- **Classification and regression.** The main loss's minimizer is the same with or without the extra loss, under two conditions: the network can represent both minimizers at once, and the main loss sees the same tables as before. If the main loss is computed on the masked tables, its inputs hold more missing cells; the minimizer for each input is still the one of section 2, but the training distribution has moved (Proposition 3). Under those conditions the auxiliary loss changes nothing at the optimum. Any effect is one of finite capacity or finite training, and its sign is not derivable. No controlled evidence exists: LimiX and LimiX-2 train with a masked objective and report no ablation of it (v2-experiment-plan §1, row 9).

**4. Required change.** Structure: an output on the cell states, used in training only if it serves as an auxiliary loss, kept if it serves imputation and density.

## 8. New task candidates

A candidate is listed with what it is, whether it can be put as classification or regression tables, a public benchmark, and earlier work with prior-fitted networks. The last column is the kind of change by the short derivations below. Sources marked † were read for this document; the others were read by a subagent and are not verified.

| Candidate | As classification or regression tables | Benchmark | Earlier work | Kind of change |
|---|---|---|---|---|
| Survival analysis with right censoring | yes: one binary row per subject and time bin | SurvSet collection; C-index, integrated Brier score | Off-the-shelf tabular models through classification reductions, 60 datasets (arXiv 2601.22259†, abstract). Models trained on a survival prior: arXiv 2603.29475, 2605.15488, 2606.04564 | inference, untested with these weights; a trained censoring prior needs a marked column |
| Multi-output and multi-label prediction | yes: one table per output, earlier outputs as features | none standard found | wrappers only | inference |
| Ranking | yes: regression on the gain of the relevance | MSLR-WEB30K, Yahoo LTRC, Istella; NDCG | a repository and a 2026 conference paper, not read | inference; the context size is the limit |
| Density estimation and synthetic rows | yes: the chain rule of section 4, sampled | no agreed benchmark; TabPFGen (arXiv 2406.05216) scores downstream AUC | TabPFGen; the TabPFN unsupervised extension | inference; no tail beyond the extreme quantiles (M1) |
| Prediction under drift over time | yes: time as a feature and a temporal split | TabReD (arXiv 2406.19380) | Drift-Resilient TabPFN (arXiv 2411.10634): better out of distribution, "lower predictive performance on ID tasks" | prior: the split of section 3, plus drift |
| Time-series anomaly detection | yes: the forecast's distribution as the score | TSB-AD; older benchmarks are criticized (arXiv 2009.13807) | none found | inference, on top of forecasting; inherits sections 3 and 4 |
| Clustering | no | the 44 datasets of TabClustPFN | TabClustPFN (arXiv 2601.21656†, abstract): a loss that is invariant to label permutation and a network for the number of clusters | loss and structure |
| Using unlabeled rows | no | none specific | SemiTabICL (arXiv 2610.14989†, abstract): a model pretrained to use them beats a matched supervised one | structure (M5) and prior |
| Intervals for effects | not from two predictive distributions (section 6) | coverage on IHDP, ACIC | CausalPFN; the martingale posterior | a third task value (structure), or many passes at inference |
| Prediction intervals with guaranteed coverage | yes: conformal calibration of the quantile outputs | any regression benchmark | conformal methods on tabular models (arXiv 2609.34887) | inference |
| Bayesian optimization | yes: an acquisition function from the predictive distribution | HPO-B | PFNs4BO (arXiv 2305.17535) | inference |

**Survival.** Time is cut into bins $k = 1, \dots, K$. A subject has an event bin $T$ and a censoring bin $C$; observed are $t = \min(T, C)$ and $\delta = \mathbf 1[T \le C]$. The hazard is $h_k(x) = P(T = k \mid T \ge k, x)$ and the survival function $S(k \mid x) = \prod_{j \le k} (1 - h_j(x))$. If censoring is independent of the event time given $x$, the likelihood of a subject is

$$
\prod_{k=1}^{t} h_k(x)^{\mathbf 1[k = t, \delta = 1]} \left( 1 - h_k(x) \right)^{1 - \mathbf 1[k = t, \delta = 1]}
$$

which is the Bernoulli likelihood of $t$ binary rows with features $(x, k)$ and the label "the event happened in this bin": one row for every bin in which the subject was at risk. The estimand, $h_k(x)$, is the class probability of a binary table, and the censoring is handled by which rows exist. Which rows exist depends on earlier labels, but a stopping rule does not change a likelihood, so a posterior over the hazards is the one of a table with these rows and independent labels. Checked: on simulated data with a censoring bin drawn uniformly, the row frequencies recover the hazards to 0.002 and their running product is the Kaplan–Meier curve; using the uncensored subjects alone is biased (a survival of 0.37 where the truth is 0.69 at the fourth bin). Two cautions. Such tables repeat a subject's features over several rows, which the prior's tables never do, so Proposition 3 gives no assurance for today's weights: the route is reported to work for other tabular models (arXiv 2601.22259) and is untested here. And the context grows to $\sum_i t_i$ rows. A prior that shows censored targets in the context and true event times in the queries would train the direct answer, but the model would have to know which column is the event indicator (M6).

**Multi-output.** $p(y_1, \dots, y_m \mid x) = \prod_k p(y_k \mid x, y_{\lt k})$. Each factor is an ordinary table with the earlier outputs as features, known in the context and filled with draws or predictions for the queries. Exact by the chain rule, at $m$ fits; errors of early outputs propagate.

**Ranking.** For one query group, the expected discounted gain of an order is a sum over positions of a decreasing weight times $\mathbb E[\mathrm{gain}(y) \mid x]$ of the item placed there. It is largest when the items are sorted by that expectation (rearrangement inequality). The estimand is the expected gain, not the expected relevance: with the usual gain $2^y - 1$ an item that is surely of relevance 1 and one that is 0 or 2 with equal chances have the same mean and expected gains of 1 and 1.5. So the regression target is the gain, or the expectation is taken from the predictive distribution. The table is the pairs of query and item. Benchmarks have millions of rows, far beyond a context.

**Density and synthetic rows.** For the exact posterior predictive distribution of a new row the chain rule holds along every column order, so drawing the columns in turn from the model's conditionals draws a row. The model's conditionals are approximations, so orders disagree and are averaged (section 4). M1 bounds every drawn value by the buckets of the extreme quantiles.

**Drift over time.** With the time as a feature, the estimand is $p(y \mid x, t_\star)$ at a time beyond the context's. It is the construction of section 3 with features: a split on a feature (Proposition 2(a)) and a prior over how the mechanism moves with that feature. Not identified without the prior, as in section 3.

**Time-series anomaly detection.** A score of the form $-\log \hat p(y_t \mid \text{past})$ from the forecaster. It needs section 3's forecast to be right and has section 4's problem: anomalous stretches in the context are a dense group.

**Clustering.** The labels are defined up to a permutation, so the loss must be invariant to it, and the decoder returns only classes that the context carries (M2). Neither holds here.

**Unlabeled rows.** The estimand is $p(y_\star \mid x_\star, D, X_U)$ with the unlabeled features $X_U$. If the prior draws the feature distribution and the outcome mechanism independently, $X_U$ changes nothing. In SCM priors they are not independent, so it can (arXiv 2610.14989). M5 keeps such rows out of the context.

**Intervals with coverage.** Split conformal calibration: hold $m$ labeled rows out of the context, score them with the model fitted on the rest, and widen the interval between two quantile outputs by the $\lceil (1 - \alpha)(m + 1) \rceil / m$ quantile of the scores. The coverage is at least $1 - \alpha$ if the held-out rows and the new rows are exchangeable, whatever the model. Inference only; it costs the rows held out, and rows that stay in the context cannot serve as calibration rows.

## 9. What the derivations say together

| Constraint | What it blocks | Tasks |
|---|---|---|
| M1: no mass beyond the buckets of the 0.001 and 0.999 quantiles | tail densities, extreme draws | outlier scores, density, synthetic rows |
| M2: classes come from the context | a class without a labeled row | one-class outlier detection, clustering |
| M4: queries are independent given the context; outputs are per row | joint distributions in one pass; evidence from the set of queries | forecast paths, effect intervals, joint draws of several cells |
| M5: a context row needs a target | unlabeled context | unlabeled rows, incomplete rows as context, clustering |
| M6: two task values and no column role | a second meaning of a regression table; a marked column | noiseless targets (a third task value); hidden confounding and a censoring indicator (a position convention or a new input) |

Impossible with the structure as it is, whatever the prior: predicting a class that the context does not hold; using a row without a target as context; returning a joint distribution over several queries or cells in one pass; serving ordinary regression and a noiseless-target task from one table; reconstructing masked cells.

By the kind of change:

- **Inference only**: the forecasting recipe without the clip and the trend outside the model; small random contexts for the outlier score; the class-frequency factor of the covering split; survival through binary rows; chains for several outputs; conformal intervals. None of these is measured, except the recipe on D1, together with two other changes.
- **Prior only**: temporal splits with trends and the context-row scale (forecasting, drift); reference rows with hidden outliers; query rows with a randomized treatment; feature masks that depend on the target (in `graph_scm_hicard_missing` already); self-masking query rows, with the limit of section 5.
- **Structure**: the masked-cell output; a value for "no target" in the context; a third task value; an input for a column's role.
- No task here needs a new loss with the outputs as they are, except clustering.

`target_train_rows` is a condition of every prior whose split is not random (sections 3 and 5). That is a reason from the derivation. Its measured effect on ordinary tables is not resolved (section 2).

## Checks and corrections

### Numerical checks

NumPy and SciPy only, no model weights, a few seconds each. The scripts are not part of the repository; the designs are stated so that each check can be redone.

| Section | What was reproduced | Design |
|---|---|---|
| 1 | the pinball minimizer is the quantile; the mean pinball loss over 999 levels against half the CRPS | five levels of a mixture of $\mathcal N(2, 0.5^2)$ and a $t_4$; a standard normal $F$ at five values of $y$ |
| 2 | the Student t predictive of the conjugate linear model; the flat-prior limit; the covering split | 40 rows, 3 features, 400,000 posterior draws; `train_rows_cover_classes` on 400,000 tables of Bernoulli labels with a Beta prior, 5 to 30 context rows and 1 to 12 queries |
| 3 | D1's seasonal naive and least-squares columns; the stand-in forecasters; a linear trend's scale; the median of a half-clipped mixture; the probabilities of step 5 | D1's generator and seeds; the stand-in as described in step 7 |
| 4 | the density score's AUROC against the share of a dense group; uniform outliers; the exact contaminated density on D3's tables; the binomial counts of small contexts | 20,000 inliers and outliers each; D3's generator with its draw order, the inlier density as a mean over 60,000 draws of its two factors |
| 5 | the self-masking means, variance and bias; the bias with a predictive column; the two feature-dependent masks | 4,000,000 draws; $a = -0.74, -0.6, -1.5$ with $b = 1, 0.5, 2$; correlations 0, 0.5, 0.9, 0.99 |
| 6 | the variance of the least-squares effect; the density ratios; the posterior of the effect against two predictive distributions | as stated in step 7 |
| 7, 8 | the chain rule along three random orders of a five-column Gaussian table; person-period frequencies against the hazards and the Kaplan–Meier product | 200,000 subjects, 8 bins |

### Review

Each derivation was read once by an agent that had not written it (three agents: sections 1, 2 and 7; sections 3 and 4; sections 5, 6 and 8), with the code and the diagnostic tables and without this document's conclusions restated. Their findings that were accepted, and what changed:

| Section | Finding | Change |
|---|---|---|
| 2 | The classifier is not trained toward the posterior predictive distribution: the prior's split puts a row of every class into the context first, so it depends on the labels | Section 2 rewritten around it; the factor $n_c/(n_c + 1)$ derived and checked on the prior's function; the Summary row split into regression and classification |
| 2 | Half of the inference members are given a power-transformed target, which the prior does not generate | Stated in step 6 as outside the derivation |
| 2 | The size of the scale term was misstated, and the all-row clip is a second channel | The term is now $1/(2 n_Q)$ of the loss, 0.07 to 0.5%; the clip is named; "measured" became "observed once" |
| 2 | The decoder's ability to return the posterior probability was asserted | Now "in principle, not demonstrated"; a single-class context is noted as outside training |
| 1 | Proposition 2(b)'s formula does not hold for a rule that draws a fixed number of rows by a score | Stated: exact for independent selection, direction only otherwise |
| 1 | Proposition 2(a) left its condition and the role of the unseen query rows unstated; Proposition 3 said "nothing" where the bound is finite and useless | Both stated |
| 1 | M5 to M7 were worded more strongly than the code supports; M6 ignored that column positions are an input and that the model has two task values | Reworded; "structure" for hidden confounding and for effect intervals became "a position convention or an input" and "a third task value" |
| 7 | The masked loss is the order-agnostic objective only with a uniform number of hidden cells and a mean over them; "changes nothing at the optimum" needs two conditions | Both stated; the claim of a likelihood bound removed |
| 3 | The stand-in forecaster's parameters were not given, so its table could not be reproduced; with a free length scale its agreement with the model is a fit | Parameters given, four length scales shown; "it is not a defect of the fit" withdrawn for "consistent with", and the saturating rank feature named as a second possible cause |
| 3 | The numbers for the trend fitted outside the model on non-linear trends depended on how the stand-in was set up | Removed; the linear case is kept as an estimate from the model's measured trend-free error |
| 3 | D1's `plain` arm changes three things, not the clip alone | Stated; the arm is no longer called "the recipe without the clip" |
| 3 | The cap holds only if half of the members transform the target; the quantile identity has a condition | Both stated (the default has 8 members, 4 of them transformed) |
| 3 | "Queries beyond the context's range never occur" and "the prior has no trend" were wrong as written | Replaced by the table of step 5 and by "the split is random, so the prior's linear mechanisms are never continued" |
| 3 | Under a temporal split the all-row scale is a leak of the future's level, not a small mismatch | Stated with $N/n_Q$ |
| 4 | The threshold $1.1 \cdot 10^{-5}$ was applied to D3, whose inliers are not a unit Gaussian | Now stated for the Gaussian case; D3's own crossing is given by seed |
| 4 | The exact-density numbers were computed on other test rows than D3's | Recomputed with the script's draw order: 0.455 at 2% where the draft had 0.42; the offset between the model's and the exact log density is reported |
| 4 | "Exactly when outliers are diffuse" and "follows from the threshold" overstated | The condition is now on $\Lambda$ as a function of $p_c$; small random contexts are labeled a hypothesis |
| 4 | Why the decoder could return the posterior outlier probability was not argued, and the reference rows' role was unclear | Argument added and marked as not demonstrated; the reference rows are said to carry no information |
| 5 | Rubin's MAR is not enough for a per-column fit: another column's mask may depend on the target | The condition is stated per column, with the reviewer's example (0.17 where the truth is 0.60), and the prior that has such masks is named |
| 5 | "The incomplete rows never reach the model" was wrong, and they could not identify the mechanism anyway | Rewritten |
| 6 | The density ratio had no reference measure; the variance statement was a bound, not a rate; the designs of two checks were not given; ACIC was left out | All four fixed |
| 8 | Survival through binary rows was called "inference" although such tables are outside the prior; the exchangeability sentence was wrong | Now "untested with these weights", with the caution; the sentence is replaced by the likelihood argument |
| 8 | Ranking by the expected relevance is right only for a linear gain; the conformal statement lacked its conditions | Both fixed |
| header | The status claimed a review that had not happened yet | Rewritten after the review |

Not changed:

- One reviewer read the two reference losses of section 2, step 6 (0.0633 and 0.0631) as two seeds of one arm. They are the control runs' loss under the two target scalings (v2-experiment-results, "Phase 1B"). The sentence was reworded to say so.
- Two reviewers asked for the check scripts to be added to the repository. This pull request is documents only; the designs are in the table above.
- A reviewer proposed to rescale the reference-row output by a formula that holds when the reference density equals the outlier density. That applies to today's weights, where reference rows give a density ratio, and not to a model trained on hidden outliers. The text now separates the two cases.

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
