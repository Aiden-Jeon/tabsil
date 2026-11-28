"""Check for v2-task-formulations.md section 2, step 7 (regression analysis, first paragraph).

Claim (stated in the document as "Checked: 400,000 posterior draws against the formula, quantiles
at 0.01 to 0.99 within 0.007"): for the conjugate linear model with a normal-inverse-gamma prior,
the posterior predictive distribution of a new target is Student t with 2 a_n degrees of freedom,
location x*' m_n and squared scale (b_n / a_n)(1 + x*' Lambda_n^{-1} x*).

Design: n = 40 rows, d = 3 features, X ~ N(0, 1), seed 0; y = X beta + N(0, 1) noise with
beta = (1, -2, 0.5); prior m0 = 0, Lambda0 = I, a0 = 2, b0 = 1. One query x* = (1, 0.5, -1).
400,000 posterior draws: sigma^2 ~ IG(a_n, b_n) (precision ~ Gamma(a_n, rate b_n)), then
beta | sigma^2 ~ N(m_n, sigma^2 Lambda_n^{-1}), then y* = x*' beta + sigma * eps.
The draws' quantiles at levels 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99 are compared with
the t quantiles.

The document's "within 0.007" was the largest difference of the run it reported; that run's design
and seed were not recorded. In this design, seed 0 gives 0.017 at the 0.99 level and four other
seeds give 0.004 to 0.008, so 0.007 is not a bound. The check therefore asserts what Monte Carlo can
support: at every level the difference is within 4 standard errors of a sample quantile, sqrt(p (1 -
p) / N) / f(q_p), with f the t density of the formula. The document's sentence was corrected to this
(v2-task-formulations.md, "Checks as scripts").
"""

import numpy as np
from scipy import stats

rng = np.random.default_rng(0)
n, d = 40, 3
X = rng.normal(size=(n, d))
y = X @ np.array([1.0, -2.0, 0.5]) + rng.normal(size=n)
m0, L0, a0, b0 = np.zeros(d), np.eye(d), 2.0, 1.0

Ln = L0 + X.T @ X
mn = np.linalg.solve(Ln, L0 @ m0 + X.T @ y)
an = a0 + n / 2
bn = b0 + 0.5 * (y @ y + m0 @ L0 @ m0 - mn @ Ln @ mn)
xs = np.array([1.0, 0.5, -1.0])
loc = xs @ mn
scale = np.sqrt(bn / an * (1 + xs @ np.linalg.solve(Ln, xs)))
df = 2 * an

N = 400_000
prec = rng.gamma(shape=an, scale=1.0 / bn, size=N)  # precision = 1 / sigma^2
sig = 1.0 / np.sqrt(prec)
Lchol = np.linalg.cholesky(np.linalg.inv(Ln))
beta = mn + (rng.normal(size=(N, d)) @ Lchol.T) * sig[:, None]
ystar = beta @ xs + sig * rng.normal(size=N)

levels = np.array([0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
emp = np.quantile(ystar, levels)
theo = loc + scale * stats.t.ppf(levels, df)
err = np.abs(emp - theo)
for p, e, t in zip(levels, emp, theo, strict=True):
    print(f"level {p:.2f}: draws {e:+.4f}  t formula {t:+.4f}  diff {e - t:+.4f}")
print(f"df = {df}, loc = {loc:.4f}, scale = {scale:.4f}")
dens = stats.t.pdf(stats.t.ppf(levels, df), df) / scale
se = np.sqrt(levels * (1 - levels) / N) / dens
z = (emp - theo) / se
print(f"max abs quantile difference = {err.max():.4f} (the document's run: 0.007)")
print("difference in Monte Carlo standard errors: " + ", ".join(f"{v:+.1f}" for v in z))
# Monte Carlo noise of the comparison: the same check with four other seeds (information only).
for seed in range(1, 5):
    r2 = np.random.default_rng(seed)
    pr = r2.gamma(shape=an, scale=1.0 / bn, size=N)
    s2 = 1.0 / np.sqrt(pr)
    b2 = mn + (r2.normal(size=(N, d)) @ Lchol.T) * s2[:, None]
    y2 = b2 @ xs + s2 * r2.normal(size=N)
    print(
        f"  seed {seed}: max abs quantile difference = "
        f"{np.abs(np.quantile(y2, levels) - theo).max():.4f}"
    )
assert np.abs(z).max() < 4, z
print("PASS s02a")
