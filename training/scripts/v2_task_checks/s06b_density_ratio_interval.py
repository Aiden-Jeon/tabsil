"""Check for v2-task-formulations.md section 6, steps 6 and 7: the density ratio of the S-learner's
queries under low overlap, and the posterior sd of the effect against two predictive distributions.

Part 1 (density ratio). With e(x) = Phi(2.5 x) clipped to [c, 1-c] and P(t = 0) = 1/2:
  w(x) = p(x | t = 0) / p(x, t = 1) = (1 - e(x)) / (e(x) P(t = 0)), whose supremum is at e = c:
  18 at c = 0.1 and 98 at c = 0.02 (stated). If the queries' treatment were drawn with probability
  1/2 independently of x, w(x) = (1 - e(x)) / (1/2 * P(t = 0)), supremum (1 - c)/0.25:
  3.6 and 3.9 (stated). Computed on a grid of x with the density of x.

Part 2 (interval). Bayesian linear model y = a + b x + t (g + d x) + eps, eps ~ N(0, 1), with
x ~ N(0, 1), t ~ Bernoulli(Phi(1.5 x)), n = 200, a N(0, 10^2) prior on the four coefficients.
The design is drawn with seed 0; the posterior of (a, b, g, d) is Gaussian with precision
X'X + I/100 (sigma = 1 known). tau(x) = g + d x.
Stated: posterior sd of tau(x) 0.19 at x = 0 and 0.34 at x = 1.5; two predictive distributions
treated as independent give 1.43 and 1.45 (sd of the difference of the two outcome predictions).
Tolerance: 0.02 on each stated number (the posterior sd depends on the drawn design).
"""

import numpy as np
from scipy import stats

# Part 1
grid = np.linspace(-6, 6, 200001)
dens = stats.norm.pdf(grid)
for c, stated in ((0.1, 18), (0.02, 98)):
    e = np.clip(stats.norm.cdf(2.5 * grid), c, 1 - c)
    P0 = np.sum((1 - e) * dens) * (grid[1] - grid[0])
    w = (1 - e) / (e * P0)
    print(f"clip {c}: P(t=0) = {P0:.4f}, sup of w = {w.max():.2f} (stated {stated})")
    assert abs(w.max() - stated) / stated < 0.01
for c, stated in ((0.1, 3.6), (0.02, 3.9)):
    e = np.clip(stats.norm.cdf(2.5 * grid), c, 1 - c)
    P0 = 0.5
    w = (1 - e) / (0.5 * P0)
    print(f"randomized queries, clip {c}: sup of w = {w.max():.2f} (stated {stated})")
    assert abs(w.max() - stated) < 0.06

# Part 2
rng = np.random.default_rng(0)
n = 200
x = rng.normal(size=n)
t = (rng.random(n) < stats.norm.cdf(1.5 * x)).astype(float)
X = np.column_stack([np.ones(n), x, t, t * x])
V = np.linalg.inv(X.T @ X + np.eye(4) / 100.0)  # posterior covariance (sigma = 1)
res = {}
for xs, stated in ((0.0, 0.19), (1.5, 0.34)):
    cvec = np.array([0.0, 0.0, 1.0, xs])  # tau(x) = g + d x
    sd_tau = np.sqrt(cvec @ V @ cvec)
    c1 = np.array([1.0, xs, 1.0, xs])
    c0 = np.array([1.0, xs, 0.0, 0.0])
    sd_y1 = np.sqrt(c1 @ V @ c1 + 1.0)  # predictive sd of y under treatment
    sd_y0 = np.sqrt(c0 @ V @ c0 + 1.0)
    diff_sd = np.sqrt(sd_y1**2 + sd_y0**2)
    res[xs] = (sd_tau, diff_sd)
    print(
        f"x = {xs}: posterior sd of tau(x) {sd_tau:.4f} (stated {stated}); "
        f"two predictive distributions as independent: sd of difference {diff_sd:.4f}"
    )
for xs, stated in ((0.0, 1.43), (1.5, 1.45)):
    assert abs(res[xs][1] - stated) < 0.02, (xs, res[xs][1])
assert abs(res[0.0][0] - 0.19) < 0.02 and abs(res[1.5][0] - 0.34) < 0.02
print("PASS s06b")
