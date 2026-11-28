"""Check for v2-task-formulations.md section 5, step 7: the MAR counter-example for a per-column
fit.

Design (the reading that reproduces the stated numbers): (x1, x2) standard bivariate normal with
correlation 0.6 (so E[x1 | x2 = 1] = 0.6, the stated truth). Column 1 (x1) is missing completely at
random with probability 0.3 (the column to impute). Column 2 (x2) is missing with probability
Phi(-0.5 + x1) in the rows that observe column 1 (MAR: it depends only on an observed value).
The context for imputing column 1 is the rows that observe column 1 and column 2 (the rows that show
column 2 in the stated sentence).

Claims reproduced:
  the truth E[x1 | x2 = 1] = 0.60;
  the context rows that show column 2, at x2 = 1, have a mean x1 of 0.17 (exact by quadrature, and
    by a window |x2 - 1| < 0.05 in simulation);
  their regression of x1 on x2 has slope 0.48 and intercept -0.30 (simulation, 4,000,000 draws).
Tolerances: 0.01 on the truth, 0.02 on the window mean and 0.01 on the quadrature, 0.02 on slope and
intercept.
"""

import numpy as np
from scipy import integrate, stats

rho = 0.6
N = 4_000_000
rng = np.random.default_rng(0)
x1 = rng.normal(size=N)
x2 = rho * x1 + np.sqrt(1 - rho**2) * rng.normal(size=N)
obs1 = rng.random(N) >= 0.3  # column 1 observed (MCAR 30% missing)
obs2 = obs1 & (rng.random(N) >= stats.norm.cdf(-0.5 + x1))  # column 2 observed given column 1

truth = rho * 1.0
ctx = obs1 & obs2
win = ctx & (np.abs(x2 - 1.0) < 0.05)
mean_win = x1[win].mean()
slope, intercept = np.polyfit(x2[ctx], x1[ctx], 1)

# quadrature for the same mean: x1 | x2 = 1 is N(0.6, 1 - 0.36); weight = P(column 2 observed | x1)
cond_mean, cond_sd = rho, np.sqrt(1 - rho**2)


def pdf(u):
    return stats.norm.pdf(u, cond_mean, cond_sd)


def w(u):
    return stats.norm.cdf(0.5 - u)  # 1 - Phi(-0.5 + u)


num, _ = integrate.quad(lambda u: u * w(u) * pdf(u), -12, 12)
den, _ = integrate.quad(lambda u: w(u) * pdf(u), -12, 12)
quad_mean = num / den

print(f"truth E[x1 | x2 = 1] = {truth:.3f} (stated 0.60)")
print(f"context mean x1 at x2 = 1: window {mean_win:.4f}, quadrature {quad_mean:.4f} (stated 0.17)")
print(
    f"context regression of x1 on x2: slope {slope:.4f} (stated 0.48), "
    f"intercept {intercept:+.4f} (stated -0.30)"
)
print(
    f"the estimate at x2 = 1 from the context regression: {intercept + slope:+.4f} "
    f"against the truth {truth:.2f}"
)

assert abs(truth - 0.60) < 0.01
assert (
    abs(mean_win - 0.17) < 0.02 and abs(quad_mean - 0.17) < 0.01
)  # quadrature 0.176: rounds to 0.18
assert abs(slope - 0.48) < 0.02, slope
assert abs(intercept - (-0.30)) < 0.02, intercept
print("PASS s05c")
