"""Check for v2-task-formulations.md section 5, step 7: a mask of the column that depends on another
column (MAR on the predictor).

Design: (x1, x) standard bivariate normal with correlation rho = 0.6 (x = rho x1 + sqrt(1 - rho^2)
z); x is missing with probability Phi(-0.74 + 1.5 x1); x1 is always observed. 4,000,000 draws, seed
0.
The context is the complete cases (x observed), the rows to impute are those with x missing.

Claims reproduced (stated as "checked"):
  regression slope 0.600 and intercept 0.001 of x on x1 among the complete cases (the truth: 0.600,
  0);
  the rows to impute have a mean x1 of 0.90 against -0.46 in the context;
  32% of the rows to impute lie beyond the context's 99th percentile of x1.
Tolerances: 0.01 on the slope and intercept (absolute), 0.02 on the means, 0.02 on the share.
"""

import numpy as np
from scipy import stats

N = 4_000_000
rng = np.random.default_rng(0)
rho = 0.6
x1 = rng.normal(size=N)
x = rho * x1 + np.sqrt(1 - rho**2) * rng.normal(size=N)
miss = rng.random(N) < stats.norm.cdf(-0.74 + 1.5 * x1)
obs = ~miss

slope, intercept = np.polyfit(x1[obs], x[obs], 1)
mean_x1_missing = x1[miss].mean()
mean_x1_context = x1[obs].mean()
q99 = np.quantile(x1[obs], 0.99)
share_beyond = np.mean(x1[miss] > q99)
print(f"missing rate {miss.mean():.4f}")
print(
    f"complete-case regression of x on x1: slope {slope:.4f} (stated 0.600), "
    f"intercept {intercept:+.4f} (stated 0.001)"
)
print(
    f"mean x1 of rows to impute {mean_x1_missing:+.4f} (stated 0.90); "
    f"context {mean_x1_context:+.4f} (stated -0.46)"
)
print(
    f"share of rows to impute beyond the context's 99th percentile of x1 ({q99:.3f}): "
    f"{share_beyond:.4f} (stated 0.32)"
)

assert abs(slope - 0.600) < 0.01, slope
assert abs(intercept - 0.001) < 0.01, intercept
assert abs(mean_x1_missing - 0.90) < 0.02, mean_x1_missing
assert abs(mean_x1_context - (-0.46)) < 0.02, mean_x1_context
assert abs(share_beyond - 0.32) < 0.02, share_beyond
print("PASS s05b")
