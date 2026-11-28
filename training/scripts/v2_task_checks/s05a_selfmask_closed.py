"""Check for v2-task-formulations.md section 5, step 7 (self-masking closed forms and the predictive
column).

Model: x ~ N(0, 1) is missing (M = 1) with probability Phi(a + b x), a = -0.74.

Claims reproduced (the "Self-masking" paragraph):
  (a) the gap of means E[x|M=0] - E[x|M=1] = -kappa phi(alpha)/(Phi(alpha)(1 - Phi(alpha))), with
      alpha = a/sqrt(1+b^2), kappa = b/sqrt(1+b^2): -1.17 at b = 1 (stated), also b = 0.5 and 2;
  (b) at a missing rate of 30%, b = 1: the error of imputing the complete-case mean is 1.42
      (stated), against 0.80 for imputing E[x|M=1] (= the conditional standard deviation of x given
      M=1);
  (c) with a predictive column of correlation rho (x1 = 0): the gap -1.17, -0.94, -0.30, -0.03 at
      rho = 0, 0.5, 0.9, 0.99 (stated), by the closed form with s^2 = 1 - rho^2 and Monte Carlo.

Design: 4,000,000 draws (seed 0). The Monte Carlo conditional means at x1 = 0 use a window
|x1| < 0.05. Tolerances: 0.01 on the closed-form gap, 0.02 on the imputation error (stated to two
decimals), 0.03 on the Monte Carlo window gap.
"""

import numpy as np
from scipy import stats

Phi, phi = stats.norm.cdf, stats.norm.pdf
N = 4_000_000


def gap(a, b, s=1.0):
    """E[z|M=0] - E[z|M=1] for z = s * N(0,1) with missingness Phi(a + b z)."""
    bb = b * s
    alpha = a / np.sqrt(1 + bb**2)
    kappa = bb / np.sqrt(1 + bb**2)
    return -s * kappa * phi(alpha) / (Phi(alpha) * (1 - Phi(alpha)))


a = -0.74
b = 1.0
al = a / np.sqrt(1 + b**2)
print(f"missing rate Phi(a/sqrt(1+b^2)) = {Phi(al):.4f} (30% is the design target)")
for bb, stated in ((1.0, -1.17), (0.5, None), (2.0, None)):
    g = gap(a, bb)
    xb = np.random.default_rng(1).normal(size=N)
    mb = np.random.default_rng(2).random(N) < Phi(a + bb * xb)
    mc_b = xb[~mb].mean() - xb[mb].mean()
    print(
        f"b={bb}: closed-form gap {g:+.4f} (Monte Carlo {mc_b:+.4f})"
        + (f" (stated {stated})" if stated else "")
    )
    assert abs(mc_b - g) < 0.01, (bb, mc_b, g)

# (b) imputation errors at b = 1 (use the closed forms and check by simulation)
rng = np.random.default_rng(0)
x = rng.normal(size=N)
M = rng.random(N) < Phi(a + b * x)
mu_c = x[~M].mean()
mu_m = x[M].mean()
err_cc = np.sqrt(np.mean((x[M] - mu_c) ** 2))
err_best = np.sqrt(np.mean((x[M] - mu_m) ** 2))
print(
    f"missing rate (MC) {M.mean():.4f}; E[x|M=0] {mu_c:+.4f}, E[x|M=1] {mu_m:+.4f}, "
    f"MC gap {mu_c - mu_m:+.4f}"
)
print(
    f"RMSE imputing the complete-case mean {err_cc:.4f} (stated 1.42); "
    f"RMSE imputing E[x|M=1] {err_best:.4f} (stated 0.80)"
)
print(f"closed-form gap at b=1 {gap(a, 1.0):+.4f}")

# (c) predictive column, x1 = 0; x = rho x1 + s z; bias = E[x|x1=0, M=0] - E[x|x1=0, M=1]
rhos = (0.0, 0.5, 0.9, 0.99)
stated_c = (-1.17, -0.94, -0.30, -0.03)
x1 = rng.normal(size=N)
ok_all = True
for rho, want in zip(rhos, stated_c, strict=True):
    s = np.sqrt(1 - rho**2)
    z = rng.normal(size=N)
    xx = rho * x1 + s * z
    mm = rng.random(N) < Phi(a + b * xx)
    # the closed form at x1 = 0: the self-masking gap of the residual with the shifted intercept
    cf = gap(a + b * rho * 0.0, b, s)
    win = np.abs(x1) < 0.05
    mc = xx[win & ~mm].mean() - xx[win & mm].mean()
    ok_all &= abs(cf - want) < 0.01
    print(
        f"rho={rho:.2f}: closed form {cf:+.4f} (stated {want:+.2f}); Monte Carlo at x1=0 {mc:+.4f}"
    )
    assert abs(mc - cf) < 0.03, (rho, mc, cf)

assert abs(gap(a, 1.0) - (-1.17)) < 0.01
assert abs(err_cc - 1.42) < 0.02, err_cc
assert abs(err_best - 0.80) < 0.01, err_best
assert ok_all
print("PASS s05a")
