"""Check for v2-task-formulations.md section 8 (ranking and split conformal intervals).

Claims reproduced:
  (a) ranking: with gain 2^y - 1, a sure relevance 1 and a 50/50 relevance 0 or 2 have the same
      mean relevance (1) and expected gains 1 and 1.5 (exact arithmetic, printed).
  (b) split conformal: with m calibration rows and the quantile of order ceil((1 - alpha)(m + 1))/m
      of the calibration scores, coverage is at least 1 - alpha. Checked by simulation with a
      predictor fitted on an independent training set (so the calibration rows are exchangeable with
      the test rows given the predictor), heavy-tailed noise t_3, alpha = 0.1, m in {19, 99, 499},
      400 replications of 1,000 test rows each (seed 0).
Tolerance for (b): the average coverage must be at least 1 - alpha - 0.003 (Monte Carlo error).
"""

import math

import numpy as np


def gain(y):
    return 2.0**y - 1.0


print(f"sure relevance 1: gain {gain(1):.1f}, mean relevance 1.0")
e_rel = 0.5 * 0 + 0.5 * 2
e_gain = 0.5 * gain(0) + 0.5 * gain(2)
print(
    f"50/50 relevance 0 or 2: mean relevance {e_rel:.1f}, expected gain {e_gain:.1f} "
    "(stated 1 and 1.5 for the sure one and this one)"
)
assert gain(1) == 1.0 and e_gain == 1.5 and e_rel == 1.0

rng = np.random.default_rng(0)
alpha = 0.1
for m in (19, 99, 499):
    k = math.ceil((1 - alpha) * (m + 1))
    cov_list = []
    for _ in range(400):
        # predictor fitted on an independent training set (linear, one feature)
        xtr = rng.normal(size=100)
        ytr = 2 * xtr + rng.standard_t(3, size=100)
        beta = np.polyfit(xtr, ytr, 1)

        def f(z, beta=beta):
            return np.polyval(beta, z)

        xc = rng.normal(size=m)
        yc = 2 * xc + rng.standard_t(3, size=m)
        scores = np.sort(np.abs(yc - f(xc)))
        q = scores[k - 1] if k <= m else np.inf
        xt = rng.normal(size=1000)
        yt = 2 * xt + rng.standard_t(3, size=1000)
        cov_list.append(np.mean(np.abs(yt - f(xt)) <= q))
    cov = float(np.mean(cov_list))
    se = float(np.std(cov_list, ddof=1) / np.sqrt(len(cov_list)))
    print(
        f"m = {m:3d}: quantile index {k}/{m}; average coverage {cov:.4f} (s.e. {se:.4f}); "
        f"required >= {1 - alpha}"
    )
    assert cov >= 1 - alpha - 0.003, (m, cov)
print("PASS s07b")
