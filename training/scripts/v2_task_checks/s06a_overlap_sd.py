"""Check for v2-task-formulations.md section 6, step 7 (the least-squares effect's sd as the
propensity clip shrinks).

Design: x ~ N(0, 1); e(x) = Phi(2.5 x) clipped to [c, 1 - c]; t ~ Bernoulli(e(x)); y = 0.5 x + t +
eps, eps ~ N(0, 1); n = 2,000; 400 replications (seeds 0..399 for the data). Clips c = 0.5, 0.1,
0.02.
Estimators of the effect (the true effect is 1):
  LS: the coefficient of t in an OLS fit of y on (1, x, t);
  T-learner: two OLS fits of y on (1, x), one per arm; the effect is the mean over all rows of
             mu1(x_i) - mu0(x_i).
Formula: sd = sigma / sqrt(sum_i (t_i - t_hat_i)^2) with t_hat the linear projection of t on (1, x);
the sum is taken as n E[(t - t_hat(x))^2], with the expectation from 4,000,000 draws.

The closed document (step 7) stated: LS sd 0.041 -> 0.062 (simulated), formula 0.045 and 0.063,
T-learner 0.060 -> 0.088. Its design is the one above, and these replications give LS 0.046 ->
0.066, formula 0.045 and 0.065, and T-learner 0.046 -> 0.066 (also the effect at x = 0; at x = 1.5
0.084 and 0.118; the root mean squared error of the effect over the rows 0.057 and 0.081). The
T-learner numbers were not reproducible by any reading, and the stated LS sd at clip 0.5 is 3
standard errors from the formula. The document's sentence was corrected to the numbers of this
script (v2-task-formulations.md, "Checks as scripts"); the claim that both estimates grow at the
same rate holds (1.45 for both). Tolerance: 0.003 absolute on each sd (about 2 Monte Carlo standard
errors at 400 replications).
"""

import numpy as np
from scipy import stats

n, reps = 2000, 400
clips = (0.5, 0.1, 0.02)


def propensity(x, c):
    return np.clip(stats.norm.cdf(2.5 * x), c, 1 - c)


def formula_sd(c):
    rng = np.random.default_rng(12345)
    N = 4_000_000
    x = rng.normal(size=N)
    t = (rng.random(N) < propensity(x, c)).astype(float)
    b = np.cov(t, x)[0, 1] / np.var(x)
    a = t.mean() - b * x.mean()
    resid = t - (a + b * x)
    return 1 / np.sqrt(n * np.mean(resid**2))


def simulate(c):
    ls, tl = [], []
    for r in range(reps):
        rng = np.random.default_rng(r)
        x = rng.normal(size=n)
        t = (rng.random(n) < propensity(x, c)).astype(float)
        y = 0.5 * x + t + rng.normal(size=n)
        D = np.column_stack([np.ones(n), x, t])
        ls.append(np.linalg.lstsq(D, y, rcond=None)[0][2])
        pred = []
        for arm in (0.0, 1.0):
            m = t == arm
            A = np.column_stack([np.ones(m.sum()), x[m]])
            coef = np.linalg.lstsq(A, y[m], rcond=None)[0]
            pred.append(coef[0] + coef[1] * x)
        tl.append(np.mean(pred[1] - pred[0]))
    return np.std(ls, ddof=1), np.std(tl, ddof=1)


stated = {0.5: (0.046, 0.045, 0.046), 0.02: (0.066, 0.065, 0.066)}  # corrected statement
ok = True
for c in clips:
    sd_ls, sd_t = simulate(c)
    f = formula_sd(c)
    line = f"clip {c:<5}: LS sd {sd_ls:.4f}, formula {f:.4f}, T-learner sd {sd_t:.4f}"
    if c in stated:
        s_ls, s_f, s_t = stated[c]
        line += f"   (stated LS {s_ls}, formula {s_f}, T-learner {s_t})"
        ok &= abs(sd_ls - s_ls) < 0.003 and abs(f - s_f) < 0.003 and abs(sd_t - s_t) < 0.003
        print(line)
        print(
            f"   check: |LS - stated| {abs(sd_ls - s_ls):.4f}, "
            f"|formula - stated| {abs(f - s_f):.4f}, "
            f"|T-learner - stated| {abs(sd_t - s_t):.4f}"
        )
    else:
        print(line)
assert ok, "a stated sd is not reproduced within 0.003 (see the check lines above)"
print("PASS s06a")
