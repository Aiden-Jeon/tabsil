"""Check for v2-task-formulations.md section 3, step 6 (the scale numbers of a temporal split).

Claims reproduced, on D1's linear series with trend strength 15 (noise sd 1, L = 365 context rows,
H = 56 horizon rows; generator copied from training/scripts/v2_prior_diag/d1_forecast.py):
  (a) the context targets, standardized over all N = 421 rows as the v1 prior does, have mean -0.23
      and standard deviation 0.87 in all-row units (stated "checked");
  (b) N / n_Q = 421 / 56 = 7.5;
  (c) in context units a linear trend puts the query targets at sqrt(12) (1/2 + h/T), h = 1..56,
      T = 365: between 1.74 and 2.26 (stated).

Design: seeds 0..3 (as D1); the mean and sd of the context targets are printed per seed and their
average over seeds is compared with the stated values. For (c) the formula is evaluated and also
the discrete standardization of the noiseless trend over the context rows (mean and sd of 0..364).
Tolerance: 0.02 for (a) on the seed average, 0.01 for (c).
"""

import numpy as np

L, H = 365, 56
N = L + H


def series(kind, strength, rng):
    t = np.arange(L + H, dtype=float)
    noise_sd = 1.0
    season = 2.0 * np.sin(2 * np.pi * t / 7) + 1.5 * np.sin(2 * np.pi * t / 30.4 + 1.0)
    trend = strength * noise_sd * t / H  # kind == "linear"
    return 10 + trend + season + rng.normal(0, noise_sd, L + H)


means, sds = [], []
for seed in range(4):
    y = series("linear", 15.0, np.random.default_rng(seed))
    z = (y - y.mean()) / y.std()  # all N rows, population sd as in the prior
    ctx = z[:L]
    means.append(ctx.mean())
    sds.append(ctx.std())
    print(f"seed {seed}: context z mean {ctx.mean():+.4f}, sd {ctx.std():.4f}")
m_avg, s_avg = np.mean(means), np.mean(sds)
print(f"seed average: mean {m_avg:+.4f} (stated -0.23), sd {s_avg:.4f} (stated 0.87)")
print(f"N / n_Q = {N / H:.3f} (stated 7.5)")

h = np.arange(1, H + 1)
z_formula = np.sqrt(12) * (0.5 + h / L)
print(
    f"sqrt(12)(1/2 + h/T), h = 1..56: from {z_formula[0]:.4f} to {z_formula[-1]:.4f} "
    "(stated 1.74 to 2.26)"
)
t_ctx = np.arange(L, dtype=float) * 15.0 / H  # noiseless trend over the context rows
tr_all = np.arange(L + H, dtype=float) * 15.0 / H
z_disc = (tr_all[L:] - t_ctx.mean()) / t_ctx.std()
print(
    f"discrete standardization of the trend over the context: {z_disc[0]:.4f} to {z_disc[-1]:.4f}"
)

assert abs(m_avg - (-0.23)) < 0.02, m_avg
assert abs(s_avg - 0.87) < 0.02, s_avg
assert abs(N / H - 7.5) < 0.05
assert abs(z_formula[0] - 1.74) < 0.01 and abs(z_formula[-1] - 2.26) < 0.01
print("PASS s03c")
