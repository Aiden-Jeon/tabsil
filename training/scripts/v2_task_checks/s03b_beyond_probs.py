"""Check for v2-task-formulations.md section 3, step 5 (the probability table).

Claims reproduced (the table of step 5, for a feature that is uniform like the running index):
  row 1: a given query lies beyond the context's largest value with probability 1/(n+1);
         D1 (n = 365): 0.3%.
  row 2: it lies beyond by a share delta of the feature's range with probability
         (1 - delta)^(n+1)/(n+1); D1's last step has delta = 56/365 = 0.15 and gives 1e-29.
  row 3: every query beyond the context's largest value has probability 1/C(N, n_Q);
         D1 (N = 421, n_Q = 56): below 1e-70.
Plus the same formulas checked by simulation at small sizes (the probabilities at D1's sizes are
far below simulation reach; the formulas are exact for continuous exchangeable values).

Design: row 1 by simulation of 2,000,000 tables of n = 20 uniform context values and one query; row
2 by simulation with n = 5 and delta = 0.2 (2,000,000 draws); row 3 by simulation with N = 6, n_Q =
2. Stated tolerances: 0.002 for the simulated probabilities (absolute). The D1 values are exact
evaluations of the formulas (log10 via lgamma).
"""

import math

import numpy as np

rng = np.random.default_rng(0)
R = 2_000_000

# row 1 (n = 20)
n = 20
ctx = rng.random((R, n))
q = rng.random(R)
p1_sim = np.mean(q > ctx.max(axis=1))
p1_formula = 1 / (n + 1)

# row 2 (n = 5, delta = 0.2)
n2, delta = 5, 0.2
ctx2 = rng.random((R, n2))
q2 = rng.random(R)
p2_sim = np.mean(q2 - ctx2.max(axis=1) > delta)
p2_formula = (1 - delta) ** (n2 + 1) / (n2 + 1)

# row 3 (N = 6 rows, n_Q = 2 random queries, 4 context rows): every query above every context value
N3, nq3 = 6, 2
R3 = R // 4
vals = rng.random((R3, N3))
perm = np.argsort(rng.random((R3, N3)), axis=1)  # a uniformly random split of the N rows
is_query = np.zeros((R3, N3), dtype=bool)
is_query[np.arange(R3)[:, None], perm[:, :nq3]] = True
q_min = np.where(is_query, vals, np.inf).min(axis=1)
c_max = np.where(is_query, -np.inf, vals).max(axis=1)
p3_sim = np.mean(q_min > c_max)
p3_formula = 1 / math.comb(N3, nq3)

print(
    f"row 1: simulated {p1_sim:.5f}, formula 1/(n+1) = {p1_formula:.5f}; "
    f"D1 value 1/366 = {1 / 366:.4f} (stated 0.3%)"
)
print(f"row 2: simulated {p2_sim:.5f}, formula {p2_formula:.5f}")
dl = 56 / 365
log10_row2 = (366 * math.log10(1 - dl)) - math.log10(366)
print(f"row 2 at D1 (n=365, delta=56/365={dl:.4f}): 10^{log10_row2:.2f} (stated 1e-29)")
N_D1, nq_D1 = 365 + 56, 56
log10_row3 = -(
    math.lgamma(N_D1 + 1) - math.lgamma(nq_D1 + 1) - math.lgamma(N_D1 - nq_D1 + 1)
) / math.log(10)
print(f"row 3 at D1 (N=421, n_Q=56): 10^{log10_row3:.1f} (stated: below 1e-70)")
print(f"row 3 small case: simulated {p3_sim:.5f}, formula 1/C(6,2) = {p3_formula:.5f}")

assert abs(p1_sim - p1_formula) < 0.002
assert abs(p2_sim - p2_formula) < 0.002
assert abs(p3_sim - p3_formula) < 0.002
assert abs(log10_row2 - (-29)) < 0.5, log10_row2
assert log10_row3 < -70
print("PASS s03b")
