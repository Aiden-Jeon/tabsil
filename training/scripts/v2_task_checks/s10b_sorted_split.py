"""Check for v2-task-formulations.md section 10, step 3 (sorted-split recognizability).

Claim reproduced: under exchangeable rows, a query lies beyond the context in some column with
probability at most 2d/(n+1) (4% at n = 1,000 and d = 20; checked). "Beyond" = above the context's
maximum or below its minimum in that column.

Design: n context rows and one query row (n + 1 exchangeable rows) with d columns. Two designs:
  (a) independent standard normal columns: the exact probability 1 - (1 - 2/(n+1))^d, compared with
      the bound 2d/(n+1) and with simulation;
  (b) equicorrelated columns (correlation 0.5), where the union bound must still hold.
Simulation: 4,000 tables per design (seed 0) at n = 1,000, d = 20, and at n = 100 for a second size.
Stated: 4% at n = 1,000, d = 20 (bound 0.0400, exact independent 0.0392).
Tolerance: simulated probability at most 2d/(n+1) + 0.01 (Monte Carlo error about 0.003).
"""

import numpy as np

rng = np.random.default_rng(0)


def p_beyond(n, d, corr, reps):
    cov = np.full((d, d), corr)
    np.fill_diagonal(cov, 1.0)
    L = np.linalg.cholesky(cov)
    hits = 0
    for _ in range(reps):
        Z = rng.normal(size=(n + 1, d)) @ L.T  # rows exchangeable: the query is a uniform row
        q = Z[rng.integers(n + 1)]
        ctx = np.delete(Z, np.where(np.all(Z == q, axis=1))[0][0], axis=0)
        beyond = (q > ctx.max(axis=0)) | (q < ctx.min(axis=0))
        hits += bool(beyond.any())
    return hits / reps


n, d = 1000, 20
bound = 2 * d / (n + 1)
exact_indep = 1 - (1 - 2 / (n + 1)) ** d
print(
    f"n = {n}, d = {d}: bound 2d/(n+1) = {bound:.4f} (stated 4%), "
    f"exact for independent columns {exact_indep:.4f}"
)
sim_a = p_beyond(n, d, 0.0, 4000)
sim_b = p_beyond(n, d, 0.5, 4000)
print(f"simulated: independent columns {sim_a:.4f}, equicorrelated 0.5 {sim_b:.4f}")
sim_small = p_beyond(100, d, 0.0, 4000)
print(f"n = 100: bound {2 * d / 101:.4f}, simulated independent {sim_small:.4f}")

assert sim_a <= bound + 0.01
assert sim_b <= bound + 0.01
assert sim_small <= 2 * d / 101 + 0.01
assert abs(bound - 0.04) < 0.001
print("PASS s10b")
