"""Check for v2-task-formulations.md section 3, step 7 ("The default recipe caps the median").

Claims reproduced:
  (a) the median of an equal-weight mixture of 8 members, 4 of which are clipped at the context
      maximum m (all their mass at or below m), is at most m: F(m) = 1/2 + (1/2) F_u(m) >= 1/2;
  (b) where the other members' quantile at level 2u - 1 is above m, the mixture's quantile at level
      u (u > 1/2) is that quantile (the "checked" sentence of the same paragraph).

Design: 20,000 random cases; members are discrete distributions on a grid of 300 values in [-10, 10]
(random Dirichlet weights, seed 0). Unclipped members: any atoms. Clipped members: their atoms above
m are moved to m, so they have all their mass at or below m. m is a random grid value. The
median and quantiles are exact on the grid (smallest grid point whose CDF reaches the level).
Stated: no case with F(m) < 1/2 (tolerance 0); identity (b) holds in every case where it applies.
"""

import numpy as np

rng = np.random.default_rng(0)
grid = np.linspace(-10, 10, 300)
cases = 20_000
bad_median = 0
bad_quant = 0
applicable = 0
min_fm = 1.0

for _ in range(cases):
    members = []
    m_idx = rng.integers(20, 280)
    m = grid[m_idx]
    for j in range(8):
        w = rng.dirichlet(np.full(300, 0.2))
        if j < 4:  # clipped members: mass above m moved to m
            w = np.concatenate([w[:m_idx], [w[m_idx:].sum()], np.zeros(299 - m_idx)])
        members.append(w)
    P = np.mean(members, axis=0)
    F = np.cumsum(P)
    Fm = F[m_idx]
    min_fm = min(min_fm, Fm)
    if Fm < 0.5 - 1e-12:
        bad_median += 1
    # quantiles of the mixture and of the unclipped members (average of the four)
    Fu_cdf = np.cumsum(np.mean(members[4:], axis=0))
    for u in rng.uniform(0.5, 0.999, size=5):
        q_mix = grid[np.searchsorted(F, u)]
        target = 2 * u - 1
        i_u = np.searchsorted(Fu_cdf, target)
        if i_u < 300 and grid[i_u] > m:
            applicable += 1
            if abs(q_mix - grid[i_u]) > 1e-12:
                bad_quant += 1

print(f"cases {cases}: min F(m) over cases = {min_fm:.6f} (claim: >= 0.5)")
print(f"cases with median above m: {bad_median} (stated 0)")
print(
    f"upper-quantile identity applicable in {applicable} (u, case) pairs, "
    f"violations {bad_quant} (stated 0)"
)
assert bad_median == 0
assert bad_quant == 0
assert applicable > 1000
print("PASS s03a")
