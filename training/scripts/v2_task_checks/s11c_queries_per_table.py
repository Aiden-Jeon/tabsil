"""Check for v2-task-formulations.md section 11.3: queries per table under two-stage sampling
(Cochran 1977).

Per-query gradient g_ij = mu + b_i + w_ij, b_i ~ N(0, sb2), w_ij ~ N(0, sw2), independent. B tables,
m queries each: Var(mean) = (sb2 + sw2/m)/B. Cost per table c_ctx n + c_q m; fixed budget, so
B = budget / (c_ctx n + c_q m) and the variance is (sb2 + sw2/m)(c_ctx n + c_q m)/budget, minimized
at m* = sqrt(c_ctx n sw2 / (c_q sb2)).

Checks:
 (a) variance formula: 200,000 replications of the mean of B = 10 tables with m = 4 queries
     (sb2 = 1, sw2 = 4): simulated variance against (sb2 + sw2/m)/B (tolerance 3%);
 (b) the optimum: grid search over integer m in 1..400 of the variance at fixed budget, for three
     settings (c_ctx n, c_q, sb2, sw2) = (100, 1, 1, 10), (500, 2, 0.5, 20), (50, 1, 4, 2); the grid
     argmin must be within 1 of m* (integer rounding);
 (c) SGD with a constant step eta = 0.05 on L(theta) = theta^2 / 2 with the gradient noise above,
     2,000 steps from theta = 0, 20,000 runs: the final excess loss divided by the noise variance
     (sb2 + sw2/m)/B is the same across (B, m) settings (ratio spread below 4%). The minibatch noise
     is drawn from its exact law N(0, (sb2 + sw2/m)/B): a mean of independent tables and queries is
     normal with that variance;
 (d) the factor by which the variance falls when m doubles, (1 + r)/(1 + r/2) with r = sw2/(m sb2):
     tends to 1 as r -> 0; checked against the simulated variances at m and 2m (tolerance 3%).
"""

import numpy as np

rng = np.random.default_rng(0)
sb2, sw2 = 1.0, 4.0


def var_formula(sb2, sw2, m, B):
    return (sb2 + sw2 / m) / B


# (a)
B, m, R = 10, 4, 200_000
b = rng.normal(0, np.sqrt(sb2), size=(R, B))
w = rng.normal(0, np.sqrt(sw2), size=(R, B, m))
means = b.mean(axis=1) + w.mean(axis=(1, 2))
sim = means.var()
form = var_formula(sb2, sw2, m, B)
print(
    f"(a) simulated variance {sim:.5f}, formula {form:.5f}, ratio {sim / form:.4f} (tolerance 3%)"
)
assert abs(sim / form - 1) < 0.03

# (b)
settings = [(100.0, 1.0, 1.0, 10.0), (500.0, 2.0, 0.5, 20.0), (50.0, 1.0, 4.0, 2.0)]
ms = np.arange(1, 401)
for cA, cq, s_b, s_w in settings:
    budget = 1.0
    cost = cA + cq * ms
    V = var_formula(s_b, s_w, ms, budget / cost)
    m_grid = ms[np.argmin(V)]
    m_star = np.sqrt(cA * s_w / (cq * s_b))
    print(
        f"(b) c_ctx*n={cA:g} c_q={cq:g} sb2={s_b:g} sw2={s_w:g}: "
        f"grid argmin {m_grid}, m* = {m_star:.2f}"
    )
    assert abs(m_grid - m_star) <= 1.0, (m_grid, m_star)

# (c) SGD
eta, T, Rs = 0.05, 2000, 20_000
ratios = []
for B_, m_ in [(1, 1), (1, 4), (4, 4), (1, 16), (16, 16)]:
    noise_sd = np.sqrt(var_formula(sb2, sw2, m_, B_))
    theta = np.zeros(Rs)
    for _ in range(T):
        theta = theta - eta * (theta + rng.normal(0, noise_sd, size=Rs))  # lambda = 1
    excess = np.mean(theta**2 / 2)
    ratio = excess / var_formula(sb2, sw2, m_, B_)
    ratios.append(ratio)
    print(f"(c) B={B_:2d} m={m_:2d}: final excess loss {excess:.5f}, excess / variance {ratio:.5f}")
spread = (max(ratios) - min(ratios)) / np.mean(ratios)
print(f"(c) relative spread of excess/variance across settings: {spread:.3%} (tolerance 4%)")
assert spread < 0.04

# (d)
for r in (0.01, 0.1, 0.5, 1.0, 2.0, 5.0):
    fac = (1 + r) / (1 + r / 2)
    print(f"(d) r = sw2/(m sb2) = {r:5.2f}: variance factor when m doubles {fac:.4f}")
m1 = 4
v1 = var_formula(sb2, sw2, m1, 1.0)
v2 = var_formula(sb2, sw2, 2 * m1, 1.0)
sims = []
for m_ in (m1, 2 * m1):
    bb = rng.normal(0, np.sqrt(sb2), size=(R, 1))
    ww = rng.normal(0, np.sqrt(sw2), size=(R, 1, m_))
    sims.append((bb.mean(axis=1) + ww.mean(axis=(1, 2))).var())
print(
    f"(d) simulated factor at m=4 -> 8 (sb2=1, sw2=4, B=1): {sims[0] / sims[1]:.4f}, "
    f"formula {v1 / v2:.4f}"
)
assert abs(sims[0] / sims[1] - v1 / v2) / (v1 / v2) < 0.03
print("PASS s11c")
