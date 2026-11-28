"""Check for v2-task-formulations.md section 10, step 3 (the mixture bound).

Claim reproduced: with pi = (1 - w) pi_0 + sum_k w_k pi_k and q* = sum_j P(j | D, x) p_j,
  0 <= E_0 KL(p_0 || q*) <= -log(1 - w)
for every case; both ends are reached (0 when the task tables can never be taken for ordinary ones,
-log(1 - w) when the two kinds of table look alike and their answers are opposite). The stated
example: 0.29 nats at w = 0.25.

Design: discrete "contexts" c = 1..C (C = 6 here, standing for (D, x)), outcomes y = 1..Y (Y = 4).
Each component j has context law pi_j(c) and answers p_j(y | c), all random Dirichlet draws
(seed 0). The mixture weight w is random in (0.05, 0.9) with one task component (or two, with the
weights split). Each case: E_0 KL = sum_c pi_0(c) KL(p_0(.|c) || q*(.|c)) with q* the posterior
mixture. 20,000 random cases; then the two extremes.
Stated: the bound holds in every case (tolerance 1e-12); 0.29 at w = 0.25 (printed).
"""

import numpy as np

rng = np.random.default_rng(0)
C, Y = 6, 4


def kl(p, q):
    m = p > 0  # 0 log 0 = 0
    return float(np.sum(p[m] * np.log(p[m] / q[m])))


def expected_kl(w_list, pi, p):
    """pi[j]: context law of component j (shape C); p[j]: answers (shape C x Y); j = 0 is
    ordinary."""
    w0 = 1 - sum(w_list)
    weights = np.array([w0] + list(w_list))
    total = 0.0
    for c in range(C):
        joint = weights * np.array([pi[j][c] for j in range(len(weights))])
        post = joint / joint.sum()
        q = post @ np.array([p[j][c] for j in range(len(weights))])
        total += pi[0][c] * kl(p[0][c], q)
    return total


worst_low, worst_high = np.inf, -np.inf
violations = 0
for _ in range(20_000):
    k = rng.integers(1, 3)
    w_total = rng.uniform(0.05, 0.9)
    split = rng.dirichlet(np.ones(k)) * w_total
    pi = [rng.dirichlet(np.full(C, 0.5)) for _ in range(k + 1)]
    p = [rng.dirichlet(np.full(Y, 0.5), size=C) for _ in range(k + 1)]
    val = expected_kl(list(split), pi, p)
    bound = -np.log(1 - w_total)
    worst_low = min(worst_low, val)
    worst_high = max(worst_high, val / bound)
    if val < -1e-12 or val > bound + 1e-12:
        violations += 1
print(
    f"20,000 random cases: violations {violations} (stated 0); min E KL {worst_low:.3e}; "
    f"max ratio to the bound {worst_high:.4f}"
)

# extremes (one task component, w = 0.25, C = 2 contexts)
w = 0.25
print(f"-log(1 - 0.25) = {-np.log(1 - w):.4f} nats (stated 0.29)")
# lower end: the task table lives on a context that ordinary tables never occupy (L_k = 0)
pi_lo = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
p_lo = [np.array([[0.2, 0.8], [0.5, 0.5]]), np.array([[0.2, 0.8], [0.5, 0.5]])]
C_save = C
C = 2  # the extremes use two contexts
low = expected_kl(
    [w], pi_lo, [np.vstack([p_lo[0][0], p_lo[0][0]]), np.vstack([p_lo[1][1], p_lo[1][1]])]
)
# upper end: same context law, answers with disjoint supports at every context
pi_same = [np.array([0.5, 0.5]), np.array([0.5, 0.5])]
p0_up = np.array([[1.0, 0.0], [1.0, 0.0]])
p1_up = np.array([[0.0, 1.0], [0.0, 1.0]])
high = expected_kl([w], pi_same, [p0_up, p1_up])
C = C_save
print(f"low extreme (task table on a context ordinary tables never take): E KL = {low:.2e}")
print(
    f"high extreme (same contexts, disjoint answers): E KL = {high:.6f}, bound {-np.log(1 - w):.6f}"
)
assert violations == 0
assert low < 1e-12
assert abs(high - (-np.log(1 - w))) < 1e-9
print("PASS s10a")
