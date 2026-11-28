"""Check for v2-task-formulations.md section 4 (diffuse outliers and the binomial counts).

Claims reproduced:
  (a) diffuse outliers uniform on the box [-6, 6]^6 with N(0, I_6) inliers: the density score and
      the likelihood-ratio score both have AUROC 0.9992 (stated), and they order the rows
      identically;
  (b) binomial counts for a context of 64 rows with group share 0.02: no group member with
      probability 0.27 and at most one with 0.63 (stated; exact binomial and 200,000 simulated
      contexts).

Design: 20,000 inliers and 20,000 uniform outliers (seeds 1, 2). Contamination share in the density
eps = 0.02 (the ordering does not depend on it, which is the point of the check); p1 = 1/12^6.
Tolerance: 0.002 on AUROC (the identity of the two orders is checked exactly), 0.005 on the
probabilities.
"""

import numpy as np
from scipy import stats

d, half, eps = 6, 6.0, 0.02
rng_in = np.random.default_rng(1)
rng_out = np.random.default_rng(2)
x_in = rng_in.normal(size=(20_000, d))
x_out = rng_out.uniform(-half, half, size=(20_000, d))
x = np.vstack([x_in, x_out])
labels = np.r_[np.zeros(len(x_in)), np.ones(len(x_out))]
p1 = (2 * half) ** (-d)
p0 = stats.multivariate_normal(np.zeros(d), np.eye(d)).pdf(x)
pc = (1 - eps) * p0 + eps * p1 * (np.all(np.abs(x) <= half, axis=1))  # box density for outliers


def auroc(score, lab):
    r = stats.rankdata(score)
    n1 = lab.sum()
    n0 = len(lab) - n1
    return (r[lab == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


s_density = auroc(-np.log(pc), labels)
s_lr = auroc(np.log(p1) - np.log(p0), labels)  # likelihood-ratio score: high = outlier
inside = np.all(np.abs(x) <= half, axis=1)  # the claim is about rows inside the box
# the density score must be non-decreasing when rows are sorted by the inlier score -log p0
# (ties allowed: in floating point the outlier term swamps p0 far in the tails, where pc is equal)
ordr = np.argsort(-np.log(p0[inside]))
order_same = bool(np.all(np.diff((-np.log(pc[inside]))[ordr]) >= 0))
ties = int(np.sum(np.diff(pc[inside][ordr]) == 0))
print(
    f"density score AUROC {s_density:.4f}, likelihood-ratio AUROC {s_lr:.4f} "
    "(stated 0.9992 for both)"
)
print(
    f"density score non-decreasing in -log p0 inside the box: {order_same}; "
    f"tied neighbours (float): {ties}"
)

p_none = (1 - 0.02) ** 64
p_le1 = p_none + 64 * 0.02 * (1 - 0.02) ** 63
rng = np.random.default_rng(0)
counts = rng.binomial(64, 0.02, size=200_000)
print(
    f"binomial, 64 rows, share 0.02: P(0 members) = {p_none:.4f} (stated 0.27), "
    f"P(<= 1) = {p_le1:.4f} (stated 0.63); "
    f"simulated {np.mean(counts == 0):.4f}, {np.mean(counts <= 1):.4f}"
)

assert abs(s_density - 0.9992) < 0.002
assert abs(s_lr - 0.9992) < 0.002
assert order_same
assert abs(p_none - 0.27) < 0.005 and abs(p_le1 - 0.63) < 0.005
print("PASS s04b")
