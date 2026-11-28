"""Check for v2-task-formulations.md section 4, step 7 (a dense group passes for inliers).

Claims reproduced (Gaussian case, the table of step 7):
  AUROC of the density score -log p_c(x) against the share eps of a dense group in the context:
     eps = 0: 0.9995; 1e-6: 0.857; 1e-5: 0.527; 1e-4: 0.168; 0.02: 0.003 (stated);
  AUROC of the likelihood ratio Lambda(x) = p1(x)/p0(x): 1.000 at every share (stated);
  the threshold eps > sigma^d/(1 + sigma^d) = 1.1e-5 for sigma = 0.15, d = 6 (stated).

Design: inliers N(0, I_6); group N(2 * 1, 0.15^2 I_6). Contaminated density
p_c = (1 - eps) N(0, I) + eps N(mu, sigma^2 I), exact. Test rows: 20,000 inliers (seed 1) and
20,000 group members (seed 2). AUROC: probability that a group member scores higher (more outlying)
than an inlier, by the rank (Mann-Whitney) formula. Same test rows for every share.
Stated tolerance: 0.01 (absolute; the stated values have three decimals and 20,000 rows per class).
"""

import numpy as np
from scipy import stats

d, sig, mu = 6, 0.15, 2.0
n_in = n_gr = 20_000
rng_in = np.random.default_rng(1)
rng_gr = np.random.default_rng(2)
x_in = rng_in.normal(size=(n_in, d))
x_gr = mu + sig * rng_gr.normal(size=(n_gr, d))
x = np.vstack([x_in, x_gr])
labels = np.r_[np.zeros(n_in), np.ones(n_gr)]


def p0(z):
    return stats.multivariate_normal(np.zeros(d), np.eye(d)).pdf(z)


def p1(z):
    return stats.multivariate_normal(np.full(d, mu), sig**2 * np.eye(d)).pdf(z)


def auroc(score, lab):
    """P(score of a group member > score of an inlier), Mann-Whitney."""
    r = stats.rankdata(score)
    n1 = lab.sum()
    n0 = len(lab) - n1
    return (r[lab == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


a0, a1 = p0(x), p1(x)
stated = {0.0: 0.9995, 1e-6: 0.857, 1e-5: 0.527, 1e-4: 0.168, 0.02: 0.003}
max_err = 0.0
for eps, want in stated.items():
    pc = (1 - eps) * a0 + eps * a1
    got = auroc(-np.log(pc), labels)
    lr = auroc(np.log(a1) - np.log(a0), labels)  # likelihood-ratio score: high = outlier
    max_err = max(max_err, abs(got - want))
    print(
        f"eps={eps:<8g} density-score AUROC {got:.4f} (stated {want}), "
        f"likelihood-ratio AUROC {lr:.4f} (stated 1.000)"
    )
thr = sig**d / (1 + sig**d)
print(f"threshold sigma^d/(1+sigma^d) = {thr:.3e} (stated 1.1e-5)")
assert max_err < 0.01, max_err
assert abs(thr - 1.1e-5) < 0.05e-5, thr  # the stated value has two significant digits
print("PASS s04a")
