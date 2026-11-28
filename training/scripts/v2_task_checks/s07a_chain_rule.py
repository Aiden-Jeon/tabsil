"""Check for v2-task-formulations.md section 7, step 7 (the chain rule for the masked-cell loss).

Claim reproduced ("For a Gaussian table the sum of the exact conditionals along any order equals the
joint log density (checked)"): for a five-column Gaussian table, the sum over the columns of
log p(x_{pi_j} | x_{pi_<j}) along three random orders pi equals log p(x), the joint log density.

Design: a random positive definite covariance (5 x 5, seed 0) with a random mean; 2,000 rows;
conditionals computed exactly from the covariance (Gaussian conditioning). Stated tolerance 1e-8
(absolute, on the summed log density; the sums are of order 10^3).
"""

import numpy as np
from scipy import stats

rng = np.random.default_rng(0)
d = 5
A = rng.normal(size=(d, d))
Sigma = A @ A.T + 0.5 * np.eye(d)
mu = rng.normal(size=d)
X = rng.multivariate_normal(mu, Sigma, size=2000)
joint = stats.multivariate_normal(mu, Sigma).logpdf(X)


def chain(order):
    total = np.zeros(len(X))
    for j in range(d):
        k = order[j]
        prev = list(order[:j])
        if not prev:
            m, v = mu[k], Sigma[k, k]
        else:
            S_pp = Sigma[np.ix_(prev, prev)]
            S_kp = Sigma[k, prev]
            w = np.linalg.solve(S_pp, S_kp)
            m = mu[k] + (X[:, prev] - mu[prev]) @ w
            v = Sigma[k, k] - S_kp @ w
        total += stats.norm.logpdf(X[:, k], m, np.sqrt(v))
    return total


max_err = 0.0
for _ in range(3):
    order = list(rng.permutation(d))
    s = chain(order)
    err = np.max(np.abs(s - joint))
    max_err = max(max_err, err)
    print(f"order {order}: max |chain rule - joint log density| = {err:.2e}")
print(f"joint log density range [{joint.min():.2f}, {joint.max():.2f}]; stated tolerance 1e-8")
assert max_err < 1e-8, max_err
print("PASS s07a")
