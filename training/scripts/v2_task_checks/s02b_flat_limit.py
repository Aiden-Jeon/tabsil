"""Check for v2-task-formulations.md section 2, step 7: the flat-prior limit of the conjugate linear
model is the textbook prediction interval.

Claim (stated as "the flat limit ... is the textbook prediction interval" in the document):
with Lambda0 -> 0, a0 -> -d/2, b0 -> 0 (improper prior proportional to 1/sigma^2), the posterior
predictive mean is the least-squares fit and its central 95% interval equals
x*' beta_hat +- t_{n-d}^{-1}(0.975) sigma_hat sqrt(1 + x*' (X'X)^{-1} x*).

Design: n = 40, d = 3, X ~ N(0, 1), seed 0, beta = (1, -2, 0.5), noise N(0, 1); Lambda0 = 1e-10 I
(numerically flat), a0 = -d/2, b0 = 1e-12 (numerically zero). Three query points. The posterior
t interval (df 2 a_n = n - d) is compared with the textbook interval. Tolerance: 1e-6 on each
endpoint. The stated design has no number beyond the identity, so the check is an identity.
"""

import numpy as np
from scipy import stats

rng = np.random.default_rng(0)
n, d = 40, 3
X = rng.normal(size=(n, d))
y = X @ np.array([1.0, -2.0, 0.5]) + rng.normal(size=n)

beta_hat, *_ = np.linalg.lstsq(X, y, rcond=None)
resid = y - X @ beta_hat
sig_hat = np.sqrt(resid @ resid / (n - d))
XtX_inv = np.linalg.inv(X.T @ X)

L0 = 1e-10 * np.eye(d)
a0, b0, m0 = -d / 2, 1e-12, np.zeros(d)
Ln = L0 + X.T @ X
mn = np.linalg.solve(Ln, L0 @ m0 + X.T @ y)
an = a0 + n / 2
bn = b0 + 0.5 * (y @ y + m0 @ L0 @ m0 - mn @ Ln @ mn)
t975 = stats.t.ppf(0.975, n - d)

max_err = 0.0
for xs in (np.array([1.0, 0.0, 0.0]), np.array([0.3, -1.2, 2.0]), np.array([-1.0, 1.0, 1.5])):
    loc = xs @ mn
    scale = np.sqrt(bn / an * (1 + xs @ np.linalg.solve(Ln, xs)))
    post = (loc - stats.t.ppf(0.975, 2 * an) * scale, loc + stats.t.ppf(0.975, 2 * an) * scale)
    half = t975 * sig_hat * np.sqrt(1 + xs @ XtX_inv @ xs)
    text = (xs @ beta_hat - half, xs @ beta_hat + half)
    err = max(abs(post[0] - text[0]), abs(post[1] - text[1]), abs(loc - xs @ beta_hat))
    max_err = max(max_err, err)
    print(
        f"x*={xs}: posterior [{post[0]:.6f}, {post[1]:.6f}]  "
        f"textbook [{text[0]:.6f}, {text[1]:.6f}]"
    )
print(f"df posterior 2a_n = {2 * an:.6f}, n - d = {n - d}")
print(f"max endpoint / mean difference = {max_err:.2e}, stated tolerance 1e-6")
assert max_err < 1e-6, max_err
print("PASS s02b")
