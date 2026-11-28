"""Check for v2-task-formulations.md section 11.6: an AUC-aligned pairwise loss against the
cross-entropy, for a linear scorer.

Binary labels, eta(x) = P(y = 1 | x), x ~ N(0, I_2).
  Well-specified: eta(x) = sigmoid(0.3 + 1.5 x1 - x2) (logistic in x).
  Misspecified:  eta(x) = sigmoid(2 x1 + 3 x2^2 - 1.5), fitted with a linear scorer s(x) = a'x + c.
Linear scorer s(x) = a'x + c, fitted two ways:
  CE:       minimize the expected cross-entropy E[-y log sigmoid(s) - (1-y) log(1 - sigmoid(s))]
            over (a, c), on a sample of 200,000 draws (seed 0);
  Pairwise: minimize the mean logistic pairwise loss over (positive, negative) pairs, E[log(1 +
            exp(-(s(x+) - s(x-))))], on 1,500 positives x 1,500 negatives (seed 0); c cancels.
Population AUC: 200,000 fresh draws (seed 1), Mann-Whitney on the score. The Bayes reference is the
AUC of eta itself (the ordering of the true conditional probability).

Claims checked:
  well-specified: the two fits give the same AUC (within 0.003, the Monte Carlo noise of the AUCs),
  and both equal the Bayes AUC (the true score is linear in x there);
  misspecified: the AUC difference (pairwise minus CE) and its sign are printed (a finding, not
  asserted). At infinite capacity both reach the Bayes ordering; the gain is a finite-capacity
  effect.
"""

import numpy as np
from scipy import optimize, stats
from scipy.special import expit, log_expit

rng = np.random.default_rng(0)
N = 200_000


def make_eta(kind):
    if kind == "well":
        return lambda x: expit(0.3 + 1.5 * x[:, 0] - 1.0 * x[:, 1])
    return lambda x: expit(2 * x[:, 0] + 3 * x[:, 1] ** 2 - 1.5)


def ce_fit(X, y):
    def loss(p):
        s = X @ p[:2] + p[2]
        return -np.mean(y * log_expit(s) + (1 - y) * log_expit(-s))

    def grad(p):
        s = X @ p[:2] + p[2]
        g = expit(s) - y
        return np.r_[X.T @ g / len(y), np.mean(g)]

    return optimize.minimize(loss, np.zeros(3), jac=grad, method="BFGS").x


def pairwise_fit(Xp, Xn):
    D = (Xp[:, None, :] - Xn[None, :, :]).reshape(-1, 2)

    def loss(a):
        m = -(D @ a)
        return np.mean(np.logaddexp(0.0, m))

    def grad(a):
        m = -(D @ a)
        w = expit(m)
        return -(D.T @ w) / len(D)

    a = optimize.minimize(loss, np.zeros(2), jac=grad, method="BFGS").x
    return a


def auc(score, y):
    r = stats.rankdata(score)
    n1 = y.sum()
    n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


for kind in ("well", "mis"):
    eta = make_eta(kind)
    X = rng.normal(size=(N, 2))
    y = (rng.random(N) < eta(X)).astype(float)
    p_ce = ce_fit(X, y)
    pos = X[y == 1][:1500]
    neg = X[y == 0][:1500]
    a_pair = pairwise_fit(pos, neg)
    Xt = np.random.default_rng(1).normal(size=(N, 2))
    yt = (np.random.default_rng(2).random(N) < eta(Xt)).astype(float)
    auc_ce = auc(Xt @ p_ce[:2], yt)
    auc_pair = auc(Xt @ a_pair, yt)
    auc_bayes = auc(eta(Xt), yt)
    print(
        f"{kind}: CE coef {np.round(p_ce, 3)}, "
        f"pairwise direction {np.round(a_pair / np.linalg.norm(a_pair), 3)}"
        f" (CE direction {np.round(p_ce[:2] / np.linalg.norm(p_ce[:2]), 3)})"
    )
    print(
        f"   AUC: CE {auc_ce:.4f}, pairwise {auc_pair:.4f}, Bayes (eta itself) {auc_bayes:.4f}; "
        f"pairwise - CE = {auc_pair - auc_ce:+.4f}"
    )
    if kind == "well":
        assert abs(auc_ce - auc_pair) < 0.003, (auc_ce, auc_pair)
        assert abs(auc_ce - auc_bayes) < 0.003 and abs(auc_pair - auc_bayes) < 0.003
print("PASS s11f (well-specified checks; misspecified difference printed, sign is a finding)")
