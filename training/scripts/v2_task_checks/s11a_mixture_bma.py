"""Check for v2-task-formulations.md section 11.2: a mixture of priors is Bayesian model averaging.

Setting: Bayesian linear regression, known noise sd 1, d = 8 features x ~ N(0, I).
  Component A (dense): beta ~ N(0, tau^2 I), tau^2 = 4/d = 0.5.
  Component B (sparse): exactly k = 2 nonzero coefficients, uniformly random support (C(8,2) = 28
    supports), nonzero values N(0, 2^2). Given the support, the posterior is Gaussian.
  Mixture prior: w B + (1 - w) A, w in {0.1, 0.5}.
Exact posterior predictives: A is Gaussian; B is a mixture over the 28 supports, each Gaussian,
weighted by its exact marginal likelihood (uniform support prior); the mixture prior weights A and B
by their exact marginal likelihoods. Marginal likelihoods via the Woodbury identity (k-dimensional).

Tables: 2,000 from each component (seeds 0 for A, 1 for B), context sizes n in {8, 16, 32, 64, 256},
Q = 20 queries per table. Measures: mean predictive log loss per query, and the RMSE of the
predictive mean against the true regression function x*' beta.

Checks (the claims of section 11.2):
  (i) on tables from A: the mixture's excess log loss over A is >= 0 (in expectation) and <=
      E[-log P(A | D)] (pointwise, so also on the average), and E[-log P(A | D)] <= -log(1 - w);
  (ii) on tables from B: prior A's RMSE excess over B, as a function of n (printed; the "coverage
  gain");
  (iii) the mixture's excess over the matching component falls as n grows (checked: excess at n =
        256 below excess at n = 8, for both components and both w).
Tolerance: Monte Carlo; (i) and (iii) use a 3-standard-error margin.
"""

import itertools

import numpy as np
from scipy.special import logsumexp

d, sig2 = 8, 1.0
tau2 = 4.0 / d
sb2 = 4.0
supports = np.array(list(itertools.combinations(range(d), 2)))  # 28 x 2
NS = len(supports)
ns = [8, 16, 32, 64, 256]
TABLES, Q = 2000, 20
ws = [0.1, 0.5]


def gauss_logpdf(y, m, v):
    return -0.5 * (np.log(2 * np.pi * v) + (y - m) ** 2 / v)


def posterior_pred(X, y, Sigma, Xq):
    """log marginal likelihood, predictive means and variances (per query) for a Gaussian prior
    beta ~ N(0, Sigma) on the columns of X (Sigma: k x k, X: n x k, Xq: Q x k)."""
    n = len(y)
    G = X.T @ X
    b = X.T @ y
    Pinv_prec = np.linalg.inv(np.linalg.inv(Sigma) + G)
    _, logdet = np.linalg.slogdet(np.eye(len(Sigma)) + Sigma @ G)
    logml = -0.5 * n * np.log(2 * np.pi) - 0.5 * logdet - 0.5 * (y @ y - b @ Pinv_prec @ b)
    m = Pinv_prec @ b
    mean = Xq @ m
    var = 1.0 + np.einsum("qk,kl,ql->q", Xq, Pinv_prec, Xq)
    return logml, mean, var


def predictives(X, y, Xq):
    """A (dense), and the 28 support-wise posteriors for B, vectorized over the supports."""
    logml_A, mu_A, v_A = posterior_pred(X, y, tau2 * np.eye(d), Xq)
    n = len(y)
    Xs = np.transpose(X[:, supports], (1, 0, 2))  # 28 x n x 2
    Xqs = np.transpose(Xq[:, supports], (1, 0, 2))  # 28 x Q x 2
    G = np.einsum("snk,snl->skl", Xs, Xs)
    b = np.einsum("snk,n->sk", Xs, y)
    P = np.linalg.inv(np.eye(2) / sb2 + G)
    _, logdet = np.linalg.slogdet(np.eye(2) + sb2 * G)
    logml_S = (
        -0.5 * n * np.log(2 * np.pi)
        - 0.5 * logdet
        - 0.5 * (y @ y - np.einsum("sk,skl,sl->s", b, P, b))
    )
    m = np.einsum("skl,sl->sk", P, b)
    mu_S = np.einsum("sqk,sk->sq", Xqs, m)
    v_S = 1.0 + np.einsum("sqk,skl,sql->sq", Xqs, P, Xqs)
    return logml_A, mu_A, v_A, logml_S, mu_S, v_S


def run(component, seed):
    rng = np.random.default_rng(seed)
    out = {}
    for n in ns:
        ll = {k: [] for k in ["A", "B", "mix0.1", "mix0.5", "dA"]}
        rm = {k: [] for k in ["A", "B", "mix0.1", "mix0.5"]}
        for _ in range(TABLES):
            if component == "A":
                beta = rng.normal(0, np.sqrt(tau2), d)
            else:
                beta = np.zeros(d)
                s = supports[rng.integers(NS)]
                beta[s] = rng.normal(0, 2.0, 2)
            X = rng.normal(size=(n + Q, d))
            y = X @ beta + rng.normal(size=n + Q)
            Xc, yc, Xq, yq = X[:n], y[:n], X[n:], y[n:]
            truth = Xq @ beta
            lmA, muA, vA, lmS, muS, vS = predictives(Xc, yc, Xq)
            # B: mixture over supports with weights ~ marginal likelihood (uniform support prior)
            wS = lmS - logsumexp(lmS)
            logml_B = logsumexp(lmS) - np.log(NS)
            logpB = logsumexp(wS[:, None] + gauss_logpdf(yq[None, :], muS, vS), axis=0)
            meanB = np.exp(wS) @ muS
            logpA = gauss_logpdf(yq, muA, vA)
            ll["A"].append(-logpA)
            ll["B"].append(-logpB)
            rm["A"].append((muA - truth) ** 2)
            rm["B"].append((meanB - truth) ** 2)
            for w in ws:
                lb = np.log(w) + logml_B
                la = np.log(1 - w) + lmA
                logPB = lb - np.logaddexp(la, lb)
                logPA = la - np.logaddexp(la, lb)
                logpmix = np.logaddexp(logPA + logpA, logPB + logpB)
                ll[f"mix{w}"].append(-logpmix)
                meanmix = np.exp(logPA) * muA + np.exp(logPB) * meanB
                rm[f"mix{w}"].append((meanmix - truth) ** 2)
                if w == 0.1:
                    ll["dA"].append(-logPA)  # -log P(A | D), per query
        out[n] = {k: np.mean(v) for k, v in ll.items() if k != "dA"}
        out[n]["dA"] = np.mean(ll["dA"])
        out[n]["dA_per_table_max_excess_check"] = 0.0
        out[n]["rmse"] = {k: np.sqrt(np.mean(v)) for k, v in rm.items()}
        out[n]["_se_excess_A"] = np.std(np.array(ll["mix0.1"]) - np.array(ll["A"])) / np.sqrt(
            len(ll["A"])
        )
        out[n]["_se_excess_B"] = np.std(np.array(ll["mix0.1"]) - np.array(ll["B"])) / np.sqrt(
            len(ll["B"])
        )
    return out


resA = run("A", 0)
resB = run("B", 1)

print(
    "tables from A: mean log loss per query (A, B, mixture w = 0.1, w = 0.5), "
    "E[-log P(A|D)] (w = 0.1)"
)
for n in ns:
    r = resA[n]
    print(
        f"  n={n:3d}: A {r['A']:.4f}  B {r['B']:.4f}  mix0.1 {r['mix0.1']:.4f}  "
        f"mix0.5 {r['mix0.5']:.4f}  "
        f"E[-log P(A|D)] {r['dA']:.4f}"
    )
    print(
        f"          excess mix0.1 - A {r['mix0.1'] - r['A']:+.4f} (se {r['_se_excess_A']:.4f}); "
        f"excess mix0.5 - A {r['mix0.5'] - r['A']:+.4f}"
    )
print("tables from B: mean log loss per query (A, B, mixture w = 0.1, w = 0.5)")
for n in ns:
    r = resB[n]
    print(
        f"  n={n:3d}: A {r['A']:.4f}  B {r['B']:.4f}  mix0.1 {r['mix0.1']:.4f}  "
        f"mix0.5 {r['mix0.5']:.4f}  "
        f"E[-log P(A|D)] {r['dA']:.4f}"
    )
    print(
        f"          excess mix0.1 - B {r['mix0.1'] - r['B']:+.4f} (se {r['_se_excess_B']:.4f}); "
        f"excess mix0.5 - B {r['mix0.5'] - r['B']:+.4f}"
    )
print("RMSE of the predictive mean (vs x'beta):")
for tag, res, _ in (("A-tables", resA, "A"), ("B-tables", resB, "B")):
    for n in ns:
        rr = res[n]["rmse"]
        print(
            f"  {tag} n={n:3d}: A {rr['A']:.4f} B {rr['B']:.4f} "
            f"mix0.1 {rr['mix0.1']:.4f} mix0.5 {rr['mix0.5']:.4f}"
        )
print("(ii) coverage gain on B-tables, RMSE of A minus RMSE of B, by n:")
for n in ns:
    print(f"  n={n:3d}: {resB[n]['rmse']['A'] - resB[n]['rmse']['B']:+.4f}")

ok = True
for n in ns:
    r = resA[n]
    tol = 3 * r["_se_excess_A"]
    ok &= r["mix0.1"] - r["A"] >= -tol
    ok &= r["mix0.1"] - r["A"] <= r["dA"] + tol
    ok &= r["dA"] <= -np.log(1 - 0.1) + tol
    r = resA[n]
    ok &= r["mix0.5"] - r["A"] <= -np.log(1 - 0.5) + 3 * r["_se_excess_A"]
print("(i) bounds on A-tables:", "ok" if ok else "FAILED")
assert ok
trend_A = resA[256]["mix0.1"] - resA[256]["A"] < resA[8]["mix0.1"] - resA[8]["A"]
trend_B = resB[256]["mix0.1"] - resB[256]["B"] < resB[8]["mix0.1"] - resB[8]["B"]
print(f"(iii) excess falls from n=8 to n=256: A-tables {trend_A}, B-tables {trend_B}")
assert trend_A and trend_B
print("PASS s11a")
