"""Check for v2-task-formulations.md section 11.4: shared representation; auxiliary targets help
when aligned and hurt when not.

Setting: x ~ N(0, I_d), d = 20; main target y = x'beta + eps; K auxiliary targets z_k = x'gamma_k +
eps_k; noise sd 1. gamma_k = U a_k with a_k ~ N(0, I_r), r = 3, U a d x r orthonormal basis (seed
0). beta = cos(theta) U a_0 + sin(theta) v, with v = ||U a_0|| u for a unit vector u orthogonal to
the span of U (so ||beta|| = ||U a_0|| for every theta; theta = 0 aligned, theta = pi/2 orthogonal).

Estimators:
 (a) OLS of y on x;
 (b) U estimated by the top min(r, K) left singular vectors of the matrix of the K auxiliary OLS
     coefficient vectors (d x K); then OLS of y on U_hat' x. For K < r only K directions exist
     (stated reading: the top min(r, K)).
 (o) oracle: OLS of y on U' x with the true U (for the checks).
Risk: E[(x'(beta_hat - beta))^2] = ||beta_hat - beta||^2 (x ~ N(0, I)), 500 replications per cell,
common random numbers across K and theta for each n (seed 1 + n).

Approximations checked (the formulas are exact in expectation, so they are checked with 20,000
replications of OLS and of the oracle, a separate run with its own seed, since the risk is
heavy-tailed at n = 30):
 OLS: d/(n - d - 1) (sigma = 1);
 oracle with the true U: the error is U(w_hat - w) - v with v = P_U-perp beta, and
     E[(w_hat - w)'(w_hat - w)] = (1 + ||v||^2) r/(n - r - 1), so the oracle risk is
     (1 + sin^2(theta)||beta||^2) r/(n - r - 1) + sin^2(theta)||beta||^2.
     (The bias ||v||^2 alone, which the brief states, is the population part; the extra term comes
     from the omitted component's effect on the estimated coefficients in finite samples. The table
     shows it is needed: 0.4873 against the simulated 0.5337 at n = 30 without it, and 0.3929
     against 0.3923 at n = 200 with it.)
Verdict printed: where the auxiliary estimator is below OLS and where above.
Tolerance: each OLS and oracle value against its formula within 4 standard errors of the
500-replication mean (the squared-error risk is heavy-tailed at n = 30, so a fixed percentage is too
tight there) plus 1%.
"""

import numpy as np

d, r, sigma = 20, 3, 1.0
rng0 = np.random.default_rng(0)
Q, _ = np.linalg.qr(rng0.normal(size=(d, d)))
U = Q[:, :r]
u_perp = Q[:, r]  # orthogonal to U
a0 = rng0.normal(size=r)
a_aux = rng0.normal(size=(50, r))
gammas = (U @ a_aux.T).T  # 50 x d
beta_norm = np.linalg.norm(U @ a0)
thetas = {"0": 0.0, "pi/8": np.pi / 8, "pi/4": np.pi / 4, "pi/2": np.pi / 2}
betas = {k: np.cos(t) * (U @ a0) + np.sin(t) * beta_norm * u_perp for k, t in thetas.items()}

ns = [30, 60, 200]
Ks = [0, 2, 10, 50]
REPS = 500
results = {}
for n in ns:
    rng = np.random.default_rng(1 + n)
    risks = {("ols", None, None): []}
    for K in Ks:
        for k in thetas:
            if K > 0:
                risks[("aux", K, k)] = []
    for k in thetas:
        risks[("oracle", None, k)] = []
    for _ in range(REPS):
        X = rng.normal(size=(n, d))
        eps = rng.normal(scale=sigma, size=n)
        epsz = rng.normal(scale=sigma, size=(n, 50))
        Z = X @ gammas.T + epsz
        XtX_inv = np.linalg.inv(X.T @ X)
        Gam_hat = XtX_inv @ X.T @ Z  # d x 50, the OLS coefficient vectors of z_k
        for k, _ in thetas.items():
            y = X @ betas[k] + eps
            b_ols = XtX_inv @ X.T @ y
            risks[("ols", None, None)].append(np.sum((b_ols - betas[k]) ** 2))
            XU = X @ U
            c_or = np.linalg.solve(XU.T @ XU, XU.T @ y)
            risks[("oracle", None, k)].append(np.sum((U @ c_or - betas[k]) ** 2))
            for K in Ks:
                if K == 0:
                    continue
                q = min(r, K)
                Uh, _, _ = np.linalg.svd(Gam_hat[:, :K], full_matrices=False)
                Uh = Uh[:, :q]
                XUh = X @ Uh
                c = np.linalg.solve(XUh.T @ XUh, XUh.T @ y)
                risks[("aux", K, k)].append(np.sum((Uh @ c - betas[k]) ** 2))
    results[n] = {key: np.mean(v) for key, v in risks.items()}
    results[n]["_se"] = {key: np.std(v, ddof=1) / np.sqrt(len(v)) for key, v in risks.items()}
    # 20,000-replication check of the OLS and oracle formulas (own seed)
    rng2 = np.random.default_rng(100 + n)
    big = 20_000
    ols_big, or_big = [], {k: [] for k in thetas}
    for _ in range(big):
        X = rng2.normal(size=(n, d))
        XtX_inv = np.linalg.inv(X.T @ X)
        e = rng2.normal(size=n)
        ols_big.append(np.sum((XtX_inv @ X.T @ e) ** 2))
        XU = X @ U
        for k in thetas:
            y = X @ betas[k] + e
            c_or = np.linalg.solve(XU.T @ XU, XU.T @ y)
            or_big[k].append(np.sum((U @ c_or - betas[k]) ** 2))
    results[n]["_big_ols"] = (np.mean(ols_big), np.std(ols_big) / np.sqrt(big))
    results[n]["_big_or"] = {k: (np.mean(v), np.std(v) / np.sqrt(big)) for k, v in or_big.items()}

print(f"beta norm (||U a0||) = {beta_norm:.3f}; ||beta||^2 = {beta_norm**2:.3f}")
ok = True
for n in ns:
    res = results[n]
    ols = res[("ols", None, None)]
    f_ols = d / (n - d - 1)
    f_or0 = r / (n - r - 1)
    print(f"n = {n}: OLS risk {ols:.4f} (formula d/(n-d-1) = {f_ols:.4f})")
    se = res["_se"]
    big_ols, big_ols_se = res["_big_ols"]
    ok_ols = abs(big_ols - f_ols) < 4 * big_ols_se + 0.01 * f_ols
    ok &= ok_ols
    print(
        f"   [check, 20,000 reps] OLS {big_ols:.4f} (se {big_ols_se:.4f}) "
        f"vs formula {f_ols:.4f}: {ok_ols}"
    )
    o0 = res[("oracle", None, "0")]
    ok_o0 = abs(res["_big_or"]["0"][0] - f_or0) < 4 * res["_big_or"]["0"][1] + 0.01 * f_or0
    ok &= ok_o0
    print(
        f"   [check, 20,000 reps] oracle theta=0 {res['_big_or']['0'][0]:.4f} "
        f"vs formula {f_or0:.4f}: {ok_o0}"
    )
    print(f"   oracle U, theta=0: {o0:.4f} (500 reps); formula r/(n-r-1) = {f_or0:.4f}")
    for k in thetas:
        o = res[("oracle", None, k)]
        bias2 = np.sin(thetas[k]) ** 2 * beta_norm**2
        formula = (1 + bias2) * f_or0 + bias2
        bo, bo_se = res["_big_or"][k]
        ok_k = abs(bo - formula) < 4 * bo_se + 0.01 * formula
        ok &= ok_k
        print(
            f"   oracle U, theta={k}: {o:.4f} (500 reps); 20,000 reps {bo:.4f} (se {bo_se:.4f}) vs "
            f"(1 + sin^2||beta||^2) r/(n-r-1) + sin^2||beta||^2 = {formula:.4f}: {ok_k}"
        )
    line = "   aux risk (500 reps; columns theta = 0, pi/8, pi/4, pi/2):"
    print(line)
    for K in Ks[1:]:
        vals = [res[("aux", K, k)] for k in thetas]
        print(
            f"     K={K:2d}: " + "  ".join(f"{v:7.4f}" for v in vals) + "   vs OLS " + f"{ols:.4f}"
        )
assert ok, "OLS or oracle value outside the stated approximations"
print("PASS s11d (OLS and oracle checks; the aux table is the result)")
