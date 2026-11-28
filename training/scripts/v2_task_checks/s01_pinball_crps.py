"""Check for v2-task-formulations.md section 1 (Proposition 1 and the "checked" sentence after the
quadrature identity, line 84 of the document read with `git show 9933bd0:...`).

Claims reproduced:
  (a) for a standard normal F, the mean pinball loss over the 999 levels tau_k = k/1000 against
      half the CRPS: ratio 1.0010 to within 1e-4;
  (b) the integral identity  int_0^1 rho_tau(y - F^{-1}(tau)) dtau = CRPS(F, y)/2  holds to 1e-8;
  (c) the pinball loss at level tau is minimized at the tau-quantile (the "numerical checks" row of
      section 1: five levels of a 50/50 mixture of N(2, 0.5^2) and a t with 4 degrees of freedom).

Design: F = N(0, 1), y in {-2, -1, 0, 1, 2}; CRPS in closed form; the integral by scipy quad.
(c) minimizes the expected pinball loss over q on a grid of quad-evaluated values, using the
mixture density; the numerical minimizer is compared with the mixture's quantile by root finding.
Stated numbers: ratio 1.0010 (tolerance 1e-4), identity 1e-8, minimizer tolerance 1e-4.
"""

import numpy as np
from scipy import integrate, optimize, stats


def rho(tau, u):
    return np.maximum(tau * u, (tau - 1.0) * u)


def crps_normal(y):
    z = y
    return z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1.0 / np.sqrt(np.pi)


levels = np.arange(1, 1000) / 1000.0
ratios = []
max_identity_err = 0.0
for y in (-2.0, -1.0, 0.0, 1.0, 2.0):
    mean_pin = np.mean(rho(levels, y - stats.norm.ppf(levels)))
    crps_half = crps_normal(y) / 2.0
    ratios.append(mean_pin / crps_half)
    integral, _ = integrate.quad(
        lambda t, y=y: rho(t, y - stats.norm.ppf(t)),
        0.0,
        1.0,
        limit=500,
        epsabs=1e-12,
        epsrel=1e-12,
    )
    max_identity_err = max(max_identity_err, abs(integral - crps_half))
    print(
        f"y={y:+.1f}  mean pinball/(CRPS/2) = {mean_pin / crps_half:.6f}   "
        f"quad integral - CRPS/2 = {integral - crps_half:.2e}"
    )

ratio_mean = float(np.mean(ratios))
print(
    f"ratio (mean over y) = {ratio_mean:.6f}, stated 1.0010, expected 1000/999 = {1000 / 999:.6f}"
)
print(f"max |integral - CRPS/2| over y = {max_identity_err:.2e}, stated 1e-8")


# (c) the minimizer of the expected pinball loss under a mixture is the quantile
def mix_pdf(y):
    return 0.5 * stats.norm.pdf(y, 2.0, 0.5) + 0.5 * stats.t.pdf(y, 4)


def mix_cdf(y):
    return 0.5 * stats.norm.cdf(y, 2.0, 0.5) + 0.5 * stats.t.cdf(y, 4)


def expected_pinball(q, tau):
    def f(y):
        return rho(tau, y - q) * mix_pdf(y)

    val, _ = integrate.quad(f, -60, 60, points=[q, 2.0], limit=500, epsabs=1e-11)
    return val


max_min_err = 0.0
for tau in (0.1, 0.25, 0.5, 0.75, 0.9):
    res = optimize.minimize_scalar(
        lambda q, tau=tau: expected_pinball(q, tau),
        bounds=(-5, 6),
        method="bounded",
        options={"xatol": 1e-8},
    )
    quant = optimize.brentq(lambda q, tau=tau: mix_cdf(q) - tau, -20, 20, xtol=1e-12)
    max_min_err = max(max_min_err, abs(res.x - quant))
    print(f"tau={tau:.2f}  argmin of expected pinball = {res.x:.6f}   quantile = {quant:.6f}")
print(f"max |argmin - quantile| = {max_min_err:.2e}, stated tolerance 1e-4")

assert abs(ratio_mean - 1.0010) < 1e-4, ratio_mean
assert max_identity_err < 1e-8, max_identity_err
assert max_min_err < 1e-4, max_min_err
print("PASS s01")
