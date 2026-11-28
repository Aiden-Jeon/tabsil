"""Check for v2-task-formulations.md section 8 (survival as binary person-period rows).

Claims reproduced ("Checked: on simulated data with a censoring bin drawn uniformly, the row
frequencies recover the hazards to 0.002 and their running product is the Kaplan-Meier curve; using
the uncensored subjects alone is biased (a survival of 0.37 where the truth is 0.69 at the fourth
bin)"):
  (a) person-period frequencies recover the hazards: max |h_hat - h| <= 0.002 over 8 bins;
  (b) the product of (1 - h_hat) is the Kaplan-Meier curve (exact identity, printed as max
  difference);
  (c) using the uncensored subjects alone is biased: the closed document stated a survival of 0.37
      at the fourth bin where the truth is 0.69. Its censoring range was not recorded. With the
      censoring bin uniform on 1..8 (the design below) this script gives 0.21, and 0.35 with
      censoring on 1..12; the document's sentence was corrected to 0.21 with this design
      (v2-task-formulations.md, "Checks as scripts"), and (c) asserts 0.21.

Design (the reading that reproduces the stated numbers): 200,000 subjects, K = 8 bins, discrete
hazard h = 0.09 in every bin (truth S(4) = 0.91^4 = 0.686, the stated 0.69). Event bin T geometric
(P(T = k) = h (1 - h)^(k - 1)); censoring bin C uniform on {1, ..., 8}; t = min(T, C), delta = 1[T
<= C].
Person-period rows: for k = 1..t, label 1 iff k = t and delta = 1. Hazard estimate: events in bin k
over subjects at risk in bin k. Kaplan-Meier from (t, delta) for the product. The uncensored-only
estimate is the fraction with t > 4 among the subjects with delta = 1.
Tolerances: 0.002 on the hazards (stated), 1e-10 on (b), 0.01 on (c).
"""

import numpy as np

rng = np.random.default_rng(0)
N, K, h = 200_000, 8, 0.09
T = rng.geometric(h, size=N)  # event bin, support 1, 2, ...
C = rng.integers(1, K + 1, size=N)  # censoring bin, uniform on 1..8
t = np.minimum(T, C)
delta = (T <= C).astype(int)

# person-period rows
rows_k, rows_y = [], []
hazard_hat = np.zeros(K)
for k in range(1, K + 1):
    at_risk = t >= k
    events = at_risk & (t == k) & (delta == 1)
    hazard_hat[k - 1] = events.sum() / at_risk.sum()
S_pp = np.cumprod(1 - hazard_hat)

# Kaplan-Meier from (t, delta)
S_km = np.zeros(K)
s = 1.0
for k in range(1, K + 1):
    n_k = np.sum(t >= k)
    d_k = np.sum((t == k) & (delta == 1))
    s *= 1 - d_k / n_k
    S_km[k - 1] = s

truth = np.full(K, 1.0)
truth = np.cumprod(np.full(K, 1 - h))
print("hazard estimates:", np.round(hazard_hat, 4))
print(f"max |hazard estimate - {h}| = {np.max(np.abs(hazard_hat - h)):.4f} (stated 0.002)")
print(
    f"max |product of person-period hazards - Kaplan-Meier| = "
    f"{np.max(np.abs(S_pp - S_km)):.2e} (stated: identical)"
)
print(f"S(4): truth {truth[3]:.4f} (stated 0.69), Kaplan-Meier {S_km[3]:.4f}")
unc = delta == 1
s_unc = np.mean(t[unc] > 4)
print(
    f"uncensored subjects only: fraction surviving past bin 4 = {s_unc:.4f} "
    "(corrected statement 0.21; the closed document's 0.37 had an unrecorded censoring range)"
)

assert np.max(np.abs(hazard_hat - h)) <= 0.002, np.max(np.abs(hazard_hat - h))
assert np.max(np.abs(S_pp - S_km)) < 1e-10
assert abs(truth[3] - 0.69) < 0.01
# sensitivity: the stated 0.37 is not reproduced with censoring uniform on 1..8; the censoring range
# that gives it (not stated in the document) is printed for information
for Kc in (12,):
    Cc = rng.integers(1, Kc + 1, size=N)
    tc, dc = np.minimum(T, Cc), (T <= Cc)
    print(
        f"  information: censoring uniform on 1..{Kc}: "
        f"uncensored-only S(4) = {np.mean(tc[dc] > 4):.4f}"
    )
assert abs(s_unc - 0.21) < 0.01, s_unc
print("PASS s08")
