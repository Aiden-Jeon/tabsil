"""Check for v2-task-formulations.md section 11.5: compute equivalence. Under the section's
assumption (equal prior loss means equal real error), a channel that multiplies the effective
training data by (1 + kappa) does what (1 + kappa) times the training does.

Implied geometric error ratio G for kappa in {0.25, 0.5, 1, 2, 3.6}, under three anchors, each as a
power law G = (1 + kappa)^(ln G0 / ln R). The printed Elo is the rule of 40 Elo per 1% of error
(Elo ~ 4000 (1 - G)), which holds for changes of #14's kind and not for broad ones; the live Elo of
section 11.5 comes from `training/scripts/gap_report.py` (`whatif_share.csv`), not from here.
  (i) second half of stage 2 within one schedule: G0 = 0.993 for R = 2 (it overstates stage 2);
  (ii)  Kumo-Tabular S -> M: R = 4.57, G0 = 0.951;
  (iii) Kumo-Tabular M -> L: R = 3.22, G0 = 0.9785.
Pure arithmetic: the exponents are printed and the anchors are reproduced exactly (G = G0 at kappa =
R - 1). Tolerance: 1e-12 on the reproduction of the anchors.
"""

import math

anchors = {
    "(i) stage 2 second half": (0.993, 2.0),
    "(ii) Kumo S->M": (0.951, 4.57),
    "(iii) Kumo M->L": (0.9785, 3.22),
}
kappas = [0.25, 0.5, 1.0, 2.0, 3.6]

print(
    "exponent e = ln G0 / ln R; G = (1 + kappa)^e; "
    "Elo ~ 4000 (1 - G), the 40-per-1% rule, not live Elo"
)
for name, (G0, R) in anchors.items():
    e = math.log(G0) / math.log(R)
    gs = [(1 + k) ** e for k in kappas]
    cells = [f"k={k}: {g:.4f} ({4000 * (1 - g):4.0f})" for k, g in zip(kappas, gs, strict=True)]
    print(f"{name:26s} e = {e:+.5f} | " + " | ".join(cells))
    # anchor: at (1 + kappa) = R the power law returns G0 (the exponent is defined by this)
    assert abs(R**e - G0) < 1e-12
    # monotone: more data, smaller error ratio; a channel of kappa > 0 never reaches
    # G < G0^(log(1+k)/log R) by construction
    assert all(b < a for a, b in zip(gs, gs[1:], strict=False))
    assert all(g <= 1 for g in gs)
print("PASS s11e (arithmetic: the anchors are reproduced at 1 + kappa = R; G falls with kappa)")
