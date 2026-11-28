"""Check for v2-task-formulations.md section 11.2: what a prior part that fixes m of 51 datasets is
worth.

If m datasets get their error multiplied by (1 - x) and the other 51 - m are unchanged, the
geometric mean of the error ratios is G = (1 - x)^(m / 51). Elo is read as "40 Elo per 1% of error"
(a 1% reduction of the geometric mean error = 40 Elo), so Elo ~ 40 * 100 * (1 - G) = 4000 (1 - G).

Output: the table of G and Elo for m in {1, 2, 4, 8, 13, 26, 51} and x in {0.05, 0.1, 0.2, 0.4}.
Check: a direct geometric mean over a simulated vector of 51 error ratios (seed 0; 2,000 random
subsets of size m, random baseline errors) agrees with the formula to 1e-12. Arithmetic; no
tolerance is stated in the brief beyond this identity.
"""

import numpy as np

ms = [1, 2, 4, 8, 13, 26, 51]
xs = [0.05, 0.1, 0.2, 0.4]
print("G = (1 - x)^(m/51)    and Elo ~ 4000 (1 - G)")
print("m \\ x " + "".join(f"{x:>18}" for x in xs))
for m in ms:
    cells = []
    for x in xs:
        G = (1 - x) ** (m / 51)
        cells.append(f"{G:.4f} ({4000 * (1 - G):5.0f})")
    print(f"{m:>5} " + "".join(f"{c:>18}" for c in cells))

rng = np.random.default_rng(0)
max_dev = 0.0
for _ in range(2000):
    base = np.exp(rng.normal(0, 0.5, size=51))  # baseline errors of the 51 datasets
    m = int(rng.choice(ms))
    x = float(rng.choice(xs))
    fixed = rng.choice(51, size=m, replace=False)
    new = base.copy()
    new[fixed] *= 1 - x
    geo = np.exp(np.mean(np.log(new / base)))
    max_dev = max(max_dev, abs(geo - (1 - x) ** (m / 51)))
print(f"max |direct geometric mean - (1 - x)^(m/51)| over 2000 random cases: {max_dev:.2e}")
assert max_dev < 1e-12
print("PASS s11b")
