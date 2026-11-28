"""Check for v2-task-formulations.md section 2, step 7 (classification with the covering split).

Uses the repository's own function `train_rows_cover_classes` from
training/src/tabsil_train/priors/transforms/classes.py (run with `uv run --frozen --package
tabsil-train python -I ...`, the only check that imports repository code).

Claims reproduced (the "Checked by running the prior's own function" sentence):
  Beta(1, 1), table of 20 context rows with one positive among them, and 1 or 10 queries:
    covering split 0.049 and 0.050 (stated), random split 0.094 (stated);
    formulas: covering 0.050, random 0.091 (posterior predictive (a + n1)/(a + b + n)).
  Beta(2, 5), 10 context rows with one positive: covering 0.107 (stated) against the formula 0.106;
    with 6 queries 0.111 (stated; the formula is approximate there).

Design: a table of N = n + n_Q rows; theta ~ Beta(a, b) per table; labels Bernoulli(theta); the
repository function orders the rows (the first n train rows are the context, the rest the
queries). Tables where the function returns None (fewer than 2 classes) are dropped as in training.
P(query positive | context holds n1 positives) is the pooled frequency over queries of the tables
whose context has n1 positives. The random split is the same conditioning with a permutation.
Seeds: numpy default_rng(0) for the labels and the splits (the function receives the generator).
Number of tables: TABLES = 60,000 per configuration (reduced from 300,000 for the 10 s budget, so
the standard error of the 1-query figures is about 0.004); the Monte Carlo standard error of each
conditional frequency is printed. Tolerance: 0.005 (absolute).
"""

import time

import numpy as np
import torch

from tabsil_train.priors.transforms.classes import train_rows_cover_classes

TABLES = 60_000  # reduced from 300,000 to keep the run near 10 s (the function is per table)
rng = np.random.default_rng(0)


def covering_formula(a, b, n, n1, nq=1):
    n0 = n - n1
    odds = (a + n1) / (b + n0) * (n1 / (n1 + 1)) / (n0 / (n0 + 1))
    return odds / (1 + odds)


def random_formula(a, b, n, n1):
    return (a + n1) / (a + b + n)


def simulate(a, b, n, nq, tables):
    """Pooled query positive frequency by context positive count, for both splits."""
    N = n + nq
    cov_hits = np.zeros(n + 1)
    cov_tot = np.zeros(n + 1)
    rnd_hits = np.zeros(n + 1)
    rnd_tot = np.zeros(n + 1)
    for _ in range(tables):
        theta = rng.beta(a, b)
        labels = (rng.random(N) < theta).astype(np.int64)
        yt = torch.from_numpy(labels)
        order = train_rows_cover_classes(yt, n, rng)
        if order is not None:
            o = order.numpy()
            k = int(labels[o[:n]].sum())
            cov_hits[k] += labels[o[n:]].sum()
            cov_tot[k] += nq
        perm = rng.permutation(N)
        k = int(labels[perm[:n]].sum())
        rnd_hits[k] += labels[perm[n:]].sum()
        rnd_tot[k] += nq
    return cov_hits, cov_tot, rnd_hits, rnd_tot


def report(name, a, b, n, nq, n1=1):
    t0 = time.time()
    ch, ct, rh, rt = simulate(a, b, n, nq, TABLES)
    cov = ch[n1] / ct[n1]
    rnd = rh[n1] / rt[n1]
    se = np.sqrt(cov * (1 - cov) / ct[n1])
    print(
        f"{name}: n={n} n1={n1} nq={nq}: covering {cov:.4f} (+-{se:.4f}), "
        f"random {rnd:.4f}  [{ct[n1] / nq:.0f} tables, {time.time() - t0:.1f}s]"
    )
    return cov, rnd, se


c1, r1, se1 = report("Beta(1,1)", 1, 1, 20, 1)
c10, r10, se10 = report("Beta(1,1)", 1, 1, 20, 10)
print(
    f"formulas Beta(1,1) n=20 n1=1: covering {covering_formula(1, 1, 20, 1):.4f}, "
    f"random {random_formula(1, 1, 20, 1):.4f}"
)
c25, _, se25 = report("Beta(2,5)", 2, 5, 10, 1)
c25q6, _, se25q6 = report("Beta(2,5)", 2, 5, 10, 6)
print(f"formula Beta(2,5) n=10 n1=1: covering {covering_formula(2, 5, 10, 1):.4f}")

tol = 0.005
checks = [
    ("covering, 1 query, Beta(1,1)", c1, 0.049),
    ("covering, 10 queries, Beta(1,1)", c10, 0.050),
    ("random, Beta(1,1)", r1, 0.094),
    ("covering, Beta(2,5), 1 query", c25, 0.107),
    ("covering, Beta(2,5), 6 queries", c25q6, 0.111),
]
ok = True
for name, got, stated in checks:
    good = abs(got - stated) < tol
    ok &= good
    print(f"{name}: stated {stated:.3f}, got {got:.4f}, {'ok' if good else 'MISMATCH'}")
assert abs(covering_formula(1, 1, 20, 1) - 0.050) < 6e-4
assert abs(random_formula(1, 1, 20, 1) - 0.091) < 6e-4
assert abs(covering_formula(2, 5, 10, 1) - 0.106) < 6e-4
assert ok, "a stated simulated value was not reproduced (see MISMATCH lines)"
print("PASS s02c")
