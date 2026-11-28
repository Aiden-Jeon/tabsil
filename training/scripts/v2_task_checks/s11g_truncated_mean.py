"""Check for v2-task-formulations.md section 11.6: the mean read from 999 quantiles, and the bias it
adds (the "mean head" price).

The output has quantiles q_k at the levels k/1000, k = 1..999, with the mass beyond the 0.001 and
0.999 quantiles put at those quantiles. Two mean readings:
  step (the output's own distribution: each 0.001 bucket at its upper quantile, the top tail at
  q_999):
       mean_step = (sum_{k=1}^{999} q_k + q_999) / 1000;
  trapezoid over the 999 quantiles, with the tails at q_1 and q_999:
       mean_trap = (sum_{k=1}^{999} q_k + (q_1 + q_999)/2) / 1000.
Distributions (each scaled, not centred, to sd 1, so the true mean is E[X]/sd(X) in sd units):
  standard normal (mean 0); Student t with 3 and 5 df scaled to sd 1 (mean 0);
  lognormal with sigma = 0.5, 1, 2 (true mean e^(sigma^2/2) / sd).
Bias b = output mean - true mean (in sd units). For a prediction that is otherwise exact, the
expected squared error of the mean is 1 + b^2 (sd units), so the head adds b^2 of MSE, a share
b^2/(1 + b^2). Quantiles are exact (scipy ppf), so this isolates the reading of the 999 values.
Checks: the normal's step bias is 0.0031 (= q_999/1000, Phi^{-1}(0.999)/1000) and its trapezoid bias
is 0; printed for every distribution; no tolerance is stated in the brief, so the printed values are
the result.
"""

import numpy as np
from scipy import stats

levels = np.arange(1, 1000) / 1000.0


def out_means(q):
    step = (q.sum() + q[-1]) / 1000.0
    trap = (q.sum() + (q[0] + q[-1]) / 2.0) / 1000.0
    return step, trap


cases = [("normal", stats.norm(), None)]
for df in (3, 5):
    # sd of t_df is sqrt(df/(df-2)); scale it to 1
    cases.append((f"t{df}", stats.t(df, scale=np.sqrt((df - 2) / df)), None))
for s in (0.5, 1.0, 2.0):
    ln = stats.lognorm(s=s)
    sd = np.sqrt((np.exp(s**2) - 1) * np.exp(s**2))
    cases.append((f"lognormal s={s}", stats.lognorm(s=s, scale=1.0 / sd), np.exp(s**2 / 2) / sd))

print(
    f"{'distribution':18s} {'true mean':>10s} {'step mean':>10s} {'bias step':>10s} "
    f"{'bias trap':>10s} "
    f"{'b^2':>9s} {'share of MSE':>13s}"
)
for name, dist, true_mean in cases:
    q = dist.ppf(levels)
    tm = dist.mean() if true_mean is None else true_mean
    step, trap = out_means(q)
    b_step = step - tm
    b_trap = trap - tm
    share = b_step**2 / (1 + b_step**2)
    print(
        f"{name:18s} {tm:10.4f} {step:10.4f} {b_step:+10.4f} {b_trap:+10.4f} "
        f"{b_step**2:9.2e} {share:13.2e}"
    )
q_norm = stats.norm.ppf(levels)
bias_norm = (q_norm.sum() + q_norm[-1]) / 1000.0
print(f"normal step bias {bias_norm:.5f} = Phi^-1(0.999)/1000 = {stats.norm.ppf(0.999) / 1000:.5f}")
assert abs(bias_norm - stats.norm.ppf(0.999) / 1000) < 1e-9
assert abs(out_means(q_norm)[1]) < 1e-9
print("PASS s11g (normal identity; the other rows are the result)")
