"""Quantile regression output: the regression predictive distribution as the values q_k at fixed
quantile levels τ_k.

The regression output of Kumo-Tabular (NVIDIA; blog and public source
NVIDIA/structured-data-models) is 999 quantiles (q001..q999) trained with the quantile (pinball)
loss. Only that behavior is reimplemented here.

- Levels: τ_k = k / (n + 1), k = 1..n (0.001 .. 0.999 for n = `N_QUANTILES` = 999)
- Loss (`pinball_loss`): the mean over levels of ρ_τ(y − q) = max(τ (y − q), (τ − 1)(y − q)); 0 for
  a missing target (as in the bar distribution loss)
- Initial bias (`initial_bias`): the standard normal quantiles Φ⁻¹(τ_k). Training targets are
  z-normalized, so the first prediction is a standard normal distribution (our choice)
- Inference (`quantiles_to_bucket_probs`): the quantiles are sorted (removing crossings, as Kumo
  does) and, with the CDF taken as linear between quantiles, turned into bucket probabilities on
  the bar distribution borders. Inference code that consumes bar distribution outputs (mean,
  target-transform members, distributional metrics) is thus reused unchanged. The mass outside the
  lowest and the highest quantile (1/(n + 1) each) goes to the bucket holding that quantile (our
  choice)
- The other direction (`bucket_quantiles`): quantiles of a bucket distribution, with the density
  taken as uniform within a bucket. It is applied to the ensemble's predictive distribution
  (numpy, float64): quantile forecasts (`tabsil.tasks.forecast`), medians and draws
  (`tabsil.tasks.unsupervised`)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

N_QUANTILES = 999
# Cap on the (rows x borders) elements computed at once when turning quantiles into bucket
# probabilities (256 MiB in float32)
CHUNK_ELEMENTS = 1 << 26


def quantile_levels(
    n: int = N_QUANTILES, device: torch.device | str | None = None, dtype=torch.float32
) -> torch.Tensor:
    """Levels τ_k = k / (n + 1), k = 1..n."""
    return torch.arange(1, n + 1, device=device, dtype=dtype) / (n + 1)


def initial_bias(n: int = N_QUANTILES) -> torch.Tensor:
    """Standard normal quantiles Φ⁻¹(τ_k) (the initial output-layer bias)."""
    return torch.special.ndtri(quantile_levels(n, dtype=torch.float64)).float()


def pinball_loss(q: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Quantile loss: q (..., n), y (...) -> (...), the mean over the n levels. 0 if y is
    missing."""
    q = q.float()
    y = y.reshape(q.shape[:-1]).to(q.device).float()
    missing = torch.isnan(y)
    diff = torch.where(missing, 0.0, y)[..., None] - q  # (..., n)
    tau = quantile_levels(q.shape[-1], q.device)
    loss = torch.maximum(tau * diff, (tau - 1) * diff).mean(-1)
    return torch.where(missing, torch.zeros_like(loss), loss)


def quantiles_to_bucket_probs(q: torch.Tensor, borders: torch.Tensor) -> torch.Tensor:
    """Quantiles q (..., n) -> bucket probabilities (..., K) on `borders` (K + 1,) (module
    docstring).

    The quantiles are sorted and clipped to the border range; the CDF at a border b is F(b) = 0
    (b < q_1), 1 (b > q_n), and in between the linear interpolation of the points (q_k, τ_k). A
    bucket probability is the difference of F at neighboring borders. Rows are independent, so they
    are computed in chunks of `CHUNK_ELEMENTS` (same result). Computed in float32."""
    shape = q.shape
    q = q.float().reshape(-1, shape[-1])
    borders = borders.float().to(q.device)
    q = q.sort(-1).values.clamp(borders[0], borders[-1])
    n = q.shape[-1]
    tau = quantile_levels(n, q.device)
    step = max(1, CHUNK_ELEMENTS // borders.numel())
    out = []
    for i in range(0, q.shape[0], step):
        qi = q[i : i + step]
        b = borders.expand(qi.shape[0], -1).contiguous()
        # hi: index of the first quantile >= b (0 if b <= q_1, n if b > q_n)
        hi = torch.searchsorted(qi.contiguous(), b, side="left")
        lo_i, hi_i = (hi - 1).clamp(0, n - 1), hi.clamp(0, n - 1)
        q_lo, q_hi = qi.gather(-1, lo_i), qi.gather(-1, hi_i)
        width = q_hi - q_lo
        frac = torch.where(width > 0, (b - q_lo) / width.clamp_min(1e-30), 1.0).clamp(0, 1)
        cdf = tau[lo_i] + frac * (tau[hi_i] - tau[lo_i])
        cdf = torch.where(hi == 0, 0.0, torch.where(hi == n, 1.0, cdf))
        # The last border is at or above the top quantile (quantiles were clipped to the
        # borders): all mass is below it
        cdf[:, -1] = 1.0
        out.append((cdf[:, 1:] - cdf[:, :-1]).clamp_min(0.0))
    return torch.cat(out).reshape(*shape[:-1], borders.numel() - 1)


def bucket_quantiles(
    log_probs: torch.Tensor | np.ndarray, borders: torch.Tensor | np.ndarray, levels: Any
) -> np.ndarray:
    """Quantiles (n, m) of bucket distributions: log probabilities (n, K) on `borders`
    (K + 1,), with the density uniform within each bucket (the CDF is linear between borders).
    `levels` is m levels shared by every row, or an (n, m) array of levels per row (uniform
    levels give draws: `tabsil.unsupervised.NumericConditional.sample`). A level q falls in the
    bucket k with cdf[k] < q <= cdf[k + 1], so a bucket without probability is never chosen."""
    probs = np.exp(np.asarray(log_probs, dtype=np.float64))
    probs = probs / probs.sum(-1, keepdims=True)
    b = np.asarray(borders, dtype=np.float64)
    cdf = np.concatenate([np.zeros((len(probs), 1)), np.cumsum(probs, -1)], axis=-1)
    cdf[:, -1] = 1.0
    levels = np.asarray(levels, dtype=np.float64)
    per_row = levels.ndim == 2
    rows = np.arange(len(probs))
    out = np.empty((len(probs), levels.shape[-1]))
    for j in range(levels.shape[-1]):
        q = levels[:, j] if per_row else np.full(len(probs), levels[j])
        k = np.clip((cdf < q[:, None]).sum(-1) - 1, 0, len(b) - 2)
        p = probs[rows, k]
        share = np.where(p > 0, (q - cdf[rows, k]) / np.where(p > 0, p, 1.0), 0.0)
        out[:, j] = b[k] + np.clip(share, 0.0, 1.0) * (b[k + 1] - b[k])
    return out
