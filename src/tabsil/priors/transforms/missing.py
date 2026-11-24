"""Missing values (TabPFN v2 paper, "Post-processing"): in some datasets, erase each cell with the
same probability (MCAR)."""

from __future__ import annotations

import torch

from tabsil.priors.pipeline import Group, Rng


class Missing:
    """With probability `dataset_prob` per dataset, set a fraction r ~ U(0, max_rate) of x to NaN.

    The v2 paper does not give these two values; they are our choice in the graph SCM prior
    (`missing_dataset_prob`, `max_missing_rate`).
    """

    def __init__(self, dataset_prob: float, max_rate: float) -> None:
        self.dataset_prob = dataset_prob
        self.max_rate = max_rate

    def __call__(self, group: Group, rng: Rng) -> Group:
        r = rng.numpy
        xs = []
        for d in range(group.x.shape[1]):
            x = group.x[:, d]
            if r.random() < self.dataset_prob:
                rate = r.uniform(0.0, self.max_rate)
                mask = torch.from_numpy(r.random(tuple(x.shape)) < rate).to(x.device)
                x = x.masked_fill(mask, float("nan"))
            xs.append(x)
        return group.with_tensors(torch.stack(xs, 1), group.y)
