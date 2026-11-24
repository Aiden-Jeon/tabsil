"""Class handling: order the rows so that the held-out rows' classes all appear in the train rows
(`train_rows_cover_classes`, from the graph SCM prior), applied to a group by `CoverClasses`.
"""

from __future__ import annotations

import numpy as np
import torch

from tabsil_train.priors.pipeline import Group, Rng


def train_rows_cover_classes(
    y: torch.Tensor, n_train: int, rng: np.random.Generator
) -> torch.Tensor | None:
    """Order the rows so that every class of the held-out rows also appears in the train rows.

    One row per class goes into the train rows first; the remaining train rows are filled at
    random. Returns None (the dataset is dropped) with fewer than 2 classes or more classes than
    train rows.
    """
    labels = y.long().cpu().numpy()
    classes = np.unique(labels)
    if len(classes) < 2 or len(classes) > n_train:
        return None
    first = np.array([rng.choice(np.flatnonzero(labels == c)) for c in classes])
    rest = rng.permutation(np.setdiff1d(np.arange(len(labels)), first))
    n_more = n_train - len(first)
    train = rng.permutation(np.concatenate([first, rest[:n_more]]))
    return torch.from_numpy(np.concatenate([train, rest[n_more:]])).to(y.device)


class CoverClasses:
    """Reorder rows so that every held-out class also appears in the train rows
    (`train_rows_cover_classes`).

    Drops the group (None) if any dataset cannot be arranged this way.
    """

    def __call__(self, group: Group, rng: Rng) -> Group | None:
        xs, ys = [], []
        for d in range(group.y.shape[1]):
            order = train_rows_cover_classes(group.y[:, d], group.n_train, rng.numpy)
            if order is None:
                return None
            xs.append(group.x[order, d])
            ys.append(group.y[order, d])
        return group.with_tensors(torch.stack(xs, 1), torch.stack(ys, 1))
