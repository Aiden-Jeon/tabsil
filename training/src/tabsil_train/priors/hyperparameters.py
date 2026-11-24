"""Hyperparameter sampling (parts for the hyperparameter stage of `Pipeline`).

- `ClassCount`: the class count of a classification dataset, bounded by its train row count.
"""

from __future__ import annotations

from tabsil_train.priors.pipeline import Rng


class ClassCount:
    """Class count `num_classes` ~ U{2, ..., min(max_classes, train rows)}.

    Every class must appear among the train rows, so the class count never exceeds the train row
    count. With `binary_prob` > 0, 2 is picked first with that probability, otherwise the count is
    drawn from the uniform distribution above (TabICLv2 `PriorDataset` uses 0.5, giving binary
    tasks 0.5 + 0.5/9 ≈ 56% of the time). With 0, no extra random number is drawn, so the datasets
    are the same as without the option.
    """

    def __init__(self, max_classes: int, binary_prob: float = 0.0) -> None:
        if not 0.0 <= binary_prob <= 1.0:
            raise ValueError(f"binary_prob must be between 0 and 1: {binary_prob}")
        self.max_classes = max_classes
        self.binary_prob = binary_prob

    def __call__(self, rng: Rng, n_train: int, n_features: int) -> dict:
        if self.binary_prob and rng.numpy.random() < self.binary_prob:
            return {"num_classes": 2}
        max_classes = min(self.max_classes, n_train)
        return {"num_classes": int(rng.numpy.integers(2, max_classes + 1))}
