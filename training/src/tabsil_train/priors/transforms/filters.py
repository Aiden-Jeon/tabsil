"""Filtering out unusable groups."""

from __future__ import annotations

from tabsil_train.priors.base import IGNORE_INDEX
from tabsil_train.priors.pipeline import Group, Rng


class DropInvalid:
    """Drop the group (None, draw again) if a dataset has all targets marked ignored (-100).

    The TabICL graph SCM fills y with -100 for datasets in which NaNs appeared.
    """

    def __call__(self, group: Group, rng: Rng) -> Group | None:
        if (group.y == IGNORE_INDEX).all(0).any():
            return None
        return group
