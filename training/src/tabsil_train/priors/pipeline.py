"""Prior assembler: hyperparameters -> generator -> transforms -> `PriorBatch`.

Tabular priors go through the same stages: fix the table size, draw hyperparameters for each group
(datasets sharing their hyperparameters), build a continuous table with a generator, and refine it
with transforms. A new prior usually adds one generator (the graph SCM prior's is
`tabsil_train.priors.graph_scm.generator`) or a few transforms (`tabsil_train.priors.transforms`)
and reuses the existing parts for the other stages.

- Table size (train rows, total rows, features) is chosen by the assembled prior and passed in
  (`tabsil_train.priors.table_size` and others).
- Hyperparameters: `(rng, n_train, n_features) -> dict`, drawn once per group (the class count,
  `tabsil_train.priors.graph_scm.ClassCount`). If they set the used feature count
  `num_features_used`, the generator builds that many and a transform (`PadFeatures`) fills the
  rest.
- Generator (`Generator`): the (x, y) of one group.
- Transform (`Transform`): takes a group and returns a new one. None drops the group, and sampling
  restarts from the hyperparameters (at most `max_attempts` times).

Tensors inside a group are ordered (rows, datasets, features) and (rows, datasets), as in the
TabPFN v1 and original PFN code, and become (datasets, rows, features) in the `PriorBatch`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Protocol

import numpy as np
import torch

from tabsil_train.priors.base import PriorBatch


@dataclass
class Rng:
    """Random number generators shared by the parts. Global random state is not used (except by
    generators that wrap an external package)."""

    numpy: np.random.Generator
    torch: torch.Generator
    device: torch.device

    @classmethod
    def from_seed(cls, seed: int, device: torch.device | str = "cpu") -> Rng:
        device = torch.device(device)
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)
        return cls(np.random.default_rng(seed), gen, device)


@dataclass
class Group:
    x: torch.Tensor  # (rows, datasets, features)
    y: torch.Tensor  # (rows, datasets)
    n_train: int  # the first n_train rows are train rows
    hp: dict  # hyperparameters shared by the group

    def with_tensors(self, x: torch.Tensor, y: torch.Tensor) -> Group:
        """The group with new x and y (the rest unchanged)."""
        return replace(self, x=x, y=y)


class Generator(Protocol):
    def __call__(
        self, n_rows: int, n_features: int, n_datasets: int, hp: dict, rng: Rng
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(x (n_rows, n_datasets, n_features), y (n_rows, n_datasets))."""
        ...


class Transform(Protocol):
    def __call__(self, group: Group, rng: Rng) -> Group | None: ...


Hyperparameters = Callable[[Rng, int, int], dict]


class Pipeline:
    """A prior assembled from parts. `sample_group` gives one group, `sample_batch` a batch of
    concatenated groups."""

    def __init__(
        self,
        generator: Generator,
        *,
        rng: Rng,
        hyperparameters: Hyperparameters | None = None,
        transforms: Sequence[Transform] = (),
        group_size: int = 1,
        max_attempts: int = 1,
    ) -> None:
        if group_size < 1 or max_attempts < 1:
            raise ValueError(
                f"group_size {group_size} and max_attempts {max_attempts} must be at least 1"
            )
        self.generator = generator
        self.rng = rng
        self.hyperparameters = hyperparameters
        self.transforms = list(transforms)
        self.group_size = group_size
        self.max_attempts = max_attempts

    def sample_group(
        self,
        n_rows: int,
        n_train: int,
        n_features: int,
        transforms: Sequence[Transform] | None = None,
    ) -> Group:
        """One group. `transforms`, if given, replaces the configured transforms (to run only some
        of the stages)."""
        transforms = self.transforms if transforms is None else transforms
        for _ in range(self.max_attempts):
            hp = self.hyperparameters(self.rng, n_train, n_features) if self.hyperparameters else {}
            n_used = hp.get("num_features_used", n_features)
            x, y = self.generator(n_rows, n_used, self.group_size, hp, self.rng)
            group: Group | None = Group(x, y, n_train, hp)
            for transform in transforms:
                group = transform(group, self.rng)
                if group is None:
                    break
            if group is not None:
                return group
        raise RuntimeError(
            f"no dataset produced after {self.max_attempts} attempts ({n_train}, {n_features})"
        )

    def sample_batch(
        self, batch_size: int, n_rows: int, n_train: int, n_features: int
    ) -> PriorBatch:
        """`batch_size` datasets (must be a multiple of group_size)."""
        if batch_size < 1 or batch_size % self.group_size:
            raise ValueError(
                f"batch_size {batch_size} must be a positive multiple of group_size"
                f" {self.group_size}"
            )
        n_groups = batch_size // self.group_size
        groups = [self.sample_group(n_rows, n_train, n_features) for _ in range(n_groups)]
        x = torch.cat([g.x.float() for g in groups], dim=1)
        y = torch.cat([g.y.float() for g in groups], dim=1)
        return PriorBatch(x.transpose(0, 1).contiguous(), y.transpose(0, 1).contiguous(), n_train)
