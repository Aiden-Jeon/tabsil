"""An IterableDataset that endlessly feeds training data to a DataLoader. Use it with
`DataLoader(batch_size=None)`.

Splitting seeds so that every worker, GPU and resume point sees different datasets is a training
concern, so it lives here; a prior only needs to be built from a single seed.

- `PriorBatches`: for the shared training loop (`tabsil.training.loop`). One `PriorBatch` from the
  prior's `sample_batch` is one forward batch (the prior decides the number of train rows)
"""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import IterableDataset, get_worker_info


class PriorBatches(IterableDataset):
    """Endlessly generates prior batches, with a different seed per DataLoader worker.

    `config` is the training config (`TrainConfig`: `make_prior(task, seed)` and `micro_batch`),
    `task` the task of the prior."""

    def __init__(self, config: Any, task: str, seed: tuple[int, ...]) -> None:
        self.config = config
        self.task = task
        self.seed = seed

    def __iter__(self):
        info = get_worker_info()
        worker = 0 if info is None else info.id
        torch.set_num_threads(1)
        seed = hash((*self.seed, worker)) % (2**31)
        prior = self.config.make_prior(self.task, seed)
        while True:
            batch = prior.sample_batch(self.config.micro_batch)
            yield batch.x, batch.y, batch.n_train
