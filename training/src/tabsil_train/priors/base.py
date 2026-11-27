"""The batch a prior hands to the training loop (`tabsil_train.training.loop`), and constants shared
by the priors and the losses."""

from __future__ import annotations

from dataclasses import dataclass

import torch

IGNORE_INDEX = -100  # target ignored by the loss (cross-entropy's ignore_index)


@dataclass
class PriorBatch:
    x: torch.Tensor  # (B, R, F), missing values are NaN
    # (B, R). Classification: class index, regression: z-normalized value (over all rows, or over
    # the train rows with `TrainConfig.target_train_rows`)
    y: torch.Tensor
    n_train: int  # the first n_train rows are train rows, the rest are eval rows (target hidden)

    @property
    def y_train(self) -> torch.Tensor:
        return self.y[:, : self.n_train]

    @property
    def y_eval(self) -> torch.Tensor:
        return self.y[:, self.n_train :]
