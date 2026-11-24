"""Training stability: collapse detection."""

from __future__ import annotations

import math


class CollapseGuard:
    """Detects a collapse: the train loss jumps suddenly and does not come back.

    In full-scale runs both classification and regression have collapsed to a constant output
    within a few hundred steps, without any grad spike. Rerunning with the same seed did not
    collapse (GPU nondeterminism), so we detect and rewind instead of trying to prevent it. Every
    log_every steps it takes the window mean loss; `patience` consecutive windows more than
    `threshold` above the median of recent healthy windows count as a collapse. Every GPU sees the
    same mean (after all-reduce), so every GPU makes the same decision.
    """

    def __init__(self, threshold: float, patience: int, n_windows: int = 20, min_windows: int = 5):
        self.threshold, self.patience, self.min_windows = threshold, patience, min_windows
        self.healthy: list[float] = []  # recent healthy window means
        self.n_windows = n_windows
        self.bad = 0  # consecutive bad windows

    def update(self, window_mean: float) -> bool:
        """Add a window mean; True if this is judged a collapse.

        A window without a single finite loss (inf) counts as bad.
        """
        bad = not math.isfinite(window_mean)
        if not bad and len(self.healthy) >= self.min_windows:
            ref = sorted(self.healthy)[len(self.healthy) // 2]
            bad = window_mean > ref + self.threshold
        if bad:
            self.bad += 1
            return self.bad >= self.patience
        self.bad = 0
        self.healthy = (self.healthy + [window_mean])[-self.n_windows :]
        return False

    @property
    def suspect(self) -> bool:
        return self.bad > 0
