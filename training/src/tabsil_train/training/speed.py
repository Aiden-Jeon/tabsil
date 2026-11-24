"""Training speed comparison: pure training time to reach a target validation loss.

The nanoTabPFN speedrun (arXiv 2606.03681) measures "time to reach a fixed evaluation score". We
apply the same idea to a training run's validation record: from the `train_seconds` (training
time excluding validation, checkpoints and logging) and `val_loss` (mean of the two tasks for
multitask) that the training loop logs at every validation, return the training time of the first
validation at or below the target. There is no interpolation between validations (the hit is
late by up to one validation interval, but comparisons with the same `val_every` are fair).

    target = val_loss_at(baseline, step=5_000)   # baseline run's validation loss at step 5k
    seconds = time_to_target(candidate, target)  # training time for another config to reach it
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class ValPoint:
    """One validation: step, pure training time so far (seconds), validation loss."""

    step: int
    train_seconds: float
    val_loss: float


def val_loss_at(points: Sequence[ValPoint], step: int) -> float:
    """Validation loss at `step` (ValueError if there was no validation at that step)."""
    for p in points:
        if p.step == step:
            return p.val_loss
    raise ValueError(f"no validation record at step {step} (steps: {[p.step for p in points]})")


def time_to_target(points: Sequence[ValPoint], target: float) -> ValPoint | None:
    """The first validation with loss at or below `target` (None if it is never reached)."""
    for p in sorted(points, key=lambda p: p.step):
        if p.val_loss <= target:
            return p
    return None
