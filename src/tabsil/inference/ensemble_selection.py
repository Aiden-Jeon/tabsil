"""Member pool + greedy selection (the ensemble selection of Caruana et al. 2004): aggregation
that spends more inference compute for better accuracy. "Test-Time Compute for Tabular Foundation
Models" (arXiv 2610.12005) reports that it lowers the error of TabPFN-3 below the default
prediction.

    clf = GreedyEnsembleClassifier(model, n_estimators=4).fit(x_train, y_train)
    proba = clf.predict_proba(x_test)

Classification only: the regression output is 999 quantiles, which only the basic
`EnsembleRegressor` reads.

What the paper (Appendix C, public code `ttc.aggregation`) does:

- a pool of 96 configurations (number 0 is the default). Each produces out-of-fold predictions
  from stratified 8-fold CV within the training rows; the fold count is lowered when a class has
  fewer than 8 training rows
- greedy: for 40 rounds, pick the candidate whose addition to the running average gives the
  lowest out-of-fold loss (log loss for classification, RMSE for regression). Candidates can be
  picked again, there is no early stopping, and ties go to the lower index. Weights are pick
  count / 40
- the selected configurations are refitted on all training rows, predict the test rows, and are
  blended with those weights
- results (51 TabArena datasets, error change vs. default): greedy −2.44%, single best −1.25%,
  uniform average of all 96 +3.68%

Our choices (the paper's configurations are TabPFN-3 preprocessing options and do not carry over
as is):

- a candidate is one member (one forward pass) combined with one softmax temperature; a paper
  configuration is an 8-member ensemble. The temperature only divides the logits, so it costs no
  extra forward pass
- `pool_size` pool members: the first `n_estimators` are the default ensemble's members (same
  seeds), the rest are random: one-hot on/off, the outlier clipping σ (`OUTLIER_SIGMAS`) and the
  member seed (feature order, class indices, sample identifier column)
- the temperature and σ candidates come from TabArena's TabPFN-2.5 search space
  (`softmax_temperature`, `OUTLIER_REMOVAL_STD`); the paper's pool was built from this search
  space too
- candidate 0 is the default ensemble (default members averaged uniformly at the default
  temperature), playing the role of the paper's configuration 0
- for out-of-fold predictions, member preprocessing is refitted on each fold's training rows
- only members with nonzero weight are refitted on all training rows (less test time)
- `selection`: `"greedy"` (the paper), `"best"` (the single candidate with the lowest out-of-fold
  loss), `"uniform"` (equal weight on all candidates except 0, the paper's uniform average)

Additions beyond the paper. The paper's method selects anything with even a slightly lower
out-of-fold loss. So on datasets whose default out-of-fold loss is already near 0 it picked a
single very sharp temperature and was overconfident on test, and it lost on the distribution shift
of grouped/temporal test splits that random folds cannot see (BeyondArena-lite). To use the
paper's method as is, set `min_improvement=0` and pass no `groups`:

- `min_improvement` (default `MIN_IMPROVEMENT` = 1%): unless the selected blend's out-of-fold loss
  is lower than the default's (candidate 0) by at least this fraction, the default ensemble is
  kept. Classification also requires the log loss to drop by at least `MIN_IMPROVEMENT_NATS`
  (when the default loss is near 0, tiny differences look large as a fraction)
- `min_temperature` (default none): drop candidate temperatures below this. Off, since it gave no
  gain in our evaluation
- `fit(X, y, groups=...)`: with per-row group ids, the out-of-fold splits are made by group (each
  group in one fold only; `StratifiedGroupKFold`, or `GroupKFold` when that cannot split). Empty
  folds are dropped, and random folds are used when fewer than 2 remain. For TabArena and
  BeyondArena, grouped splits pass the group column and temporal splits pass time intervals as
  groups (`tabsil.benchmarks.arena.split_groups`). Temporal splits are also interval-wise k-fold,
  so some folds fit the past from future intervals (not a forward-only split; the same way
  AutoGluon builds bagging folds in TabArena). Wrapped in `DiagScale`, groups are used only for
  greedy selection and DiagScale's validation split stays random
"""

from __future__ import annotations

import contextlib
import copy
import warnings
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold, StratifiedKFold

from tabsil.inference.ensemble import EnsembleClassifier, check_classification_only
from tabsil.inference.preprocessing import Member

POOL_SIZE = 24
N_FOLDS = 8
N_ROUNDS = 40  # TabArena/AutoGluon ensemble selection default (as in the paper)
SELECTIONS = ("greedy", "best", "uniform")
# softmax_temperature and OUTLIER_REMOVAL_STD of the TabArena TabPFN-2.5 search space
TEMPERATURES = (0.25, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.25, 1.5)
OUTLIER_SIGMAS = (3.0, 6.0, 12.0)
# default minimum relative improvement (measured: BeyondArena-lite grouped split Elo 1023 -> 1054,
# TabArena-Lite unchanged)
MIN_IMPROVEMENT = 0.01
# minimum classification log-loss decrease also required with min_improvement (nats)
MIN_IMPROVEMENT_NATS = 1e-3


def greedy_weights(preds: np.ndarray, loss, n_rounds: int = N_ROUNDS) -> np.ndarray:
    """Caruana's greedy selection (with replacement). `preds` (n_candidates, n) holds each
    candidate's predictions, `loss` maps a (b, n) batch of predictions to (b,) losses. Weights are
    pick count / `n_rounds`. Ties and all-NaN losses pick the lower index."""
    total = np.zeros(preds.shape[1])
    counts = np.zeros(len(preds))
    for t in range(n_rounds):
        scores = loss((total + preds) / (t + 1))
        best = int(np.nanargmin(scores)) if np.isfinite(scores).any() else 0
        total += preds[best]
        counts[best] += 1
    return counts / n_rounds


def _log_loss(true_class_probs: np.ndarray) -> np.ndarray:
    # averages of candidates whose probabilities sum to 1 also sum to 1, so the true-class
    # probability alone determines the log loss
    return -np.log(np.clip(true_class_probs, 1e-15, 1.0)).mean(axis=1)


def check_selection(
    *,
    n_estimators: int,
    pool_size: int = POOL_SIZE,
    selection: str = "greedy",
    temperatures: tuple[float, ...] = TEMPERATURES,
    min_improvement: float = MIN_IMPROVEMENT,
    min_temperature: float | None = None,
) -> tuple[float, ...]:
    """Validate the selection settings and return the pool temperatures (at least
    `min_temperature`); ValueError otherwise. Also used to check settings before building an
    estimator (`scripts/eval.py`)."""
    if selection not in SELECTIONS:
        raise ValueError(f"selection must be one of {SELECTIONS}: {selection}")
    if pool_size < n_estimators:
        raise ValueError(f"pool_size ({pool_size}) must be at least n_estimators ({n_estimators})")
    if not 0.0 <= min_improvement < 1.0:
        raise ValueError(f"min_improvement must be in [0, 1): {min_improvement}")
    if min_temperature is not None:
        temperatures = tuple(t for t in temperatures if t >= min_temperature)
        if not temperatures:
            raise ValueError(f"no temperature is at least min_temperature ({min_temperature})")
    return tuple(temperatures)


class _Selection:
    """Pool, fold and selection settings and shared computation. Inherited with
    `EnsembleClassifier`."""

    def __init__(
        self,
        *args: Any,
        pool_size: int = POOL_SIZE,
        temperatures: tuple[float, ...] = TEMPERATURES,
        n_folds: int = N_FOLDS,
        n_rounds: int = N_ROUNDS,
        selection: str = "greedy",
        min_improvement: float = MIN_IMPROVEMENT,
        min_temperature: float | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        check_classification_only(self.task, "greedy selection")
        self.temperatures = check_selection(
            n_estimators=self.n_estimators,
            pool_size=pool_size,
            selection=selection,
            temperatures=temperatures,
            min_improvement=min_improvement,
            min_temperature=min_temperature,
        )
        self.pool_size = pool_size
        self.min_improvement = min_improvement
        self.min_temperature = min_temperature
        self.n_folds = n_folds
        self.n_rounds = n_rounds
        self.selection = selection

    def _pool(self, rng: np.random.Generator, defaults: list[Member], n_classes: int):
        # random members after the default ones; rng continues from drawing the default members
        members = list(defaults)
        for _ in range(self.pool_size - len(defaults)):
            members.append(
                Member(
                    class_permutation=rng.permutation(n_classes),
                    target_power=False,
                    seed=int(rng.integers(0, 2**31 - 1)),
                    one_hot=bool(rng.integers(2)),
                    outlier_sigma=OUTLIER_SIGMAS[int(rng.integers(len(OUTLIER_SIGMAS)))],
                )
            )
        return members

    def _group_splits(self, y: np.ndarray, groups: Any) -> list | None:
        """Group-wise out-of-fold (train rows, validation rows) splits, or None when the groups
        cannot be split (random folds are used). Empty folds are dropped. Whether groups were used
        is recorded in `grouped_`."""
        self.grouped_ = False
        if groups is None:
            return None
        groups = pd.factorize(pd.Series(np.asarray(groups, dtype=object)), use_na_sentinel=False)[0]
        if len(groups) != len(y):
            raise ValueError(f"len(groups) ({len(groups)}) != training rows ({len(y)})")
        n = int(min(self.n_folds, len(np.unique(groups))))
        if n < 2:
            return None
        rows = np.zeros(len(y))
        splitter = StratifiedGroupKFold(n, shuffle=True, random_state=self.random_state)
        try:
            with warnings.catch_warnings():
                # classes with fewer training rows than folds: folds missing them are kept
                warnings.simplefilter("ignore", UserWarning)
                splits = list(splitter.split(rows, y, groups))
        except ValueError:  # every class has fewer rows than folds
            splits = None
        if splits is None:
            splits = list(GroupKFold(n).split(rows, y, groups))  # split by group size
        # stratification can pile groups into one fold, leaving folds without validation rows
        splits = [(tr, te) for tr, te in splits if len(tr) and len(te)]
        if len(splits) < 2:
            return None
        self.grouped_ = True
        return splits

    def _out_of_fold(self, X: Any, y: np.ndarray, members: list[Member], splits) -> np.ndarray:
        """Out-of-fold predictions per candidate (n_candidates, n). Candidate 0 is the default
        ensemble, then member × temperature (member p, temperature t -> 1 + p × n_temps + t)."""
        X = pd.DataFrame(X).reset_index(drop=True)
        n_t = len(self.temperatures)
        oof = np.zeros((1 + len(members) * n_t, len(X)))
        for train_ix, test_ix in splits:
            fold = copy.copy(self)
            fold.fit_members(X.iloc[train_ix], y[train_ix], members)
            fold_members = fold._member_predictions(X.iloc[test_ix], y[test_ix])
            with contextlib.closing(fold_members) as predictions:
                for p, preds in predictions:
                    oof[1 + p * n_t : 1 + (p + 1) * n_t, test_ix] = preds[:n_t]
                    if p < self.n_estimators:  # default ensemble: uniform average at default temp
                        oof[0, test_ix] += preds[n_t] / self.n_estimators
        return oof

    _min_improvement_abs = 0.0  # MIN_IMPROVEMENT_NATS for classification

    def _guard(self, weights: np.ndarray, oof: np.ndarray, loss) -> tuple[np.ndarray, bool]:
        """Fall back to the default ensemble (candidate 0) unless the selected blend's out-of-fold
        loss is sufficiently below the default's (like DiagScale's "keep the untrained state if no
        epoch improves"). Returns (weights, reverted)."""
        if self.min_improvement <= 0 or weights[0] == 1.0:
            return weights, False
        default, selected = loss(np.stack([oof[0], weights @ oof]))
        need = max(self.min_improvement * default, self._min_improvement_abs)
        if default - selected >= need:
            return weights, False
        return np.eye(len(oof))[0], True  # also the default when the loss is NaN

    def _select(self, oof: np.ndarray, loss) -> np.ndarray:
        if self.selection == "greedy":
            return greedy_weights(oof, loss, self.n_rounds)
        weights = np.zeros(len(oof))
        if self.selection == "best":
            scores = loss(oof)
            weights[int(np.nanargmin(scores)) if np.isfinite(scores).any() else 0] = 1.0
        else:  # uniform: all candidates except 0 (the default ensemble)
            weights[1:] = 1.0 / (len(oof) - 1)
        return weights

    def _fit_selected(self, X: Any, y: np.ndarray, members: list[Member], oof, loss):
        """Select the weights and refit only the weighted members on all training rows."""
        weights, reverted = self._guard(self._select(oof, loss), oof, loss)
        n_t = len(self.temperatures)
        # {temperature: weight} per member. Candidate 0's weight is split over the default
        # members at the default temperature
        per_member: list[dict[float, float]] = [{} for _ in members]
        for c in np.flatnonzero(weights):
            if c == 0:
                for p in range(self.n_estimators):
                    t = self.softmax_temperature
                    per_member[p][t] = per_member[p].get(t, 0.0) + weights[0] / self.n_estimators
            else:
                p, t = divmod(c - 1, n_t)
                t = self.temperatures[t]
                per_member[p][t] = per_member[p].get(t, 0.0) + weights[c]
        used = [p for p, w in enumerate(per_member) if w]
        self.selection_ = {
            "n_folds": self.n_folds_,
            "candidates": len(oof),
            "pool_members": len(members),
            "used_members": len(used),
            "oof_loss_default": float(loss(oof[:1])[0]),
            "oof_loss_selected": float(loss((weights @ oof)[None])[0]),
            "weight_default": float(weights[0]),
            "reverted": reverted,
            "grouped": self.grouped_,
        }
        self.member_weights_ = [per_member[p] for p in used]
        return self.fit_members(X, y, [members[p] for p in used])


class GreedyEnsembleClassifier(_Selection, EnsembleClassifier):
    """Classification: greedily select and blend member × temperature candidates by out-of-fold
    log loss (module docstring).

    `n_estimators` is the default ensemble's member count (candidate 0 and the first pool
    members)."""

    _min_improvement_abs = MIN_IMPROVEMENT_NATS

    def fit(self, X: Any, y: Any, groups: Any = None) -> GreedyEnsembleClassifier:
        """With `groups` (per-row group ids), the out-of-fold splits are made by group."""
        rng = np.random.default_rng(self.random_state)
        y = self._fit_labels(y)
        n_classes = len(self.classes_)
        members = self._pool(rng, self._default_members(rng, n_classes), n_classes)
        splits = self._group_splits(y, groups)
        if splits is None:
            counts = np.bincount(y, minlength=n_classes)
            n_folds = int(np.clip(counts[counts > 0].min(), 2, self.n_folds))
            splitter = StratifiedKFold(n_folds, shuffle=True, random_state=self.random_state)
            with warnings.catch_warnings():
                # classes with fewer training rows than folds: folds missing them are kept
                warnings.simplefilter("ignore", UserWarning)
                splits = list(splitter.split(np.zeros(len(y)), y))
        self.n_folds_ = len(splits)
        oof = self._out_of_fold(X, y, members, splits)
        return self._fit_selected(X, y, members, oof, _log_loss)

    def _member_predictions(self, X: Any, y: np.ndarray):
        """Per member (index, true-class probabilities (n_temps + 1, n)); the last row is the
        default temperature."""
        temps = (*self.temperatures, self.softmax_temperature)
        with contextlib.closing(self.member_logits(X)) as members:
            for i, logits in members:
                rows = torch.arange(len(y), device=logits.device)
                target = torch.as_tensor(y, device=logits.device)
                probs = [torch.softmax(logits / t, dim=-1)[rows, target] for t in temps]
                yield i, torch.stack(probs).double().cpu().numpy()

    def predict_proba(self, X: Any) -> np.ndarray:
        total = 0.0
        with contextlib.closing(self.member_logits(X)) as members:
            for i, logits in members:
                for t, w in self.member_weights_[i].items():
                    total = total + w * torch.softmax(logits / t, dim=-1)
        return total.cpu().numpy()
