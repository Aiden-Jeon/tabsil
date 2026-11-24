"""DiagScale: test-time adaptation that learns, per dataset, only a diagonal scale on the
attention query-key similarity ("Test-Time Compute for Tabular Foundation Models", arXiv
2610.12005, "Adaptation" and Appendix B). All other weights stay fixed.

    clf = DiagScale(TabSilClassifier(model)).fit(x_train, y_train)
    proba = clf.predict_proba(x_test)

Classification only: the regression output is 999 quantiles, which only the basic
`EnsembleRegressor` reads.

`DiagScale` wraps one ensemble estimator (a class from `tabsil.inference.ensemble`, including
greedy selection in `tabsil.inference.ensemble_selection`). fit learns the scale on a copy of the
model, then fits the wrapped estimator with that copy on all training rows.

What the paper (and its public code `ttc.backbones.tabicl_finetune`) does:

- each attention gets a vector δ (head × head dim) initialized to 0, and the logit becomes
  qᵀ(I + diag(δ))k. δ = 0 is the original model. Only δ is trained
- AdamW, lr 3e-2 (the value chosen for both TabPFN-3 and TabICL v2), weight decay 0.01, linear
  warmup over 10% of all steps then cosine to 0, gradient norm clipped at 1
- at most 30 epochs, patience 8. Validation rows (10%, at least min(200, 25%)) are held out from
  the training rows, and the epoch is selected by validation loss (log loss for classification,
  RMSE for regression). If no epoch beats the untrained state, the untrained state is kept
- episode: draw at most 10,000 of the remaining rows, 20% as query and the rest as context. Steps
  per epoch: max(1, min(10, round(n_rows / n_drawn)))
- 2 members for training and validation, the default member count for deployment
- results (TabArena Elo): TabPFN-3 +26, TabICL v2 +82, TabFM +20

Our choices:

- δ multiplies the query part of the qkv Linear output (forward hook). A module with `qkv` (an
  `nn.Linear` ordered query, key, value, each contiguous per head) and `nhead` counts as an
  attention. The product is computed in float32 and cast back to the compute dtype. Training uses
  a copy of the model, so the original model is unchanged
- attentions that use the qkv weight directly (`diag_scale_in_forward`, column_row_transformer)
  multiply q by δ in their forward instead of through a hook. It comes after the q/k RMSNorm, RoPE
  and temperature, so the logit is qᵀ(I + diag(δ))k. δ is attached to the column, row and ICL
  attentions; the classification retrieval decoder is not an attention module and gets none
- episode forwards use the same arguments as inference (`forward_kwargs`: the task of a multitask
  model, the large-table computation, and grouped plus original indices when there are more
  classes than label embeddings)
- the cell count of one episode (rows × columns after preprocessing) is capped at `max_cells`
  (GPU memory)
- episode members fit their preprocessing on the context rows (as in inference)
- an episode whose query has a class missing from the context is skipped (as in the paper code)
"""

from __future__ import annotations

import copy
import functools
import math
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch import nn

from tabsil.inference.ensemble import (
    EnsembleClassifier,
    _Base,
    check_classification_only,
    forward_kwargs,
)

EPOCHS = 30
LEARNING_RATE = 3e-2
WEIGHT_DECAY = 0.01
PATIENCE = 8
MIN_IMPROVEMENT = 1e-4  # minimum validation-loss decrease that counts as improvement (paper code)
MAX_ROWS = 10_000
MAX_CELLS = 400_000
MIN_ROWS = 256  # minimum episode rows, even on wide tables
QUERY_FRACTION = 0.2
N_VIEWS = 2
GRAD_CLIP = 1.0


def _scale_query(attention: nn.Module, linear: nn.Module, inputs, output: torch.Tensor):
    delta = attention.diag_scale
    e = delta.numel()
    q = (output[..., :e].float() * (1.0 + delta)).to(output.dtype)
    return torch.cat([q, output[..., e:]], dim=-1)


def add_diag_scale(model: nn.Module) -> list[nn.Parameter]:
    """Attach δ (initialized to 0, head × head dim = E entries) to each attention and return the
    list. Modifies `model` in place (use it on a copy). δ is the attention module's parameter
    `diag_scale`."""
    deltas = []
    for module in model.modules():
        qkv = getattr(module, "qkv", None)
        if not (isinstance(qkv, nn.Linear) and hasattr(module, "nhead")):
            continue
        if hasattr(module, "diag_scale"):
            raise ValueError("DiagScale is already attached to this model")
        delta = nn.Parameter(torch.zeros(qkv.out_features // 3, device=qkv.weight.device))
        module.register_parameter("diag_scale", delta)
        if not getattr(module, "diag_scale_in_forward", False):  # multiplied in their forward
            qkv.register_forward_hook(functools.partial(_scale_query, module))
        deltas.append(delta)
    if not deltas:
        raise ValueError("no attention module with qkv and nhead")
    return deltas


def _views(
    estimator: _Base, model: nn.Module, n_views: int, random_state: int
) -> EnsembleClassifier:
    # basic ensemble with the wrapped estimator's inference settings and a different member count
    # (training episodes, validation)
    kwargs = {
        "n_estimators": n_views,
        "categorical_features_indices": estimator.categorical_features_indices,
        "member_kinds": estimator.member_kinds,
        "softmax_temperature": estimator.softmax_temperature,
        "max_one_hot_width": estimator.max_one_hot_width,
        "max_features_per_member": estimator.max_features_per_member,
        "feature_subsampling": estimator.feature_subsampling,
        "device": estimator.device,
        "random_state": random_state,
    }
    return EnsembleClassifier(model, **kwargs)


def _log_loss(classes: np.ndarray, proba: np.ndarray, y: np.ndarray) -> float:
    # classes absent from the training rows get probability 0 (sklearn log_loss fails when the
    # labels do not match)
    ix = np.clip(np.searchsorted(classes, y), 0, len(classes) - 1)
    p = np.where(classes[ix] == y, proba[np.arange(len(y)), ix], 0.0)
    return float(-np.log(np.clip(p, 1e-15, 1.0)).mean())


def _validation_loss(views: EnsembleClassifier, X: pd.DataFrame, y: np.ndarray) -> float:
    return _log_loss(views.classes_, views.predict_proba(X), y)


def _episode_backward(
    views: EnsembleClassifier, model: nn.Module, X_query, y_query
) -> float | None:
    """Compute the query-row loss with `views` fitted on the context rows and backpropagate it
    (member average). Returns the loss, or None if the query has a class missing from the
    context."""
    if not np.isin(y_query, views.classes_).all():
        return None
    target = views.label_encoder_.transform(y_query)
    autocast = views.device.type == "cuda"  # bfloat16, as in inference
    target = torch.as_tensor(target, device=views.device)
    total = 0.0
    n = len(views.preprocessors_)
    kwargs = forward_kwargs(model, views.task)  # same arguments as inference
    for i, x, y_train in views._iter_inputs(X_query, views.y_trains_):
        y_train, extra = views._model_targets(y_train)
        with torch.autocast(views.device.type, dtype=torch.bfloat16, enabled=autocast):
            logits = model(x, y_train, **kwargs, **extra)[0].float()
        perm = views.preprocessors_[i].member.class_permutation
        logits = logits[:, : len(views.classes_)][:, perm]
        loss = F.cross_entropy(logits, target.long())
        (loss / n).backward()
        total += float(loss.detach()) / n
    return total


def fit_diag_scale(
    estimator: _Base,
    X: Any,
    y: Any,
    *,
    epochs: int = EPOCHS,
    lr: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    patience: int = PATIENCE,
    max_rows: int = MAX_ROWS,
    max_cells: int = MAX_CELLS,
    query_fraction: float = QUERY_FRACTION,
    n_views: int = N_VIEWS,
    grad_clip: float = GRAD_CLIP,
) -> tuple[nn.Module, dict]:
    """Attach DiagScale to a copy of `estimator`'s model and train it on (`X`, `y`).

    Returns (model copy, training record)."""
    start = time.perf_counter()
    seed = estimator.random_state
    check_classification_only(estimator.task, "DiagScale")
    model = copy.deepcopy(estimator.model).to(estimator.device).eval()
    model.requires_grad_(False)
    deltas = add_diag_scale(model)

    X = pd.DataFrame(X).reset_index(drop=True)
    y = np.asarray(y)
    # at least 1 row (with 2 rows or fewer round gives 0 and train_test_split fails)
    n_val = int(max(1, round(0.1 * len(y)), min(200, round(0.25 * len(y)))))
    try:
        fit_ix, val_ix = train_test_split(
            np.arange(len(y)),
            test_size=n_val,
            random_state=seed,
            stratify=y,
        )
    except ValueError:  # a class with very few rows: no stratification
        fit_ix, val_ix = train_test_split(np.arange(len(y)), test_size=n_val, random_state=seed)
    X_fit, y_fit = X.iloc[fit_ix].reset_index(drop=True), y[fit_ix]
    X_val, y_val = X.iloc[val_ix], y[val_ix]
    val_views = _views(estimator, model, n_views, seed).fit(X_fit, y_fit)

    width = max(x.shape[1] for x in val_views.X_trains_)
    take = min(len(y_fit), max_rows, max(MIN_ROWS, max_cells // width))
    steps = max(1, min(10, round(len(y_fit) / take)))
    total_steps = steps * epochs
    warmup = max(1, int(0.1 * total_steps))

    def lr_factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, (step - warmup) / max(1, total_steps - warmup))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    optimizer = torch.optim.AdamW(deltas, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    before = best = _validation_loss(val_views, X_val, y_val)
    best_state = [d.detach().clone() for d in deltas]
    best_epoch, bad_epochs, curve, losses = -1, 0, [before], []
    for epoch in range(epochs):
        rng = np.random.default_rng(seed * 100_003 + epoch)
        for _ in range(steps):
            idx = rng.permutation(len(y_fit))[:take]
            n_query = int(np.clip(round(len(idx) * query_fraction), 1, max(1, len(idx) - 8)))
            query, context = idx[:n_query], idx[n_query:]
            views = _views(estimator, model, n_views, int(rng.integers(0, 2**31 - 1)))
            views.fit(X_fit.iloc[context], y_fit[context])
            optimizer.zero_grad(set_to_none=True)
            loss = _episode_backward(views, model, X_fit.iloc[query], y_fit[query])
            if loss is None:
                continue
            torch.nn.utils.clip_grad_norm_(deltas, grad_clip)
            optimizer.step()
            scheduler.step()
            losses.append(loss)
        score = _validation_loss(val_views, X_val, y_val)
        curve.append(score)
        if score < best - MIN_IMPROVEMENT:
            best, best_epoch, bad_epochs = score, epoch, 0
            best_state = [d.detach().clone() for d in deltas]
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break
    with torch.no_grad():  # best epoch (untrained, δ = 0, if none)
        for d, s in zip(deltas, best_state, strict=True):
            d.copy_(s)
    info = {
        "n_params": int(sum(d.numel() for d in deltas)),
        "val_rows": len(val_ix),
        "episode_rows": int(take),
        "steps_per_epoch": int(steps),
        "epochs_run": len(curve) - 1,
        "best_epoch": best_epoch,  # -1: kept untrained
        "val_loss_before": before,
        "val_loss_best": best,
        "val_curve": curve,
        "train_loss": losses,
        "delta_norm": float(torch.cat([d.detach().flatten() for d in deltas]).norm()),
        "seconds": time.perf_counter() - start,
    }
    return model, info


class DiagScale:
    """Wrap an ensemble estimator and learn DiagScale in fit (module docstring). The kwargs are
    those of `fit_diag_scale`. After fit, predictions come from a copy of the wrapped estimator
    (`estimator_`, with the adapted model), whose attributes (`classes_`, `predict_full`, ...) are
    exposed as is. The training record is `adaptation_`."""

    def __init__(self, estimator: _Base, **kwargs: Any) -> None:
        check_classification_only(estimator.task, "DiagScale")
        self.estimator = estimator
        self.kwargs = kwargs

    def fit(self, X: Any, y: Any, **fit_kwargs: Any) -> DiagScale:
        """`fit_kwargs` (e.g. `groups` for greedy selection) go to the wrapped estimator's fit."""
        model, self.adaptation_ = fit_diag_scale(self.estimator, X, y, **self.kwargs)
        estimator = copy.copy(self.estimator)
        estimator.model = model
        self.estimator_ = estimator.fit(X, y, **fit_kwargs)
        return self

    def __getattr__(self, name: str):
        # predict, predict_proba, classes_, ... come from the fitted estimator (absent before fit)
        estimator = self.__dict__.get("estimator_")
        if estimator is None:
            raise AttributeError(name)
        return getattr(estimator, name)
