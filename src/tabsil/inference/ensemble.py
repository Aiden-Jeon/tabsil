"""Preprocessing member ensemble: sklearn-style `EnsembleClassifier` and `EnsembleRegressor` for a
model with `forward(x, y_train, task)` (the `tabsil.models` model).

    clf = EnsembleClassifier(model, n_estimators=8).fit(x_train, y_train)
    proba = clf.predict_proba(x_test)

`tabsil.recipe.TabSilClassifier` and `TabSilRegressor` set TabSil's defaults (8 `standard` members,
at most 768 columns per member chosen `balanced`).

`fit` trains no weights. Each member fits its preprocessing to the training rows
(`tabsil.inference.preprocessing`), and prediction runs the training and test rows through the
model together.

- per member: shuffled feature order, (classification) shuffled class indices, and a sample
  identifier column of standard normal noise
- numeric values more than 12 standard deviations from the column mean are clipped to that bound
- categorical columns are one-hot encoded; columns with more than `MAX_ONE_HOT_CATEGORIES`
  categories stay codes. If the column count after expansion exceeds `max_one_hot_width`
  (default: `max_features` of the model config, the feature count seen in training, or
  `MAX_FEATURES` if the config has none), columns revert to codes starting from the one with the
  most categories. Two categories become one 0/1 column. Missing values and categories unseen in
  training (which become missing when encoded) are NaN
- (with `max_features_per_member`) on tables with more features than that, each member picks that
  many original columns: `feature_subsampling="random"` draws independently per member
  (`tabsil.inference.preprocessing.feature_subset`), `"balanced"` spreads the columns evenly
- classification: softmax of the logits divided by a temperature (`softmax_temperature`, default
  0.9), averaged over members
- regression: y is z-normalized, and half of the members also apply a power transform to y (off,
  off, on, on, ...), exactly half when the member count is a multiple of 4. The model's 999
  quantile outputs become bucket probabilities on the regression borders per member (no
  temperature, `tabsil.models.quantile_distribution.quantiles_to_bucket_probs`). For the predictive
  distribution only the borders are inverse-transformed, moved onto the common borders, and
  averaged

Choices for the regression prediction (the mean):

- Values of the two tail buckets: in the borders built from the prior the outermost buckets are
  very wide (z width 24-28), so the mean of a half-normal tail lands near z ±34. On data whose
  training targets go beyond the inner borders (skewed y), the 1-2% of probability in the tails
  would be valued that far out and shift the prediction a lot, so a tail bucket is valued at the
  end of the training target range (only when that lies beyond the inner border, clipped inside
  the outermost border).
- The mean is computed per member in its own space rather than on the common borders; target
  power members inverse-transform their bucket values. Moving to the common borders assumes a
  uniform density within each bucket, so when mass is concentrated near a common border
  (cpu_activity's y=0, just above an inner border) part of the probability leaks into a tail
  bucket.
- Depending on λ, the Yeo-Johnson inverse is undefined (NaN) beyond one end and arbitrarily steep
  near it. Values outside the training target range are set to its end values, so the bucket
  values of target power members lie within the training target range.
"""

from __future__ import annotations

import contextlib
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, PowerTransformer
from torch import nn

from tabsil.inference.preprocessing import (
    FEATURE_SUBSAMPLINGS,
    MEMBER_KINDS,
    Member,
    MemberPreprocessor,
    apply_encoder,
    balanced_feature_subsets,
    feature_subset,
    select_features,
    to_numeric,
)
from tabsil.models.quantile_distribution import quantiles_to_bucket_probs

# One-hot width cap of a model whose config records no feature cap (`max_features_of`)
MAX_FEATURES = 160

SOFTMAX_TEMPERATURE = 0.9
DEFAULT_MEMBER_KINDS = MEMBER_KINDS


def resolve_member_kinds(
    value: str | tuple[str, ...] | list[str], presets: dict[str, tuple[str, ...]] | None = None
) -> tuple[str, ...]:
    """Member preprocessing kinds: a name in `presets` (the recipe's `MEMBER_KIND_PRESETS`), a comma
    separated string, or a sequence of kinds from `MEMBER_KINDS`."""
    presets = presets or {}
    if isinstance(value, str):
        if value in presets:
            return presets[value]
        value = [k.strip() for k in value.split(",") if k.strip()]
    kinds = tuple(value)
    if not kinds or set(kinds) - set(MEMBER_KINDS):
        names = f" or one of the presets {tuple(presets)}" if presets else ""
        raise ValueError(f"member_kinds are chosen from {MEMBER_KINDS}{names}: {value}")
    return kinds


def max_features_of(model: nn.Module) -> int:
    """The feature cap the model saw in training (`config.max_features`).

    It is not part of the registry contract, so `MAX_FEATURES` is used when it is missing. Shared
    by the inference defaults and the evaluation scripts."""
    return getattr(model.config, "max_features", MAX_FEATURES)


def max_classes_of(model: nn.Module) -> int:
    """The maximum number of classes for classification: the model's `max_classes` (the
    column_row_transformer retrieval decoder groups label embeddings and accepts more classes than
    outputs), else the output count (`config.n_out`). Shared by `EnsembleClassifier` and the
    TabArena/BeyondArena limits of the evaluation scripts."""
    return int(getattr(model, "max_classes", model.config.n_out))


def check_classification_only(task: str, method: str) -> None:
    """`method` (greedy selection, DiagScale) is for classification only: the regression output is
    999 quantiles, which only the basic `EnsembleRegressor` reads."""
    if task != "classification":
        raise ValueError(f"{method} takes classification only (regression outputs are quantiles)")


def forward_kwargs(model: nn.Module, task: str) -> dict:
    """Keyword arguments for the model forward: the task, and `inference=True` (the
    memory-saving computation of large tables) for a model that takes inference options
    (`supports_inference_options`)."""
    kwargs = {"task": task}
    if getattr(model, "supports_inference_options", False):
        kwargs["inference"] = True
    return kwargs


class _Base:
    # the task this ensemble predicts (passed to the forward of a multitask model)
    task = ""

    def __init__(
        self,
        model: nn.Module,
        *,
        n_estimators: int,
        categorical_features_indices: list[int] | None = None,
        member_kinds: tuple[str, ...] | str = DEFAULT_MEMBER_KINDS,
        softmax_temperature: float = SOFTMAX_TEMPERATURE,
        max_one_hot_width: int | None = None,
        max_features_per_member: int | None = None,
        feature_subsampling: str = "random",
        device: str = "cpu",
        random_state: int = 0,
    ) -> None:
        if feature_subsampling not in FEATURE_SUBSAMPLINGS:
            raise ValueError(
                f"feature_subsampling is one of {FEATURE_SUBSAMPLINGS}: {feature_subsampling}"
            )
        self.feature_subsampling = feature_subsampling
        # `member_kinds` is kept for the saved inference settings; TabSil has one kind, "standard"
        self.member_kinds = resolve_member_kinds(member_kinds)
        self.softmax_temperature = softmax_temperature
        # cap on the column count after one-hot expansion: the feature count the model saw in
        # training (max_features of the model config)
        if max_one_hot_width is None:
            max_one_hot_width = max_features_of(model)
        self.max_one_hot_width = max_one_hot_width
        # cap on the original features one member uses (None: all, `feature_subsets`)
        self.max_features_per_member = max_features_per_member
        self.model = model
        self.n_estimators = n_estimators
        self.categorical_features_indices = categorical_features_indices
        self.device = torch.device(device)
        self.random_state = random_state

    def _fit_members(self, X: Any, members: list[Member]) -> None:
        # training rows as a numeric array: `cat_ix_` are the categorical columns (strings and
        # categories turned into codes)
        X_num, self.cat_ix_, self.encoder_ = to_numeric(X, self.categorical_features_indices)
        # columns per member (only when features exceed the cap; set from the member seed, so a
        # member always gets the same columns)
        if self.feature_subsampling == "balanced":
            rng = np.random.default_rng([m.seed for m in members])
            self.feature_subsets_ = balanced_feature_subsets(
                X_num.shape[1], self.max_features_per_member, len(members), rng
            )
        else:
            self.feature_subsets_ = [
                feature_subset(X_num.shape[1], self.max_features_per_member, m.seed + 4)
                for m in members
            ]
        self.preprocessors_ = []
        self.X_trains_ = []
        for m, cols in zip(members, self.feature_subsets_, strict=True):
            x, cat_ix = select_features(X_num, self.cat_ix_, cols)
            prep = MemberPreprocessor(m, cat_ix, self.max_one_hot_width).fit(x)
            self.preprocessors_.append(prep)
            # the transformed training rows do not change, so compute them once per member
            self.X_trains_.append(prep.transform(x, np.random.default_rng(m.seed + 2)))

    def _model_targets(self, y: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """The training-row targets for the model (1, train rows) and extra forward kwargs."""
        return y, {}

    def _iter_inputs(self, X: Any, y_trains: list[np.ndarray]):
        """Per member (index, model input x (1, train + test rows, F), y_train (1, train rows))."""
        X_test = apply_encoder(pd.DataFrame(X), self.cat_ix_, self.encoder_)
        for i, (prep, y_train) in enumerate(zip(self.preprocessors_, y_trains, strict=True)):
            x_test, _ = select_features(X_test, self.cat_ix_, self.feature_subsets_[i])
            x_test = prep.transform(x_test, np.random.default_rng(prep.member.seed + 3))
            x = np.concatenate([self.X_trains_[i], x_test])
            x = torch.as_tensor(x, dtype=torch.float32, device=self.device).unsqueeze(0)
            y = torch.as_tensor(y_train, dtype=torch.float32, device=self.device).unsqueeze(0)
            yield i, x, y

    def _iter_logits(self, X: Any, y_trains: list[np.ndarray]):
        model = self.model.to(self.device).eval()
        autocast = self.device.type == "cuda"  # bfloat16, as in training
        kwargs = forward_kwargs(model, self.task)
        # member forwards reuse one large CPU buffer (the cell state of the interleaved
        # column_row_transformer), released after the last member (also on an exception)
        hold = getattr(model, "hold_host_state", None)
        with hold() if hold is not None else contextlib.nullcontext():
            for i, x, y in self._iter_inputs(X, y_trains):
                y, extra = self._model_targets(y)
                with (
                    torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=autocast),
                    torch.inference_mode(),
                ):
                    logits = model(x, y, **kwargs, **extra)[0].float()
                # yielded outside autocast: the consumer's math (e.g. the regression mean matmul)
                # runs in float32
                yield i, logits


class EnsembleClassifier(_Base):
    """Classification: shuffle the class indices per member, softmax the logits divided by a
    temperature, and average.

    More classes than model outputs (`config.n_out`, the number of label embeddings) are accepted
    only by a retrieval decoder model (one with a larger `max_classes_of`, column_row_transformer).
    The label embedding gets the shuffled index modulo `n_out` (its group), the decoder gets the
    shuffled index itself. The decoder averages the training rows' original-class one-hots with
    its attention weights, so every class gets a probability. Indices are shuffled per member, so
    the classes sharing an embedding differ between members. With 11 classes only one group has 2.
    (Our choice; the official TabPFN-3.5 learns 160 label embeddings and does not group them.)
    """

    task = "classification"

    def fit(self, X: Any, y: Any) -> EnsembleClassifier:
        rng = np.random.default_rng(self.random_state)
        y = self._fit_labels(y)
        return self.fit_members(X, y, self._default_members(rng, len(self.classes_)))

    def _default_members(self, rng: np.random.Generator, n_classes: int) -> list[Member]:
        return [
            Member(
                class_permutation=rng.permutation(n_classes),
                target_power=False,
                seed=int(rng.integers(0, 2**31 - 1)),
            )
            for i in range(self.n_estimators)
        ]

    def _fit_labels(self, y: Any) -> np.ndarray:
        """Set the classes (`classes_`) and return the targets as indices starting at 0."""
        self.label_encoder_ = LabelEncoder()
        y = self.label_encoder_.fit_transform(np.asarray(y))
        self.classes_ = self.label_encoder_.classes_
        n_classes = len(self.classes_)
        if n_classes > max_classes_of(self.model):
            raise ValueError(
                f"{n_classes} classes exceed what the model accepts ({max_classes_of(self.model)})."
            )
        return y

    def _model_targets(self, y: torch.Tensor) -> tuple[torch.Tensor, dict]:
        # with more classes than label embeddings: group index for the embedding, original index
        # for the decoder (class docstring)
        n_out, n_classes = self.model.config.n_out, len(self.classes_)
        if n_classes <= n_out:
            return y, {}
        return torch.remainder(y, n_out), {"y_classes": y, "n_classes": n_classes}

    def fit_members(self, X: Any, y: np.ndarray, members: list[Member]) -> EnsembleClassifier:
        """Fit with the given members. `y` holds `classes_` indices (after `_fit_labels`)."""
        self._fit_members(X, members)
        # each member gets the training targets as shuffled indices: original class c -> perm[c]
        self.y_trains_ = [m.class_permutation[y] for m in members]
        return self

    def member_logits(self, X: Any):
        """Per member (index, logits in the original class order (n, n_classes))."""
        n_classes = len(self.classes_)
        with contextlib.closing(self._iter_logits(X, self.y_trains_)) as members:
            for i, logits in members:
                perm = self.preprocessors_[i].member.class_permutation
                yield i, logits[:, :n_classes][:, perm]  # shuffled-index logits back to class order

    def predict_proba(self, X: Any) -> np.ndarray:
        with contextlib.closing(self.member_logits(X)) as members:
            probs = [
                torch.softmax(logits / self.softmax_temperature, dim=-1) for _, logits in members
            ]
        return torch.stack(probs).mean(0).cpu().numpy()

    def predict(self, X: Any) -> np.ndarray:
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


def _probs_on_borders(probs: torch.Tensor, frm: torch.Tensor, to: torch.Tensor) -> torch.Tensor:
    """Move bucket probabilities on borders `frm` onto borders `to` (uniform within a bucket)."""
    cdf = torch.cat([torch.zeros_like(probs[:, :1]), probs.cumsum(-1)], dim=-1)  # CDF on frm
    idx = (torch.searchsorted(frm, to) - 1).clamp(0, len(frm) - 2)
    width = (frm[idx + 1] - frm[idx]).clamp_min(1e-12)
    share = ((to - frm[idx]) / width).clamp(0, 1)
    at_to = cdf[:, idx] + probs[:, idx] * share
    at_to[:, to <= frm[0]] = 0.0
    at_to[:, to >= frm[-1]] = 1.0
    at_to[:, 0], at_to[:, -1] = 0.0, 1.0
    return (at_to[:, 1:] - at_to[:, :-1]).clamp_min(0)


def _clipped_inverse(transform: PowerTransformer, u_train: np.ndarray):
    """Inverse of the power transform, with values outside the training target range [lo, hi]
    set to its end values.

    The inverse is undefined beyond the right end for λ < 0 and beyond the left end for λ > 2, and
    arbitrarily steep near it, so extending it past the range (even linearly with the end slope)
    sends z into the thousands.
    """
    lo, hi = float(u_train.min()), float(u_train.max())

    def g(u: np.ndarray) -> np.ndarray:
        u = np.clip(np.asarray(u, dtype=np.float64).reshape(-1, 1), lo, hi)
        with np.errstate(all="ignore"):
            return transform.inverse_transform(u).ravel()

    return g


def _bucket_values(borders: torch.Tensor, u_train: np.ndarray) -> np.ndarray:
    """The value of each bucket for the mean: inner buckets use their midpoint, the two tail
    buckets the ends of the training target range.

    Tail values are clipped inside the outermost borders (so a single extreme row does not set
    them).
    """
    b = borders.double().numpy()
    values = (b[1:] + b[:-1]) / 2
    values[0] = max(b[0], min(b[1], float(u_train.min())))
    values[-1] = min(b[-1], max(b[-2], float(u_train.max())))
    return values


class EnsembleRegressor(_Base):
    """Regression: bucket probabilities computed per member in its own space and averaged (module
    docstring). Half of the members apply a Yeo-Johnson power transform to the target."""

    task = "regression"

    def __init__(self, model: nn.Module, borders: torch.Tensor, **kwargs: Any) -> None:
        super().__init__(model, **kwargs)
        self.borders = borders.float().cpu()  # moved to the device only to compute; results on CPU

    def fit(self, X: Any, y: Any) -> EnsembleRegressor:
        rng = np.random.default_rng(self.random_state)
        return self.fit_members(X, y, self._default_members(rng))

    def _default_members(self, rng: np.random.Generator) -> list[Member]:
        return [
            Member(
                class_permutation=None,
                target_power=(i // 2) % 2 == 1,  # off, off, on, on, ...
                seed=int(rng.integers(0, 2**31 - 1)),
            )
            for i in range(self.n_estimators)
        ]

    def fit_members(self, X: Any, y: Any, members: list[Member]) -> EnsembleRegressor:
        """Fit with the given members. `y` is in original units."""
        y = np.asarray(y, dtype=np.float64)
        self.y_mean_, self.y_std_ = float(y.mean()), float(y.std()) + 1e-12
        z = (y - self.y_mean_) / self.y_std_
        self._fit_members(X, members)
        b = self.borders.double().numpy()
        self.y_trains_, self.bucket_values_, self.border_maps_ = [], [], []
        for m in members:
            u, values, border_map = z, _bucket_values(self.borders, z), None
            if m.target_power:
                transform = PowerTransformer(method="yeo-johnson", standardize=True)
                u = np.asarray(transform.fit_transform(z.reshape(-1, 1))).ravel()
                inverse = _clipped_inverse(transform, u)
                # bucket values and borders from member space to z space (the borders are used
                # by predict_distribution)
                values = np.clip(inverse(_bucket_values(self.borders, u)), b[0], b[-1])
                back = np.clip(np.maximum.accumulate(inverse(b)), b[0], b[-1])
                border_map = torch.as_tensor(back, dtype=torch.float32)
            self.y_trains_.append(u)
            self.bucket_values_.append(torch.as_tensor(values, dtype=torch.float32))
            self.border_maps_.append(border_map)
        return self

    def predict_distribution(self, X: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """(bucket log probabilities (n, n_buckets), borders in original units (float64)).

        The mixture of the members moved onto the common borders. Its tails are the common
        borders' half-normal tails, so the mean of this distribution differs from `predict` (use
        `predict` for the mean). Borders are mapped to original units in float64: inner buckets
        are as narrow as 1e-6 in z, so with a large target mean and small spread several borders
        would collapse to one value in float32.
        """
        with contextlib.closing(self._member_probs(X)) as members:
            probs = [self._on_common_borders(i, p) for i, p in members]
        return self._mixture(probs)

    def predict(self, X: Any) -> np.ndarray:
        """Mean of the predictive distribution. The mean of the member mixture is the average of
        the member means, so each member uses its own bucket values (`bucket_values_`) without
        moving to the common borders."""
        with contextlib.closing(self._member_probs(X)) as members:
            return self._mean([self._member_mean(i, p) for i, p in members])

    def predict_full(self, X: Any) -> tuple[np.ndarray, torch.Tensor, torch.Tensor]:
        """(the `predict` mean, the `predict_distribution` log probabilities and borders) from one
        forward pass. Distribution-quality evaluation (`tabsil_train.benchmarks.uncertainty`) uses
        the point prediction and the distribution together."""
        means, probs = [], []
        with contextlib.closing(self._member_probs(X)) as members:
            for i, p in members:
                means.append(self._member_mean(i, p))
                probs.append(self._on_common_borders(i, p))
        return (self._mean(means), *self._mixture(probs))

    def _member_probs(self, X: Any):
        """Per member (index, bucket probabilities in its own space): the quantile outputs turned
        into bucket probabilities on the borders, without a temperature (module docstring)."""
        with contextlib.closing(self._iter_logits(X, self.y_trains_)) as members:
            for i, logits in members:
                p = quantiles_to_bucket_probs(logits, self.borders)
                yield i, torch.softmax(p.clamp_min(1e-30).log(), dim=-1)

    def _member_mean(self, i: int, p: torch.Tensor) -> torch.Tensor:
        return p @ self.bucket_values_[i].to(p.device)

    def _mean(self, means: list[torch.Tensor]) -> np.ndarray:
        return torch.stack(means).mean(0).cpu().numpy() * self.y_std_ + self.y_mean_

    def _on_common_borders(self, i: int, p: torch.Tensor) -> torch.Tensor:
        if self.border_maps_[i] is None:
            return p
        # move the probabilities from the power-transformed borders mapped back to z space
        borders = self.borders.to(p.device)
        return _probs_on_borders(p, self.border_maps_[i].to(p.device), borders)

    def _mixture(self, probs: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        log_probs = torch.stack(probs).mean(0).clamp_min(1e-12).log().cpu()
        return log_probs, self.borders.double() * self.y_std_ + self.y_mean_
