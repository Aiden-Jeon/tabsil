import pytest
import torch

from tabsil_train.priors.base import IGNORE_INDEX, PriorBatch
from tabsil_train.priors.hyperparameters import ClassCount
from tabsil_train.priors.pipeline import Group, Pipeline, Rng
from tabsil_train.priors.transforms import CoverClasses, DropInvalid, Missing


class PadFeatures:
    """Test transform: zero-pad the features up to `n_features` columns."""

    def __init__(self, n_features: int) -> None:
        self.n_features = n_features

    def __call__(self, group, rng):
        x = group.x
        pad = torch.zeros(*x.shape[:2], self.n_features - x.shape[-1], device=x.device)
        return group.with_tensors(torch.cat([x, pad], dim=-1), group.y)


class SignClasses:
    """Test transform: classes from the sign of a continuous target."""

    def __call__(self, group, rng):
        return group.with_tensors(group.x, (group.y > 0).float())


class Constant:
    """Fixed-value generator: x = value, y = row index % 2."""

    def __init__(self, value: float) -> None:
        self.value = value

    def __call__(self, n_rows, n_features, n_datasets, hp, rng):
        x = torch.full((n_rows, n_datasets, n_features), self.value)
        y = (torch.arange(n_rows) % 2).float().unsqueeze(1).expand(n_rows, n_datasets).clone()
        return x, y


class RejectFirst:
    def __init__(self, n: int) -> None:
        self.n = n
        self.calls = 0

    def __call__(self, group, rng):
        self.calls += 1
        return None if self.calls <= self.n else group


def _counter():
    draws = []

    def hyperparameters(rng, n_train, n_features):
        draws.append(1)
        return {"draw": len(draws)}

    return hyperparameters, draws


def test_rejected_groups_are_redrawn_from_the_hyperparameters():
    hyperparameters, draws = _counter()
    reject = RejectFirst(2)
    pipe = Pipeline(
        Constant(1.0),
        rng=Rng.from_seed(0),
        hyperparameters=hyperparameters,
        transforms=[reject],
        max_attempts=5,
    )
    group = pipe.sample_group(n_rows=6, n_train=4, n_features=3)
    assert reject.calls == 3 and len(draws) == 3
    assert group.hp == {"draw": 3}


def test_gives_up_after_max_attempts():
    pipe = Pipeline(
        Constant(1.0), rng=Rng.from_seed(0), transforms=[RejectFirst(10)], max_attempts=3
    )
    with pytest.raises(RuntimeError, match="after 3 attempts"):
        pipe.sample_group(6, 4, 3)


def test_generator_gets_the_used_feature_count_and_batch_is_datasets_first():
    pipe = Pipeline(
        Constant(1.0),
        rng=Rng.from_seed(0),
        hyperparameters=lambda rng, n_train, n_features: {"num_features_used": 2},
        transforms=[PadFeatures(5)],
        group_size=2,
    )
    batch = pipe.sample_batch(4, n_rows=7, n_train=5, n_features=5)
    assert isinstance(batch, PriorBatch) and batch.n_train == 5
    assert batch.x.shape == (4, 7, 5) and batch.y.shape == (4, 7)
    assert (batch.x[..., :2] == 1).all() and (batch.x[..., 2:] == 0).all()
    assert torch.equal(batch.y[0], torch.arange(7.0) % 2)
    for bad in (3, 0, -2):
        with pytest.raises(ValueError, match="multiple of group_size"):
            pipe.sample_batch(bad, 7, 5, 5)


def test_sample_group_can_run_only_some_transforms():
    pipe = Pipeline(
        Constant(1.0), rng=Rng.from_seed(0), transforms=[RejectFirst(10)], max_attempts=2
    )
    assert (pipe.sample_group(5, 3, 2, transforms=[]).x == 1).all()
    with pytest.raises(RuntimeError):
        pipe.sample_group(5, 3, 2)


class Gaussian:
    """A random generator: x ~ N(0, 1), y = sum of the features plus noise."""

    def __call__(self, n_rows, n_features, n_datasets, hp, rng):
        x = torch.randn(n_rows, n_datasets, n_features, generator=rng.torch)
        y = x.sum(-1) + 0.1 * torch.randn(n_rows, n_datasets, generator=rng.torch)
        return x, y


def test_parts_combine_into_a_classification_batch_with_missing_values():
    # Combining parts: a generator, the class count, classes, padding and missing values
    pipe = Pipeline(
        Gaussian(),
        rng=Rng.from_seed(0),
        hyperparameters=ClassCount(10),
        transforms=[
            SignClasses(),
            CoverClasses(),
            PadFeatures(8),
            Missing(1.0, 0.5),
        ],
        group_size=4,
    )
    batch = pipe.sample_batch(8, n_rows=40, n_train=30, n_features=4)
    assert batch.x.shape == (8, 40, 8)
    assert torch.isnan(batch.x).any()
    assert ((batch.y >= 0) | (batch.y == IGNORE_INDEX)).all()


def test_filters_and_cover_classes_look_at_every_dataset():
    rng = Rng.from_seed(0)
    y = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])  # 2 datasets
    group = Group(torch.arange(8.0).reshape(4, 2, 1), y, n_train=2, hp={})
    covered = CoverClasses()(group, rng)
    for d in range(2):
        assert set(covered.y[:2, d].tolist()) == {0.0, 1.0}
        # Only the row order changes; x and y move together
        pairs = {
            (float(a), float(b)) for a, b in zip(covered.x[:, d, 0], covered.y[:, d], strict=True)
        }
        assert pairs == {
            (float(a), float(b)) for a, b in zip(group.x[:, d, 0], y[:, d], strict=True)
        }
    one_class = Group(group.x, torch.zeros(4, 2), 2, {})
    assert CoverClasses()(one_class, rng) is None
    invalid = Group(
        group.x, torch.stack([y[:, 0], torch.full((4,), float(IGNORE_INDEX))], 1), 2, {}
    )
    assert DropInvalid()(invalid, rng) is None and DropInvalid()(group, rng) is group


def test_missing_only_touches_x_of_chosen_datasets():
    group = Group(torch.ones(50, 3, 4), torch.zeros(50, 3), 40, {})
    assert not torch.isnan(Missing(0.0, 0.9)(group, Rng.from_seed(0)).x).any()
    out = Missing(1.0, 0.9)(group, Rng.from_seed(1))
    assert torch.isnan(out.x).any() and torch.equal(out.y, group.y)


def test_settings_baked_into_parts_are_read_only():
    from tabsil_train.priors.graph_scm import GraphSCMPrior

    graph = GraphSCMPrior("regression", seed=0, missing_dataset_prob=0.5)
    assert graph.missing_dataset_prob == 0.5 and graph.task == "regression"
    for name in ("missing_dataset_prob", "task"):
        with pytest.raises(AttributeError):
            setattr(graph, name, 0)


ACCELERATOR = (
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else None
)


@pytest.mark.skipif(ACCELERATOR is None, reason="no GPU (cuda/mps)")
def test_transforms_keep_tensors_on_their_device():
    # Generator yields GPU tensors while Rng is on CPU (device is read from the tensor, not Rng)
    dev = torch.device(ACCELERATOR)
    y = torch.tensor([[0.0, 1.0]] * 5 + [[1.0, 0.0]] * 5, device=dev)
    group = Group(torch.randn(10, 2, 3, device=dev), y, 6, {})
    rng = Rng.from_seed(0)
    for transform in (Missing(1.0, 0.5), CoverClasses(), PadFeatures(5)):
        out = transform(group, rng)
        assert out.x.device.type == dev.type and out.y.device.type == dev.type
