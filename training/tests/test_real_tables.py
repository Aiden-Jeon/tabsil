"""`tabsil_train.priors.real_tables`: subsamples of pinned corpus tables, given to the model as
inference gives it a table, and a training run that continues a checkpoint on them."""

import hashlib
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import torch
from train_helpers import tabsil_smoke

from tabsil.models import load_checkpoint
from tabsil_train.priors import prior_module, real_tables
from tabsil_train.priors.base import IGNORE_INDEX
from tabsil_train.training.loop import train

N_ROWS = 600


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_corpus(folder) -> str:
    """A corpus of three tables in the format of `real_corpus.py`; returns the manifest digest."""
    rng = np.random.default_rng(0)
    a, c = rng.normal(size=N_ROWS), rng.normal(size=N_ROWS)
    many = rng.choice([f"k{i}" for i in range(40)], N_ROWS)
    tables = {
        # 3 categories: one-hot. Width: a, c, 3 one-hot columns, the identifier
        1: ("binary", pd.DataFrame({"a": a, "b": rng.choice(list("xyz"), N_ROWS), "c": c})),
        # 40 categories: codes and a count column. Width: a, m, the counts, the identifier
        2: ("multiclass", pd.DataFrame({"a": a, "m": many})),
        3: ("regression", pd.DataFrame({"a": a, "c": c})),
    }
    targets = {
        1: np.where(a + c > 0, "yes", "no"),
        2: np.digitize(a, [-1, 0, 1]).astype(str),
        3: 3 * a - c + 10,
    }
    rows = []
    (folder / "tables").mkdir(parents=True)
    for openml_id, (task, frame) in tables.items():
        frame = frame.assign(target=targets[openml_id])
        if task == "regression":
            frame.loc[::7, "c"] = np.nan  # missing cells stay NaN
            frame.loc[3, "target"] = np.nan  # a row without a target is not drawn
        path = folder / "tables" / f"{openml_id}.parquet"
        frame.to_parquet(path, index=False)
        rows.append(
            {
                "openml_id": openml_id,
                "name": f"table{openml_id}",
                "task_type": task,
                "target": "target",
                "sha256": _sha256(path),
                "file": f"tables/{openml_id}.parquet",
            }
        )
    pd.DataFrame(rows).to_csv(folder / "manifest.csv", index=False)
    return _sha256(folder / "manifest.csv")


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    folder = tmp_path / "corpus"
    digest = _write_corpus(folder)
    monkeypatch.setattr(real_tables, "corpus_folder", lambda: folder)
    monkeypatch.setattr(real_tables, "MANIFEST_SHA256", digest)
    monkeypatch.setattr(real_tables, "_TABLES", {})
    return folder


def _config(**changes):
    sizes = {"min_rows": 100, "max_rows": 300, "max_features": 100, "max_cells": 100 * 300}
    return tabsil_smoke(prior_type="real_tables", **{**sizes, **changes})


def test_classification_rows_are_inference_inputs(corpus):
    config = _config()
    prior = real_tables.make_prior(config, "classification", seed=0)
    widths = set()
    for _ in range(20):
        batch = prior.sample_batch(1)
        n_rows = batch.x.shape[1]
        assert batch.x.dtype == torch.float32 and batch.x.shape[0] == 1
        assert 100 <= n_rows <= 300
        assert config.min_train_fraction - 0.01 <= batch.n_train / n_rows
        assert batch.n_train / n_rows <= config.max_train_fraction + 0.01
        train_classes = set(batch.y_train.flatten().tolist())
        assert len(train_classes) >= 2 and train_classes <= set(range(4))
        # eval rows: a class of the train rows, or ignored
        assert set(batch.y_eval.flatten().tolist()) <= train_classes | {IGNORE_INDEX}
        widths.add(batch.x.shape[2])
    assert widths == {6, 4}  # the two tables' member widths (see `_write_corpus`)


def test_a_batch_of_several_tables_has_one_shape(corpus):
    batch = real_tables.make_prior(_config(), "classification", seed=1).sample_batch(8)
    assert batch.x.shape[0] == 8 and batch.y.shape == batch.x.shape[:2]
    assert batch.x.shape[2] <= 6


def test_regression_targets_are_standardized_as_the_graph_scm_prior_does(corpus):
    batch = real_tables.make_prior(_config(), "regression", seed=2).sample_batch(1)
    y = batch.y[0].double()
    assert abs(y.mean()) < 1e-6 and abs(y.std(unbiased=False) - 1) < 1e-6  # over all rows
    assert torch.isnan(batch.x).any()  # the missing cells
    assert not torch.isnan(batch.y).any()  # rows without a target are never drawn
    on_train = real_tables.make_prior(_config(target_train_rows=True), "regression", seed=2)
    y_train = on_train.sample_batch(1).y_train[0].double()
    assert abs(y_train.mean()) < 1e-6


def test_eval_rows_of_a_class_the_train_rows_lack_are_ignored(corpus):
    prior = real_tables.make_prior(_config(), "classification", seed=3)
    table = real_tables.Table("t", pd.DataFrame({"a": np.arange(11.0)}), np.array([*"aaaaabbbbbc"]))
    rows = np.arange(11)
    y = prior._targets(table, rows[:10], rows)
    assert y[-1] == IGNORE_INDEX and sorted(set(y[:10])) == [0, 1]
    assert prior._targets(table, rows[:5], rows) is None  # one train class: drawn again


def test_the_same_seed_gives_the_same_batch(corpus):
    first = real_tables.make_prior(_config(), "classification", seed=4).sample_batch(2)
    second = real_tables.make_prior(_config(), "classification", seed=4).sample_batch(2)
    assert torch.equal(first.x.nan_to_num(-7), second.x.nan_to_num(-7))
    assert torch.equal(first.y, second.y) and first.n_train == second.n_train


def test_a_corpus_that_is_not_the_pinned_one_is_refused(corpus):
    with pytest.raises(ValueError, match="pinned corpus manifest"):
        real_tables.load_tables(corpus, "0" * 64)
    pd.DataFrame({"a": [1.0]}).to_parquet(corpus / "tables" / "3.parquet", index=False)
    with pytest.raises(ValueError, match="differs from the manifest"):
        real_tables.load_tables(corpus, _sha256(corpus / "manifest.csv"))


def test_registered_and_without_borders_of_its_own():
    assert prior_module("real_tables") is real_tables
    config = _config()  # check_config passes without the corpus
    assert config.n_out == real_tables.MAX_CLASSES == 10  # the classifier output of #14's model
    with pytest.raises(ValueError, match="start checkpoint"):
        real_tables.make_borders(config, 32, seed=0)
    with pytest.raises(ValueError, match="min_rows"):
        _config(min_rows=1)


def test_training_continues_a_checkpoint_on_real_tables(corpus, tmp_path):
    small = {
        "filter_unpredictable": False,
        "border_datasets": 10,
        "batch_size": 4,
        "micro_batch": 4,
        "min_rows": 32,
        "max_rows": 32,
        "max_features": 4,
        "max_cells": 128,
        "collapse_threshold": None,
    }
    start = tabsil_smoke(steps=2, **small)
    train(start, tmp_path / "start", mlflow_log=False)
    init = tmp_path / "start" / "latest.pt"
    config = replace(start, prior_type="real_tables")
    result = train(config, tmp_path / "real", init_from=init, mlflow_log=False)
    _, borders = load_checkpoint(result["final_path"])
    assert torch.equal(borders, torch.load(init, weights_only=True)["borders"])
    assert np.isfinite(result["history"]["train_loss"]).all()
    assert np.isfinite(result["history"]["val_loss"]).all()  # validated on real tables too
