"""Tests for benchmark scoring and task selection in the evaluation script."""

import argparse

import numpy as np
import pytest
import train_helpers as helpers

from tabsil_train.benchmarks.tabular.amlb import score_fold


def test_test_rows_of_classes_unseen_in_training_are_ignored():
    classes = np.array([0, 1, 2])
    y_true = np.array([0, 1, 2, 0, 1, 2, 3])  # class 3 was not in the training fold
    proba = np.full((7, 3), 1 / 3)
    proba[np.arange(6), y_true[:6]] = 0.8
    proba = proba / proba.sum(axis=1, keepdims=True)
    assert score_fold("classification", y_true, proba, classes) == pytest.approx(1.0)


def test_single_class_fold_scores_nan():
    proba = np.array([[0.7, 0.3], [0.6, 0.4]])
    assert np.isnan(score_fold("classification", np.array([0, 0]), proba, np.array([0, 1])))


def _eval_script():
    return helpers.load_script("scripts/eval.py", "eval_script_benchmark")


def test_unknown_task_ids_are_reported():
    script = _eval_script()
    with pytest.raises(SystemExit, match="35995"):
        script.selected_tasks(argparse.Namespace(suite="all", tasks="359955,35995"))
    assert script.selected_tasks(argparse.Namespace(suite="all", tasks="359955")) == [
        ("classification", 359955)
    ]


def test_removed_inference_flags_are_rejected():
    script = _eval_script()
    for flag in ("--one-hot", "--official-v35", "--target-transform"):
        with pytest.raises(SystemExit):
            script.parse_args([flag, "all"])
    # `--member-kinds` is back as an inference option: kinds from MEMBER_KINDS or a preset
    assert script.parse_args(["--member-kinds", "numeric-cycle"]).member_kinds == "numeric-cycle"
