"""Tests for the pure parts of the gap report (`scripts/gap_report.py`): the Bradley-Terry Elo on
toy pools with known ratings, the geometric mean and its bootstrap, the property buckets, the
fill of missing results, and the what-ifs (one changed error, the reduction to a rank)."""

import math

import numpy as np
import pandas as pd
import pytest
import train_helpers as helpers

gap = helpers.load_script("scripts/gap_report.py", "gap_report_script")
RF = gap.CALIBRATION


def test_elo_of_a_two_method_pool_follows_the_win_probability():
    # A beats RF on 3 datasets and ties on 1: W_A = 3.5, W_RF = 0.5, so p_A = 3.5 / 4 and the
    # Elo gap is 400 * log10(p / (1 - p)) = 400 * log10(7)
    pool = pd.DataFrame(
        {"d0": [0.1, 0.5], "d1": [0.1, 0.5], "d2": [0.1, 0.5], "d3": [0.4, 0.4]},
        index=["A", RF],
    )
    elo = gap.bradley_terry(pool)
    assert elo[RF] == pytest.approx(1000.0)
    assert elo["A"] - elo[RF] == pytest.approx(400 * math.log10(7), abs=1e-3)


def test_elo_is_calibrated_to_the_random_forest_and_symmetric_pools_tie():
    # three methods, each best on one dataset and worst on one: every method has the same wins
    pool = pd.DataFrame(
        {"d0": [0.1, 0.2, 0.3], "d1": [0.2, 0.3, 0.1], "d2": [0.3, 0.1, 0.2]},
        index=["X", RF, "Y"],
    )
    elo = gap.bradley_terry(pool)
    assert elo.to_dict() == pytest.approx({"X": 1000.0, RF: 1000.0, "Y": 1000.0}, abs=1e-4)


def test_identical_methods_get_the_same_elo_and_a_dominant_one_the_top():
    pool = pd.DataFrame(
        {"d0": [0.1, 0.1, 0.9, 0.2], "d1": [0.3, 0.3, 0.8, 0.5], "d2": [0.2, 0.2, 0.7, 0.4]},
        index=["best", "twin", RF, "other"],
    )
    elo = gap.bradley_terry(pool)
    assert elo["best"] == pytest.approx(elo["twin"])
    assert elo["best"] == elo.max()


def test_elo_needs_every_method_on_every_dataset():
    pool = pd.DataFrame({"d0": [0.1, np.nan], "d1": [0.2, 0.3]}, index=["A", RF])
    with pytest.raises(ValueError, match="every method"):
        gap.bradley_terry(pool)


def test_missing_results_take_the_random_forest_error_of_the_dataset():
    pool = pd.DataFrame({"d0": [0.1, np.nan, 0.3], "d1": [np.nan, 0.5, 0.6]}, index=["A", "B", RF])
    filled = gap.fill_missing(pool)
    assert filled.loc["B", "d0"] == pytest.approx(0.3)
    assert filled.loc["A", "d1"] == pytest.approx(0.6)
    assert filled.loc["A", "d0"] == pytest.approx(0.1)


def test_geometric_mean_and_bootstrap_interval():
    assert gap.geo_mean([2.0, 0.5]) == pytest.approx(1.0)
    assert gap.geo_mean([1.2, 1.2, 1.2]) == pytest.approx(1.2)
    low, high = gap.bootstrap_ci([1.1] * 10)
    assert low == pytest.approx(1.1) and high == pytest.approx(1.1)
    ratios = np.array([0.9, 1.1, 1.3, 0.8, 1.05, 1.4])
    assert gap.bootstrap_ci(ratios) == gap.bootstrap_ci(ratios)  # fixed seed
    low, high = gap.bootstrap_ci(ratios)
    assert low <= gap.geo_mean(ratios) <= high


def test_rank_counts_strictly_better_errors():
    assert gap.rank_among(0.5, [0.4, 0.6]) == 2
    assert gap.rank_among(0.3, [0.4, 0.6]) == 1
    assert gap.rank_among(0.5, [0.5, 0.5]) == 1  # a tie is not better


def test_property_buckets_have_the_stated_edges():
    props = pd.DataFrame(
        {
            "task": ["binary"] * 4,
            "train_rows": [1999, 2000, 10000, 10001],
            "features": [20, 21, 100, 101],
            "cat_features": [0, 1, 0, 0],
            "max_cardinality": [0, 21, 20, 0],
            "missing_any": [False, False, True, False],
            "minority_share": [0.09, 0.10, 0.5, 0.5],
            "classes": [2, 3, 2, 2],
        },
        index=["a", "b", "c", "d"],
    )
    buckets = gap.bucket_frame(props)
    assert list(buckets["train rows"]) == ["<2K", "2K-10K", "2K-10K", ">10K"]
    assert list(buckets["features"]) == ["<=20", "21-100", "21-100", ">100"]
    assert list(buckets["categorical"]) == ["none", "any", "none", "none"]
    assert list(buckets["max cardinality"]) == ["<21", ">=21", "<21", "<21"]
    assert list(buckets["minority share"]) == ["<10%", ">=10%", ">=10%", ">=10%"]
    assert list(buckets["classes"]) == ["2", ">2", "2", "2"]
    assert list(buckets["missing"]) == ["none", "none", "any", "none"]


def test_regression_has_no_minority_or_class_bucket():
    props = pd.DataFrame(
        {
            "task": ["regression"],
            "train_rows": [5000],
            "features": [9],
            "cat_features": [0],
            "max_cardinality": [0],
            "missing_any": [False],
            "minority_share": [np.nan],
            "classes": [np.nan],
        },
        index=["r"],
    )
    buckets = gap.bucket_frame(props)
    assert buckets.loc["r", "minority share"] == "n/a"
    assert buckets.loc["r", "classes"] == "n/a"


def test_changing_one_error_moves_that_method_only_through_its_wins():
    pool = pd.DataFrame(
        {"d0": [0.1, 0.2, 0.3, 0.4, 0.5], "d1": [0.1, 0.2, 0.3, 0.4, 0.5]},
        index=["m0", "m1", "m2", RF, "m4"],
    )
    worse = gap.with_error(pool, "m0", pd.Series({"d0": 0.9}))
    assert worse.loc["m0", "d0"] == 0.9 and pool.loc["m0", "d0"] == 0.1  # a copy
    elo_before, rank_before = gap.pool_elo(pool)
    elo_after, _ = gap.pool_elo(worse)
    assert elo_after["m0"] < elo_before["m0"]
    assert rank_before["m0"] == 1


def _seven_method_pool(datasets: int = 10) -> pd.DataFrame:
    # "C" is the worst method on every dataset (rank 7 of 7)
    rng = np.random.default_rng(1)
    base = rng.uniform(0.2, 0.4, size=(6, datasets))
    values = np.vstack([np.full(datasets, 0.9), base])
    names = ["C", "m1", "m2", "m3", RF, "m5", "m6"]
    return pd.DataFrame(values, index=names, columns=[f"d{i}" for i in range(datasets)])


def test_reduction_to_rank_finds_the_first_cut_that_reaches_the_target():
    pool = _seven_method_pool()
    mask = pd.Series(True, index=pool.columns)
    assert gap.rank_with_reduction(pool, "C", mask, 0.0) == 7
    r = gap.reduction_to_rank(pool, "C", mask)
    assert r is not None and 0.0 < r < 0.99
    assert gap.rank_with_reduction(pool, "C", mask, r) <= gap.TARGET_RANK
    assert gap.rank_with_reduction(pool, "C", mask, r - 0.01) > gap.TARGET_RANK


def test_a_scope_that_cannot_lift_the_rank_gives_none():
    pool = _seven_method_pool()
    nothing = pd.Series(False, index=pool.columns)
    assert gap.reduction_to_rank(pool, "C", nothing) is None


def test_closing_the_gap_moves_only_the_datasets_behind_the_target():
    run = pd.Series({"d0": 0.4, "d1": 0.1, "d2": 0.4})
    target = pd.Series({"d0": 0.1, "d1": 0.4, "d2": 0.1})
    mask = pd.Series({"d0": True, "d1": True, "d2": False})
    half = gap.close_gap(run, target, mask, 0.5)
    assert half["d0"] == pytest.approx(0.2)  # the geometric midpoint of 0.4 and 0.1
    assert half["d1"] == pytest.approx(0.1)  # not behind: unchanged
    assert half["d2"] == pytest.approx(0.4)  # outside the mask: unchanged
    assert gap.close_gap(run, target, mask, 1.0)["d0"] == pytest.approx(0.1)


def test_a_transplant_applies_the_other_entrants_ratio_dataset_by_dataset():
    pool = _seven_method_pool(4)
    board = pool.copy()
    board.loc["after"] = board.loc["m1"] * np.array([0.5, 1.0, 2.0, 1.0])
    pool = pd.concat([pool, board.loc[["after"]]])
    table = gap.transplant_whatif(pool, board, "C", pairs=(("m1", "after", "m1 -> after"),))
    assert list(table["change"]) == ["m1 -> after"]
    assert table.loc[0, "datasets better"] == 1
    assert table.loc[0, "G"] == pytest.approx(1.0)
    skipped = gap.transplant_whatif(pool, board, "C", pairs=(("m1", "absent", "x"),))
    assert skipped.empty


def test_the_noise_what_if_without_noise_is_the_uniform_cut():
    pool = _seven_method_pool()
    table = gap.noise_whatif(pool, "C", cuts=(0.5,), sds=(0.0,), draws=3)
    elo_before, _ = gap.pool_elo(pool)
    elo_after, rank_after = gap.pool_elo(gap.scaled(pool, "C", math.exp(-0.5)))
    assert table.loc[0, "elo gain mean"] == pytest.approx(elo_after["C"] - elo_before["C"])
    assert table.loc[0, "rank median"] == rank_after["C"]
    again = gap.noise_whatif(pool, "C", cuts=(0.1,), sds=(0.05,), draws=3)
    assert again.equals(gap.noise_whatif(pool, "C", cuts=(0.1,), sds=(0.05,), draws=3))  # seeded


def test_the_split_what_if_takes_the_after_errors_on_the_named_datasets_only():
    board = _seven_method_pool(4).drop(index="C")
    before = pd.Series(0.9, index=board.columns)
    after = pd.Series([0.1, 0.1, 0.9, 0.9], index=board.columns)
    table = gap.split_whatif(board, before, after, ("d0", "d1", "not a dataset"))
    assert list(table["case"]) == [
        "before",
        "after",
        "before with after on d0",
        "before with after on d1",
        "before with after on all of these",
    ]
    assert table.loc[0, "elo gain over before"] == 0.0
    assert table.iloc[-1]["elo"] == pytest.approx(table.loc[1, "elo"])  # d2, d3 equal anyway


def test_the_crowd_counts_the_top_entrants_within_a_relative_width():
    pool = pd.DataFrame(
        {"d0": [0.100, 0.1005, 0.15, 0.9], "d1": [0.200, 0.30, 0.2010, 0.9]},
        index=["run", "a", "b", RF],
    )
    table = gap.crowd(pool, "run", top=2, widths=(0.01,))
    # a is within 1% of the run on d0 only, b on d1 only: 2 of the 4 (dataset, entrant) pairs
    assert table.loc[0, "share of pairs"] == pytest.approx(0.5)
    assert (
        gap.crowd(pool, "run", exclude=("a",), top=1, widths=(0.01,)).loc[0, "share of pairs"]
        == 0.5
    )
