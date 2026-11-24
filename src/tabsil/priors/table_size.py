"""Table size rule: the train rows, eval rows and features of each batch (TabICLv2 training
scripts, soda-inria/tabicl `scripts/train_v2_clf_stage*.sh`, `tabicl.prior._dataset.PriorDataset`).
"""

from __future__ import annotations

import numpy as np

# TabICLv2's feature cap for tables with many rows (`PriorDataset.adjust_max_features`):
# (up to this many rows, feature cap). Tables of at most 10,240 rows keep `max_features`
LONG_TABLE_FEATURE_CAPS = (
    (10_240, None),
    (20_000, 80),
    (30_000, 60),
    (40_000, 40),
    (50_000, 30),
    (60_000, 20),
    (65_000, 15),
)
# Feature cap for tables longer than the last row count above (the else of TabICLv2's
# `adjust_max_features`)
LONG_TABLE_BEYOND_CAP = 10


def long_table_feature_cap(n_rows: int, max_features: int) -> int:
    """Feature cap of a table with `n_rows` rows (TabICLv2 `adjust_max_features`). Tables above
    65,000 rows get 10, as in TabICLv2 (also as there, without comparing against `max_features`)."""
    for rows, cap in LONG_TABLE_FEATURE_CAPS:
        if n_rows <= rows:
            return max_features if cap is None else min(cap, max_features)
    return LONG_TABLE_BEYOND_CAP


def check_split_sizes(
    *,
    min_rows: int,
    max_rows: int,
    min_train_fraction: float,
    max_train_fraction: float,
    max_features: int,
    max_cells: int,
    min_train_rows: int = 1,
) -> None:
    """Whether the table size settings make sense (raised before a DataLoader worker would fail on
    them). The shortest table must still have `min_train_rows` train rows (2 for classification)
    and one eval row."""
    if min_rows < max(min_train_rows, 1) + 1 or max_rows < min_rows:
        raise ValueError(
            f"row counts must satisfy min_train_rows + 1 ≤ min_rows ≤ max_rows: {min_train_rows},"
            f" {min_rows}, {max_rows}"
        )
    if not 0.0 < min_train_fraction <= max_train_fraction < 1.0:
        raise ValueError(
            "train fractions must satisfy 0 < min_train_fraction ≤ max_train_fraction < 1:"
            f" {min_train_fraction}, {max_train_fraction}"
        )
    if max_features < 1:
        raise ValueError(f"max_features must be at least 1: {max_features}")
    if max_rows > max_cells:
        raise ValueError(
            f"max_rows {max_rows} is larger than max_cells {max_cells} (not even one feature fits)"
        )


def sample_split_size(
    rng: np.random.Generator,
    *,
    min_rows: int,
    max_rows: int,
    min_train_fraction: float,
    max_train_fraction: float,
    max_features: int,
    log_uniform_rows: bool = False,
    max_cells: int,
    min_train_rows: int = 1,
) -> tuple[int, int, int]:
    """(train rows, eval rows, features) by the TabICLv2 rule.

    1. Rows (train + eval): the value itself if min_rows == max_rows, otherwise log-uniform
       (`log_uniform_rows`) or uniform in [min_rows, max_rows] (`PriorDataset.sample_seq_len`)
    2. Train rows = int(rows × U(min_train_fraction, max_train_fraction)) (the fraction case of
       `sample_train_size`), clipped so that at least one train and one eval row remain
    3. Features = round(U(1, cap)), with the cap lowered for long tables (`long_table_feature_cap`).
       TabICLv2 has no cell limit; here the feature count is reduced to stay within `max_cells`.
    """
    if min_rows == max_rows:
        n_rows = max_rows
    elif log_uniform_rows:
        n_rows = int(np.exp(rng.uniform(np.log(min_rows), np.log(max_rows))))
    else:
        n_rows = int(rng.integers(min_rows, max_rows))  # upper end excluded, like TabICL's randint
    n_train = int(n_rows * rng.uniform(min_train_fraction, max_train_fraction))
    n_train = min(max(n_train, min_train_rows, 1), n_rows - 1)
    cap = long_table_feature_cap(n_rows, max_features)
    n_features = int(round(rng.uniform(1, cap)))
    n_features = max(1, min(n_features, max_cells // n_rows))
    return n_train, n_rows - n_train, n_features
