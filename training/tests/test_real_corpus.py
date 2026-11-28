"""Tests of the real-table corpus script (`training/scripts/real_corpus.py`): name normalization,
the row hashes, the 1% overlap rule, the shape and Jaccard rules and the candidate filters, on
small synthetic frames. No download happens here."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# registered in sys.modules before execution: the script's dataclasses look their module up there
SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "real_corpus.py"
_spec = importlib.util.spec_from_file_location("real_corpus", SCRIPT)
real_corpus = importlib.util.module_from_spec(_spec)
sys.modules["real_corpus"] = real_corpus
_spec.loader.exec_module(real_corpus)


def test_normalize_name_removes_separators_and_version_suffixes():
    assert real_corpus.normalize_name("Blood-Transfusion-Service-Center") == (
        "bloodtransfusionservicecenter"
    )
    assert real_corpus.normalize_name("credit-g") == "creditg"
    assert real_corpus.normalize_name("dataset_v2") == "dataset"
    assert real_corpus.normalize_name("wine quality (version 3)") == "winequality"
    assert real_corpus.normalize_name("pc4") == "pc4"


def test_normalize_feature_keeps_only_lower_alphanumerics():
    assert real_corpus.normalize_feature("Feature_#1 (x)") == "feature1x"


def test_jaccard_and_empty_sets():
    assert real_corpus.jaccard({"a", "b"}, {"b", "c"}) == pytest.approx(1 / 3)
    assert real_corpus.jaccard(set(), {"a"}) == 0.0


def test_shape_close_is_within_one_percent_and_same_predictors():
    assert real_corpus.shape_close(1005, 4, 1000, 4)
    assert not real_corpus.shape_close(1020, 4, 1000, 4)
    assert not real_corpus.shape_close(1000, 5, 1000, 4)


def test_numbers_round_to_six_significant_digits_and_mark_missing():
    text = real_corpus.numbers_to_text([1.23456789, 0.0, -0.0, np.nan, np.inf, 1.0000001])
    assert text[0] == "1.23457"
    assert text[1] == text[2] == "0"
    assert text[3] == real_corpus.MISSING
    assert text[4] == "+inf"
    assert text[5] == "1"


def test_named_hash_ignores_column_order_but_not_names():
    a = pd.DataFrame({"x": [1.0, 2.0], "y": ["p", None]})
    b = pd.DataFrame({"y": ["p", None], "x": [1.0, 2.0]})
    renamed = a.rename(columns={"x": "z"})
    named_a, values_a = real_corpus.row_keys(a)
    named_b, _ = real_corpus.row_keys(b)
    named_r, values_r = real_corpus.row_keys(renamed)
    assert named_a == named_b
    assert named_a != named_r
    assert values_a == values_r


def test_rounding_makes_near_equal_rows_equal_and_missing_differs_from_text():
    a = pd.DataFrame({"x": [1.0000001], "y": [np.nan]})
    b = pd.DataFrame({"x": [1.0], "y": ["nan"]})
    named_a, _ = real_corpus.row_keys(a)
    named_b, _ = real_corpus.row_keys(b)
    assert named_a != named_b
    assert (
        real_corpus.row_keys(a)[0]
        == real_corpus.row_keys(pd.DataFrame({"x": [1.0], "y": [None]}))[0]
    )


def test_first_over_needs_more_than_one_percent_of_rows():
    hashes = [bytes([i]) for i in range(200)]
    index = real_corpus.build_index({"bench": set(hashes[:2])})
    assert real_corpus.first_over(index, hashes[:200]) is None  # 2 of 200 is 1%, not above
    index = real_corpus.build_index({"bench": set(hashes[:3])})
    assert real_corpus.first_over(index, hashes[:200]) == ("bench", 3)


def _cand(**kw):
    base = dict(
        id=1000,
        name="cand",
        version="1",
        rows=1000,
        classes=2,
        target="y",
        predictors=["a", "b", "c", "d"],
        tags=[],
        desc={},
        features=[],
    )
    base.update(kw)
    return real_corpus.Cand(**base)


def _bench(**kw):
    base = dict(suite="s", name="bench", openml_id=5, rows=1000, predictors=["a", "b", "c", "d"])
    base.update(kw)
    return real_corpus.Bench(**base)


def test_rule_one_matches_the_id_then_the_normalized_name():
    bench = [_bench(name="Other-Set")]
    by_id = {5: bench[0]}
    by_name = {real_corpus.normalize_name("Other-Set"): bench[0]}
    rule, label = real_corpus.metadata_match(_cand(id=5), bench, by_id, by_name)
    assert rule.startswith("1: same OpenML")
    rule, _ = real_corpus.metadata_match(_cand(id=7, name="other_set"), bench, {}, by_name)
    assert rule == "1: name equals a benchmark name"


def test_rule_two_needs_shape_and_half_the_names_rule_three_needs_most():
    bench = [_bench(openml_id=None, name="far", rows=5000, predictors=["a", "b", "c", "d", "e"])]
    cand = _cand(predictors=["a", "b", "c", "x"])  # Jaccard 3/5 = 0.6, shapes differ
    assert real_corpus.metadata_match(cand, bench, {}, {}) is None
    bench = [_bench(openml_id=None, name="near", rows=1004, predictors=["a", "b", "c", "x"])]
    rule, _ = real_corpus.metadata_match(cand, bench, {}, {})
    assert rule.startswith("2:")
    bench = [_bench(openml_id=None, name="far", rows=5000, predictors=["a", "b", "c", "x", "z"])]
    rule, _ = real_corpus.metadata_match(_cand(predictors=["a", "b", "c", "x", "z"]), bench, {}, {})
    assert rule.startswith("3:")


def test_candidate_filters_reject_text_and_wrong_targets():
    base = dict(
        desc={"status": "active", "default_target_attribute": "y", "name": "tab"},
        target="y",
        features=[{"name": "y", "data_type": "nominal"}],
        predictors=[{"name": "a"}, {"name": "b"}],
        missing=0,
        rows=1000,
        classes=2,
    )
    assert real_corpus.filter_reason(**base) is None
    text = dict(base, desc={**base["desc"], "tag": ["image"]})
    assert "image" in real_corpus.filter_reason(**text)
    numeric_class = dict(base, features=[{"name": "y", "data_type": "numeric"}])
    assert "classification with a numeric" in real_corpus.filter_reason(**numeric_class)
    context = dict(base, desc={**base["desc"], "name": "contexts"})
    assert real_corpus.filter_reason(**context) is None


@pytest.mark.parametrize(
    "name",
    [
        "fri_c0_1000_50",
        "fried",
        "friedman_x",
        "2dplanes",
        "mv",
        "bank8FM",
        "puma32H",
        "kin8nm",
        "BNG(breast-w)",
        "led24",
        "LED-display-domain-7digit",
        "waveform-5000",
        "fourclass",
        "COMET_MC_SAMPLE",
        "delta_elevators",
        "EpistasisBench",
    ],
)
def test_synthetic_rule_excludes_generator_name_families(name):
    assert real_corpus.synthetic_reason({"name": name, "description": ""}) is not None


def test_synthetic_rule_reads_the_description_as_whole_words():
    assert "artificial" in real_corpus.synthetic_reason(
        {"name": "tab", "description": "This is an Artificial table."}
    )
    assert real_corpus.synthetic_reason({"name": "tab", "description": "Simulated data."})
    assert real_corpus.synthetic_reason(
        {"name": "t", "description": "from Monte Carlo simulations"}
    )
    # a word inside another word does not count, and real names pass
    assert (
        real_corpus.synthetic_reason({"name": "tab", "description": "regenerated tissue"}) is None
    )
    assert real_corpus.synthetic_reason({"name": "hypothyroid", "description": "patients"}) is None
    assert real_corpus.synthetic_reason({"name": "contexts", "description": ""}) is None


def test_name_stem_is_the_first_token():
    assert real_corpus.name_stem("disclosure_x_noise") == "disclosure"
    assert real_corpus.name_stem("Blood-Transfusion") == "blood"
    assert real_corpus.name_stem("pc4") == "pc4"


def test_a_candidate_the_server_does_not_serve_is_recorded_and_skipped():
    # OpenML answers HTTP 500 for one candidate's features after the retries: that candidate is
    # recorded as a filter exclusion and the pool goes on with the next one
    import urllib.error
    from types import SimpleNamespace

    def quality(rows, classes):
        return [
            {"name": "NumberOfClasses", "value": str(classes)},
            {"name": "NumberOfInstances", "value": str(rows)},
        ]

    listed = {
        "data": {
            "dataset": [
                {"did": 1, "name": "broken", "quality": quality(1000, 2)},
                {"did": 2, "name": "fine", "quality": quality(1000, 2)},
            ]
        }
    }
    desc = {"status": "active", "default_target_attribute": "y", "description": ""}
    features = [
        {"name": "y", "data_type": "nominal", "is_target": "true"},
        {"name": "a", "data_type": "numeric"},
        {"name": "b", "data_type": "numeric"},
    ]

    class FakeWeb:
        def json(self, url):
            if url == real_corpus.LIST_URL:
                return listed
            if url.endswith("/data/features/1"):
                raise urllib.error.HTTPError(url, 500, "Internal Server Error", {}, None)
            if "/data/features/" in url:
                return {"data_features": {"feature": features}}
            did = int(url.rsplit("/", 1)[1])
            return {"data_set_description": {**desc, "name": f"tab{did}"}}

    args = SimpleNamespace(max_candidates=10, pool_deadline=float("inf"))
    cands, excluded, examined, _ = real_corpus.candidate_pool(FakeWeb(), args)
    assert [c.id for c in cands] == [2] and examined == 2
    assert excluded[0]["openml_id"] == 1 and "not served" in excluded[0]["reason"]


def test_rule_six_matches_the_row_count_alone():
    # another encoding, other names and another column count: only the rows tell the copy
    bench = [_bench(openml_id=None, name="credit", rows=1000, predictors=["p", "q"])]
    rule, label = real_corpus.metadata_match(_cand(rows=1008), bench, {}, {})
    assert rule.startswith("6:") and "credit" in label
    assert real_corpus.metadata_match(_cand(rows=1020), bench, {}, {}) is None


def _credit_like(rng, n=1000):
    # a benchmark-like table: a continuous amount, two labelled categories, an age, a binary target
    purpose = rng.choice(list("ABCDEF"), size=n, p=[0.35, 0.25, 0.15, 0.12, 0.08, 0.05])
    job = rng.choice(["skilled", "unskilled", "manager", "none"], size=n, p=[0.6, 0.2, 0.15, 0.05])
    return pd.DataFrame(
        {
            "credit_amount": np.round(rng.lognormal(8, 0.7, size=n)),
            "purpose": purpose,
            "job": job,
            "age": rng.integers(19, 76, size=n),
            "risk": rng.choice(["good", "bad"], size=n, p=[0.7, 0.3]),
        }
    )


def _index(frame):
    source = real_corpus.Source("openml:1", labels=["tabarena-51:credit"])
    source.profiles = real_corpus.column_profiles(frame, real_corpus.SAMPLE)
    return real_corpus.ColumnIndex({"openml:1": source})


def test_rule_seven_finds_relabelled_subset_and_scaled_copies():
    rng = np.random.default_rng(0)
    bench = _credit_like(rng)
    index = _index(bench)
    codes = {c: i for i, c in enumerate("ABCDEF")}
    # categories as codes, columns renamed, one column dropped, the target as 0/1
    relabelled = pd.DataFrame(
        {
            "Purpose": bench["purpose"].map(codes),
            "Amount": bench["credit_amount"],
            "Job": bench["job"].map({"skilled": 2, "unskilled": 1, "manager": 3, "none": 0}),
            "target": (bench["risk"] == "bad").astype(int),
        }
    )
    assert index.match(real_corpus.column_profiles(relabelled)) is not None
    # 80% of the rows, shuffled
    subset = bench.sample(frac=0.8, random_state=1)
    assert index.match(real_corpus.column_profiles(subset)) is not None
    # the amount standardized and the age shifted: matched by the normalized values
    scaled = bench.assign(
        credit_amount=(bench["credit_amount"] - bench["credit_amount"].mean())
        / bench["credit_amount"].std(),
        age=bench["age"] + 0.5,
    )
    assert index.match(real_corpus.column_profiles(scaled)) is not None


def test_rule_seven_leaves_an_unrelated_table():
    index = _index(_credit_like(np.random.default_rng(0)))
    rng = np.random.default_rng(5)
    n = 900
    other = pd.DataFrame(
        {
            "income": np.round(rng.gamma(3, 2000, size=n), 2),
            "region": rng.choice(list("NSEW"), size=n),
            "level": rng.choice(["low", "mid", "high"], size=n, p=[0.3, 0.4, 0.3]),
            "y": rng.normal(size=n),
        }
    )
    assert index.match(real_corpus.column_profiles(other)) is None


def test_column_profiles_sample_the_smallest_hashes_and_skip_weak_columns():
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        {
            "constant": np.ones(5000),
            "few_ints": rng.integers(0, 90, size=5000),  # integers: need more than 100 values
            "many": rng.normal(size=5000),
        }
    )
    profiles = real_corpus.column_profiles(frame, sample=100)
    # the constant column is left out, and so is the column of 90 distinct integers (more than
    # LOW_CARD for its shares, at most MIN_DISTINCT_INT for its values); `many` is sampled
    assert len(profiles) == 1
    (many,) = profiles
    assert len(many.raw) == 100 and many.raw_cut == int(many.raw[-1])
    assert np.all(np.diff(many.raw.astype(np.float64)) >= 0)
