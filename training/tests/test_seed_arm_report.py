"""Tests of the seed report for arms trained from scratch (`scripts/seed_arm_report.py`, v2 plan
§3, "Arm trained from scratch")."""

import json
import math

import numpy as np
import pandas as pd
import pytest
from train_helpers import load_script

from tabsil_train.benchmarks import paired

KIND = {"a": "binary", "b": "multiclass", "r": "regression", "q": "regression"}


def _file(path, errors: dict, splits: int = 2):
    rows = [
        {"dataset": d, "fold": s, "metric_error": e, "problem_type": KIND[d], "imputed": False}
        for d, e in errors.items()
        for s in range(splits)
    ]
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


def _script():
    return load_script("scripts/seed_arm_report.py", "seed_arm_report")


def _errors(script, errors: dict) -> pd.DataFrame:
    frame = pd.DataFrame(
        [{"dataset": d, "split": 0, "problem_type": KIND[d], "error": e} for d, e in errors.items()]
    )
    return script.dataset_errors(frame)


def test_score_is_the_mean_log_error_and_g_uses_the_seed_means():
    script = _script()
    control = {
        "0": {"s": _errors(script, {"a": 0.2, "b": 1.0, "r": 4.0, "q": 2.0})},
        "1": {"s": _errors(script, {"a": 0.4, "b": 1.0, "r": 4.0, "q": 2.0})},
    }
    arm = {"0": {"s": _errors(script, {"a": 0.15, "b": 0.5, "r": 4.0, "q": 2.2})}}
    out = script.compare_arm(control, arm)
    assert set(out) == {"s"}  # one suite: nothing to pool
    r = out["s"]
    expected = np.mean(np.log([0.2, 1.0, 4.0, 2.0]))
    assert r["scores"]["control"]["0"] == pytest.approx(expected)
    assert r["scores_by_task"]["regression"]["arm"]["0"] == pytest.approx(
        np.mean(np.log([4.0, 2.2]))
    )
    # the control error of "a" is the mean of its seeds, 0.3
    ratios = r["ratios"].set_index("dataset")["ratio"]
    assert ratios["s/a"] == pytest.approx(0.5) and ratios["s/q"] == pytest.approx(1.1)
    assert r["stats"]["all"]["G"] == pytest.approx((0.5 * 0.5 * 1.0 * 1.1) ** 0.25)
    assert r["stats"]["regression"]["G"] == pytest.approx(math.sqrt(1.1))
    assert (r["stats"]["all"]["better"], r["stats"]["all"]["worse"]) == (2, 1)
    # the arm seed against the control seed of the same number
    assert r["pairs"]["0"]["all"]["G"] == pytest.approx((0.75 * 0.5 * 1.0 * 1.1) ** 0.25)
    assert set(r["pairs"]) == {"0"}


def test_all_below_needs_every_arm_seed_below_every_control_seed():
    script = _script()

    def runs(values):
        return {str(i): {"s": _errors(script, {"a": v, "r": 1.0})} for i, v in enumerate(values)}

    below = script.compare_arm(runs([0.30, 0.32, 0.31]), runs([0.20, 0.29]))["s"]
    assert below["all_below"] and below["p_if_all_below"] == pytest.approx(0.1)
    assert below["reading"].startswith("third seed")
    mixed = script.compare_arm(runs([0.30, 0.32, 0.31]), runs([0.20, 0.305]))["s"]
    assert not mixed["all_below"] and mixed["reading"].startswith("not resolved (G at most")
    three = script.compare_arm(runs([0.30, 0.32, 0.31]), runs([0.20, 0.29, 0.28]))["s"]
    assert three["all_below"] and three["p_if_all_below"] == pytest.approx(0.05)
    assert three["reading"].startswith("adopted on the evidence")
    # three seeds below, but G of the seed means above 0.98
    close = script.compare_arm(runs([0.300, 0.302, 0.301]), runs([0.299, 0.298, 0.297]))["s"]
    assert close["all_below"] and close["reading"].startswith("not resolved (G at most")
    assert script.permutation_p(3, 2) == pytest.approx(0.1)


def test_reading_drops_an_arm_only_with_a_ci_above_one():
    script = _script()
    assert script.reading(False, 2, {"G": 1.03, "ci_low": 1.01}).startswith("dropped")
    assert script.reading(False, 2, {"G": 1.03, "ci_low": 0.99}) == "not resolved (G above 1.01)"
    assert script.reading(False, 2, {"G": 1.005, "ci_low": 0.99}) == "not resolved (G at most 1.01)"


def test_pooled_puts_the_suites_together_with_one_weight_per_dataset():
    script = _script()
    control = {
        "0": {
            "one": _errors(script, {"a": 0.2, "r": 1.0}),
            "two": _errors(script, {"a": 0.4}),  # the same name in another suite stays apart
        }
    }
    arm = {"0": {"one": _errors(script, {"a": 0.1, "r": 1.0}), "two": _errors(script, {"a": 0.4})}}
    out = script.compare_arm(control, arm)
    assert list(out) == ["one", "two", "pooled"]
    assert out["pooled"]["stats"]["all"]["datasets"] == 3
    assert out["pooled"]["stats"]["all"]["G"] == pytest.approx(0.5 ** (1 / 3))
    assert out["pooled"]["scores"]["control"]["0"] == pytest.approx(
        np.mean(np.log([0.2, 1.0, 0.4]))
    )


def test_seeds_must_share_their_splits():
    script = _script()
    control = {
        "0": {"s": _errors(script, {"a": 0.2, "r": 1.0})},
        "1": {"s": _errors(script, {"a": 0.2})},
    }
    with pytest.raises(ValueError, match="same splits"):
        script.compare_arm(control, {"0": {"s": _errors(script, {"a": 0.2, "r": 1.0})}})
    with pytest.raises(ValueError, match="same datasets"):
        script.compare_arm(
            {"0": {"s": _errors(script, {"a": 0.2, "r": 1.0})}},
            {"0": {"s": _errors(script, {"a": 0.2})}},
        )


def test_the_script_reads_a_spec_of_local_files(tmp_path, capsys):
    script = _script()
    spec = {
        "control": {
            "0": {"s": _file(tmp_path / "c0.csv", {"a": 0.30, "b": 1.0, "r": 4.0, "q": 2.0})},
            "1": {"s": _file(tmp_path / "c1.csv", {"a": 0.32, "b": 1.1, "r": 4.1, "q": 2.1})},
        },
        "arms": {
            "x": {"0": {"s": _file(tmp_path / "x0.csv", {"a": 0.2, "b": 0.9, "r": 3.9, "q": 1.9})}}
        },
    }
    (tmp_path / "spec.json").write_text(json.dumps(spec))
    script.main([str(tmp_path / "spec.json"), "--out-dir", str(tmp_path / "out")])
    text = capsys.readouterr().out
    assert "## x" in text and "third seed" in text
    saved = json.loads((tmp_path / "out" / "report.json").read_text())
    assert saved["x"]["s"]["all_below"] is True
    assert saved["x"]["s"]["stats"]["all"]["G"] < 1
    ratios = pd.read_csv(tmp_path / "out" / "ratios_x_s.csv")
    assert len(ratios) == 4 and (ratios["splits"] == 2).all()
    # an imputed split is an error, as in the paired report
    rows = pd.read_csv(tmp_path / "x0.csv")
    rows.loc[0, "imputed"] = True
    rows.to_csv(tmp_path / "x0.csv", index=False)
    with pytest.raises(ValueError, match="imputed"):
        script.main([str(tmp_path / "spec.json")])
    assert paired.ERROR_FLOOR > 0
