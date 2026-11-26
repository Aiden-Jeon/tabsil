"""fev-bench forecasting evaluation helpers (`tabsil_train.benchmarks.fevbench`) that need neither
the fev package nor the data: reference files, frequency seasonality, the Seasonal Naive check,
Elo and its pools."""

import numpy as np
import pandas as pd
import pytest

from tabsil_train.benchmarks import fevbench as fb


def _ours(task: str = "ecdc_ili", scale: float = 0.5) -> pd.DataFrame:
    """TabSil rows (and our Seasonal Naive) for every window of `task`: TabSil's errors are the
    reference Seasonal Naive's times `scale`."""
    snaive = fb.reference_windows().query("method == @fb.ANCHOR and task == @task")
    tabsil = snaive.assign(method=fb.METHOD, **{m: snaive[m] * scale for m in fb.METRICS})
    ours = snaive.assign(method=f"{fb.ANCHOR} (ours)")
    rows = pd.concat([tabsil, ours], ignore_index=True)
    return rows.assign(cutoff=rows.cutoff.astype(str))


def test_reference_files_cover_every_window_of_every_task():
    ref = fb.reference_windows()
    assert ref.method.nunique() == 7
    for method, g in ref.groupby("method"):
        assert set(g.task) == set(fb.TASKS), method
        for task, w in g.groupby("task"):
            assert sorted(w.window) == list(range(fb.TASKS[task]["num_windows"])), (method, task)
    public = fb.reference_public()
    assert set(fb.PUBLIC_ONLY) <= set(public.method)
    assert set(public.task) == set(fb.TASKS)


def test_freq_seasonality_follows_autogluon():
    cases = {"h": 24, "2h": 12, "D": 7, "W-SUN": 1, "MS": 12, "ME": 12, "QS-DEC": 4, "15min": 96}
    for freq, expected in cases.items():
        assert fb.freq_seasonality(freq) == expected, freq
    assert fb.freq_seasonality("5h") == 1  # 24 is not a multiple of 5


def test_seasonal_naive_check():
    ours = _ours()
    assert max(fb.check_seasonal_naive(ours).values()) == 0.0
    off = ours.copy()
    off.loc[off.method == f"{fb.ANCHOR} (ours)", "SQL"] *= 1.001
    with pytest.raises(ValueError, match="differs"):
        fb.check_seasonal_naive(off)


def test_elo_anchor_order_and_missing_results():
    rows = []
    for t in range(3):
        for s in range(2):
            for method, err in [("a", 1.0), ("b", 2.0), (fb.ANCHOR, 3.0)]:
                rows.append({"method": method, "task": f"t{t}", "split": s, "error": err + 0.1 * s})
    long = pd.DataFrame(rows)
    out = fb.elo(long, rounds=0)
    assert out.loc[fb.ANCHOR, "elo"] == pytest.approx(1000.0)
    assert list(out.index) == ["a", "b", fb.ANCHOR]
    assert np.isclose(out.loc["a", "winrate"], 1.0) and out["elo+"].isna().all()
    with pytest.raises(ValueError, match="missing"):
        fb.elo(long.iloc[1:], rounds=0)
    with pytest.raises(ValueError, match="anchor"):
        fb.elo(long[long.method != fb.ANCHOR], rounds=0)


def test_pools_and_comparison():
    ours = _ours()
    pools = fb.pools(ours, "SQL")
    windows, mean = pools["windows"], pools["mean"]
    assert windows.method.nunique() == 8 and len(windows) == 8 * 10
    assert set(fb.PUBLIC_ONLY) <= set(mean.method) and (mean.split == "all").all()
    assert mean.groupby("method").size().eq(1).all()
    table = fb.comparison(ours).set_index("method")
    assert table.loc[fb.METHOD, "SQL_vs_snaive"] == pytest.approx(0.5)
    assert table.loc[fb.ANCHOR, "SQL_vs_snaive"] == pytest.approx(1.0)
    # with only some windows there is no task-mean pool
    assert set(fb.pools(ours[ours.window >= 8], "SQL")) == {"windows"}
