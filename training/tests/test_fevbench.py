"""fev-bench forecasting evaluation helpers (`tabsil_train.benchmarks.forecast.fevbench`) that need
neither the fev package nor the data: reference files, frequency seasonality, the Seasonal Naive
check, Elo and its pools, and the arguments `eval.py --suite fev` hands to the benchmark."""

import numpy as np
import pandas as pd
import pytest

from tabsil_train.benchmarks.forecast import fevbench as fb


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


def test_eval_suite_fev_runs_the_benchmark_with_the_forecaster(tmp_path, monkeypatch, capsys):
    # `eval.py --suite fev` (`tabsil_train.evaluation.fev`) with the benchmark's run replaced, so
    # neither the fev package nor the data is needed: the arguments reach the forecaster and the
    # run, and the tables it wrote are printed
    from train_helpers import load_script, tiny_checkpoint

    from tabsil import TabSilForecaster

    script = load_script("scripts/eval.py", "eval_script_fev")
    seen = {}

    def run(forecaster, tasks, windows, out_dir, log=print):
        seen.update(forecaster=forecaster, tasks=tasks, windows=windows, out_dir=out_dir)
        out_dir.mkdir(parents=True)
        table = pd.DataFrame({"task": ["ecdc_ili"] * 2, "method": [fb.METHOD, fb.ANCHOR]})
        table = table.assign(SQL=[1.0, 2.0], MASE=[1.5, 2.5])
        table.to_csv(out_dir / "comparison.csv", index=False)
        elo = pd.DataFrame({"pool": ["windows"], "metric": ["SQL"], "method": [fb.METHOD]})
        elo.assign(elo=[1100.0]).to_csv(out_dir / "elo.csv", index=False)
        return {"tabsil_seconds": 1.0, "seconds": {"total": 2.0}}

    monkeypatch.setattr(fb, "run", run)
    monkeypatch.delenv("MLFLOW_RUN_ID", raising=False)
    out = tmp_path / "eval"
    args = ["--suite", "fev", "--datasets", "ecdc_ili", "--fev-windows", "2", "--out-dir", str(out)]
    model = tiny_checkpoint(tmp_path / "m.pt")
    script.main([*args, "--regressor-path", model, "--n-estimators", "2", "--seed", "3",
                 "--max-context-length", "64", "--device", "cpu"])  # fmt: skip
    assert seen["tasks"] == ["ecdc_ili"] and seen["windows"] == 2
    assert seen["out_dir"] == out / "fev"
    forecaster = seen["forecaster"]
    assert isinstance(forecaster, TabSilForecaster) and forecaster.max_context_length == 64
    assert (forecaster.n_estimators, forecaster.random_state, forecaster.device) == (2, 3, "cpu")
    printed = capsys.readouterr().out
    assert "task mean SQL" in printed and "Elo (Seasonal Naive = 1000)" in printed
    assert "TabSil time 1s, total 2s" in printed
    with pytest.raises(SystemExit, match="regressor-path"):
        script.main(args)
    # a window count below 1 is refused with the arguments, before the model is loaded and the
    # run is logged as one over all windows
    seen.clear()
    capsys.readouterr()
    with pytest.raises(SystemExit):
        script.main(["--suite", "fev", "--fev-windows", "0", "--regressor-path", model])
    assert "--fev-windows must be >= 1" in capsys.readouterr().err and not seen
