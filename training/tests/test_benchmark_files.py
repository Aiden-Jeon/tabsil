"""Pinned benchmark downloads and result tables shared by the suites (`benchmarks._files`): the
settings string of a model's rows (with `tabsil_train.hub.weights_digest`), its check, and one
run's bookkeeping of `results.csv` (`Results`)."""

import hashlib
import io
import json

import pandas as pd
import pytest
import torch
from train_helpers import tiny_checkpoint, tiny_model

from tabsil.recipe import default_inference
from tabsil_train.benchmarks import _files
from tabsil_train.benchmarks.unsupervised import adbench
from tabsil_train.hub import load_model, weights_digest

GOOD = b"benchmark bytes"
PIN = hashlib.sha256(GOOD).hexdigest()


@pytest.fixture
def serve(monkeypatch):
    """Serve `body[0]` for every URL; returns the list of requested URLs."""
    body, urls = [GOOD], []

    def urlopen(url, timeout):
        urls.append(url)
        return io.BytesIO(body[0])

    monkeypatch.setattr(_files.urllib.request, "urlopen", urlopen)
    return body, urls


def test_a_good_download_is_cached_and_reused(tmp_path, serve):
    _, urls = serve
    path = tmp_path / "sub" / "data.npz"
    assert _files.download_verified("https://x/data.npz", path, PIN) == path
    assert path.read_bytes() == GOOD
    _files.download_verified("https://x/data.npz", path, PIN)
    assert len(urls) == 1  # the second call uses the cache
    assert sorted(p.name for p in path.parent.iterdir()) == ["data.npz"]


def test_a_bad_download_is_not_cached(tmp_path, serve):
    body, urls = serve
    body[0] = b"<html>error page</html>"
    path = tmp_path / "data.npz"
    with pytest.raises(ValueError, match="sha256"):
        _files.download_verified("https://x/data.npz", path, PIN)
    assert not any(tmp_path.iterdir())  # neither the file nor the .part
    body[0] = GOOD  # the next call downloads again and succeeds
    _files.download_verified("https://x/data.npz", path, PIN)
    assert len(urls) == 2 and path.read_bytes() == GOOD


def test_a_changed_cached_file_is_rejected(tmp_path, serve):
    path = tmp_path / "data.npz"
    _files.download_verified("https://x/data.npz", path, PIN)
    path.write_bytes(b"edited")
    with pytest.raises(ValueError, match="not the pinned file"):
        _files.download_verified("https://x/data.npz", path, PIN)


def test_adbench_does_not_cache_a_bad_download(tmp_path, serve):
    body, _ = serve
    body[0] = b"<html>"
    with pytest.raises(ValueError, match="ADBench @ 3dac822"):
        adbench.npz_path("yeast", tmp_path)
    assert not any(tmp_path.iterdir())


def test_read_results(tmp_path):
    empty = _files.read_results(tmp_path / "results.csv", ["a", "b"])
    assert empty.empty and list(empty.columns) == ["a", "b"]
    (tmp_path / "results.csv").write_text("a,b\n1,2\n")
    assert _files.read_results(tmp_path / "results.csv", ["a", "b"]).to_dict("records") == [
        {"a": 1, "b": 2}
    ]


def _linear():
    model = torch.nn.Linear(2, 1)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1.0, -2.0]]))
        model.bias.copy_(torch.tensor([0.5]))
    return model


def test_weights_digest_value():
    # the definition is dev-v2's (the arena cache key of its eval.py): the same weights must
    # keep their digest in both
    assert weights_digest(_linear()) == "9d90b5d8bf1471b05a4b836ce924b02e57f53004"
    borders = torch.tensor([0.0, 1.0, 2.0])
    assert weights_digest(_linear(), borders) == "225f440d777bddaa0987a45a83f24947afbad948"


def test_weights_digest_follows_the_weights_not_the_path(tmp_path):
    first = load_model(tiny_checkpoint(tmp_path / "a.pt"), "regression", tmp_path)
    copy = load_model(tiny_checkpoint(tmp_path / "copy.pt"), "regression", tmp_path)
    other = load_model(tiny_checkpoint(tmp_path / "b.pt", seed=1), "regression", tmp_path)
    assert weights_digest(*first) == weights_digest(*copy)
    assert weights_digest(*first) != weights_digest(*other)
    model, borders = first
    assert weights_digest(model, borders) != weights_digest(model, borders + 1)  # borders count
    assert weights_digest(model, borders) != weights_digest(model)


def test_model_settings_name_the_weights_and_the_saved_inference_settings():
    model, borders = tiny_model(), torch.linspace(-4, 4, 33)
    text = _files.model_settings(model, borders, 2, 3, n_permutations=5)
    assert text.startswith(
        f"weights={weights_digest(model, borders)} n_estimators=2 seed=3 n_permutations=5 "
    )
    inference = default_inference(model, "classification")
    assert f"member_kinds={'/'.join(inference['member_kinds'])}" in text
    for key in ("max_features_per_member", "feature_subsampling", "categories"):
        assert f" {key}={inference[key]}" in text
    assert text.endswith(f"softmax_temperature={inference['softmax_temperature']}")
    assert _files.model_settings(model, borders, 2, 3, n_permutations=5) == text
    for changed in (
        _files.model_settings(model, borders, 4, 3, n_permutations=5),
        _files.model_settings(model, borders, 2, 0, n_permutations=5),
        _files.model_settings(model, borders, 2, 3, n_permutations=10),
        _files.model_settings(tiny_model(seed=1), borders, 2, 3, n_permutations=5),
    ):
        assert changed != text
    # two Hub tags of the same weights with other settings in config.json (v1 and v1.1)
    model.inference_config = {"member_kinds": ["standard"], "softmax_temperature": 0.9}
    tagged = _files.model_settings(model, borders, 2, 3, n_permutations=5)
    assert tagged != text and tagged.split()[0] == text.split()[0]  # same weights
    assert " member_kinds=standard " in tagged and tagged.endswith("softmax_temperature=0.9")


def _rows(*rows):
    return pd.DataFrame(rows, columns=["method", "opts"])


def test_check_settings(tmp_path):
    path = tmp_path / "results.csv"
    settings = {"ours": "weights=a seed=0", "ours_median": "weights=a seed=0", "baseline": ""}
    done = _rows(
        ("ours", "weights=a seed=0"),
        ("ours_median", "weights=a seed=0"),
        ("baseline", float("nan")),  # an empty cell
        ("not_requested", "weights=b"),  # a method this run does not ask for
    )
    assert not _files.other_settings(done, settings).any()
    _files.check_settings(done, settings, path)
    _files.check_settings(_files.read_results(path, ["method", "opts"]), settings, path)  # no file
    for stale in (("ours_median", "weights=b seed=0"), ("ours", float("nan")), ("baseline", "x")):
        rows = pd.concat([done, _rows(stale)], ignore_index=True)
        assert _files.other_settings(rows, settings).tolist() == [False] * 4 + [True]
        with pytest.raises(ValueError, match="other settings") as error:
            _files.check_settings(rows, settings, path)
        assert stale[0] in str(error.value) and "another output directory" in str(error.value)
    # the message shows the row's settings and this run's
    with pytest.raises(ValueError, match=r"'weights=b seed=0'; this run: 'weights=a seed=0'"):
        _files.check_settings(_rows(("ours", "weights=b seed=0")), settings, path)


def test_results_appends_rows_and_records_failures(tmp_path):
    out, columns, logs = tmp_path / "out", ["method", "value", "opts"], []
    results = _files.Results(out, columns, {"a": "s"}, "suite", logs.append)
    assert results.done.empty and list(results.done.columns) == columns
    results.append([{"method": "a", "value": 1.5, "opts": "s"}])
    results.append([{"method": "a", "value": 2.5, "opts": "s"}, {"method": "b", "value": 3.0}])
    try:
        raise RuntimeError("boom")
    except RuntimeError as error:
        results.fail({"dataset": "d", "method": "a"}, error)
    rows, failures = results.finish()
    lines = (out / "results.csv").read_text().splitlines()
    assert lines[0] == "method,value,opts" and len(lines) == 4  # the header once
    assert rows.value.tolist() == [1.5, 2.5, 3.0]
    assert failures == [{"dataset": "d", "method": "a", "error": "RuntimeError('boom')"}]
    assert json.loads((out / "failures.json").read_text()) == failures
    assert logs[0].startswith("[suite] d a FAILED:\nTraceback")
    assert logs[0].rstrip().endswith("RuntimeError: boom")
    # a rerun starts from those rows; without a failed cell it removes the earlier failures.json
    again = _files.Results(out, columns, {"a": "s"}, "suite")
    assert len(again.done) == 3 and again.finish()[1] == []
    assert not (out / "failures.json").exists()
    # rows of other settings stop a run when it opens
    with pytest.raises(ValueError, match="other settings"):
        _files.Results(out, columns, {"a": "other"}, "suite")
