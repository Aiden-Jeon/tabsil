"""Pinned benchmark downloads and result tables shared by the suites (`benchmarks._files`)."""

import hashlib
import io

import pytest

from tabsil_train.benchmarks import _files, adbench

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
