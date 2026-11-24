"""Network retries for benchmark data downloads (`tabsil_train.benchmarks._retry`)."""

import openml.exceptions as oe
import pandas as pd
import pytest
import requests
import urllib3

from tabsil_train.benchmarks import _retry
from tabsil_train.benchmarks._retry import is_network_error, retry


@pytest.fixture
def waits(monkeypatch):
    out = []
    monkeypatch.setattr(_retry.time, "sleep", out.append)
    return out


def _flaky(errors, value="ok"):
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        if errors:
            raise errors.pop(0)
        return value

    return fn, calls


def _raise(error, *, cause=None, context=None, suppress=False):
    # `raise error from cause`, or `error` raised while handling `context`
    try:
        if context is not None:
            try:
                raise context
            except type(context):
                if suppress:
                    raise error from None
                raise error  # noqa: B904
        raise error from cause
    except type(error) as e:
        return e


def test_network_errors_are_recognised_through_the_cause_chain():
    assert is_network_error(requests.exceptions.ChunkedEncodingError("IncompleteRead"))
    protocol = urllib3.exceptions.ProtocolError("Connection broken")
    assert is_network_error(_raise(RuntimeError("wrapped"), cause=protocol))
    # A different error raised while handling a network error is not caused by it
    assert not is_network_error(_raise(ValueError("bug"), context=TimeoutError()))
    assert not is_network_error(_raise(KeyError("x"), context=ConnectionError(), suppress=True))
    assert not is_network_error(ValueError("bad data"))
    # Proxy/SSL configuration errors do not go away by waiting
    assert not is_network_error(requests.exceptions.ProxyError("proxy"))
    assert not is_network_error(requests.exceptions.SSLError("cert"))


def test_openml_errors_are_retried_only_when_transient():
    def server_error(status):
        return oe.OpenMLServerError(
            f"Unexpected server error when calling https://x. Please contact the developers!\n"
            f"Status code: {status}\n<html>...</html>"
        )

    assert is_network_error(server_error(503))
    assert is_network_error(server_error(429))
    assert is_network_error(server_error(200))  # corrupted body
    assert not is_network_error(server_error(404))
    assert not is_network_error(server_error(403))
    assert not is_network_error(oe.OpenMLServerError("URI too long! (https://x)"))
    assert is_network_error(oe.OpenMLHashException("checksum"))
    # API responses such as a missing task (subclass) are not retried
    assert not is_network_error(oe.OpenMLServerException("Unknown task", code=151))

    class OpenMLServerError(Exception):  # a different class with the same name
        pass

    assert not is_network_error(OpenMLServerError("Status code: 503"))


def test_retry_waits_and_tries_again(waits):
    errors = [requests.exceptions.ChunkedEncodingError("a"), ConnectionError("b")]
    fn, calls = _flaky(errors)
    assert retry(fn, what="t") == "ok"
    assert calls["n"] == 3 and waits == [30.0, 60.0]


def test_retry_raises_other_errors_at_once(waits):
    fn, calls = _flaky([ValueError("bad")])
    with pytest.raises(ValueError):
        retry(fn, what="t")
    assert calls["n"] == 1 and waits == []


def test_retry_gives_up_after_the_last_attempt(waits):
    fn, calls = _flaky([TimeoutError(str(i)) for i in range(5)])
    with pytest.raises(TimeoutError):
        retry(fn, what="t", attempts=3)
    assert calls["n"] == 3 and waits == [30.0, 60.0]


def test_retry_attempts(waits):
    fn, calls = _flaky([TimeoutError()])
    with pytest.raises(TimeoutError):  # called only once
        retry(fn, what="t", attempts=1)
    assert calls["n"] == 1 and waits == []
    with pytest.raises(ValueError, match="attempts"):
        retry(fn, what="t", attempts=0)
    assert calls["n"] == 1


def test_load_openml_task_retries(monkeypatch, waits):
    import openml

    from tabsil_train.benchmarks import amlb

    class Dataset:
        name = "toy"

        def get_data(self, target, dataset_format):
            return pd.DataFrame({"a": [1.0, 2.0]}), pd.Series([0, 1]), [False], ["a"]

    class Task:
        dataset_id = 7
        target_name = "y"
        task_type_id = openml.tasks.TaskType.SUPERVISED_CLASSIFICATION

        def get_dataset(self, download_data):
            if not attempts:  # the first attempt is cut off mid-download
                attempts.append(1)
                raise requests.exceptions.ChunkedEncodingError("IncompleteRead")
            return Dataset()

    attempts = []
    monkeypatch.setattr(openml.tasks, "get_task", lambda task_id, download_splits: Task())
    task = amlb.load_openml_task(42)
    assert task.name == "toy" and task.task_type == "classification"
    assert waits == [30.0]


def test_arena_prefetch_tasks_retries_each_task(monkeypatch, waits):
    from types import SimpleNamespace

    from tabsil_train.benchmarks import arena

    fetched = []
    materialized = []

    class Collection:
        # OpenML tasks 41 (two splits) and 42, and a Data Foundry task (non-integer id)
        tasks = [SimpleNamespace(task_id_str=s) for s in ("41", "41", "42", "UserTask|abc")]

        def __iter__(self):
            return iter(self.tasks)

        def materialize(self):
            materialized.append(1)
            if len(materialized) == 1:
                raise requests.exceptions.ChunkedEncodingError("IncompleteRead")
            return self

    jobs = ["job"]

    class TaskMetadata:
        def subset_to_jobs(self, js):
            assert js is jobs
            return Collection()

    context = SimpleNamespace(task_metadata_collection=TaskMetadata())
    monkeypatch.setattr(_retry, "fetch_openml_task", fetched.append)
    arena.prefetch_tasks(context, jobs, what="suite")
    assert fetched == [41, 42] and len(materialized) == 2 and waits == [30.0]
