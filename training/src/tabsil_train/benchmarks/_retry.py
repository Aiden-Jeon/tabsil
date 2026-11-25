"""Retry benchmark data downloads so a network drop does not abort the whole evaluation.

A connection dropping mid-download of OpenML data (`IncompleteRead`) has repeatedly forced
evaluations that had run for tens of minutes to restart from scratch. We wait briefly and retry
only on network errors; other errors, such as a "no such data" response, are raised as is.

openml itself retries each API request a few times (5 by default, 1-4 s apart), but longer outages
still propagate, and parquet downloads (minio) are not retried. So we wait longer, outside it.

Partially downloaded caches are not deleted. openml removes the cache folder itself when
`get_task`/`get_dataset` fails, writes API responses to file only after they are fully received,
and downloads parquet to `*.part.minio` before renaming it (the next attempt resumes from there).
"""

from __future__ import annotations

import http.client
import re
import time
from collections.abc import Callable
from typing import TypeVar

# Number of attempts and first wait (seconds). The wait doubles each time (30, 60, 120 s)
ATTEMPTS = 4
FIRST_WAIT = 30.0

# HTTP status in openml `OpenMLServerError` messages
_STATUS = re.compile(r"Status code: (\d{3})")


def _network_error_types() -> tuple[type[BaseException], ...]:
    types: list[type[BaseException]] = [ConnectionError, TimeoutError, http.client.IncompleteRead]
    try:
        import requests

        types += [
            requests.exceptions.ConnectionError,
            requests.exceptions.ChunkedEncodingError,
            requests.exceptions.Timeout,
        ]
    except ImportError:
        pass
    try:
        import urllib3

        types += [urllib3.exceptions.ProtocolError, urllib3.exceptions.TimeoutError]
    except ImportError:
        pass
    return tuple(types)


def _config_error_types() -> tuple[type[BaseException], ...]:
    # Among requests' `ConnectionError`s, proxy and SSL errors are configuration problems that
    # waiting will not fix
    try:
        import requests
    except ImportError:
        return ()
    return (requests.exceptions.ProxyError, requests.exceptions.SSLError)


def _is_transient_openml_error(error: BaseException) -> bool:
    try:
        import openml.exceptions as oe
    except ImportError:
        return False
    if isinstance(error, oe.OpenMLHashException):  # downloaded file is corrupted
        return True
    # Subclasses (`OpenMLServerException`: API responses like "unknown task") are not retried
    if type(error) is not oe.OpenMLServerError:
        return False
    # Only status 200 (corrupted body), 429 or 5xx. 4xx (missing or forbidden data) and errors
    # without a status ("URI too long") fail the same way on retry
    m = _STATUS.search(str(error))
    return m is not None and (int(m.group(1)) in (200, 429) or int(m.group(1)) >= 500)


def is_network_error(error: BaseException) -> bool:
    """Whether the error is a network drop or a transient server failure.

    Follows the chain of explicit causes (`raise ... from e`). An error raised while handling
    another (`__context__`) is not attributed to it. Among OpenML errors, only
    `OpenMLHashException` (corrupted download) and `OpenMLServerError` with status 200, 429 or
    5xx (unexpected server error) are retried."""
    network, config = _network_error_types(), _config_error_types()
    seen: set[int] = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        if isinstance(error, config):
            return False
        if isinstance(error, network) or _is_transient_openml_error(error):
            return True
        error = error.__cause__
    return False


# a TypeVar rather than `def retry[T]` (PEP 695 needs Python 3.12; the package supports 3.11)
T = TypeVar("T")


def retry(
    fn: Callable[[], T], *, what: str, attempts: int = ATTEMPTS, wait: float = FIRST_WAIT
) -> T:
    """Call `fn()`; on a network error (`is_network_error`), wait and retry, up to `attempts`
    calls in total. Other errors, or a failure on the last attempt, are raised as is."""
    if attempts < 1:
        raise ValueError(f"attempts must be at least 1: {attempts}")
    for attempt in range(1, attempts):
        try:
            return fn()
        except Exception as e:
            if not is_network_error(e):
                raise
            print(
                f"{what}: network error, retrying in {wait:.0f} s ({attempt}/{attempts - 1}):"
                f" {type(e).__name__}: {e}",
                flush=True,
            )
            time.sleep(wait)
            wait *= 2
    return fn()


def fetch_openml_task(task_id: int):
    """Download an OpenML task with its splits and dataset (description, features, data; kept in
    the cache), retrying on network errors. Returns
    `(task, dataset, result of dataset.get_data(target))`."""
    import openml

    def fetch():
        task = openml.tasks.get_task(task_id, download_splits=True)
        dataset = task.get_dataset(download_data=True)
        return task, dataset, dataset.get_data(target=task.target_name, dataset_format="dataframe")

    return retry(fetch, what=f"OpenML task {task_id}")
