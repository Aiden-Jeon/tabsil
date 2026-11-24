"""CPU cell state of interleaved chunked inference (`column_row_transformer._host_state`):
instead of the caching host allocator (`pin_memory=True`), allocate an exact-size pageable tensor,
register just that memory as pinned, and unregister it when forward ends (even on an exception).
Values are bit-identical to keeping the state on the device."""

import warnings

import pytest
import torch

from tabsil.models import column_row_transformer as M
from tabsil.models.column_row_transformer import ColumnRowTransformer, ColumnRowTransformerConfig

# two interleaved column/row stage pairs at tiny size
TINY_STAGES = {"column_blocks": 2, "row_blocks": 2}
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _model(**changes) -> ColumnRowTransformer:
    torch.manual_seed(0)
    config = ColumnRowTransformerConfig.tiny(**changes)
    model = ColumnRowTransformer(config).eval()
    model.inference_chunk_cells = 37  # several row chunks, several column chunks
    model.inference_column_chunk = 2
    return model


def _inputs(device="cpu"):
    g = torch.Generator().manual_seed(0)
    x = torch.randn(1, 40, 5, generator=g)
    x[0, 3, 1] = float("nan")
    yc = torch.randint(0, 3, (1, 30), generator=g).float()
    yr = torch.randn(1, 30, generator=g)
    return x.to(device), yc.to(device), yr.to(device)


def _forward(model, x, y, task):
    with torch.inference_mode():
        return model(x, y, task, inference=True)


def test_offloaded_state_matches_device_state(monkeypatch):
    # On CPU too, the offload path (`_host_state`, pageable without registration when there is no
    # CUDA) gives the same values
    model = _model(**TINY_STAGES)
    x, yc, yr = _inputs()
    want = {t: _forward(model, x, y, t) for t, y in (("classification", yc), ("regression", yr))}
    calls = []
    real = M._host_state

    def host_state(shape, dtype, pin):
        calls.append((shape, dtype, pin))
        return real(shape, dtype, pin)

    monkeypatch.setattr(M, "_host_state", host_state)
    monkeypatch.setattr(ColumnRowTransformer, "_offload_interleaved_state", lambda *_: True)
    for task, y in (("classification", yc), ("regression", yr)):
        assert torch.equal(_forward(model, x, y, task), want[task])
    assert len(calls) == 2 and calls[0][0] == (1, 40, 5, model.config.emsize)


def test_registration_is_released_after_success_and_error(monkeypatch):
    model = _model(**TINY_STAGES)
    x, yc, _ = _inputs()
    events = []

    def host_state(shape, dtype, pin):
        state = torch.empty(shape, dtype=dtype)
        events.append(("register", state.data_ptr()))
        return state, True

    def release(state, registered):
        events.append(("release", state.data_ptr(), registered))

    monkeypatch.setattr(M, "_host_state", host_state)
    monkeypatch.setattr(M, "_release_host_state", release)
    monkeypatch.setattr(ColumnRowTransformer, "_offload_interleaved_state", lambda *_: True)
    _forward(model, x, yc, "classification")
    assert events == [("register", events[0][1]), ("release", events[0][1], True)]

    # Unregisters even when a row block raises (if left registered, the OS cannot reclaim it)
    events.clear()

    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(model.row_blocks[0], "forward", boom)
    with pytest.raises(RuntimeError, match="boom"):
        _forward(model, x, yc, "classification")
    assert events == [("register", events[0][1]), ("release", events[0][1], True)]


def _counting(monkeypatch):
    # Count `_host_state`/`_release_host_state` calls (the real functions still run)
    events = []
    real_state, real_release = M._host_state, M._release_host_state

    def host_state(shape, dtype, pin):
        state, registered = real_state(shape, dtype, pin)
        events.append(("state", tuple(shape), state.data_ptr()))
        return state, registered

    def release(state, registered):
        events.append(("release", state.data_ptr()))
        real_release(state, registered)

    monkeypatch.setattr(M, "_host_state", host_state)
    monkeypatch.setattr(M, "_release_host_state", release)
    monkeypatch.setattr(ColumnRowTransformer, "_offload_interleaved_state", lambda *_: True)
    return events


def test_hold_host_state_reuses_one_buffer_across_forwards(monkeypatch):
    model = _model(**TINY_STAGES)
    x, yc, yr = _inputs()
    want = [_forward(model, x, yc, "classification"), _forward(model, x, yr, "regression")]
    small = _forward(model, x[:, :, :3], yc, "classification")
    events = _counting(monkeypatch)
    with model.hold_host_state():
        got = [_forward(model, x, yc, "classification"), _forward(model, x, yr, "regression")]
        # A smaller state reuses the front of the same buffer (no stale values left)
        got_small = _forward(model, x[:, :, :3], yc, "classification")
        assert [e[0] for e in events] == ["state"]  # allocated once and not released
    assert all(torch.equal(g, w) for g, w in zip(got, want, strict=True))
    assert torch.equal(got_small, small)
    ptr = events[0][2]
    assert events == [("state", (40 * 5 * model.config.emsize,), ptr), ("release", ptr)]
    assert model._held_host_state is None


def test_hold_host_state_grows_and_releases_on_error(monkeypatch):
    model = _model(**TINY_STAGES)
    x, yc, _ = _inputs()
    events = _counting(monkeypatch)
    with pytest.raises(RuntimeError, match="boom"), model.hold_host_state():
        _forward(model, x[:, :, :3], yc, "classification")
        _forward(model, x, yc, "classification")  # larger state: free the old buffer, allocate anew
        raise RuntimeError("boom")
    kinds = [e[0] for e in events]
    assert kinds == ["state", "release", "state", "release"]
    assert events[1][1] == events[0][2] and events[3][1] == events[2][2]
    assert model._held_host_state is None and not model._holding_host_state


def test_ensemble_holds_one_state_per_predict(monkeypatch):
    import numpy as np

    from tabsil.inference.ensemble import EnsembleClassifier

    model = _model(**TINY_STAGES)
    model.inference_chunk_cells = 37
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(40, 4)), rng.integers(0, 3, size=40)
    clf = EnsembleClassifier(model, n_estimators=3, member_kinds=("standard",)).fit(X, y)
    want = clf.predict_proba(X[:9])
    events = _counting(monkeypatch)
    got = clf.predict_proba(X[:9])
    assert np.array_equal(got, want)
    # Three members share one buffer, released when predict ends
    assert [e[0] for e in events] == ["state", "release"]


def test_state_stays_on_device_without_offload(monkeypatch):
    # A state that fits on the device does not call `_host_state` (always so on the CPU device)
    model = _model(**TINY_STAGES)
    x, yc, _ = _inputs()
    monkeypatch.setattr(M, "_host_state", lambda *a: pytest.fail("should not offload"))
    _forward(model, x, yc, "classification")


def test_failed_registration_falls_back_to_pageable(monkeypatch):
    cleared = []

    class FakeCudart:
        def cudaHostRegister(self, ptr, size, flags):
            return 2  # cudaErrorMemoryAllocation

        def cudaHostUnregister(self, ptr):
            raise AssertionError("unregistered memory must not be unregistered")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "cudart", lambda: FakeCudart())
    monkeypatch.setattr(M, "_clear_cuda_error", lambda: cleared.append(True))
    with pytest.warns(RuntimeWarning, match="could not register"):
        state, registered = M._host_state((2, 3), torch.bfloat16, True)
    assert not registered and state.shape == (2, 3) and state.device.type == "cpu"
    assert not state.is_pinned() and cleared == [True]
    M._release_host_state(state, registered)  # no-op when nothing was registered


def test_pin_off_does_not_register(monkeypatch):
    monkeypatch.setattr(torch.cuda, "cudart", lambda: pytest.fail("should not register"))
    state, registered = M._host_state((4, 5), torch.float32, False)
    assert not registered and state.shape == (4, 5)


def test_inference_host_state_bytes():
    model = _model(**TINY_STAGES)
    e = model.config.emsize
    model.interleaved_state_device_bytes = 1000
    model.inference_chunk_cells = 100
    assert model.inference_host_state_bytes(50, 3) == 50 * 3 * e * 4  # cells > 100, bytes > 1000
    assert model.inference_host_state_bytes(20, 3) == 0  # 60 cells: not chunked
    model.interleaved_state_device_bytes = 50 * 3 * e * 4
    assert model.inference_host_state_bytes(50, 3) == 0  # stays on the device
    # Default: BeyondArena maps_router_eta_1m (1,250,000 train + test rows, 768 + 1 id columns)
    default = _model(**TINY_STAGES)
    default.inference_chunk_cells = ColumnRowTransformer.inference_chunk_cells
    assert default.inference_host_state_bytes(1_250_000, 769) == 1_250_000 * 769 * e * 4


# --- GPU -----------------------------------------------------------------------------------------


@cuda
@pytest.mark.parametrize("pin", [True, False])
def test_gpu_offload_is_bit_identical(pin):
    model = _model(**TINY_STAGES).cuda()
    x, yc, yr = _inputs("cuda")
    for task, y in (("classification", yc), ("regression", yr)):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            want = _forward(model, x, y, task)  # state on the device
            model.interleaved_state_device_bytes = 0  # state on the CPU
            model.interleaved_state_pin = pin
            before = torch.cuda.host_memory_stats().get("allocated_bytes.all.peak", 0)
            got = _forward(model, x, y, task)
            after = torch.cuda.host_memory_stats().get("allocated_bytes.all.peak", 0)
            model.interleaved_state_device_bytes = (
                ColumnRowTransformer.interleaved_state_device_bytes
            )
            model.interleaved_state_pin = ColumnRowTransformer.interleaved_state_pin
        assert torch.equal(got, want)
        assert after == before  # the caching host allocator is not used


@cuda
def test_gpu_state_size_matches_the_estimate(monkeypatch):
    # Cell state is float32 even under bfloat16 autocast: the evaluation queue's estimate
    # (`inference_host_state_bytes`) must match the actual allocation
    model = _model(**TINY_STAGES).cuda()
    model.interleaved_state_device_bytes = 0
    x, yc, _ = _inputs("cuda")
    sizes = []
    real = M._host_state

    def host_state(shape, dtype, pin):
        state, registered = real(shape, dtype, pin)
        sizes.append((state.dtype, state.nbytes))
        return state, registered

    monkeypatch.setattr(M, "_host_state", host_state)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        _forward(model, x, yc, "classification")
    assert sizes == [(torch.float32, model.inference_host_state_bytes(*x.shape[1:]))]


@cuda
def test_gpu_register_roundtrip_and_error_clearing():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        state, registered = M._host_state((1024, 1024), torch.bfloat16, True)
    assert registered and state.is_pinned()
    # Re-registering the same memory fails: once the leftover error is cleared, the next kernel runs
    err = int(torch.cuda.cudart().cudaHostRegister(state.data_ptr(), state.nbytes, 0))
    assert err != 0
    M._clear_cuda_error()
    torch.ones(8, device="cuda").sum().item()
    M._release_host_state(state, registered)
    assert not state.is_pinned()
