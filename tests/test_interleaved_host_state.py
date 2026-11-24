"""CPU cell state of interleaved chunked inference (`column_row_transformer._host_state`):
instead of the caching host allocator (`pin_memory=True`), allocate an exact-size pageable tensor,
register just that memory as pinned, and unregister it when forward ends (even on an exception).

The offload path (`_rows_interleaved_offload`) accumulates the column summaries over train-row
chunks (online softmax) and computes the test-row chunks to the end on the device. Its values equal
those with the state on the device within numerical error (float32, relative 1e-5). The freshly
initialized model has attention output layers at 0, so the summaries do not reach its values; the
tests that compare values use a model with perturbed weights (`_model(random=True)`)."""

import warnings

import pytest
import torch

from tabsil.models import column_row_transformer as M
from tabsil.models.column_row_transformer import ColumnRowTransformer, ColumnRowTransformerConfig

# two interleaved column/row stage pairs at tiny size
TINY_STAGES = {"column_blocks": 2, "row_blocks": 2}
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _model(random: bool = False, **changes) -> ColumnRowTransformer:
    torch.manual_seed(0)
    config = ColumnRowTransformerConfig.tiny(**changes)
    model = ColumnRowTransformer(config).eval()
    model.inference_chunk_cells = 37  # several row chunks, several column chunks
    model.inference_column_chunk = 2
    if random:
        _shake(model)
    return model


def _shake(module: torch.nn.Module, scale: float = 0.3) -> None:
    # Noise on every weight, including the output layers that start at 0 (so that the summary and
    # spread attentions reach the values)
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for p in module.parameters():
            p.add_(torch.randn(p.shape, generator=g).to(p) * scale)


def _rel(got: torch.Tensor, want: torch.Tensor) -> float:
    return float((got.float() - want.float()).abs().max() / want.float().abs().max())


def _held_rows(model, x, n_train) -> int:
    # Rows of the state the offload carries: up to the last row chunk with train rows
    step = model.inference_chunk_cells // (x.shape[0] * x.shape[-1])
    return min(x.shape[1], -(-n_train // step) * step)


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


@pytest.mark.parametrize("random", [False, True])
def test_offloaded_state_matches_device_state(monkeypatch, random):
    # On CPU too, the offload path (`_host_state`, pageable without registration when there is no
    # CUDA) gives the same values within numerical error
    model = _model(random, **TINY_STAGES)
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
        assert _rel(_forward(model, x, y, task), want[task]) < 1e-5
    # Only the chunks with train rows (5 chunks of 7 rows = 35 rows) keep a state
    held = _held_rows(model, x, yc.shape[1])
    assert held == 35
    assert len(calls) == 2 and calls[0][0] == (1, held, 5, model.config.emsize)


def test_offload_matches_device_state_in_float64(monkeypatch):
    # The two paths differ only in the order of the summary computation: in float64 the row
    # vectors (`_rows_interleaved_chunked`) agree to rounding. 3 block pairs, 2 tables
    model = _model(True, column_blocks=3, row_blocks=3).double()
    g = torch.Generator().manual_seed(0)
    x = torch.randn(2, 61, 7, generator=g, dtype=torch.float64)
    x[0, 3, 1] = float("nan")
    y = torch.randint(0, 3, (2, 45), generator=g).double()

    def rows():
        with torch.inference_mode():
            return model._rows_interleaved_chunked(x, y, "classification")

    want = rows()
    monkeypatch.setattr(ColumnRowTransformer, "_offload_interleaved_state", lambda *_: True)
    got = rows()
    assert got.dtype == torch.float64 and _rel(got, want) < 1e-12


def test_held_state_stays_on_device_when_it_fits(monkeypatch):
    # The whole state does not fit on the device, but that of the train-row chunks does: it is not
    # kept on the CPU
    model = _model(True, **TINY_STAGES)
    x, yc, _ = _inputs()
    want = _forward(model, x, yc, "classification")
    e = model.config.emsize
    model.interleaved_state_device_bytes = 35 * 5 * e * 4  # train-row chunks 35 rows <= this < 40
    monkeypatch.setattr(M, "_host_state", lambda *a: pytest.fail("should not offload"))
    monkeypatch.setattr(ColumnRowTransformer, "_offload_interleaved_state", _offload_any_device)
    offload = []
    real = ColumnRowTransformer._rows_interleaved_offload

    def spy(self, *args):
        offload.append(True)
        return real(self, *args)

    monkeypatch.setattr(ColumnRowTransformer, "_rows_interleaved_offload", spy)
    assert _rel(_forward(model, x, yc, "classification"), want) < 1e-5
    assert offload == [True]


def _offload_any_device(self, device, size):
    # Decide by size alone, on the CPU device too (for real only CUDA offloads)
    return size > self.interleaved_state_device_bytes


@pytest.mark.parametrize("temperature", ["logscale", "gated_logscale", "none"])
def test_split_attention_matches_forward(temperature):
    # The computation split into queries, keys/values and output (`split_*`, `_attention_lse`)
    # equals forward
    torch.manual_seed(0)
    att = M.Attention(16, 2, temperature=temperature, gate_hidden=8).eval()
    _shake(att)
    att.diag_scale = torch.nn.Parameter(torch.randn(2, 8) * 0.1)
    x, ctx = torch.randn(3, 4, 16), torch.randn(3, 9, 16)
    with torch.no_grad():
        want = att(x, context=ctx)
        q = att.split_queries(x, 9)
        k, v = att.split_keys(ctx)
        out, lse = M._attention_lse(q, k, v)
        got = att.split_output(out)
    assert _rel(got, want) < 1e-6
    assert lse.shape == (3, 2, 4) and lse.dtype == torch.float32


def test_split_queries_rejects_rope():
    att = M.Attention(16, 2, rope_base=100.0)
    with pytest.raises(ValueError, match="RoPE"):
        att.split_queries(torch.randn(1, 3, 16), 3)


@pytest.mark.parametrize("chunk", [1, 5, 23])
def test_online_summary_matches_summary(chunk):
    # The summary accumulated over train-row chunks equals `InducedBlock.summary` computed at once
    torch.manual_seed(0)
    block = M.InducedBlock(16, 2, 4, 2, 8).eval()
    _shake(block)
    h = torch.randn(3, 23, 16) * 2
    with torch.no_grad():
        want = block.summary(h)
        acc = M._OnlineSummary(block, 3, 23, h.dtype)
        for r0 in range(0, 23, chunk):
            acc.add(h[:, r0 : r0 + chunk])
        got = acc.finish()
    assert _rel(got, want) < 1e-5


def test_online_summary_needs_rows():
    block = M.InducedBlock(16, 2, 4, 2, 8)
    with pytest.raises(ValueError, match="no train rows"):
        M._OnlineSummary(block, 1, 5, torch.float32).finish()


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
    held = _held_rows(model, x, yc.shape[1])
    assert events == [("state", (held * 5 * model.config.emsize,), ptr), ("release", ptr)]
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
def test_gpu_offload_matches_device_state(monkeypatch, pin):
    # A small `interleaved_state_device_bytes` keeps the state on the CPU. float32: within relative
    # 1e-5 of the state on the device. bfloat16 autocast: the offload is not further from the
    # float32 values than the bfloat16 computation with the state on the device (the per-chunk
    # summaries are not rounded to bfloat16)
    model = _model(True, **TINY_STAGES).cuda()
    x, yc, yr = _inputs("cuda")

    def offloaded(y, task):
        with monkeypatch.context() as m:
            m.setattr(model, "interleaved_state_device_bytes", 0)
            m.setattr(model, "interleaved_state_pin", pin)
            stats = torch.cuda.host_memory_stats
            before = stats().get("allocated_bytes.all.peak", 0)
            got = _forward(model, x, y, task)
            assert stats().get("allocated_bytes.all.peak", 0) == before  # no caching allocator
        return got

    for task, y in (("classification", yc), ("regression", yr)):
        want = _forward(model, x, y, task)  # float32, state on the device
        assert _rel(offloaded(y, task), want) < 1e-5, task
        with torch.autocast("cuda", dtype=torch.bfloat16):
            device_bf16 = _forward(model, x, y, task)
            got_bf16 = offloaded(y, task)
        assert _rel(got_bf16, want) <= 1.5 * _rel(device_bf16, want) + 1e-4, task


@cuda
@pytest.mark.parametrize("pin", [True, False])
def test_gpu_offload_pipeline_many_chunks(monkeypatch, pin):
    # A small model at the published width (cells 256, 4 heads), 30 row chunks (20 with train
    # rows): with the reads and writes overlapping the compute (`_HostRows`) the values equal those
    # with the state on the device, and the same forward twice gives the same values
    torch.manual_seed(0)
    config = ColumnRowTransformerConfig(
        nlayers=2, nhead=4, decoder_heads=2, n_out=3, column_blocks=3, row_blocks=3, n_inducing=32
    )
    model = ColumnRowTransformer(config).eval()
    _shake(model, 0.05)
    model = model.cuda()
    model.inference_chunk_cells = 100 * 24
    g = torch.Generator().manual_seed(0)
    x = torch.randn(1, 3000, 24, generator=g).cuda()
    yr = torch.randn(1, 2000, generator=g).cuda()
    want = _forward(model, x, yr, "regression")
    monkeypatch.setattr(model, "interleaved_state_device_bytes", 0)
    monkeypatch.setattr(model, "interleaved_state_pin", pin)
    writes = []
    real = M._HostRows.write

    def write(self, r0, r1, src):
        writes.append(r0)
        return real(self, r0, r1, src)

    monkeypatch.setattr(M._HostRows, "write", write)
    got = _forward(model, x, yr, "regression")
    again = _forward(model, x, yr, "regression")
    assert _rel(got, want) < 1e-5
    assert torch.equal(got, again)
    # Block pairs 0 and 1 (the last pair writes nothing) x 20 train-row chunks, two forwards
    assert len(writes) == 2 * 2 * 20 and max(writes) == 1900


@cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_gpu_attention_lse_kernel_matches_math(dtype):
    # The GPU memory-efficient kernel path equals the path that builds the logits (the CPU one)
    g = torch.Generator().manual_seed(0)
    q, k, v = (torch.randn(5, 4, n, 64, generator=g).cuda().to(dtype) for n in (40, 77, 77))
    out, lse = M._attention_lse(q, k, v)
    s = (q.float() @ k.float().transpose(-1, -2)) / 8
    want_lse = s.logsumexp(-1)
    want = torch.softmax(s, -1) @ v.float()
    tol = 1e-5 if dtype == torch.float32 else 1e-2
    assert out.dtype == dtype and lse.dtype == torch.float32
    assert _rel(out, want) < tol and _rel(lse, want_lse) < 1e-5


@cuda
def test_gpu_state_size_matches_the_estimate(monkeypatch):
    # Cell state is float32 even under bfloat16 autocast. The carried state (the row chunks with
    # train rows) is not larger than the evaluation queue's estimate (`inference_host_state_bytes`,
    # all rows)
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
    held = _held_rows(model, x, yc.shape[1]) * x.shape[-1] * model.config.emsize * 4
    assert sizes == [(torch.float32, held)]
    assert held <= model.inference_host_state_bytes(*x.shape[1:])


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
