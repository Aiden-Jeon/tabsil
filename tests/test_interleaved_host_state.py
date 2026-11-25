"""CPU cell state of interleaved chunked inference (`column_row_transformer._host_state`):
instead of the caching host allocator (`pin_memory=True`), allocate an exact-size pageable tensor,
register just that memory as pinned, and unregister it when forward ends (even on an exception).

The offload path (`_rows_interleaved_offload`) accumulates the column summaries over train-row
chunks (online softmax) and computes the test-row chunks to the end on the device. Its values equal
those with the state on the device within float32 rounding: the row vectors before the decoder
within relative 1e-5 (`ROWS_TOL`), the outputs within 1e-4 (`OUTPUT_TOL`: the decoder's log p
magnifies the rounding), and float64 within 1e-12. The freshly
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
# Offload vs state on the device, float32, relative to the largest value: the row vectors before
# the decoder (`_rows_interleaved_chunked`), where the summary order differs, and the model output.
# The classification output is log p, which magnifies float32 rounding near p = 1 (on Linux CPU and
# H100 the outputs differ by 1.1e-5 to 1.8e-5 while the row vectors agree much more closely)
ROWS_TOL = 1e-5
OUTPUT_TOL = 1e-4


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


def _rows(model, x, y, task):
    # The row vectors of the chunked inference, before the row ICL and the decoder
    with torch.inference_mode():
        return model._rows_interleaved_chunked(x, y, task)


@pytest.mark.parametrize("random", [False, True])
def test_offloaded_state_matches_device_state(monkeypatch, random):
    # On CPU too, the offload path (`_host_state`, pageable without registration when there is no
    # CUDA) gives the same values within numerical error
    model = _model(random, **TINY_STAGES)
    x, yc, yr = _inputs()
    tasks = (("classification", yc), ("regression", yr))
    want = {t: _forward(model, x, y, t) for t, y in tasks}
    want_rows = {t: _rows(model, x, y, t) for t, y in tasks}
    calls = []
    real = M._host_state

    def host_state(shape, dtype, pin):
        calls.append((shape, dtype, pin))
        return real(shape, dtype, pin)

    monkeypatch.setattr(M, "_host_state", host_state)
    monkeypatch.setattr(ColumnRowTransformer, "_offload_interleaved_state", lambda *_: True)
    for task, y in tasks:
        assert _rel(_forward(model, x, y, task), want[task]) < OUTPUT_TOL
    # Only the chunks with train rows (5 chunks of 7 rows = 35 rows) keep a state
    held = _held_rows(model, x, yc.shape[1])
    assert held == 35
    assert len(calls) == 2 and calls[0][0] == (1, held, 5, model.config.emsize)
    for task, y in tasks:
        assert _rel(_rows(model, x, y, task), want_rows[task]) < ROWS_TOL, task


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


def test_offload_without_train_rows_matches_device_state(monkeypatch):
    # No train rows: nothing is kept and the summaries attend to no keys, as on the device
    model = _model(True, **TINY_STAGES)
    x, _, _ = _inputs()
    y = torch.zeros(1, 0)
    tasks = ("classification", "regression")
    want = {t: _forward(model, x, y, t) for t in tasks}
    want_rows = {t: _rows(model, x, y, t) for t in tasks}
    monkeypatch.setattr(ColumnRowTransformer, "_offload_interleaved_state", lambda *_: True)
    for task in tasks:
        got = _forward(model, x, y, task)
        assert torch.isfinite(got).all() and _rel(got, want[task]) < OUTPUT_TOL, task
        assert _rel(_rows(model, x, y, task), want_rows[task]) < ROWS_TOL, task


def test_offload_with_one_block_pair_keeps_no_state(monkeypatch):
    # One block pair: block 0 reads the cell encoder output and the last block writes no state, so
    # nothing is allocated on the CPU; the values still match the device
    model = _model(True)  # the tiny config has one column/row stage pair
    x, yc, yr = _inputs()
    tasks = (("classification", yc), ("regression", yr))
    want = {t: _forward(model, x, y, t) for t, y in tasks}
    want_rows = {t: _rows(model, x, y, t) for t, y in tasks}
    monkeypatch.setattr(M, "_host_state", lambda *a: pytest.fail("should not keep a state"))
    monkeypatch.setattr(ColumnRowTransformer, "_offload_interleaved_state", lambda *_: True)
    for task, y in tasks:
        assert _rel(_forward(model, x, y, task), want[task]) < OUTPUT_TOL, task
        assert _rel(_rows(model, x, y, task), want_rows[task]) < ROWS_TOL, task


def test_held_state_stays_on_device_when_it_fits(monkeypatch):
    # The whole state does not fit on the device, but that of the train-row chunks does: it is not
    # kept on the CPU
    model = _model(True, **TINY_STAGES)
    x, yc, _ = _inputs()
    want = _forward(model, x, yc, "classification")
    want_rows = _rows(model, x, yc, "classification")
    e = model.config.emsize
    model.interleaved_state_device_bytes = 35 * 5 * e * 4  # train-row chunks 35 rows <= this < 40
    monkeypatch.setattr(M, "_host_state", lambda *a: pytest.fail("should not offload"))
    monkeypatch.setattr(ColumnRowTransformer, "_offload_interleaved_state", _offload_any_device)
    offload = []
    real = ColumnRowTransformer._rows_interleaved_offload

    def spy(self, *args, **kwargs):
        offload.append(True)
        return real(self, *args, **kwargs)

    monkeypatch.setattr(ColumnRowTransformer, "_rows_interleaved_offload", spy)
    assert _rel(_forward(model, x, yc, "classification"), want) < OUTPUT_TOL
    assert _rel(_rows(model, x, yc, "classification"), want_rows) < ROWS_TOL
    assert offload == [True, True]


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


def test_online_summary_without_rows_matches_summary():
    # No chunk added: the same as `InducedBlock.summary` over zero train rows
    torch.manual_seed(0)
    block = M.InducedBlock(16, 2, 4, 2, 8).eval()
    _shake(block)
    with torch.no_grad():
        want = block.summary(torch.zeros(3, 0, 16))
        got = M._OnlineSummary(block, 3, 0, torch.float32).finish()
    assert torch.isfinite(got).all() and _rel(got, want) < 1e-6


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


def test_ensemble_holds_one_state_per_pass_over_the_training_rows(monkeypatch):
    import numpy as np

    from tabsil.inference.ensemble import EnsembleClassifier

    model = _model(**TINY_STAGES)
    model.inference_chunk_cells = 37
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(40, 4)), rng.integers(0, 3, size=40)
    kw = {"n_estimators": 3, "member_kinds": ("standard",)}
    clf = EnsembleClassifier(model, train_context=False, **kw).fit(X, y)
    want = clf.predict_proba(X[:9])
    events = _counting(monkeypatch)
    got = clf.predict_proba(X[:9])
    assert np.array_equal(got, want)
    # Without train contexts every predict runs the training rows: three members share one
    # buffer, released when predict ends
    assert [e[0] for e in events] == ["state", "release"]
    # With train contexts the training rows run once, at fit (one buffer for the three members),
    # and predict carries no state
    events.clear()
    ctx = EnsembleClassifier(model, **kw).fit(X, y)
    assert [e[0] for e in events] == ["state", "release"]
    events.clear()
    np.testing.assert_allclose(ctx.predict_proba(X[:9]), want, atol=1e-5, rtol=0)
    assert events == []


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
    # With the train rows: only the row chunks with train rows (2045 rows each at 769 columns)
    step = ColumnRowTransformer.inference_chunk_cells // 769
    held = -(-1_000_000 // step) * step
    got = default.inference_host_state_bytes(1_250_000, 769, train_rows=1_000_000)
    assert got == held * 769 * e * 4 < 1_250_000 * 769 * e * 4
    # One block pair carries nothing
    one = _model()
    one.inference_chunk_cells = ColumnRowTransformer.inference_chunk_cells
    one.interleaved_state_device_bytes = 0
    assert one.inference_host_state_bytes(1_250_000, 769, train_rows=1_000_000) == 0


@pytest.mark.parametrize(
    ("changes", "n_train", "budget_rows", "kept_rows"),
    [
        (TINY_STAGES, 30, 0, 35),  # 5 train-row chunks of 7 rows on the CPU
        (TINY_STAGES, 3, 0, 7),  # one chunk
        (TINY_STAGES, 30, 35, 0),  # the train-row chunks fit on the device (the whole state not)
        ({}, 30, 0, 0),  # one block pair: nothing carried
    ],
)
def test_host_state_estimate_matches_the_allocation(
    monkeypatch, changes, n_train, budget_rows, kept_rows
):
    # `inference_host_state_bytes(..., train_rows=)` is the CPU state the forward allocates
    model = _model(**changes)
    x, _, _ = _inputs()
    y = torch.zeros(1, n_train)
    e = model.config.emsize
    model.interleaved_state_device_bytes = budget_rows * 5 * e * 4
    sizes = []
    real = M._host_state

    def host_state(shape, dtype, pin):
        state, registered = real(shape, dtype, pin)
        sizes.append(state.nbytes)
        return state, registered

    monkeypatch.setattr(M, "_host_state", host_state)
    monkeypatch.setattr(ColumnRowTransformer, "_offload_interleaved_state", _offload_any_device)
    _forward(model, x, y, "classification")
    want = model.inference_host_state_bytes(*x.shape[1:], train_rows=n_train)
    assert want == kept_rows * 5 * e * 4
    assert sizes == ([want] if want else [])


# --- GPU -----------------------------------------------------------------------------------------


@cuda
@pytest.mark.parametrize("pin", [True, False])
def test_gpu_offload_matches_device_state(monkeypatch, pin):
    # A small `interleaved_state_device_bytes` keeps the state on the CPU. float32: row vectors
    # within `ROWS_TOL` and outputs within `OUTPUT_TOL` of the state on the device. bfloat16
    # autocast: the offload is not further from the
    # float32 values than the bfloat16 computation with the state on the device (the per-chunk
    # summaries are not rounded to bfloat16)
    model = _model(True, **TINY_STAGES).cuda()
    x, yc, yr = _inputs("cuda")

    def offloaded(y, task, run=_forward):
        with monkeypatch.context() as m:
            m.setattr(model, "interleaved_state_device_bytes", 0)
            m.setattr(model, "interleaved_state_pin", pin)
            stats = torch.cuda.host_memory_stats
            before = stats().get("allocated_bytes.all.peak", 0)
            got = run(model, x, y, task)
            assert stats().get("allocated_bytes.all.peak", 0) == before  # no caching allocator
        return got

    for task, y in (("classification", yc), ("regression", yr)):
        want = _forward(model, x, y, task)  # float32, state on the device
        assert _rel(offloaded(y, task), want) < OUTPUT_TOL, task
        assert _rel(offloaded(y, task, _rows), _rows(model, x, y, task)) < ROWS_TOL, task
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
    want_rows = _rows(model, x, yr, "regression")
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
    assert _rel(got, want) < OUTPUT_TOL
    assert torch.equal(got, again)
    # Block pairs 0 and 1 (the last pair writes nothing) x 20 train-row chunks, two forwards
    assert len(writes) == 2 * 2 * 20 and max(writes) == 1900
    assert _rel(_rows(model, x, yr, "regression"), want_rows) < ROWS_TOL


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
    # train rows) is what the evaluation queue's estimate gives with the train rows
    # (`inference_host_state_bytes`), and not more than the estimate without them (all rows)
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
    assert held == model.inference_host_state_bytes(*x.shape[1:], train_rows=yc.shape[1])
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
