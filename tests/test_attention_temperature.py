"""Attention temperatures of the model (`column_row_transformer` module docstring, attention
temperatures): gain-free q/k RMSNorm, `logscale` for the column summary, row ICL and decoder,
`gated_logscale` for the row stage, none for the column spread."""

import math

import helpers
import pytest
import torch

from tabsil.models import column_row_transformer as M
from tabsil.models import config_dict, model_from_state
from tabsil.models.column_row_transformer import (
    Attention,
    ColumnRowTransformer,
    ColumnRowTransformerConfig,
    LogScale,
)


def test_log_scale_multiplies_query_by_head_scale_times_log_n():
    torch.manual_seed(0)
    scale = LogScale(nhead=3, head_dim=4, hidden=8)
    assert ((scale.head_scale >= 0) & (scale.head_scale < 1)).all()  # U(0, 1) (same as Kumo)
    q = torch.randn(2, 3, 5, 4)
    expected = q * scale.head_scale.view(1, 3, 1, 1) * math.log(37)
    assert torch.allclose(scale(q, 37), expected)
    assert torch.equal(scale(q, 0), torch.zeros_like(q))  # log(max(n, 1))
    gated = LogScale(nhead=3, head_dim=4, hidden=8, gated=True)
    with torch.no_grad():
        gated.head_scale.copy_(scale.head_scale)
    assert torch.allclose(gated(q, 37), expected)  # zero last gate layer: initial scale 1
    torch.nn.init.normal_(gated.gate[2].weight, std=10.0)
    ratio = gated(q, 37) / expected
    assert (ratio >= 0).all() and (ratio <= 2 + 1e-5).all()
    assert not torch.allclose(ratio, torch.ones(1))


def test_affine_free_qk_norm_with_log_scale_bounds_logits(monkeypatch):
    # Without gain |q|=|k|=√d, so logit ≤ √d · a_h · log n, however large Wq and Wk grow
    torch.manual_seed(0)
    dim, nhead, n = 32, 4, 50
    attn = Attention(dim, nhead, temperature="logscale")
    assert not list(attn.q_norm.parameters()) and not list(attn.k_norm.parameters())
    with torch.no_grad():
        attn.qkv.weight.mul_(1000.0)
    seen = {}

    def spy(q, k, v):
        seen["logits"] = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
        return torch.nn.functional.scaled_dot_product_attention(q, k, v)

    monkeypatch.setattr(M, "_batched_sdpa", spy)
    attn(torch.randn(2, n, dim))
    d = dim // nhead
    bound = math.sqrt(d) * attn.log_scale.head_scale.view(1, nhead, 1, 1) * math.log(n)
    assert (seen["logits"].abs() <= bound * (1 + 1e-4)).all()


def test_temperatures_go_to_the_right_stages():
    model = ColumnRowTransformer(ColumnRowTransformerConfig.tiny())
    block = model.column_blocks[0]
    assert (
        isinstance(block.summarize.log_scale, LogScale) and block.summarize.log_scale.gate is None
    )
    assert block.broadcast.log_scale is None
    assert model.row_blocks[0].attention.log_scale.gate is not None  # gated_logscale
    assert model.icl_layers[0].attention.log_scale.gate is None
    assert model.heads["classification"].log_scale.gate is None
    # Gain-free QK-norm: q_norm and k_norm have no parameters
    assert not any("q_norm" in n or "k_norm" in n for n, _ in model.named_parameters())
    # the gate MLP width is `gate_hidden`
    gate = model.row_blocks[0].attention.log_scale.gate
    assert gate[0].out_features == ColumnRowTransformerConfig.tiny().gate_hidden


def test_model_runs_saves_and_loads():
    torch.manual_seed(0)
    model = helpers.tiny_model()
    x = torch.randn(2, 40, 5)
    yc, yr = torch.randint(0, 3, (2, 30)).float(), torch.randn(2, 30)
    out = {
        "classification": model(x, yc, "classification"),
        "regression": model(x, yr, "regression"),
    }
    assert all(torch.isfinite(o).all() for o in out.values())
    # Same values on the inference path (row-chunked, fused decoder)
    for task, y in (("classification", yc), ("regression", yr)):
        torch.testing.assert_close(model(x, y, task, inference=True), out[task])
    state = {"model_type": model.model_type, "config": config_dict(model.config)}
    state |= {"model": model.state_dict(), "borders": None}
    loaded, _ = model_from_state(state)
    assert loaded.config == model.config
    torch.testing.assert_close(loaded.eval()(x, yc, "classification"), out["classification"])


def test_temperature_validation():
    with pytest.raises(ValueError, match="temperature"):
        Attention(16, 2, temperature="qassmax")
