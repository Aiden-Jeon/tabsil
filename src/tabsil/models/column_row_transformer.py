"""Column-Row Transformer: the TabSil backbone. A table is read in three stages, column -> row ->
row ICL (the TabICL family layout used by TabPFN-3, arXiv 2605.13986 Fig. 5 / Appendix C, and
TabPFN-3.5, arXiv 2609.17895), with column and row stages interleaved as in Kumo-Tabular.

One model serves classification and regression (`TASKS`): only the target encoders and the output
heads are per task, everything else is shared. The `task` argument of `forward` picks the output.

1. Cell encoder (TabPFN-3.5 Fourier + ECDF): each feature is z-normalized with the train rows
   (missing cells become 0 and get a separate indicator). The token of feature j groups the cells of
   columns (j + 2^k - 1) mod C, k = 0..G-1 (`features_per_group` G). The z values times 32 learned
   frequencies go through sin/cos and a Linear; [z, missing indicator, sin/cos(pi k u), k = 1..4 of
   the in-column ECDF midrank u] go through another Linear; the sum is LayerNorm'd. A column that is
   constant on the train rows gets u = 0.5 on every row (so that it carries no test-only signal).
   Train-row tokens get a target embedding (classification: one orthogonally initialized vector per
   class, regression: Linear(y)), followed by a LayerNorm shared by the two tasks
2. Column and row stages, interleaved (`column_blocks` == `row_blocks` pairs): the cell tokens pass
   [column block i -> row block i] in turn.
   - Column block (ISAB): `n_inducing` inducing points per column cross-attend to the train rows to
     summarize the column, then every row (train and test) attends to that summary. Test rows never
     see each other
   - Row block: `n_cls` learned CLS tokens and the feature tokens of one row attend to each other
     (RoPE, base 1e5). The CLS tokens carry over between row blocks (they are not part of the column
     blocks). The last row block uses only the CLS tokens as queries; their outputs, concatenated,
     are the row vector (width `n_cls` x `emsize`)
3. Row ICL (`nlayers` layers): the train row vectors get a second target embedding (+ LayerNorm).
   Train rows attend to train rows; test rows attend to train rows only, with one key/value head
   (multi-query)
4. Output: a per-task residual MLP (pre-norm, last layer zero-initialized), then
   - classification: many-class retrieval decoder. Test rows (queries) attend to train rows (keys)
     and the attention weights average the train rows' one-hot labels into probabilities p; the
     logits are log(max(p, 1e-5) + 3e-5) (`decoder_heads` heads, averaged). It has no per-class
     parameters; the target embedding count (`n_out`) caps the classes seen in training
   - regression: one Linear to 999 quantiles (`tabsil.models.quantile_distribution`: levels
     k/1000, pinball loss, weight 0 and bias at the standard normal quantiles at the start).
     Inference turns the quantiles into bucket probabilities on the regression borders

Normalization is RMSNorm with a learned gain (pre-norm, plus one at the end of each stage) and the
MLP width factor is `ff_factor` (2).

Attention temperatures (Kumo-Tabular; `LogScale` is adapted from NVIDIA structured-data-models
`sdm/nn/scaling.py` `LogScale` / `GatedLogScale`, Apache-2.0, see NOTICE): q and k get an RMSNorm
without gain, and q is multiplied by a per-head scalar times log n (n = number of keys), so that the
logits are bounded by sqrt(head dim) * a_h * log n (`LogScale`):
- `logscale` (column summary: inducing points -> train rows, row ICL, retrieval decoder): q <- q *
  a_h * log n, a_h learned per head and drawn from U(0, 1)
- `gated_logscale` (row stage): logscale times (1 + tanh(gate(q))), gate a head dim ->
  `gate_hidden` -> head dim MLP whose last layer starts at 0
- none (column spread: rows -> inducing points)
Kumo applies RoPE before the q/k RMSNorm; without a gain the rotation does not change the norm, so
the order here (norm -> RoPE) gives the same values.

Choices made here (not in the reports): the row-stage attention output layer does not start at 0
(the CLS tokens only receive the row's cells through attention; starting at 0 gave all rows the
same vector and gradients near 1e-8); inducing points and CLS tokens start with std 0.02, attention
and MLP output layers at 0. Structure values of the published model are in `tabsil.recipe`.
"""

from __future__ import annotations

import contextlib
import math
from dataclasses import asdict, dataclass
from typing import Any, ClassVar

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from tabsil.models.quantile_distribution import N_QUANTILES, initial_bias

# The model outputs both tasks; `forward(task=...)` picks one
TASKS = ("classification", "regression")
# Attention temperatures used inside the model (module docstring): q is multiplied by these
TEMPERATURES = ("none", "logscale", "gated_logscale")
N_FOURIER = 32  # frequencies of the TabPFN-3.5 cell encoder
N_HARMONICS = 4  # sin/cos orders of the TabPFN-3.5 ECDF features
NAN_INDICATOR = -2.0  # indicator value of a missing cell (0 otherwise)
# Class cap at inference (as in the official TabPFN-3.5). The retrieval decoder has no per-class
# parameters, so more classes than target embeddings (`n_out`) are accepted by grouping the
# embeddings (`y_classes` of `forward`)
MAX_CLASSES = 160
# The CUDA attention kernels (`F.scaled_dot_product_attention`, flash etc.) have batch on one grid
# axis and fail with "invalid argument" above this many (stage 2 check: 8 tables x 10,240 rows = a
# row-stage batch of 81,920). Larger batches are split (each batch element is independent, so the
# values are the same). Training batches are smaller, so training is unchanged
SDPA_MAX_BATCH = 65_535
# Cap on the logits built at once when attention with log-sum-exp (`_attention_lse`) is computed
# off the GPU kernel (CPU): the batch is split above it (each batch element is independent, so the
# values are the same). 2^28 logits = 1 GiB of float32
LSE_CHUNK_LOGITS = 1 << 28
# Whether the classification retrieval decoder skips the score matrix (`fused` of
# `RetrievalDecoder`) in training forwards with activation checkpointing on
# (`ColumnRowTransformer.activation_checkpointing_rows`). On a 60,000-row table that matrix is
# 13 GB in float32, with several more copies through the softmax. Under bf16 autocast the
# score-matrix path computes q·k in bf16 and this path in float32, so the values differ slightly.
# A switch for comparing the two in measurements
CHECKPOINT_FUSED_DECODER = True


def standardize_cells(
    x: torch.Tensor, n_train: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Cell values (B, R, F) -> (z values, missing indicator, used-column mask (B, 1, F)).

    1) Build the missing indicator and z-normalize with the train rows (missing cells get the
       mean, i.e. 0)
    2) Columns constant on the train rows carry no information and are set to 0
    """
    missing = ~torch.isfinite(x)
    x = torch.where(missing, torch.nan, x)
    mean = torch.nanmean(x[:, :n_train], dim=1, keepdim=True)
    var = torch.nanmean(torch.square(x[:, :n_train] - mean), dim=1, keepdim=True)
    std = torch.sqrt(torch.nan_to_num(var, nan=0.0))
    x = torch.nan_to_num((x - mean) / std.clamp_min(1e-6), nan=0.0)
    # A column that is nearly constant on the train rows can get very large values on other rows.
    # Clip to ±100 so that the bf16 backward does not overflow
    x = x.clamp(-100.0, 100.0)
    indicator = missing.to(x.dtype) * NAN_INDICATOR
    used = (std > 1e-6).to(x.dtype)
    return x * used, indicator, used


@dataclass
class ColumnRowTransformerConfig:
    """Model size. The defaults are the TabSil structure (`tabsil.recipe.STRUCTURE`)."""

    emsize: int = 256  # width E of the cell, column and row stages
    features_per_group: int = 3
    n_inducing: int = 256
    column_blocks: int = 6
    column_heads: int = 4
    n_cls: int = 4  # row ICL width = n_cls x emsize
    row_blocks: int = 6  # interleaved with the column blocks: must equal column_blocks
    row_heads: int = 4
    rope_base: float = 100_000.0
    nlayers: int = 24  # row ICL layers
    nhead: int = 16  # row ICL heads
    ff_factor: int = 2
    decoder_heads: int = 6  # classification retrieval decoder
    decoder_head_dim: int = 64
    # hidden width of the gate MLP of the gated_logscale temperature
    gate_hidden: int = 64
    # number of target embeddings of the classification output (the class cap in training)
    n_out: int = 10
    # feature cap seen in training (`TrainConfig.max_features` of tabsil-train). Not part of the
    # structure; read by inference
    max_features: int = 100

    INFO_FIELDS: ClassVar[tuple[str, ...]] = ("max_features",)

    def __post_init__(self) -> None:
        if self.column_blocks != self.row_blocks or self.row_blocks < 1:
            raise ValueError(
                "the interleaved column/row stages need as many column blocks as row blocks (at"
                f" least one): column_blocks {self.column_blocks}, row_blocks {self.row_blocks}"
            )
        for name, heads, dim in (
            ("column_heads", self.column_heads, self.emsize),
            ("row_heads", self.row_heads, self.emsize),
            ("nhead", self.nhead, self.icl_dim),
        ):
            if dim % heads:
                raise ValueError(f"{name} {heads} does not divide the width {dim}")
        if (self.emsize // self.row_heads) % 2:
            raise ValueError("RoPE needs an even row-stage head dim")

    @property
    def icl_dim(self) -> int:
        return self.n_cls * self.emsize

    @property
    def tasks(self) -> tuple[str, ...]:
        """The tasks this model outputs."""
        return TASKS

    def output_size(self, task: str) -> int:
        """Outputs of `task`: the class cap (classification) or the quantile count (regression)."""
        return N_QUANTILES if task == "regression" else self.n_out

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def tiny(cls, **changes: Any) -> ColumnRowTransformerConfig:
        """A config small enough to follow by hand (notebooks, tests): E 16, 4 inducing points, one
        column/row stage pair, 2 CLS tokens, 2 row ICL layers, 2 heads each, class cap 3. Fields to
        change are given as arguments."""
        values = {
            "emsize": 16,
            "n_inducing": 4,
            "column_blocks": 1,
            "column_heads": 2,
            "n_cls": 2,
            "row_blocks": 1,
            "row_heads": 2,
            "nlayers": 2,
            "nhead": 2,
            "decoder_heads": 2,
            "decoder_head_dim": 8,
            "gate_hidden": 8,
            "n_out": 3,
        }
        return cls(**(values | changes))


# Top-level module name -> stage (`ColumnRowTransformer.num_parameters_by_stage`)
STAGES: dict[str, str] = {
    "cell_encoder": "cell_encoder",
    "cell_target": "cell_encoder",
    "cell_target_norm": "cell_encoder",
    "column_blocks": "column_stage",
    "cls_tokens": "row_stage",
    "row_blocks": "row_stage",
    "row_norm": "row_stage",
    "icl_target": "row_icl",
    "icl_target_norm": "row_icl",
    "icl_layers": "row_icl",
    "icl_norm": "row_icl",
    "pre_head": "output",
    "heads": "output",
}
STAGE_ORDER = ("cell_encoder", "column_stage", "row_stage", "row_icl", "output")


def group_offsets(g: int) -> tuple[int, ...]:
    """Offsets of the columns grouped into the token of feature j: 2^k − 1 (k = 0..g−1);
    (0, 1, 3) for g = 3."""
    return tuple(2**k - 1 for k in range(g))


def _group(t: torch.Tensor, offsets: tuple[int, ...]) -> torch.Tensor:
    # (B, R, C) -> (B, R, C, G): column (j + offset) mod C at feature j
    c = t.shape[-1]
    idx = (torch.arange(c, device=t.device)[:, None] + torch.tensor(offsets, device=t.device)) % c
    return t[..., idx]


def ecdf_midrank(x: torch.Tensor, n_train: int) -> torch.Tensor:
    """Per column, the midrank u ∈ [0, 1] against the train rows: (number smaller + number
    equal / 2) / number of non-missing train rows. Missing cells get 0.5. x: (B, R, C)."""
    finite = torch.isfinite(x)
    train = torch.where(finite[:, :n_train], x[:, :n_train], torch.inf)
    sorted_train = train.transpose(1, 2).sort(-1).values.contiguous()  # (B, C, N), missing last
    values = torch.where(finite, x, 0.0).transpose(1, 2).contiguous()  # (B, C, R)
    left = torch.searchsorted(sorted_train, values, side="left")
    right = torch.searchsorted(sorted_train, values, side="right")
    n_valid = finite[:, :n_train].sum(1).clamp_min(1)  # (B, C)
    u = ((left + right).to(x.dtype) / 2 / n_valid[..., None].to(x.dtype)).clamp(0, 1)
    return torch.where(finite, u.transpose(1, 2), 0.5)


def _rope(t: torch.Tensor, base: float) -> torch.Tensor:
    # t: (N, H, L, d). Rotate the two halves by the positions 0..L-1 (rotate-half)
    length, d = t.shape[-2], t.shape[-1]
    inv = base ** (-torch.arange(0, d, 2, device=t.device, dtype=torch.float32) / d)
    angle = torch.arange(length, device=t.device, dtype=torch.float32)[:, None] * inv
    cos, sin = angle.cos().to(t.dtype), angle.sin().to(t.dtype)
    a, b = t[..., : d // 2], t[..., d // 2 :]
    return torch.cat([a * cos - b * sin, a * sin + b * cos], dim=-1)


class LogScale(nn.Module):
    """q ← q · a_h · log n (module docstring, attention temperatures). a_h (`head_scale`) is a
    learned per-head scalar drawn from U(0, 1). With `gated`, q is also multiplied by
    (1 + tanh(gate(q))) (the gate starts at 0, so the first factor is 1).

    Adapted from NVIDIA structured-data-models (Apache-2.0) `sdm/nn/scaling.py` `LogScale` and
    `GatedLogScale`, merged into one class. The query gate follows TabICLv2's QASSMax query MLP
    (BSD-3-Clause, `tabicl/_model/ssmax.py` `QASSMaxMLP`). See NOTICE."""

    def __init__(self, nhead: int, head_dim: int, hidden: int, gated: bool = False) -> None:
        super().__init__()
        self.head_scale = nn.Parameter(torch.rand(nhead))
        self.gate = None
        if gated:
            self.gate = nn.Sequential(
                nn.Linear(head_dim, hidden), nn.GELU(), nn.Linear(hidden, head_dim)
            )
            nn.init.zeros_(self.gate[2].weight)
            nn.init.zeros_(self.gate[2].bias)

    def forward(self, q: torch.Tensor, n_keys: int) -> torch.Tensor:
        # q: (N, H, L, d). log n is a tensor under compile (so that a new n does not recompile)
        if torch.compiler.is_compiling():
            log_n = (torch.ones((), device=q.device) * n_keys).clamp(min=1).log()
        else:
            log_n = q.new_full((), math.log(max(n_keys, 1)), dtype=torch.float32)
        scale = (self.head_scale.float() * log_n).to(q.dtype).view(1, -1, 1, 1)
        out = q * scale
        if self.gate is not None:
            out = out * (1 + torch.tanh(self.gate(q)))
        return out


def _log_scale(temperature: str, nhead: int, head_dim: int, hidden: int) -> LogScale | None:
    """The temperature module of `temperature` (`TEMPERATURES`), None for "none"."""
    if temperature not in TEMPERATURES:
        raise ValueError(f"the temperature is one of {TEMPERATURES}: {temperature}")
    if temperature == "none":
        return None
    return LogScale(nhead, head_dim, hidden, gated=temperature == "gated_logscale")


def _scale_query(module: nn.Module, q: torch.Tensor, n_keys: int) -> torch.Tensor:
    """Multiply q by the module's temperature (`module.log_scale`, if any)."""
    if module.log_scale is not None:
        return module.log_scale(q, n_keys)
    return q


class Attention(nn.Module):
    """Multi-head attention with one weight for q, k and v (`qkv`, (3D, D)).

    - With `context`, keys and values come from it (cross-attention), otherwise from the query input
    - With `n_context` (self-attention), only the first n_context tokens are keys and values, and
      the queries after them (test rows) see only the keys and values of the first head
      (multi-query)
    - q and k get an RMSNorm without gain; `temperature` (`TEMPERATURES`, module docstring) scales
      q through the `log_scale` module (`gate_hidden`: its gate MLP width)
    - `rope_base`: RoPE on q and k
    - `zero_out`: the output layer starts at 0 (at first it adds nothing to the residual)
    """

    def __init__(
        self,
        dim: int,
        nhead: int,
        *,
        temperature: str = "none",
        gate_hidden: int = 64,
        rope_base: float | None = None,
        zero_out: bool = True,
    ) -> None:
        super().__init__()
        self.nhead, self.rope_base = nhead, rope_base
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.q_norm = nn.RMSNorm(dim // nhead, elementwise_affine=False)
        self.k_norm = nn.RMSNorm(dim // nhead, elementwise_affine=False)
        if zero_out:
            nn.init.zeros_(self.out.weight)
        self.log_scale = _log_scale(temperature, nhead, dim // nhead, gate_hidden)

    def _heads(self, t: torch.Tensor) -> torch.Tensor:
        n, length, dim = t.shape
        return t.view(n, length, self.nhead, dim // self.nhead).transpose(1, 2)  # (N, H, L, d)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        n_context: int | None = None,
    ) -> torch.Tensor:
        n, length, dim = x.shape
        w = self.qkv.weight
        source = x if context is None else context
        if n_context is not None:
            source = source[:, :n_context]
        q = self._heads(F.linear(x, w[:dim]))
        k = self._heads(F.linear(source, w[dim : 2 * dim]))
        v = self._heads(F.linear(source, w[2 * dim :]))
        q, k = self.q_norm(q), self.k_norm(k)
        if self.rope_base is not None:
            # Even when the queries are a prefix (CLS), their positions start at 0 like the keys
            q, k = _rope(q, self.rope_base), _rope(k, self.rope_base)
        q = _scale_query(self, q, k.shape[2])
        if n_context is None:
            out = _batched_sdpa(q, k, v)
        else:
            train = _batched_sdpa(q[:, :, :n_context], k, v)
            # Test rows: every query head sees the keys/values of the first head. The heads are
            # folded into the query length and computed against one key/value head (so the
            # keys/values are not repeated per head)
            test_q = q[:, :, n_context:]
            h, t, d = test_q.shape[1:]
            test = _batched_sdpa(test_q.reshape(n, 1, h * t, d), k[:, :1], v[:, :1])
            test = test.view(n, h, t, d)
            out = torch.cat([train, test], dim=2)
        return self.out(out.transpose(1, 2).reshape(n, length, dim))

    # The two methods below split self-attention with `n_context` (row ICL) into the context rows
    # and the query rows after them, for inference with a train context
    # (`ColumnRowTransformer.train_context`). They compute the same formulas as `forward`

    def forward_context(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """`forward(x, n_context=len)` when every row is a context row (train rows only), and the
        keys and values of the first head (N, 1, L, d) that later query rows attend to (multi-query,
        `forward_queries`). The keys and values are cast to the dtype the attention kernel gets
        (bfloat16 under autocast), so storing them changes no value."""
        n, length, dim = x.shape
        w = self.qkv.weight
        q = self._heads(F.linear(x, w[:dim]))
        k = self._heads(F.linear(x, w[dim : 2 * dim]))
        v = self._heads(F.linear(x, w[2 * dim :]))
        q, k = self.q_norm(q), self.k_norm(k)
        if self.rope_base is not None:
            q, k = _rope(q, self.rope_base), _rope(k, self.rope_base)
        q = _scale_query(self, q, k.shape[2])
        out = _batched_sdpa(q, k, v)
        dtype = _attention_dtype(v)
        # copies: a view of the first head would keep every head's keys and values alive
        first = (k[:, :1].to(dtype, copy=True), v[:, :1].to(dtype, copy=True))
        return self.out(out.transpose(1, 2).reshape(n, length, dim)), *first

    def forward_queries(
        self, x: torch.Tensor, k: torch.Tensor, v: torch.Tensor, n_keys: int
    ) -> torch.Tensor:
        """The query rows after the context rows (test rows) of `forward(x, n_context=...)`: every
        query head attends to the first head's keys and values `k`, `v` (N, 1, n_keys, d) of the
        context rows (`forward_context`)."""
        if self.rope_base is not None:
            raise ValueError("RoPE attention has no query-only form (positions follow the keys)")
        n, length, dim = x.shape
        q = _scale_query(self, self.q_norm(self._heads(F.linear(x, self.qkv.weight[:dim]))), n_keys)
        h, t, d = q.shape[1:]
        out = _batched_sdpa(q.reshape(n, 1, h * t, d), k, v).view(n, h, t, d)
        return self.out(out.transpose(1, 2).reshape(n, length, dim))

    # The three methods below split `forward` (cross-attention, no `n_context`) into queries,
    # keys/values and output. Inference that accumulates the softmax over chunks of keys
    # (`_OnlineSummary`) uses them. They compute the same formulas as `forward` (which is left as
    # it is, so that training and unchunked inference keep their computation order)

    def split_queries(self, x: torch.Tensor, n_keys: int) -> torch.Tensor:
        """Queries (N, H, L, d): q RMSNorm, temperature (for `n_keys` keys). RoPE is not supported:
        the key positions would differ per chunk."""
        if self.rope_base is not None:
            raise ValueError("RoPE attention cannot be computed over chunks of keys")
        dim = x.shape[-1]
        q = self.q_norm(self._heads(F.linear(x, self.qkv.weight[:dim])))
        return _scale_query(self, q, n_keys)

    def split_keys(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Keys and values (N, H, L, d) of `context` (k RMSNorm)."""
        w = self.qkv.weight
        dim = w.shape[1]
        k = self.k_norm(self._heads(F.linear(context, w[dim : 2 * dim])))
        v = self._heads(F.linear(context, w[2 * dim :]))
        return k, v

    def split_output(self, out: torch.Tensor) -> torch.Tensor:
        """Per-head attention output (N, H, L, d) -> output layer (N, L, D)."""
        n, heads, length, d = out.shape
        return self.out(out.transpose(1, 2).reshape(n, length, heads * d))


def _batched_sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Scaled dot-product attention; a batch larger than `SDPA_MAX_BATCH` is split into batches of
    that size (the CUDA kernel grid limit)."""
    n = q.shape[0]
    if n <= SDPA_MAX_BATCH:
        return F.scaled_dot_product_attention(q, k, v)
    step = SDPA_MAX_BATCH
    return torch.cat(
        [
            F.scaled_dot_product_attention(q[i : i + step], k[i : i + step], v[i : i + step])
            for i in range(0, n, step)
        ]
    )


def _attention_dtype(t: torch.Tensor) -> torch.dtype:
    """The dtype the attention kernel computes `t` in: the autocast dtype when autocast is on for
    t's device (SDPA is in its lower-precision list), else t's own dtype."""
    device = t.device.type
    if t.is_floating_point() and torch.is_autocast_enabled(device):
        return torch.get_autocast_dtype(device)
    return t.dtype


def _wide(t: torch.Tensor) -> torch.Tensor:
    """t in float32, or in its own dtype if that is wider (float64)."""
    return t.to(torch.promote_types(t.dtype, torch.float32))


def _attention_lse(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Softmax attention output (N, H, Lq, d) and the per-query log-sum-exp (N, H, Lq) in float32
    (float64 for float64 inputs; `_OnlineSummary` merges chunks of keys with it). q and k are cast
    to v's dtype (as SDPA under autocast).

    On the GPU this is SDPA's memory-efficient kernel (the aten op that also returns the
    log-sum-exp): logits and softmax are float32 inside the kernel and the output has the input
    dtype. Elsewhere (CPU, float64) the logits are built in float32 or wider, `LSE_CHUNK_LOGITS` at
    a time."""
    q, k = q.to(v.dtype), k.to(v.dtype)
    n = q.shape[0]
    if q.is_cuda and v.dtype != torch.float64:
        if n > SDPA_MAX_BATCH:
            parts = [
                _attention_lse(
                    q[i : i + SDPA_MAX_BATCH], k[i : i + SDPA_MAX_BATCH], v[i : i + SDPA_MAX_BATCH]
                )
                for i in range(0, n, SDPA_MAX_BATCH)
            ]
            return torch.cat([p[0] for p in parts]), torch.cat([p[1] for p in parts])
        with torch.autocast("cuda", enabled=False):
            out, lse = torch.ops.aten._scaled_dot_product_efficient_attention(q, k, v, None, True)[
                :2
            ]
        return out, lse[..., : q.shape[-2]].float()  # the kernel may pad the query length
    per_item = q[0].numel() // q.shape[-1] * k.shape[-2]  # logits of one batch element
    step = max(1, LSE_CHUNK_LOGITS // max(1, per_item))
    outs, lses = [], []
    with torch.autocast(q.device.type, enabled=False):
        for i in range(0, n, step):
            qi, ki, vi = (_wide(t[i : i + step]) for t in (q, k, v))
            s = (qi @ ki.transpose(-1, -2)) * q.shape[-1] ** -0.5
            lse = s.logsumexp(-1)
            outs.append((torch.exp(s - lse[..., None]) @ vi).to(v.dtype))
            lses.append(lse)
    return torch.cat(outs), torch.cat(lses)


class _OnlineSummary:
    """`InducedBlock.summary` accumulated over chunks of train rows (online softmax; inference,
    `ColumnRowTransformer._rows_interleaved_offload`). The queries of the summary attention are the
    inducing points, the same for every column; the keys and values are the train rows. The
    attention output and log-sum-exp of each chunk, merged in float32 (`_wide`; weights
    exp(lse_j - lse)), equal one attention over all train rows (only the computation order
    differs; in float64 they agree to rounding).

    The temperature depends on the key count, so the train-row count `n_train` is given up front.
    `add` takes a chunk (N, rows, E), `finish` returns the summary (N, n_inducing, E). N = datasets
    x columns."""

    def __init__(self, block: InducedBlock, n: int, n_train: int, dtype: torch.dtype) -> None:
        self.block, self.n = block, n
        self.ind = block.inducing.to(dtype)[None]  # (1, n_inducing, E): the same for all columns
        self.q = block.summarize.split_queries(block.norms["ind"](self.ind), n_train)
        self.q_batch: torch.Tensor | None = None  # q expanded to N (`_wide`, at the first `add`)
        self.acc: torch.Tensor | None = None  # output so far (N, H, n_inducing, d), `_wide`
        self.lse: torch.Tensor | None = None  # log-sum-exp so far (N, H, n_inducing)
        self.value_dtype = dtype  # SDPA output dtype (the dtype of v, set at the first `add`)

    def add(self, h_rows: torch.Tensor) -> None:
        attention = self.block.summarize
        k, v = attention.split_keys(self.block.norms["rows"](h_rows))
        # q, k and v are rounded to the dtype SDPA gets when computed at once (bfloat16 under
        # autocast) and then computed in float32 (`_wide`), so that the per-chunk outputs are not
        # rounded to bfloat16 (computed at once, the output is rounded once, at the end)
        self.value_dtype = v.dtype
        if self.q_batch is None:
            q = _wide(self.q.to(v.dtype))
            self.q_batch = q.expand(self.n, -1, -1, -1).contiguous()
        k, v = _wide(k.to(v.dtype)), _wide(v)
        out, lse = _attention_lse(self.q_batch, k, v)
        if self.acc is None:
            self.acc, self.lse = out, lse
            return
        total = torch.logaddexp(self.lse, lse)
        self.acc.mul_(torch.exp(self.lse - total)[..., None])
        self.acc.add_(out * torch.exp(lse - total)[..., None])
        self.lse = total

    def finish(self) -> torch.Tensor:
        """The summary (N, n_inducing, E). Without train rows the attention output is 0 (the
        output of SDPA over zero keys in `InducedBlock.summary`)."""
        block = self.block
        if self.acc is None:
            _, heads, n_ind, d = self.q.shape
            acc = self.q.new_zeros(self.n, heads, n_ind, d)
        else:
            acc = self.acc
        out = block.summarize.split_output(acc.to(self.value_dtype))
        ind = self.ind.expand(self.n, -1, -1) + out
        return ind + block.summarize_mlp(block.norms["ind_mlp"](ind))


def _call(block: nn.Module, ckpt: bool, *args, **kwargs) -> torch.Tensor:
    """Call one block. With `ckpt`, use activation checkpointing: the forward keeps only the block
    inputs (not its activations) until backward, which recomputes the block forward. Blocks
    compiled one by one by the `compile` speed-up option work as they are (the recomputation also
    runs the compiled forward). Non-reentrant, so it works with DDP. The forward has no randomness,
    so the RNG state is not saved."""
    if not ckpt:
        return block(*args, **kwargs)
    return checkpoint(block, *args, use_reentrant=False, preserve_rng_state=False, **kwargs)


def _mlp(dim: int, ff_factor: int) -> nn.Sequential:
    mlp = nn.Sequential(
        nn.Linear(dim, ff_factor * dim, bias=False),
        nn.GELU(),
        nn.Linear(ff_factor * dim, dim, bias=False),
    )
    nn.init.zeros_(mlp[2].weight)
    return mlp


class Block(nn.Module):
    """Pre-norm self-attention + MLP (row stage, row ICL)."""

    def __init__(self, dim: int, nhead: int, ff_factor: int, **attention) -> None:
        super().__init__()
        self.attention = Attention(dim, nhead, **attention)
        self.attention_norm = nn.RMSNorm(dim)
        self.mlp = _mlp(dim, ff_factor)
        self.mlp_norm = nn.RMSNorm(dim)

    def forward(
        self, h: torch.Tensor, n_context: int | None = None, n_queries: int | None = None
    ) -> torch.Tensor:
        """With `n_queries`, only the first n_queries tokens are queries and only their outputs are
        returned (keys and values are all tokens; the CLS tokens of the last row block)."""
        normed = self.attention_norm(h)
        if n_queries is None:
            h = h + self.attention(normed, n_context=n_context)
        else:
            h = h[:, :n_queries] + self.attention(normed[:, :n_queries], context=normed)
        return h + self.mlp(self.mlp_norm(h))

    def forward_context(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """`forward(h, n_context=len)` for context rows only (row ICL train rows), and the first
        head's keys and values the query rows attend to (`Attention.forward_context`)."""
        out, k, v = self.attention.forward_context(self.attention_norm(h))
        h = h + out
        return h + self.mlp(self.mlp_norm(h)), k, v

    def forward_queries(
        self, h: torch.Tensor, k: torch.Tensor, v: torch.Tensor, n_keys: int
    ) -> torch.Tensor:
        """The query rows (row ICL test rows) of `forward(h, n_context=...)`, attending to the
        stored keys and values of the context rows (`forward_context`)."""
        h = h + self.attention.forward_queries(self.attention_norm(h), k, v, n_keys)
        return h + self.mlp(self.mlp_norm(h))


class InducedBlock(nn.Module):
    """One column-stage block (ISAB): the inducing points summarize the train rows (`logscale`
    temperature) and every row attends to that summary (no temperature)."""

    def __init__(
        self, dim: int, nhead: int, n_inducing: int, ff_factor: int, gate_hidden: int
    ) -> None:
        super().__init__()
        self.inducing = nn.Parameter(nn.init.trunc_normal_(torch.empty(n_inducing, dim), std=0.02))
        self.summarize = Attention(dim, nhead, temperature="logscale", gate_hidden=gate_hidden)
        self.summarize_mlp = _mlp(dim, ff_factor)
        self.broadcast = Attention(dim, nhead, gate_hidden=gate_hidden)
        self.broadcast_mlp = _mlp(dim, ff_factor)
        # Separate norms (they have learned gains) for queries, context and each MLP input
        self.norms = nn.ModuleDict(
            {name: nn.RMSNorm(dim) for name in ("ind", "rows", "ind_mlp", "h", "hidden", "h_mlp")}
        )

    def forward(self, h: torch.Tensor, n_train: int) -> torch.Tensor:
        # h: (N, R, E), N = datasets x columns. The first n_train rows are the train rows
        return self.spread(h, self.summary(h[:, :n_train]))

    def summary(self, h_train: torch.Tensor) -> torch.Tensor:
        """The inducing points' summary (N, n_inducing, E) of the train rows (N, n_train, E). It
        does not see the test rows, so chunked inference computes it once per column."""
        nm = self.norms
        ind = self.inducing.to(h_train.dtype).expand(h_train.shape[0], -1, -1)
        ind = ind + self.summarize(nm["ind"](ind), context=nm["rows"](h_train))
        return ind + self.summarize_mlp(nm["ind_mlp"](ind))

    def spread(self, h: torch.Tensor, ind: torch.Tensor) -> torch.Tensor:
        """Each row attends to the summary (`summary`). Rows do not see each other."""
        nm = self.norms
        h = h + self.broadcast(nm["h"](h), context=nm["hidden"](ind))
        return h + self.broadcast_mlp(nm["h_mlp"](h))


class FourierCellEncoder(nn.Module):
    """TabPFN-3.5 cell encoder: learned-frequency sin/cos terms + [z value, missing indicator,
    ECDF harmonics] terms -> LayerNorm."""

    def __init__(self, g: int, dim: int) -> None:
        super().__init__()
        self.frequencies = nn.Parameter(torch.logspace(-1, 1, N_FOURIER))
        self.fourier = nn.Linear(g * 2 * N_FOURIER, dim)
        self.metadata = nn.Linear(g * (2 + 2 * N_HARMONICS), dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, z: torch.Tensor, indicator: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        # each (B, R, C, G)
        b, r, c, g = z.shape
        k = torch.arange(1, N_HARMONICS + 1, device=u.device, dtype=u.dtype)
        harmonics = math.pi * u[..., None] * k
        meta = torch.cat(
            [z[..., None], indicator[..., None], harmonics.sin(), harmonics.cos()], dim=-1
        ).reshape(b, r, c, -1)
        phase = z[..., None] * self.frequencies.to(z.dtype)  # (B, R, C, G, F)
        fourier = torch.cat([phase.sin(), phase.cos()], dim=-1).reshape(b, r, c, -1)
        return self.norm(self.fourier(fourier) + self.metadata(meta))


class TargetEncoder(nn.Module):
    """Train-row targets -> embeddings: one orthogonally initialized vector per class
    (classification), Linear(y) (regression)."""

    def __init__(self, task: str, n_classes: int, dim: int) -> None:
        super().__init__()
        self.task, self.n_classes = task, n_classes
        if task == "classification":
            self.encode = nn.Embedding(n_classes, dim)
            nn.init.orthogonal_(self.encode.weight)
        else:
            self.encode = nn.Linear(1, dim)

    def forward(self, y_train: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        # (B, N) → (B, N, dim)
        if self.task == "classification":
            return self.encode(y_train.long().clamp(0, self.n_classes - 1)).to(dtype)
        return self.encode(y_train.to(dtype)[..., None])


class PreHeadMLP(nn.Module):
    """Residual MLP before the output layer (pre-norm; the last layer starts at 0, so it is the
    identity at first)."""

    def __init__(self, dim: int, ff_factor: int) -> None:
        super().__init__()
        self.norm = nn.RMSNorm(dim)
        self.mlp = _mlp(dim, ff_factor)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return h + self.mlp(self.norm(h))


def _quantile_head(dim: int, n: int) -> nn.Linear:
    """The regression quantile output: one Linear (after the pre-head MLP) with weight 0 and bias at
    the standard normal quantiles (the first prediction is a standard normal distribution,
    `tabsil.models.quantile_distribution`)."""
    head = nn.Linear(dim, n)
    nn.init.zeros_(head.weight)
    with torch.no_grad():
        head.bias.copy_(initial_bias(n))
    return head


class _CellStats:
    """Computes the train-row statistics of the cell normalization (`standardize_cells`) and the
    ECDF midrank (`ecdf_midrank`) once, and applies the same formulas to slices of rows and columns
    (`features`, for `ColumnRowTransformer._rows_interleaved_chunked`). The computation is per cell,
    so the values equal those computed on the whole table at once."""

    def __init__(self, x: torch.Tensor, n_train: int) -> None:
        train = x[:, :n_train]
        finite = torch.isfinite(train)
        t = torch.where(finite, train, torch.nan)
        self.mean = torch.nanmean(t, dim=1, keepdim=True)  # (B, 1, C)
        var = torch.nanmean(torch.square(t - self.mean), dim=1, keepdim=True)
        self.std = torch.sqrt(torch.nan_to_num(var, nan=0.0))
        self.used = (self.std > 1e-6).to(x.dtype)
        train = torch.where(finite, train.float(), torch.inf)
        self.sorted = train.transpose(1, 2).sort(-1).values.contiguous()  # (B, C, N)
        self.n_valid = finite.sum(1).clamp_min(1)  # (B, C)

    FIELDS = ("mean", "std", "used", "sorted", "n_valid")

    @classmethod
    def from_fields(cls, values: dict[str, torch.Tensor]) -> _CellStats:
        stats = cls.__new__(cls)
        for name in cls.FIELDS:
            setattr(stats, name, values[name])
        return stats

    def map(self, fn) -> _CellStats:
        """The statistics with `fn` applied to each tensor (batch first)."""
        return _CellStats.from_fields({n: fn(getattr(self, n)) for n in self.FIELDS})

    def features(
        self, x: torch.Tensor, cols: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(z values, missing indicator, ECDF u) of a row slice x (B, Rj, Cj). `cols` are the
        indices of x's columns in the original table (None: all columns)."""

        def pick(t: torch.Tensor) -> torch.Tensor:
            return t if cols is None else t[..., cols]

        mean, std, used = pick(self.mean), pick(self.std), pick(self.used)
        missing = ~torch.isfinite(x)
        z = torch.where(missing, torch.nan, x)
        z = torch.nan_to_num((z - mean) / std.clamp_min(1e-6), nan=0.0).clamp(-100.0, 100.0)
        indicator = missing.to(z.dtype) * NAN_INDICATOR
        z = z * used
        sorted_train = self.sorted if cols is None else self.sorted[:, cols]
        n_valid = self.n_valid if cols is None else self.n_valid[:, cols]
        values = torch.where(~missing, x.float(), 0.0).transpose(1, 2).contiguous()
        left = torch.searchsorted(sorted_train, values, side="left")
        right = torch.searchsorted(sorted_train, values, side="right")
        u = ((left + right).float() / 2 / n_valid[..., None].float()).clamp(0, 1)
        u = torch.where(~missing, u.transpose(1, 2), 0.5).to(z.dtype)
        return z, indicator, torch.where(used > 0, u, 0.5)


class RetrievalDecoder(nn.Module):
    """Many-class retrieval decoder: test rows (queries) attend to train rows (keys), and the
    attention weights average the train rows' one-hot labels (then averaged over heads). The logits
    are log(max(p, 1e-5) + 3e-5). q gets the `logscale` temperature."""

    def __init__(self, dim: int, heads: int, head_dim: int, n_classes: int, hidden: int) -> None:
        super().__init__()
        self.heads, self.head_dim, self.n_classes = heads, head_dim, n_classes
        self.q_projection = nn.Linear(dim, heads * head_dim)
        self.k_projection = nn.Linear(dim, heads * head_dim)
        self.log_scale = _log_scale("logscale", heads, head_dim, hidden)

    def forward(
        self,
        h: torch.Tensor,
        y_train: torch.Tensor,
        n_train: int,
        *,
        n_classes: int | None = None,
        fused: bool = False,
    ) -> torch.Tensor:
        """h: (B, R, D) -> test-row logits (B, T, n_classes). Queries come from the test rows,
        keys only from the train rows.

        `n_classes` (default: the number of target embeddings) is the one-hot class count. The
        decoder has no per-class parameters, so it may exceed the number of target embeddings
        (when inference groups the target embeddings). With `fused`, the (test rows x train rows)
        score matrix is not built; a single attention with the one-hot labels as values computes
        the result (for inference: on large tables that matrix reaches hundreds of GB). Same
        values, only the computation order differs."""
        b = h.shape[0]
        n_classes = n_classes or self.n_classes
        q = self.q_projection(h[:, n_train:]).view(b, -1, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_projection(h[:, :n_train]).view(b, -1, self.heads, self.head_dim).transpose(1, 2)
        q = _scale_query(self, q, n_train)
        onehot = F.one_hot(y_train.long().clamp(0, n_classes - 1), n_classes).float()
        if fused:
            p = _retrieve(q, k, onehot).mean(1)
        else:
            # Probabilities and log in float32 (bf16 blurs the values near the clip)
            scores = (q @ k.transpose(-1, -2)).float() / math.sqrt(self.head_dim)
            p = (torch.softmax(scores, dim=-1) @ onehot[:, None]).mean(1)  # (B, T, n_classes)
        return torch.log(p.clamp_min(1e-5) + 3e-5)

    def keys(self, h_train: torch.Tensor) -> torch.Tensor:
        """The keys (B, heads, n_train, head_dim) of the train rows (B, n_train, D), for
        `from_keys` (inference with a train context)."""
        b = h_train.shape[0]
        k = self.k_projection(h_train).view(b, -1, self.heads, self.head_dim).transpose(1, 2)
        return k.contiguous()

    def from_keys(
        self, h_test: torch.Tensor, keys: torch.Tensor, onehot: torch.Tensor, n_train: int
    ) -> torch.Tensor:
        """`forward(..., fused=True)` for the test rows h_test (B, T, D) with the train rows' keys
        (`keys`) and one-hot labels (B, n_train, n_classes, float32) computed beforehand."""
        b = h_test.shape[0]
        q = self.q_projection(h_test).view(b, -1, self.heads, self.head_dim).transpose(1, 2)
        p = _retrieve(_scale_query(self, q, n_train), keys, onehot).mean(1)
        return torch.log(p.clamp_min(1e-5) + 3e-5)


def _retrieve(q: torch.Tensor, k: torch.Tensor, onehot: torch.Tensor) -> torch.Tensor:
    """softmax(q·kᵀ/√d) @ onehot per head (B, H, T, n_classes), in float32 and without the score
    matrix.

    q (B, H, T, d), k (B, H, N, d), onehot (B, N, n_classes). To keep the value width of the
    attention kernel equal to that of q and k, the one-hot labels are cut into pieces of d columns,
    each with its own attention (the method of `_class_attention` in the official TabPFN-3.5
    inference code). q and k are the same for every piece and are not copied. Computed in float32
    with autocast off.

    Adapted from TabPFN (Apache-2.0) `src/tabpfn/architectures/tabpfn_v3_5.py` `_class_attention`;
    see NOTICE."""
    b, heads, t, d = q.shape
    n, n_classes = onehot.shape[1:]
    pieces = math.ceil(n_classes / d)
    v = F.pad(onehot, (0, pieces * d - n_classes)).view(b, n, pieces, d)
    q, k = q.float(), k.float()
    with torch.autocast(q.device.type, enabled=False):
        out = [  # (B, H, T, d) per piece
            F.scaled_dot_product_attention(q, k, v[:, None, :, j].expand(-1, heads, -1, -1))
            for j in range(pieces)
        ]
    return torch.cat(out, dim=-1)[..., :n_classes]


def _clear_cuda_error() -> None:
    """Clear the last error left by a failed CUDA runtime call (`cudaHostRegister`). Otherwise the
    next kernel launch check (`cudaGetLastError`) stops on that error. torch's cudart binding has no
    `cudaGetLastError`, so the already loaded libcudart is called through ctypes (Linux; a no-op if
    it is not found)."""
    import ctypes

    try:
        with open("/proc/self/maps") as f:
            paths = {line.split()[-1] for line in f if "libcudart.so" in line}
    except OSError:
        return
    for path in sorted(paths):
        try:
            ctypes.CDLL(path).cudaGetLastError()
        except (OSError, AttributeError):
            continue


def _host_state(shape: tuple[int, ...], dtype: torch.dtype, pin: bool) -> tuple[torch.Tensor, bool]:
    """The CPU cell state (offload of `ColumnRowTransformer._rows_interleaved_offload`): a regular
    (pageable) CPU tensor of the exact size, registered as pinned in place (`cudaHostRegister`) if
    `pin`. `torch.empty(..., pin_memory=True)` goes through the caching host allocator, which rounds
    sizes up to a power of two (984 GB -> 1 TiB for BeyondArena's maps_router_eta_1m) and keeps
    freed blocks in the process until `empty_cache`, so an eval worker that has gone through a large
    table keeps holding hundreds of GB it no longer uses. The returned bool says whether the tensor
    was registered (undo with `_release_host_state`). If registration fails (memlock limit etc.),
    warn and use pageable memory (same values, slower copies)."""
    state = torch.empty(shape, dtype=dtype, device="cpu")
    if not pin or state.numel() == 0 or not torch.cuda.is_available():
        return state, False
    err = int(torch.cuda.cudart().cudaHostRegister(state.data_ptr(), state.nbytes, 0))
    if err != 0:
        _clear_cuda_error()
        import warnings

        warnings.warn(
            f"could not register the {state.nbytes / 2**30:.1f} GiB cell state as pinned"
            f" (cudaError {err}). Continuing with pageable memory (same values, slower device"
            " copies)",
            RuntimeWarning,
            stacklevel=3,
        )
        return state, False
    return state, True


def _release_host_state(state: torch.Tensor, registered: bool) -> None:
    """Undo the registration of `_host_state` (the memory returns to the OS when the tensor is
    freed). The caller first waits for its copies (`_HostRows.close`: they are asynchronous, on
    their own streams and threads); the device is also synchronized here in case work is still
    pending. Also called on the way out of an exception (`finally`), so it raises
    no new exception (so as not to mask the original one)."""
    if not registered:
        return
    import warnings

    try:
        torch.cuda.synchronize()
    except RuntimeError:  # device already in an error state: the original exception says why
        pass
    err = int(torch.cuda.cudart().cudaHostUnregister(state.data_ptr()))
    if err != 0:
        _clear_cuda_error()
        warnings.warn(
            f"could not unregister the pinned cell state (cudaError {err})",
            RuntimeWarning,
            stacklevel=3,
        )


class _HostRows:
    """Reads the cell state h (B, R, C, E) to the device in row chunks and writes it back
    (inference, `ColumnRowTransformer._rows_interleaved_offload`). When h is on the CPU and the
    device is a GPU, the copies overlap the compute:

    - Reads (`rows`): the next chunk is read ahead on its own stream (H2D), alternating between two
      device buffers. The compute stream waits for that chunk's copy event, and a buffer is
      refilled only after the compute is done with the chunk it held
    - Writes (`write`): another stream (D2H) waits for the compute event and copies. At most two
      chunks are being written at a time (they hold device memory). `flush` waits until every write
      has reached the CPU
    - Each direction's copies are issued by its own thread. With pageable (unregistered) state a
      copy call holds the CPU until it is done, so it must be off the thread that launches the
      compute to overlap (with registered state the call returns at once)

    The chunk being read (the next one) and the chunks being written (earlier ones) are different
    rows, so they never overlap. `flush` before moving to the next block. Otherwise (h on the
    device, or a CPU device in tests) the chunks are copied in turn."""

    WRITES_IN_FLIGHT = 2

    def __init__(self, h: torch.Tensor, step: int, device: torch.device) -> None:
        import collections
        from concurrent.futures import ThreadPoolExecutor

        self.h, self.step, self.device = h, step, device
        self.overlap = device.type == "cuda" and h.device.type == "cpu"
        # Inference mode is per thread, so the copy threads use the caller's (writing to an
        # inference tensor must happen inside inference mode)
        self.inference = torch.is_inference_mode_enabled()
        self.pending: collections.deque = collections.deque()
        if self.overlap:
            self.read_stream = torch.cuda.Stream(device)
            self.write_stream = torch.cuda.Stream(device)
            self.reader = ThreadPoolExecutor(1, thread_name_prefix="h2d")
            self.writer = ThreadPoolExecutor(1, thread_name_prefix="d2h")
            self.buffers: list[torch.Tensor] | None = None
            # event of the compute being done with each buffer (carried into the next `rows`)
            self.free: list[torch.cuda.Event | None] = [None, None]

    def chunks(self) -> list[tuple[int, int]]:
        r = self.h.shape[1]
        return [(r0, min(r0 + self.step, r)) for r0 in range(0, r, self.step)]

    def rows(self):
        """(r0, r1, h[:, r0:r1] on the device) per row chunk. A yielded tensor is valid only until
        the next chunk is requested."""
        chunks = self.chunks()
        if not self.overlap:
            for r0, r1 in chunks:
                yield r0, r1, self.h[:, r0:r1].to(self.device)
            return
        if self.buffers is None:
            b, _, c, e = self.h.shape
            shape = (b, min(self.step, self.h.shape[1]), c, e)
            self.buffers = [
                torch.empty(shape, dtype=self.h.dtype, device=self.device) for _ in range(2)
            ]
        free, buffers = self.free, self.buffers
        stream = torch.cuda.current_stream(self.device)
        # The writes of the previous block have reached the CPU (`flush`)
        futures = {0: self.reader.submit(self._read, *chunks[0], buffers[0], free[0])}
        for j, (r0, r1) in enumerate(chunks):
            if j + 1 < len(chunks):
                slot = (j + 1) % 2
                futures[j + 1] = self.reader.submit(
                    self._read, *chunks[j + 1], buffers[slot], free[slot]
                )
            stream.wait_event(futures.pop(j).result())
            yield r0, r1, buffers[j % 2][:, : r1 - r0]
            event = torch.cuda.Event()
            event.record(stream)
            free[j % 2] = event

    def _read(self, r0: int, r1: int, buffer: torch.Tensor, after) -> torch.cuda.Event:
        with (
            torch.inference_mode(self.inference),
            torch.cuda.device(self.device),
            torch.cuda.stream(self.read_stream),
        ):
            if after is not None:
                self.read_stream.wait_event(after)
            for i in range(self.h.shape[0]):
                buffer[i, : r1 - r0].copy_(self.h[i, r0:r1], non_blocking=True)
            event = torch.cuda.Event()
            event.record(self.read_stream)
        return event

    def write(self, r0: int, r1: int, src: torch.Tensor) -> None:
        """Write the device tensor src (B, r1 - r0, C, E) to h[:, r0:r1] (after the compute stream
        has produced src)."""
        if not self.overlap:
            self.h[:, r0:r1] = src.to(self.h.dtype)
            return
        ready = torch.cuda.Event()
        ready.record(torch.cuda.current_stream(self.device))
        while len(self.pending) >= self.WRITES_IN_FLIGHT:
            self.pending.popleft().result()
        self.pending.append(self.writer.submit(self._write, r0, r1, src, ready))

    def _write(self, r0: int, r1: int, src: torch.Tensor, ready: torch.cuda.Event) -> None:
        with (
            torch.inference_mode(self.inference),
            torch.cuda.device(self.device),
            torch.cuda.stream(self.write_stream),
        ):
            self.write_stream.wait_event(ready)
            src = src.to(self.h.dtype)
            for i in range(self.h.shape[0]):
                self.h[i, r0:r1].copy_(src[i], non_blocking=True)
            done = torch.cuda.Event()
            done.record(self.write_stream)
        # Keep src alive until it has reached the CPU (after that the compute stream reuses its
        # memory)
        done.synchronize()

    def flush(self) -> None:
        while self.pending:
            self.pending.popleft().result()

    def close(self) -> None:
        """Wait for the remaining copies and shut the threads down (also on the way out of an
        exception, so it raises no new exception)."""
        if not self.overlap:
            return
        while self.pending:
            try:
                self.pending.popleft().result()
            except Exception:  # noqa: BLE001 — the original exception says why
                pass
        self.reader.shutdown(wait=True)
        self.writer.shutdown(wait=True)
        with contextlib.suppress(RuntimeError):
            self.read_stream.synchronize()
            self.write_stream.synchronize()


@dataclass
class TrainContext:
    """What the test rows read from the train rows (`ColumnRowTransformer.train_context`), for B
    tables with the same train-row count and column count (one ensemble member per table):

    - `stats`: the train-row statistics of the cell normalization and the ECDF (`_CellStats`)
    - `summaries`: per column block, the inducing points' summary (B x C, n_inducing, E)
    - `icl_kv`: per row ICL layer, the first head's keys and values of the train rows
      (B, 1, n_train, head dim) that the test rows attend to (multi-query)
    - classification: the retrieval decoder's train-row keys (B, heads, n_train, head dim) and
      the one-hot labels it averages (B, n_train, classes)

    Tensors are batch first, except `summaries` (batch x columns first, batch major)."""

    task: str
    n_train: int
    n_columns: int
    stats: _CellStats
    summaries: list[torch.Tensor]
    icl_kv: list[tuple[torch.Tensor, torch.Tensor]]
    decoder_keys: torch.Tensor | None = None
    labels: torch.Tensor | None = None

    @property
    def batch(self) -> int:
        return self.stats.mean.shape[0]

    def _map(self, batch_first, per_column) -> TrainContext:
        # a new context with `batch_first` applied to the batch-first tensors and `per_column` to
        # the summaries
        def opt(t):
            return None if t is None else batch_first(t)

        return TrainContext(
            self.task,
            self.n_train,
            self.n_columns,
            self.stats.map(batch_first),
            [per_column(s) for s in self.summaries],
            [(batch_first(k), batch_first(v)) for k, v in self.icl_kv],
            opt(self.decoder_keys),
            opt(self.labels),
        )

    def select(self, start: int, stop: int) -> TrainContext:
        """Tables start..stop-1 (views, no copy)."""
        c = self.n_columns
        return self._map(lambda t: t[start:stop], lambda s: s[start * c : stop * c])

    @staticmethod
    def cat(contexts: list[TrainContext]) -> TrainContext:
        """The contexts stacked along the batch (they share the task, train-row and column
        counts)."""
        if len(contexts) == 1:
            return contexts[0]
        first = contexts[0]
        if any(
            (c.task, c.n_train, c.n_columns) != (first.task, first.n_train, first.n_columns)
            for c in contexts
        ):
            raise ValueError("only train contexts of the same task and table shape are stacked")

        def stack(get):
            parts = [get(c) for c in contexts]
            return None if parts[0] is None else torch.cat(parts)

        def field(name):
            return stack(lambda c: getattr(c.stats, name))

        return TrainContext(
            first.task,
            first.n_train,
            first.n_columns,
            _CellStats.from_fields({n: field(n) for n in _CellStats.FIELDS}),
            [stack(lambda c, i=i: c.summaries[i]) for i in range(len(first.summaries))],
            [
                (stack(lambda c, i=i: c.icl_kv[i][0]), stack(lambda c, i=i: c.icl_kv[i][1]))
                for i in range(len(first.icl_kv))
            ],
            stack(lambda c: c.decoder_keys),
            stack(lambda c: c.labels),
        )

    def tensors(self) -> list[torch.Tensor]:
        out = [getattr(self.stats, n) for n in _CellStats.FIELDS] + list(self.summaries)
        out += [t for kv in self.icl_kv for t in kv]
        return out + [t for t in (self.decoder_keys, self.labels) if t is not None]

    @property
    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.tensors())


class ColumnRowTransformer(nn.Module):
    # Model type (`tabsil.models.MODEL_TYPES`)
    model_type = "column_row_transformer"
    config_class = ColumnRowTransformerConfig
    # Chunk sizes of stages 0-2 at inference (`inference` of `forward`), the defaults of the
    # official TabPFN-3.5 inference code: the cells (batch x rows x columns) of one row chunk, and
    # the columns summarized at once from the train rows
    inference_chunk_cells = 2048 * 768
    inference_column_chunk = 4
    # When the cell state (B, R, C, E) of chunked inference (`_rows_interleaved_chunked`) is larger
    # than this, only the state of the row chunks with train rows is carried, and it is kept on the
    # CPU and moved to the device per chunk if it is also larger than this
    # (`_rows_interleaved_offload`; bounds the GPU memory on very large tables)
    interleaved_state_device_bytes = 16 << 30
    # Register the CPU cell state as pinned in place (`_host_state`; with False, copies go from
    # pageable memory. Same values)
    interleaved_state_pin = True
    # forward accepts the inference arguments (`inference`, `y_classes`/`n_classes`)
    # (`tabsil.inference.ensemble.forward_kwargs`)
    supports_inference_options = True
    # Activation checkpointing (tabsil-train config `TrainConfig.activation_checkpointing_rows`). In
    # training forwards on tables with more rows (train + test) than this, the activations of the
    # column/row stage blocks and the row ICL layers are recomputed in backward instead of kept
    # (`torch.utils.checkpoint`), and the classification decoder skips the score matrix
    # (`CHECKPOINT_FUSED_DECODER`). None turns it off (default). Set by the training loop, not the
    # model config, so it is not saved in checkpoint files or config.json
    activation_checkpointing_rows: int | None = None

    def __init__(self, config: ColumnRowTransformerConfig | None = None) -> None:
        super().__init__()
        self.config = config = config or ColumnRowTransformerConfig()
        # CPU cell state reused across forwards inside `hold_host_state` (1-D tensor,
        # registered flag)
        self._held_host_state: tuple[torch.Tensor, bool] | None = None
        self._holding_host_state = False
        e, g, d = config.emsize, config.features_per_group, config.icl_dim
        hidden = config.gate_hidden
        self.offsets = group_offsets(g)
        self.cell_encoder = FourierCellEncoder(g, e)
        # Target encoders: one for the cell stage and one for the row ICL input, per task
        self.cell_target = nn.ModuleDict({t: TargetEncoder(t, config.n_out, e) for t in TASKS})
        self.icl_target = nn.ModuleDict({t: TargetEncoder(t, config.n_out, d) for t in TASKS})
        # shared by the two tasks
        self.cell_target_norm = nn.LayerNorm(e)
        self.icl_target_norm = nn.LayerNorm(d)
        self.column_blocks = nn.ModuleList(
            InducedBlock(e, config.column_heads, config.n_inducing, config.ff_factor, hidden)
            for _ in range(config.column_blocks)
        )
        self.cls_tokens = nn.Parameter(
            nn.init.trunc_normal_(torch.empty(config.n_cls, e), std=0.02)
        )
        self.row_blocks = nn.ModuleList(
            Block(
                e,
                config.row_heads,
                config.ff_factor,
                rope_base=config.rope_base,
                zero_out=False,  # module docstring, choices made here
                temperature="gated_logscale",
                gate_hidden=hidden,
            )
            for _ in range(config.row_blocks)
        )
        self.row_norm = nn.RMSNorm(e)
        self.icl_layers = nn.ModuleList(
            Block(d, config.nhead, config.ff_factor, temperature="logscale", gate_hidden=hidden)
            for _ in range(config.nlayers)
        )
        self.icl_norm = nn.RMSNorm(d)
        self.pre_head = nn.ModuleDict({t: PreHeadMLP(d, config.ff_factor) for t in TASKS})
        self.heads = nn.ModuleDict(
            {
                "classification": RetrievalDecoder(
                    d,
                    config.decoder_heads,
                    config.decoder_head_dim,
                    config.output_size("classification"),
                    hidden,
                ),
                "regression": _quantile_head(d, config.output_size("regression")),
            }
        )

    @property
    def tasks(self) -> tuple[str, ...]:
        """The tasks this model outputs (picked by the `task` argument of forward)."""
        return TASKS

    @property
    def max_classes(self) -> int:
        """Class cap at inference (`MAX_CLASSES`, or the number of target embeddings if larger)."""
        return max(self.config.output_size("classification"), MAX_CLASSES)

    def _checkpointing(self, rows: int) -> bool:
        """Whether this forward uses activation checkpointing: it is on, the table has more rows
        than the threshold, and this is a training forward that computes gradients."""
        limit = self.activation_checkpointing_rows
        return limit is not None and rows > limit and self.training and torch.is_grad_enabled()

    def repeated_blocks(self) -> list[nn.Module]:
        """The repeated blocks (the `compile` speed-up option compiles each of them)."""
        return [*self.column_blocks, *self.row_blocks, *self.icl_layers]

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def parameter_stage(self, name: str) -> str:
        """The stage of a parameter name (`STAGES`). Used by the per-group norms of the training
        metrics (`training.diagnostics.parameter_groups`)."""
        return STAGES[name.split(".")[0]]

    def num_parameters_by_stage(self) -> dict[str, int]:
        """Parameter count per stage (in `STAGE_ORDER`): cell encoder (with its target embeddings),
        column stage, row stage, row ICL (with its target embeddings), output."""
        counts = dict.fromkeys(STAGE_ORDER, 0)
        for name, p in self.named_parameters():
            counts[self.parameter_stage(name)] += p.numel()
        return counts

    def num_parameters_by_task(self) -> dict[str, int]:
        """Parameter counts shared by the two tasks ("shared") and specific to each task. Parts
        whose name contains a task (target encoders, pre-head MLP, output layer) belong to it."""
        counts = {"shared": 0} | dict.fromkeys(TASKS, 0)
        for name, p in self.named_parameters():
            parts = name.split(".")
            task = next((t for t in TASKS if t in parts), "shared")
            counts[task] += p.numel()
        return counts

    def decoder_weight(self) -> torch.Tensor:
        """The training metric `decoder_norm`: the weight of the regression output layer."""
        return self.heads["regression"].weight

    def _embed_cells(self, x: torch.Tensor, n_train: int) -> torch.Tensor:
        # x: (B, R, C) -> tokens (B, R, C, E). The token of feature j holds the cells of columns
        # j + offsets
        z, indicator, used = standardize_cells(x, n_train)
        zg, ig = _group(z, self.offsets), _group(indicator, self.offsets)
        u = ecdf_midrank(x.float(), n_train).to(z.dtype)
        # Columns constant on the train rows: like z (set to 0), u is 0.5 on every row (the train
        # rows are all 0.5, so test rows getting 0 or 1 would be a test-only signal)
        u = torch.where(used > 0, u, 0.5)
        return self.cell_encoder(zg, ig, _group(u, self.offsets))

    def _target(self, stage: str, task: str, y_train: torch.Tensor, rows: int, like: torch.Tensor):
        # Train-row target embeddings (B, R, dim); 0 for the test rows
        target = getattr(self, f"{stage}_target")[task](y_train, like.dtype)
        target = getattr(self, f"{stage}_target_norm")(target)
        return F.pad(target, (0, 0, 0, rows - y_train.shape[1]))

    def forward(
        self,
        x: torch.Tensor,
        y_train: torch.Tensor,
        task: str,
        *,
        y_classes: torch.Tensor | None = None,
        n_classes: int | None = None,
        inference: bool = False,
    ) -> torch.Tensor:
        """Test-row logits (B, R - n_train, outputs).

        x: (B, R, C), the train rows followed by the test rows. y_train: (B, n_train). `task` is
        "classification" (y: class indices from 0) or "regression" (y: z-normalized values; the
        output is 999 quantiles, not sorted).

        Inference-only arguments:

        - `y_classes`/`n_classes` (classification): the train-row class indices the retrieval
          decoder averages, and the class count. When given, `y_train` is used only for the target
          embeddings. With more classes than target embeddings (`n_out`), pass the grouped
          embedding indices as `y_train` and the original indices as `y_classes`
          (`tabsil.inference.ensemble.EnsembleClassifier`). The output is (B, T, n_classes)
        - `inference`: reduces memory on large tables. With more cells (B x R x C) than
          `inference_chunk_cells`, stages 0-2 are computed in row chunks
          (`_rows_interleaved_chunked`), and the classification decoder skips the score matrix.
          Test rows do not see each other, so the values are the same (only the computation order
          differs)
        """
        self._check_targets(y_train, task, y_classes, n_classes)
        b, r, c = x.shape
        n_train = y_train.shape[1]
        ckpt = self._checkpointing(r)
        if inference and b * r * c > self.inference_chunk_cells:
            rows = self._rows_interleaved_chunked(x, y_train, task)
        else:
            rows = self._rows(x, y_train, task, ckpt=ckpt)

        # 3) Row ICL: add a second target embedding to the train row vectors
        rows = rows + self._target("icl", task, y_train, r, rows)
        for layer in self.icl_layers:
            rows = _call(layer, ckpt, rows, n_context=n_train)
        rows = self.icl_norm(rows)

        # 4) Output
        rows = self.pre_head[task](rows)
        if task == "classification":
            labels = y_train if y_classes is None else y_classes
            fused = inference or (ckpt and CHECKPOINT_FUSED_DECODER)
            return self.heads[task](rows, labels, n_train, n_classes=n_classes, fused=fused)
        return self.heads[task](rows[:, n_train:])

    def _check_targets(
        self,
        y_train: torch.Tensor,
        task: str,
        y_classes: torch.Tensor | None,
        n_classes: int | None,
    ) -> None:
        """The argument checks of `forward` (and `train_context`)."""
        cfg = self.config
        if task not in TASKS:
            raise ValueError(f"the task of this model is one of {TASKS}: {task}")
        if (y_classes is None) != (n_classes is None):
            raise ValueError("y_classes and n_classes must be given together")
        if n_classes is not None and n_classes > self.max_classes:
            raise ValueError(f"{n_classes} classes exceed the cap ({self.max_classes})")
        if task == "classification" and not self.training:
            # Indices beyond the target embeddings would be merged into the last class by the
            # embeddings and the decoder. In training the prior keeps labels in range, so (to
            # avoid a GPU sync) this is checked only in evaluation and inference
            n_labels = cfg.output_size("classification") if y_classes is None else n_classes
            labels = y_train if y_classes is None else y_classes
            if labels.numel() and int(labels.max()) >= n_labels:
                raise ValueError(
                    f"class index {int(labels.max())} is out of range for {n_labels} classes."
                    " With more classes than target embeddings"
                    f" ({cfg.output_size('classification')}), pass the grouped indices as y_train"
                    " and the original indices as y_classes/n_classes"
                )
            if y_classes is not None and int(y_train.max()) >= cfg.output_size("classification"):
                raise ValueError(
                    "with y_classes, y_train must hold grouped indices within the target embeddings"
                )

    # Inference with a train context: the train rows never see the test rows (module docstring:
    # the column summaries read train rows only, the spread and the row blocks work per row, and
    # train rows attend to train rows in the row ICL), so everything the test rows read from them
    # is computed once (`train_context`, at fit) and each prediction runs the test rows only
    # (`forward_test`). `tabsil.inference.ensemble` uses it when the model has this flag
    supports_train_context = True

    def train_context(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        task: str,
        *,
        y_classes: torch.Tensor | None = None,
        n_classes: int | None = None,
    ) -> TrainContext:
        """What the test rows of `forward(x, y_train, task, inference=True)` read from the train
        rows x_train (B, n_train, C) (`TrainContext`), for `forward_test`. The arguments are those
        of `forward` (inference). The train rows take the same path as in `forward`: in row chunks
        (and with the cell state on the CPU) when they are too many for the device."""
        self._check_targets(y_train, task, y_classes, n_classes)
        b, n, c = x_train.shape
        summaries: list[torch.Tensor] = []
        if b * n * c > self.inference_chunk_cells:
            rows = self._rows_interleaved_chunked(x_train, y_train, task, summaries=summaries)
        else:
            rows = self._rows(x_train, y_train, task, summaries=summaries)
        rows = rows + self._target("icl", task, y_train, n, rows)
        icl_kv = []
        for layer in self.icl_layers:
            rows, k, v = layer.forward_context(rows)
            icl_kv.append((k, v))
        keys = labels = None
        if task == "classification":
            decoder = self.heads[task]
            keys = decoder.keys(self.pre_head[task](self.icl_norm(rows)))
            width = n_classes or decoder.n_classes
            classes = y_train if y_classes is None else y_classes
            labels = F.one_hot(classes.long().clamp(0, width - 1), width).float()
        return TrainContext(task, n, c, _CellStats(x_train, n), summaries, icl_kv, keys, labels)

    def forward_test(self, x_test: torch.Tensor, context: TrainContext) -> torch.Tensor:
        """Test-row outputs (B, T, outputs), the values of `forward(x, y_train, task,
        inference=True)` for x = [train rows, x_test] (within float rounding: the train rows are
        computed in other shapes), from the train context (`train_context`). Test rows do not see
        each other, so they are computed `inference_chunk_cells / (B x C)` rows at a time."""
        b, t, c = x_test.shape
        if (b, c) != (context.batch, context.n_columns):
            raise ValueError(
                f"test rows (batch {b}, {c} columns) do not match the train context (batch"
                f" {context.batch}, {context.n_columns} columns)"
            )
        if t == 0:
            width = (
                context.labels.shape[-1]
                if context.task == "classification"
                else self.config.output_size("regression")
            )
            return x_test.new_zeros(b, 0, width)
        step = max(1, self.inference_chunk_cells // (b * c))
        parts = [self._test_rows(x_test[:, r0 : r0 + step], context) for r0 in range(0, t, step)]
        return parts[0] if len(parts) == 1 else torch.cat(parts, dim=1)

    def _test_rows(self, x: torch.Tensor, context: TrainContext) -> torch.Tensor:
        # `forward_test` on one chunk of test rows: stages 0-2 with the stored column summaries,
        # row ICL against the stored train keys/values, then the output head
        cfg, task = self.config, context.task
        b, t, _ = x.shape
        h = self._encode_cells(*context.stats.features(x, None))  # (B, t, C, E)
        cls = self.cls_tokens.to(h.dtype).expand(b, t, -1, -1).clone()  # (B, t, n_cls, E)
        last = len(self.row_blocks) - 1
        blocks = zip(self.column_blocks, self.row_blocks, context.summaries, strict=True)
        for i, (column, row, summary) in enumerate(blocks):
            new = self._interleaved_pair(column, row, h, summary, cls, 0, i == last)
            if new is not None:
                h = new.to(h.dtype)
        rows = self.row_norm(cls).reshape(b, t, cfg.icl_dim)
        for layer, (k, v) in zip(self.icl_layers, context.icl_kv, strict=True):
            rows = layer.forward_queries(rows, k, v, context.n_train)
        rows = self.pre_head[task](self.icl_norm(rows))
        if task == "classification":
            decoder = self.heads[task]
            return decoder.from_keys(rows, context.decoder_keys, context.labels, context.n_train)
        return self.heads[task](rows)

    def train_context_bytes(
        self,
        n_train: int,
        columns: int,
        *,
        classes: int | None,
        batch: int = 1,
        attention_itemsize: int = 2,
    ) -> int:
        """Bytes of the train context (`train_context`) of `batch` tables with `n_train` rows and
        `columns` columns: float32 column summaries and train-row statistics, and the row ICL
        keys/values in the attention dtype (`attention_itemsize`: 2 under bfloat16 autocast, 4 in
        float32). For classification (`classes`: the one-hot label width, at least the target
        embedding count `n_out`) also the decoder keys and the float32 one-hot labels; None for
        regression, which keeps neither."""
        cfg = self.config
        summaries = cfg.column_blocks * columns * cfg.n_inducing * cfg.emsize * 4
        stats = columns * (n_train * 4 + 3 * 4 + 8)  # sorted train values, mean/std/used, count
        icl = cfg.nlayers * 2 * n_train * (cfg.icl_dim // cfg.nhead) * attention_itemsize
        head = 0
        if classes is not None:
            decoder = n_train * cfg.decoder_heads * cfg.decoder_head_dim * attention_itemsize
            head = decoder + n_train * max(classes, cfg.n_out) * 4
        return batch * (summaries + stats + icl + head)

    def _rows(
        self,
        x: torch.Tensor,
        y_train: torch.Tensor,
        task: str,
        *,
        ckpt: bool = False,
        summaries: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Stages 0-2 (cell encoder, interleaved column and row stages): x (B, R, C) -> row vectors
        (B, R, n_cls x E). The cell tokens pass [column block i -> row block i] in turn; the CLS
        tokens carry over between row blocks and the last row block has only the CLS tokens as
        queries. With `ckpt`, each block uses activation checkpointing (`_call`). With
        `summaries` (a list, inference), each column block's summary (B x C, n_inducing, E) is
        appended to it."""
        cfg = self.config
        n_train = y_train.shape[1]
        cells = self._embed_cells(x, n_train)  # (B, R, C, E)
        h = cells + self._target("cell", task, y_train, x.shape[1], cells)[:, :, None]
        b, r, c, e = h.shape
        cls = self.cls_tokens.to(h.dtype).expand(b * r, -1, -1)
        last = len(self.row_blocks) - 1
        for i, (column, row) in enumerate(zip(self.column_blocks, self.row_blocks, strict=True)):
            hc = h.permute(0, 2, 1, 3).reshape(b * c, r, e)
            if summaries is None:
                hc = _call(column, ckpt, hc, n_train)
            else:  # the same as `column(hc, n_train)`, keeping the summary
                summaries.append(column.summary(hc[:, :n_train]))
                hc = column.spread(hc, summaries[-1])
            hr = hc.view(b, c, r, e).transpose(1, 2).reshape(b * r, c, e)
            tokens = torch.cat([cls.to(hr.dtype), hr], dim=1)
            if i == last:
                cls = _call(row, ckpt, tokens, n_queries=cfg.n_cls)
            else:
                tokens = _call(row, ckpt, tokens)
                cls, h = tokens[:, : cfg.n_cls], tokens[:, cfg.n_cls :].view(b, r, c, e)
        return self.row_norm(cls).reshape(b, r, cfg.icl_dim)

    def _chunk_embedder(self, x: torch.Tensor, y_train: torch.Tensor, task: str):
        """A function `embed` that builds the cell tokens of a slice of rows
        (`_rows_interleaved_chunked`). The train-row statistics (`_CellStats`) are computed once.
        `embed(x_part, rows=(r0, r1))` gives the tokens of rows r0..r1 and all columns, with the
        cell target embedding added to the train rows."""
        n_train = y_train.shape[1]
        stats = _CellStats(x, n_train)
        target = None  # train-row cell target embedding (B, n_train, E), built once the cell
        # dtype is known

        def embed(x_part, *, rows: tuple[int, int]):
            nonlocal target
            cells = self._encode_cells(*stats.features(x_part, None))
            if target is None:
                target = self._target("cell", task, y_train, n_train, cells)
            r0, r1 = rows
            if r0 < n_train:  # target embedding on the train rows only
                k = min(r1, n_train) - r0
                cells = torch.cat([cells[:, :k] + target[:, r0 : r0 + k, None], cells[:, k:]], 1)
            return cells

        return embed

    def _rows_interleaved_chunked(
        self,
        x: torch.Tensor,
        y_train: torch.Tensor,
        task: str,
        *,
        summaries: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """The values of `_rows`, computed in chunks (inference).

        The row blocks mix the columns, so the summary of the next column block depends on all
        train rows after the previous row block. One cell state (B, R, C, E) and one CLS state
        (B, R, n_cls, E) are therefore carried, and for each block pair (1) the columns are taken
        `inference_column_chunk` at a time and summarized from the current train-row state, then
        (2) the rows are taken `inference_chunk_cells / (B x C)` at a time through the spread and
        the row block, which update the state. All summaries are computed before (2), so they see
        the train-row state before it changes. The cell encoder also runs in row chunks. When the
        cell state is larger than `interleaved_state_device_bytes`, the computation takes another
        order that keeps only part of it, on the CPU if needed (`_rows_interleaved_offload`). With
        `summaries` (a list), each column block's summary (B x C, n_inducing, E) is appended to it
        (`train_context`)."""
        b, r, c = x.shape
        n_train = y_train.shape[1]
        embed = self._chunk_embedder(x, y_train, task)
        step = max(1, self.inference_chunk_cells // (b * c))
        device = x.device
        first = embed(x[:, :step], rows=(0, min(step, r)))
        shape = (b, r, c, first.shape[-1])
        if self._offload_interleaved_state(device, math.prod(shape) * first.element_size()):
            # The offload recomputes this chunk when it needs it: holding it for the whole forward
            # would keep a chunk of device memory (up to `inference_chunk_cells` x E) busy
            e, dtype = first.shape[-1], first.dtype
            del first
            return self._rows_interleaved_offload(
                x, n_train, embed, step, e, dtype, summaries=summaries
            )
        h = torch.empty(shape, dtype=first.dtype, device=device)
        for r0 in range(0, r, step):
            r1 = min(r0 + step, r)
            h[:, r0:r1] = first if r0 == 0 else embed(x[:, r0:r1], rows=(r0, r1))
        del first  # copied into h; do not keep a second chunk on the device through the blocks
        e = h.shape[-1]
        cls = self.cls_tokens.to(h.dtype).expand(b, r, -1, -1).clone()  # (B, R, n_cls, E)
        last = len(self.row_blocks) - 1
        blocks = zip(self.column_blocks, self.row_blocks, strict=True)
        for i, (column, row) in enumerate(blocks):
            # (1) Per-column summaries from the current train-row state (B, C, n_inducing, E)
            summary = None
            for c0 in range(0, c, self.inference_column_chunk):
                c1 = min(c0 + self.inference_column_chunk, c)
                ht = h[:, :n_train, c0:c1].permute(0, 2, 1, 3)
                ht = ht.reshape(b * (c1 - c0), n_train, e)
                ind = column.summary(ht)
                if summary is None:
                    summary = ind.new_empty(b, c, *ind.shape[1:])
                summary[:, c0:c1] = ind.view(b, c1 - c0, *ind.shape[1:])
            summary = summary.reshape(b * c, *summary.shape[2:])
            if summaries is not None:
                summaries.append(summary)
            # (2) Per row chunk: spread -> row block
            for r0 in range(0, r, step):
                r1 = min(r0 + step, r)
                new = self._interleaved_pair(column, row, h[:, r0:r1], summary, cls, r0, i == last)
                if new is not None:
                    h[:, r0:r1] = new.to(h.dtype)
        return self.row_norm(cls).reshape(b, r, self.config.icl_dim)

    def _interleaved_pair(
        self,
        column: InducedBlock,
        row: Block,
        h: torch.Tensor,
        summary: torch.Tensor,
        cls: torch.Tensor,
        r0: int,
        last: bool,
    ) -> torch.Tensor | None:
        """One block pair on a row chunk h (B, Rj, C, E) (`_rows_interleaved_chunked`): spread with
        the summary (B x C, n_inducing, E), then the row block. Updates the CLS state
        cls[:, r0:r0 + Rj] and returns the new cell state (B, Rj, C, E), or None for the last block
        (its queries are the CLS tokens only)."""
        cfg = self.config
        b, rj, c, e = h.shape
        hc = h.permute(0, 2, 1, 3).reshape(b * c, rj, e)
        hc = column.spread(hc, summary)
        hr = hc.view(b, c, rj, e).transpose(1, 2).reshape(b * rj, c, e)
        part = cls[:, r0 : r0 + rj].reshape(b * rj, cfg.n_cls, e)
        tokens = torch.cat([part.to(hr.dtype), hr], dim=1)
        if last:
            out = row(tokens, n_queries=cfg.n_cls)
            cls[:, r0 : r0 + rj] = out.view(b, rj, cfg.n_cls, e).to(cls.dtype)
            return None
        tokens = row(tokens)
        cls[:, r0 : r0 + rj] = tokens[:, : cfg.n_cls].view(b, rj, cfg.n_cls, e).to(cls.dtype)
        return tokens[:, cfg.n_cls :].view(b, rj, c, e)

    def _rows_interleaved_offload(
        self,
        x: torch.Tensor,
        n_train: int,
        embed,
        step: int,
        e: int,
        dtype: torch.dtype,
        *,
        summaries: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """`_rows_interleaved_chunked` when the cell state (B, R, C, E) does not fit on the device.
        The values are equal within numerical error (only the computation order of the column
        summaries differs). The row chunks are the same (`step` rows each); `e` and `dtype` are the
        width and dtype of the cell encoder output.

        - The column summaries are accumulated over chunks of train rows (`_OnlineSummary`): the
          summary of block i + 1 takes each train-row chunk produced by block pair i while it is
          still on the device (block 0 takes the cell encoder output). So the train rows are never
          read back from the CPU column by column (strided). All summaries of a block are done
          before block pair i + 1 starts, so they see the train-row state before it changes
        - Test rows are not used by the summaries and do not see each other. So only the chunks
          with train rows (the leading ones) keep a state across block pairs; each chunk without
          train rows goes through all block pairs on the device at once after the summaries are
          known (it never goes through the CPU)
        - The kept state (B, rows of the chunks with train rows, C, E) is kept on the CPU if it is
          larger than `interleaved_state_device_bytes`, and its reads and writes overlap the compute
          (`_HostRows`). The CPU state is allocated at its exact size (bypassing the caching host
          allocator) and released as soon as the forward ends, even on an exception
          (`_host_state`)
        - Block 0 does not read the state: it recomputes the cell encoder (same input and chunks, so
          the same values)
        - Without train rows, or with one block pair, nothing is kept (without train rows the
          summaries attend to no keys, as on the device)
        - With `summaries` (a list), the column summaries are appended to it, as in
          `_rows_interleaved_chunked`"""
        cfg = self.config
        b, r, c = x.shape
        device = x.device
        blocks = list(zip(self.column_blocks, self.row_blocks, strict=True))
        last = len(blocks) - 1
        chunks = [(r0, min(r0 + step, r)) for r0 in range(0, r, step)]
        n_held = self._held_rows(r, n_train, step)
        block_summaries = [] if summaries is None else summaries

        def cells(r0: int, r1: int) -> torch.Tensor:
            return embed(x[:, r0:r1], rows=(r0, r1))

        def add(acc: _OnlineSummary, new: torch.Tensor, r0: int) -> None:
            if r0 < n_train:  # the chunk's train rows (B, k, C, E) -> (B x C, k, E)
                k = min(r0 + new.shape[1], n_train) - r0
                acc.add(new[:, :k].to(dtype).permute(0, 2, 1, 3).reshape(b * c, k, e))

        shape = (b, n_held, c, e)
        release, rows = None, None
        if n_held and self._offload_interleaved_state(device, math.prod(shape) * dtype.itemsize):
            h, release = self._acquire_host_state(shape, dtype)
        else:
            h = torch.empty(shape, dtype=dtype, device=device)
        try:
            rows = _HostRows(h, step, device)
            cls = self.cls_tokens.to(dtype).expand(b, r, -1, -1).clone()  # (B, R, n_cls, E)
            acc = _OnlineSummary(blocks[0][0], b * c, n_train, dtype)
            for r0, r1 in chunks:
                if r0 < n_train:
                    add(acc, cells(r0, r1), r0)
            block_summaries.append(acc.finish())  # per block (B x C, n_inducing, E)
            # Chunks with train rows: every chunk, block pair by block pair
            for i, (column, row) in enumerate(blocks):
                acc = None if i == last else _OnlineSummary(blocks[i + 1][0], b * c, n_train, dtype)
                source = (
                    ((r0, r1, cells(r0, r1)) for r0, r1 in rows.chunks()) if i == 0 else rows.rows()
                )
                for r0, r1, part in source:
                    new = self._interleaved_pair(
                        column, row, part, block_summaries[i], cls, r0, i == last
                    )
                    if new is not None:
                        add(acc, new, r0)
                        rows.write(r0, r1, new)
                rows.flush()
                if acc is not None:
                    block_summaries.append(acc.finish())
            # Chunks without train rows: every block pair, chunk by chunk. The state between block
            # pairs is cast to the state dtype, as when it is kept
            for r0, r1 in chunks:
                if r0 < n_held:
                    continue
                part = cells(r0, r1)
                for i, (column, row) in enumerate(blocks):
                    part = self._interleaved_pair(
                        column, row, part, block_summaries[i], cls, r0, i == last
                    )
                    if part is not None:
                        part = part.to(dtype)
            return self.row_norm(cls).reshape(b, r, cfg.icl_dim)
        finally:
            # Even on an exception, stop the copies and unregister the pinned memory (while
            # registered, the OS cannot reclaim it)
            if rows is not None:
                rows.close()
            if release is not None:
                release()

    def _acquire_host_state(self, shape: tuple[int, ...], dtype: torch.dtype):
        """The CPU cell state, and a cleanup function to call when the forward ends (None:
        `hold_host_state` releases it). Outside `hold_host_state`, each forward allocates and
        releases its own. Inside, one 1-D buffer is kept and, if it fits (same dtype, enough
        elements), its front is reused as `shape`. Filling overwrites every row, so no values of an
        earlier forward remain."""
        if not self._holding_host_state:
            state, registered = _host_state(shape, dtype, self.interleaved_state_pin)
            return state, lambda: _release_host_state(state, registered)
        numel = math.prod(shape)
        held = self._held_host_state
        if held is not None and (held[0].dtype != dtype or held[0].numel() < numel):
            self._held_host_state = None
            _release_host_state(*held)
            held = None
        if held is None:
            held = _host_state((numel,), dtype, self.interleaved_state_pin)
            self._held_host_state = held
        return held[0][:numel].view(shape), None

    @contextlib.contextmanager
    def hold_host_state(self):
        """Inference forwards inside this context reuse the CPU cell state (offload of
        `_rows_interleaved_offload`) instead of releasing it, so that each ensemble member does not
        allocate hundreds of GB anew (page faults) and pin/unpin them (`tabsil.inference.ensemble`
        calls the member forwards inside it). Released on exit, even on an exception. Same values.
        When nested, the outermost context releases it."""
        if self._holding_host_state:
            yield
            return
        self._holding_host_state = True
        try:
            yield
        finally:
            self._holding_host_state = False
            held, self._held_host_state = self._held_host_state, None
            if held is not None:
                _release_host_state(*held)

    def _offload_interleaved_state(self, device: torch.device, size: int) -> bool:
        """Whether the cell state (`size` bytes) of `_rows_interleaved_chunked` is too large for the
        device (then `_rows_interleaved_offload`, which asks again for the state it carries)."""
        return device.type == "cuda" and size > self.interleaved_state_device_bytes

    def _held_rows(self, rows: int, n_train: int, step: int) -> int:
        """Rows of the cell state `_rows_interleaved_offload` carries across block pairs: the row
        chunks (`step` rows each) with train rows. With one block pair nothing is carried (block 0
        reads the cell encoder output and the last block writes no state)."""
        if len(self.row_blocks) < 2:
            return 0
        return min(rows, -(-n_train // step) * step)

    def inference_host_state_bytes(
        self,
        rows: int,
        columns: int,
        *,
        batch: int = 1,
        itemsize: int = 4,
        train_rows: int | None = None,
    ) -> int:
        """Bytes of cell state that one inference forward (`inference=True`, on a GPU) keeps on the
        CPU (offload of `_rows_interleaved_offload`; the eval queue uses it to choose which large
        tables run concurrently, `training/scripts/eval.py --host-memory-budget-gb`). `rows` are
        train + test rows; the cell state (B, R, C, E) has one token per column.

        0 when the forward is not chunked or the whole state fits on the device. Otherwise the
        offload carries only the row chunks with train rows (`_held_rows`), on the CPU only if that
        part does not fit on the device either. With `train_rows` (the forward's train rows, which
        come first) this is exact; without it, the whole state (an upper bound). `itemsize` is the
        size of the cell token dtype: 4, because the cell encoder outputs float32 even under
        bfloat16 autocast (so the state is float32 too; checked on H100)."""
        cells = batch * rows * columns
        if cells <= self.inference_chunk_cells:
            return 0
        token = self.config.emsize * itemsize
        if cells * token <= self.interleaved_state_device_bytes:
            return 0
        if train_rows is None:
            return cells * token
        step = max(1, self.inference_chunk_cells // (batch * columns))
        held = batch * self._held_rows(rows, train_rows, step) * columns * token
        return held if held > self.interleaved_state_device_bytes else 0

    def _encode_cells(self, z, indicator, u) -> torch.Tensor:
        """Cell values (`_CellStats.features`) of all C columns -> tokens (B, rows, C, E); the
        columns are grouped by circular shift (`_group`)."""
        return self.cell_encoder(*(_group(t, self.offsets) for t in (z, indicator, u)))
