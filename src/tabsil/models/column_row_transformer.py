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

Attention temperatures (Kumo-Tabular, NVIDIA structured-data-models `sdm/nn/scaling.py`, Apache
2.0, reimplemented from its behavior): q and k get an RMSNorm without gain, and q is multiplied by a
per-head scalar times log n (n = number of keys), so that the logits are bounded by
sqrt(head dim) * a_h * log n (`LogScale`):
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
from collections.abc import Mapping
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
# Whether the classification retrieval decoder skips the score matrix (`fused` of
# `RetrievalDecoder`) in training forwards with activation checkpointing on
# (`ColumnRowTransformer.activation_checkpointing_rows`). On a 60,000-row table that matrix is
# 13 GB in float32, with several more copies through the softmax. Under bf16 autocast the
# score-matrix path computes q·k in bf16 and this path in float32, so the values differ slightly.
# A switch for comparing the two in measurements
CHECKPOINT_FUSED_DECODER = True

# Fields that minipfn's config of this model had and TabSil fixes: name -> (minipfn default, the
# only value TabSil accepts). minipfn checkpoints, Hub config.json files and MLflow run params may
# still hold them (`fixed_fields_check`). minipfn left a field at its default out of the saved dict
# when the field was added after the first release, hence the defaults
FIXED_FIELDS: dict[str, tuple[Any, Any]] = {
    "task": ("classification", "multitask"),
    "qk_norm": (False, True),
    "encoder": ("linear", "fourier"),
    "target_norm": (False, True),
    "head_mlp": (False, True),
    "n_buckets": (5000, 5000),
    "logit_softcap": (None, None),
    "qk_norm_affine": (True, False),
    "column_summary_temperature": ("qassmax", "logscale"),
    "column_spread_temperature": ("none", "none"),
    "row_temperature": ("none", "gated_logscale"),
    "icl_temperature": ("qassmax", "logscale"),
    "decoder_temperature": ("qassmax", "logscale"),
    "decoder_qk_norm": (False, False),
    "stage_layout": ("sequential", "interleaved"),
    "regression_head": ("bar", "quantile"),
    "categorical_encoder": (False, False),
}
# minipfn always wrote this field, TabSil never does: its presence marks a minipfn dict
LEGACY_MARKER = "task"
# Fields renamed since minipfn: old name -> current name
RENAMED_FIELDS = {"qassmax_hidden": "gate_hidden"}


def fixed_fields_check(values: dict, *, as_text: bool = False) -> None:
    """Remove the fields TabSil fixes (`FIXED_FIELDS`) from `values` in place, and raise a
    ValueError if one of them differs from the TabSil value.

    `values` is a saved model config dict or the MLflow params of a training run (`as_text`: the
    values are strings). A minipfn dict or run (it has `LEGACY_MARKER`) that lacks one of the fields
    had it at minipfn's default; a dict without the marker is a TabSil one and is left as it is."""
    if LEGACY_MARKER not in values:
        return
    wrong = {}
    for name, (old_default, fixed) in FIXED_FIELDS.items():
        value = values.pop(name, str(old_default) if as_text else old_default)
        if value != (str(fixed) if as_text else fixed):
            wrong[name] = (value, fixed)
    if wrong:
        raise ValueError(
            "this minipfn model is not the TabSil structure; TabSil fixes these fields"
            f" (saved value, TabSil value): {wrong}"
        )


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
    """Model size. The defaults are the published TabSil model (`tabsil.recipe.STRUCTURE`)."""

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
    # hidden width of the gate MLP of the gated_logscale temperature (minipfn: `qassmax_hidden`)
    gate_hidden: int = 64
    # number of target embeddings of the classification output (the class cap in training)
    n_out: int = 10
    # feature cap seen in training (`TrainConfig.max_features`). Not part of the structure;
    # read by inference
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
    def from_dict(cls, values: Mapping[str, Any]) -> ColumnRowTransformerConfig:
        """The config of a saved dict: a TabSil one, or a minipfn one of the same structure (the
        fields TabSil fixes must hold the TabSil value, `fixed_fields_check`). Missing fields take
        the defaults; a field this code does not know raises a ValueError naming it."""
        values = dict(values)
        fixed_fields_check(values)
        for old, new in RENAMED_FIELDS.items():
            if old in values:
                values[new] = values.pop(old)
        unknown = sorted(set(values) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(
                "the column_row_transformer model config has fields this code does not know:"
                f" {unknown} (the checkpoint may have been saved by newer code or another branch)"
            )
        return cls(**values)

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
    (1 + tanh(gate(q))) (the gate starts at 0, so the first factor is 1)."""

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
    - When DiagScale (`tabsil.inference.diag_scale`) attaches the parameter `diag_scale` (δ,
      heads x head dim), the q·k similarity becomes qᵀ(I + diag(δ))k: δ multiplies q after the
      RMSNorm, RoPE and temperature. Without it the computation is unchanged
    """

    # q, k and v come from the `qkv` weight rather than its forward, so DiagScale's δ is applied
    # here (`tabsil.inference.diag_scale.add_diag_scale` does not hook the forward of qkv)
    diag_scale_in_forward = True

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
        delta = getattr(self, "diag_scale", None)  # δ attached by DiagScale (class docstring)
        if delta is not None:
            h, d = q.shape[1], q.shape[3]
            q = (q.float() * (1.0 + delta.view(1, h, 1, d))).to(q.dtype)
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


def _retrieve(q: torch.Tensor, k: torch.Tensor, onehot: torch.Tensor) -> torch.Tensor:
    """softmax(q·kᵀ/√d) @ onehot per head (B, H, T, n_classes), in float32 and without the score
    matrix.

    q (B, H, T, d), k (B, H, N, d), onehot (B, N, n_classes). To keep the value width of the
    attention kernel equal to that of q and k, the one-hot labels are cut into pieces of d columns,
    each with its own attention (the method of `_chunked_class_attention` in the official
    TabPFN-3.5 inference code). q and k are the same for every piece and are not copied. Computed
    in float32 with autocast off."""
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
    """The CPU cell state (offload of `ColumnRowTransformer._rows_interleaved_chunked`): a regular
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
    freed). All copies are synchronous and already done, but the device is synchronized first in
    case work is still pending. Also called on the way out of an exception (`finally`), so it raises
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


class ColumnRowTransformer(nn.Module):
    # Model type (`tabsil.models.MODEL_TYPES`)
    model_type = "column_row_transformer"
    config_class = ColumnRowTransformerConfig
    # Chunk sizes of stages 0-2 at inference (`inference` of `forward`), the defaults of the
    # official TabPFN-3.5 inference code: the cells (batch x rows x columns) of one row chunk, and
    # the columns summarized at once from the train rows
    inference_chunk_cells = 2048 * 768
    inference_column_chunk = 4
    # The cell state (B, R, C, E) that chunked inference (`_rows_interleaved_chunked`) carries is
    # kept on the CPU and moved to the device per chunk when it is larger than this (bounds the GPU
    # memory on very large tables)
    interleaved_state_device_bytes = 16 << 30
    # Register the CPU cell state as pinned in place (`_host_state`; with False, copies go from
    # pageable memory. Same values)
    interleaved_state_pin = True
    # forward accepts the inference arguments (`inference`, `y_classes`/`n_classes`)
    # (`tabsil.inference.ensemble.forward_kwargs`)
    supports_inference_options = True
    # Activation checkpointing (training config `TrainConfig.activation_checkpointing_rows`). In
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

    def _rows(
        self, x: torch.Tensor, y_train: torch.Tensor, task: str, *, ckpt: bool = False
    ) -> torch.Tensor:
        """Stages 0-2 (cell encoder, interleaved column and row stages): x (B, R, C) -> row vectors
        (B, R, n_cls x E). The cell tokens pass [column block i -> row block i] in turn; the CLS
        tokens carry over between row blocks and the last row block has only the CLS tokens as
        queries. With `ckpt`, each block uses activation checkpointing (`_call`)."""
        cfg = self.config
        n_train = y_train.shape[1]
        cells = self._embed_cells(x, n_train)  # (B, R, C, E)
        h = cells + self._target("cell", task, y_train, x.shape[1], cells)[:, :, None]
        b, r, c, e = h.shape
        cls = self.cls_tokens.to(h.dtype).expand(b * r, -1, -1)
        last = len(self.row_blocks) - 1
        for i, (column, row) in enumerate(zip(self.column_blocks, self.row_blocks, strict=True)):
            hc = h.permute(0, 2, 1, 3).reshape(b * c, r, e)
            hc = _call(column, ckpt, hc, n_train)
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
        self, x: torch.Tensor, y_train: torch.Tensor, task: str
    ) -> torch.Tensor:
        """The values of `_rows`, computed in chunks (inference).

        The row blocks mix the columns, so the summary of the next column block depends on all
        train rows after the previous row block. One cell state (B, R, C, E) and one CLS state
        (B, R, n_cls, E) are therefore carried, and for each block pair (1) the columns are taken
        `inference_column_chunk` at a time and summarized from the current train-row state, then
        (2) the rows are taken `inference_chunk_cells / (B x C)` at a time through the spread and
        the row block, which update the state. All summaries are computed before (2), so they see
        the train-row state before it changes. The cell encoder also runs in row chunks. A cell
        state larger than `interleaved_state_device_bytes` is kept on the CPU, allocated at its
        exact size (bypassing the caching host allocator) and released as soon as the forward ends,
        even on an exception (`_host_state`)."""
        cfg = self.config
        b, r, c = x.shape
        n_train = y_train.shape[1]
        embed = self._chunk_embedder(x, y_train, task)
        step = max(1, self.inference_chunk_cells // (b * c))
        device = x.device
        h, release = None, None
        try:
            for r0 in range(0, r, step):
                r1 = min(r0 + step, r)
                cells = embed(x[:, r0:r1], rows=(r0, r1))
                if h is None:
                    shape = (b, r, c, cells.shape[-1])
                    size = math.prod(shape) * cells.element_size()
                    if self._offload_interleaved_state(device, size):
                        h, release = self._acquire_host_state(shape, cells.dtype)
                    else:
                        h = torch.empty(shape, dtype=cells.dtype, device=device)
                h[:, r0:r1] = cells
            e = h.shape[-1]
            cls = self.cls_tokens.to(h.dtype).expand(b, r, -1, -1).clone()  # (B, R, n_cls, E)
            last = len(self.row_blocks) - 1
            blocks = zip(self.column_blocks, self.row_blocks, strict=True)
            for i, (column, row) in enumerate(blocks):
                # (1) Per-column summaries from the current train-row state
                # (B, C, n_inducing, E)
                summary = None
                for c0 in range(0, c, self.inference_column_chunk):
                    c1 = min(c0 + self.inference_column_chunk, c)
                    ht = h[:, :n_train, c0:c1].to(device).permute(0, 2, 1, 3)
                    ht = ht.reshape(b * (c1 - c0), n_train, e)
                    ind = column.summary(ht)
                    if summary is None:
                        summary = ind.new_empty(b, c, *ind.shape[1:])
                    summary[:, c0:c1] = ind.view(b, c1 - c0, *ind.shape[1:])
                summary = summary.reshape(b * c, *summary.shape[2:])
                # (2) Per row chunk: spread -> row block
                for r0 in range(0, r, step):
                    r1 = min(r0 + step, r)
                    rj = r1 - r0
                    hc = h[:, r0:r1].to(device).permute(0, 2, 1, 3).reshape(b * c, rj, e)
                    hc = column.spread(hc, summary)
                    hr = hc.view(b, c, rj, e).transpose(1, 2).reshape(b * rj, c, e)
                    part = cls[:, r0:r1].reshape(b * rj, cfg.n_cls, e)
                    tokens = torch.cat([part.to(hr.dtype), hr], dim=1)
                    if i == last:
                        out = row(tokens, n_queries=cfg.n_cls)
                        cls[:, r0:r1] = out.view(b, rj, cfg.n_cls, e).to(cls.dtype)
                    else:
                        tokens = row(tokens)
                        new_cls = tokens[:, : cfg.n_cls].view(b, rj, cfg.n_cls, e)
                        cls[:, r0:r1] = new_cls.to(cls.dtype)
                        h[:, r0:r1] = tokens[:, cfg.n_cls :].view(b, rj, c, e).to(h.dtype)
            return self.row_norm(cls).reshape(b, r, cfg.icl_dim)
        finally:
            # Unregister the pinned memory even on an exception (while registered, the OS
            # cannot reclaim it)
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
        `_rows_interleaved_chunked`) instead of releasing it, so that each ensemble member does not
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
        """Whether to keep the cell state (`size` bytes) of `_rows_interleaved_chunked` on the
        CPU."""
        return device.type == "cuda" and size > self.interleaved_state_device_bytes

    def inference_host_state_bytes(
        self, rows: int, columns: int, *, batch: int = 1, itemsize: int = 4
    ) -> int:
        """Bytes of cell state that one inference forward (`inference=True`) keeps on the CPU
        (offload of `_rows_interleaved_chunked`; the eval queue uses it to choose which large tables
        run concurrently, `scripts/eval.py --host-memory-budget-gb`). The cell state (B, R, C, E)
        has one token per column. 0 when the forward is not chunked or the state stays on the
        device. `itemsize` is the size of the cell token dtype: 4, because the cell encoder outputs
        float32 even under bfloat16 autocast (so the state is float32 too; checked on H100)."""
        cells = batch * rows * columns
        if cells <= self.inference_chunk_cells:
            return 0
        size = cells * self.config.emsize * itemsize
        return size if size > self.interleaved_state_device_bytes else 0

    def _encode_cells(self, z, indicator, u) -> torch.Tensor:
        """Cell values (`_CellStats.features`) of all C columns -> tokens (B, rows, C, E); the
        columns are grouped by circular shift (`_group`)."""
        return self.cell_encoder(*(_group(t, self.offsets) for t in (z, indicator, u)))
