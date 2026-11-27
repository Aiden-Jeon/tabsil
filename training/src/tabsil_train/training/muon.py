"""Muon on every parameter (TabSil's optimizer; TabICLv2 training scripts' `--muon True`), or
Muon on the weight matrices and AdamW on the rest (`TrainConfig.optimizer="muon_adamw"`).

Muon (Jordan et al. 2024) updates 2-D weight matrices with their gradient momentum
orthogonalized by Newton–Schulz iterations. TabICLv2 (arXiv 2602.11139), EXAONE Tabular and
Mitra-v2 use it instead of (or alongside) AdamW.

- Every parameter is Muon's. A parameter that is not 2-D is flattened to a (first dim, product of
  the rest) matrix as in TabICL `Muon` ((n,) -> (n, 1), whose orthogonalization is a unit vector;
  0-D -> (1, 1)), and the lr correction uses that matrix size. The weight decay covers every
  parameter
- A fused q·k·v weight (`qkv` Linear, (3D, D)) is split into three along dim 0 and each part is
  orthogonalized separately (as Kimi/Moonshot do; TabICL does not split). Kept fused, the three
  would share one orthogonal matrix and each update would shrink, and the lr correction would use
  the (3D, D) size instead of (D, D)
- lr: Moonshot's "match_rms_adamw" (arXiv 2502.16982, `0.2·sqrt(max(A, B))`, using the size of
  one piece for a split matrix). Matching the update size to the AdamW RMS lets us use the same
  lr, warmup and cosine as Adam
- The Muon computation matches `torch.optim.Muon` (PyTorch 2.11; 5 Newton–Schulz steps in bf16,
  Nesterov). Unsplit matrices are bit-identical (tested). The step lives here only to add the
  split

Muon with AdamW (`optimizer="muon_adamw"`, `Muon.for_model(..., adamw=True)`): Kumo-Tabular's split
(NVIDIA structured-data-models `train/kumo/tabular/optim.py` `configure_optimizers`, Apache-2.0;
see NOTICE). Muon takes the weights of the `nn.Linear` layers, except those of the target
embeddings, the output layers and the attention logit scales; AdamW takes everything else
(`adamw_parameters`):

- Muon: every `nn.Linear` weight outside `ADAMW_MODULES` and the `LogScale` modules
- AdamW with weight decay `ADAMW_WEIGHT_DECAY`: the `LogScale` parameters (per-head scales and their
  query gates) and the other parameters with at least 2 dimensions (embeddings, inducing points,
  CLS tokens, the Linear weights of `ADAMW_MODULES`)
- AdamW without weight decay: the remaining 1-D parameters (biases, norms, Fourier frequencies)

`ADAMW_MODULES` are TabSil's counterparts of Kumo's `y_emb`, `y_lin` and `head`: the target encoders
of the cell stage and the row ICL (`cell_target`, `icl_target`), and the output (`pre_head`, the
residual MLP before the output layer, and `heads`, the classification retrieval decoder and the
regression quantile layer). AdamW is `torch.optim.AdamW`'s computation with its defaults (betas
0.9 and 0.999, eps 1e-8, decoupled weight decay 0.01), as Kumo uses it. Both parts share the lr and
its schedule, as in Kumo (one `LambdaLR` for both). The AdamW groups live in the same optimizer as
extra param groups (`adamw` key), so the scheduler, the checkpoint and the collapse rewinds treat
it as one optimizer; the Muon weight decay setting (`muon_weight_decay`, its phases) changes only
the Muon groups, and the AdamW decay stays 0.01. Kumo's Muon uses `torch.optim.Muon`'s momentum
0.95 (TabSil's default is 0.9, `TrainConfig.muon_momentum`).

Distributed Muon (training option `muon_distributed`): by default every GPU computes the same
Newton–Schulz for every matrix. With `process_group`, the GPUs of that group split the matrix
pieces (q·k·v split pieces as units, `shard_pieces`), each orthogonalizes its own share, and the
results (bf16) are all-gathered (the same idea as the distributed Muon of Moonlight/Kimi). The
momentum buffers stay complete on every GPU (unchanged checkpoint format, so runs can resume from
checkpoints saved without sharding). Received pieces are restored to tensors with the same values
and strides as the single-GPU update before being applied, so the weights are bit-identical to
the single-GPU computation (tested)
"""

from __future__ import annotations

import math
import os

import torch
import torch.distributed as dist
from torch import nn

MUON_MOMENTUM = 0.95  # the `torch.optim.Muon` default (TabSil trains with 0.9, `TrainConfig`)
# quintic Newton–Schulz coefficients and step count (Keller Jordan; same as `torch.optim.Muon`)
NS_COEFFICIENTS = (3.4445, -4.7750, 2.0315)
NS_STEPS = 5
NS_EPS = 1e-7
# AdamW of the `muon_adamw` optimizer: `torch.optim.AdamW` defaults, as Kumo-Tabular uses them
ADAMW_BETAS = (0.9, 0.999)
ADAMW_EPS = 1e-8
ADAMW_WEIGHT_DECAY = 0.01
# Top-level modules whose parameters go to AdamW (module docstring): target encoders and output
ADAMW_MODULES = ("cell_target", "icl_target", "pre_head", "heads")


def muon_parameters(model: nn.Module) -> list[tuple[nn.Parameter, int]]:
    """[(parameter, pieces to split it into)] in `model.parameters()` order. The pieces are 3 for a
    weight that holds q, k and v together (`qkv` Linears of the repeated blocks,
    `model.repeated_blocks()`, and `nn.MultiheadAttention` input projections) and 1 otherwise."""
    blocks = model.repeated_blocks() if hasattr(model, "repeated_blocks") else []
    splits: dict[int, int] = {}
    for block in blocks:
        for name, module in block.named_modules():
            if isinstance(module, nn.Linear) and min(module.weight.shape) > 1:
                qkv = name.rsplit(".", 1)[-1] == "qkv" and module.out_features % 3 == 0
                splits[id(module.weight)] = 3 if qkv else 1
            elif isinstance(module, nn.MultiheadAttention) and module.in_proj_weight is not None:
                splits[id(module.in_proj_weight)] = 3
    return [(p, splits.get(id(p), 1)) for p in model.parameters()]


def adamw_parameters(model: nn.Module) -> tuple[set[int], set[int]]:
    """(ids of the parameters AdamW takes with weight decay, ids of those it takes without) for
    `optimizer="muon_adamw"` (module docstring); the other parameters are Muon's. ValueError if the
    model lacks one of `ADAMW_MODULES` (the split would silently differ from the documented one)."""
    from tabsil.models.column_row_transformer import LogScale

    children = dict(model.named_children())
    missing = [name for name in ADAMW_MODULES if name not in children]
    if missing:
        raise ValueError(f"the muon_adamw optimizer needs the modules {missing} of TabSil")
    adamw = {id(p) for name in ADAMW_MODULES for p in children[name].parameters()}
    scales = {id(p) for m in model.modules() if isinstance(m, LogScale) for p in m.parameters()}
    muon = {id(m.weight) for m in model.modules() if isinstance(m, nn.Linear)} - adamw - scales
    decay, no_decay = set(), set()
    for p in model.parameters():
        if id(p) in muon:
            continue
        (decay if id(p) in scales or p.ndim >= 2 else no_decay).add(id(p))
    return decay, no_decay


def _matrix(t: torch.Tensor) -> torch.Tensor:
    """The 2-D view to orthogonalize: (first dim, product of the rest); 0-D is (1, 1) (like
    `reshape(len(g), -1)` in TabICL `Muon`). Must be a view since parameters change in place."""
    if t.ndim == 2:
        return t
    return t.view(1, 1) if t.ndim == 0 else t.view(t.shape[0], -1)


def _newton_schulz(grad: torch.Tensor) -> torch.Tensor:
    """Push a 2-D matrix toward orthogonal (as `torch.optim._muon._zeropower_via_newtonschulz`)."""
    a, b, c = NS_COEFFICIENTS
    x = grad.bfloat16()
    tall = grad.size(0) > grad.size(1)
    if tall:
        x = x.T
    x.div_(x.norm().clamp(min=NS_EPS))  # spectral norm at most 1
    for _ in range(NS_STEPS):
        gram = x @ x.T
        x = torch.addmm(x, torch.addmm(gram, gram, gram, beta=b, alpha=c), x, beta=a)
    return x.T if tall else x


def _nesterov(grad: torch.Tensor, buf: torch.Tensor, group: dict) -> torch.Tensor:
    """The Nesterov update to orthogonalize."""
    return grad.lerp(buf, group["momentum"])


def _apply(p: torch.Tensor, update: torch.Tensor, group: dict) -> None:
    """Apply weight decay and the orthogonalized update, with the lr corrected by the piece size
    (match_rms_adamw). Non-2-D parameters are seen as their flattened matrix (`_matrix`)."""
    p = _matrix(p)
    rows, cols = p.shape[0] // group["split"], p.shape[1]
    lr = group["lr"]
    p.mul_(1 - lr * group["weight_decay"])
    p.add_(update, alpha=-(lr * (0.2 * math.sqrt(max(rows, cols)))))


def _piece_cost(rows: int, cols: int) -> int:
    """Newton–Schulz matmul cost of one piece (per iteration, short side m, long side n:
    m²·(2n + m))."""
    m, n = min(rows, cols), max(rows, cols)
    return m * m * (2 * n + m)


def shard_pieces(shapes: list[tuple[int, int]], n_ranks: int) -> list[int]:
    """The rank that owns each (rows, cols) piece. Largest cost first, each to the rank with the
    least load so far (LPT; ties go to the lower rank). Integer-only, so every GPU agrees."""
    loads, owner = [0] * n_ranks, [0] * len(shapes)
    for i in sorted(range(len(shapes)), key=lambda i: (-_piece_cost(*shapes[i]), i)):
        rank = min(range(n_ranks), key=lambda r: (loads[r], r))
        owner[i] = rank
        loads[rank] += _piece_cost(*shapes[i])
    return owner


def shard_group(scope: str):
    """The process group distributed Muon splits matrices over. "all": every GPU; "node": the GPUs
    of a node (one group per node; across nodes the computation is duplicated). Every GPU must call
    it. None with a single GPU (no splitting)."""
    if scope not in ("node", "all"):
        raise ValueError(f"distributed Muon scope must be node or all: {scope}")
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return None
    if scope == "all":
        return dist.group.WORLD
    # set by torchrun; otherwise (another launcher) use this node's GPU count. Defaulting to the
    # world size would silently turn node into all, sending all-gathers across nodes
    local = int(os.environ.get("LOCAL_WORLD_SIZE") or torch.cuda.device_count() or 1)
    if dist.get_world_size() % local:
        raise ValueError(f"cannot split {dist.get_world_size()} GPUs into {local} per node")
    if local == dist.get_world_size():
        return dist.group.WORLD
    group, _ = dist.new_subgroups(group_size=local)
    return group


class Muon(torch.optim.Optimizer):
    """The `torch.optim.Muon` computation (Nesterov, match_rms_adamw). With a group `split` above
    1, the weight is split that many ways along dim 0, each piece orthogonalized separately and the
    lr corrected by the piece size.

    With `process_group`, the GPUs of that group split the pieces (module docstring, distributed
    Muon).
    `Muon.for_model` builds TabSil's optimizer (every parameter of a model)."""

    def __init__(
        self,
        params,
        lr: float,
        *,
        weight_decay: float = 0.0,
        split: int = 1,
        momentum: float = MUON_MOMENTUM,
        any_shape: bool = False,
        process_group=None,
    ) -> None:
        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "momentum": momentum,
            "split": split,
        }
        super().__init__(params, defaults)
        for group in self.param_groups:
            if group.get("adamw"):
                continue
            for p in group["params"]:
                # with any_shape, non-2-D parameters are accepted as flattened matrices
                shape = _matrix(p).shape if any_shape else p.shape
                if len(shape) != 2 or shape[0] % group["split"]:
                    raise ValueError(
                        f"Muon takes only 2-D matrices splittable into {group['split']} pieces"
                    )
        if process_group is not None and dist.get_world_size(process_group) == 1:
            process_group = None  # no GPUs to split over
        self.process_group = process_group
        # piece owners per (group size, set of parameters with gradients)
        self._owners: dict[tuple, list[int]] = {}

    @classmethod
    def for_model(
        cls,
        model: nn.Module,
        lr: float,
        *,
        weight_decay: float = 0.0,
        momentum: float = MUON_MOMENTUM,
        process_group=None,
        adamw: bool = False,
    ) -> Muon:
        """Muon on every parameter of `model` (module docstring), one param group per split count
        (`muon_parameters`). With `adamw`, the parameters of `adamw_parameters` go to two AdamW
        groups instead (with and without weight decay), after the Muon groups."""
        muon = muon_parameters(model)
        decay, no_decay = adamw_parameters(model) if adamw else (set(), set())
        muon = [(p, s) for p, s in muon if id(p) not in decay | no_decay]
        groups = [
            {"params": [p for p, s in muon if s == split], "split": split}
            for split in sorted({s for _, s in muon})
        ]
        for ids, decay_value in ((decay, ADAMW_WEIGHT_DECAY), (no_decay, 0.0)):
            params = [p for p in model.parameters() if id(p) in ids]
            if params:
                groups.append(
                    {
                        "params": params,
                        "adamw": True,
                        "weight_decay": decay_value,
                        "betas": ADAMW_BETAS,
                        "eps": ADAMW_EPS,
                    }
                )
        return cls(
            groups,
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            any_shape=True,
            process_group=process_group,
        )

    def load_state_dict(self, state_dict: dict) -> None:
        """Load a state; the group settings other than lr (weight decay, momentum) stay as this
        optimizer has them (lr is set again by the training loop, `_set_base_lr`)."""
        if "param_groups" not in state_dict:
            raise ValueError("not a Muon optimizer state")
        settings = [
            {k: v for k, v in group.items() if k not in ("params", "lr", "initial_lr")}
            for group in self.param_groups
        ]
        super().load_state_dict(state_dict)
        for group, kept in zip(self.param_groups, settings, strict=True):
            group.update(kept)

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise ValueError("Muon does not take a closure")
        for group in self.param_groups:
            if group.get("adamw"):
                self._adamw_step(group)
        if self.process_group is not None:
            self._step_sharded()
            return
        for group in self.param_groups:
            if group.get("adamw"):
                continue
            split = group["split"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                buf = self._momentum(p, group)
                update = _matrix(_nesterov(p.grad, buf, group))
                rows = update.shape[0] // split
                if split == 1:
                    update = _newton_schulz(update)
                else:
                    update = torch.cat([_newton_schulz(u) for u in update.split(rows)])
                _apply(p, update, group)

    def _adamw_step(self, group: dict) -> None:
        """`torch.optim.AdamW`'s update (single-tensor form, decoupled weight decay, bias
        correction) for the parameters of an AdamW group that have a gradient. Every GPU computes
        it (it is elementwise and cheap)."""
        beta1, beta2 = group["betas"]
        lr, eps, decay = group["lr"], group["eps"], group["weight_decay"]
        for p in group["params"]:
            if p.grad is None:
                continue
            state = self.state[p]
            if "step" not in state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)
            state["step"] += 1
            step, grad = state["step"], p.grad
            p.mul_(1 - lr * decay)
            state["exp_avg"].lerp_(grad, 1 - beta1)
            state["exp_avg_sq"].mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
            correction1 = 1 - beta1**step
            correction2 = 1 - beta2**step
            denom = (state["exp_avg_sq"].sqrt() / math.sqrt(correction2)).add_(eps)
            p.addcdiv_(state["exp_avg"], denom, value=-lr / correction1)

    def _momentum(self, p: torch.Tensor, group: dict) -> torch.Tensor:
        """Update the momentum buffer with this gradient and return it (shared by both paths)."""
        state = self.state[p]
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros_like(p.grad)
        buf = state["momentum_buffer"]
        buf.lerp_(p.grad, 1 - group["momentum"])
        return buf

    def _step_sharded(self) -> None:
        """The same update as `step`, with the pieces split among the group's GPUs.

        1. Every GPU updates all momentum buffers (unchanged checkpoint format)
        2. Orthogonalize the Nesterov update of owned pieces only, appended to the send buffer
        3. All-gather every piece; the length matches the GPU with the largest share (the rest is
           padding)
        4. Restore the received pieces to the shape and strides of the single-GPU `_newton_schulz`
           result and apply them. Tall pieces are orthogonalized transposed, so the computed
           (wide) form is sent
        """
        group_ = self.process_group
        me, n_ranks = dist.get_rank(group_), dist.get_world_size(group_)
        active = [  # (parameter, group, piece rows); no gradient here means none on any GPU (DDP)
            (p, g, _matrix(p).shape[0] // g["split"])
            for g in self.param_groups
            if not g.get("adamw")
            for p in g["params"]
            if p.grad is not None
        ]
        if not active:
            return
        key = (n_ranks, *(id(p) for p, _, _ in active))
        if key not in self._owners:
            shapes = [
                (rows, _matrix(p).shape[1]) for p, g, rows in active for _ in range(g["split"])
            ]
            self._owners[key] = shard_pieces(shapes, n_ranks)
        owner = self._owners[key]

        sends, sizes, k = [], [0] * n_ranks, 0
        for p, g, rows in active:
            split = g["split"]
            buf = _matrix(self._momentum(p, g))
            grad, cols = _matrix(p.grad), _matrix(p).shape[1]
            for j in range(split):
                if owner[k + j] != me:
                    continue
                # rows of the owned piece only (elementwise per row, same as computing all and
                # slicing)
                rows_j = slice(j * rows, (j + 1) * rows)
                orth = _newton_schulz(_nesterov(grad[rows_j], buf[rows_j], g))
                sends.append((orth.T if rows > cols else orth).reshape(-1))
            for j in range(split):
                sizes[owner[k + j]] += rows * cols
            k += split

        width = max(sizes)
        send = torch.zeros(width, dtype=torch.bfloat16, device=active[0][0].device)
        if sends:
            torch.cat(sends, out=send[: sizes[me]])
        received = torch.empty(n_ranks * width, dtype=send.dtype, device=send.device)
        dist.all_gather_into_tensor(received, send, group=group_)

        offsets, k = [r * width for r in range(n_ranks)], 0
        for p, g, rows in active:
            cols, parts = _matrix(p).shape[1], []
            for j in range(g["split"]):
                r = owner[k + j]
                flat = received[offsets[r] : offsets[r] + rows * cols]
                offsets[r] += rows * cols
                parts.append(flat.view(cols, rows).T if rows > cols else flat.view(rows, cols))
            k += g["split"]
            _apply(p, parts[0] if g["split"] == 1 else torch.cat(parts), g)
