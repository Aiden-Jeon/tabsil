"""Muon on every parameter (TabSil's optimizer; TabICLv2 training scripts' `--muon True`).

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
    ) -> Muon:
        """Muon on every parameter of `model` (module docstring), one param group per split count
        (`muon_parameters`)."""
        muon = muon_parameters(model)
        groups = [
            {"params": [p for p, s in muon if s == split], "split": split}
            for split in sorted({s for _, s in muon})
        ]
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
        if self.process_group is not None:
            self._step_sharded()
            return
        for group in self.param_groups:
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
