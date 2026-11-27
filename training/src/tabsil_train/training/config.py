"""Training config: model, prior, optimization and run settings, flat in one dataclass.

Why flat: MLflow run params keep exactly these field names, and evaluation and Hub upload rebuild
the model from those params (`tabsil_train.hub.load_model`). `model_type` and `prior_type` pick the
model and the prior; each model and prior reads only the fields it has.

TabSil trains one model for classification and regression (`TASKS`, alternating per step). The
defaults are TabSil's curriculum stage 1 (`tabsil_train.recipe.curriculum`), so `TrainConfig()` is a
TabSil config.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, fields, replace

from tabsil.models import (
    DEFAULT_MODEL_TYPE,
    build_model,
    canonical_model_type,
    config_fields,
    model_class,
)
from tabsil_train.priors import DEFAULT_PRIOR_TYPE, canonical_prior_type, prior_module
from tabsil_train.training.reference import REFERENCE_BATCH

N_BUCKETS = 5000
# The tasks one training run learns (one model, the training loop alternates them per step)
TASKS = ("classification", "regression")
# Training speed options and optimizer settings. A resumed run takes the original run's values
# (training script, `source_speed_options`)
SPEED_OPTIONS = (
    "compile",
    "tf32",
    "weight_average",
    "weight_average_every",
    "muon_weight_decay",
    "muon_distributed",
    "muon_momentum",
    "phases",
)
# Settings that can change per step range (`TrainConfig.phases`)
PHASE_KEYS = ("lr", "muon_weight_decay")
# Scope of distributed Muon (`muon_distributed`)
MUON_DISTRIBUTED = ("off", "node", "all")
# Optimizers (`optimizer`, `training.muon`) and how messages name them
OPTIMIZER_NAMES = {"muon": "Muon", "muon_adamw": "Muon + AdamW"}


@dataclass
class TrainConfig:
    """Model, prior and optimization settings. The defaults are TabSil's curriculum stage 1; the
    smoke (local CPU) and medium (short measurement) sizes are constructors here, the curriculum
    is in `tabsil_train.recipe`."""

    # Training data kind (`tabsil_train.priors.PRIOR_TYPES`). Old names (`PRIOR_TYPE_ALIASES`) map
    # to the current ones
    prior_type: str = DEFAULT_PRIOR_TYPE
    # Model class (`tabsil.models.MODEL_TYPES`). Of the model settings below, only those in that
    # class's config are used. Old names (`MODEL_TYPE_ALIASES`) map to the current ones
    model_type: str = DEFAULT_MODEL_TYPE
    # Model structure (`ColumnRowTransformerConfig`, TabSil's `tabsil.recipe.STRUCTURE`): cell
    # width, row ICL layers and heads, cells per token, inducing points, CLS tokens, and the
    # interleaved column/row stage pairs
    emsize: int = 256
    nlayers: int = 24
    nhead: int = 16
    features_per_group: int = 3
    n_inducing: int = 256
    n_cls: int = 4
    column_blocks: int = 6
    row_blocks: int = 6
    column_heads: int = 4
    row_heads: int = 4
    # prior / table size (TabICLv2 training scripts, `tabsil_train.priors.graph_scm`). max_features
    # is also kept in the model config, where inference reads it (feature count seen in training)
    # - rows (train + eval) from [min_rows, max_rows] (log-uniform with log_uniform_rows), the
    #   train rows a U(min_train_fraction, max_train_fraction) share of them, features
    #   U(1, max_features) capped for long tables (`table_size.LONG_TABLE_FEATURE_CAPS`) and by
    #   max_cells (cells per table, eval rows included)
    # - binary_prob: probability that a classification table has 2 classes (TabICLv2 0.5)
    # - filter_unpredictable: resample unpredictable graphs and datasets
    max_features: int = 100
    log_uniform_rows: bool = False
    max_cells: int = 1_024 * 100
    min_rows: int = 1_024
    max_rows: int = 1_024
    min_train_fraction: float = 0.3
    max_train_fraction: float = 0.9
    binary_prob: float = 0.5
    filter_unpredictable: bool = True
    # Standardize the regression target again with the train rows (mean, sd with ddof 0), after
    # the prior's clip and its standardization over all rows, as inference does; tables whose
    # train-row sd is at most 0.1 are drawn again (`priors.transforms.TrainRowTargetScale`,
    # Kumo-Tabular's loader). The regression borders follow it. Off by default (v1)
    target_train_rows: bool = False
    missing_dataset_prob: float = 0.3
    border_datasets: int = 20_000  # datasets sampled to set the regression borders (our choice)
    # optimization (Muon on all parameters or Muon + AdamW, `optimizer`; warmup + cosine)
    batch_size: int = 256
    micro_batch: int = 4
    steps: int = 250_000  # total length of the lr schedule
    stop_step: int | None = None  # step to stop this run at (None: steps); for chunked training
    deadline: float | None = None  # past this time (Unix time), save a checkpoint and stop
    warmup_steps: int = 2_500
    lr: float = 8e-4
    grad_clip: float = 10.0
    # run
    seed: int = 0
    num_workers: int = 8
    # Micro batches each prior worker has requested ahead (the DataLoader's `prefetch_factor`). The
    # loader hands them over in request order, so one that takes longer to make than its head
    # start (workers per task x prefetch_factor micro batches) stops its GPU, and every other GPU
    # waits for it in the gradient all-reduce. A larger value absorbs slower tables at the same
    # tables in the same order. Priors with `corrected_cat_sizes` have such tables: at stage 1 on
    # 8 GPUs the default's head start is about 14 s, and `graph_scm_base_realism` has micro
    # batches that take 20 to 45 s
    prefetch_factor: int = 4
    # Python's garbage collector. A full pass walks every object it tracks, and nearly all of them
    # live for the whole run (the model, the optimizer state, the compiled graphs): 1.2 to 2.1
    # million per process, 0.7 to 1.2 s per pass on average. A pass stops its process, and every
    # other GPU waits for it in the gradient all-reduce. On (default): at every log step each
    # process collects once and then calls `gc.freeze()`, which moves the objects alive at that
    # point out of the collector's reach, so later passes walk only what was made since. Every GPU
    # does it at the same step, so the passes run side by side. Until the first log step of a run
    # or a resumed chunk nothing is frozen. `train` unfreezes at the end. The training computation
    # is the same on or off. It is set per chunk, and the run param holds the first chunk's value.
    # Measured on 8 H100 over the same tables: 6.7% less per step in stage 1 and 2.3% in stage 2
    # (`training/docs/v2-experiment-results.md`, "Collector pauses" and "The collector change as
    # merged")
    gc_freeze: bool = True
    log_every: int = 100
    val_every: int = 5_000
    val_batch: int = 8  # datasets per val batch; separate from micro_batch so the val set is fixed
    # Tables per task of the reference validation set (`training.reference`, metrics `ref_*`): drawn
    # from one fixed prior and seed, so that runs of different seeds and priors are validated on
    # the same tables. A multiple of 8 (its batch size). 0 turns it off (default)
    reference_val_tables: int = 0
    checkpoint_every: int = 5_000
    # latest.pt is overwritten, so every this many steps also keep `step_XXXXXXX.pt` (to go back)
    keep_every: int | None = 25_000
    # Collapse detection (CollapseGuard): if the train loss mean per log_every stays more than
    # collapse_threshold above the recent healthy mean for collapse_patience windows, rewind to the
    # healthy state kept in memory and continue with a new data seed. None disables it. Our
    # choice of values: in full-scale runs a collapse jumped classification 0.8 -> 2.0 and
    # regression -1.0 -> +1.1, while normal noise was about 0.1
    collapse_threshold: float | None = 0.5
    collapse_patience: int = 3
    snapshot_every: int = 2_000  # keep a healthy state in memory this often (latest 2)
    max_rewinds: int = 10  # stop if one run needs more rewinds than this
    # Factor applied to the base lr at every collapse rewind (1 keeps the lr). With the same lr
    # after a rewind the collapse repeated (8-10 rewinds in a regression run); lowering only the lr
    # from the same checkpoint stopped it (35k steps from 275k: 1e-4 collapsed 3 times even with
    # weight decay, 5e-5 never). A fixed rule lowers it instead of a manual lr change mid-run. The
    # warmup/cosine shape is unchanged and the factor carries over to the next chunk
    rewind_lr_factor: float = 0.5
    # The rule never lowers the base lr below this (after that it only rewinds). An lr given
    # explicitly when resuming (training script `--lr`) ignores it. If it is above
    # lr x rewind_lr_factor a rewind keeps the lr, so a recipe with a lower lr must lower this too
    rewind_min_lr: float = 2.5e-5
    # Training speedups (from the nanoTabPFN speedrun, arXiv 2606.03681). All off by default
    # - compile: torch.compile(dynamic=True) for the training forward. Table sizes vary per batch,
    #   so shapes stay dynamic
    # - tf32: float32 matmuls in TF32 (matmuls outside bf16 autocast,
    #   `set_float32_matmul_precision`)
    compile: bool = False
    tf32: bool = False
    # LAWA (latest weight averaging, speedrun record 7): average the latest K weights kept every
    # weight_average_every steps, validate it (`val_*_avg`) and save it as `final_avg.pt` at the
    # end. Training itself is unchanged. 0 disables it
    weight_average: int = 0
    weight_average_every: int = 500
    # Muon on every parameter (`training.muon`, TabICLv2 `--muon True`): parameters that are not 2-D
    # are flattened to (first dim, rest) matrices. The update size matches the AdamW RMS, so lr,
    # warmup and cosine are those of Adam. muon_weight_decay is the decoupled weight decay of all
    # parameters, muon_momentum the Muon momentum (TabICLv2 0.9 = `--beta1`)
    muon_weight_decay: float = 0.01
    muon_momentum: float = 0.9
    # "muon": Muon on every parameter (above, v1). "muon_adamw": Kumo-Tabular's split, Muon on the
    # Linear weights and AdamW (torch defaults, weight decay 0.01 on matrices and the attention
    # logit scales, 0 on the other 1-D parameters) on the target encoders, the output and the rest
    # (`training.muon`). Both share lr and schedule; muon_weight_decay sets only the Muon part. A
    # run resumes only from a checkpoint of the same optimizer
    optimizer: str = "muon"
    # Distributed Muon (`training.muon`): instead of every GPU computing the same Newton–Schulz
    # orthogonalization, GPUs split the matrix pieces and all-gather the results. "node" splits
    # among the GPUs of a node, "all" among all GPUs. Updates are bit-identical to "off" (default,
    # every GPU computes everything) and the checkpoint format is the same, so it can be switched
    # on or off when resuming
    muon_distributed: str = "off"
    # Per-step-range setting changes:
    # `((start step, {"lr": ..., "muon_weight_decay": ...}), ...)`.
    # Values apply from the update at the start step; before the first phase the `lr` and
    # `muon_weight_decay` fields above apply. A setting missing from a phase keeps the previous
    # phase's value. lr changes only the base lr; the warmup/cosine shape is unchanged. The lr
    # factor from collapse rewinds (`rewind_lr_factor`) carries across phases. Also accepts the
    # MLflow param string (`normalize_phases`)
    phases: tuple = ()
    # Training diagnostics (`training.diagnostics`). They do not change the training computation
    # (bit-identical when on). Per-group weight and gradient norms are always logged
    # - grad_cosine: every log_every steps, the per-group cosine between the two task gradients
    #   from the preceding 4 steps. Rank 0 briefly holds 3 extra steps of gradients (bf16)
    # - rank_timing: per-GPU forward/backward time and table size, gathered to rank 0 every
    #   log_every steps (metrics `rank_*`, one line per step in `rank_timing.jsonl`)
    grad_cosine: bool = False
    rank_timing: bool = False
    # Activation checkpointing: for batches whose tables have more rows (train + eval) than this,
    # each block recomputes its activations in backward instead of keeping them
    # (`ColumnRowTransformer.activation_checkpointing_rows`, 0: every batch). Validation batches of
    # that size use the model's inference path (`inference=True`, a decoder without row chunks or
    # a score matrix). Block values differ only in computation order, and the classification
    # decoder computes its scores in float32 even under bf16 autocast (off: the q·k product is
    # bf16), so slightly more precise. None disables it. Enabled so that the long tables of
    # curriculum stages 2 and 3 fit on one GPU
    activation_checkpointing_rows: int | None = None

    def __post_init__(self) -> None:
        self.phases = normalize_phases(self.phases)
        self.model_type = canonical_model_type(self.model_type)
        self.prior_type = canonical_prior_type(self.prior_type)
        model_class(self.model_type)  # ValueError for an unknown name
        prior_module(self.prior_type).check_config(self)  # unknown name or bad sizes
        if not 0.0 < self.rewind_lr_factor <= 1.0:
            raise ValueError(f"rewind_lr_factor must be in (0, 1]: {self.rewind_lr_factor}")
        if self.muon_weight_decay < 0:
            raise ValueError(f"muon_weight_decay must be at least 0: {self.muon_weight_decay}")
        if self.muon_distributed not in MUON_DISTRIBUTED:
            raise ValueError(
                f"muon_distributed must be one of {MUON_DISTRIBUTED}: {self.muon_distributed}"
            )
        if self.optimizer not in OPTIMIZER_NAMES:
            raise ValueError(f"optimizer must be one of {tuple(OPTIMIZER_NAMES)}: {self.optimizer}")
        if self.prefetch_factor < 1:
            raise ValueError(f"prefetch_factor must be at least 1: {self.prefetch_factor}")
        if not 0.0 <= self.muon_momentum < 1.0:
            raise ValueError(f"muon_momentum must be in [0, 1): {self.muon_momentum}")
        if self.reference_val_tables < 0 or self.reference_val_tables % REFERENCE_BATCH:
            raise ValueError(
                f"reference_val_tables must be a multiple of {REFERENCE_BATCH} (0 turns it off):"
                f" {self.reference_val_tables}"
            )
        if not 0.0 <= self.binary_prob <= 1.0:
            raise ValueError(f"binary_prob must be in [0, 1]: {self.binary_prob}")
        if self.activation_checkpointing_rows is not None:
            if self.activation_checkpointing_rows < 0:
                raise ValueError(
                    "activation_checkpointing_rows must be at least 0 (None disables it):"
                    f" {self.activation_checkpointing_rows}"
                )
            if not hasattr(model_class(self.model_type), "activation_checkpointing_rows"):
                raise ValueError(f"the {self.model_type} model has no activation checkpointing")
        if self.weight_average < 0 or self.weight_average_every < 1:
            raise ValueError(
                "weight_average must be at least 0 and weight_average_every at least 1"
            )
        if self.grad_cosine and self.log_every < 4:
            raise ValueError("grad_cosine uses the 4 steps before each log (needs log_every >= 4)")
        self._check_phases()

    def _check_phases(self) -> None:
        starts = [start for start, _ in self.phases]
        if any(b <= a for a, b in zip(starts, starts[1:], strict=False)) or any(
            s < 0 for s in starts
        ):
            raise ValueError(f"phase start steps must be at least 0 and increasing: {starts}")
        if starts and starts[-1] >= self.steps:
            # an unreachable phase would be logged with the run but never take effect
            raise ValueError(f"phase start steps must be below steps ({self.steps}): {starts}")
        for start, changes in self.phases:
            settings = dict(changes)
            if not settings or set(settings) - set(PHASE_KEYS):
                raise ValueError(f"phases change only {PHASE_KEYS} (step {start}: {settings})")
            if settings.get("lr", 1.0) <= 0:
                raise ValueError(f"phase lr must be above 0 (step {start})")
            if settings.get("muon_weight_decay", 0.0) < 0:
                raise ValueError(f"phase muon_weight_decay must be at least 0 (step {start})")

    def phase_value(self, name: str, step: int) -> float:
        """Value of `name` (`PHASE_KEYS`) for update `step` (0-based): from the last phase starting
        at or before `step` that sets it, else the field value."""
        value = getattr(self, name)
        for start, changes in self.phases:
            if start > step:
                break
            value = dict(changes).get(name, value)
        return value

    @property
    def n_out(self) -> int:
        """The class cap of the classification output: the prior's class count cap."""
        return prior_module(self.prior_type).MAX_CLASSES

    @property
    def tasks(self) -> tuple[str, ...]:
        """The tasks of the run (one model; the training loop alternates them per step)."""
        return TASKS

    @classmethod
    def medium(cls) -> TrainConfig:
        """The stage 1 model, prior and batch with 20k steps, for measuring step time and curves."""
        return cls(
            steps=20_000,
            warmup_steps=1_000,
            val_every=1_000,
            checkpoint_every=1_000,
            keep_every=5_000,
            border_datasets=5_000,
        )

    @classmethod
    def smoke(cls) -> TrainConfig:
        """Finishes within a minute on a local CPU, for checking code paths: width 32, 2 row ICL
        layers, 2 heads, 8 inducing points, 2 column/row stage pairs, tables of 64 rows x up to 8
        features."""
        return cls(
            emsize=32,
            nlayers=2,
            nhead=2,
            n_inducing=8,
            column_blocks=2,
            row_blocks=2,
            min_rows=64,
            max_rows=64,
            max_features=8,
            max_cells=512,
            border_datasets=40,
            batch_size=8,
            micro_batch=4,
            steps=20,
            warmup_steps=2,
            lr=1e-3,
            num_workers=0,
            log_every=5,
            val_every=10,
            checkpoint_every=10,
            keep_every=10,
        )

    def model_config(self):
        """The config of the `model_type` model: its fields that also exist here are copied, the
        rest keep that config's defaults."""
        mine = {f.name for f in fields(self)}
        values = {n: getattr(self, n) for n in config_fields(self.model_type) if n in mine}
        return model_class(self.model_type).config_class(n_out=self.n_out, **values)

    def build_model(self):
        return build_model(self.model_type, self.model_config())

    def for_world_size(self, world_size: int) -> TrainConfig:
        """Shrink the per-GPU micro batch when micro_batch x world_size exceeds the batch."""
        if self.micro_batch * world_size <= self.batch_size:
            return self
        return replace(self, micro_batch=self.batch_size // world_size)

    def grad_accumulation(self, world_size: int) -> int:
        per_forward = self.micro_batch * world_size
        if self.batch_size % per_forward:
            raise ValueError(
                f"batch_size {self.batch_size} is not divisible by micro_batch x world_size ="
                f" {per_forward}"
            )
        return self.batch_size // per_forward

    def make_prior(self, task: str, seed: int):
        """The `prior_type` prior of `task` ("classification" or "regression") with this config's
        dataset sizes."""
        if task not in TASKS:
            raise ValueError(f"the prior's task is one of {TASKS}: {task}")
        return prior_module(self.prior_type).make_prior(self, task, seed)


def normalize_phases(phases) -> tuple:
    """`TrainConfig.phases` in the form `((start step, (("lr", value), ...)), ...)` (settings
    sorted by name, float values). Also accepts a `{start step: {setting: value}}` dict, a list of
    (start step, dict), or the string stored in MLflow params/tags (`str(phases)`)."""
    if isinstance(phases, str):
        phases = ast.literal_eval(phases) if phases.strip() else ()
    if isinstance(phases, dict):  # {start step: {setting: value}}
        phases = sorted(phases.items())
    out = []
    for start, changes in phases or ():
        items = changes.items() if isinstance(changes, dict) else changes
        out.append((int(start), tuple(sorted((str(k), float(v)) for k, v in items))))
    return tuple(out)
