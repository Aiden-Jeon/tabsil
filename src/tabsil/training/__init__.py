"""Training.

- `config`: the training config `TrainConfig` (model, prior, optimization and run settings in one
  flat dataclass; MLflow params keep these names)
- `loop`: the shared prior-fitting training loop `train` (DDP, checkpoints and resuming, collapse
  rewinds)
- `data`: an IterableDataset that feeds the prior to a DataLoader (`PriorBatches`)
- `losses`: classification cross-entropy, regression pinball loss
- `muon`: Muon on every parameter (distributed Newton–Schulz included)
- `stability`: collapse detection
- `diagnostics`: training diagnostics (per-group weight and gradient norms, task gradient cosine,
  per-rank timing and table sizes). Read-only; they do not change the training computation
"""
