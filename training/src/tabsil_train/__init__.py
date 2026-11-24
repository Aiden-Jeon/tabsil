"""tabsil-train: training and evaluation of TabSil (the `tabsil` package is the inference side).

- `priors`: training data (assembler, generators, transforms, table size rules, graph SCM prior)
- `training`: training config, the training loop, data feeding, losses, stability
- `benchmarks`: evaluation (AMLB/CTR23, TabArena, BeyondArena, uncertainty metrics)
- `recipe`: TabSil's training curriculum (the structure and inference defaults are in
  `tabsil.recipe`)
- `hub`: checkpoints of training runs (MLflow `runs:/` artifacts, the structure in run params)
- `utils`: MLflow run logging, the learning rate schedule, torchrun setup

The scripts are in `training/scripts` (`train.py`, `eval.py`) and `training/hub` (Hub upload).
"""
