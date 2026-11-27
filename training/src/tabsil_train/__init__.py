"""tabsil-train: training and evaluation of TabSil (the `tabsil` package is the inference side).

- `priors`: training data (assembler, generators, transforms, table size rules, graph SCM prior)
- `training`: training config, the training loop, data feeding, losses, stability
- `benchmarks`: evaluation (AMLB/CTR23, TabArena, BeyondArena, uncertainty metrics)
- `recipe`: TabSil's training curriculum (the structure and inference defaults are in
  `tabsil.recipe`)
- `hub`: `load_model` for the scripts (Hub repos, `save_pretrained` folders, checkpoint files,
  MLflow `runs:/` artifacts) and the model structure in run params
- `subsample`: `SubsampledEstimator`, which reduces the rows passed to fit to N (class-stratified
  for classification; for context-length experiments)
- `utils`: MLflow run logging, the learning rate schedule, torchrun setup

The scripts are in `training/scripts` (`train.py`, `eval.py`) and `training/hub` (Hub upload).
"""
