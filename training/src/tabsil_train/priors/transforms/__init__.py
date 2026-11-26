"""Post-processing transforms: they refine the groups a generator makes.

A transform is a callable in the shape of `tabsil_train.priors.pipeline.Transform`: `(group, rng) ->
group`. Returning None makes the `Pipeline` drop the group and draw again.

- `classes.CoverClasses`: reorder rows so that every held-out class also appears in the train rows
- `missing.Missing`: missing values (MCAR)
- `missing.StructuredMissing`: value-dependent (MAR/MNAR) missing values in column groups
- `categorical.HighCardinality`: recode columns as categorical columns with many categories
- `features.Coarsen`, `features.MeasurementNoise`: coarse values, noisy measurements
- `dependence.CorrelateColumns`, `dependence.Relabel`: correlated column blocks, a target
  recomputed from a few columns
- `filters.DropInvalid`: drop a group with a dataset whose targets are all marked ignored
- `target.TrainRowTargetScale`: standardize a regression target again with its train rows

The graph SCM prior uses DropInvalid -> [CorrelateColumns -> Relabel] -> CoverClasses
(classification) or [TrainRowTargetScale] (regression) -> [MeasurementNoise -> Coarsen ->
HighCardinality -> StructuredMissing] -> Missing, in that order. A bracketed part is only there
when its option is on.
"""

from tabsil_train.priors.transforms.categorical import HighCardinality
from tabsil_train.priors.transforms.classes import CoverClasses
from tabsil_train.priors.transforms.dependence import CorrelateColumns, Relabel
from tabsil_train.priors.transforms.features import Coarsen, MeasurementNoise
from tabsil_train.priors.transforms.filters import DropInvalid
from tabsil_train.priors.transforms.missing import Missing, StructuredMissing
from tabsil_train.priors.transforms.target import TrainRowTargetScale

__all__ = [
    "Coarsen",
    "CorrelateColumns",
    "CoverClasses",
    "DropInvalid",
    "HighCardinality",
    "MeasurementNoise",
    "Missing",
    "Relabel",
    "StructuredMissing",
    "TrainRowTargetScale",
]
