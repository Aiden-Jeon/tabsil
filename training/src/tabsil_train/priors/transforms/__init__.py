"""Post-processing transforms: they refine the groups a generator makes.

A transform is a callable in the shape of `tabsil_train.priors.pipeline.Transform`: `(group, rng) ->
group`. Returning None makes the `Pipeline` drop the group and draw again.

- `classes.CoverClasses`: reorder rows so that every held-out class also appears in the train rows
- `missing.Missing`: missing values (MCAR)
- `filters.DropInvalid`: drop a group with a dataset whose targets are all marked ignored

The graph SCM prior uses DropInvalid -> CoverClasses (classification) -> Missing, in that order.
"""

from tabsil_train.priors.transforms.classes import CoverClasses
from tabsil_train.priors.transforms.filters import DropInvalid
from tabsil_train.priors.transforms.missing import Missing

__all__ = ["CoverClasses", "DropInvalid", "Missing"]
