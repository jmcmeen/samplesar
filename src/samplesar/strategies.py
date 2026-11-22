"""Sampling strategies: each nests a [`Sample`][samplesar.Sample] into a
[`Nest`][samplesar.Nest], and is registered so it can also be run by name.

See the strategies page of the documentation for what each removes, and
``papers/hull-strategies/strategies.md`` for their formal specification.
"""

from samplesar._hull import (
    concave_peel,
    hull_area,
    hull_distance,
    hull_peel,
    hull_random,
    hull_species,
)
from samplesar._nest import from_order, random
from samplesar._registry import available, register, run

__all__ = [
    "available",
    "concave_peel",
    "from_order",
    "hull_area",
    "hull_distance",
    "hull_peel",
    "hull_random",
    "hull_species",
    "random",
    "register",
    "run",
]
