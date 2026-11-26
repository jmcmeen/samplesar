"""Sampling strategies: each nests a [`Sample`][samplesar.Sample] into a
[`Nest`][samplesar.Nest], and is registered so it can also be run by name.

See the strategies page of the documentation for what each removes, and
``papers/hull-strategies/strategies.md`` for their formal specification.
"""

from samplesar._hull import (
    concave_peel,
    grid_peel,
    hull_area,
    hull_distance,
    hull_masked_area,
    hull_peel,
    hull_random,
    hull_species,
    hull_weighted,
)
from samplesar._nest import from_order, random
from samplesar._ranked import (
    band,
    density_peel,
    elevation,
    grow,
    radial,
    sector,
    sweep,
    temporal,
)
from samplesar._rarity import hull_rarity
from samplesar._registry import available, register, run

__all__ = [
    "available",
    "band",
    "concave_peel",
    "density_peel",
    "elevation",
    "from_order",
    "grid_peel",
    "grow",
    "hull_area",
    "hull_distance",
    "hull_masked_area",
    "hull_peel",
    "hull_random",
    "hull_rarity",
    "hull_species",
    "hull_weighted",
    "radial",
    "random",
    "register",
    "run",
    "sector",
    "sweep",
    "temporal",
]
