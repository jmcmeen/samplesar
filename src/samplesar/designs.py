"""Sampling designs that aren't nests: regions as sampling units.

[`islands`][samplesar.designs.islands] divides a sample among regions you
give and counts the species in each, for species-area data of the island
type.
"""

from samplesar._islands import islands

__all__ = ["islands"]
