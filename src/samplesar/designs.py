"""Sampling designs that aren't nests: regions and patches as sampling units.

[`islands`][samplesar.designs.islands] divides a sample among regions you
give and counts the species in each, for species-area data of the island
type. [`quadrats`][samplesar.designs.quadrats] lays out square patches of
equal size instead, in a grid or at random, and counts the species in each.
"""

from samplesar._islands import islands
from samplesar._quadrats import quadrats

__all__ = ["islands", "quadrats"]
