"""Regions to restrict samples to, as polygons for [`Sample.within`][samplesar.Sample.within]
and [`sources.open_data`][samplesar.sources.open_data]'s ``boundary``.

Administrative units from geoBoundaries, ecoregions from known datasets, and
any other boundary file or URL. [`admin_units`][samplesar.regions.admin_units]
and [`read_units`][samplesar.regions.read_units] give many regions at once,
by name, for [`designs.islands`][samplesar.designs.islands].
"""

from samplesar._geoboundaries import admin, admin_units
from samplesar._regions import ECOREGIONS, ecoregion, read, read_units

__all__ = [
    "ECOREGIONS",
    "admin",
    "admin_units",
    "ecoregion",
    "read",
    "read_units",
]
