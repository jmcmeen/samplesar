"""Regions to restrict samples to, as polygons for [`Sample.within`][samplesar.Sample.within]
and [`sources.open_data`][samplesar.sources.open_data]'s ``boundary``.

Administrative units from geoBoundaries, ecoregions from known datasets, and
any other boundary file or URL.
"""

from samplesar._geoboundaries import admin
from samplesar._regions import ECOREGIONS, ecoregion, read

__all__ = ["ECOREGIONS", "admin", "ecoregion", "read"]
