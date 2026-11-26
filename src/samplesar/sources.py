"""Sources of observations: each function returns a [`Sample`][samplesar.Sample].

From a DataFrame, a CSV file or a GeoDataFrame you already have; from
iNaturalist, through its API or a local copy of its open-data export; and
from GBIF. [`simulated`][samplesar.sources.simulated] makes up a landscape of
observations, for examples and teaching.
"""

from samplesar._opendata import download_open_data
from samplesar._simulated import simulated
from samplesar._sources import csv, dataframe, gbif, geodataframe, inaturalist, open_data

__all__ = [
    "csv",
    "dataframe",
    "download_open_data",
    "gbif",
    "geodataframe",
    "inaturalist",
    "open_data",
    "simulated",
]
