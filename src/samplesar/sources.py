"""Sources of observations: each function returns a [`Sample`][samplesar.Sample].

From a DataFrame, a CSV file or a GeoDataFrame you already have; from
iNaturalist, through its API or a local copy of its open-data export.
"""

from samplesar._opendata import download_open_data
from samplesar._sources import csv, dataframe, geodataframe, inaturalist, open_data

__all__ = ["csv", "dataframe", "download_open_data", "geodataframe", "inaturalist", "open_data"]
