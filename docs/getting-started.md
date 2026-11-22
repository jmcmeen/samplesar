# Getting started

## A sample

A `Sample` is a set of species observations, each at a longitude and latitude
in decimal degrees. Several observations can share a point.

```python
import pandas as pd
import samplesar

df = pd.DataFrame(
    {
        "species": ["a", "b", "a", "c", "d", "b"],
        "lon": [-82.30, -82.28, -82.25, -82.27, -82.21, -82.24],
        "lat": [36.50, 36.55, 36.52, 36.58, 36.51, 36.56],
    }
)
sample = samplesar.sources.dataframe(df)
sample.n_points, sample.n_species, sample.area   # area in km²
```

`sources.csv` reads the same three columns from a file; pass
`species_col`, `lon_col` and `lat_col` if yours are named differently.

## Coordinate systems

Areas are measured in km² on an equal-area projection centered on the sample.
If your coordinates are already projected, say a grid laid out in UTM, give
the CRS, and areas and distances are measured in it, so the grid stays
exactly straight:

```python
utm = samplesar.sources.dataframe(df, lon_col="easting", lat_col="northing", crs="EPSG:32617")
```

A GeoDataFrame of points works directly; its CRS is read from the frame
(geopandas isn't needed by samplesar itself):

```python
sample = samplesar.sources.geodataframe(gdf, species_col="species")
```

## From iNaturalist

With `pip install 'samplesar[inat]'`, a sample can come straight from
iNaturalist, fetched through pyinaturalist (or from records you already
have):

```python
sample = samplesar.sources.inaturalist(
    taxon_id=20979, place_id=45, quality_grade="research"   # frogs in Tennessee
)
```

Only identifications at species level or below are kept (a subspecies
counts as its species), and obscured coordinates
(randomized by iNaturalist within about 0.2°) are dropped, unless you ask
otherwise (`ranks=None`, `skip_obscured=False`).

## From GBIF

With `pip install 'samplesar[gbif]'`, `sources.gbif` builds a sample from
GBIF, which gathers occurrences from museums, surveys, eBird, iNaturalist and
more:

```python
frogs = samplesar.sources.gbif(taxonKey=952, country="CR")   # Anura in Costa Rica
```

It keeps present, georeferenced records identified at species level or
below, without geospatial issues; `max_uncertainty` (meters) and `boundary`
narrow it further. The search API serves up to 100,000 records per query;
for more, use GBIF's download service. (`taxonKey` takes GBIF Backbone keys,
which pygbif warns are being replaced by Catalogue of Life keys.)

## From a local copy of iNaturalist's open data

For many samples, or large ones, keep a local copy of iNaturalist's
open-data export instead of calling the API (`pip install
'samplesar[opendata]'`, which adds DuckDB). The first call downloads about
14 GB and builds a 12 GB Parquet file; later queries take moments:

```python
from pathlib import Path

parquet = samplesar.sources.download_open_data(Path.home() / "data" / "inaturalist-open-data")
frogs = samplesar.sources.open_data(
    parquet, bbox=(-85.0, 35.0, -81.5, 37.0), taxon_id=20979   # frogs, East Tennessee
)
```

`taxon_id` takes any taxon and everything under it. By default only research
grade observations at species level or below are kept. The export holds only
observations with openly licensed photos, and obscured observations carry
their public, randomized coordinates without a flag; `max_accuracy` (meters)
is the filter available for that.

## Within a boundary

`Sample.within` keeps the observations inside a region's outline, and
`sources.open_data(boundary=...)` reads only those. `samplesar.regions.admin` fetches
countries, states and counties from geoBoundaries (cached after the first
download); any shapely polygon, or a GeoDataFrame in any CRS, works too:

```python
costa_rica = samplesar.regions.admin("CRI")
frogs = samplesar.sources.open_data(parquet, boundary=costa_rica, taxon_id=20979)

tennessee = samplesar.regions.admin("USA", level=1, name="Tennessee")
sullivan = samplesar.regions.admin("USA", 2, "Sullivan", inside=tennessee)
```

Regions that cross political lines come from boundary files.
`samplesar.regions.ecoregion` reads known ecoregion datasets (US EPA Level III and
IV, and RESOLVE 2017 worldwide), and `samplesar.regions.read` any other
file or URL (physiographic provinces, watersheds, protected areas), with
`pip install 'samplesar[geo]'`. Downloads are cached, so a script that names
its sources is reproducible:

```python
# The Southern Appalachians: three EPA ecoregions south of the Roanoke Gap.
southern_appalachians = samplesar.regions.ecoregion(
    "epa_l3", codes=["66", "67", "68"], clip=(-90, 30, -75, 37.3)
)
salamanders = samplesar.sources.open_data(
    parquet, boundary=southern_appalachians, taxon_id=26718
)
```

geoBoundaries' files carry
their own licenses (CC BY 4.0 or ODbL) and need attribution when published.

## A nest

A nesting strategy shrinks the sample one step at a time. Each step keeps a
subset of the points of the step before, so the areas and species counts form
a nested species-area curve.

```python
nest = samplesar.strategies.random(sample, seed=0)
nest.to_frame()      # step, n_points, area (km²), species
```

## Fitting the curve

`Nest.sar_data()` gives the `area` and `species` columns
[sars](https://jmcmeen.github.io/sars/) fits, without the zero-area steps at
the end:

```python
import sars

fit = sars.sar_power(nest.sar_data())
fit.params["z"], fit.r_squared
```

## Plotting

```python
samplesar.plot.nest(nest, step=10)       # the points and hull at step 10
samplesar.plot.curve(nest)          # species against area, log axes, with a power fit
animation = samplesar.plot.animate(nest)
animation.save("nest.gif", writer="pillow")
```

matplotlib comes with sars, so plotting needs nothing extra; an MP4 needs
ffmpeg installed.

## Saving a nest

A nest can be saved and loaded, so an expensive one is built once:

```python
nest.to_json("nest.json")
again = samplesar.Nest.from_json("nest.json")
```

The file holds the observations, the strategy and its options, and the
points each step removes.
