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
sample = samplesar.Sample.from_df(df)
sample.n_points, sample.n_species, sample.area   # area in km²
```

`Sample.from_csv` reads the same three columns from a file; pass
`species_col`, `lon_col` and `lat_col` if yours are named differently.

## Coordinate systems

Areas are measured in km² on an equal-area projection centred on the sample.
If your coordinates are already projected, say a grid laid out in UTM, give
the CRS, and areas and distances are measured in it, so the grid stays
exactly straight:

```python
utm = samplesar.Sample.from_df(df, lon_col="easting", lat_col="northing", crs="EPSG:32617")
```

A GeoDataFrame of points works directly; its CRS is read from the frame
(geopandas isn't needed by samplesar itself):

```python
sample = samplesar.Sample.from_geodataframe(gdf, species_col="species")
```

## From iNaturalist

With `pip install 'samplesar[inat]'`, a sample can come straight from
iNaturalist, fetched through pyinaturalist (or from records you already
have):

```python
sample = samplesar.Sample.from_pyinaturalist(
    taxon_id=20979, place_id=45, quality_grade="research"   # frogs in Tennessee
)
```

Only species-level identifications are kept, and obscured coordinates
(randomised by iNaturalist within about 0.2°) are dropped, unless you ask
otherwise (`ranks=None`, `skip_obscured=False`).

## A nest

A nesting strategy shrinks the sample one step at a time. Each step keeps a
subset of the points of the step before, so the areas and species counts form
a nested species-area curve.

```python
nest = samplesar.random_nest(sample, seed=0)
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

## Saving a nest

A nest can be saved and loaded, so an expensive one is built once:

```python
nest.to_json("nest.json")
again = samplesar.Nest.from_json("nest.json")
```

The file holds the observations, the strategy and its options, and the
points each step removes.
