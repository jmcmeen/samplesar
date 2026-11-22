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
sample.n_points, sample.n_species, sample.area
```

`Sample.from_csv` reads the same three columns from a file; pass
`species_col`, `lon_col` and `lat_col` if yours are named differently.

## A nest

A nesting strategy shrinks the sample one step at a time. Each step keeps a
subset of the points of the step before, so the areas and species counts form
a nested species-area curve.

```python
nest = samplesar.random_nest(sample, seed=0)
nest.to_frame()      # step, n_points, area, species
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
