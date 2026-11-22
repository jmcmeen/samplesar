# samplesar

Species-area relationship (SAR) datasets from geometric sampling of point
observations. samplesar takes georeferenced species records, samples them with
a geometric design, such as a sequence of nested convex hulls, and returns the
area and richness table that [sars](https://jmcmeen.github.io/sars/) fits.

```bash
pip install samplesar
```

```python
import samplesar, sars

sample = samplesar.Sample.from_csv("observations.csv")   # species, lon, lat
nest = samplesar.random_nest(sample, seed=0)
fit = sars.sar_power(nest.sar_data())
```

- [Getting started](getting-started.md): load a sample, nest it, fit the curve.
- [Sampling strategies](strategies.md): what each design removes, step by step.
- [API reference](api/sample.md).

!!! note "Areas are in square degrees"
    Areas are convex hull areas of longitude and latitude, in square degrees.
    That is fine for ordering nested samples and for fitting within one
    sample, but it isn't a physical area: don't compare `c` across regions at
    different latitudes. A projected area is planned.
