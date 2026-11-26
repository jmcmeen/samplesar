# samplesar

Species-area relationship (SAR) datasets from georeferenced species records.
samplesar builds the table of areas and species counts that
[sars](https://jmcmeen.github.io/sars/) fits, in two ways: by shrinking one
sample step by step with a sampling strategy, such as a sequence of nested
convex hulls, or by counting the species in each of several regions you
choose.

```bash
pip install samplesar
```

```python
import samplesar, sars

sample = samplesar.sources.simulated(seed=1)         # or sources.csv("observations.csv")
nest = samplesar.strategies.hull_peel(sample)        # shrink it from the outside in
fit = sars.sar_power(nest.sar_data())
```

- [Getting started](getting-started.md): load a sample, nest it, fit the curve.
- [Sampling strategies](strategies.md): what each design removes, step by step.
- [Regions as sampling units](islands.md): states, parks or islands, one row each.
- Examples, as notebooks: [a first nest](examples/first_nest.ipynb),
  [comparing strategies](examples/strategies.ipynb) and
  [regions as islands](examples/islands.ipynb).
- [API reference](api/core.md).

!!! note "Areas are in km²"
    Areas are convex hull areas in km², measured on an equal-area projection
    of the WGS84 ellipsoid centered on each sample, or in the sample's own
    projected coordinate system when it has one. Distances are in km.
