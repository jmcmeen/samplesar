# samplesar

Species-area relationship datasets from geometric sampling of point
observations. Sample georeferenced species records with a geometric design,
such as a sequence of nested hulls, and hand the resulting area and richness
table to [sars](https://pypi.org/project/sars/) for curve fitting.

```python
import samplesar, sars

sample = samplesar.sources.csv("observations.csv")   # species, lon, lat
nest = samplesar.strategies.random(sample, seed=0)
fit = sars.sar_power(nest.sar_data())
```

Documentation: <https://jmcmeen.github.io/samplesar/>

## Development

```sh
make install   # create .venv with dev tooling
make check     # lint, format check, type check, tests
make docs      # build the documentation site
```
