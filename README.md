# nestsar

Nested species-area relationship datasets from point observations. Build a
sequence of nested sub-samples from georeferenced species records and hand
the resulting area and richness table to [sars](https://pypi.org/project/sars/)
for curve fitting.

```python
import nestsar, sars

sample = nestsar.Sample.from_csv("observations.csv")   # species, lon, lat
nest = nestsar.random_nest(sample, seed=0)
fit = sars.sar_power(nest.sar_data())
```

## Development

```sh
make install   # create .venv with dev tooling
make check     # lint, format check, tests
```
