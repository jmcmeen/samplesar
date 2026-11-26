# samplesar

Species-area relationships from one set of georeferenced species records.

A species-area relationship (SAR) is usually built from separate islands or
plots. samplesar builds one from a single sample of point observations
(iNaturalist, GBIF, a survey) instead, by sampling it with a geometric
design and recording the area and the number of species of each subset. The
resulting table goes to [sars](https://jmcmeen.github.io/sars/) for curve
fitting.

```python
import samplesar

sample = samplesar.sources.csv("observations.csv")   # species, lon, lat
nest = samplesar.strategies.hull_peel(sample)        # shrink it from the outside in
nest.to_frame()                                      # step, n_points, area (km²), species
fit = nest.fit("power")                              # sars' power model: S = c·A^z
```

Documentation: <https://jmcmeen.github.io/samplesar/>

## Install

```sh
pip install samplesar
```

Optional extras add data sources:

| Extra | For |
| --- | --- |
| `samplesar[inat]` | `sources.inaturalist`: observations fetched from iNaturalist's API |
| `samplesar[opendata]` | `sources.open_data`: iNaturalist's open-data export, read from a local copy |
| `samplesar[gbif]` | `sources.gbif`: occurrences from GBIF |
| `samplesar[geo]` | `regions.read` and `regions.ecoregion`: regions from boundary files |

## Terms

- **Sample:** a set of species observations, each at a longitude and latitude.
- **Strategy:** a sampling design: the rule that picks which points make up
  each subset of the sample.
- **Nest:** what a strategy returns: a sequence of subsets, each inside the
  one before.
- **Area:** each subset's area in km², measured on an equal-area projection
  centered on the sample (and, with a mask, only within it).

## Areas within a mask

A convex hull can reach over ground that isn't habitat: the sea along a
coast, an inlet, a lake, the bay of a concave island. Left alone, that
ground counts as area sampled, most at the large end of the curve.

With a **mask**, every area is the part of the hull inside a region you give
(land, say). The hull itself, and which points each strategy removes, stay
the same. In plots, the orange outline is the hull and **only the shaded
part counts toward the area**:

![The same hull without a mask (left: the whole hull counts) and with one (right: the inlet is left out)](https://raw.githubusercontent.com/jmcmeen/samplesar/main/docs/assets/mask.webp)

```python
costa_rica = samplesar.regions.admin("CRI")
frogs = samplesar.sources.open_data(parquet, boundary=costa_rica, taxon_id=20979)

land = frogs.with_mask(costa_rica)     # or frogs.within(costa_rica, mask=True)
nest = samplesar.strategies.hull_peel(land)
nest.to_frame()                        # area: the land in each hull; hull_area: the whole hull
```

For iNaturalist's frog records across Costa Rica, a quarter of the full hull
is sea. Measured on land only, the fitted z of hull peeling rises from 0.18
to 0.20.

## Strategies

Each strategy is one function in `samplesar.strategies`, also callable by
name through `strategies.run(sample, "hull_peel")`:

- **Baselines:** `random` removes points uniformly at random; `from_order`
  removes them in any order you give.
- **From the outside in:** `hull_random` and `hull_weighted` remove a hull
  vertex at random (uniformly, or favoring isolated ones); `hull_peel`,
  `grid_peel` and `concave_peel` remove whole layers.
- **Greedy hull choices:** `hull_area`, `hull_species` and `hull_distance`
  remove the hull vertex that changes the area or species count least or
  most, or lies farthest from or nearest to the center; `hull_rarity`
  removes the one whose species are the most common, or the rarest.
- **Fixed rankings**, fast on large samples: `radial` (shrinking discs),
  `sweep` (a line moving across), `density_peel` (sparsest first) and
  `grow` (growing out from a seed point, in reverse).

[Sampling strategies](https://jmcmeen.github.io/samplesar/strategies/) describes each one.

## Data and regions

- `samplesar.sources` builds samples from a DataFrame, CSV or GeoDataFrame;
  from iNaturalist, its open-data export or GBIF.
- `samplesar.regions` gives polygons to restrict a sample to (`Sample.within`)
  or to mask its areas: countries, states and counties from geoBoundaries,
  US EPA and RESOLVE ecoregions, or any boundary file or URL.
- `Sample.merge_within(meters)` merges records made at one spot (a
  trailhead, a feeder) that differ only by GPS jitter. Left apart, they make
  the last steps of a hull nest measure hulls centimeters across.

## Evaluating and plotting

- `samplesar.evaluate`: `compare` fits several nests side by side,
  `baseline` and `against_baseline` set a strategy against many random
  nests, and `diagnostics` and `autocorrelation` describe how each nest
  shrinks.
- `samplesar.plot`: `nest` draws one step, `curve` the species-area curve
  with its power fit, and `animate` the whole nest.

## Development

```sh
make install   # create .venv with dev tooling
make check     # lint, format check, type check, tests
make docs      # build the documentation site
```
