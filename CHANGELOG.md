# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `strategies.hull_rarity(sample, remove="common")`: removes the hull vertex
  whose species are the most common in the sample (or, with
  `remove="rare"`, the rarest), by occupancy, abundance or range size, with
  the vertex's species combined by sum (rarity-weighted richness), mean or
  maximum. `update=True` measures commonness among the remaining points, and
  `power` draws the vertex at random, favoring high scores for
  `remove="rare"` and low ones for `remove="common"`.
- `strategies.band(sample, bearing=0)`: closes in on a line from both
  sides, so each step keeps a band: a belt of latitude by default, or with
  `bearing=90` a north-south strip, so a gradient along it survives.
- `strategies.sector(sample, start=0, sectors=4)`: removes one compass
  sector after another, to test whether a curve depends on which side of a
  region is lost first.
- `strategies.radial` takes `stretch` and `bearing`: the discs become
  ellipses, so what remains is longer in one direction than the other.
- Samples can carry each observation's elevation and date:
  `sources.dataframe` and `sources.csv` take `elevation_col` and `date_col`,
  `sources.gbif` keeps GBIF's elevation and event date, and
  `sources.inaturalist` and `sources.open_data` the observation date.
  `Sample.elevations` and `Sample.dates` give them per point (the mean
  elevation and the first date), `Sample.with_elevations` sets an elevation
  per point, and saved nests keep both.
- `strategies.elevation(sample, remove="highest")`: removes points from the
  summits down, from the valleys up (`"lowest"`), or outside a narrowing
  elevational band (`"outer"`).
- `strategies.temporal(sample, remove="newest")`: removes points in order of
  when they were first observed, so the nest runs back through time.
- `strategies.split(sample, "hull_peel", parts=4)`: splits the sample into
  parts (k-means clusters, a grid, your labels, nearest seeds or polygons)
  and runs a strategy inside each, giving one nest of several patches
  shrinking at once. `strategies.partition` returns the split alone.
- Nests can have `parts`: each step's area is then the ground its parts'
  convex hulls cover together, not one hull around them all. `to_frame`
  adds `n_parts`, `plot.nest` draws each part's hull, `evaluate.diagnostics`
  describes them together, and saved nests keep the parts (nest format 5;
  nests without parts are saved as before). `Nest.groups`, `Nest.footprint`
  and `Nest.areas_at` give a step's points by part, the ground it covers
  and its areas.
- `designs.islands(sample, regions)`: regions as sampling units (states,
  parks, real islands), one row each with the region's area and the species
  observed in it, for species-area data of the island type. `Islands` has
  `to_frame`, `sar_data`, `fit` and `sample_of`; areas are each region's own
  on the ellipsoid (optionally within a mask), or the hull of its points.
- `strategies.circle(sample)`: shrinks the sample's smallest enclosing
  circle as fast as possible, removing at each step the point the circle
  rests on whose loss shrinks it most.
- `strategies.density_peel` takes `method="voronoi"`: density as one over
  the area of each point's Voronoi cell, with no bandwidth to choose.
- `Sample.thin(km, seed=None)`: thins the sample to at most one point per
  grid cell, to even out sampling effort before nesting.
- `strategies.concave_peel` takes `ratio=(start, end)`: the ratio changes
  as the sample shrinks, in proportion to the points removed.
- `evaluate.ensemble(sample, strategy, n)`: many seeded nests of a random
  strategy averaged into one curve, with the band a single nest falls in.
- `evaluate.bootstrap(sample, strategy, model, n)`: fits to nests of
  resamples of the sample's points (drawn with replacement; each holds
  about 63% of them), for how much a deterministic strategy's fit owes to
  which places were sampled.
- `sources.simulated(n_points, n_species, seed)`: a made-up landscape of
  observations with elevations and dates (a ridge, species with home ranges
  and preferred heights, effort piled around a few places), for examples,
  teaching and trying strategies without data.
- Example scripts in `examples/`: a first nest, comparing strategies
  against a baseline, regions as islands, elevation and time, patches and
  effort, and real observations from GBIF with a land mask. All but the
  last run offline in seconds.
- `regions.admin_units(country, level)` and `regions.read_units(source,
  column)`: many named regions at once, each as its own polygon.

### Fixed

- Downloads name samplesar and its repository as their client. geoBoundaries now refuses
  Python's default `User-Agent` with a 403, so `regions.admin` failed for
  any boundary not already cached.

### Changed

- sars 0.11.12 or newer is required (was 0.11.5).

## [0.2.1] - 2026-11-23

### Added

- `strategies.hull_masked_area(sample, keep="largest")`: `hull_area`'s choice
  of hull vertex, made by each hull's area within the sample's mask, so on a
  coast a vertex whose loss gives up a wide stretch of sea and little land
  costs little. It needs a mask.

### Changed

- `Nest.to_frame` (and so `sar_data`, `fit` and the plots) keeps each step's
  hull and species counts from the step before instead of recomputing them,
  so nests that remove one point at a time take seconds instead of a minute:
  on 13,500 points, 6.5 s instead of 46 s for `radial`, and 0.3 s instead of
  44 s for `random`. The values agree with before to floating-point rounding.

## [0.2.0] - 2026-11-23

### Added

- Masks: `Sample.with_mask(region)` and `Sample.within(boundary, mask=True)`
  measure every area within a region (land, say), as the area of each hull's
  intersection with it, so a hull reaching over the sea or a lake counts only
  the ground inside. `Sample.area`, `Nest.to_frame`, `sar_data`, `fit` and
  `evaluate.diagnostics` use it; `to_frame` adds the whole hull's area as
  `hull_area`, and `diagnostics` the share inside as `mask_share`. Subsets
  keep the mask, saved nests store it (nest format 4), and `plot.nest` and
  `plot.animate` draw its outline and fill only the hull inside it. Nests
  themselves are unchanged.
- `Sample.hull_area_of(points)` and `Sample.areas_of(points)`: the area of some points' hull, within the
  mask if there is one (and, from `areas_of`, the whole hull's area too).

### Changed

- The README is rewritten: what a nested SAR is, how areas are measured,
  masks (with a figure), the strategies, data sources and evaluation.
- This release is 0.2.0 because 0.1.4 broke the API without a minor
  release: the modules regrouped by job (`sources`, `regions`,
  `strategies`, `evaluate`, `plot`) and the US spellings (`center`,
  `meters`) replaced the old names. See 0.1.4 below.

## [0.1.4] - 2026-11-23

### Added

- `Sample.within(boundary)` and `sources.open_data(..., boundary=...)`:
  keep only the observations inside a region's outline, from a shapely
  polygon in WGS84 or a GeoDataFrame or GeoSeries in any CRS.
- `regions.admin(country, level=0, name=None)`: a country's, state's or
  county's boundary from geoBoundaries' open release, cached, for
  `within` and `boundary=`.
- `regions.read(source, column, values, clip=...)` and
  `regions.ecoregion(source, codes=..., names=...)`: regions from any
  boundary file or URL (cached), or from the US EPA's Level III and IV and
  RESOLVE 2017 ecoregions, through the new `geo` extra (geopandas).
- `sources.gbif()`: a sample from GBIF occurrences, records you have or a
  query fetched through pygbif (the new `gbif` extra) up to the search API's
  100,000-record limit, keeping present, georeferenced, species-level
  records without geospatial issues, optionally by coordinate uncertainty
  and boundary.

- Four strategies that rank every point once, which is fast on large samples:
  `strategies.radial` (shrinking discs around a center), `strategies.sweep`
  (a front moving along a bearing), `strategies.density_peel` (lowest kernel
  density first, giving kernel-smoothed boundaries) and `strategies.grow`
  (the reverse of a region grown from a seed by nearest neighbors).

- `strategies.hull_weighted`: a random hull vertex, picked with probability
  proportional to its isolation (distance to the nearest other point) raised
  to `power`, so isolated outliers tend to go first.

- `strategies.grid_peel(sample, cell_km)`: hull peeling of the occupied grid
  cells at a chosen scale, removing whole cells together, for multi-scale
  removal.

### Changed

- **Breaking: the API is grouped by job, and the old top-level names are
  gone** (no aliases):
  - `samplesar.sources`: `dataframe`, `csv`, `geodataframe`, `inaturalist`,
    `open_data` and `download_open_data` replace `Sample.from_df`,
    `from_csv`, `from_geodataframe`, `from_pyinaturalist`, `from_open_data`
    and `samplesar.download_open_data`.
  - `samplesar.regions`: `admin`, `ecoregion` and `read`.
  - `samplesar.strategies`: `random`, `from_order`, `hull_random`,
    `hull_peel`, `hull_area`, `hull_species`, `hull_distance` and
    `concave_peel` replace the `*_nest` functions and `nest_from_order`;
    `run`, `available` and `register` replace `nest`, `strategies` and
    `register_strategy`.
  - `samplesar.evaluate`: `compare`, `baseline`, `against_baseline`,
    `diagnostics` and `autocorrelation` replace `compare_nests`,
    `random_baseline`, `compare_to_baseline`, `nest_diagnostics` and
    `residual_autocorrelation`.
  - `samplesar.plot`: `nest`, `curve` and `animate` replace `plot_nest`,
    `plot_nest_curve` and `animate_nest`.

  `Sample`, `Nest` and `hull_area` stay at the top level, as do methods such
  as `Sample.within` and `Nest.fit`. Registered strategy names and nest
  labels are unchanged by the regrouping.

- **Breaking: US English spellings.** `Sample(centre=...)` is now
  `Sample(center=...)`, `Sample.projection_centre` is
  `Sample.projection_center`, and `Sample.merge_within(metres)` is
  `merge_within(meters)`; the docs use US spellings throughout. Saved nests
  store the sample's `center` (format version 3); files saved with `centre`
  still load, as do nests saved with a `centre` option, read as `center`.

- samplesar needs sars 0.11.5 or newer, which reports the parameters a fit
  left on a bound (`at_bounds` in `evaluate.compare` and the baseline's
  `at_bounds_share`).

## [0.1.3] - 2026-11-22

### Added

- `concave_peel_nest()` (registered as `"concave_peel"`): peel a sample by
  the boundary of a concave hull, which follows its outline instead of
  spanning its gaps.
- `nest_diagnostics()` and `residual_autocorrelation()`: each step's hull
  perimeter, compactness, area share, density and centroid shift, and the
  lag-1 autocorrelation and Durbin-Watson statistic of a fit's residuals
  along the nest.
- `random_baseline()` and `compare_to_baseline()`: fit a model to many
  seeded random nests of a sample, and place a strategy's fit among them
  (mean, spread, quantiles and a rank p-value per parameter), since one
  nest's R² and AICc aren't independent evidence.
- `Sample.merge_within(metres)`: merge points within a distance of each
  other into one, keeping every observation, so records at one spot (GPS jitter) don't
  make the last steps of a hull nest measure centimetre-wide hulls.
- `compare_nests()` has an `at_bounds` column naming the parameters that
  stopped on a bound of sars' search (such as the power model's z at 5 for a
  near-vertical curve), and warns about them; it needs a sars with
  `SARFit.at_bounds`, and is left unknown with an older one.
- `download_open_data()` and `Sample.from_open_data()`: a local copy of
  iNaturalist's open-data export as one Parquet file, and samples queried
  from it by region, taxon (and everything under it), grade, date and
  positional accuracy, through the new `opendata` extra (DuckDB).
- `papers/`: manuscripts built on samplesar. The first, on the hull
  strategies, has an outline and a formal specification of strategies S1 to
  S9 that the code is checked against.

### Fixed

- `plot_nest` shows at most a handful of coordinate ticks, so long
  longitudes no longer run into each other.

- A point on a hull edge that rounding pushed a hair outward (as in a grid
  laid out in a sample's UTM CRS) no longer counts as a hull vertex, so the
  vertex strategies can't pick it and it doesn't move the vertex centroid;
  vertices and the peeling boundary now share one tolerance.

## [0.1.2] - 2026-11-22

### Added

- `nest(sample, strategy, **options)` runs any strategy by name, and
  `strategies()` lists them with a one-line description each.
- `register_strategy(name)`: register a new sampling design so it runs by
  name and labels its nests like the built-in strategies.
- `Nest.strategy`, `Nest.options` and `Nest.label` record how a nest was made;
  `compare_nests()` also takes a plain sequence of nests, named by label
  (repeated labels are numbered).
- `nest_from_order(sample, order)`: nest a sample by removing its points in
  a given order, so any ordering can be evaluated.
- `Nest.to_dict()` / `Nest.from_dict()` and `Nest.to_json()` /
  `Nest.from_json()`: save and load a nest, with its strategy and options.
- `Sample.from_df(..., crs=...)` takes coordinates in any coordinate reference
  system; in a projected one (UTM, a national grid) areas and distances are
  measured in it. `Sample.from_geodataframe()` reads point GeoDataFrames
  (geopandas isn't required). `Sample.xy`, `Sample.projection`; areas in a
  CRS that isn't equal-area are corrected by its area distortion
  (`Sample.area_scale`).
- `Sample.from_pyinaturalist()`: a sample from iNaturalist observations,
  fetched through pyinaturalist (the new `inat` extra) past the API's
  10,000-result limit, keeping identifications at species level or below
  (a subspecies counts as its species) with true coordinates by default.
- `plot_nest()`, `plot_nest_curve()` and `animate_nest()`: draw a nest's
  points and hull at a step, its species-area curve with a power fit, and an
  animation of it shrinking (saved as GIF or MP4).

### Changed

- `Nest.to_frame()` (and so `sar_data()`, `fit()` and `compare_nests()`) is
  5 to 7 times faster: species counts are updated by the points each step
  drops instead of recounted. `Sample.points` is computed once and is
  read-only (an in-place edit of the returned array now raises).
- **Areas are in km², distances in km.** Hulls are measured on an equal-area
  projection of the WGS84 ellipsoid centred on each sample (pyproj, a new
  dependency), not in square degrees. Every reported area and the fitted `c`
  change; `z` doesn't. A degree of longitude no longer counts as much as one
  of latitude in the distance strategy.

## [0.1.1] - 2026-11-22

### Added

- `hull_random_nest()`: nest a sample by removing a random vertex of its
  convex hull at each step, so it shrinks from the outside in.
- `hull_peel_nest()`: nest a sample by peeling off its whole convex hull
  boundary at each step (convex layers).
- `hull_area_nest()`: nest a sample by removing, at each step, the hull
  vertex whose loss leaves the largest (or smallest) remaining hull.
- `hull_species_nest()`: nest a sample by removing, at each step, the hull
  vertex whose loss leaves the most (or fewest) species.
- `hull_distance_nest()`: nest a sample by removing, at each step, the hull
  vertex farthest from (or closest to) the centroid of the hull's vertices.
- `Nest.fit()` and `compare_nests()`: fit a sars model to a nest, or the same
  model to several nests (one row each: steps, parameters, R², AICc), to
  compare sampling strategies.
- `make typecheck` (mypy), part of `make check`; tests run in parallel.
- A documentation site: getting started, the sampling strategies and an API
  reference (`make docs`, built on every pull request, deployed on each
  release).
- CONTRIBUTING.md and CODE_OF_CONDUCT.md.

### Changed

- **Python 3.11 or newer is required.** 3.10 reached end of life in October
  2026. The dependency floors move to versions with Python 3.11 wheels:
  numpy 1.23.2 and pandas 1.5.
- CI lints with the locked ruff version, type-checks with mypy, tests the
  oldest and newest Pythons on pull requests and every supported Python on
  `main`, and checks the minimum and newest allowed dependency versions.

### Fixed

- `random_nest()` rejects `min_points` below 1 with a clear message; 0 gave
  a nest ending in an empty step and a negative value failed inside numpy.

## [0.1.0] - 2026-11-22

The first release as samplesar. Releases 0.1.0 to 0.1.3 were published as
nestsar, a project that no longer exists; samplesar starts its version numbers
again at 0.1.0.

### Added

- `Sample`: georeferenced species observations from a DataFrame or CSV, with
  distinct points, species count, and convex hull area.
- `Nest` and `random_nest()`: nest a sample by removing one point at a time
  uniformly at random, the baseline for every other strategy, and export the
  area and species table that `sars` fits.
- Tests and PyPI classifiers cover Python 3.10 to 3.15.

### Changed

- **Renamed from nestsar to samplesar**, for the distribution and the import
  name alike: `pip install samplesar` and `import samplesar` replace
  `nestsar`. The package now covers geometric sampling designs in general,
  with nesting as one of them. Apart from the import name the API is
  unchanged, so `nestsar.random_nest(...)` becomes
  `samplesar.random_nest(...)`.

