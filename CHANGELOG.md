# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

