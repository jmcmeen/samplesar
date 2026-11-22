# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

