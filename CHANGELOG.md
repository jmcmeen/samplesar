# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.1] - 2026-10-27

### Added

- CI tests and PyPI classifiers cover Python 3.14 and 3.15.

## [0.1.0] - 2026-10-27

### Added

- `Sample`: georeferenced species observations from a DataFrame or CSV, with
  distinct points, species count, and convex hull area.
- `Nest` and `random_nest()`: nest a sample by removing one point at a time
  uniformly at random, the baseline for every other strategy, and export the
  area and species table that `sars` fits.

