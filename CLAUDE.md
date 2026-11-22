# CLAUDE.md

Guidance for working in this repo. When something here conflicts with the code, trust the
code and flag the drift.

## What this is

`samplesar` builds species-area relationship (SAR) datasets from georeferenced species
observations by geometric sampling: nested hulls, islands and other sampling designs. Each
design turns a `Sample` of points into an area and richness table, and
[sars](https://github.com/jmcmeen/sars) fits and evaluates the curves.

- Python **>= 3.11**, `src/` layout, package `samplesar`.
- Tooling: **uv** for environments, **ruff** for lint and format, **mypy** for types,
  **pytest** (with xdist) for tests. Drive everything through the `Makefile`;
  `make check` runs what CI runs.

## Design

1. **`Sample` holds the observations; a `Nest` holds index arrays into its points.** A
   strategy only decides which point (or points) to drop next; it returns a `Nest` whose
   `steps[i]` are row indices into `sample.points`, each step a subset of the one before.
   Don't copy observation data per step.
2. **One public function per strategy family**, named for what it removes
   (`random_nest`, `hull_*_nest`), with `min_points` and, if random, `seed`. Choices that
   pick one candidate are deterministic: ties are broken by a stated rule, never by set
   or dict order.
3. **sars is a required dependency.** Evaluating a sampling design means fitting its SAR
   data, so `samplesar` uses sars directly rather than as an optional extra.
4. **Areas are convex hull areas in square degrees** until a projected area lands. Say so
   wherever an area is reported.
5. **Errors are exceptions** (`ValueError`, `KeyError`) with messages that name the
   problem; no sentinel returns.

## Tests

- Every strategy gets tests for nestedness, its own selection rule (checked by brute
  force on small samples), reproducibility, and a hand-made sample with a known answer.
- No real observation data in the repo: tests build synthetic samples.
