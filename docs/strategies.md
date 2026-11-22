# Sampling strategies

Each strategy takes a `Sample` and returns a `Nest`: a sequence of ever-smaller
subsets of the sample's points, stopping when `min_points` remain. Strategies
differ only in which point they drop next.

## Running a strategy by name

Every strategy below is registered under a short name, so it can be chosen by
a string, from a script or a config file:

```python
samplesar.strategies()                      # name -> one-line description
nest = samplesar.nest(sample, "hull_area", keep="smallest")
nest.label                                  # "hull_area(keep='smallest')"
```

Each nest records the strategy that made it and the options it was given
(`nest.strategy`, `nest.options`), whether it was made by name or by calling
the function directly. `compare_nests` names its rows by these labels when it
gets a plain list of nests.

## Random removal

`random_nest(sample, min_points=3, seed=None)` removes one point at a time,
uniformly at random. It's the baseline every other strategy is compared
against.

## A given order

`nest_from_order(sample, order, min_points=3)` removes points in the order
given, one per step: any ordering, from a heuristic, an optimiser or an
experiment's design, becomes a nest to evaluate like the others. The order
needn't name every point; the nest stops when it runs out or `min_points`
remain.

## From the outside in

These strategies only remove points on the sample's convex hull, so the
sample shrinks from its edges inwards and each step's hull lies inside the
one before.

A point is a hull **vertex** if the hull bends there; a point exactly on a
straight edge between two vertices isn't one. "Exactly" is in floating point:
a point rounding puts a hair outside the edge is a vertex. When the remaining points are collinear
the hull is a segment, and its vertices are its two ends.

### Random hull vertex

`hull_random_nest(sample, min_points=3, seed=None)` removes one hull vertex
per step, chosen uniformly at random: the outside-in counterpart of
`random_nest`.

### Hull peeling

`hull_peel_nest(sample, min_points=3)` removes the whole hull boundary at each
step: every vertex and every point on an edge, leaving only the points
strictly inside. Edges are straight on the sample's projection: a grid laid
out in a projected CRS (UTM, say) peels ring by ring if the sample is given
that CRS, but rows along parallels of longitude and latitude are curved on
the Earth, so a grid laid out in degrees doesn't peel by whole rows. These are the sample's convex layers (onion peeling). A step
can remove many points, so the nest has few steps; on a regular grid each step
takes off the outer ring. It stops before a layer would leave fewer than
`min_points` points.

### Concave hull peeling

`concave_peel_nest(sample, ratio=0.3, min_points=3)` peels the boundary of a
concave hull instead, which follows the sample's outline rather than spanning
its gaps: suited to L-shaped or scattered samples. `ratio` runs from 1 (the
convex hull, as `hull_peel_nest`) down towards 0 (tighter). Areas are still
convex hull areas.

## Choosing a vertex

The next strategies each remove one hull vertex per step, chosen by a rule
rather than at random. Every vertex of the current hull is a candidate; ties
go to the vertex farthest from the centroid of the hull's vertices, then to
the lowest point index, so the nest is always the same for the same sample.

### By area change

`hull_area_nest(sample, keep="largest", min_points=3)` tries removing each
vertex and keeps the largest remaining hull: the vertex whose loss costs the
least area goes first, so the area falls as slowly as the points allow.
`keep="smallest"` does the opposite, cutting the area as fast as possible.

### By species change

`hull_species_nest(sample, keep="most", min_points=3)` removes the vertex
whose loss leaves the most species: a vertex whose species all occur at other
remaining points costs nothing, so species richness falls as late as
possible. `keep="fewest"` removes first the vertices holding species found
nowhere else, so richness falls as fast as possible.

### By distance from the centre

`hull_distance_nest(sample, remove="farthest", min_points=3)` removes the
vertex farthest from the centroid of the hull's vertices (in km on the
sample's projection), trimming outlying corners first. `remove="closest"` removes the vertex nearest
the centre and leaves the outliers to last. Ties go to the lowest point index.

## Comparing strategies

`Nest.fit(model="power")` fits one of [sars](https://jmcmeen.github.io/sars/)'
models to a nest, and `compare_nests` fits the same model to several nests,
one row each, so strategies can be compared on one sample:

```python
import samplesar

nests = {
    "random": samplesar.random_nest(sample, seed=0),
    "hull random": samplesar.hull_random_nest(sample, seed=0),
    "peel": samplesar.hull_peel_nest(sample),
    "largest area": samplesar.hull_area_nest(sample, keep="largest"),
    "most species": samplesar.hull_species_nest(sample, keep="most"),
    "farthest": samplesar.hull_distance_nest(sample, remove="farthest"),
}
samplesar.compare_nests(nests)   # n_steps, n_fitted, c, z, r_squared, aicc, converged
```

The areas are convex hull areas in km², so `c` is per km²; `z` doesn't
depend on the unit. A nest needs at least three steps with
a positive area to fit (a peeled nest with few layers may not have them).

Random removal keeps the sample's whole extent until its last few points, so
it gives a steep curve; the outside-in strategies shrink the area from the
first step, so their slopes are shallower. Nested steps share most of their
points, so R² and AICc describe how smooth a strategy's curve is, not
independent evidence: compare them across strategies on the same sample.

## Against a random baseline

Steps of one nest share most of their points, so its R² and AICc aren't
independent evidence. Compare a strategy with the spread of fits from many
random nests of the same sample instead:

```python
baseline = samplesar.random_baseline(sample, n=99, strategy="hull_random")
samplesar.compare_to_baseline(samplesar.hull_peel_nest(sample), baseline)
```

Each parameter gets the nest's value, the baseline's mean, spread and 95%
range, and a two-sided rank p-value.

## Adding a strategy

A new sampling design is a function that takes a `Sample` and its options as
keywords and returns a `Nest` whose `steps` are index arrays into
`sample.points`, each a subset of the one before. Register it, and it runs by
name, is listed by `strategies()` and labels its nests like the built-in ones:

```python
import numpy as np
import samplesar

@samplesar.register_strategy("every_other")
def every_other_nest(sample, min_points=3):
    """Drop every other point at each step."""
    remaining = np.arange(sample.n_points)
    steps = [remaining]
    while len(remaining) // 2 >= min_points:
        remaining = remaining[::2]
        steps.append(remaining)
    return samplesar.Nest(sample=sample, steps=tuple(steps))

samplesar.compare_nests([
    samplesar.nest(sample, "every_other"),
    samplesar.nest(sample, "hull_peel"),
])
```
