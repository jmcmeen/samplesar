# Sampling strategies

Each strategy takes a `Sample` and returns a `Nest`: a sequence of ever-smaller
subsets of the sample's points, stopping when `min_points` remain. Strategies
differ only in which point they drop next.

## Running a strategy by name

Every strategy below is registered under a short name, so it can be chosen by
a string, from a script or a config file:

```python
samplesar.strategies.available()                      # name -> one-line description
nest = samplesar.strategies.run(sample, "hull_area", keep="smallest")
nest.label                                  # "hull_area(keep='smallest')"
```

Each nest records the strategy that made it and the options it was given
(`nest.strategy`, `nest.options`), whether it was made by name or by calling
the function directly. `evaluate.compare` names its rows by these labels when it
gets a plain list of nests.

## Random removal

`strategies.random(sample, min_points=3, seed=None)` removes one point at a time,
uniformly at random. It's the baseline every other strategy is compared
against.

## A given order

`strategies.from_order(sample, order, min_points=3)` removes points in the order
given, one per step: any ordering, from a heuristic, an optimizer or an
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

`strategies.hull_random(sample, min_points=3, seed=None)` removes one hull vertex
per step, chosen uniformly at random: the outside-in counterpart of
`strategies.random`.

### Isolation-weighted hull vertex

`strategies.hull_weighted(sample, power=1.0, min_points=3, seed=None)` removes a
random hull vertex too, but picks each one with probability proportional to
its distance from the nearest other remaining point, raised to `power`.
Isolated outliers (vagrants, georeferencing errors, one-off trips) tend to go
first, and clustered edges last longer. With `power=0` it is `hull_random`.

### Hull peeling

`strategies.hull_peel(sample, min_points=3)` removes the whole hull boundary at each
step: every vertex and every point on an edge, leaving only the points
strictly inside. Edges are straight on the sample's projection: a grid laid
out in a projected CRS (UTM, say) peels ring by ring if the sample is given
that CRS, but rows along parallels of longitude and latitude are curved on
the Earth, so a grid laid out in degrees doesn't peel by whole rows. These are the sample's convex layers (onion peeling). A step
can remove many points, so the nest has few steps; on a regular grid each step
takes off the outer ring. It stops before a layer would leave fewer than
`min_points` points.

### Grid peeling

`strategies.grid_peel(sample, cell_km, min_points=3)` peels at a chosen scale.
Points are snapped to square cells of `cell_km`, and each step peels the
outer layer of occupied cells, removing all their points together. Small
cells behave like `hull_peel`. Large cells give a few coarse steps that
ignore GPS jitter and detail within a cell, so running a range of sizes shows
how much the curve depends on the scale of removal.

### Concave hull peeling

`strategies.concave_peel(sample, ratio=0.3, min_points=3)` peels the boundary of a
concave hull instead, which follows the sample's outline rather than spanning
its gaps: suited to L-shaped or scattered samples. `ratio` runs from 1 (the
convex hull, as `strategies.hull_peel`) down towards 0 (tighter). Areas are still
convex hull areas.

## Choosing a vertex

The next strategies each remove one hull vertex per step, chosen by a rule
rather than at random. Every vertex of the current hull is a candidate; ties
go to the vertex farthest from the centroid of the hull's vertices, then to
the lowest point index, so the nest is always the same for the same sample.

### By area change

`strategies.hull_area(sample, keep="largest", min_points=3)` tries removing each
vertex and keeps the largest remaining hull: the vertex whose loss costs the
least area goes first, so the area falls as slowly as the points allow.
`keep="smallest"` does the opposite, cutting the area as fast as possible.

### By area change within a mask

`strategies.hull_masked_area(sample, keep="largest", min_points=3)` makes the
same choice by each hull's area within the sample's mask (`Sample.with_mask`),
so on a coast a vertex whose loss gives up a wide stretch of sea and little
land costs little. It needs a mask, and chooses as `hull_area` does once
every hull lies inside it.

### By species change

`strategies.hull_species(sample, keep="most", min_points=3)` removes the vertex
whose loss leaves the most species: a vertex whose species all occur at other
remaining points costs nothing, so species richness falls as late as
possible. `keep="fewest"` removes first the vertices holding species found
nowhere else, so richness falls as fast as possible.

### By distance from the center

`strategies.hull_distance(sample, remove="farthest", min_points=3)` removes the
vertex farthest from the centroid of the hull's vertices (in km on the
sample's projection), trimming outlying corners first. `remove="closest"` removes the vertex nearest
the center and leaves the outliers to last. Ties go to the lowest point index.

## By a fixed ranking

These strategies rank every point once and remove them in that order, one per
step. They don't re-measure the hull at each step, so they're fast on large
samples. Each one models a different way a study area can shrink. Ties go to
the lowest point index. Areas are still convex hull areas.

### Radial

`strategies.radial(sample, center=None, min_points=3)` removes the point
farthest from a center, so the sample shrinks as concentric discs: the
point-data analogue of nested circular plots around a focal site. `center` is
a (longitude, latitude), by default the mean of the projected points.

### Sweep

`strategies.sweep(sample, bearing=90.0, min_points=3)` removes points in order
along a bearing (degrees clockwise from grid north), like a front moving
across the sample: a transect walked from one end, or a range edge retreating.
With the default of 90°, the westernmost point goes first.

### Density peeling

`strategies.density_peel(sample, bandwidth_km=None, min_points=3)` smooths the
density of points once with a Gaussian kernel and removes the point of lowest
density first. Each step keeps a superlevel set of that density, so the
boundary follows where the observations are, not the outermost few, and
isolated records go first. The default bandwidth uses Scott's rule.

### Growing from a seed

`strategies.grow(sample, seed=None, min_points=3)` grows a region from the
point nearest `seed` by adding, each time, the point nearest the region
(Prim's algorithm), and removes points in the reverse of that order. The last
points left are around the seed, and the region follows dense corridors of
observations rather than convex shapes.

## Comparing strategies

`Nest.fit(model="power")` fits one of [sars](https://jmcmeen.github.io/sars/)'
models to a nest, and `evaluate.compare` fits the same model to several nests,
one row each, so strategies can be compared on one sample:

```python
import samplesar

nests = {
    "random": samplesar.strategies.random(sample, seed=0),
    "hull random": samplesar.strategies.hull_random(sample, seed=0),
    "peel": samplesar.strategies.hull_peel(sample),
    "largest area": samplesar.strategies.hull_area(sample, keep="largest"),
    "most species": samplesar.strategies.hull_species(sample, keep="most"),
    "farthest": samplesar.strategies.hull_distance(sample, remove="farthest"),
}
samplesar.evaluate.compare(nests)   # n_steps, n_fitted, c, z, r_squared, aicc, converged
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
baseline = samplesar.evaluate.baseline(sample, n=99, strategy="hull_random")
samplesar.evaluate.against_baseline(samplesar.strategies.hull_peel(sample), baseline)
```

Each parameter gets the nest's value, the baseline's mean, spread and 95%
range, and a two-sided rank p-value.

## Adding a strategy

A new sampling design is a function that takes a `Sample` and its options as
keywords and returns a `Nest` whose `steps` are index arrays into
`sample.points`, each a subset of the one before. Register it, and it runs by
name, is listed by `strategies.available()` and labels its nests like the built-in ones:

```python
import numpy as np
import samplesar

@samplesar.strategies.register("every_other")
def every_other_nest(sample, min_points=3):
    """Drop every other point at each step."""
    remaining = np.arange(sample.n_points)
    steps = [remaining]
    while len(remaining) // 2 >= min_points:
        remaining = remaining[::2]
        steps.append(remaining)
    return samplesar.Nest(sample=sample, steps=tuple(steps))

samplesar.evaluate.compare([
    samplesar.strategies.run(sample, "every_other"),
    samplesar.strategies.run(sample, "hull_peel"),
])
```
