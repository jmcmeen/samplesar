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

A pair, `ratio=(0.6, 0.1)`, changes the ratio as the sample shrinks, in
proportion to the points removed: loose while there are many points to
outline, tighter as they go. shapely's ratio is already relative to the
spacing of the points left, so a single value is the default; the pair is
for testing how much a curve depends on it.

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

### By rarity

`strategies.hull_rarity(sample, remove="common", measure="occupancy", score="sum")`
removes the vertex whose species are the most common in the sample, so the
rare species stay to the end and richness falls slowly. `remove="rare"`
removes the vertex with the rarest species first, which shows how fast a
shrinking region loses its distinctive species.

A species' rarity is one over how common it is, by `measure`: the number of
points it occurs at (`"occupancy"`), its observations (`"abundance"`) or the
area of the convex hull of its own points (`"range"`). A vertex's score
combines the rarities of the species observed at it: `"sum"` (with
occupancy, this is rarity-weighted richness, which also favors rich
vertices), `"mean"` or `"max"`. Commonness is measured once on the whole
sample, or with `update=True` among the points remaining at each step. A
species whose points make no hull (one or two points, or a line) has the
smallest range any species has in the sample.

`hull_species` differs: it asks only whether a vertex holds a species found
at no other remaining point, so it can't tell a widespread species from a
scarce one until the scarce one is down to its last point.

With `power`, the vertex is drawn at random with probability proportional to
its score raised to that power (or the inverse, for `remove="common"`), as
`hull_weighted` does with isolation: `power=0` is `hull_random`, and `seed`
makes it reproducible.

### By the smallest enclosing circle

`strategies.circle(sample, min_points=3)` finds the smallest circle around
the remaining points, which rests on two or three of them (more when
several lie on it), and removes the one whose loss shrinks the circle most. It trims the sample where it reaches
farthest, as `radial` does, but toward no fixed center: the circle moves as
it shrinks, so a sample with a long arm loses the arm first and then closes
on its bulk.

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

With `stretch` the discs become ellipses, `stretch` times longer along
`bearing` (degrees clockwise from grid north) than across it:
`radial(sample, stretch=2, bearing=0)` keeps, at each step, an ellipse twice
as long north-south as east-west.

### Band

`strategies.band(sample, bearing=0, center=None, min_points=3)` closes in on
a line from both sides: each step removes the point farthest from the line
through `center`, measured along `bearing`. The default keeps a belt of
latitude that narrows from the north and the south, as a study of a
latitudinal gradient might. `bearing=90` narrows from the east and the west
instead, so what remains is a north-south strip: it keeps the sample's
north-south span, and any gradient along it, as far as the points near the
line reach.

### Sector

`strategies.sector(sample, start=0, sectors=4, clockwise=True, center=None, min_points=3)`
removes one compass sector after another, each from its far end inwards.
With the defaults the north-east quadrant goes first, then the south-east,
south-west and north-west. Running it from several `start` bearings tests
whether a curve depends on which side of a region is lost first, as when a
range retreats toward a pole or upslope.

### Sweep

`strategies.sweep(sample, bearing=90.0, min_points=3)` removes points in order
along a bearing (degrees clockwise from grid north), like a front moving
across the sample: a transect walked from one end, or a range edge retreating.
With the default of 90°, the westernmost point goes first.

### Elevation

`strategies.elevation(sample, remove="highest", around=None, min_points=3)`
removes points in order of elevation. `"highest"` works from the summits
down, so each step keeps everything below a falling contour; `"lowest"`
works from the valleys up and keeps the high ground, as a climate envelope
retreating upslope would. `"outer"` removes the point farthest in elevation
from `around` (by default the median), so each step keeps a narrowing
elevational band.

The sample needs an elevation in meters at every point: from a column
(`sources.dataframe(df, elevation_col="elevation")`; `sources.gbif` keeps
GBIF's, where records have one) or set per point with
`Sample.with_elevations`. Areas stay planimetric: each step's convex hull on
the map, not surface area over the terrain.

### Through time

`strategies.temporal(sample, remove="newest", min_points=3)` removes the most
recently found point at each step, so the nest runs back through time: each
step holds the places visited by some date, and the curve shows how the
sampled area and its species list grew with effort. `"oldest"` removes the
earliest places first. A point's date is that of its first observation
(`sources.dataframe(df, date_col="date")`; `sources.inaturalist`,
`sources.open_data` and `sources.gbif` keep theirs).

### Density peeling

`strategies.density_peel(sample, bandwidth_km=None, min_points=3)` smooths the
density of points once with a Gaussian kernel and removes the point of lowest
density first. Each step keeps a superlevel set of that density, so the
boundary follows where the observations are, not the outermost few, and
isolated records go first. The default bandwidth uses Scott's rule.

`method="voronoi"` measures density without a bandwidth: a point's density
is one over the area of its Voronoi cell, the ground nearer to it than to
any other point, within the sample's hull. The points that each stand for
the most ground, where effort was thinnest, go first.

### Growing from a seed

`strategies.grow(sample, seed=None, min_points=3)` grows a region from the
point nearest `seed` by adding, each time, the point nearest the region
(Prim's algorithm), and removes points in the reverse of that order. The last
points left are around the seed, and the region follows dense corridors of
observations rather than convex shapes.

## Several patches at once

One hull around a sample of separate clusters (two valleys, a chain of
reserves, an archipelago) spans the empty ground between them, and that
ground is most of its area. `strategies.split` nests the clusters as
separate patches instead:

```python
nest = samplesar.strategies.split(sample, "hull_peel", parts=4)
nest.to_frame()   # area: the four hulls together; n_parts: parts with points left
```

It splits the sample into parts, runs the named strategy inside each, and
joins their steps into one nest. Each step's area is the ground its parts'
hulls cover together (overlaps counted once, and within the mask if the
sample has one), so the curve is that of several patches shrinking at once.
`plot.nest` draws every part's hull, and `evaluate.diagnostics` describes
them together.

`by` chooses the split, and `strategies.partition(sample, parts, by)` returns
it on its own, as one label per point:

- `"kmeans"` (the default): k-means clusters of the projected points. Give
  a `seed`: without one the clusters can differ from run to run.
- `"grid"`: equal rectangles over the points' bounding box, in the most
  nearly square layout with at least `parts` cells, or
  `parts=(columns, rows)`.
- labels you give, one per point.
- (longitude, latitude) seeds: each point joins its nearest seed.
- polygons (shapely, or a GeoSeries or GeoDataFrame): ecoregions, islands,
  counties.

`pace` sets how the parts shrink together. With `"even"` (the default) each
part goes at its own pace and all reach their last step together; with
`"turn"` each part takes one step per round, so parts with few steps finish
early. A part with no more than `min_points` points stays whole. For one
curve per part instead of one curve for all, nest each part yourself:
`strategies.hull_peel(sample.subset(np.flatnonzero(labels == 0)))`.

## Thinning before nesting

Observation data pile up where people go. `sample.thin(km)` keeps one point
in each occupied grid cell of that size (the one nearest the cell's center,
or with `seed` a random one), so no neighborhood holds more than its share
and a nest doesn't spend most of its steps inside one trailhead:

```python
even = sample.thin(1.0)                      # at most one point per km²
nest = samplesar.strategies.hull_peel(even)
```

The points not kept are dropped with their observations, as spatial
thinning of occurrence records usually is, so species seen only at dropped
points are lost. The cells start at the sample's own south-west corner, so
thin the whole sample once, before taking parts of it. To keep every observation and only merge points a few
meters apart, use `Sample.merge_within`.

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

## One curve from many random nests

A single random nest is one draw. `evaluate.ensemble` runs many and reads
each one's species count at the same areas, giving the strategy's expected
curve and the band a single nest falls in:

```python
band = samplesar.evaluate.ensemble(sample, "hull_random", n=99)
ax.fill_between(band["area"], band["q025"], band["q975"], alpha=0.3)
ax.plot(band["area"], band["mean"])
```

## How much a fit owes to the points sampled

A deterministic strategy gives one nest per sample. `evaluate.bootstrap`
draws the sample's points with replacement, nests each resample with the
same strategy and fits it, so the spread of the fits shows how sensitive
the result is to which places happened to be sampled:

```python
fits = samplesar.evaluate.bootstrap(sample, "hull_peel", n=99)
fits["z"].quantile([0.025, 0.5, 0.975])
```

Steps within a nest aren't independent, so this spread describes
sensitivity; it isn't a confidence interval of the usual kind.

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
