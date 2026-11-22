# Sampling strategies

Each strategy takes a `Sample` and returns a `Nest`: a sequence of ever-smaller
subsets of the sample's points, stopping when `min_points` remain. Strategies
differ only in which point they drop next.

## Random removal

`random_nest(sample, min_points=3, seed=None)` removes one point at a time,
uniformly at random. It's the baseline every other strategy is compared
against.

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
strictly inside. These are the sample's convex layers (onion peeling). A step
can remove many points, so the nest has few steps; on a regular grid each step
takes off the outer ring. It stops before a layer would leave fewer than
`min_points` points.
