# Sampling strategies

Each strategy takes a `Sample` and returns a `Nest`: a sequence of ever-smaller
subsets of the sample's points, stopping when `min_points` remain. Strategies
differ only in which point they drop next.

## Random removal

`random_nest(sample, min_points=3, seed=None)` removes one point at a time,
uniformly at random. It's the baseline every other strategy is compared
against.
