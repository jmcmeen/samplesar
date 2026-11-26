"""Hull reduction by how rare each vertex's species are."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import numpy as np
import pandas as pd

from samplesar._hull import _center_distance, _pick, _remove_one
from samplesar._nest import Nest
from samplesar._registry import register
from samplesar._sample import Sample, point_species
from samplesar._sample import hull_area as convex_area

Measure = Literal["occupancy", "abundance", "range"]
Score = Literal["sum", "mean", "max"]

_SCORES: dict[str, Callable[[np.ndarray], float]] = {"sum": np.sum, "mean": np.mean, "max": np.max}


class _Commonness:
    """How common each species is among some of the sample's points: the
    number of points holding it (occupancy), its observations (abundance) or
    the area of its own convex hull in km² (range)."""

    def __init__(self, sample: Sample, measure: str) -> None:
        self.measure = measure
        self.xy = np.asarray(sample.xy)
        self.species, self.n_species = point_species(sample)
        n = sample.n_points
        # Each point's observations of each of its species, in the order of
        # ``self.species``.
        data = sample.data
        codes, _ = pd.factorize(data["species"])
        keys = pd.MultiIndex.from_frame(data[["lon", "lat"]])
        rows = pd.MultiIndex.from_arrays([sample.points[:, 0], sample.points[:, 1]]).get_indexer(
            keys
        )
        counts = pd.Series(1, index=pd.MultiIndex.from_arrays([rows, codes])).groupby(level=[0, 1])
        table = counts.sum()
        self.observations = [
            np.array([table[(p, s)] for s in self.species[p]], dtype=float) for p in range(n)
        ]
        self.holders: list[set[int]] = [set() for _ in range(self.n_species)]
        for p, found in enumerate(self.species):
            for s in found:
                self.holders[s].add(p)
        self.value = np.zeros(self.n_species)
        for p in range(n):
            if measure == "occupancy":
                self.value[self.species[p]] += 1
            elif measure == "abundance":
                self.value[self.species[p]] += self.observations[p]
        if measure == "range":
            for s in range(self.n_species):
                self.value[s] = self._range(s)

    def _range(self, s: int) -> float:
        return convex_area(self.xy[sorted(self.holders[s])])

    def remove(self, point: int) -> None:
        """Take a point's observations out of every measure."""
        found = self.species[point]
        for s in found:
            self.holders[s].discard(point)
        if self.measure == "occupancy":
            self.value[found] -= 1
        elif self.measure == "abundance":
            self.value[found] -= self.observations[point]
        else:
            for s in found:
                self.value[s] = self._range(s)

    def rarity(self) -> np.ndarray:
        """Each species' rarity: the inverse of its commonness. A range of no
        area (a species at one or two points, or along a line) counts as the
        smallest positive range, or 1 km² if there is none."""
        value = self.value
        positive = value[value > 0]
        floor = float(positive.min()) if positive.size else 1.0
        return 1.0 / np.maximum(value, floor)


def rarity_weights(score: np.ndarray, power: float, rare_first: bool) -> np.ndarray:
    """Probabilities proportional to ``score ** power`` (rare first) or its
    inverse (common first), computed on scores scaled into (0, 1] so a large
    power neither underflows to nothing nor overflows."""
    scaled = score / score.max() if rare_first else score.min() / score
    weight = scaled**power
    return np.asarray(weight / weight.sum())


@register("hull_rarity")
def hull_rarity(
    sample: Sample,
    remove: Literal["common", "rare"] = "common",
    measure: Measure = "occupancy",
    score: Score = "sum",
    update: bool = False,
    power: float | None = None,
    min_points: int = 3,
    seed: int | None = None,
) -> Nest:
    """Nest a sample by removing the hull vertex whose species are most common, or rarest.

    Each species has a rarity, the inverse of how common it is in the
    sample, and each vertex of the current hull a score from the rarities of
    the species observed at it. With ``score="sum"`` and
    ``measure="occupancy"`` the score is rarity-weighted richness: each
    species counts one over the number of points it occurs at.
    ``remove="common"`` takes the lowest-scoring vertex, keeping the sample's
    rare species to the end; ``remove="rare"`` takes the highest, showing how
    fast a shrinking region loses its distinctive species.
    [`hull_species`][samplesar.strategies.hull_species] differs: it asks only
    whether a vertex holds a species found at no other remaining point.

    Ties, to within floating-point rounding, go to the vertex farthest from
    the centroid of the hull's vertices, then to the lowest point index.

    Parameters
    ----------
    sample : Sample
        The sample to nest.
    remove : {"common", "rare"}
        Remove the vertex whose species are most common, or rarest.
    measure : {"occupancy", "abundance", "range"}
        What makes a species common: the number of points it occurs at, its
        number of observations, or the area of the convex hull of its own
        points (km²; a species without a hull of positive area counts as the
        smallest such range in the sample).
    score : {"sum", "mean", "max"}
        How a vertex's species are combined. ``"sum"`` favors rich vertices
        as well as rare species; ``"mean"`` and ``"max"`` don't.
    update : bool
        Measure commonness among the points still remaining at each step,
        instead of once on the whole sample.
    power : float, optional
        Choose a vertex at random instead, with probability proportional to
        its score raised to ``power`` (``remove="rare"``) or to the inverse
        of that (``remove="common"``); 0 is
        [`hull_random`][samplesar.strategies.hull_random], and a large power
        approaches the deterministic choice.
    min_points : int
        Stop when this many points remain.
    seed : int, optional
        Seed for the random choices when ``power`` is given.

    Returns
    -------
    Nest

    Raises
    ------
    ValueError
        If ``remove``, ``measure`` or ``score`` is unknown, ``power`` is
        negative or not finite, ``seed`` is given without ``power``, or
        ``min_points`` is below 1 or the sample has fewer points.
    """
    if remove not in ("common", "rare"):
        raise ValueError(f"remove must be 'common' or 'rare', got {remove!r}")
    if measure not in ("occupancy", "abundance", "range"):
        raise ValueError(f"measure must be 'occupancy', 'abundance' or 'range', got {measure!r}")
    if score not in _SCORES:
        raise ValueError(f"score must be 'sum', 'mean' or 'max', got {score!r}")
    if power is not None and not (np.isfinite(power) and power >= 0):
        raise ValueError(f"power must be a finite number of at least 0, got {power}.")
    if power is None and seed is not None:
        raise ValueError("seed is only used with power; without it the choice is deterministic.")
    rare_first = remove == "rare"
    combine = _SCORES[score]
    points = sample.xy
    commonness = _Commonness(sample, measure)
    rarity = commonness.rarity()
    rng = np.random.default_rng(seed)

    def choose(remaining: np.ndarray, vertices: np.ndarray) -> int:
        nonlocal rarity
        scores = np.array([combine(rarity[commonness.species[v]]) for v in vertices])
        if power is None:
            sign = 1.0 if rare_first else -1.0
            drop = _pick(vertices, sign * scores, _center_distance(points, vertices))
        elif power == 0:  # the same draws as hull_random
            drop = int(rng.choice(vertices))
        else:
            drop = int(rng.choice(vertices, p=rarity_weights(scores, power, rare_first)))
        if update:
            commonness.remove(drop)
            rarity = commonness.rarity()
        return drop

    return _remove_one(sample, min_points, choose)
