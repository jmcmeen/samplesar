"""Nesting strategies that remove points from the convex hull inwards."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import numpy as np
import pandas as pd
import shapely
from shapely.geometry import MultiPoint, Polygon

from samplesar._nest import Nest, check_size
from samplesar._registry import register_strategy
from samplesar._sample import Sample, hull_area


def _hull(points: np.ndarray, idx: np.ndarray) -> shapely.Geometry:
    return MultiPoint(points[idx]).convex_hull


def hull_vertices(points: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """The rows of ``idx`` whose points are vertices of their convex hull.

    Points on an edge between two vertices aren't vertices. When the points
    are collinear the hull is a segment and its vertices are its two ends.
    Returned in the order of ``idx``.
    """
    hull = _hull(points, idx)
    coords = hull.exterior.coords if isinstance(hull, Polygon) else hull.coords
    corners = {tuple(c) for c in coords}
    return np.array([i for i in idx if tuple(points[i]) in corners], dtype=int)


def _on_boundary(points: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """The rows of ``idx`` on the boundary of their convex hull: the vertices
    and any points on its edges. A degenerate hull (a point or a segment) has
    every point on it."""
    hull = _hull(points, idx)
    if not isinstance(hull, Polygon):
        return idx.copy()
    return idx[shapely.intersects(hull.boundary, shapely.points(points[idx]))]


def _remove_one(
    sample: Sample, min_points: int, choose: Callable[[np.ndarray, np.ndarray], int]
) -> Nest:
    """Drop one hull vertex per step: ``choose(remaining, vertices)`` names it."""
    check_size(sample, min_points)
    points = sample.points
    remaining = np.arange(sample.n_points)
    steps = [remaining.copy()]
    while len(remaining) > min_points:
        drop = choose(remaining, hull_vertices(points, remaining))
        remaining = remaining[remaining != drop]
        steps.append(remaining.copy())
    return Nest(sample=sample, steps=tuple(steps))


@register_strategy("hull_random")
def hull_random_nest(sample: Sample, min_points: int = 3, seed: int | None = None) -> Nest:
    """Nest a sample by removing a random vertex of its convex hull at each step.

    Like `random_nest`, but only points on the outside can go, so the sample
    shrinks from its edges inwards.

    Parameters
    ----------
    sample : Sample
        The sample to nest.
    min_points : int
        Stop when this many points remain.
    seed : int, optional
        Seed for the random choices, for a reproducible nest.

    Returns
    -------
    Nest

    Raises
    ------
    ValueError
        If ``min_points`` is below 1 or the sample has fewer points.
    """
    rng = np.random.default_rng(seed)
    return _remove_one(sample, min_points, lambda _, vertices: int(rng.choice(vertices)))


@register_strategy("hull_peel")
def hull_peel_nest(sample: Sample, min_points: int = 3) -> Nest:
    """Nest a sample by peeling off its whole convex hull at each step.

    Each step removes every point on the hull's boundary, the vertices and
    any points on its edges, leaving the points strictly inside: the sample's
    convex layers, or onion peeling. A step can remove many points, so the
    nest has far fewer steps than the one-at-a-time strategies. It stops
    before a layer would leave fewer than ``min_points`` points.

    Parameters
    ----------
    sample : Sample
        The sample to nest.
    min_points : int
        Stop rather than leave fewer than this many points.

    Returns
    -------
    Nest

    Raises
    ------
    ValueError
        If ``min_points`` is below 1, the sample has fewer points, or no layer
        can be peeled (every point but fewer than ``min_points`` lies on the
        hull).
    """
    check_size(sample, min_points)
    points = sample.points
    remaining = np.arange(sample.n_points)
    steps = [remaining.copy()]
    while True:
        inside = remaining[~np.isin(remaining, _on_boundary(points, remaining))]
        if len(inside) < min_points:
            break
        remaining = inside
        steps.append(remaining.copy())
    if len(steps) == 1:
        raise ValueError(
            f"No layer can be peeled: fewer than {min_points} points lie strictly inside the "
            "sample's convex hull."
        )
    return Nest(sample=sample, steps=tuple(steps))


def _centre_distance(points: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    """Each vertex's distance from the centroid of the vertices (degrees)."""
    corners = points[vertices]
    return np.hypot(*(corners - corners.mean(axis=0)).T)


def _near_max(values: np.ndarray) -> np.ndarray:
    """Which values tie for the maximum, allowing for floating-point rounding:
    geometrically equal areas or distances differ in their last bits once the
    points sit at real longitudes and latitudes."""
    top = values.max()
    return values >= top - 1e-9 * max(abs(float(top)), float(np.abs(values).max()), 1e-300)


def _pick(vertices: np.ndarray, score: np.ndarray, distance: np.ndarray) -> int:
    """The vertex with the highest score; ties (to within rounding) go to the
    one farthest from the centre, then to the lowest point index."""
    tied = np.flatnonzero(_near_max(score))
    farthest = tied[_near_max(distance[tied])]
    return int(vertices[farthest].min())


@register_strategy("hull_area")
def hull_area_nest(
    sample: Sample, keep: Literal["largest", "smallest"] = "largest", min_points: int = 3
) -> Nest:
    """Nest a sample by removing the hull vertex that changes its area least or most.

    At each step every vertex of the current convex hull is tried, and the
    one whose removal leaves the largest hull (``keep="largest"``, the
    smallest loss of area) or the smallest hull (``keep="smallest"``, the
    largest loss) goes. Ties, to within floating-point rounding, go to the
    vertex farthest from the centroid of the hull's vertices, then to the
    lowest point index, so the nest is deterministic and doesn't depend on
    where the sample lies.

    Parameters
    ----------
    sample : Sample
        The sample to nest.
    keep : {"largest", "smallest"}
        Which remaining hull to keep at each step.
    min_points : int
        Stop when this many points remain.

    Returns
    -------
    Nest

    Raises
    ------
    ValueError
        If ``keep`` is unknown, ``min_points`` is below 1 or the sample has
        fewer points.
    """
    if keep not in ("largest", "smallest"):
        raise ValueError(f"keep must be 'largest' or 'smallest', got {keep!r}")
    sign = 1.0 if keep == "largest" else -1.0
    points = sample.points

    def choose(remaining: np.ndarray, vertices: np.ndarray) -> int:
        areas = np.array([hull_area(points[remaining[remaining != v]]) for v in vertices])
        return _pick(vertices, sign * areas, _centre_distance(points, vertices))

    return _remove_one(sample, min_points, choose)


def _point_species(sample: Sample) -> tuple[list[np.ndarray], int]:
    """The species codes found at each of the sample's points, and the number
    of species."""
    data = sample.data
    codes, uniques = pd.factorize(data["species"])
    lookup = {(x, y): i for i, (x, y) in enumerate(map(tuple, sample.points))}
    which = np.array([lookup[(x, y)] for x, y in zip(data["lon"], data["lat"], strict=True)])
    grouped = pd.Series(codes).groupby(which).unique()
    return [np.asarray(grouped[i], dtype=int) for i in range(sample.n_points)], len(uniques)


@register_strategy("hull_species")
def hull_species_nest(
    sample: Sample, keep: Literal["most", "fewest"] = "most", min_points: int = 3
) -> Nest:
    """Nest a sample by removing the hull vertex that loses the fewest or most species.

    At each step every vertex of the current convex hull is tried, and the
    one whose removal leaves the most species (``keep="most"``: a vertex
    whose species all occur elsewhere costs nothing) or the fewest
    (``keep="fewest"``: a vertex holding species found nowhere else goes
    first) is removed. Ties go to the vertex farthest from the centroid of
    the hull's vertices, then to the lowest point index, so the nest is
    deterministic.

    Parameters
    ----------
    sample : Sample
        The sample to nest.
    keep : {"most", "fewest"}
        Which remaining species count to keep at each step.
    min_points : int
        Stop when this many points remain.

    Returns
    -------
    Nest

    Raises
    ------
    ValueError
        If ``keep`` is unknown, ``min_points`` is below 1 or the sample has
        fewer points.
    """
    if keep not in ("most", "fewest"):
        raise ValueError(f"keep must be 'most' or 'fewest', got {keep!r}")
    sign = 1.0 if keep == "most" else -1.0
    points = sample.points
    species, n_species = _point_species(sample)
    # How many remaining points hold each species: removing a point loses the
    # species it is the last holder of.
    holders = np.zeros(n_species, dtype=int)
    for found in species:
        holders[found] += 1

    def choose(remaining: np.ndarray, vertices: np.ndarray) -> int:
        lost = np.array([np.count_nonzero(holders[species[v]] == 1) for v in vertices])
        drop = _pick(vertices, -sign * lost, _centre_distance(points, vertices))
        holders[species[drop]] -= 1
        return drop

    return _remove_one(sample, min_points, choose)


@register_strategy("hull_distance")
def hull_distance_nest(
    sample: Sample, remove: Literal["farthest", "closest"] = "farthest", min_points: int = 3
) -> Nest:
    """Nest a sample by removing the hull vertex farthest from, or closest to, its centre.

    The centre is the centroid of the current hull's vertices, and distances
    are Euclidean in degrees: a degree of longitude counts as much as a
    degree of latitude, as in the hull areas. ``remove="farthest"`` trims the
    outlying corners first; ``remove="closest"`` removes the vertex nearest
    the centre, leaving the outliers to last. Ties, to within floating-point
    rounding, go to the lowest point index.

    Parameters
    ----------
    sample : Sample
        The sample to nest.
    remove : {"farthest", "closest"}
        Which vertex to remove at each step.
    min_points : int
        Stop when this many points remain.

    Returns
    -------
    Nest

    Raises
    ------
    ValueError
        If ``remove`` is unknown, ``min_points`` is below 1 or the sample has
        fewer points.
    """
    if remove not in ("farthest", "closest"):
        raise ValueError(f"remove must be 'farthest' or 'closest', got {remove!r}")
    sign = 1.0 if remove == "farthest" else -1.0
    points = sample.points

    def choose(remaining: np.ndarray, vertices: np.ndarray) -> int:
        distance = _centre_distance(points, vertices)
        return _pick(vertices, sign * distance, np.zeros(len(vertices)))

    return _remove_one(sample, min_points, choose)
