"""Nesting strategies that remove points from the convex hull inwards."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import shapely
from shapely.geometry import MultiPoint, Polygon

from samplesar._nest import Nest
from samplesar._sample import Sample


def _check(sample: Sample, min_points: int) -> None:
    if min_points < 1:
        raise ValueError(f"min_points must be at least 1, got {min_points}")
    if sample.n_points < min_points:
        raise ValueError(f"Sample has {sample.n_points} points; need at least {min_points}")


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
    _check(sample, min_points)
    points = sample.points
    remaining = np.arange(sample.n_points)
    steps = [remaining.copy()]
    while len(remaining) > min_points:
        drop = choose(remaining, hull_vertices(points, remaining))
        remaining = remaining[remaining != drop]
        steps.append(remaining.copy())
    return Nest(sample=sample, steps=tuple(steps))


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
        If ``min_points`` is below 1 or the sample has fewer points.
    """
    _check(sample, min_points)
    points = sample.points
    remaining = np.arange(sample.n_points)
    steps = [remaining.copy()]
    while True:
        inside = remaining[~np.isin(remaining, _on_boundary(points, remaining))]
        if len(inside) < min_points:
            break
        remaining = inside
        steps.append(remaining.copy())
    return Nest(sample=sample, steps=tuple(steps))
