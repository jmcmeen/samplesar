"""Strategies that rank every point once and remove them in that order.

Each is fast (no hull is re-measured per step) and has its own geometric
meaning: concentric discs, a moving front, a smoothed density boundary, or a
region grown from a seed. Ties are broken by point index.
"""

from __future__ import annotations

import heapq

import numpy as np
from pyproj import Transformer

from samplesar._nest import Nest, check_size, from_order
from samplesar._registry import register
from samplesar._sample import WGS84, Sample


def _to_xy(sample: Sample, lon: float, lat: float) -> np.ndarray:
    """A longitude and latitude in the sample's projected km."""
    projection = sample.projection
    x, y = Transformer.from_crs(WGS84, projection, always_xy=True).transform(lon, lat)
    meters = projection.axis_info[0].unit_conversion_factor
    return np.array([x, y]) * meters / 1000.0


def _center_xy(sample: Sample, center: tuple[float, float] | None) -> np.ndarray:
    if center is None:
        return np.asarray(sample.xy.mean(axis=0))
    lon, lat = center
    if not (-180 <= lon <= 180 and -90 <= lat <= 90):
        raise ValueError(f"center must be (longitude, latitude) in degrees, got {center}.")
    return _to_xy(sample, lon, lat)


def _tied(key: np.ndarray) -> np.ndarray:
    """``key`` rounded to a billionth of its range, so values equal but for
    floating-point rounding (points round-tripped through longitude and
    latitude) tie, and the tie goes by index."""
    key = np.asarray(key, dtype=float)
    span = float(np.ptp(key)) if key.size else 0.0
    if span == 0:
        return np.zeros_like(key)
    return np.round((key - key.min()) / span * 1e9)


def _by_rank(sample: Sample, key: np.ndarray, min_points: int) -> Nest:
    """Remove points in increasing ``key``, ties by index."""
    order = np.lexsort((np.arange(sample.n_points), _tied(key)))
    return from_order(sample, order, min_points=min_points)


@register("radial")
def radial(sample: Sample, center: tuple[float, float] | None = None, min_points: int = 3) -> Nest:
    """Nest a sample by removing the point farthest from a center at each step.

    The sample shrinks as concentric discs around the center, the point-data
    analogue of nested circular plots around a focal site.

    Parameters
    ----------
    sample : Sample
        The sample to nest.
    center : tuple of float, optional
        The (longitude, latitude) to shrink towards; by default the mean of
        the projected points.
    min_points : int
        Stop when this many points remain.

    Returns
    -------
    Nest

    Raises
    ------
    ValueError
        If ``center`` isn't a longitude and latitude, or ``min_points`` is
        below 1 or above the sample's size.
    """
    check_size(sample, min_points)
    distance = np.hypot(*(sample.xy - _center_xy(sample, center)).T)
    return _by_rank(sample, -distance, min_points)


@register("sweep")
def sweep(sample: Sample, bearing: float = 90.0, min_points: int = 3) -> Nest:
    """Nest a sample by sweeping a straight front across it.

    Points are removed in order along ``bearing``: with the default 90°, the
    front moves east, removing the westernmost point first, like a transect
    walked from one end or a range edge retreating.

    Parameters
    ----------
    sample : Sample
        The sample to nest.
    bearing : float
        The direction the front moves, in degrees clockwise from grid north
        of the sample's projection.
    min_points : int
        Stop when this many points remain.

    Returns
    -------
    Nest

    Raises
    ------
    ValueError
        If ``bearing`` isn't finite, or ``min_points`` is below 1 or above the
        sample's size.
    """
    check_size(sample, min_points)
    if not np.isfinite(bearing):
        raise ValueError(f"bearing must be a finite number of degrees, got {bearing}.")
    angle = np.radians(bearing)
    # Rounded so that bearings like 90° give exact 0 and 1, not 6e-17.
    east, north = np.round(np.sin(angle), 15), np.round(np.cos(angle), 15)
    along = sample.xy[:, 0] * east + sample.xy[:, 1] * north
    return _by_rank(sample, along, min_points)


def scott_bandwidth(xy: np.ndarray) -> float:
    """Scott's rule for a two-dimensional Gaussian kernel, in km: the mean of
    the coordinates' standard deviations times n^(-1/6)."""
    spread = float(np.mean(np.std(xy, axis=0)))
    return spread * len(xy) ** (-1.0 / 6.0) if spread > 0 else 1.0


BLOCK = 1024  # points whose neighbors are summed at once, to bound memory


def kernel_density(xy: np.ndarray, bandwidth: float) -> np.ndarray:
    """Each point's Gaussian kernel density from all the points (unnormalized,
    its own kernel included), summed over neighbors within five bandwidths,
    a block of points at a time so memory stays bounded."""
    from scipy.spatial import cKDTree

    tree = cKDTree(xy)
    density = np.empty(len(xy))
    for start in range(0, len(xy), BLOCK):
        block = cKDTree(xy[start : start + BLOCK])
        pairs = block.sparse_distance_matrix(tree, 5.0 * bandwidth, output_type="ndarray")
        density[start : start + block.n] = np.bincount(
            pairs["i"], weights=np.exp(-0.5 * (pairs["v"] / bandwidth) ** 2), minlength=block.n
        )
    return density


@register("density_peel")
def density_peel(sample: Sample, bandwidth_km: float | None = None, min_points: int = 3) -> Nest:
    """Nest a sample by removing the point of lowest kernel density at each step.

    The density of points is smoothed once with a Gaussian kernel, and points
    go in order of increasing density, so each step keeps a superlevel set of
    the smoothed density: the boundary follows where the observations are,
    not the outermost few, and isolated records go first.

    Parameters
    ----------
    sample : Sample
        The sample to nest.
    bandwidth_km : float, optional
        The kernel's standard deviation in km; by default Scott's rule.
    min_points : int
        Stop when this many points remain.

    Returns
    -------
    Nest

    Raises
    ------
    ValueError
        If ``bandwidth_km`` isn't positive, or ``min_points`` is below 1 or
        above the sample's size.
    """
    check_size(sample, min_points)
    if bandwidth_km is None:
        bandwidth_km = scott_bandwidth(sample.xy)
    elif not (np.isfinite(bandwidth_km) and bandwidth_km > 0):
        raise ValueError(f"bandwidth_km must be positive, got {bandwidth_km}.")
    return _by_rank(sample, kernel_density(np.asarray(sample.xy), bandwidth_km), min_points)


def _spanning_edges(xy: np.ndarray) -> list[list[tuple[float, int]]]:
    """Each point's neighbors on the Euclidean minimum spanning tree of
    distinct points, with distances: from the Delaunay triangulation; along
    the line for collinear points; from all pairs for fewer than four."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import minimum_spanning_tree
    from scipy.spatial import Delaunay, QhullError

    n = len(xy)
    if n < 4:
        i, j = np.triu_indices(n, 1)
        pairs = np.column_stack([i, j])
    else:
        try:
            simplices = Delaunay(xy).simplices
            pairs = np.vstack([simplices[:, [0, 1]], simplices[:, [1, 2]], simplices[:, [0, 2]]])
        except QhullError:  # collinear: the tree joins neighbors along the line
            centered = xy - xy.mean(axis=0)
            direction = np.linalg.svd(centered, full_matrices=False)[2][0]
            along = np.argsort(centered @ direction, kind="stable")
            pairs = np.column_stack([along[:-1], along[1:]])
    edges: list[list[tuple[float, int]]] = [[] for _ in range(n)]
    if n < 2:
        return edges
    pairs = np.unique(np.sort(pairs, axis=1), axis=0)
    weight = np.hypot(*(xy[pairs[:, 0]] - xy[pairs[:, 1]]).T)
    tree = minimum_spanning_tree(coo_matrix((weight, (pairs[:, 0], pairs[:, 1])), shape=(n, n)))
    tree = tree.tocoo()
    for a, b, w in zip(tree.row, tree.col, tree.data, strict=True):
        edges[a].append((float(w), int(b)))
        edges[b].append((float(w), int(a)))
    return edges


def growth_order(xy: np.ndarray, seed: int) -> np.ndarray:
    """The order Prim's algorithm adds points to a region grown from ``seed``,
    each time the point nearest the region (ties by index). Prim's choices
    are edges of the minimum spanning tree, so it runs on that tree, built on
    the distinct positions; points at one position are added together, in
    index order, the seed first."""
    positions, where = np.unique(xy, axis=0, return_inverse=True)
    where = np.asarray(where).ravel()
    edges = _spanning_edges(positions)
    by_position = np.argsort(where, kind="stable")  # each position's points, in index order
    bounds = np.searchsorted(where[by_position], np.arange(len(positions) + 1))
    first = by_position[bounds[:-1]]  # each position's lowest point index, for ties
    added = np.zeros(len(positions), dtype=bool)
    order: list[int] = []
    frontier: list[tuple[float, int, int]] = [(0.0, seed, int(where[seed]))]
    while frontier:
        _, _, position = heapq.heappop(frontier)
        if added[position]:
            continue
        added[position] = True
        members = by_position[bounds[position] : bounds[position + 1]]
        if position == where[seed]:
            members = np.concatenate([[seed], members[members != seed]])
        order.extend(int(m) for m in members)
        for w, other in edges[position]:
            if not added[other]:
                heapq.heappush(frontier, (w, int(first[other]), other))
    if len(order) != len(xy):  # the tree spans every position
        raise RuntimeError("The spanning tree missed points; please report this.")
    return np.asarray(order)


@register("grow")
def grow(sample: Sample, seed: tuple[float, float] | None = None, min_points: int = 3) -> Nest:
    """Nest a sample by reversing a region grown from a seed point.

    A region is grown from the point nearest ``seed`` by adding, each time,
    the point nearest the region (Prim's algorithm); the nest removes points
    in the reverse of that order. The last points left are around the seed,
    and the region follows dense corridors of observations rather than
    convex shapes.

    Parameters
    ----------
    sample : Sample
        The sample to nest.
    seed : tuple of float, optional
        The (longitude, latitude) to grow from; by default the mean of the
        projected points. The region starts at the point nearest it.
    min_points : int
        Stop when this many points remain.

    Returns
    -------
    Nest

    Raises
    ------
    ValueError
        If ``seed`` isn't a longitude and latitude, or ``min_points`` is below
        1 or above the sample's size.
    """
    check_size(sample, min_points)
    xy = np.asarray(sample.xy)
    distance = np.hypot(*(xy - _center_xy(sample, seed)).T)
    start = int(np.lexsort((np.arange(len(xy)), distance))[0])
    return from_order(sample, growth_order(xy, start)[::-1], min_points=min_points)
