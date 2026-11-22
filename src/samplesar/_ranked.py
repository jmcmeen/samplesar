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
    metres = projection.axis_info[0].unit_conversion_factor
    return np.array([x, y]) * metres / 1000.0


def _centre_xy(sample: Sample, centre: tuple[float, float] | None) -> np.ndarray:
    if centre is None:
        return np.asarray(sample.xy.mean(axis=0))
    lon, lat = centre
    if not (-180 <= lon <= 180 and -90 <= lat <= 90):
        raise ValueError(f"centre must be (longitude, latitude) in degrees, got {centre}.")
    return _to_xy(sample, lon, lat)


def _by_rank(sample: Sample, key: np.ndarray, min_points: int) -> Nest:
    """Remove points in increasing ``key``, ties by index."""
    order = np.lexsort((np.arange(sample.n_points), key))
    return from_order(sample, order, min_points=min_points)


@register("radial")
def radial(sample: Sample, centre: tuple[float, float] | None = None, min_points: int = 3) -> Nest:
    """Nest a sample by removing the point farthest from a centre at each step.

    The sample shrinks as concentric discs around the centre, the point-data
    analogue of nested circular plots around a focal site.

    Parameters
    ----------
    sample : Sample
        The sample to nest.
    centre : tuple of float, optional
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
        If ``centre`` isn't a longitude and latitude, or ``min_points`` is
        below 1 or above the sample's size.
    """
    check_size(sample, min_points)
    distance = np.hypot(*(sample.xy - _centre_xy(sample, centre)).T)
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
    along = sample.xy[:, 0] * np.sin(angle) + sample.xy[:, 1] * np.cos(angle)
    return _by_rank(sample, along, min_points)


def scott_bandwidth(xy: np.ndarray) -> float:
    """Scott's rule for a two-dimensional Gaussian kernel, in km: the mean of
    the coordinates' standard deviations times n^(-1/6)."""
    spread = float(np.mean(np.std(xy, axis=0)))
    return spread * len(xy) ** (-1.0 / 6.0) if spread > 0 else 1.0


def kernel_density(xy: np.ndarray, bandwidth: float) -> np.ndarray:
    """Each point's Gaussian kernel density from all the points (unnormalised),
    summed over neighbours within five bandwidths."""
    from scipy.spatial import cKDTree

    tree = cKDTree(xy)
    pairs = tree.query_pairs(5.0 * bandwidth, output_type="ndarray")
    density = np.ones(len(xy))  # each point's own kernel
    if len(pairs):
        d2 = np.sum((xy[pairs[:, 0]] - xy[pairs[:, 1]]) ** 2, axis=1)
        weight = np.exp(-0.5 * d2 / bandwidth**2)
        np.add.at(density, pairs[:, 0], weight)
        np.add.at(density, pairs[:, 1], weight)
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
    """Each point's neighbours on the Euclidean minimum spanning tree, with
    distances: from the Delaunay triangulation, or from all pairs when the
    points are collinear or too few."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import minimum_spanning_tree
    from scipy.spatial import Delaunay, QhullError

    n = len(xy)
    try:
        if n < 4:
            raise QhullError("too few points")
        simplices = Delaunay(xy).simplices
        pairs = np.vstack([simplices[:, [0, 1]], simplices[:, [1, 2]], simplices[:, [0, 2]]])
    except QhullError:
        i, j = np.triu_indices(n, 1)
        pairs = np.column_stack([i, j])
    pairs = np.unique(np.sort(pairs, axis=1), axis=0)
    weight = np.hypot(*(xy[pairs[:, 0]] - xy[pairs[:, 1]]).T)
    tree = minimum_spanning_tree(coo_matrix((weight, (pairs[:, 0], pairs[:, 1])), shape=(n, n)))
    tree = tree.tocoo()
    edges: list[list[tuple[float, int]]] = [[] for _ in range(n)]
    for a, b, w in zip(tree.row, tree.col, tree.data, strict=True):
        edges[a].append((float(w), int(b)))
        edges[b].append((float(w), int(a)))
    return edges


def growth_order(xy: np.ndarray, seed: int) -> np.ndarray:
    """The order Prim's algorithm adds points to a region grown from ``seed``,
    each time the point nearest the region (ties by index). Prim's choices
    are edges of the minimum spanning tree, so it runs on that tree."""
    edges = _spanning_edges(xy)
    added = np.zeros(len(xy), dtype=bool)
    order = []
    frontier: list[tuple[float, int]] = [(0.0, seed)]
    while frontier:
        _, point = heapq.heappop(frontier)
        if added[point]:
            continue
        added[point] = True
        order.append(point)
        for w, other in edges[point]:
            if not added[other]:
                heapq.heappush(frontier, (w, other))
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
    distance = np.hypot(*(xy - _centre_xy(sample, seed)).T)
    start = int(np.lexsort((np.arange(len(xy)), distance))[0])
    return from_order(sample, growth_order(xy, start)[::-1], min_points=min_points)
