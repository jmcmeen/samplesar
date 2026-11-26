"""Strategies that rank every point once and remove them in that order.

Each is fast (no hull is re-measured per step) and has its own geometric
meaning: concentric discs or ellipses, a moving front, a narrowing band,
compass sectors, a smoothed density boundary, or a region grown from a seed.
Ties are broken by point index.
"""

from __future__ import annotations

import heapq
import numbers
from typing import Any, Literal

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


def _finite(name: str, value: Any) -> float:
    """``value`` as a float, or a ValueError naming the option if it isn't a
    finite real number (True and False aren't numbers here)."""
    if isinstance(value, bool | np.bool_) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a finite number, got {value!r}.")
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{name} must be a finite number, got {value!r}.")
    return number


def _along_across(
    sample: Sample, bearing: float, origin: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Each point's offset from ``origin`` along a bearing and across it, in km."""
    angle = np.radians(_finite("bearing", bearing))
    # Rounded so that bearings like 90° give exact 0 and 1, not 6e-17.
    east, north = np.round(np.sin(angle), 15), np.round(np.cos(angle), 15)
    offset = np.asarray(sample.xy) - origin
    return offset[:, 0] * east + offset[:, 1] * north, offset[:, 0] * north - offset[:, 1] * east


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
def radial(
    sample: Sample,
    center: tuple[float, float] | None = None,
    min_points: int = 3,
    stretch: float = 1.0,
    bearing: float = 0.0,
) -> Nest:
    """Nest a sample by removing the point farthest from a center at each step.

    The sample shrinks as concentric discs around the center, the point-data
    analogue of nested circular plots around a focal site. With ``stretch``
    the discs become ellipses, so what remains is longer in one direction
    than the other.

    Parameters
    ----------
    sample : Sample
        The sample to nest.
    center : tuple of float, optional
        The (longitude, latitude) to shrink towards; by default the mean of
        the projected points.
    min_points : int
        Stop when this many points remain.
    stretch : float
        How many times longer each ellipse is along ``bearing`` than across
        it: distances along the bearing are divided by it before the
        farthest point is chosen. 1 gives discs; above 1 the points far
        across the bearing go first and what remains is elongated along it;
        below 1 the reverse.
    bearing : float
        The direction of the ellipses' stretched axis, in degrees clockwise
        from grid north of the sample's projection. With ``stretch=2`` and
        the default 0, each step keeps an ellipse twice as long north-south
        as east-west.

    Returns
    -------
    Nest

    Raises
    ------
    ValueError
        If ``center`` isn't a longitude and latitude, ``stretch`` isn't a
        positive number, ``bearing`` isn't a finite number, or
        ``min_points`` is below 1 or above the sample's size.
    """
    check_size(sample, min_points)
    if _finite("stretch", stretch) <= 0:
        raise ValueError(f"stretch must be a positive number, got {stretch!r}.")
    along, across = _along_across(sample, bearing, _center_xy(sample, center))
    with np.errstate(over="ignore"):
        distance = np.hypot(along / stretch, across)
    if not np.isfinite(distance).all():
        raise ValueError(
            f"stretch={stretch!r} is too small to rank these points; for a band use "
            "strategies.band."
        )
    return _by_rank(sample, -distance, min_points)


@register("band")
def band(
    sample: Sample,
    bearing: float = 0.0,
    center: tuple[float, float] | None = None,
    min_points: int = 3,
) -> Nest:
    """Nest a sample by closing in on a line from both sides, keeping a band.

    At each step the point farthest from a line through ``center``, measured
    along ``bearing``, is removed, so each step keeps a band around the
    line, as long across the bearing as the points in it reach. With the
    default bearing of 0 the band is a belt of latitude closing in from the
    north and the south; with 90 it closes in from the east and the west,
    leaving a north-south strip that keeps whatever gradient runs along it.
    It is the limit of [`radial`][samplesar.strategies.radial] on
    the same bearing as its ``stretch`` goes to 0.

    Parameters
    ----------
    sample : Sample
        The sample to nest.
    bearing : float
        The direction the band narrows along, in degrees clockwise from grid
        north of the sample's projection.
    center : tuple of float, optional
        A (longitude, latitude) on the line the band closes in on; by
        default the mean of the projected points.
    min_points : int
        Stop when this many points remain.

    Returns
    -------
    Nest

    Raises
    ------
    ValueError
        If ``bearing`` isn't finite, ``center`` isn't a longitude and
        latitude, or ``min_points`` is below 1 or above the sample's size.
    """
    check_size(sample, min_points)
    along, _ = _along_across(sample, bearing, _center_xy(sample, center))
    return _by_rank(sample, -np.abs(along), min_points)


@register("sector")
def sector(
    sample: Sample,
    start: float = 0.0,
    sectors: int = 4,
    clockwise: bool = True,
    center: tuple[float, float] | None = None,
    min_points: int = 3,
) -> Nest:
    """Nest a sample by removing one compass sector after another.

    The plane around ``center`` is cut into equal sectors, the first
    starting at the bearing ``start`` and the rest following clockwise (or
    counterclockwise). The points of the first sector are removed, farthest
    from the center first, then those of the next, and so on. With the
    defaults the north-east quadrant goes first, then the south-east, the
    south-west and the north-west: a way to test whether a curve depends on
    which side of a region is lost first.

    A point on the line where a sector begins belongs to that sector, and a
    point at the center to the last one, so a focal site stays to the end.

    Parameters
    ----------
    sample : Sample
        The sample to nest.
    start : float
        The bearing where the first sector begins, in degrees clockwise from
        grid north of the sample's projection.
    sectors : int
        The number of equal sectors, at least 1. With 1 the sample shrinks
        as [`radial`][samplesar.strategies.radial] does.
    clockwise : bool
        Whether the sectors are taken clockwise from ``start``; otherwise
        counterclockwise.
    center : tuple of float, optional
        The (longitude, latitude) the sectors meet at; by default the mean
        of the projected points.
    min_points : int
        Stop when this many points remain.

    Returns
    -------
    Nest

    Raises
    ------
    ValueError
        If ``start`` isn't a finite number, ``sectors`` isn't a whole number
        of at least 1, ``clockwise`` isn't true or false, ``center`` isn't a
        longitude and latitude, or ``min_points`` is below 1 or above the
        sample's size.
    """
    check_size(sample, min_points)
    start = _finite("start", start)
    if isinstance(sectors, bool) or not isinstance(sectors, int | np.integer) or sectors < 1:
        raise ValueError(f"sectors must be a whole number of at least 1, got {sectors!r}.")
    sectors = int(sectors)
    if not isinstance(clockwise, bool | np.bool_):
        raise ValueError(f"clockwise must be True or False, got {clockwise!r}.")
    offset = np.asarray(sample.xy) - _center_xy(sample, center)
    distance = np.hypot(offset[:, 0], offset[:, 1])
    bearing = np.degrees(np.arctan2(offset[:, 0], offset[:, 1]))  # clockwise from north
    turned = np.mod((bearing - start) if clockwise else (start - bearing), 360.0)
    # The share of the way round, in sectors, rounded so a point on the line
    # where a sector begins is in it, not, by 1e-14 of a degree, in the one
    # before.
    which = np.floor(np.round(turned * sectors / 360.0, 9)).astype(int) % sectors
    which[distance <= 1e-9 * distance.max()] = sectors - 1  # at the center, to rounding
    order = np.lexsort((np.arange(sample.n_points), _tied(-distance), which))
    return from_order(sample, order, min_points=min_points)


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
    angle = np.radians(_finite("bearing", bearing))
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


def voronoi_density(xy: np.ndarray) -> np.ndarray:
    """Each point's density as one over the area of its Voronoi cell (the
    ground nearer to it than to any other point), cut to the points' convex
    hull, in points per km². Points at one position share its cell. Zero
    where the points make no hull (fewer than three, or in a line)."""
    import shapely

    positions, where, counts = np.unique(xy, axis=0, return_inverse=True, return_counts=True)
    where = np.asarray(where).ravel()
    cloud = shapely.MultiPoint(positions)
    hull = cloud.convex_hull
    if not isinstance(hull, shapely.Polygon) or not hull.area > 0:
        return np.zeros(len(xy))
    cells = shapely.get_parts(shapely.voronoi_polygons(cloud, extend_to=hull))
    # GEOS returns the cells in its own order: find each position's.
    tree = shapely.STRtree(cells)
    spots = shapely.points(positions)
    pairs = tree.query(spots, predicate="intersects")
    gap = shapely.distance(cells[pairs[1]], spots[pairs[0]])  # 0 inside; ties on shared edges
    best = np.lexsort((pairs[1], gap, pairs[0]))
    first = best[np.unique(pairs[0][best], return_index=True)[1]]
    area = np.zeros(len(positions))
    area[pairs[0][first]] = shapely.area(shapely.intersection(cells[pairs[1][first]], hull))
    density = np.where(area > 0, counts / np.where(area > 0, area, 1.0), np.inf)
    return np.asarray(density[where])


@register("density_peel")
def density_peel(
    sample: Sample,
    bandwidth_km: float | None = None,
    min_points: int = 3,
    method: Literal["kernel", "voronoi"] = "kernel",
) -> Nest:
    """Nest a sample by removing the point of lowest density at each step.

    The density of points is measured once, and points go in order of
    increasing density, so the boundary follows where the observations are,
    not the outermost few, and isolated records go first.

    With ``method="kernel"`` the density is smoothed with a Gaussian kernel,
    and each step keeps a superlevel set of it. With ``method="voronoi"`` a
    point's density is one over the area of its Voronoi cell, the ground
    nearer to it than to any other point (within the sample's convex hull):
    no bandwidth to choose, and the measure adapts to the local spacing, so
    the points standing for the most ground each, where effort was thinnest,
    go first.

    Parameters
    ----------
    sample : Sample
        The sample to nest.
    bandwidth_km : float, optional
        The kernel's standard deviation in km; by default Scott's rule.
        Only for ``method="kernel"``.
    min_points : int
        Stop when this many points remain.
    method : {"kernel", "voronoi"}
        How density is measured.

    Returns
    -------
    Nest

    Raises
    ------
    ValueError
        If ``method`` is unknown, ``bandwidth_km`` isn't positive or is
        given with ``method="voronoi"``, or ``min_points`` is below 1 or
        above the sample's size.
    """
    check_size(sample, min_points)
    if method not in ("kernel", "voronoi"):
        raise ValueError(f"method must be 'kernel' or 'voronoi', got {method!r}")
    if method == "voronoi":
        if bandwidth_km is not None:
            raise ValueError("bandwidth_km is only used with method='kernel'.")
        return _by_rank(sample, voronoi_density(np.asarray(sample.xy)), min_points)
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


@register("elevation")
def elevation(
    sample: Sample,
    remove: Literal["highest", "lowest", "outer"] = "highest",
    around: float | None = None,
    min_points: int = 3,
) -> Nest:
    """Nest a sample by removing points in order of elevation.

    ``remove="highest"`` takes the highest point at each step, from the
    summits down, so each step keeps everything below a falling contour;
    ``"lowest"`` works from the valleys up, keeping the high ground, as a
    climate envelope retreating upslope would. ``"outer"`` removes the point
    farthest in elevation from ``around``, so each step keeps a narrowing
    elevational band. Areas are still each step's convex hull on the map
    (planimetric), not surface area over the terrain.

    Parameters
    ----------
    sample : Sample
        The sample to nest, with an elevation at every point (see
        [`Sample.elevations`][samplesar.Sample.elevations]).
    remove : {"highest", "lowest", "outer"}
        Which point to remove at each step.
    around : float, optional
        For ``remove="outer"``, the elevation in meters the band closes in
        on; by default the median of the points' elevations.
    min_points : int
        Stop when this many points remain.

    Returns
    -------
    Nest

    Raises
    ------
    ValueError
        If the sample has no elevations or some points lack one, ``remove``
        is unknown, ``around`` isn't a finite number or is given without
        ``remove="outer"``, or ``min_points`` is below 1 or above the
        sample's size.
    """
    check_size(sample, min_points)
    if remove not in ("highest", "lowest", "outer"):
        raise ValueError(f"remove must be 'highest', 'lowest' or 'outer', got {remove!r}")
    if around is not None and remove != "outer":
        raise ValueError("around is only used with remove='outer'.")
    if around is not None:
        around = _finite("around", around)
    height = sample.elevations
    missing = int(np.count_nonzero(np.isnan(height)))
    if missing:
        raise ValueError(
            f"{missing} of the sample's {len(height)} points have no elevation; set them "
            "with Sample.with_elevations, or keep only the points that have one "
            "(Sample.subset)."
        )
    if remove == "highest":
        key = -height
    elif remove == "lowest":
        key = height
    else:
        key = -np.abs(height - (float(np.median(height)) if around is None else around))
    return _by_rank(sample, key, min_points)


@register("temporal")
def temporal(
    sample: Sample, remove: Literal["newest", "oldest"] = "newest", min_points: int = 3
) -> Nest:
    """Nest a sample by removing points in order of when they were first observed.

    ``remove="newest"`` takes the most recently found point at each step, so
    the nest runs back through time: each step holds the places that had
    been visited by some date, and its curve shows how the sampled area and
    its species list grew with effort. ``"oldest"`` removes the earliest
    places first. A point's date is that of its first observation; a point
    keeps all its observations while it remains, including later ones.

    Parameters
    ----------
    sample : Sample
        The sample to nest, with a date at every point (see
        [`Sample.dates`][samplesar.Sample.dates]).
    remove : {"newest", "oldest"}
        Which point to remove at each step.
    min_points : int
        Stop when this many points remain.

    Returns
    -------
    Nest

    Raises
    ------
    ValueError
        If the sample has no dates or some points lack one, ``remove`` is
        unknown, or ``min_points`` is below 1 or above the sample's size.
    """
    check_size(sample, min_points)
    if remove not in ("newest", "oldest"):
        raise ValueError(f"remove must be 'newest' or 'oldest', got {remove!r}")
    dates = sample.dates
    missing = int(np.count_nonzero(np.isnat(dates)))
    if missing:
        raise ValueError(
            f"{missing} of the sample's {len(dates)} points have no date; give every "
            "observation one, or keep only the points that have one (Sample.subset)."
        )
    # Whole nanoseconds, compared exactly: _tied's rounding would merge dates
    # a billionth of the sample's span apart.
    when = dates.astype("datetime64[ns]").astype(np.int64)
    order = np.lexsort((np.arange(sample.n_points), -when if remove == "newest" else when))
    return from_order(sample, order, min_points=min_points)
