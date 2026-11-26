"""Nesting strategies that remove points from the convex hull inwards."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Literal

import numpy as np
import shapely
from shapely.geometry import MultiPoint, Polygon

from samplesar._nest import Nest, NoStepError, check_size
from samplesar._registry import register
from samplesar._sample import Sample, point_species
from samplesar._sample import hull_area as convex_area


def _hull(points: np.ndarray, idx: np.ndarray) -> shapely.Geometry:
    return MultiPoint(points[idx]).convex_hull


def _tolerance(hull: Polygon) -> float:
    """How near counts as on the hull's boundary: 1e-9 of its bounding box's
    diagonal. A row of points laid out straight in the sample's crs (a UTM
    grid, say) is straight here only to within rounding after its round trip
    through longitude and latitude."""
    x0, y0, x1, y1 = hull.bounds
    return 1e-9 * float(np.hypot(x1 - x0, y1 - y0))


def _off_line(corners: np.ndarray, before: np.ndarray, after: np.ndarray) -> np.ndarray:
    """Each corner's distance from the line through its two neighbors."""
    edge = after - before
    cross = np.abs(edge[:, 0] * (corners - before)[:, 1] - edge[:, 1] * (corners - before)[:, 0])
    return cross / np.maximum(np.hypot(edge[:, 0], edge[:, 1]), 1e-300)


def _true_corners(ring: np.ndarray, tolerance: float) -> np.ndarray:
    """The ring's corners without those within ``tolerance`` of the line
    through their neighbors: edge points that rounding pushed outward. If
    what's left is a sliver (points collinear but for rounding), its two far
    ends."""
    corners = ring
    while len(corners) > 3:
        off = _off_line(corners, np.roll(corners, 1, axis=0), np.roll(corners, -1, axis=0))
        # One corner per pass, the flattest, so each later decision is made
        # against the corner's current neighbors, not ones already dropped.
        flattest = int(np.argmin(off))
        if off[flattest] > tolerance:
            break
        corners = np.delete(corners, flattest, axis=0)
    if len(corners) == 3:
        off = _off_line(corners, np.roll(corners, 1, axis=0), np.roll(corners, -1, axis=0))
        if off.min() <= tolerance:
            gaps = [np.hypot(*(corners[a] - corners[b])) for a, b in ((0, 1), (1, 2), (0, 2))]
            a, b = ((0, 1), (1, 2), (0, 2))[int(np.argmax(gaps))]
            corners = corners[[a, b]]
    return corners


def hull_vertices(points: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """The rows of ``idx`` whose points are vertices of their convex hull.

    Points on an edge between two vertices aren't vertices, to within
    rounding (1e-9 of the hull's bounding-box diagonal). When the points are collinear the hull is
    a segment and its vertices are its two ends. Returned in the order of
    ``idx``.
    """
    hull = _hull(points, idx)
    if isinstance(hull, Polygon):
        coords = _true_corners(np.asarray(hull.exterior.coords)[:-1], _tolerance(hull))
    else:
        coords = np.asarray(hull.coords)
    corners = {tuple(c) for c in coords}
    return np.array([i for i in idx if tuple(points[i]) in corners], dtype=int)


def _on_boundary(points: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """The rows of ``idx`` on the boundary of their convex hull, to within
    rounding: the vertices and any points on its edges. A degenerate hull (a
    point or a segment) has every point on it."""
    hull = _hull(points, idx)
    if not isinstance(hull, Polygon):
        return idx.copy()
    distance = shapely.distance(hull.boundary, shapely.points(points[idx]))
    return idx[distance <= _tolerance(hull)]


def _remove_one(
    sample: Sample, min_points: int, choose: Callable[[np.ndarray, np.ndarray], int]
) -> Nest:
    """Drop one hull vertex per step: ``choose(remaining, vertices)`` names it."""
    check_size(sample, min_points)
    points = sample.xy
    remaining = np.arange(sample.n_points)
    steps = [remaining.copy()]
    while len(remaining) > min_points:
        drop = choose(remaining, hull_vertices(points, remaining))
        remaining = remaining[remaining != drop]
        steps.append(remaining.copy())
    return Nest(sample=sample, steps=tuple(steps))


@register("hull_random")
def hull_random(sample: Sample, min_points: int = 3, seed: int | None = None) -> Nest:
    """Nest a sample by removing a random vertex of its convex hull at each step.

    Like `strategies.random`, but only points on the outside can go, so the sample
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


class _Isolation:
    """Each hull vertex's distance to its nearest other remaining point (km),
    from one KD-tree of all the points: the nearest few neighbors are looked
    up and the first still remaining is taken, falling back to a tree of the
    remaining points only when all of those are gone."""

    LOOK = 16

    def __init__(self, points: np.ndarray) -> None:
        from scipy.spatial import cKDTree

        self.points = points
        self.tree = cKDTree(points)
        self.alive = np.ones(len(points), dtype=bool)

    def __call__(self, remaining: np.ndarray, vertices: np.ndarray) -> np.ndarray:
        from scipy.spatial import cKDTree

        self.alive[:] = False
        self.alive[remaining] = True
        k = min(self.LOOK + 1, len(self.points))
        distance, index = self.tree.query(self.points[vertices], k=k)
        distance, index = distance.reshape(len(vertices), k), index.reshape(len(vertices), k)
        usable = self.alive[index] & (index != vertices[:, None])
        found = usable.any(axis=1)
        out = np.empty(len(vertices))
        out[found] = distance[found, usable[found].argmax(axis=1)]
        if not found.all():
            near, _ = cKDTree(self.points[remaining]).query(self.points[vertices[~found]], k=2)
            out[~found] = near[:, 1]
        return out


def isolation_weights(isolation: np.ndarray, power: float) -> np.ndarray:
    """Probabilities proportional to ``isolation ** power``, computed on
    isolation scaled to its maximum so a large power neither underflows nor
    overflows; equal when every vertex shares its position with another."""
    top = float(isolation.max())
    if top <= 0:
        return np.full(len(isolation), 1.0 / len(isolation))
    weight = (isolation / top) ** power
    return weight / weight.sum()


@register("hull_weighted")
def hull_weighted(
    sample: Sample, power: float = 1.0, min_points: int = 3, seed: int | None = None
) -> Nest:
    """Nest a sample by removing a random hull vertex, weighted by its isolation.

    Each vertex of the current hull is picked with probability proportional
    to its distance from the nearest other remaining point, raised to
    ``power``. Isolated outliers (vagrants, georeferencing errors, one-off
    trips) then tend to go first, while clustered edges last longer. A power
    of 0 is `strategies.hull_random` (the same nest for the same seed); a
    large power approaches always removing the most isolated vertex.

    Parameters
    ----------
    sample : Sample
        The sample to nest.
    power : float
        The exponent on each vertex's isolation; at least 0.
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
        If ``power`` is negative or not finite, or ``min_points`` is below 1
        or the sample has fewer points.
    """
    if not (np.isfinite(power) and power >= 0):
        raise ValueError(f"power must be a finite number of at least 0, got {power}.")
    rng = np.random.default_rng(seed)
    isolation = _Isolation(np.asarray(sample.xy))

    def choose(remaining: np.ndarray, vertices: np.ndarray) -> int:
        if power == 0 or len(remaining) < 2:
            return int(rng.choice(vertices))
        weight = isolation_weights(isolation(remaining, vertices), power)
        return int(rng.choice(vertices, p=weight))

    return _remove_one(sample, min_points, choose)


@register("hull_peel")
def hull_peel(sample: Sample, min_points: int = 3) -> Nest:
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
    points = sample.xy
    remaining = np.arange(sample.n_points)
    steps = [remaining.copy()]
    while True:
        inside = remaining[~np.isin(remaining, _on_boundary(points, remaining))]
        if len(inside) < min_points:
            break
        remaining = inside
        steps.append(remaining.copy())
    if len(steps) == 1:
        raise NoStepError(
            f"No layer can be peeled: fewer than {min_points} points lie strictly inside the "
            "sample's convex hull."
        )
    return Nest(sample=sample, steps=tuple(steps))


def grid_cells(xy: np.ndarray, cell_km: float) -> np.ndarray:
    """Each point's grid cell, as an index into the occupied cells: square
    cells of ``cell_km`` anchored at the points' south-west corner. A point
    on a cell edge, to within rounding, is in the cell above or to its right."""
    corner = xy.min(axis=0)
    ij = np.floor((xy - corner) / cell_km + 1e-9).astype(np.int64)
    _, cell = np.unique(ij, axis=0, return_inverse=True)
    return np.asarray(cell).ravel()


@register("grid_peel")
def grid_peel(sample: Sample, cell_km: float, min_points: int = 3) -> Nest:
    """Nest a sample by peeling the convex layers of its occupied grid cells.

    Points are snapped to square cells of ``cell_km``, anchored at the
    sample's south-west corner in its projected km. Each cell is represented
    by the mean of its points, and each step peels the cells on the convex
    hull of the remaining cells, removing every point in them together. Small cells behave like
    `strategies.hull_peel`; large cells give a few coarse steps that ignore
    detail within a cell, so a range of sizes shows how much the curve
    depends on the scale of removal. It stops before a layer would leave
    fewer than ``min_points`` points.

    Parameters
    ----------
    sample : Sample
        The sample to nest.
    cell_km : float
        The side of a grid cell, in km.
    min_points : int
        Stop rather than leave fewer than this many points.

    Returns
    -------
    Nest

    Raises
    ------
    ValueError
        If ``cell_km`` isn't positive, ``min_points`` is below 1 or the
        sample has fewer points, or no layer of cells can be peeled.
    """
    check_size(sample, min_points)
    if not (np.isfinite(cell_km) and cell_km > 0):
        raise ValueError(f"cell_km must be positive, got {cell_km}.")
    xy = np.asarray(sample.xy)
    cell = grid_cells(xy, cell_km)
    n_cells = int(cell.max()) + 1
    centers = np.zeros((n_cells, 2))
    np.add.at(centers, cell, xy)  # each cell's center: the mean of its points
    centers /= np.bincount(cell, minlength=n_cells)[:, None]
    cells_left = np.arange(n_cells)
    remaining = np.arange(sample.n_points)
    steps = [remaining.copy()]
    while True:
        cells_left = cells_left[~np.isin(cells_left, _on_boundary(centers, cells_left))]
        inside = remaining[np.isin(cell[remaining], cells_left)]
        if len(inside) < min_points:
            break
        remaining = inside
        steps.append(remaining.copy())
    if len(steps) == 1:
        raise NoStepError(
            f"No layer of {cell_km} km cells can be peeled: fewer than {min_points} points "
            "lie in cells strictly inside the hull of the occupied cells."
        )
    return Nest(sample=sample, steps=tuple(steps))


def _center_distance(points: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    """Each vertex's distance from the centroid of the vertices (km)."""
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
    one farthest from the center, then to the lowest point index."""
    tied = np.flatnonzero(_near_max(score))
    farthest = tied[_near_max(distance[tied])]
    return int(vertices[farthest].min())


@register("hull_area")
def hull_area(
    sample: Sample, keep: Literal["largest", "smallest"] = "largest", min_points: int = 3
) -> Nest:
    """Nest a sample by removing the hull vertex that changes its area least or most.

    Not to be confused with [`samplesar.hull_area`][samplesar.hull_area],
    which measures a set of points.

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
    points = sample.xy

    def choose(remaining: np.ndarray, vertices: np.ndarray) -> int:
        areas = np.array([convex_area(points[remaining[remaining != v]]) for v in vertices])
        return _pick(vertices, sign * areas, _center_distance(points, vertices))

    return _remove_one(sample, min_points, choose)


@register("hull_masked_area")
def hull_masked_area(
    sample: Sample, keep: Literal["largest", "smallest"] = "largest", min_points: int = 3
) -> Nest:
    """Nest a sample by removing the hull vertex that changes its area within the mask least or most.

    [`hull_area`][samplesar.strategies.hull_area] measured within the sample's
    mask (see [`Sample.with_mask`][samplesar.Sample.with_mask]): every vertex
    of the current hull is tried, and the one whose removal leaves the
    largest hull area inside the mask (``keep="largest"``) or the smallest
    (``keep="smallest"``) goes. On a coast, a vertex whose loss gives up a
    wide stretch of sea and little land costs little here, where
    ``hull_area`` counts the sea as well; the two choose alike once every
    hull lies inside the mask. Ties go as in ``hull_area``: to within
    floating-point rounding, to the vertex farthest from the centroid of the
    hull's vertices, then to the lowest point index.

    Parameters
    ----------
    sample : Sample
        The sample to nest, with a mask.
    keep : {"largest", "smallest"}
        Which remaining hull to keep at each step, by its area within the mask.
    min_points : int
        Stop when this many points remain.

    Returns
    -------
    Nest

    Raises
    ------
    ValueError
        If the sample has no mask (without one this is ``hull_area``),
        ``keep`` is unknown, ``min_points`` is below 1 or the sample has fewer
        points.
    """
    if sample.mask is None:
        raise ValueError(
            "hull_masked_area needs a sample with a mask (Sample.with_mask, or "
            "within(..., mask=True)); without one, use hull_area."
        )
    if keep not in ("largest", "smallest"):
        raise ValueError(f"keep must be 'largest' or 'smallest', got {keep!r}")
    sign = 1.0 if keep == "largest" else -1.0
    points = sample.xy

    def choose(remaining: np.ndarray, vertices: np.ndarray) -> int:
        areas = np.array([sample.hull_area_of(remaining[remaining != v]) for v in vertices])
        return _pick(vertices, sign * areas, _center_distance(points, vertices))

    return _remove_one(sample, min_points, choose)


@register("hull_species")
def hull_species(
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
    points = sample.xy
    species, n_species = point_species(sample)
    # How many remaining points hold each species: removing a point loses the
    # species it is the last holder of.
    holders = np.zeros(n_species, dtype=int)
    for found in species:
        holders[found] += 1

    def choose(remaining: np.ndarray, vertices: np.ndarray) -> int:
        lost = np.array([np.count_nonzero(holders[species[v]] == 1) for v in vertices])
        drop = _pick(vertices, -sign * lost, _center_distance(points, vertices))
        holders[species[drop]] -= 1
        return drop

    return _remove_one(sample, min_points, choose)


@register("hull_distance")
def hull_distance(
    sample: Sample, remove: Literal["farthest", "closest"] = "farthest", min_points: int = 3
) -> Nest:
    """Nest a sample by removing the hull vertex farthest from, or closest to, its center.

    The center is the centroid of the current hull's vertices, and distances
    are in km on the sample's projection (see
    [`Sample.xy`][samplesar.Sample.xy]). ``remove="farthest"`` trims the
    outlying corners first; ``remove="closest"`` removes the vertex nearest
    the center, leaving the outliers to last. Ties, to within floating-point
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
    points = sample.xy

    def choose(remaining: np.ndarray, vertices: np.ndarray) -> int:
        distance = _center_distance(points, vertices)
        return _pick(vertices, sign * distance, np.zeros(len(vertices)))

    return _remove_one(sample, min_points, choose)


@register("concave_peel")
def concave_peel(sample: Sample, ratio: float = 0.3, min_points: int = 3) -> Nest:
    """Nest a sample by peeling off its concave hull at each step.

    Like [`strategies.hull_peel`][samplesar.strategies.hull_peel], but each step removes
    the points on the boundary of a concave hull, which follows the sample's
    outline instead of spanning its gaps: suited to L-shaped or scattered
    samples (a valley, a coastline, a trail network). The concave hull is
    shapely's, with ``ratio`` setting how tightly it follows the points. The
    nest's areas are still convex hull areas, as for every strategy.

    Parameters
    ----------
    sample : Sample
        The sample to nest.
    ratio : float
        Between 0 and 1: 1 is the convex hull (the same as ``strategies.hull_peel``),
        smaller values follow the outline more tightly.
    min_points : int
        Stop rather than leave fewer than this many points.

    Returns
    -------
    Nest

    Raises
    ------
    ValueError
        If ``ratio`` is outside 0 to 1, ``min_points`` is below 1, the sample
        has fewer points, or no layer can be peeled.
    RuntimeError
        If shapely is built on a GEOS older than 3.11 (the PyPI wheels of
        shapely 2.0 and later aren't).
    """
    if not 0 <= ratio <= 1:
        raise ValueError(f"ratio must be between 0 and 1, got {ratio!r}.")
    if shapely.geos_version < (3, 11):
        raise RuntimeError(
            f"strategies.concave_peel needs GEOS 3.11 or newer for shapely's concave_hull; this "
            f"shapely uses GEOS {'.'.join(map(str, shapely.geos_version))}."
        )
    check_size(sample, min_points)
    points = sample.xy
    remaining = np.arange(sample.n_points)
    steps = [remaining.copy()]
    while True:
        hull = shapely.concave_hull(MultiPoint(points[remaining]), ratio=ratio)
        if isinstance(hull, Polygon):
            distance = shapely.distance(hull.exterior, shapely.points(points[remaining]))
            on = remaining[distance <= _tolerance(hull)]
        else:
            on = remaining
        inside = remaining[~np.isin(remaining, on)]
        if len(inside) < min_points:
            break
        remaining = inside
        steps.append(remaining.copy())
    if len(steps) == 1:
        raise NoStepError(
            f"No layer can be peeled: fewer than {min_points} points lie inside the "
            "sample's concave hull."
        )
    return Nest(sample=sample, steps=tuple(steps))


def _circle2(ax: float, ay: float, bx: float, by: float) -> tuple[float, float, float]:
    """The circle on two points as a diameter: (x, y, radius)."""
    cx, cy = (ax + bx) / 2.0, (ay + by) / 2.0
    return cx, cy, math.hypot(ax - cx, ay - cy)


def _circle3(
    ax: float, ay: float, bx: float, by: float, cx: float, cy: float
) -> tuple[float, float, float]:
    """The circle through three points: (x, y, radius). Three points in a
    line (to within rounding) give the circle on the two farthest apart."""
    d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    span = max(abs(ax - bx), abs(ax - cx), abs(bx - cx), abs(ay - by), abs(ay - cy), abs(by - cy))
    if abs(d) <= 1e-12 * span * span or span == 0:
        pairs = ((ax, ay, bx, by), (bx, by, cx, cy), (ax, ay, cx, cy))
        return _circle2(*max(pairs, key=lambda q: math.hypot(q[0] - q[2], q[1] - q[3])))
    a2, b2, c2 = ax * ax + ay * ay, bx * bx + by * by, cx * cx + cy * cy
    x = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / d
    y = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / d
    return x, y, math.hypot(ax - x, ay - y)


def _circle_of(points: np.ndarray) -> tuple[np.ndarray, float]:
    """The circle through one, two or three points: (center, radius)."""
    flat = [float(v) for v in np.asarray(points, dtype=float).ravel()]
    if len(flat) == 2:
        return np.array(flat), 0.0
    x, y, radius = _circle2(*flat) if len(flat) == 4 else _circle3(*flat)
    return np.array([x, y]), radius


def enclosing_circle(points: np.ndarray) -> tuple[np.ndarray, float]:
    """The smallest circle around some points, as (center, radius): Welzl's
    incremental algorithm, with the points taken in a fixed shuffled order
    so it runs in about linear time and always gives the same answer. Plain
    floats throughout: the point sets are hull vertices, a few dozen at
    most, where array calls would cost more than the arithmetic."""
    points = np.asarray(points, dtype=float)
    if len(points) == 0:
        raise ValueError("No points to enclose.")
    # Measured from the points' own middle: far from the origin (projected
    # coordinates run to thousands of km) the circle through three points
    # loses most of its digits to cancellation.
    origin = (points.min(axis=0) + points.max(axis=0)) / 2.0
    order = np.random.default_rng(0).permutation(len(points))
    p = [(float(x), float(y)) for x, y in points[order] - origin]
    x, y, r = p[0][0], p[0][1], 0.0
    for i in range(1, len(p)):
        ix, iy = p[i]
        if math.hypot(ix - x, iy - y) <= r * (1.0 + 1e-12):
            continue
        x, y, r = ix, iy, 0.0  # p[i] lies on the circle of p[: i + 1]
        for j in range(i):
            jx, jy = p[j]
            if math.hypot(jx - x, jy - y) <= r * (1.0 + 1e-12):
                continue
            x, y, r = _circle2(ix, iy, jx, jy)  # and so does p[j]
            for k in range(j):
                kx, ky = p[k]
                if math.hypot(kx - x, ky - y) > r * (1.0 + 1e-12):
                    x, y, r = _circle3(ix, iy, jx, jy, kx, ky)
    return np.array([x, y]) + origin, r


def _outline(points: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Rows of ``idx`` that include every vertex of their convex hull: from
    qhull, which is fast, or all of them when it can't build a hull (few
    points, or points in a line)."""
    if len(idx) < 4:
        return idx
    from scipy.spatial import ConvexHull, QhullError

    try:
        return idx[ConvexHull(points[idx]).vertices]
    except QhullError:
        return idx


@register("circle")
def circle(sample: Sample, min_points: int = 3) -> Nest:
    """Nest a sample by shrinking its smallest enclosing circle as fast as possible.

    At each step the smallest circle around the remaining points is found.
    It rests on two or three of them; the one whose removal leaves the
    smallest circle goes. The sample is trimmed where it reaches farthest,
    like [`radial`][samplesar.strategies.radial], but toward no fixed
    center: the circle moves as it shrinks, so a sample with a long arm
    loses the arm first and then closes on its bulk. Ties, to within
    floating-point rounding, go to the lowest point index. Areas are still
    convex hull areas.

    Parameters
    ----------
    sample : Sample
        The sample to nest.
    min_points : int
        Stop when this many points remain.

    Returns
    -------
    Nest

    Raises
    ------
    ValueError
        If ``min_points`` is below 1 or the sample has fewer points.
    """
    points = np.asarray(sample.xy)

    def choose(remaining: np.ndarray, vertices: np.ndarray) -> int:
        # The circle rests on hull vertices only, so they stand for the rest.
        center, radius = enclosing_circle(points[vertices])
        distance = np.hypot(*(points[vertices] - center).T)
        resting = vertices[distance >= radius * (1 - 1e-9)]
        if len(resting) == 1 or len(remaining) <= 2:
            return int(resting.min())
        left = []
        for v in resting:
            rest = remaining[remaining != v]
            left.append(enclosing_circle(points[_outline(points, rest)])[1])
        radii = np.asarray(left)
        tied = resting[radii <= radii.min() * (1 + 1e-9) + 1e-300]
        return int(tied.min())

    return _remove_one(sample, min_points, choose)
