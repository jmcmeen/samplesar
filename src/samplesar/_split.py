"""Splitting a sample into parts and nesting each: several patches at once."""

from __future__ import annotations

import inspect
import warnings
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import shapely

from samplesar import _registry
from samplesar._nest import Nest, NoStepError, check_size
from samplesar._registry import register
from samplesar._sample import WGS84, Sample


def _renumber(labels: np.ndarray) -> np.ndarray:
    """Labels as 0, 1, 2, ... in the order each part's first point appears,
    so the numbering doesn't depend on how the parts were found."""
    _, first, inverse = np.unique(labels, return_index=True, return_inverse=True)
    rank = np.empty(len(first), dtype=int)
    rank[np.argsort(first, kind="stable")] = np.arange(len(first))
    return np.asarray(rank[np.asarray(inverse).ravel()])


def _grid_shape(parts: Any) -> tuple[int, int]:
    """Columns and rows for a grid: as given, or the most nearly square
    layout with at least ``parts`` cells."""
    if isinstance(parts, tuple | list) and len(parts) == 2:
        columns, rows = (int(v) for v in parts)
        if columns < 1 or rows < 1 or (columns, rows) != tuple(parts):
            raise ValueError(f"A grid needs whole (columns, rows) of at least 1, got {parts!r}.")
        return columns, rows
    columns = int(np.ceil(np.sqrt(parts)))
    return columns, int(np.ceil(parts / columns))


def partition(
    sample: Sample,
    parts: int | tuple[int, int] = 4,
    by: Any = "kmeans",
    seed: int | None = None,
) -> np.ndarray:
    """Split a sample's points into parts.

    Parameters
    ----------
    sample : Sample
        The sample to split.
    parts : int or tuple of int
        How many parts to make, for ``by="kmeans"`` and ``by="grid"``; for a
        grid it may also be ``(columns, rows)``. Fewer come back when some
        would be empty.
    by : str, array or sequence of polygons
        How to split:

        - ``"kmeans"``: k-means clusters of the projected points, so each
          part is compact and the parts follow where the observations are.
        - ``"grid"``: equal rectangles over the points' bounding box on the
          sample's projection, the most nearly square layout with at least
          ``parts`` cells, so 3 gives 2 by 2 and 5 gives 3 by 2 (or
          ``parts=(columns, rows)``).
        - a one-dimensional array of labels, one per row of
          ``sample.points``: your own split.
        - an array of (longitude, latitude) seeds, shape (n, 2): each point
          joins its nearest seed (the seeds' Voronoi cells).
        - a sequence of shapely polygons in WGS84, or a GeoSeries or
          GeoDataFrame of them (ecoregions, islands, counties; one without
          a CRS is taken to be in WGS84): each point joins the first
          polygon that holds it.
    seed : int, optional
        Seed for k-means' starting centers. Without one the clusters, and
        so the split, can differ from run to run.

    Returns
    -------
    np.ndarray
        Each point's part: whole numbers from 0, numbered in the order
        their first points appear in ``sample.points``.

    Raises
    ------
    ValueError
        If ``parts`` isn't a whole number from 1 to the number of points,
        ``by`` isn't one of the above, labels or seeds have the wrong shape,
        or a point lies in none of the polygons.
    """
    n = sample.n_points
    xy = np.asarray(sample.xy)
    if isinstance(by, str):
        if by not in ("kmeans", "grid"):
            raise ValueError(f"by must be 'kmeans', 'grid', labels, seeds or polygons, got {by!r}")
        if by == "grid":
            columns, rows = _grid_shape(_whole(parts, n) if not _pair(parts) else parts)
            low, span = xy.min(axis=0), np.ptp(xy, axis=0)
            span[span == 0] = 1.0
            cell = np.floor((xy - low) / span * (columns, rows) * (1 - 1e-12)).astype(int)
            return _renumber(cell[:, 1] * columns + cell[:, 0])
        k = _whole(parts, n)
        if k == 1:
            return np.zeros(n, dtype=int)
        from scipy.cluster.vq import kmeans2

        with warnings.catch_warnings():  # an empty cluster just means fewer parts
            warnings.simplefilter("ignore", UserWarning)
            _, labels = kmeans2(xy, k, minit="++", seed=seed, missing="warn")
        return _renumber(labels)
    if hasattr(by, "to_crs") and hasattr(by, "crs"):  # a GeoSeries or GeoDataFrame
        frame = by if by.crs is None else by.to_crs(4326)
        by = list(getattr(frame, "geometry", frame))
    if isinstance(by, Sequence) and len(by) and all(isinstance(g, shapely.Geometry) for g in by):
        labels = np.full(n, -1)
        lon, lat = sample.points[:, 0], sample.points[:, 1]
        for i, polygon in enumerate(by):
            free = np.flatnonzero(labels < 0)
            if not len(free):
                break
            shapely.prepare(polygon)
            labels[free[shapely.intersects_xy(polygon, lon[free], lat[free])]] = i
        if (labels < 0).any():
            raise ValueError(
                f"{int((labels < 0).sum())} of the sample's {n} points lie in none of the "
                "polygons; restrict the sample first (Sample.within) or add a polygon for them."
            )
        return _renumber(labels)
    try:
        given = np.asarray(by)
    except (TypeError, ValueError):
        given = np.empty(0, dtype=object)
    if given.ndim == 1 and given.shape == (n,) and given.dtype.kind in "iub":
        return _renumber(given)
    if given.ndim == 1 and given.shape == (n,) and given.dtype.kind == "f":
        if not (np.isfinite(given).all() and (given == np.round(given)).all()):
            raise ValueError("Labels must be whole numbers or text, one per point, none missing.")
        return _renumber(given.astype(int))
    if given.ndim == 1 and given.shape == (n,) and given.dtype.kind in "OUS":
        return _renumber(np.unique(given.astype(str), return_inverse=True)[1])
    if given.ndim == 2 and given.shape[1] == 2 and given.dtype.kind in "if" and len(given):
        if not (np.abs(given[:, 0]) <= 180).all() or not (np.abs(given[:, 1]) <= 90).all():
            raise ValueError("Seeds must be (longitude, latitude) pairs in degrees.")
        from pyproj import Transformer
        from scipy.spatial import cKDTree

        projection = sample.projection
        to_plane = Transformer.from_crs(WGS84, projection, always_xy=True)
        x, y = to_plane.transform(given[:, 0], given[:, 1])
        seeds = np.column_stack([x, y]) * projection.axis_info[0].unit_conversion_factor / 1000.0

        return _renumber(cKDTree(seeds).query(xy)[1])
    raise ValueError(
        "by must be 'kmeans', 'grid', one label per point, (longitude, latitude) seeds of "
        f"shape (n, 2), or polygons; got {type(by).__name__}"
        + (f" of shape {given.shape}" if given.size else "")
        + "."
    )


def _pair(parts: Any) -> bool:
    return isinstance(parts, tuple | list) and len(parts) == 2


def _whole(parts: Any, n: int) -> int:
    if isinstance(parts, bool) or not isinstance(parts, int | np.integer) or not 1 <= parts <= n:
        raise ValueError(f"parts must be a whole number from 1 to the {n} points, got {parts!r}.")
    return int(parts)


@register("split")
def split(
    sample: Sample,
    strategy: str = "hull_random",
    parts: int | tuple[int, int] = 4,
    by: Any = "kmeans",
    pace: Literal["even", "turn"] = "even",
    min_points: int = 3,
    seed: int | None = None,
    **options: Any,
) -> Nest:
    """Nest a sample as several patches: split it into parts and run a strategy in each.

    One hull around a sample of separate clusters spans the empty ground
    between them. This splits the sample into parts (see
    [`strategies.partition`][samplesar.strategies.partition]), nests each
    part on its own with ``strategy``, and joins the parts' steps into one
    nest whose areas are those of the parts' hulls together: the curve of
    several separate patches shrinking at once, as an archipelago's is.

    Parameters
    ----------
    sample : Sample
        The sample to nest.
    strategy : str
        The registered strategy to run inside each part, such as
        ``"hull_peel"`` or ``"radial"``.
    parts : int or tuple of int
        How many parts, as [`strategies.partition`][samplesar.strategies.partition] takes it.
    by : str, array or sequence of polygons
        How to split, as ``partition`` takes it: ``"kmeans"``, ``"grid"``,
        labels, seeds or polygons.
    pace : {"even", "turn"}
        How the parts shrink together. ``"even"``: each at its own pace, so
        all reach their last step together (a part with twice the steps
        takes two for another's one). ``"turn"``: one step in each part per
        round, so parts with few steps finish early. Either way each step of
        the nest is one step of one part.
    min_points : int
        Each part stops when this many of its points remain, so the last
        step holds about that many for every part; a part with no more than
        that stays whole throughout, as does one a layer-peeling strategy
        can't take a layer from.
    seed : int, optional
        Seed for the split (k-means) and, when ``strategy`` is random, for
        each part's choices. Without one a k-means split can differ from
        run to run, even with a deterministic strategy inside.
    **options
        The inner strategy's own options, such as ``keep`` or ``bearing``.

    Returns
    -------
    Nest
        With [`parts`][samplesar.Nest]: each point's part.

    Raises
    ------
    ValueError
        If ``strategy`` isn't registered or is ``"split"`` or ``"order"``,
        ``pace`` is unknown, the split is invalid (see ``partition``),
        ``min_points`` is below 1, the strategy can't take a step in any
        part, or it rejects its options or any part (a part with points
        lacking a date, for ``"temporal"``, say).
    TypeError
        If an option isn't one the inner strategy takes.
    """
    if strategy == "split":
        raise ValueError("split can't run inside itself; split by labels instead.")
    if strategy == "order":
        raise ValueError(
            "split can't run 'order' inside its parts: an order names rows of the whole "
            "sample. Build the nest with strategies.from_order and set Nest.parts instead."
        )
    inner = _registry._REGISTRY.get(strategy)
    if inner is None:
        raise ValueError(f"No strategy {strategy!r}; known: {', '.join(_registry._REGISTRY)}.")
    if pace not in ("even", "turn"):
        raise ValueError(f"pace must be 'even' or 'turn', got {pace!r}")
    check_size(sample, min_points)
    labels = partition(sample, parts=parts, by=by, seed=seed)
    n_parts = int(labels.max()) + 1
    seeded = "seed" in inspect.signature(inner.original).parameters
    seeds = np.random.SeedSequence(seed).generate_state(n_parts) if seed is not None else None
    # Each part's steps, as rows of the whole sample's points.
    nests: list[list[np.ndarray]] = []
    failure: NoStepError | None = None
    for part in range(n_parts):
        members = np.flatnonzero(labels == part)
        if len(members) <= min_points:
            nests.append([members])
            continue
        own = dict(options)
        if seeded:
            own["seed"] = None if seeds is None else int(seeds[part])
        part_sample = sample.subset(members)
        if not np.array_equal(part_sample.points, sample.points[members]):
            raise RuntimeError("A part's points came back out of order; please report this.")
        try:
            nest = inner.original(part_sample, min_points=min_points, **own)
        except NoStepError as err:  # this part has no layer to peel: it stays whole
            failure = err
            nests.append([members])
            continue
        nests.append([members[np.asarray(step, dtype=int)] for step in nest.steps])
    if all(len(steps) == 1 for steps in nests):
        if failure is not None:
            raise failure
        raise ValueError(
            f"No part has more than min_points={min_points} points, so there is nothing to nest."
        )
    # Every step of every part, in the order the nest takes them.
    events = [
        (step / (len(steps) - 1) if pace == "even" else float(step), part, step)
        for part, steps in enumerate(nests)
        for step in range(1, len(steps))
    ]
    events.sort()
    current = [steps[0] for steps in nests]
    joined = [np.sort(np.concatenate(current))]
    for _, part, step in events:
        current[part] = nests[part][step]
        joined.append(np.sort(np.concatenate(current)))
    return Nest(sample=sample, steps=tuple(joined), parts=labels)
