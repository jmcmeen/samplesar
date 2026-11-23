"""Regions to restrict samples to: polygons, from any source."""

from __future__ import annotations

from typing import Any

import numpy as np
import shapely
from shapely.geometry import MultiPolygon, Polygon, box


def as_geometry(boundary: Any) -> shapely.Geometry:
    """A boundary as one polygonal shapely geometry in WGS84 longitude and
    latitude: a shapely geometry is taken to be in WGS84 already; a
    GeoDataFrame or GeoSeries is converted from its CRS and dissolved."""
    if hasattr(boundary, "to_crs") and hasattr(boundary, "crs"):
        frame = boundary if boundary.crs is None else boundary.to_crs(4326)
        geometry = getattr(frame, "geometry", frame)
        boundary = shapely.union_all(np.asarray(geometry))
    if not isinstance(boundary, Polygon | MultiPolygon):
        raise ValueError(
            "boundary must be a polygon or multipolygon (a shapely geometry in WGS84, or a "
            f"GeoDataFrame/GeoSeries of them), got {type(boundary).__name__}."
        )
    if boundary.is_empty:
        raise ValueError("boundary is empty.")
    return boundary


def inside(boundary: shapely.Geometry, lon: Any, lat: Any) -> np.ndarray:
    """Which (lon, lat) points lie in the boundary or on its edge."""
    shapely.prepare(boundary)
    return np.asarray(
        shapely.intersects_xy(boundary, np.asarray(lon, float), np.asarray(lat, float))
    )


def clip_bbox(
    boundary: shapely.Geometry, bbox: tuple[float, float, float, float] | None
) -> tuple[float, float, float, float]:
    """The boundary's bounding box, narrowed by ``bbox`` if one is given (an
    antimeridian-crossing ``bbox`` is kept as it is). Raises if ``bbox``
    isn't (west, south, east, north) or doesn't overlap the boundary."""
    west, south, east, north = boundary.bounds
    if bbox is None:
        return (west, south, east, north)
    try:
        b_west, b_south, b_east, b_north = (float(v) for v in bbox)
    except (TypeError, ValueError):
        raise ValueError(f"bbox must be (west, south, east, north), got {bbox!r}.") from None
    if not (b_south <= b_north and -90 <= b_south and b_north <= 90):
        raise ValueError(f"bbox must be (west, south, east, north), got {bbox!r}.")
    south, north = max(south, b_south), min(north, b_north)
    if b_west <= b_east:
        west, east = max(west, b_west), min(east, b_east)
        overlaps = west <= east
    else:  # the caller's bbox crosses the antimeridian
        overlaps = boundary.bounds[2] >= b_west or boundary.bounds[0] <= b_east
        west, east = b_west, b_east
    if south > north or not overlaps:
        raise ValueError("bbox and boundary don't overlap.")
    return (west, south, east, north)


def polygonal(geometry: shapely.Geometry) -> shapely.Geometry:
    """Only the polygons of a geometry: ``make_valid`` and intersections can
    leave stray lines and points in a collection, which some GEOS versions
    refuse in later operations."""
    polygons: list[Polygon] = []
    stack = [geometry]
    while stack:
        part = stack.pop()
        if isinstance(part, Polygon):
            if not part.is_empty:
                polygons.append(part)
        elif hasattr(part, "geoms"):
            stack.extend(part.geoms)
    if not polygons:
        return Polygon()
    return polygons[0] if len(polygons) == 1 else MultiPolygon(polygons)


def unwrap(lon: Any, lon0: float) -> np.ndarray:
    """Longitudes within 180° of ``lon0``, so ones across the antimeridian
    from it run on past ±180 instead of jumping."""
    return (np.asarray(lon, dtype=float) - lon0 + 180.0) % 360.0 - 180.0 + lon0


def near(region: shapely.Geometry, points: np.ndarray, lon0: float) -> shapely.Geometry:
    """The part of a region near some (lon, lat) points, in longitudes
    unwrapped about ``lon0``: within a box around the points, padded by half
    their extent and a degree. Raises if none of the region is there."""
    lon = unwrap(points[:, 0], lon0)
    west, east = float(lon.min()), float(lon.max())
    south, north = float(points[:, 1].min()), float(points[:, 1].max())
    pad = 0.5 * max(east - west, north - south) + 1.0
    if east - west + 2 * pad < 360:
        frame = box(west - pad, max(south - pad, -90.0), east + pad, min(north + pad, 90.0))
        # The region in each of its three positions around the globe, cut to the box.
        copies = [
            shapely.transform(region, lambda c, d=shift: c + np.array([d, 0.0])).intersection(frame)
            for shift in (-360.0, 0.0, 360.0)
        ]
        region = shapely.union_all([polygonal(c) for c in copies])
    region = polygonal(region)
    if region.is_empty:
        raise ValueError("The mask doesn't reach the sample's region.")
    return region
