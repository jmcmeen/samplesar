"""Regions to restrict samples to: polygons, from any source."""

from __future__ import annotations

from typing import Any

import numpy as np
import shapely
from shapely.geometry import MultiPolygon, Polygon


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
    antimeridian-crossing ``bbox`` is kept as it is)."""
    west, south, east, north = boundary.bounds
    if bbox is None:
        return (west, south, east, north)
    b_west, b_south, b_east, b_north = (float(v) for v in bbox)
    south, north = max(south, b_south), min(north, b_north)
    if b_west <= b_east:
        west, east = max(west, b_west), min(east, b_east)
    else:
        west, east = b_west, b_east
    return (west, south, east, north)
