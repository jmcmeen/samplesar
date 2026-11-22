"""Point observations of species, the input every nesting strategy starts from."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pyproj import CRS, Transformer
from shapely.geometry import MultiPoint

COLUMNS = ("species", "lon", "lat")


WGS84 = CRS.from_epsg(4326)


def _validate(df: pd.DataFrame, crs: CRS | None = None) -> pd.DataFrame:
    """Check and normalise the columns; with a CRS, convert its coordinates
    (in the ``lon`` and ``lat`` columns) to WGS84 longitude and latitude."""
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"Missing column(s): {missing}. Need {list(COLUMNS)}.")
    out = df.loc[:, list(COLUMNS)].reset_index(drop=True)
    if len(out) == 0:
        raise ValueError("Sample has no observations")
    if out["species"].isna().any():
        raise ValueError("species contains missing values")
    lon = pd.to_numeric(out["lon"], errors="coerce")
    lat = pd.to_numeric(out["lat"], errors="coerce")
    if lon.isna().any() or lat.isna().any():
        raise ValueError("Coordinates must be numeric with no missing values")
    if crs is not None and crs != WGS84:
        to_wgs84 = Transformer.from_crs(crs, WGS84, always_xy=True)
        x, y = to_wgs84.transform(lon.to_numpy(float), lat.to_numpy(float))
        if not (np.isfinite(x).all() and np.isfinite(y).all()):
            raise ValueError(f"Some coordinates fall outside {crs.name}.")
        lon, lat = pd.Series(x), pd.Series(y)
    if ((lat < -90) | (lat > 90)).any():
        raise ValueError("lat must be within [-90, 90]")
    if ((lon < -180) | (lon > 180)).any():
        raise ValueError("lon must be within [-180, 180]")
    out["lon"] = lon.astype(float)
    out["lat"] = lat.astype(float)
    out["species"] = out["species"].astype(str)
    return out


@dataclass(frozen=True)
class Sample:
    """A set of georeferenced species observations.

    Build one with [`from_df`][samplesar.Sample.from_df] or
    [`from_csv`][samplesar.Sample.from_csv] rather than directly: they check
    and normalise the columns.

    Attributes
    ----------
    data : pd.DataFrame
        One row per observation, with columns ``species`` (str), ``lon`` and
        ``lat`` (float, decimal degrees). Several observations may share a
        point. Treat it as read-only: the points and other derived values
        are computed once, so build a new sample rather than editing it.
    centre : tuple of float, optional
        The (lon, lat) the points are projected about for areas and
        distances (see [`xy`][samplesar.Sample.xy]), when there's no
        ``crs``. By default the centre of the sample's points; a
        [`subset`][samplesar.Sample.subset] keeps its parent's, so their
        areas are comparable.
    crs : pyproj.CRS, optional
        A projected coordinate reference system to measure areas and
        distances in, such as the UTM zone the sample was laid out in. Set by
        [`from_df`][samplesar.Sample.from_df] when the coordinates come in
        one.
    """

    data: pd.DataFrame
    centre: tuple[float, float] | None = None
    crs: CRS | None = None

    @classmethod
    def from_df(
        cls,
        df: pd.DataFrame,
        species_col: str = "species",
        lon_col: str = "lon",
        lat_col: str = "lat",
        crs: Any = None,
    ) -> Sample:
        """Build a sample from a DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            One row per observation. Other columns are dropped.
        species_col : str
            The column holding the species name.
        lon_col : str
            The column holding the longitude in decimal degrees, or the x
            (easting) coordinate when ``crs`` is projected.
        lat_col : str
            The column holding the latitude in decimal degrees, or the y
            (northing) coordinate when ``crs`` is projected.
        crs : str, int or pyproj.CRS, optional
            The coordinates' reference system, as pyproj accepts it
            (``"EPSG:32617"``, ``32617``, a WKT string). By default WGS84
            longitude and latitude. Coordinates in another geographic system
            are converted to WGS84. Coordinates in a projected system are
            converted to WGS84 for ``data``, and areas and distances are
            measured in that system, so a grid laid out in it stays straight.

        Returns
        -------
        Sample

        Raises
        ------
        KeyError
            If a column is missing.
        ValueError
            If there are no rows, a species is missing, a coordinate is
            missing, not numeric or out of range, or ``crs`` isn't a
            coordinate reference system.
        """
        renamed = df.rename(columns={species_col: "species", lon_col: "lon", lat_col: "lat"})
        parsed = _parse_crs(crs)
        projected = parsed if parsed is not None and parsed.is_projected else None
        return cls(_validate(renamed, parsed), crs=projected)

    @classmethod
    def from_geodataframe(cls, gdf: Any, species_col: str = "species") -> Sample:
        """Build a sample from a GeoDataFrame of point observations.

        The geometries' coordinates and the frame's CRS are read directly, so
        geopandas isn't a requirement of samplesar; the frame just needs a
        point ``geometry`` and a ``crs``.

        Parameters
        ----------
        gdf : geopandas.GeoDataFrame
            One point per observation. A frame without a CRS is taken to be
            in WGS84 longitude and latitude.
        species_col : str
            The column holding the species name.

        Returns
        -------
        Sample

        Raises
        ------
        ValueError
            If a geometry isn't a point, or as [`from_df`][samplesar.Sample.from_df].
        KeyError
            If the species column is missing.
        """
        geometry = gdf.geometry
        kinds = set(geometry.geom_type)
        if kinds - {"Point"}:
            raise ValueError(f"Every geometry must be a point; found {', '.join(sorted(kinds))}.")
        if species_col not in gdf.columns:
            raise KeyError(f"Missing column {species_col!r}.")
        frame = pd.DataFrame(
            {
                "species": np.asarray(gdf[species_col]),
                "lon": np.asarray(geometry.x),
                "lat": np.asarray(geometry.y),
            }
        )
        return cls.from_df(frame, crs=gdf.crs)

    @classmethod
    def from_csv(cls, path: str | Path, **kwargs: str) -> Sample:
        """Build a sample from a CSV file.

        Parameters
        ----------
        path : str or Path
            The CSV file, one row per observation.
        **kwargs : str
            ``species_col``, ``lon_col`` and ``lat_col``, as in
            [`from_df`][samplesar.Sample.from_df]. Nothing else is passed to
            the CSV reader.

        Returns
        -------
        Sample

        Raises
        ------
        KeyError
            If a column is missing.
        ValueError
            As in [`from_df`][samplesar.Sample.from_df].
        TypeError
            If a keyword other than the three column names is given.
        """
        return cls.from_df(pd.read_csv(path), **kwargs)

    @property
    def n_observations(self) -> int:
        """The number of observations."""
        return len(self.data)

    @property
    def n_species(self) -> int:
        """The number of distinct species."""
        return int(self.data["species"].nunique())

    @cached_property
    def points(self) -> np.ndarray:
        """Distinct (lon, lat) points, shape (n_points, 2), in first-seen order.

        Computed once and read-only.
        """
        points = self.data[["lon", "lat"]].drop_duplicates().to_numpy(dtype=float)
        points.flags.writeable = False
        return points

    @property
    def n_points(self) -> int:
        """The number of distinct points."""
        return len(self.points)

    @property
    def projection_centre(self) -> tuple[float, float]:
        """The (lon, lat) the points are projected about when there's no
        ``crs``: ``centre`` if given, else the centre of the points (their
        mean direction, so a sample across the antimeridian is centred
        correctly)."""
        if self.centre is not None:
            return self.centre
        return centre_of(self.points)

    @property
    def projection(self) -> CRS:
        """The projected CRS areas and distances are measured in: ``crs`` if
        set, else a Lambert azimuthal equal-area projection on the WGS84
        ellipsoid, centred on
        [`projection_centre`][samplesar.Sample.projection_centre]."""
        if self.crs is not None:
            return self.crs
        lon, lat = self.projection_centre
        return CRS.from_proj4(f"+proj=laea +lat_0={lat!r} +lon_0={lon!r} +ellps=WGS84 +units=m")

    @cached_property
    def xy(self) -> np.ndarray:
        """The points in km on [`projection`][samplesar.Sample.projection],
        shape (n_points, 2), in the order of [`points`][samplesar.Sample.points].

        All the package's hulls, areas and distances use these. By default
        the projection is equal-area, so areas measured on it are true areas
        on the ellipsoid. Computed once and read-only.
        """
        projection = self.projection
        to_plane = Transformer.from_crs(WGS84, projection, always_xy=True)
        x, y = to_plane.transform(self.points[:, 0], self.points[:, 1])
        metres = projection.axis_info[0].unit_conversion_factor
        xy = np.column_stack([x, y]) * metres / 1000.0
        xy.flags.writeable = False
        return xy

    @property
    def area(self) -> float:
        """Area of the convex hull of the points, in km².

        Measured on [`xy`][samplesar.Sample.xy]: by default an equal-area
        projection centred on the sample, or the sample's ``crs``.
        Zero when fewer than three distinct, non-collinear points remain.
        """
        return hull_area(self.xy)

    def subset(self, point_indices: Sequence[int] | np.ndarray) -> Sample:
        """The observations at some of the sample's points.

        Parameters
        ----------
        point_indices : sequence of int or np.ndarray
            Rows of [`points`][samplesar.Sample.points] to keep.

        Returns
        -------
        Sample
            Every observation at those points.
        """
        keep = self.points[list(point_indices)]
        mask = pd.MultiIndex.from_frame(self.data[["lon", "lat"]]).isin([tuple(p) for p in keep])
        centre = self.projection_centre if self.crs is None else self.centre
        return Sample(self.data.loc[mask].reset_index(drop=True), centre=centre, crs=self.crs)

    def __repr__(self) -> str:
        return (
            f"Sample(n_observations={self.n_observations}, n_points={self.n_points}, "
            f"n_species={self.n_species}, area={self.area:.6g})"
        )


def _parse_crs(crs: Any) -> CRS | None:
    if crs is None:
        return None
    try:
        return CRS.from_user_input(crs)
    except Exception as err:  # pyproj raises CRSError; anything else is equally unusable
        raise ValueError(f"Not a coordinate reference system: {crs!r} ({err})") from err


def centre_of(points: np.ndarray) -> tuple[float, float]:
    """The centre of (lon, lat) points, as the mean of their directions from
    the Earth's centre."""
    lon, lat = np.radians(np.asarray(points, dtype=float)).T
    x, y, z = (
        (np.cos(lat) * np.cos(lon)).mean(),
        (np.cos(lat) * np.sin(lon)).mean(),
        np.sin(lat).mean(),
    )
    return float(np.degrees(np.arctan2(y, x))), float(np.degrees(np.arctan2(z, np.hypot(x, y))))


def hull_area(points: np.ndarray) -> float:
    """Convex hull area of a set of points, in their own units.

    For a sample's area in km², pass its projected points,
    [`Sample.xy`][samplesar.Sample.xy] (as [`Sample.area`][samplesar.Sample.area]
    does).

    Parameters
    ----------
    points : np.ndarray
        Shape (n, 2).

    Returns
    -------
    float
        Zero for fewer than three points or collinear points.
    """
    pts = np.asarray(points, dtype=float)
    if len(pts) < 3:
        return 0.0
    return float(MultiPoint(pts).convex_hull.area)


def point_species(sample: Sample) -> tuple[list[np.ndarray], int]:
    """The species codes found at each of the sample's points, and the number
    of species."""
    data = sample.data
    codes, uniques = pd.factorize(data["species"])
    lookup = {(x, y): i for i, (x, y) in enumerate(map(tuple, sample.points))}
    which = np.array([lookup[(x, y)] for x, y in zip(data["lon"], data["lat"], strict=True)])
    grouped = pd.Series(codes).groupby(which).unique()
    return [np.asarray(grouped[i], dtype=int) for i in range(sample.n_points)], len(uniques)
