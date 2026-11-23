"""Point observations of species, the input every nesting strategy starts from."""

from __future__ import annotations

import numbers
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Any

import numpy as np
import pandas as pd
from pyproj import CRS, Proj, Transformer
from pyproj.exceptions import ProjError
from shapely.geometry import MultiPoint

COLUMNS = ("species", "lon", "lat")


WGS84 = CRS.from_epsg(4326)


def _validate(df: pd.DataFrame, crs: CRS | None = None) -> pd.DataFrame:
    """Check and normalize the columns; with a CRS, convert its coordinates
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
    converted = crs is not None and crs != WGS84
    if converted:
        assert crs is not None
        try:
            to_wgs84 = Transformer.from_crs(crs, WGS84, always_xy=True)
            x, y = to_wgs84.transform(lon.to_numpy(float), lat.to_numpy(float))
        except ProjError as err:
            raise ValueError(f"Can't convert coordinates in {crs.name} to WGS84: {err}") from err
        if not (np.isfinite(x).all() and np.isfinite(y).all()):
            raise ValueError(f"Some coordinates fall outside {crs.name}.")
        lon, lat = pd.Series(x), pd.Series(y)
    out_of_range = ((lat < -90) | (lat > 90)).any() or ((lon < -180) | (lon > 180)).any()
    if out_of_range and converted:
        assert crs is not None
        raise ValueError(f"Some coordinates fall outside {crs.name}'s area once converted.")
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

    Build one with [`sources.dataframe`][samplesar.sources.dataframe] or
    [`sources.csv`][samplesar.sources.csv] rather than directly: they check
    and normalize the columns.

    Attributes
    ----------
    data : pd.DataFrame
        One row per observation, with columns ``species`` (str), ``lon`` and
        ``lat`` (float, decimal degrees). Several observations may share a
        point. Treat it as read-only: the points and other derived values
        are computed once, so build a new sample rather than editing it.
    center : tuple of float, optional
        The (lon, lat) the points are projected about for areas and
        distances (see [`xy`][samplesar.Sample.xy]), when there's no
        ``crs``. By default the center of the sample's points; a
        [`subset`][samplesar.Sample.subset] keeps its parent's, so their
        areas are comparable.
    crs : pyproj.CRS, optional
        A projected coordinate reference system to measure areas and
        distances in, such as the UTM zone the sample was laid out in. Set by
        [`sources.dataframe`][samplesar.sources.dataframe] when the coordinates come in
        one.
    """

    data: pd.DataFrame
    center: tuple[float, float] | None = None
    crs: CRS | None = None

    def __post_init__(self) -> None:
        if self.center is not None:
            lon, lat = (float(v) for v in self.center)
            if not (np.isfinite(lon) and np.isfinite(lat) and -90 <= lat <= 90):
                raise ValueError(f"center must be a (lon, lat) in degrees, got {self.center!r}.")
            object.__setattr__(self, "center", (lon, lat))

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
    def projection_center(self) -> tuple[float, float]:
        """The (lon, lat) the points are projected about when there's no
        ``crs``: ``center`` if given, else the center of the points (their
        mean direction, so a sample across the antimeridian is centered
        correctly)."""
        if self.center is not None:
            return self.center
        return center_of(self.points)

    @property
    def projection(self) -> CRS:
        """The projected CRS areas and distances are measured in: ``crs`` if
        set, else a Lambert azimuthal equal-area projection on the WGS84
        ellipsoid, centered on
        [`projection_center`][samplesar.Sample.projection_center]."""
        if self.crs is not None:
            return self.crs
        lon, lat = self.projection_center
        return CRS.from_proj4(
            f"+proj=laea +lat_0={lat:.12f} +lon_0={lon:.12f} +ellps=WGS84 +units=m"
        )

    @cached_property
    def area_scale(self) -> float:
        """How much [`projection`][samplesar.Sample.projection] enlarges areas
        at [`projection_center`][samplesar.Sample.projection_center]; areas
        are divided by it.

        1 for the default equal-area projection, about 1 for UTM within its
        zone, and 4 for Web Mercator at 60° latitude (where the corrected
        areas are good to a few 0.1%: Web Mercator treats WGS84 coordinates as
        if on a sphere). A warning is given if it
        varies by more than 1% across the sample, where one factor can't
        correct every part of it.
        """
        if self.crs is None:
            return 1.0
        proj = Proj(self.crs)
        lon, lat = self.projection_center
        scale = float(proj.get_factors(lon, lat).areal_scale)
        across = np.asarray(proj.get_factors(self.points[:, 0], self.points[:, 1]).areal_scale)
        if across.size and (across.max() / across.min() > 1.01):
            warnings.warn(
                f"{self.crs.name} distorts areas by {across.min():.3g}x to {across.max():.3g}x "
                "across this sample, so its areas are only approximate; a CRS that is "
                "equal-area here, or none (the default projection), measures them truly.",
                stacklevel=2,
            )
        return scale

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
        meters = projection.axis_info[0].unit_conversion_factor
        xy = np.column_stack([x, y]) * meters / 1000.0
        xy.flags.writeable = False
        return xy

    @property
    def area(self) -> float:
        """Area of the convex hull of the points, in km².

        Measured on [`xy`][samplesar.Sample.xy] and divided by
        [`area_scale`][samplesar.Sample.area_scale]: true areas on the
        default equal-area projection, and corrected for a ``crs``'s area
        distortion. Zero when fewer than three distinct, non-collinear points
        remain.
        """
        return hull_area(self.xy) / self.area_scale

    def merge_within(self, meters: float) -> Sample:
        """Merge points within a distance of each other into one, keeping every observation.

        Records made at one spot (a trailhead, a feeder, a car park) differ
        only by GPS jitter. Left apart, they make the last steps of a hull
        nest measure hulls centimeters across, which aren't area but dominate
        the species-area curve. Points at most ``meters`` apart are linked
        (on [`xy`][samplesar.Sample.xy], with a distorting ``crs`` corrected
        by its scale at the sample's center), and each linked group becomes
        one point at its first-seen member's coordinates, holding all the
        group's observations. Linking chains: three points 0.8 apart with
        ``meters=1`` become one, and a warning says when a chain spans more
        than ten times ``meters``, since its observations then move far.

        Parameters
        ----------
        meters : float
            Merge points at most this far apart (positive).

        Returns
        -------
        Sample
            With the same projection as this one.

        Raises
        ------
        ValueError
            If ``meters`` isn't a positive number.
        """
        if isinstance(meters, bool) or not isinstance(meters, numbers.Real):
            raise ValueError(f"meters must be a positive number, got {meters!r}.")
        meters = float(meters)
        if not (np.isfinite(meters) and meters > 0):
            raise ValueError(f"meters must be a positive number, got {meters!r}.")
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        from scipy.spatial import cKDTree

        n = self.n_points
        # xy is in the projection's km; a distorting crs stretches lengths by
        # about the square root of its areal scale.
        radius = meters / 1000.0 * np.sqrt(self.area_scale)
        pairs = cKDTree(self.xy).query_pairs(radius, output_type="ndarray")
        graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
        _, group = connected_components(graph, directed=False)
        # Each group's first-seen point represents it.
        first = np.full(group.max() + 1, n)
        np.minimum.at(first, group, np.arange(n))
        target = self.points[first[group]]
        shared = np.unique(group[np.bincount(group)[group] > 1])
        spans = [np.ptp(self.xy[group == g], axis=0).max() for g in shared]
        if spans and max(spans) > 10 * radius:
            warnings.warn(
                f"Linked points chain over {max(spans) * 1000 / np.sqrt(self.area_scale):.0f} m, "
                f"more than ten times {meters:g} m, so some observations move far; a "
                "smaller distance keeps them nearer.",
                stacklevel=2,
            )
        keys = pd.MultiIndex.from_frame(self.data[["lon", "lat"]])
        rows = pd.MultiIndex.from_arrays([self.points[:, 0], self.points[:, 1]]).get_indexer(keys)
        data = self.data.copy()
        data["lon"], data["lat"] = target[rows, 0], target[rows, 1]
        return Sample(data, center=self.projection_center, crs=self.crs)

    def within(self, boundary: Any) -> Sample:
        """The observations inside a region (or on its edge).

        Parameters
        ----------
        boundary : polygon
            A shapely polygon or multipolygon in WGS84 longitude and
            latitude, or a GeoDataFrame or GeoSeries of them in any CRS (it
            is converted and dissolved), such as
            [`regions.admin`][samplesar.regions.admin] returns.

        Returns
        -------
        Sample
            With the same ``crs`` and ``center``; without an explicit
            ``center``, its projection is centered on the points kept, as for
            a new sample.

        Raises
        ------
        ValueError
            If ``boundary`` isn't polygonal, or no observation lies inside it.
        """
        from samplesar._boundary import as_geometry, inside

        keep = inside(as_geometry(boundary), self.data["lon"], self.data["lat"])
        if not keep.any():
            raise ValueError("No observations lie inside the boundary.")
        return Sample(self.data[keep].reset_index(drop=True), center=self.center, crs=self.crs)

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
        data = self.data.loc[mask].reset_index(drop=True)
        return Sample(data, center=self.projection_center, crs=self.crs)

    def __repr__(self) -> str:
        return (
            f"Sample(n_observations={self.n_observations}, n_points={self.n_points}, "
            f"n_species={self.n_species}, area={self.area:.6g})"
        )


def _parse_crs(crs: Any) -> CRS | None:
    if crs is None:
        return None
    try:
        parsed = CRS.from_user_input(crs)
    except Exception as err:  # pyproj raises CRSError; anything else is equally unusable
        raise ValueError(f"Not a coordinate reference system: {crs!r} ({err})") from err
    if not (parsed.is_geographic or parsed.is_projected):
        raise ValueError(
            f"{parsed.name} is neither geographic nor projected; give a CRS for points."
        )
    return parsed


def center_of(points: np.ndarray) -> tuple[float, float]:
    """The center of (lon, lat) points, as the mean of their directions from
    the Earth's center."""
    lon, lat = np.radians(np.asarray(points, dtype=float)).T
    x, y, z = (
        (np.cos(lat) * np.cos(lon)).mean(),
        (np.cos(lat) * np.sin(lon)).mean(),
        np.sin(lat).mean(),
    )
    return float(np.degrees(np.arctan2(y, x))), float(np.degrees(np.arctan2(z, np.hypot(x, y))))


def hull_area(points: np.ndarray) -> float:
    """Convex hull area of a set of points, in their own units.

    Not to be confused with the strategy
    [`strategies.hull_area`][samplesar.strategies.hull_area], which nests a
    sample.

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
