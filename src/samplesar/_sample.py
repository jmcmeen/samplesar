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
import shapely
from pyproj import CRS, Proj, Transformer
from pyproj.exceptions import ProjError
from shapely.geometry import MultiPoint

COLUMNS = ("species", "lon", "lat")
# Kept when a frame has them: meters above sea level, and when it was observed.
OPTIONAL = ("elevation", "date")


WGS84 = CRS.from_epsg(4326)
MASK_SEGMENT_DEGREES = 0.05  # a mask's longest edge before projecting


def _validate(df: pd.DataFrame, crs: CRS | None = None) -> pd.DataFrame:
    """Check and normalize the columns; with a CRS, convert its coordinates
    (in the ``lon`` and ``lat`` columns) to WGS84 longitude and latitude."""
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"Missing column(s): {missing}. Need {list(COLUMNS)}.")
    extra = [c for c in OPTIONAL if c in df.columns]
    out = df.loc[:, [*COLUMNS, *extra]].reset_index(drop=True)
    if "elevation" in extra:
        try:
            elevation = pd.to_numeric(out["elevation"]).astype(float)
        except (TypeError, ValueError) as err:
            raise ValueError(f"elevation must be numeric (meters): {err}") from err
        if np.isinf(elevation).any():
            raise ValueError("elevation must be finite or missing.")
        out["elevation"] = elevation
    if "date" in extra:
        out["date"] = _dates(out["date"])
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


def _dates(column: pd.Series) -> pd.Series:
    """A column of dates, times or text as UTC times without a zone. Text is
    parsed in one pass when it shares a format, and value by value when it
    doesn't (a date here, a time with a zone there)."""
    try:
        with warnings.catch_warnings():  # pandas warns before falling back itself
            warnings.simplefilter("ignore", UserWarning)
            when = pd.to_datetime(column, utc=True)
    except (TypeError, ValueError):
        try:
            when = column.map(lambda v: pd.to_datetime(v, utc=True))
            when = pd.to_datetime(when, utc=True)
        except (TypeError, ValueError, OverflowError) as err:
            raise ValueError(f"date must hold dates or times: {err}") from err
    return when.dt.tz_localize(None).astype("datetime64[ns]")


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
        ``lat`` (float, decimal degrees), and optionally ``elevation``
        (float, meters above sea level) and ``date`` (when it was observed),
        either of which may be missing on some rows. Several observations
        may share a point. Treat it as read-only: the points and other derived values
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
    mask : shapely.Polygon or shapely.MultiPolygon, optional
        The ground areas are measured within, in WGS84 longitude and
        latitude: land, say, so a hull reaching over the sea counts only its
        land. Set by [`with_mask`][samplesar.Sample.with_mask] or
        ``within(boundary, mask=True)``, and kept by subsets. Only its part
        near the points is kept (within a box padded by half their extent
        and a degree), with longitudes unwrapped about the projection
        center, so it may run past ±180 for a sample across the
        antimeridian. It changes areas only; which points a strategy
        removes is unchanged.
    """

    data: pd.DataFrame
    center: tuple[float, float] | None = None
    crs: CRS | None = None
    mask: Any = None

    def __post_init__(self) -> None:
        if self.center is not None:
            lon, lat = (float(v) for v in self.center)
            if not (np.isfinite(lon) and np.isfinite(lat) and -90 <= lat <= 90):
                raise ValueError(f"center must be a (lon, lat) in degrees, got {self.center!r}.")
            object.__setattr__(self, "center", (lon, lat))
        if self.mask is not None:
            from samplesar._boundary import as_geometry, near, polygonal

            mask = polygonal(shapely.make_valid(as_geometry(self.mask)))
            # Only the part near the points matters; cutting it there keeps
            # the far side of the globe out of the projection, and the saved
            # nest small.
            mask = near(mask, self.points, self.projection_center[0])
            object.__setattr__(self, "mask", mask)

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

    @cached_property
    def _point_of_row(self) -> np.ndarray:
        """The row of ``points`` each observation is at."""
        keys = pd.MultiIndex.from_frame(self.data[["lon", "lat"]])
        points = pd.MultiIndex.from_arrays([self.points[:, 0], self.points[:, 1]])
        return np.asarray(points.get_indexer(keys))

    @cached_property
    def elevations(self) -> np.ndarray:
        """Each point's elevation in meters, in the order of
        [`points`][samplesar.Sample.points]: the mean over its observations
        that have one, and NaN where none does.

        Raises
        ------
        ValueError
            If the sample has no ``elevation`` column (see
            [`sources.dataframe`][samplesar.sources.dataframe] and
            [`with_elevations`][samplesar.Sample.with_elevations]).
        """
        if "elevation" not in self.data.columns:
            raise ValueError(
                "The sample has no elevations; give sources.dataframe an elevation_col, or "
                "set them with Sample.with_elevations."
            )
        mean = self.data["elevation"].groupby(self._point_of_row).mean()
        out = mean.reindex(range(self.n_points)).to_numpy(dtype=float)
        out.flags.writeable = False
        return out

    @cached_property
    def dates(self) -> np.ndarray:
        """Each point's date, in the order of
        [`points`][samplesar.Sample.points]: when it was first observed
        (the earliest of its observations' dates), as ``datetime64[ns]``,
        and NaT where no observation has one.

        Raises
        ------
        ValueError
            If the sample has no ``date`` column (see
            [`sources.dataframe`][samplesar.sources.dataframe]).
        """
        if "date" not in self.data.columns:
            raise ValueError("The sample has no dates; give sources.dataframe a date_col.")
        first = self.data["date"].groupby(self._point_of_row).min()
        out = first.reindex(range(self.n_points)).to_numpy(dtype="datetime64[ns]")
        out.flags.writeable = False
        return out

    def with_elevations(self, elevations: Sequence[float] | np.ndarray) -> Sample:
        """The same sample with an elevation for each point.

        Parameters
        ----------
        elevations : sequence of float
            Meters above sea level, one per row of
            [`points`][samplesar.Sample.points]; NaN where unknown. Every
            observation at a point gets its point's value, replacing any
            elevations the sample had.

        Returns
        -------
        Sample

        Raises
        ------
        ValueError
            If there isn't one value per point, or a value is infinite or
            not a number.
        """
        try:
            values = np.asarray(elevations, dtype=float)
        except (TypeError, ValueError) as err:
            raise ValueError(f"elevations must be numbers (meters): {err}") from err
        if values.shape != (self.n_points,):
            raise ValueError(
                f"Need one elevation per point ({self.n_points}), got shape {values.shape}."
            )
        if np.isinf(values).any():
            raise ValueError("elevations must be finite or NaN.")
        data = self.data.copy()
        data["elevation"] = values[self._point_of_row]
        return Sample(data, center=self.center, crs=self.crs, mask=self.mask)

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

    @cached_property
    def _mask_plane(self) -> shapely.Geometry:
        """The whole mask in km on the projection (a subset with the same
        projection inherits it rather than projecting it again)."""
        from samplesar._boundary import polygonal

        to_plane = Transformer.from_crs(WGS84, self.projection, always_xy=True)
        km = self.projection.axis_info[0].unit_conversion_factor / 1000.0
        # Short edges, so straight lines in degrees stay true once projected.
        dense = shapely.segmentize(self.mask, MASK_SEGMENT_DEGREES)

        def project(coords: np.ndarray) -> np.ndarray:
            x, y = to_plane.transform(coords[:, 0], coords[:, 1])
            return np.column_stack([x, y]) * km

        projected = shapely.transform(dense, project)
        if not np.isfinite(shapely.get_coordinates(projected)).all():
            raise ValueError(
                f"Part of the mask can't be projected to {self.projection.name}; cut it to "
                "the sample's region first (regions.read's clip, say)."
            )
        return polygonal(shapely.make_valid(projected))

    @cached_property
    def mask_xy(self) -> shapely.Geometry | None:
        """The mask (see [`with_mask`][samplesar.Sample.with_mask]) in km on
        [`projection`][samplesar.Sample.projection], cut to the convex hull of
        all the points (every step's hull lies inside it); None without a
        mask. Computed once.

        Raises
        ------
        ValueError
            If part of the mask can't be projected to ``crs``.
        """
        if self.mask is None:
            return None
        from samplesar._boundary import polygonal

        hull = MultiPoint(self.xy).convex_hull
        clipped = polygonal(self._mask_plane.intersection(hull))
        shapely.prepare(clipped)
        return clipped

    def _derived(self, data: pd.DataFrame) -> Sample:
        """A sample of some of these observations, on the same projection and
        mask, reusing the projected mask if it's been computed."""
        child = Sample(data, center=self.projection_center, crs=self.crs, mask=self.mask)
        if "_mask_plane" in self.__dict__:
            child.__dict__["_mask_plane"] = self.__dict__["_mask_plane"]
        return child

    def hull_area_of(self, point_indices: Sequence[int] | np.ndarray) -> float:
        """The area, in km², of the convex hull of some of the sample's points,
        within the mask (see [`with_mask`][samplesar.Sample.with_mask]) if there is one.

        Measured as [`area`][samplesar.Sample.area] is.

        Parameters
        ----------
        point_indices : sequence of int or np.ndarray
            Rows of [`points`][samplesar.Sample.points].

        Returns
        -------
        float
            Zero for fewer than three points, collinear points, or a hull
            wholly outside the mask.
        """
        return self.areas_of(point_indices)[0]

    def areas_of(self, point_indices: Sequence[int] | np.ndarray) -> tuple[float, float]:
        """The area of some points' convex hull within the mask, and the
        whole hull's area, both in km² (the same without a mask).

        Parameters
        ----------
        point_indices : sequence of int or np.ndarray
            Rows of [`points`][samplesar.Sample.points].

        Returns
        -------
        tuple of float
        """
        xy = self.xy[np.asarray(point_indices, dtype=int)]
        if len(xy) < 3:
            return 0.0, 0.0
        hull = MultiPoint(xy).convex_hull
        whole = float(hull.area) / self.area_scale
        if self.mask is None or whole == 0:
            return whole, whole
        return self._mask_tiles.area_within(hull) / self.area_scale, whole

    @cached_property
    def _mask_tiles(self) -> _Tiles:
        mask = self.mask_xy
        assert mask is not None
        return _Tiles(mask)

    @property
    def area(self) -> float:
        """Area of the convex hull of the points, in km², within the
        mask (see [`with_mask`][samplesar.Sample.with_mask]) if there is one.

        Measured on [`xy`][samplesar.Sample.xy] and divided by
        [`area_scale`][samplesar.Sample.area_scale]: true areas on the
        default equal-area projection, and corrected for a ``crs``'s area
        distortion. Zero when fewer than three distinct, non-collinear points
        remain.
        """
        return self.hull_area_of(np.arange(self.n_points))

    def with_mask(self, mask: Any) -> Sample:
        """The same sample, with its areas measured within a region.

        A convex hull can span ground that isn't habitat: sea along a coast,
        an inlet, a lake, the bay of a concave island. Its area then
        overstates the area sampled, most at the large end of a nest. With a
        mask, every area (the sample's, and each step's in
        [`Nest.to_frame`][samplesar.Nest.to_frame], ``sar_data`` and ``fit``)
        is the area of the hull's intersection with the mask, on the same
        equal-area projection. Nests are unchanged: strategies choose points
        by their hulls as before.

        Parameters
        ----------
        mask : polygon or None
            A shapely polygon or multipolygon in WGS84 longitude and
            latitude, or a GeoDataFrame or GeoSeries in any CRS (converted
            and dissolved), such as [`regions.admin`][samplesar.regions.admin]
            returns; None removes the mask. Observations outside it are kept
            (use [`within`][samplesar.Sample.within] to drop them).

        Returns
        -------
        Sample
            With the same observations, ``center`` and ``crs``.

        Raises
        ------
        ValueError
            If ``mask`` isn't polygonal or is empty.
        """
        return Sample(self.data, center=self.center, crs=self.crs, mask=mask)

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
        rows = self._point_of_row
        data = self.data.copy()
        data["lon"], data["lat"] = target[rows, 0], target[rows, 1]
        return self._derived(data)

    def within(self, boundary: Any, mask: bool = False) -> Sample:
        """The observations inside a region (or on its edge).

        Parameters
        ----------
        boundary : polygon
            A shapely polygon or multipolygon in WGS84 longitude and
            latitude, or a GeoDataFrame or GeoSeries of them in any CRS (it
            is converted and dissolved), such as
            [`regions.admin`][samplesar.regions.admin] returns.
        mask : bool
            Also measure areas within the region (see
            [`with_mask`][samplesar.Sample.with_mask]), so a hull reaching
            beyond it counts only its part inside. By default the sample's
            mask, if any, is kept.

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

        region = as_geometry(boundary)
        keep = inside(region, self.data["lon"], self.data["lat"])
        if not keep.any():
            raise ValueError("No observations lie inside the boundary.")
        return Sample(
            self.data[keep].reset_index(drop=True),
            center=self.center,
            crs=self.crs,
            mask=region if mask else self.mask,
        )

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
        return self._derived(data)

    def __repr__(self) -> str:
        return (
            f"Sample(n_observations={self.n_observations}, n_points={self.n_points}, "
            f"n_species={self.n_species}, area={self.area:.6g}"
            + (", masked" if self.mask is not None else "")
            + ")"
        )


class _Tiles:
    """A mask cut into a grid of tiles, so the area of a polygon within it
    costs one intersection per tile along the polygon's edge, not one with
    the whole mask (a coastline of many thousand vertices)."""

    GRID = 32  # tiles across the mask's longer side

    def __init__(self, mask: shapely.Geometry) -> None:
        self.mask = mask
        west, south, east, north = mask.bounds if not mask.is_empty else (0, 0, 0, 0)
        from samplesar._boundary import polygonal

        size = max(east - west, north - south) / self.GRID
        tiles: list[shapely.Geometry] = []
        if size > 0:
            # Edges from the counts, so the last tiles reach the far bounds.
            xs = np.linspace(west, east, max(1, round((east - west) / size)) + 1)
            ys = np.linspace(south, north, max(1, round((north - south) / size)) + 1)
            for x0, x1 in zip(xs, xs[1:], strict=False):
                for y0, y1 in zip(ys, ys[1:], strict=False):
                    tile = polygonal(mask.intersection(shapely.box(x0, y0, x1, y1)))
                    if tile.area > 0:
                        tiles.append(tile)
        self.tiles = np.array(tiles, dtype=object)
        self.areas = shapely.area(self.tiles) if tiles else np.empty(0)
        self.tree = shapely.STRtree(self.tiles)

    def area_within(self, polygon: shapely.Geometry) -> float:
        """The area of ``polygon`` inside the mask, in the mask's units."""
        if shapely.contains(self.mask, polygon):
            return float(polygon.area)
        near = self.tree.query(polygon)
        if not len(near):
            return 0.0
        shapely.prepare(polygon)
        inside = shapely.contains_properly(polygon, self.tiles[near])
        edge = self.tiles[near[~inside]]
        return float(
            self.areas[near[inside]].sum() + shapely.area(shapely.intersection(edge, polygon)).sum()
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
