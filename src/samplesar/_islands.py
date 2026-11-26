"""Regions as sampling units: one area and species count per region."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Literal

import numpy as np
import pandas as pd
import sars
import shapely
from pyproj import Geod, Transformer

from samplesar._boundary import as_geometry, polygonal
from samplesar._sample import MASK_SEGMENT_DEGREES, WGS84, Sample, point_species

GEOD = Geod(ellps="WGS84")
MIN_FIT_ROWS = 3


def _ring_area(ring: shapely.Geometry) -> float:
    lon, lat = np.asarray(ring.coords).T
    return abs(float(GEOD.polygon_area_perimeter(lon, lat)[0]))


def geodesic_area(region: shapely.Geometry) -> float:
    """A WGS84 polygon's or multipolygon's area on the ellipsoid, in km².

    Its edges are taken as straight in longitude and latitude, as the
    point-in-region tests take them, so long edges are cut into short
    pieces before measuring (a geodesic between two far corners of a
    parallel bows away from it). Each ring is measured on its own, so the
    result doesn't depend on which way rings are wound."""
    if region.is_empty:
        return 0.0
    total = 0.0
    for polygon in shapely.get_parts(shapely.segmentize(region, MASK_SEGMENT_DEGREES)):
        if polygon.geom_type != "Polygon":
            continue
        total += _ring_area(polygon.exterior) - sum(_ring_area(hole) for hole in polygon.interiors)
    return max(total, 0.0) / 1e6


def _named(regions: Any, names: Any) -> dict[str, shapely.Geometry]:
    """Regions as a dict of name to valid WGS84 polygon, from a mapping, a
    sequence or array of polygons, or a GeoSeries or GeoDataFrame."""
    if hasattr(regions, "to_crs") and hasattr(regions, "crs"):
        frame = regions if regions.crs is None else regions.to_crs(4326)
        geometry = list(getattr(frame, "geometry", frame))
        if isinstance(names, str):
            if names not in getattr(frame, "columns", ()):
                raise KeyError(f"The regions have no column {names!r} to name them by.")
            names = list(frame[names])
        elif names is None:
            names = list(frame.index)
        regions = geometry
    if isinstance(regions, Mapping):
        if names is not None:
            raise ValueError("A mapping of regions already names them; leave names out.")
        names, regions = list(regions), list(regions.values())
    elif isinstance(regions, shapely.Geometry | str) or not isinstance(regions, Iterable):
        raise ValueError(
            "regions must be a mapping of name to polygon, a sequence of polygons, or a "
            f"GeoSeries or GeoDataFrame; got {type(regions).__name__}."
        )
    regions = list(regions)
    if not regions:
        raise ValueError("No regions given.")
    if names is None:
        names = [f"region {i}" for i in range(len(regions))]
    elif isinstance(names, str):
        raise ValueError("names as a column name needs a GeoDataFrame; give one name per region.")
    names = [str(name) for name in names]
    if len(names) != len(regions):
        raise ValueError(f"{len(regions)} regions but {len(names)} names.")
    seen: set[str] = set()
    repeated = sorted({name for name in names if name in seen or seen.add(name)})  # type: ignore[func-returns-value]
    if repeated:
        raise ValueError(
            f"Region names must differ; repeated: {', '.join(repeated[:5])}. Dissolve the "
            "rows of one region first, or name them apart."
        )
    named = {}
    for name, region in zip(names, regions, strict=True):
        try:
            # Published outlines sometimes cross themselves: repair them, as
            # a sample's mask is repaired.
            named[name] = polygonal(shapely.make_valid(as_geometry(region)))
        except (ValueError, TypeError, shapely.errors.GEOSException) as err:
            raise ValueError(f"Region {name!r} isn't a usable polygon: {err}") from err
        if named[name].is_empty:
            raise ValueError(f"Region {name!r} is empty.")
    return named


@dataclass(frozen=True, eq=False)
class Islands:
    """A sample divided among regions, each one a sampling unit.

    Made by [`designs.islands`][samplesar.designs.islands]. Where a
    [`Nest`][samplesar.Nest] shrinks one sample step by step, this counts
    the species in each of several given regions (states, parks, real
    islands), so its species-area data are of the island type: one
    independent row per region.

    Attributes
    ----------
    sample : Sample
        The observations.
    regions : mapping of str to polygon
        Each region's name and outline, in WGS84 longitude and latitude.
    area : {"region", "hull"}
        What each region's area is: its own (``"region"``), or the convex
        hull of its points, cut to the region (``"hull"``).
    mask : polygon, optional
        For ``area="region"``: ground the regions' areas are measured
        within, such as land.
    """

    sample: Sample
    regions: Mapping[str, Any]
    area: str = "region"
    mask: Any = None

    def __post_init__(self) -> None:
        if self.area not in ("region", "hull"):
            raise ValueError(f"area must be 'region' or 'hull', got {self.area!r}")

    @property
    def label(self) -> str:
        """``islands (n regions)``, for plots and tables."""
        return f"islands ({len(self.regions)} regions)"

    @cached_property
    def _members(self) -> dict[str, np.ndarray]:
        """The rows of ``sample.points`` in each region (or on its edge):
        one indexed search of the points for all the regions."""
        points = self.sample.points
        tree = shapely.STRtree(shapely.points(points))
        regions = np.array(list(self.regions.values()), dtype=object)
        which, point = tree.query(regions, predicate="intersects")
        order = np.lexsort((point, which))
        which, point = which[order], point[order]
        bounds = np.searchsorted(which, np.arange(len(regions) + 1))
        return {name: point[bounds[i] : bounds[i + 1]] for i, name in enumerate(self.regions)}

    def _hull_area(self, points: np.ndarray, region: Any) -> float:
        """The area of the convex hull of some points (on the sample's
        projection) that lies within a region, in km²."""
        sample = self.sample
        hull = shapely.MultiPoint(sample.xy[points]).convex_hull
        if not isinstance(hull, shapely.Polygon) or not hull.area > 0:
            return 0.0
        projection = sample.projection
        km = projection.axis_info[0].unit_conversion_factor / 1000.0
        back = Transformer.from_crs(projection, WGS84, always_xy=True)
        # Cut into short edges first: a straight edge on the projection
        # curves in longitude and latitude.
        fine = shapely.segmentize(hull, max(float(np.sqrt(hull.area)) / 50.0, 1e-6))
        outline = shapely.transform(
            fine, lambda c: np.column_stack(back.transform(c[:, 0] / km, c[:, 1] / km))
        )
        return geodesic_area(polygonal(shapely.make_valid(outline).intersection(region)))

    def sample_of(self, name: str) -> Sample:
        """The observations in one region, as a [`Sample`][samplesar.Sample]
        whose areas are measured within the region, ready to nest. The
        region replaces any mask the sample had.

        Raises
        ------
        KeyError
            If there is no such region.
        ValueError
            If the region holds no observations.
        """
        if name not in self.regions:
            raise KeyError(f"No region {name!r}; known: {', '.join(list(self.regions)[:10])}.")
        inside = self._members[name]
        if not len(inside):
            raise ValueError(f"Region {name!r} holds no observations.")
        return self.sample.subset(inside).with_mask(self.regions[name])

    def to_frame(self) -> pd.DataFrame:
        """One row per region.

        Returns
        -------
        pd.DataFrame
            ``region``, ``area`` (km²), ``species``, ``n_points`` and
            ``n_observations``, in the order the regions were given; and
            ``region_area``, each region's whole area on the ellipsoid, when
            ``area`` is something else (a hull, or the part within a mask).
            A region without observations has zero species.
        """
        return self._frame.copy()

    @cached_property
    def _frame(self) -> pd.DataFrame:
        sample = self.sample
        species, _ = point_species(sample)
        observations = np.bincount(sample._point_of_row, minlength=sample.n_points)
        rows = []
        for name, region in self.regions.items():
            inside = self._members[name]
            whole = geodesic_area(region)
            if self.area == "hull":
                area = self._hull_area(inside, region)
            elif self.mask is not None:
                area = geodesic_area(polygonal(region.intersection(self.mask)))
            else:
                area = whole
            found = np.unique(np.concatenate([species[p] for p in inside])) if len(inside) else ()
            row = {
                "region": name,
                "area": area,
                "species": len(found),
                "n_points": len(inside),
                "n_observations": int(observations[inside].sum()),
            }
            if self.area == "hull" or self.mask is not None:
                row["region_area"] = whole
            rows.append(row)
        return pd.DataFrame(rows)

    def sar_data(self, min_observations: int = 1) -> pd.DataFrame:
        """The species-area data sars fits.

        Parameters
        ----------
        min_observations : int
            Leave out regions with fewer observations than this. With the
            default, regions without any are left out: an empty region in
            observation data is more often unvisited than without species.
            0 keeps them, as islands with no species.

        Returns
        -------
        pd.DataFrame
            ``area`` (km²) and ``species``, one row per region kept, without
            regions of no area.
        """
        frame = self.to_frame()
        keep = (frame["n_observations"] >= min_observations) & (frame["area"] > 0)
        return frame.loc[keep, ["area", "species"]].reset_index(drop=True)

    def fit(self, model: str = "power", min_observations: int = 1, **kwargs: Any) -> Any:
        """Fit one of sars' models to the regions' species-area data.

        Parameters
        ----------
        model : str
            A sars model: ``"power"`` calls ``sars.sar_power``, ``"loga"``
            ``sars.sar_loga``, and so on.
        min_observations : int
            As in [`sar_data`][samplesar.Islands.sar_data].
        **kwargs
            Passed to the sars function.

        Returns
        -------
        object
            sars' fit (``SARFit`` for a single model).

        Raises
        ------
        ValueError
            If sars has no such model, or fewer than three regions have
            data to fit.
        """
        func = getattr(sars, f"sar_{model}", None)
        if not callable(func):
            raise ValueError(f"sars has no model {model!r} (no sars.sar_{model})")
        data = self.sar_data(min_observations)
        if len(data) < MIN_FIT_ROWS:
            raise ValueError(
                f"{len(data)} region(s) have observations and an area; a fit needs at least "
                f"{MIN_FIT_ROWS}."
            )
        return func(data, **kwargs)


def islands(
    sample: Sample,
    regions: Any,
    names: Sequence[str] | str | None = None,
    area: Literal["region", "hull"] = "region",
    mask: Any = None,
) -> Islands:
    """Divide a sample among regions, each one a sampling unit.

    For species-area data of the island type from the same observations:
    states, counties, parks, ecoregions or real islands, one row each, with
    the region's area and the species observed in it. The regions are
    yours to choose, so this is a different design from the nesting
    strategies, which sample one region arbitrarily; the two kinds of curve
    aren't comparable point for point.

    Regions with more observers have more species recorded, whatever their
    area, so the table also gives each region's observations and points.

    Parameters
    ----------
    sample : Sample
        The observations. Those outside every region are ignored; one in
        two overlapping regions counts in both.
    regions : mapping, sequence, GeoSeries or GeoDataFrame
        The regions: a mapping of name to shapely polygon in WGS84 (as
        [`regions.admin_units`][samplesar.regions.admin_units] and
        [`regions.read_units`][samplesar.regions.read_units] return), a
        sequence or array of polygons, or a GeoSeries or GeoDataFrame in any
        CRS. Outlines that cross themselves are repaired. Longitudes must
        be within -180 to 180, as the sample's are.
    names : sequence of str or str, optional
        A name per region, for a sequence; or, for a GeoDataFrame, the
        column that names them (by default its index). Names must differ.
    area : {"region", "hull"}
        Each region's area: ``"region"``, its own area on the WGS84
        ellipsoid (the island's area, as island SARs use), or ``"hull"``,
        the area of the convex hull of its points that lies within it (the
        ground the observations span; the hull is drawn on the sample's
        projection).
    mask : polygon, optional
        With ``area="region"``, measure each region within this ground
        (land, say): a shapely polygon in WGS84, or a GeoDataFrame or
        GeoSeries. The sample's own mask isn't used: it is kept only near
        the sample's points.

    Returns
    -------
    Islands

    Raises
    ------
    ValueError
        If ``regions`` isn't polygons, is empty, has repeated names or not
        as many names as regions, or a region is empty or unusable;
        ``area`` is unknown; or ``mask`` is given with ``area="hull"`` or
        isn't polygonal.
    KeyError
        If ``names`` is a column the regions don't have.

    Examples
    --------
    >>> states = samplesar.regions.admin_units("USA", level=1)  # doctest: +SKIP
    >>> islands = samplesar.designs.islands(sample, states)  # doctest: +SKIP
    >>> islands.to_frame()  # doctest: +SKIP
    >>> islands.fit("power")  # doctest: +SKIP
    """
    if mask is not None and area == "hull":
        raise ValueError("mask applies to area='region'; a hull is already cut to its region.")
    named = _named(regions, names)
    ground = None if mask is None else shapely.make_valid(as_geometry(mask))
    return Islands(sample=sample, regions=named, area=area, mask=ground)
