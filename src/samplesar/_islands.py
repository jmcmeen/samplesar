"""Regions as sampling units: one area and species count per region."""

from __future__ import annotations

import numbers
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
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

    sample: Sample | None
    regions: Mapping[str, Any]
    area: str = "region"
    mask: Any = None
    areas: Mapping[str, float] = field(default_factory=dict)
    units: Mapping[str, tuple[float, int]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.area not in ("region", "hull"):
            raise ValueError(f"area must be 'region' or 'hull', got {self.area!r}")
        if self.regions and self.sample is None:
            raise ValueError("Regions need a sample to count species in; give one, or only units.")
        both = sorted(set(self.regions) & set(self.units))
        if both:
            raise ValueError(f"Names must differ; both a region and a unit: {', '.join(both[:5])}.")
        unknown = sorted(set(self.areas) - set(self.regions))
        if unknown:
            raise ValueError(
                f"areas names no region: {', '.join(unknown[:5])}. A unit's area is given with it."
            )
        if not self.regions and not self.units:
            raise ValueError("No regions or units given.")

    @property
    def names(self) -> list[str]:
        """Every unit's name: the regions, then the units given by hand."""
        return [*self.regions, *self.units]

    @property
    def label(self) -> str:
        """``islands (n regions)``, for plots and tables."""
        return f"islands ({len(self.names)} regions)"

    def add(
        self,
        regions: Any = None,
        names: Sequence[str] | str | None = None,
        *,
        units: Any = None,
        areas: Mapping[str, float] | None = None,
    ) -> Islands:
        """The same design with more regions or units, from any source.

        Parameters
        ----------
        regions : mapping, sequence, GeoSeries or GeoDataFrame, optional
            More regions, as [`designs.islands`][samplesar.designs.islands]
            takes them.
        names : sequence of str or str, optional
            Their names, as ``designs.islands`` takes them.
        units : mapping or DataFrame, optional
            More units given by hand, as ``designs.islands`` takes them.
        areas : mapping of str to float, optional
            Areas (km²) to use for regions instead of measuring them, added
            to those already set.

        Returns
        -------
        Islands

        Raises
        ------
        ValueError
            If a name is already in the design, or as ``designs.islands``.
        """
        more_regions = {} if regions is None else _named(regions, names)
        more_units = _units(units)
        taken = sorted((set(more_regions) | set(more_units)) & set(self.names))
        if taken:
            raise ValueError(
                f"Already in the design: {', '.join(taken[:5])}. Names must differ; drop the "
                "old one first to replace it."
            )
        return Islands(
            sample=self.sample,
            regions={**self.regions, **more_regions},
            area=self.area,
            mask=self.mask,
            areas={**self.areas, **_areas(areas)},
            units={**self.units, **more_units},
        )

    def select(self, names: Iterable[str]) -> Islands:
        """The design with only the units named, in the order named.

        Raises
        ------
        KeyError
            If a name isn't in the design.
        """
        wanted = [names] if isinstance(names, str) else list(names)
        missing = [name for name in wanted if name not in self.regions and name not in self.units]
        if missing:
            raise KeyError(
                f"Not in the design: {', '.join(map(str, missing[:5]))}; known: "
                f"{', '.join(self.names[:10])}."
            )
        return Islands(
            sample=self.sample if any(n in self.regions for n in wanted) else None,
            regions={n: self.regions[n] for n in wanted if n in self.regions},
            area=self.area,
            mask=self.mask,
            areas={n: self.areas[n] for n in wanted if n in self.areas},
            units={n: self.units[n] for n in wanted if n in self.units},
        )

    def drop(self, names: Iterable[str]) -> Islands:
        """The design without the units named.

        Raises
        ------
        KeyError
            If a name isn't in the design.
        ValueError
            If nothing would be left.
        """
        gone = [names] if isinstance(names, str) else list(names)
        missing = [name for name in gone if name not in self.regions and name not in self.units]
        if missing:
            raise KeyError(f"Not in the design: {', '.join(map(str, missing[:5]))}.")
        return self.select([name for name in self.names if name not in gone])

    @cached_property
    def _members(self) -> dict[str, np.ndarray]:
        """The rows of ``sample.points`` in each region (or on its edge):
        one indexed search of the points for all the regions."""
        if self.sample is None:
            return {}
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
        assert sample is not None
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
        if name in self.units:
            raise KeyError(f"{name!r} was given by hand: it has no boundary or observations.")
        if name not in self.regions:
            raise KeyError(f"No region {name!r}; known: {', '.join(list(self.regions)[:10])}.")
        inside = self._members[name]
        if not len(inside):
            raise ValueError(f"Region {name!r} holds no observations.")
        assert self.sample is not None
        return self.sample.subset(inside).with_mask(self.regions[name])

    def to_frame(self) -> pd.DataFrame:
        """One row per region.

        Returns
        -------
        pd.DataFrame
            ``region``, ``area`` (km²), ``species``, ``n_points``,
            ``n_observations`` and ``source``, the regions first and then
            the units given by hand, each in the order given. ``source``
            says where a row's numbers come from: ``"measured"`` (area and
            species from the region and the observations in it), ``"area
            given"`` (species counted, area yours) or ``"given"`` (both
            yours; its point and observation counts are missing).
            ``region_area``, each region's whole area on the ellipsoid, is
            added when ``area`` is something else (a hull, the part within
            a mask, or an area you gave). A region without observations has
            zero species.
        """
        return self._frame.copy()

    @cached_property
    def _frame(self) -> pd.DataFrame:
        rows = []
        sample = self.sample
        if sample is not None and self.regions:
            species, _ = point_species(sample)
            observations = np.bincount(sample._point_of_row, minlength=sample.n_points)
        extra = self.area == "hull" or self.mask is not None or bool(self.areas)
        for name, region in self.regions.items():
            inside = self._members[name]
            whole = geodesic_area(region)
            if name in self.areas:
                area, source = float(self.areas[name]), "area given"
            elif self.area == "hull":
                area, source = self._hull_area(inside, region), "measured"
            elif self.mask is not None:
                area = geodesic_area(polygonal(region.intersection(self.mask)))
                source = "measured"
            else:
                area, source = whole, "measured"
            found = np.unique(np.concatenate([species[p] for p in inside])) if len(inside) else ()
            row = {
                "region": name,
                "area": area,
                "species": len(found),
                "n_points": len(inside),
                "n_observations": int(observations[inside].sum()),
                "source": source,
            }
            if extra:
                row["region_area"] = whole
            rows.append(row)
        for name, (area, count) in self.units.items():
            row = {
                "region": name,
                "area": float(area),
                "species": int(count),
                "n_points": pd.NA,
                "n_observations": pd.NA,
                "source": "given",
            }
            if extra:
                row["region_area"] = np.nan
            rows.append(row)
        frame = pd.DataFrame(rows)
        if self.units:
            frame["n_points"] = frame["n_points"].astype("Int64")
            frame["n_observations"] = frame["n_observations"].astype("Int64")
        return frame

    def sar_data(self, min_observations: int = 1) -> pd.DataFrame:
        """The species-area data sars fits.

        Parameters
        ----------
        min_observations : int
            Leave out regions with fewer observations than this. With the
            default, regions without any are left out: an empty region in
            observation data is more often unvisited than without species.
            0 keeps them, as islands with no species. Units given by hand
            are always kept.

        Returns
        -------
        pd.DataFrame
            ``area`` (km²) and ``species``, one row per region kept, without
            regions of no area.
        """
        frame = self.to_frame()
        enough = (frame["n_observations"] >= min_observations).fillna(True).astype(bool)
        keep = (enough | (frame["source"] == "given")) & (frame["area"] > 0)
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


def _areas(areas: Any) -> dict[str, float]:
    """Areas given for regions, checked: positive km² by name."""
    if areas is None:
        return {}
    if not isinstance(areas, Mapping):
        raise ValueError(f"areas must map region names to km², got {type(areas).__name__}.")
    out = {}
    for name, value in areas.items():
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise ValueError(f"The area of {name!r} must be a number of km², got {value!r}.")
        km2 = float(value)
        if not (np.isfinite(km2) and km2 > 0):
            raise ValueError(f"The area of {name!r} must be positive, got {value!r}.")
        out[str(name)] = km2
    return out


def _units(units: Any) -> dict[str, tuple[float, int]]:
    """Units given by hand, as name to (area in km², species): from a mapping
    of name to a pair or to a dict with ``area`` and ``species``, or a
    DataFrame with ``region`` (or its index), ``area`` and ``species``."""
    if units is None:
        return {}
    if isinstance(units, pd.DataFrame):
        missing = [c for c in ("area", "species") if c not in units.columns]
        if missing:
            raise ValueError(f"A table of units needs columns area and species; no {missing}.")
        names = units["region"] if "region" in units.columns else units.index
        units = {
            str(name): (area, count)
            for name, area, count in zip(names, units["area"], units["species"], strict=True)
        }
        if len(units) < len(names):
            raise ValueError("Unit names must differ.")
    if not isinstance(units, Mapping):
        raise ValueError(
            "units must map each name to (area in km², species), or be a DataFrame with "
            f"region, area and species; got {type(units).__name__}."
        )
    out: dict[str, tuple[float, int]] = {}
    for name, value in units.items():
        if isinstance(value, Mapping):
            if not {"area", "species"} <= set(value):
                raise ValueError(f"Unit {name!r} needs an area and a species count.")
            area, count = value["area"], value["species"]
        else:
            try:
                area, count = value
            except (TypeError, ValueError):
                raise ValueError(
                    f"Unit {name!r} must be (area in km², species), got {value!r}."
                ) from None
        if isinstance(count, list | tuple | set | frozenset):  # a list of species names
            count = len(set(count))
        for what, number in (("area", area), ("species count", count)):
            if isinstance(number, bool) or not isinstance(number, numbers.Real):
                raise ValueError(f"The {what} of unit {name!r} must be a number, got {number!r}.")
        if not (np.isfinite(area) and area > 0):
            raise ValueError(f"The area of unit {name!r} must be positive, got {area!r}.")
        if not (np.isfinite(count) and count >= 0 and float(count).is_integer()):
            raise ValueError(
                f"The species count of unit {name!r} must be a whole number, got {count!r}."
            )
        out[str(name)] = (float(area), int(count))
    return out


def islands(
    sample: Sample | None = None,
    regions: Any = None,
    names: Sequence[str] | str | None = None,
    area: Literal["region", "hull"] = "region",
    mask: Any = None,
    *,
    areas: Mapping[str, float] | None = None,
    units: Any = None,
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

    A design can be put together by hand from several sources. Regions are
    a plain mapping of name to polygon, so mappings from different places
    combine (``{**counties, **parks}``); ``areas`` replaces a region's
    measured area with one you know (a park's official area); and ``units``
    adds rows you give outright, a name with its area and species count,
    with no boundary or observations behind them. The result's
    [`add`][samplesar.Islands.add], [`select`][samplesar.Islands.select]
    and [`drop`][samplesar.Islands.drop] go on from there.

    Parameters
    ----------
    sample : Sample, optional
        The observations. Those outside every region are ignored; one in
        two overlapping regions counts in both. Not needed for a design of
        ``units`` only.
    regions : mapping, sequence, GeoSeries or GeoDataFrame, optional
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
    areas : mapping of str to float, optional
        An area in km² to use for a region instead of measuring it, by the
        region's name. Its species are still counted from the observations.
    units : mapping or DataFrame, optional
        Units given by hand: each name with its area in km² and its number
        of species, as ``{"Mammoth Cave": (214, 12)}``, as
        ``{"Mammoth Cave": {"area": 214, "species": 12}}``, or as a
        DataFrame with ``region`` (or its index), ``area`` and ``species``
        columns. A list of species names in place of the count is counted.

    Returns
    -------
    Islands

    Raises
    ------
    ValueError
        If neither regions nor units are given, regions are given without a
        sample, a name is used twice, ``areas`` names no region, an area
        isn't positive or a species count isn't a whole number;
        ``regions`` isn't polygons, is empty, has repeated names or not
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

    A design put together by hand:

    >>> tennessee = samplesar.regions.admin("USA", 1, "Tennessee")  # doctest: +SKIP
    >>> counties = samplesar.regions.admin_units(  # doctest: +SKIP
    ...     "USA", 2, names=["Sevier", "Blount", "Cocke"], inside=tennessee
    ... )
    >>> parks = samplesar.regions.read_units("parks.gpkg", "UNIT_NAME", ["Shenandoah"])  # doctest: +SKIP
    >>> islands = samplesar.designs.islands(  # doctest: +SKIP
    ...     sample,
    ...     {**counties, **parks},
    ...     areas={"Shenandoah": 806},            # its official area
    ...     units={"Mammoth Cave": (214, 12)},    # a row typed in
    ... )
    """
    if mask is not None and area == "hull":
        raise ValueError("mask applies to area='region'; a hull is already cut to its region.")
    named = {} if regions is None else _named(regions, names)
    ground = None if mask is None else shapely.make_valid(as_geometry(mask))
    return Islands(
        sample=sample if named else None,
        regions=named,
        area=area,
        mask=ground,
        areas=_areas(areas),
        units=_units(units),
    )
