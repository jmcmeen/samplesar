"""Nested samples: a sequence of ever-smaller subsets of one sample."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shapely
from pyproj import CRS
from pyproj.exceptions import CRSError
from shapely.errors import GEOSException

from samplesar._registry import Options, register
from samplesar._sample import Sample, _validate, point_species


@dataclass(frozen=True)
class Nest:
    """A nested sequence of sub-samples of one [`Sample`][samplesar.Sample].

    Made by a nesting strategy such as [`strategies.random`][samplesar.strategies.random]. Strategies differ only
    in which points they drop at each step.

    Attributes
    ----------
    sample : Sample
        The full sample.
    steps : tuple of np.ndarray
        ``steps[i]`` holds the rows of ``sample.points`` that remain at step
        ``i``. ``steps[0]`` is every point and each later step is a subset of
        the one before.
    strategy : str
        The registered strategy that made the nest, such as ``"hull_area"``;
        empty for a nest built directly.
    options : mapping
        The options the strategy was called with, as given (defaults left
        out), read-only.
    parts : np.ndarray, optional
        For a nest of several patches (see
        [`strategies.split`][samplesar.strategies.split]): the part each of
        the sample's points belongs to, one whole number per row of
        ``sample.points``. Each step's area is then the area covered by its
        parts' convex hulls together, not one hull around them all.
    """

    sample: Sample
    steps: tuple[np.ndarray, ...]
    strategy: str = ""
    options: Mapping[str, Any] = field(default_factory=Options)
    parts: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.parts is None:
            return
        parts = np.asarray(self.parts)
        if parts.shape != (self.sample.n_points,) or not np.issubdtype(parts.dtype, np.integer):
            raise ValueError(
                f"parts must hold one whole number per point ({self.sample.n_points}), got "
                f"shape {parts.shape} of {parts.dtype}."
            )
        parts = parts.astype(int)
        parts.flags.writeable = False
        object.__setattr__(self, "parts", parts)

    def groups(self, step: int) -> list[np.ndarray]:
        """A step's points, part by part: the rows of ``sample.points`` in
        each part that still has any, in order of part. One group, the whole
        step, for a nest without parts."""
        idx = np.asarray(self.steps[step], dtype=int)
        if self.parts is None:
            return [idx]
        labels = self.parts[idx]
        return [idx[labels == part] for part in np.unique(labels)]

    def footprint(self, step: int) -> shapely.Geometry:
        """The ground a step covers, on the sample's projection in km: its
        convex hull, or for a nest with parts the union of each part's hull.
        Not cut to the sample's mask. A point or a line where the points
        make no polygon."""
        xy = self.sample.xy
        hulls = [shapely.MultiPoint(xy[group]).convex_hull for group in self.groups(step)]
        if not hulls:  # a step with no points
            return shapely.GeometryCollection()
        if len(hulls) == 1:
            return hulls[0]
        polygons = [hull for hull in hulls if isinstance(hull, shapely.Polygon)]
        return shapely.union_all(polygons) if polygons else hulls[0]

    @property
    def label(self) -> str:
        """The strategy and its options, such as ``hull_area(keep='smallest')``.

        ``"nest"`` for a nest built directly rather than by a strategy.
        """
        name = self.strategy or "nest"
        if not self.options:
            return name
        return f"{name}({', '.join(f'{k}={_short(v)}' for k, v in self.options.items())})"

    @property
    def n_steps(self) -> int:
        """The number of steps, including the full sample."""
        return len(self.steps)

    def sample_at(self, step: int) -> Sample:
        """The observations remaining at a step, as a [`Sample`][samplesar.Sample]."""
        return self.sample.subset(self.steps[step])

    def areas_at(self, step: int) -> tuple[float, float]:
        """A step's area within the sample's mask and its whole area, both
        in km² (the same without a mask): of its convex hull, or for a nest
        with parts of its parts' hulls together."""
        if self.parts is None:
            return self.sample.areas_of(self.steps[step])
        return _cover_areas(self.sample, self.footprint(step))

    def to_frame(self) -> pd.DataFrame:
        """The nest as a table.

        Returns
        -------
        pd.DataFrame
            One row per step: ``step``, ``n_points``, ``area`` (convex hull,
            km², within the sample's mask (see [`with_mask`][samplesar.Sample.with_mask]) if it
            has one) and ``species`` (distinct species remaining). With a
            mask, also ``hull_area``: the whole hull's area, km². For a
            nest with ``parts``, the areas are of the parts' hulls together
            (overlaps counted once), and ``n_parts`` counts the parts with
            points left.
        """
        sample = self.sample
        masked = sample.mask is not None
        species, n_species = point_species(self.sample)
        # How many of the step's points hold each species, and the step's
        # hull, updated from the points each step drops (or adds) rather than
        # recomputed from every point at every step.
        holders = np.zeros(n_species, dtype=int)
        present = np.zeros(sample.n_points, dtype=bool)
        areas: _StepAreas | _PartAreas = (
            _StepAreas(sample) if self.parts is None else _PartAreas(sample, self.parts)
        )
        rows = []
        for i, idx in enumerate(self.steps):
            idx = np.asarray(idx, dtype=int)
            now = np.zeros(sample.n_points, dtype=bool)
            now[idx] = True
            gone, added = present & ~now, now & ~present
            present = now
            for change, sign in ((np.flatnonzero(gone), -1), (np.flatnonzero(added), 1)):
                if len(change):
                    np.add.at(holders, np.concatenate([species[p] for p in change]), sign)
            area, whole = areas.update(idx, gone, added)
            row = {
                "step": i,
                "n_points": len(idx),
                "area": area,
                "species": int(np.count_nonzero(holders)),
            }
            if masked:
                row["hull_area"] = whole
            if self.parts is not None:
                row["n_parts"] = len(np.unique(self.parts[idx]))
            rows.append(row)
        return pd.DataFrame(rows)

    def sar_data(self) -> pd.DataFrame:
        """The species-area data sars fits.

        Returns
        -------
        pd.DataFrame
            ``area`` (convex hull, km², within the sample's mask if it has
            one) and ``species`` columns,
            one row per step, without the zero-area steps at the end.
        """
        frame = self.to_frame()
        return frame.loc[frame["area"] > 0, ["area", "species"]].reset_index(drop=True)

    def to_dict(self) -> dict[str, Any]:
        """The nest as plain data, ready for JSON.

        Holds the observations (with their elevations and dates, if any, and
        their center, crs and mask), the strategy and its options, and the steps,
        stored as the points each step removes rather than in full. Options
        that are tuples or arrays are stored as lists.

        Returns
        -------
        dict

        Raises
        ------
        ValueError
            If the steps aren't nested (only a nest built by hand can be
            like that), or an option isn't a plain value JSON can hold (a
            ``Generator`` seed, say).
        """
        removed = []
        for before, after in zip(self.steps, self.steps[1:], strict=False):
            if not np.isin(after, before).all():
                raise ValueError("Only a nest whose steps are nested can be saved.")
            removed.append(np.setdiff1d(before, after).tolist())
        data = self.sample.data
        mask = self.sample.mask
        saved = {
            "format": FORMAT,
            # A nest without a mask or parts is saved as version 3, and one
            # without parts as 4, which older releases still read.
            "version": FORMAT_VERSION if self.parts is not None else 4 if mask is not None else 3,
            "samplesar": _package_version(),
            "strategy": self.strategy,
            "options": {k: _plain(k, v) for k, v in self.options.items()},
            "observations": _observations(data),
            "center": list(self.sample.center) if self.sample.center is not None else None,
            "crs": self.sample.crs.to_wkt() if self.sample.crs is not None else None,
            "first": np.asarray(self.steps[0]).tolist(),
            "removed": removed,
        }
        if mask is not None:
            saved["mask"] = shapely.to_wkt(mask, rounding_precision=-1)
        if self.parts is not None:
            saved["parts"] = self.parts.tolist()
        return saved

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Nest:
        """Rebuild a nest from [`to_dict`][samplesar.Nest.to_dict]'s output.

        Parameters
        ----------
        data : mapping
            As written by ``to_dict`` (or read back from its JSON).

        Returns
        -------
        Nest
            Each step's points in the order of the first step.

        Raises
        ------
        ValueError
            If the data isn't a saved nest, is from a newer format version,
            or its steps don't fit its sample.
        """
        missing = {"format", "version", "observations", "first", "removed"} - set(data)
        if data.get("format") != FORMAT or missing:
            raise ValueError("Not a saved samplesar nest.")
        version = data["version"]
        if not isinstance(version, int) or isinstance(version, bool):
            raise ValueError(f"The saved format version must be a whole number, got {version!r}.")
        if version > FORMAT_VERSION:
            raise ValueError(
                f"Saved with nest format {version}; this samplesar reads up to "
                f"{FORMAT_VERSION}. Upgrade samplesar to load it."
            )
        sample = Sample(_validate(pd.DataFrame(data["observations"])))
        center = data.get("center")
        if center is None and version < 3:
            center = data.get("centre")  # its key before version 3
        crs = data.get("crs")
        mask = data.get("mask")
        if center is not None or crs is not None or mask is not None:
            try:
                sample = dataclasses.replace(
                    sample,
                    center=tuple(center) if center is not None else None,
                    crs=CRS.from_wkt(crs) if crs is not None else None,
                    mask=shapely.from_wkt(mask) if mask is not None else None,
                )
            except (TypeError, ValueError, CRSError, GEOSException) as err:
                raise ValueError(f"The saved center, crs or mask is unusable: {err}") from err
        remaining = _indices(data["first"])
        if remaining.size and (remaining.min() < 0 or remaining.max() >= sample.n_points):
            raise ValueError("The saved steps name points outside the saved sample.")
        if len(np.unique(remaining)) < len(remaining):
            raise ValueError("The saved first step repeats a point.")
        steps = [remaining]
        for gone in data["removed"]:
            gone = _indices(gone)
            if not np.isin(gone, remaining).all():
                raise ValueError("A saved step removes a point that isn't there.")
            remaining = remaining[~np.isin(remaining, gone)]
            steps.append(remaining)
        options = dict(data.get("options", {}))
        if version < 3 and "centre" in options:  # radial's keyword before version 3
            options["center"] = options.pop("centre")
        parts = data.get("parts")
        if parts is not None:
            try:
                parts = _indices(parts)
            except ValueError:
                raise ValueError("The saved parts must be whole numbers, one per point.") from None
            if len(parts) != sample.n_points:
                raise ValueError("The saved parts don't match the saved sample's points.")
        return cls(
            sample=sample,
            steps=tuple(steps),
            strategy=data.get("strategy", ""),
            options=Options(options),
            parts=parts,
        )

    def to_json(self, path: str | Path) -> None:
        """Save the nest to a JSON file (see [`to_dict`][samplesar.Nest.to_dict]).

        Parameters
        ----------
        path : str or Path
            The file to write.
        """
        Path(path).write_text(json.dumps(self.to_dict()))

    @classmethod
    def from_json(cls, path: str | Path) -> Nest:
        """Load a nest saved with [`to_json`][samplesar.Nest.to_json].

        Parameters
        ----------
        path : str or Path
            The file to read.

        Returns
        -------
        Nest

        Raises
        ------
        ValueError
            As [`from_dict`][samplesar.Nest.from_dict].
        """
        return cls.from_dict(json.loads(Path(path).read_text()))

    def fit(self, model: str = "power", **kwargs: Any) -> Any:
        """Fit one of sars' models to the nest's species-area data.

        The areas are convex hull areas in km² (within the sample's mask,
        if it has one), so a parameter that depends on the unit of area is
        for km²: ``c`` of the power model is the species count at 1 km².

        Parameters
        ----------
        model : str
            A sars model: ``"power"`` calls ``sars.sar_power``, ``"loga"``
            ``sars.sar_loga``, and so on.
        **kwargs
            Passed to the sars function.

        Returns
        -------
        object
            sars' fit (``SARFit`` for a single model).

        Raises
        ------
        ValueError
            If sars has no such model, or the nest has fewer than three steps
            with a positive area to fit.
        """
        from samplesar._evaluate import fit_nest

        return fit_nest(self, model, **kwargs)


class _StepAreas:
    """Each step's hull areas (within the mask, and whole), in km², kept from
    the step before. The hull is reused when no vertex leaves and nothing is
    added; when vertices leave, it's rebuilt from the surviving vertices and
    the points outside their polygon, since every other point stays inside."""

    def __init__(self, sample: Sample) -> None:
        self.sample = sample
        self.xy = sample.xy
        self.vertices: np.ndarray | None = None  # point indices, in hull order
        self.areas = (0.0, 0.0)

    def update(self, idx: np.ndarray, gone: np.ndarray, added: np.ndarray) -> tuple[float, float]:
        if self.vertices is not None and not added.any():
            if not gone[self.vertices].any():
                return self.areas
            candidates = self._outside(idx, self.vertices[~gone[self.vertices]])
        else:
            candidates = idx
        self.vertices = _hull_vertices(self.xy, candidates)
        if self.vertices is None:  # fewer than three points, or all in a line
            self.areas = self.sample.areas_of(idx)
        else:
            hull = shapely.Polygon(self.xy[self.vertices])
            scale = self.sample.area_scale
            whole = hull.area / scale
            masked = (
                whole
                if self.sample.mask is None
                else self.sample._mask_tiles.area_within(hull) / scale
            )
            self.areas = (masked, whole)
        return self.areas

    def _outside(self, idx: np.ndarray, kept: np.ndarray) -> np.ndarray:
        """The surviving vertices and the points not strictly inside their polygon."""
        if len(kept) < 3:
            return idx
        polygon = shapely.Polygon(self.xy[kept])
        if not polygon.area > 0:
            return idx
        shapely.prepare(polygon)
        inside = shapely.contains_xy(polygon, self.xy[idx, 0], self.xy[idx, 1])
        return np.union1d(idx[~inside], kept)


def _cover_areas(sample: Sample, cover: shapely.Geometry) -> tuple[float, float]:
    """The area of some ground (on the sample's projection, km) within the
    sample's mask, and its whole area, in km²."""
    if not cover.area > 0:
        return 0.0, 0.0
    whole = float(cover.area) / sample.area_scale
    if sample.mask is None:
        return whole, whole
    return sample._mask_tiles.area_within(cover) / sample.area_scale, whole


class _PartAreas:
    """Each step's areas for a nest of several parts, in km²: the union of
    the parts' convex hulls, within the mask and whole. A part's hull is
    rebuilt only when it loses a vertex or gains a point, and the union only
    when some hull changed."""

    def __init__(self, sample: Sample, parts: np.ndarray) -> None:
        self.sample = sample
        self.xy = sample.xy
        self.parts = parts
        self.vertices: dict[int, np.ndarray | None] = {}
        self.hulls: dict[int, shapely.Polygon] = {}
        self.areas = (0.0, 0.0)

    def update(self, idx: np.ndarray, gone: np.ndarray, added: np.ndarray) -> tuple[float, float]:
        labels = self.parts[idx]
        changed = False
        for part in np.unique(self.parts[gone | added]):
            part = int(part)
            vertices = self.vertices.get(part)
            if vertices is not None and not added[self.parts == part].any():
                if not gone[vertices].any():
                    continue
            members = idx[labels == part]
            vertices = _hull_vertices(self.xy, members)
            self.vertices[part] = vertices
            if vertices is not None:
                self.hulls[part] = shapely.Polygon(self.xy[vertices])
            else:
                # Too few points, a line, or a sliver qhull won't take: GEOS
                # decides, as footprint does.
                hull = shapely.MultiPoint(self.xy[members]).convex_hull
                if isinstance(hull, shapely.Polygon) and hull.area > 0:
                    self.hulls[part] = hull
                else:
                    self.hulls.pop(part, None)
            changed = True
        if changed:
            hulls = list(self.hulls.values())
            cover = (
                shapely.GeometryCollection()
                if not hulls
                else hulls[0]
                if len(hulls) == 1
                else shapely.union_all(hulls)
            )
            self.areas = _cover_areas(self.sample, cover)
        return self.areas


def _hull_vertices(xy: np.ndarray, idx: np.ndarray) -> np.ndarray | None:
    """The convex hull's vertices, as indices into ``xy`` in hull order, or
    None when the points make no polygon (or qhull can't tell).

    qhull's precision rules differ from GEOS's (``areas_of``) only at rounding
    level: a corner within rounding of a straight edge may count as a vertex
    in one and not the other, which changes an area by no more than rounding.
    """
    if len(idx) < 3:
        return None
    from scipy.spatial import ConvexHull, QhullError

    try:
        return idx[ConvexHull(xy[idx]).vertices]
    except QhullError:
        return None


def _observations(data: pd.DataFrame) -> dict[str, list[Any]]:
    """The observations as plain lists: missing elevations and dates as None,
    dates as ISO text."""
    saved: dict[str, list[Any]] = {
        "species": data["species"].tolist(),
        "lon": data["lon"].tolist(),
        "lat": data["lat"].tolist(),
    }
    if "elevation" in data.columns:
        saved["elevation"] = [None if np.isnan(v) else float(v) for v in data["elevation"]]
    if "date" in data.columns:
        saved["date"] = [None if pd.isna(v) else v.isoformat() for v in data["date"]]
    return saved


def _short(value: Any) -> str:
    """An option value for a label: numpy scalars as plain numbers, and long
    sequences (a removal order, say) abbreviated with their length."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, shapely.Geometry):
        return f"<{value.geom_type}>"
    if hasattr(value, "to_numpy"):  # a pandas or geopandas column or table
        return f"<{type(value).__name__} ({len(value)})>"
    if isinstance(value, list | tuple) and any(isinstance(v, shapely.Geometry) for v in value):
        kinds = sorted({v.geom_type for v in value if isinstance(v, shapely.Geometry)})
        return f"<{len(value)} {'/'.join(kinds)}>"
    if isinstance(value, np.ndarray | list | tuple) and len(value) > 6:
        head = ", ".join(
            repr(v.item() if isinstance(v, np.generic) else v) for v in list(value)[:3]
        )
        return f"[{head}, ... ({len(value)})]"
    return repr(value)


class NoStepError(ValueError):
    """A strategy that removes whole layers couldn't remove even one."""


def check_size(sample: Sample, min_points: int) -> None:
    """Raise ValueError unless ``1 <= min_points <= sample.n_points``."""
    if min_points < 1:
        raise ValueError(f"min_points must be at least 1, got {min_points}")
    if sample.n_points < min_points:
        raise ValueError(f"Sample has {sample.n_points} points; need at least {min_points}")


FORMAT = "samplesar.nest"
# 2 adds the sample's center and crs; 3 spells the key "center"; 4 adds the mask;
# 5 adds parts
FORMAT_VERSION = 5


def _package_version() -> str:
    from samplesar import __version__

    return __version__


def _indices(values: Any) -> np.ndarray:
    """Saved point indices as an int array; anything else isn't a saved nest."""
    array = np.asarray(values)
    if array.size == 0:
        return np.empty(0, dtype=int)
    if array.ndim != 1 or not np.issubdtype(array.dtype, np.integer):
        raise ValueError("The saved steps must be lists of whole-number point indices.")
    return array.astype(int)


def _plain(name: str, value: Any) -> Any:
    """An option value as plain JSON data (tuples and arrays become lists),
    or a ValueError naming the option if it can't be stored."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, shapely.Geometry):
        return shapely.to_wkt(value, rounding_precision=-1)
    if hasattr(value, "to_crs") and hasattr(value, "crs"):  # a GeoSeries or GeoDataFrame
        frame = value if value.crs is None else value.to_crs(4326)
        return [_plain(name, g) for g in getattr(frame, "geometry", frame)]
    if isinstance(value, pd.Series | pd.Index):
        return [_plain(name, v) for v in value.tolist()]
    if isinstance(value, np.ndarray | tuple | list):
        return [_plain(name, v) for v in value]
    if isinstance(value, Mapping):
        return {str(k): _plain(name, v) for k, v in value.items()}
    if value is None or isinstance(value, bool | int | float | str):
        return value
    raise ValueError(
        f"Option {name}={value!r} can't be saved as JSON; pass a plain value (a seed as an "
        "int, not a Generator) to save the nest."
    )


@register("random")
def random(sample: Sample, min_points: int = 3, seed: int | None = None) -> Nest:
    """Nest a sample by removing one point at a time, uniformly at random.

    The baseline every other strategy is compared against.

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
    check_size(sample, min_points)
    n = sample.n_points
    rng = np.random.default_rng(seed)
    remaining = np.arange(n)
    steps = [remaining.copy()]
    while len(remaining) > min_points:
        drop = rng.integers(len(remaining))
        remaining = np.delete(remaining, drop)
        steps.append(remaining.copy())
    return Nest(sample=sample, steps=tuple(steps))


@register("order")
def from_order(sample: Sample, order: Sequence[int] | np.ndarray, min_points: int = 3) -> Nest:
    """Nest a sample by removing its points in a given order.

    Any ordering, from a heuristic, an optimizer or an experiment's design,
    becomes a [`Nest`][samplesar.Nest] that the rest of the package can
    evaluate. One point goes per step, in ``order``, until ``min_points``
    remain or the order runs out; the order needn't name every point.

    Parameters
    ----------
    sample : Sample
        The sample to nest.
    order : sequence of int or np.ndarray
        Rows of ``sample.points``, in the order to remove them. Floats are
        accepted if they're whole numbers.
    min_points : int
        Stop when this many points remain, even if the order goes on.

    Returns
    -------
    Nest
        Registered as ``"order"``, so ``samplesar.strategies.run(sample, "order",
        order=...)`` is the same.

    Raises
    ------
    ValueError
        If ``order`` repeats a point or names one outside the sample, isn't
        a sequence of whole numbers, or ``min_points`` is below 1 or above the
        sample's size.
    """
    check_size(sample, min_points)
    order = np.asarray(order)
    if order.ndim != 1:
        raise ValueError("order must be a one-dimensional sequence of point indices.")
    if order.size and not np.issubdtype(order.dtype, np.integer):
        if not np.issubdtype(order.dtype, np.floating) or not np.all(order == np.round(order)):
            raise ValueError("order must hold whole numbers: indices of the sample's points.")
        order = order.astype(int)
    n = sample.n_points
    if order.size and (order.min() < 0 or order.max() >= n):
        raise ValueError(f"order names a point outside the sample's {n} points.")
    if len(np.unique(order)) < len(order):
        raise ValueError("order repeats a point.")
    remaining = np.arange(n)
    steps = [remaining.copy()]
    for point in order[: n - min_points]:
        remaining = remaining[remaining != point]
        steps.append(remaining.copy())
    return Nest(sample=sample, steps=tuple(steps))
