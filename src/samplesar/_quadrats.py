"""Equal-area patches as sampling units: squares of a given side, in a grid
or scattered, with the species counted in each."""

from __future__ import annotations

import numbers
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

import numpy as np
import pandas as pd
import shapely
from pyproj import Transformer
from scipy.special import gammaln

from samplesar._boundary import as_geometry, polygonal
from samplesar._islands import Islands, geodesic_area
from samplesar._sample import WGS84, Sample, point_species

MAX_PATCHES = 20_000  # each is a polygon measured on the ellipsoid
EDGE_PIECES = 16  # a patch's straight sides curve in longitude and latitude
HALF_TURN = 180.0


def _outlines(sample: Sample, squares: np.ndarray) -> list[shapely.Geometry]:
    """Squares on the sample's projection, rows of (side, x0, y0) in km, as
    WGS84 polygons."""
    projection = sample.projection
    km = projection.axis_info[0].unit_conversion_factor / 1000.0
    back = Transformer.from_crs(projection, WGS84, always_xy=True)
    out = []
    for side, x0, y0 in squares:
        square = shapely.segmentize(shapely.box(x0, y0, x0 + side, y0 + side), side / EDGE_PIECES)
        outline = shapely.transform(
            square, lambda c: np.column_stack(back.transform(c[:, 0] / km, c[:, 1] / km))
        )
        west, _, east, _ = outline.bounds
        if not np.all(np.isfinite(outline.bounds)) or east - west > HALF_TURN:
            raise ValueError(
                f"A {side:g} km patch (the grid reaches up to half a patch beyond the sample) "
                "crosses the antimeridian or covers a pole, where its outline can't be drawn "
                "in longitude and latitude. Use smaller patches."
            )
        out.append(outline)
    return out


@dataclass(frozen=True, eq=False)
class Quadrats(Islands):
    """A sample divided among square patches of equal size.

    Made by [`designs.quadrats`][samplesar.designs.quadrats]. It is an
    [`Islands`][samplesar.Islands] design whose regions are the patches, so
    its table, its species-area data, its fit and its map work the same
    way; ``select`` and ``drop`` keep or remove patches by name.

    Attributes
    ----------
    sample : Sample
        The observations.
    regions : mapping of str to polygon
        Each patch's name and outline, in WGS84 longitude and latitude.
    mask : polygon, optional
        Ground the patches' areas are measured within, such as land.
    layout : {"grid", "random"}
        How the patches were laid out.
    squares : mapping of str to (float, float, float)
        Each patch's side and the x and y of its lower left corner, in km
        on the sample's projection.
    """

    layout: str = "grid"
    squares: Mapping[str, tuple[float, float, float]] = field(default_factory=dict)
    cells: Mapping[str, tuple[int, int]] = field(default_factory=dict)
    origins: Mapping[float, tuple[float, float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.units or self.known_areas or self.area != "region":
            raise ValueError(
                "A quadrat design holds only patches measured as they are: no units given by "
                "hand, known areas or hull areas."
            )
        missing = [name for name in self.regions if name not in self.squares]
        if missing:
            raise ValueError(f"No square given for patch {missing[0]!r}.")

    @property
    def sizes(self) -> list[float]:
        """The patch sides in the design, in km, smallest first."""
        return sorted({self.squares[name][0] for name in self.regions})

    @property
    def label(self) -> str:
        """``quadrats (n patches, k sizes)``, for plots and tables."""
        count, sizes = len(self.regions), len(self.sizes)
        return (
            f"quadrats ({count} patch{'es' if count != 1 else ''}, "
            f"{sizes} size{'s' if sizes != 1 else ''})"
        )

    def add(self, *args: Any, **kwargs: Any) -> Islands:
        """Not available: a quadrat design holds only its own patches.

        Raises
        ------
        ValueError
            Always. To mix patches with other regions, give
            [`designs.islands`][samplesar.designs.islands] this design's
            ``regions`` along with the others.
        """
        raise ValueError(
            "A quadrat design holds only its own patches. To mix them with other regions, "
            "give designs.islands this design's regions along with the others."
        )

    @cached_property
    def _members(self) -> dict[str, np.ndarray]:
        """The rows of ``sample.points`` in each patch. Patches are half
        open (their lower and left edges belong to them, their upper and
        right edges to the next patch), so a point on a shared edge of a
        grid is in exactly one patch."""
        assert self.sample is not None
        xy = self.sample.xy
        members: dict[str, np.ndarray] = {}
        by_size: dict[float, list[str]] = {}
        for name in self.regions:
            by_size.setdefault(self.squares[name][0], []).append(name)
        for side, names in by_size.items():
            if side in self.origins:
                x0, y0 = self.origins[side]
                col = np.floor((xy[:, 0] - x0) / side).astype(np.int64)
                row = np.floor((xy[:, 1] - y0) / side).astype(np.int64)
                span = int(row.max() - row.min()) + 2 if len(row) else 1
                key = col * span + (row - (row.min() if len(row) else 0))
                order = np.argsort(key, kind="stable")
                ordered = key[order]
                low = int(row.min()) if len(row) else 0
                for name in names:
                    c, r = self.cells[name]
                    if not 0 <= r - low < span:
                        members[name] = np.empty(0, dtype=np.int64)
                        continue
                    wanted = c * span + (r - low)
                    start, stop = np.searchsorted(ordered, [wanted, wanted + 1])
                    members[name] = np.sort(order[start:stop])
            else:
                for name in names:
                    _, x0, y0 = self.squares[name]
                    inside = (
                        (xy[:, 0] >= x0)
                        & (xy[:, 0] < x0 + side)
                        & (xy[:, 1] >= y0)
                        & (xy[:, 1] < y0 + side)
                    )
                    members[name] = np.flatnonzero(inside)
        return {name: members[name] for name in self.regions}

    def to_frame(self) -> pd.DataFrame:
        """One row per patch.

        Returns
        -------
        pd.DataFrame
            ``region`` (the patch's name), ``size_km`` (its side), ``area``
            (km²), ``species``, ``n_points``, ``n_observations`` and
            ``source``, as [`Islands.to_frame`][samplesar.Islands.to_frame]
            gives them. With a mask, ``area`` is the part of the patch
            within it and ``region_area`` the whole patch.
        """
        return self._frame.copy()

    @cached_property
    def _frame(self) -> pd.DataFrame:
        frame = super()._frame
        frame.insert(1, "size_km", [self.squares[name][0] for name in frame["region"]])
        return frame

    def _sizes_with_data(self, min_observations: int = 1) -> int:
        """How many patch sizes have a row in the species-area data."""
        frame = self._frame
        keep = (frame["n_observations"] >= min_observations) & (frame["area"] > 0)
        return int(frame.loc[keep, "size_km"].nunique())

    def fit(self, model: str = "power", min_observations: int = 1, **kwargs: Any) -> Any:
        """Fit one of sars' models to the patches' species-area data, one
        row per patch, as [`Islands.fit`][samplesar.Islands.fit] does.

        Raises
        ------
        ValueError
            If the patches with data are of one size only (they give one
            point on the curve), sars has no such model, or fewer than
            three patches have data to fit.
        """
        if self._sizes_with_data(min_observations) < 2:  # noqa: PLR2004
            raise ValueError(
                "The patches with observations are all one size, which gives one point on a "
                "species-area curve; lay out several sizes to fit one."
            )
        return super().fit(model, min_observations, **kwargs)

    def accumulate(
        self,
        size_km: float | Sequence[float] | None = None,
        order: str = "any",
        min_observations: int = 1,
    ) -> pd.DataFrame:
        """The species-area curve from combining patches of one size.

        Patches of a size are one point on a curve when each is a row of
        its own. Taken together they give a whole curve: the species in
        one patch, in two, in three and so on, against their combined
        area. This is how curves are drawn from a grid of quadrats or from
        scattered ones, and it is a different curve from
        [`fit`][samplesar.Quadrats.fit]'s, which has one row per patch and
        needs several sizes.

        Parameters
        ----------
        size_km : float or sequence of float, optional
            The patch size to combine, or several; by default every size
            in the design, each on its own.
        order : {"any", "nearest"}
            Which patches are combined. ``"any"``: every combination of k
            patches, wherever they lie; the mean species count over all of
            them is computed exactly from the number of patches each
            species is in, without drawing any. ``"nearest"``: only
            neighbors. From each patch in turn the others are added
            nearest first (by the distance between centers; equally near
            patches in the design's order), and the curves from all the
            starting patches are averaged. Nearby patches usually share
            more species than distant ones, and then this curve lies below
            the other.
        min_observations : int
            Leave out patches with fewer observations than this, as
            [`sar_data`][samplesar.Islands.sar_data] does: by default the
            patches without any. Patches of no area are left out too, and
            a size left with no patches has no rows.

        Returns
        -------
        pd.DataFrame
            One row per size and number of patches: ``size_km``,
            ``patches`` (how many are combined), ``area`` (their mean
            combined area, km²) and ``species`` (their mean species
            count). The last row of a size has every patch, so its species
            are all those the patches hold.

        Raises
        ------
        ValueError
            If ``order`` is unknown, or no patch of any size asked for has
            enough observations.
        KeyError
            If the design has no patches of a size asked for.
        """
        if order not in ("any", "nearest"):
            raise ValueError(f"order must be 'any' or 'nearest', got {order!r}")
        known = self.sizes
        if size_km is None:
            wanted = known
        else:
            # Sizes are told apart by the six digits that name their patches,
            # so 0.1 + 0.2 finds the 0.3 km patches.
            by_label = {f"{side:g}": side for side in known}
            asked = _sizes(size_km)
            missing = [side for side in asked if f"{side:g}" not in by_label]
            wanted = [by_label[f"{side:g}"] for side in asked if f"{side:g}" in by_label]
            if missing:
                raise KeyError(
                    f"No patches of {', '.join(f'{side:g}' for side in missing)} km; the design "
                    f"has {', '.join(f'{side:g}' for side in known)} km."
                )
        assert self.sample is not None
        species, n_species = point_species(self.sample)
        frame = self._frame
        keep = (frame["n_observations"] >= min_observations) & (frame["area"] > 0)
        parts = []
        for side in wanted:
            rows = frame[keep & (frame["size_km"] == side)]
            if rows.empty:
                continue
            names = rows["region"].tolist()
            areas = rows["area"].to_numpy(dtype=float)
            count = len(names)
            # Which species each patch holds, as pairs (patch, species) sorted by species.
            patch: list[int] = []
            found: list[int] = []
            for i, name in enumerate(names):
                inside = self._members[name]
                if len(inside):
                    held = np.unique(np.concatenate([species[p] for p in inside])).tolist()
                    patch.extend([i] * len(held))
                    found.extend(held)
            patch_of = np.asarray(patch, dtype=np.int64)
            species_of = np.asarray(found, dtype=np.int64)
            by_species = np.argsort(species_of, kind="stable")
            patch_of, species_of = patch_of[by_species], species_of[by_species]
            k = np.arange(1, count + 1)
            richness: np.ndarray
            area: np.ndarray
            if order == "any":
                occupied = np.bincount(species_of, minlength=n_species)
                # Species in the same number of patches share a row.
                occupied, shared = np.unique(occupied[occupied > 0], return_counts=True)
                # A species in m of the N patches is missed by k of them
                # with probability C(N - m, k) / C(N, k).
                free = count - occupied[:, None]
                with np.errstate(invalid="ignore"):
                    log_missed = (
                        gammaln(free + 1)
                        - gammaln(free - k[None, :] + 1)
                        - gammaln(count + 1)
                        + gammaln(count - k[None, :] + 1)
                    )
                missed = np.where(free >= k[None, :], np.exp(log_missed), 0.0)
                richness = shared.sum() - shared @ missed
                area = k * areas.mean()
            else:
                # How far apart two patches are: in a grid, counted in whole
                # cells, so that equally near patches are exactly tied.
                places: np.ndarray
                if side in self.origins:
                    places = np.array([self.cells[n] for n in names], dtype=np.int64)
                else:
                    places = np.array([self.squares[n][1:] for n in names], dtype=float)
                starts = np.flatnonzero(np.diff(species_of, prepend=-1))
                seen = np.zeros(count)
                covered = np.zeros(count)
                rank = np.empty(count, dtype=np.int64)
                for start in range(count):
                    distance = ((places - places[start]) ** 2).sum(axis=1)
                    distance[start] = -1  # the starting patch comes first
                    nearest = np.argsort(distance, kind="stable")
                    rank[nearest] = np.arange(count)
                    # Each species enters with the first of its patches reached.
                    if len(starts):
                        first = np.minimum.reduceat(rank[patch_of], starts)
                        seen += np.bincount(first, minlength=count).cumsum()
                    covered += np.cumsum(areas[nearest])
                richness = seen / count
                area = covered / count
            parts.append(
                pd.DataFrame({"size_km": side, "patches": k, "area": area, "species": richness})
            )
        if not parts:
            raise ValueError(
                f"No patch of the sizes asked for has {min_observations} observation(s) or more."
            )
        return pd.concat(parts, ignore_index=True)


def _sizes(size_km: Any) -> list[float]:
    if isinstance(size_km, Mapping):
        raise ValueError("size_km must be a number of km or a sequence of them, got a mapping.")
    scalar = isinstance(size_km, str | bytes) or not isinstance(size_km, Iterable)
    if isinstance(size_km, np.ndarray) and size_km.ndim == 0:
        size_km = size_km.item()
    values = [size_km] if scalar or np.ndim(size_km) == 0 else list(size_km)
    if not values:
        raise ValueError("No patch size given.")
    out = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise ValueError(f"A patch size must be a number of km, got {value!r}.")
        side = float(value)
        if not (np.isfinite(side) and side > 0):
            raise ValueError(f"A patch size must be positive, got {value!r}.")
        out.append(side)
    labels = [f"{side:g}" for side in out]
    if len(set(labels)) < len(labels):
        raise ValueError(
            f"Patch sizes must differ in their first six digits, which name the patches; "
            f"got {', '.join(labels)}."
        )
    return out


def _grid(side: float, low: np.ndarray, high: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The cells of a grid centered on an extent, covering it with room to
    spare on every side: its origin, and each cell's (col, row)."""
    across = np.floor((high - low) / side) + 1.0
    total = float(across[0]) * float(across[1])
    if not total <= MAX_PATCHES:
        raise ValueError(
            f"A grid of {side:g} km patches over this sample has {total:.6g} "
            f"cells; the most is {MAX_PATCHES}. Use larger patches, or place some at random (n)."
        )
    counts = across.astype(np.int64)
    origin = low - (counts * side - (high - low)) / 2.0
    cols, rows = np.meshgrid(np.arange(counts[0]), np.arange(counts[1]), indexing="ij")
    return origin, np.column_stack([cols.ravel(), rows.ravel()])


def _fraction(outline: shapely.Geometry, ground: shapely.Geometry) -> float:
    whole = geodesic_area(outline)
    return geodesic_area(polygonal(outline.intersection(ground))) / whole if whole > 0 else 0.0


def _enough(fraction: float, min_fraction: float) -> bool:
    return fraction > 0 and fraction >= min_fraction


def quadrats(
    sample: Sample,
    size_km: float | Sequence[float],
    n: int | None = None,
    seed: int | None = None,
    mask: Any = None,
    min_fraction: float = 0.5,
) -> Quadrats:
    """Divide a sample among square patches of equal size.

    Where a nest shrinks one sample and
    [`designs.islands`][samplesar.designs.islands] counts species in
    regions you give, this lays out patches of a set size and counts the
    species in each, as quadrat surveys do. Patches of one size give one
    point on a species-area curve, so give several sizes to fit one; each
    size is laid out on its own, and patches of different sizes overlap.

    The patches are squares on the sample's projection, their sides along
    its axes. On the default projection, which is equal-area, patches of a
    size all have the same area; under a ``crs`` of the sample's own they
    differ as far as it distorts areas, and each is reported at its true
    area on the ellipsoid.

    Patches of a size still differ in how hard they were sampled, and the
    table gives each one's points and observations.
    [`Sample.thin`][samplesar.Sample.thin] evens effort out beforehand.

    Parameters
    ----------
    sample : Sample
        The observations.
    size_km : float or sequence of float
        The side of a patch, in km; or several, for patches of several
        sizes.
    n : int, optional
        By default the patches of a size form a grid over the sample's
        extent, centered on it, patch against patch. With ``n``, that many
        patches of each size are placed at random within the extent
        instead, none overlapping another of its size. Placed one after
        another at random, patches leave gaps too small for the next, so
        they can fill about half of the extent at most.
    seed : int, optional
        Seeds the random placement, so a design can be made again. A grid
        has nothing random in it and doesn't use it.
    mask : polygon, optional
        Ground the patches' areas are measured within (land, say): a
        shapely polygon in WGS84, or a GeoDataFrame or GeoSeries. The
        sample's own mask isn't used: it is kept only near the sample's
        points.
    min_fraction : float
        With a mask, leave out a patch with less than this share of its
        area within it, rather than count it as a small patch. A patch
        wholly outside the mask is always left out.

    Returns
    -------
    Quadrats
        Patches are named for their size and place: ``"10 km (2, 0)"`` is
        the grid cell in column 2 from the west, row 0 from the south, and
        ``"10 km #3"`` the third patch placed at random. A point belongs to
        one patch of each size at most; in a grid, one on a shared edge
        goes to the patch east or north of it.

    Raises
    ------
    ValueError
        If the sample has no points; a size isn't a positive number or two
        are the same; ``n`` isn't a positive whole number or ``seed`` isn't
        a whole number; a grid would
        have more than 20,000 cells or ``n`` is more than that; a patch
        placed at random is larger than the sample's extent, or ``n`` of
        them don't fit without overlapping; ``min_fraction`` isn't between
        0 and 1; the mask isn't polygonal or leaves no patch; or a patch
        crosses the antimeridian or covers a pole.

    Examples
    --------
    >>> patches = samplesar.designs.quadrats(sample, [5, 10, 20, 40])  # doctest: +SKIP
    >>> patches.to_frame()  # doctest: +SKIP
    >>> patches.fit("power")  # doctest: +SKIP

    Thirty patches of each size, scattered, on land only:

    >>> patches = samplesar.designs.quadrats(  # doctest: +SKIP
    ...     sample, [5, 10, 20], n=30, seed=1, mask=land
    ... )
    """
    if not isinstance(sample, Sample):
        raise ValueError(f"quadrats needs a Sample, got {type(sample).__name__}.")
    if sample.n_points == 0:
        raise ValueError("The sample has no points to lay patches over.")
    sides = _sizes(size_km)
    if n is not None:
        if isinstance(n, bool) or not isinstance(n, numbers.Integral) or n < 1:
            raise ValueError(f"n must be a positive whole number of patches, got {n!r}.")
        if n > MAX_PATCHES:
            raise ValueError(f"n is {n}; the most patches of a size is {MAX_PATCHES}.")
    if isinstance(min_fraction, bool) or not isinstance(min_fraction, numbers.Real):
        raise ValueError(f"min_fraction must be a number from 0 to 1, got {min_fraction!r}.")
    if not 0 <= min_fraction <= 1:
        raise ValueError(f"min_fraction must be from 0 to 1, got {min_fraction!r}.")
    ground = None if mask is None else shapely.make_valid(as_geometry(mask))
    if ground is not None:
        shapely.prepare(ground)
    try:
        rng = np.random.default_rng(seed)
    except (TypeError, ValueError) as err:
        raise ValueError(f"seed must be a whole number or None, got {seed!r}.") from err
    xy = sample.xy
    low, high = xy.min(axis=0), xy.max(axis=0)

    regions: dict[str, shapely.Geometry] = {}
    squares: dict[str, tuple[float, float, float]] = {}
    cells: dict[str, tuple[int, int]] = {}
    origins: dict[float, tuple[float, float]] = {}
    for side in sides:
        if n is None:
            origin, grid = _grid(side, low, high)
            corners = np.column_stack([np.full(len(grid), side), origin + grid * side])
            origins[side] = (float(origin[0]), float(origin[1]))
            for (col, row), square, outline in zip(
                grid, corners, _outlines(sample, corners), strict=True
            ):
                if ground is not None and not _enough(_fraction(outline, ground), min_fraction):
                    continue
                name = f"{side:g} km ({col}, {row})"
                regions[name] = outline
                squares[name] = (side, float(square[1]), float(square[2]))
                cells[name] = (int(col), int(row))
        else:
            room = high - low - side
            if np.any(room < 0):
                raise ValueError(
                    f"A {side:g} km patch is larger than the sample's extent "
                    f"({high[0] - low[0]:.3g} by {high[1] - low[1]:.3g} km), so it can't be "
                    "placed at random within it. Use smaller patches, or a grid (no n)."
                )
            placed: list[tuple[float, float]] = []
            for _ in range(200 * n + 1000):
                if len(placed) == n:
                    break
                x0, y0 = (low + rng.uniform(size=2) * room).tolist()
                if any(abs(x0 - x) < side and abs(y0 - y) < side for x, y in placed):
                    continue
                outline = _outlines(sample, np.array([[side, x0, y0]]))[0]
                if ground is not None and not _enough(_fraction(outline, ground), min_fraction):
                    continue
                placed.append((x0, y0))
                name = f"{side:g} km #{len(placed)}"
                regions[name] = outline
                squares[name] = (side, x0, y0)
            if len(placed) < n:
                where = "" if ground is None else " within the mask"
                raise ValueError(
                    f"Only {len(placed)} of {n} patches of {side:g} km fit{where} without "
                    "overlapping; patches placed at random fill about half of the sample's "
                    "extent at most. Ask for fewer or smaller patches, or use a grid (no n)."
                )
    if not regions:
        raise ValueError(
            "No patch has enough of its area within the mask; lower min_fraction, or check "
            "that the mask covers the sample."
        )
    return Quadrats(
        sample=sample,
        regions=regions,
        area="region",
        mask=ground,
        layout="grid" if n is None else "random",
        squares=squares,
        cells=cells,
        origins=origins,
    )
