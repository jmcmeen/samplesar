"""Elevations for a sample's points, read from a digital elevation model."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import numpy as np

from samplesar._regions import _fetch_to, cache_root
from samplesar._sample import WGS84, Sample

logger = logging.getLogger(__name__)

# Copernicus GLO-30: one-degree tiles of about 30 m cells, as cloud-optimized
# GeoTIFFs in a public bucket, named by their south-west corner.
GLO30 = "https://copernicus-dem-30m.s3.amazonaws.com/{name}/{name}.tif"


def glo30_tile(lon: float, lat: float) -> str:
    """The name of the Copernicus GLO-30 tile for a point: the one whose
    south-west corner is the whole degrees at or below it (the 180th
    meridian belongs to the tile west of it)."""
    south, west = min(math.floor(lat), 89), math.floor(lon)
    if west >= 180:
        west = 179
    ns = f"{'N' if south >= 0 else 'S'}{abs(south):02d}"
    ew = f"{'E' if west >= 0 else 'W'}{abs(west):03d}"
    return f"Copernicus_DSM_COG_10_{ns}_00_{ew}_00_DEM"


def _rasterio() -> Any:
    try:
        import rasterio
    except ImportError as err:
        raise ImportError(
            "Reading elevation models needs rasterio: pip install 'samplesar[elevation]'"
        ) from err
    return rasterio


WHOLE = 60_000_000  # cells: a raster up to this size is read whole, a larger one point by point


def _read(path: Path | str, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """The raster's first band at some (lon, lat) points, as floats: NaN at
    its nodata value and outside it."""
    rasterio = _rasterio()
    from pyproj import CRS, Transformer
    from rasterio.transform import rowcol

    with rasterio.open(path) as raster:
        if raster.crs is None:
            raise ValueError("the raster has no coordinate reference system")
        crs = CRS.from_wkt(raster.crs.to_wkt())
        x, y = np.asarray(lon, dtype=float), np.asarray(lat, dtype=float)
        if crs != WGS84:
            x, y = Transformer.from_crs(WGS84, crs, always_xy=True).transform(x, y)
        if raster.width * raster.height > WHOLE:
            read = raster.sample(list(zip(x, y, strict=True)), indexes=1, masked=True)
            return np.array([np.nan if v[0] is np.ma.masked else float(v[0]) for v in read])
        band = raster.read(1, masked=True)
        rows, cols = (np.asarray(v, dtype=int) for v in rowcol(raster.transform, x, y))
        inside = (rows >= 0) & (rows < raster.height) & (cols >= 0) & (cols < raster.width)
        out = np.full(len(x), np.nan)
        found = band[rows[inside], cols[inside]]
        out[inside] = np.ma.filled(found.astype(float), np.nan)
        return out


def _tile(name: str, cache: Path) -> Path | None:
    """A GLO-30 tile's local file, downloaded if need be, or None where
    there is no tile (open sea)."""
    path = cache / f"{name}.tif"
    sea = cache / f"{name}.none"  # remembered, so the sea isn't asked for again
    if sea.exists():
        return None
    if not path.exists():
        logger.info("Downloading elevation tile %s", name)
        try:
            _fetch_to(GLO30.format(name=name), path)
        except ValueError as err:
            if getattr(err.__cause__, "code", None) == 404:  # no such tile: open sea
                sea.touch()
                return None
            raise
    return path


def _from_tiles(tiles: np.ndarray, lon: np.ndarray, lat: np.ndarray, cache: Path) -> np.ndarray:
    heights = np.full(len(lon), np.nan)
    for name in sorted(set(tiles)):
        path = _tile(name, cache)
        if path is None:
            continue
        here = tiles == name
        try:
            heights[here] = _read(path, lon[here], lat[here])
        except Exception as err:
            path.unlink(missing_ok=True)  # a damaged download: fetched again next time
            raise ValueError(
                f"Couldn't read the elevation tile {name} ({err}); it was removed from the "
                "cache, so try again."
            ) from err
    return heights


def dem_elevations(
    sample: Sample, raster: str | Path | None = None, *, cache_dir: str | Path | None = None
) -> np.ndarray:
    """Each point's elevation, read from a digital elevation model.

    For samples whose records carry no elevation, or an unreliable one:
    pass the result to [`Sample.with_elevations`][samplesar.Sample.with_elevations]
    and nest with [`strategies.elevation`][samplesar.strategies.elevation].

    By default the model is Copernicus GLO-30 (about 30 m cells, worldwide
    land): the one-degree tiles the points fall in are downloaded once, about
    40 MB each, and cached. It is a surface model, so heights include forest
    canopy and buildings. Give ``raster`` to read your own model instead
    (SRTM, a lidar mosaic, a national model): the first band of any raster
    rasterio opens, in any coordinate system it declares. Cells equal to the
    raster's declared nodata value are missing; a raster that declares none
    returns its sentinel values (-9999, say) as heights. Needs ``pip install
    'samplesar[elevation]'``.

    Copernicus DEM is free to use with attribution: "produced using
    Copernicus WorldDEM-30 © DLR e.V. 2010-2014 and © Airbus Defence and
    Space GmbH 2014-2018 provided under COPERNICUS by the European Union and
    ESA; all rights reserved".

    Parameters
    ----------
    sample : Sample
        The sample whose points to look up.
    raster : str or Path, optional
        A local elevation raster in meters. By default Copernicus GLO-30.
    cache_dir : str or Path, optional
        Where downloaded tiles are kept; by default ``$SAMPLESAR_CACHE/elevation``
        if that's set, else ``samplesar/elevation`` under ``$XDG_CACHE_HOME``
        or ``~/.cache``.

    Returns
    -------
    np.ndarray
        Meters above sea level (above the EGM2008 geoid, for GLO-30), one
        per row of
        [`points`][samplesar.Sample.points]; NaN where the model has no
        value (the sea in GLO-30, which has no tiles there; outside your
        raster, or at its nodata value).

    Raises
    ------
    ImportError
        If rasterio isn't installed.
    ValueError
        If ``raster`` doesn't exist, has no coordinate reference system or
        can't be read, or a tile can't be downloaded or read.

    Examples
    --------
    >>> heights = samplesar.sources.dem_elevations(sample)  # doctest: +SKIP
    >>> sample = sample.with_elevations(heights)  # doctest: +SKIP
    >>> nest = samplesar.strategies.elevation(sample, remove="lowest")  # doctest: +SKIP
    """
    _rasterio()
    lon, lat = sample.points[:, 0], sample.points[:, 1]
    if raster is not None:
        path = Path(raster).expanduser()
        if not path.exists():
            raise ValueError(f"No elevation raster at {path}.")
        try:
            return _read(path, lon, lat)
        except Exception as err:
            if isinstance(err, ImportError):
                raise
            raise ValueError(f"Couldn't read elevations from {path}: {err}") from err
    cache = cache_root(cache_dir, "elevation")
    cache.mkdir(parents=True, exist_ok=True)
    tiles = np.array([glo30_tile(x, y) for x, y in zip(lon, lat, strict=True)])
    heights = _from_tiles(tiles, lon, lat, cache)
    # Each tile sits half a cell north-west of its whole degrees, so a point
    # within half a cell of a tile's south or east edge is in the next one.
    for east, south in ((0.0, 1.0), (1.0, 0.0), (1.0, 1.0)):
        missing = np.flatnonzero(np.isnan(heights))
        if not len(missing):
            break
        cell = 1.0 / 1200.0  # the widest cells, at high latitudes
        near = np.array(
            [
                glo30_tile(x + east * cell, y - south * cell)
                for x, y in zip(lon[missing], lat[missing], strict=True)
            ]
        )
        moved = near != tiles[missing]
        if moved.any():
            heights[missing[moved]] = _from_tiles(
                near[moved], lon[missing[moved]], lat[missing[moved]], cache
            )
    return heights
