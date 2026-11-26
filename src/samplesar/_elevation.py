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
    """The name of the Copernicus GLO-30 tile holding a point: the one whose
    south-west corner is the whole degrees at or below it."""
    south, west = math.floor(lat), math.floor(lon)
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


def _read(path: Path | str, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """The raster's first band at some (lon, lat) points, as floats: NaN at
    its nodata value and outside its extent."""
    rasterio = _rasterio()
    from pyproj import Transformer

    with rasterio.open(path) as raster:
        x, y = lon, lat
        if raster.crs is not None and raster.crs.to_wkt() != WGS84.to_wkt():
            to_raster = Transformer.from_crs(WGS84, raster.crs.to_wkt(), always_xy=True)
            x, y = to_raster.transform(lon, lat)
        values = np.array(
            [v[0] for v in raster.sample(list(zip(x, y, strict=True)), indexes=1, masked=True)],
            dtype=object,
        )
        out = np.array([np.nan if v is np.ma.masked else float(v) for v in values], dtype=float)
        left, bottom, right, top = raster.bounds
        outside = (np.asarray(x) < left) | (np.asarray(x) > right)
        outside |= (np.asarray(y) < bottom) | (np.asarray(y) > top)
        out[outside] = np.nan
    return out


def dem_elevations(
    sample: Sample, raster: str | Path | None = None, *, cache_dir: str | Path | None = None
) -> np.ndarray:
    """Each point's elevation, read from a digital elevation model.

    For samples whose records carry no elevation, or an unreliable one:
    pass the result to [`Sample.with_elevations`][samplesar.Sample.with_elevations]
    and nest with [`strategies.elevation`][samplesar.strategies.elevation].

    By default the model is Copernicus GLO-30 (about 30 m cells, worldwide
    land): the one-degree tiles the points fall in are downloaded once, about
    40 MB each, and cached. Give ``raster`` to read your own model instead
    (SRTM, a lidar mosaic, a national model): any single-band raster rasterio
    opens, in any coordinate system. Needs ``pip install
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
        Meters above sea level, one per row of
        [`points`][samplesar.Sample.points]; NaN where the model has no
        value (the sea in GLO-30, which has no tiles there; outside your
        raster, or at its nodata value).

    Raises
    ------
    ImportError
        If rasterio isn't installed.
    ValueError
        If ``raster`` doesn't exist or can't be read, or a tile can't be
        downloaded.

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
    heights = np.full(sample.n_points, np.nan)
    for name in sorted(set(tiles)):
        path = cache / f"{name}.tif"
        sea = cache / f"{name}.none"  # remembered, so the sea isn't asked for again
        if sea.exists():
            continue
        if not path.exists():
            logger.info("Downloading elevation tile %s", name)
            try:
                _fetch_to(GLO30.format(name=name), path)
            except ValueError as err:
                if "404" in str(err):  # no tile: open sea
                    sea.touch()
                    continue
                raise
        here = tiles == name
        heights[here] = _read(path, lon[here], lat[here])
    return heights
