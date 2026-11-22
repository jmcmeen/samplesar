"""Regions from any boundary file, and from known ecoregion datasets."""

from __future__ import annotations

import hashlib
import logging
import os
import urllib.request
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import shapely
from shapely.geometry import box

logger = logging.getLogger(__name__)

TIMEOUT = 300  # seconds without data before a download fails

ECOREGIONS: dict[str, dict[str, str]] = {
    "epa_l3": {
        "url": "https://dmap-prod-oms-edc.s3.us-east-1.amazonaws.com/ORD/Ecoregions/us/us_eco_l3.zip",
        "code": "US_L3CODE",
        "name": "US_L3NAME",
        "about": "US EPA Level III ecoregions of the conterminous US (Omernik & Griffith 2014); public domain",
    },
    "epa_l4": {
        "url": "https://dmap-prod-oms-edc.s3.us-east-1.amazonaws.com/ORD/Ecoregions/us/us_eco_l4.zip",
        "code": "US_L4CODE",
        "name": "US_L4NAME",
        "about": "US EPA Level IV ecoregions of the conterminous US (Omernik & Griffith 2014); public domain",
    },
    "resolve_2017": {
        "url": "https://storage.googleapis.com/teow2016/Ecoregions2017.zip",
        "code": "ECO_ID",
        "name": "ECO_NAME",
        "about": "RESOLVE Ecoregions 2017, worldwide (Dinerstein et al. 2017); CC BY 4.0",
    },
}


def cache_root(given: str | Path | None, subdir: str) -> Path:
    """Where downloads are kept: ``given`` if set, else ``subdir`` under
    ``$SAMPLESAR_CACHE``, or under ``samplesar`` in ``$XDG_CACHE_HOME`` or
    ``~/.cache``."""
    if given is not None:
        return Path(given).expanduser()
    if os.environ.get("SAMPLESAR_CACHE"):
        return Path(os.environ["SAMPLESAR_CACHE"]).expanduser() / subdir
    base = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(base) / "samplesar" / subdir


def _fetch_to(url: str, path: Path) -> None:
    """Download ``url`` to ``path`` through a temporary file, so an
    interrupted download leaves nothing behind."""
    partial = path.with_name(path.name + ".partial")
    try:
        with urllib.request.urlopen(url, timeout=TIMEOUT) as response, partial.open("wb") as out:
            while chunk := response.read(8 * 1024 * 1024):
                out.write(chunk)
    except Exception as err:
        partial.unlink(missing_ok=True)
        raise ValueError(f"Couldn't download {url}: {err}") from err
    os.replace(partial, path)


def _local(source: str | Path, cache_dir: str | Path | None) -> Path:
    """A local file for ``source``: a path as it is, a URL downloaded once into
    the cache (named by the URL, so two sources never collide)."""
    text = str(source)
    if not text.startswith(("http://", "https://")):
        path = Path(text).expanduser()
        if not path.exists():
            raise ValueError(f"No boundary file at {path}.")
        return path
    cache = cache_root(cache_dir, "regions")
    cache.mkdir(parents=True, exist_ok=True)
    name = text.rstrip("/").rsplit("/", 1)[-1] or "boundary"
    path = cache / f"{hashlib.sha1(text.encode()).hexdigest()[:10]}_{name}"
    if not path.exists():
        logger.info("Downloading %s", text)
        _fetch_to(text, path)
    return path


def _geopandas() -> Any:
    try:
        import geopandas
    except ImportError as err:
        raise ImportError(
            "Reading boundary files needs geopandas: pip install 'samplesar[geo]'"
        ) from err
    return geopandas


def read_boundary(
    source: str | Path,
    column: str | None = None,
    values: Iterable[Any] | None = None,
    *,
    clip: Any = None,
    cache_dir: str | Path | None = None,
) -> Any:
    """A region from any boundary file: some of its rows, dissolved into one polygon.

    For regions that cross political lines (ecoregions, physiographic
    provinces, watersheds, protected areas) or any other polygons, to pass to
    [`Sample.within`][samplesar.Sample.within] and
    [`Sample.from_open_data`][samplesar.Sample.from_open_data]'s
    ``boundary``. The file is read with geopandas (``pip install
    'samplesar[geo]'``); a URL is downloaded once and cached, so a script
    that names its sources is reproducible.

    Parameters
    ----------
    source : str or Path
        A local path or URL to a vector file geopandas reads: a shapefile
        (or a zip of one), GeoPackage, GeoJSON, and so on.
    column : str, optional
        The column to select rows by. By default every row is used.
    values : iterable, optional
        Keep rows whose ``column`` is one of these, compared as text without
        regard to case (so codes may be given as ``"66"`` or ``66``).
    clip : tuple of float or polygon, optional
        Cut the region to a (west, south, east, north) box in degrees, or to
        a polygon (as ``Sample.within`` takes).
    cache_dir : str or Path, optional
        Where downloads are kept; by default ``regions`` under
        ``$SAMPLESAR_CACHE``, or ``samplesar/regions`` under
        ``$XDG_CACHE_HOME`` or ``~/.cache``.

    Returns
    -------
    shapely.Polygon or shapely.MultiPolygon
        In WGS84 longitude and latitude.

    Raises
    ------
    ValueError
        If the file is missing or can't be downloaded, ``column`` is given
        without ``values`` (or the reverse), a value matches no row (the
        message lists the column's values), or nothing is left after
        clipping.
    KeyError
        If ``column`` isn't in the file (the message lists its columns).
    ImportError
        If geopandas isn't installed.

    Examples
    --------
    >>> region = samplesar.read_boundary(  # doctest: +SKIP
    ...     "https://example.org/provinces.zip", "PROVINCE", ["Blue Ridge"]
    ... )
    """
    if (column is None) != (values is None):
        raise ValueError("Give column and values together, or neither.")
    gpd = _geopandas()
    path = _local(source, cache_dir)
    frame = gpd.read_file(f"zip://{path}" if path.suffix == ".zip" else path)
    if column is not None and values is not None:
        if column not in frame.columns:
            raise KeyError(
                f"No column {column!r}; the file has {', '.join(map(str, frame.columns))}."
            )
        wanted = [str(v).casefold() for v in values]
        present = frame[column].astype(str).str.casefold()
        missing = [v for v in wanted if v not in set(present)]
        if missing:
            available = sorted(set(frame[column].astype(str)))
            shown = ", ".join(available[:30]) + (" ..." if len(available) > 30 else "")
            raise ValueError(f"No row has {column} = {', '.join(missing)}. Values: {shown}")
        frame = frame[present.isin(wanted)]
    if frame.crs is not None:
        frame = frame.to_crs(4326)
    region = shapely.union_all(shapely.make_valid(frame.geometry.to_numpy()))
    if clip is not None:
        if isinstance(clip, tuple | list) and len(clip) == 4:
            region = region.intersection(box(*clip))
        else:
            from samplesar._boundary import as_geometry

            region = region.intersection(as_geometry(clip))
    parts = [
        g
        for g in getattr(region, "geoms", [region])
        if g.geom_type in ("Polygon", "MultiPolygon") and not g.is_empty
    ]
    if not parts:
        raise ValueError("Nothing of the region is left after clipping.")
    return parts[0] if len(parts) == 1 else shapely.union_all(parts)


def ecoregion(
    source: str = "epa_l3",
    codes: Iterable[Any] | None = None,
    names: Iterable[str] | None = None,
    *,
    clip: Any = None,
    cache_dir: str | Path | None = None,
) -> Any:
    """Ecoregions from a known dataset, by code or name, as one polygon.

    The datasets (downloaded once and cached, read with geopandas: ``pip
    install 'samplesar[geo]'``):

    - ``"epa_l3"``: US EPA Level III ecoregions of the conterminous US
      (Omernik & Griffith 2014), public domain; codes like ``"66"`` (Blue
      Ridge).
    - ``"epa_l4"``: US EPA Level IV ecoregions, public domain; codes like
      ``"66d"``.
    - ``"resolve_2017"``: RESOLVE Ecoregions 2017, worldwide (Dinerstein et
      al. 2017), CC BY 4.0; ``ECO_ID`` codes or ``ECO_NAME`` names.

    Any other file works through [`read_boundary`][samplesar.read_boundary].

    Parameters
    ----------
    source : str
        One of the datasets above.
    codes : iterable, optional
        Ecoregion codes to dissolve together.
    names : iterable of str, optional
        Ecoregion names (case doesn't matter), instead of codes.
    clip : tuple of float or polygon, optional
        Cut the region to a (west, south, east, north) box or a polygon, as
        [`read_boundary`][samplesar.read_boundary] does.
    cache_dir : str or Path, optional
        As [`read_boundary`][samplesar.read_boundary].

    Returns
    -------
    shapely.Polygon or shapely.MultiPolygon
        In WGS84 longitude and latitude.

    Raises
    ------
    ValueError
        If ``source`` is unknown, both or neither of ``codes`` and ``names``
        are given, or as [`read_boundary`][samplesar.read_boundary].

    Examples
    --------
    The Southern Appalachians: the Blue Ridge, Ridge and Valley and
    Southwestern Appalachians south of the Roanoke Gap.

    >>> region = samplesar.ecoregion(  # doctest: +SKIP
    ...     "epa_l3", codes=["66", "67", "68"], clip=(-90, 30, -75, 37.3)
    ... )
    """
    spec = ECOREGIONS.get(source)
    if spec is None:
        raise ValueError(f"Unknown ecoregion source {source!r}; known: {', '.join(ECOREGIONS)}.")
    if (codes is None) == (names is None):
        raise ValueError("Give codes or names (one of them).")
    column, values = (spec["code"], codes) if codes is not None else (spec["name"], names)
    return read_boundary(spec["url"], column, values, clip=clip, cache_dir=cache_dir)
