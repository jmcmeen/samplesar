"""Administrative boundaries from geoBoundaries (www.geoboundaries.org)."""

from __future__ import annotations

import difflib
import json
import logging
import os
import re
import urllib.request
from pathlib import Path
from typing import Any

import shapely
from shapely.geometry import shape

logger = logging.getLogger(__name__)

API_URL = "https://www.geoboundaries.org/api/current/gbOpen/{iso}/ADM{level}/"


def _cache_dir(given: str | Path | None) -> Path:
    from samplesar._regions import cache_root

    return cache_root(given, "geoboundaries")


TIMEOUT = 120  # seconds without data before a download fails


def _fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=TIMEOUT) as response:
        return response.read()


def _cached(path: Path, url: str) -> bytes:
    """``url``'s content, kept at ``path``: downloaded to a temporary file and
    moved into place, so an interrupted download leaves no broken cache, and
    fetched again if the cached copy doesn't parse."""
    if path.exists():
        content = path.read_bytes()
        try:
            json.loads(content)
            return content
        except ValueError:
            path.unlink()
    try:
        content = _fetch(url)
        json.loads(content)
    except Exception as err:
        raise ValueError(f"Couldn't download {url}: {err}") from err
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_bytes(content)
    os.replace(partial, path)
    return content


def _features(iso: str, level: int, simplified: bool, cache: Path) -> tuple[list[dict], dict]:
    """The GeoJSON features of one country's level, and its metadata,
    downloaded once and cached."""
    cache.mkdir(parents=True, exist_ok=True)
    stem = f"{iso}_ADM{level}"
    meta_path = cache / f"{stem}.json"
    try:
        meta = json.loads(_cached(meta_path, API_URL.format(iso=iso, level=level)))
    except ValueError as err:
        raise ValueError(
            f"geoBoundaries has no ADM{level} boundaries for {iso!r} (or couldn't be "
            f"reached): {err}"
        ) from err
    key = "simplifiedGeometryGeoJSON" if simplified else "gjDownloadURL"
    if not meta.get(key):
        kind = "simplified " if simplified else ""
        raise ValueError(f"geoBoundaries gives no {kind}file for {iso} ADM{level}.")
    geo_path = cache / f"{stem}{'_simplified' if simplified else ''}.geojson"
    if not geo_path.exists():
        logger.info("Downloading %s ADM%d boundaries from geoBoundaries", iso, level)
    return json.loads(_cached(geo_path, meta[key]))["features"], meta


def boundary(
    country: str,
    level: int = 0,
    name: str | None = None,
    *,
    simplified: bool = False,
    inside: Any = None,
    cache_dir: str | Path | None = None,
) -> Any:
    """A country's boundary, or a state's, county's or district's, from geoBoundaries.

    For [`Sample.within`][samplesar.Sample.within] and
    [`Sample.from_open_data`][samplesar.Sample.from_open_data]'s
    ``boundary``. Boundaries come from geoBoundaries' open release (Runfola
    et al. 2020), downloaded once and cached. Each country's boundaries carry
    their own license (CC BY 4.0 or ODbL; see geoboundaries.org) and need
    attribution when published. For US cities, the Census Bureau's TIGER
    "places" are better; any polygon works with ``within``.

    Parameters
    ----------
    country : str
        The country's ISO 3166-1 alpha-3 code, such as ``"CRI"`` or
        ``"USA"``.
    level : int
        0 for the country, 1 for its first-level units (states, provinces),
        2 for the next (counties, districts), up to 5 where a country has
        them.
    name : str, optional
        The unit to return, matched without regard to case against its name
        (``"Tennessee"``) or ISO 3166-2 code (``"US-TN"``). By default every
        unit at the level, dissolved into one.
    simplified : bool
        Use geoBoundaries' simplified geometry: much smaller, slightly less
        exact near the edges.
    inside : polygon, optional
        Keep only units lying in this region (by a point inside each unit),
        to tell apart units with one name: ``inside=samplesar.boundary("USA",
        1, "Tennessee")`` picks Tennessee's Sullivan County.
    cache_dir : str or Path, optional
        Where downloads are kept; by default ``$SAMPLESAR_CACHE/geoboundaries``
        if that's set, else ``samplesar/geoboundaries`` under
        ``$XDG_CACHE_HOME`` or ``~/.cache``.

    Returns
    -------
    shapely.Polygon or shapely.MultiPolygon
        In WGS84 longitude and latitude.

    Raises
    ------
    ValueError
        If the country code or level is invalid or unavailable, a download
        fails, no unit lies ``inside`` the region, or ``name`` matches no
        unit or several (the message lists them).

    Examples
    --------
    >>> costa_rica = samplesar.boundary("CRI")  # doctest: +SKIP
    >>> tennessee = samplesar.boundary("USA", level=1, name="Tennessee")  # doctest: +SKIP
    >>> county = samplesar.boundary("USA", 2, "Sullivan", inside=tennessee)  # doctest: +SKIP
    """
    if not (isinstance(country, str) and re.fullmatch(r"[A-Za-z]{3}", country)):
        raise ValueError(
            f"country must be an ISO 3166-1 alpha-3 code such as 'CRI', got {country!r}."
        )
    if level not in range(6):
        raise ValueError(f"level must be 0 to 5, got {level!r}.")
    iso = country.upper()
    features, _ = _features(iso, level, simplified, _cache_dir(cache_dir))
    # Some published rings self-intersect; repair them so unions and tests work.
    for f in features:
        f["_geometry"] = shapely.make_valid(shape(f["geometry"]))
    if inside is not None:
        from samplesar._boundary import as_geometry

        region = as_geometry(inside)
        shapely.prepare(region)
        features = [f for f in features if region.contains(f["_geometry"].representative_point())]
        if not features:
            raise ValueError(f"No ADM{level} unit of {iso} lies inside the given region.")
    if name is not None:
        wanted = name.casefold()
        matches = [
            f
            for f in features
            if wanted
            in (
                str(f["properties"].get("shapeName", "")).casefold(),
                str(f["properties"].get("shapeISO", "")).casefold(),
            )
        ]
        if not matches:
            names = sorted({str(f["properties"].get("shapeName", "")) for f in features})
            close = difflib.get_close_matches(name, names, n=5, cutoff=0.6)
            hint = f" Close: {', '.join(close)}." if close else ""
            raise ValueError(f"No ADM{level} unit of {iso} is named {name!r}.{hint}")
        if len(matches) > 1:
            listed = ", ".join(
                f"{f['properties'].get('shapeName')} ({f['properties'].get('shapeID')})"
                for f in matches
            )
            raise ValueError(
                f"{len(matches)} ADM{level} units of {iso} are named {name!r}: {listed}. "
                "Narrow them with inside=, a boundary of the level above."
            )
        features = matches
    merged = shapely.union_all([f["_geometry"] for f in features])
    # make_valid can leave stray lines or points beside the polygons.
    parts = [
        g for g in getattr(merged, "geoms", [merged]) if g.geom_type in ("Polygon", "MultiPolygon")
    ]
    return parts[0] if len(parts) == 1 else shapely.union_all(parts)
