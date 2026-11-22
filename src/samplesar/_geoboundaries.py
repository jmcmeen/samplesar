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
    if given is not None:
        return Path(given).expanduser()
    if os.environ.get("SAMPLESAR_CACHE"):
        return Path(os.environ["SAMPLESAR_CACHE"]).expanduser() / "geoboundaries"
    base = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(base) / "samplesar" / "geoboundaries"


def _fetch(url: str) -> bytes:
    with urllib.request.urlopen(url) as response:
        return response.read()


def _features(iso: str, level: int, simplified: bool, cache: Path) -> tuple[list[dict], dict]:
    """The GeoJSON features of one country's level, and its metadata,
    downloaded once and cached."""
    cache.mkdir(parents=True, exist_ok=True)
    stem = f"{iso}_ADM{level}"
    meta_path = cache / f"{stem}.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
    else:
        try:
            meta = json.loads(_fetch(API_URL.format(iso=iso, level=level)))
        except Exception as err:
            raise ValueError(
                f"geoBoundaries has no ADM{level} boundaries for {iso!r} (or couldn't be "
                f"reached): {err}"
            ) from err
        meta_path.write_text(json.dumps(meta))
    key = "simplifiedGeometryGeoJSON" if simplified else "gjDownloadURL"
    geo_path = cache / f"{stem}{'_simplified' if simplified else ''}.geojson"
    if not geo_path.exists():
        logger.info("Downloading %s ADM%d boundaries from geoBoundaries", iso, level)
        geo_path.write_bytes(_fetch(meta[key]))
    return json.loads(geo_path.read_text())["features"], meta


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
        Where downloads are kept; by default ``$SAMPLESAR_CACHE`` or
        ``~/.cache/samplesar``.

    Returns
    -------
    shapely.Polygon or shapely.MultiPolygon
        In WGS84 longitude and latitude.

    Raises
    ------
    ValueError
        If the country code or level is invalid or unavailable, or ``name``
        matches no unit or several (the message lists them).

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
    if inside is not None:
        from samplesar._boundary import as_geometry

        region = as_geometry(inside)
        shapely.prepare(region)
        features = [
            f for f in features if region.contains(shape(f["geometry"]).representative_point())
        ]
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
    return shapely.union_all([shape(f["geometry"]) for f in features])
