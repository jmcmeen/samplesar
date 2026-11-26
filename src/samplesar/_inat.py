"""Samples from iNaturalist observations, through pyinaturalist."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

PER_PAGE = 200  # the API's largest page
SPECIES_LEVEL = ("species", "subspecies", "variety", "form")
INFRASPECIFIC = {"subspecies", "variety", "form"}


def _coordinates(record: Mapping[str, Any]) -> tuple[float, float] | None:
    """(lon, lat) from a record's GeoJSON point, or its location (a
    "lat,lon" string, or [lat, lon] as pyinaturalist parses it)."""
    geojson = record.get("geojson")
    if geojson and geojson.get("coordinates"):
        lon, lat = geojson["coordinates"][:2]
        return float(lon), float(lat)
    location = record.get("location")
    if isinstance(location, str) and "," in location:
        lat, lon = (float(v) for v in location.split(",")[:2])
        return lon, lat
    if isinstance(location, list | tuple) and len(location) >= 2:
        return float(location[1]), float(location[0])
    return None


def _species_name(taxon: Mapping[str, Any]) -> str:
    """The taxon's name, with a subspecies, variety or form counted under its
    species (the first two words of its name)."""
    name = str(taxon["name"])
    if taxon.get("rank") in INFRASPECIFIC:
        return " ".join(name.split()[:2])
    return name


def inaturalist_frame(
    records: Iterable[Mapping[str, Any]],
    ranks: Iterable[str] | None = SPECIES_LEVEL,
    skip_obscured: bool = True,
) -> pd.DataFrame:
    """iNaturalist observation records as ``species``, ``lon``, ``lat``, ``date`` rows.

    Records without a taxon or coordinates are dropped, as are those whose
    taxon isn't at one of ``ranks`` (``None`` keeps every rank) and, with
    ``skip_obscured``, those whose coordinates iNaturalist has obscured. A
    subspecies, variety or form counts as its species.
    """
    wanted = None if ranks is None else set(ranks)
    rows = []
    dropped = {"taxon": 0, "rank": 0, "obscured": 0, "coordinates": 0}
    for record in records:
        taxon = record.get("taxon") or {}
        if not taxon.get("name"):
            dropped["taxon"] += 1
            continue
        if wanted is not None and taxon.get("rank") not in wanted:
            dropped["rank"] += 1
            continue
        if skip_obscured and (
            record.get("obscured")
            or record.get("geoprivacy") in ("obscured", "private")
            or record.get("taxon_geoprivacy") in ("obscured", "private")
        ):
            dropped["obscured"] += 1
            continue
        point = _coordinates(record)
        if point is None:
            dropped["coordinates"] += 1
            continue
        rows.append(
            {
                "species": _species_name(taxon),
                "lon": point[0],
                "lat": point[1],
                "date": _observed(record),
            }
        )
    if any(dropped.values()):
        logger.info("Dropped iNaturalist records: %s", dropped)
    return pd.DataFrame(rows, columns=["species", "lon", "lat", "date"])


def _observed(record: Mapping[str, Any]) -> Any:
    """When the record was observed, as a UTC time without a zone; NaT when
    it has no usable date."""
    # The time when there is one: the date alone ties every point of a day.
    value = record.get("time_observed_at") or record.get("observed_on")
    if not value:
        return pd.NaT
    try:
        when = pd.to_datetime(value, utc=True)
    except (TypeError, ValueError, OverflowError):
        return pd.NaT
    return when.tz_localize(None)


def fetch_inaturalist(max_results: int | None = None, **query: Any) -> list[dict[str, Any]]:
    """Every observation matching ``query``, paging by id past the API's
    10,000-result limit (pyinaturalist keeps to the rate limit). A query's
    ``id_above`` is where the fetch starts, so an interrupted one can resume."""
    if max_results is not None and max_results < 1:
        raise ValueError(f"max_results must be at least 1, got {max_results}.")
    per_page = query.pop("per_page", None)
    per_page = PER_PAGE if per_page is None else int(per_page)
    if per_page < 1:
        raise ValueError(f"per_page must be at least 1, got {per_page}.")
    per_page = min(per_page, PER_PAGE)  # the API serves no more than this a page
    try:
        from pyinaturalist import get_observations
    except ImportError as err:
        raise ImportError(
            "Fetching from iNaturalist needs pyinaturalist: pip install 'samplesar[inat]'"
        ) from err
    last_id = query.pop("id_above", None)
    query = {k: v for k, v in query.items() if k not in ("page", "order", "order_by")}
    records: list[dict[str, Any]] = []
    while max_results is None or len(records) < max_results:
        size = per_page if max_results is None else min(per_page, max_results - len(records))
        page = get_observations(
            **query,
            per_page=size,
            order="asc",
            order_by="id",
            **({"id_above": last_id} if last_id else {}),
        )
        results = page.get("results", [])
        if not results:
            break
        records.extend(results)
        last_id = results[-1]["id"]
        logger.info("Fetched %d iNaturalist observations", len(records))
        if len(results) < size:
            break
    return records
