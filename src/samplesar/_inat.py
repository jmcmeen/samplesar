"""Samples from iNaturalist observations, through pyinaturalist."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

PER_PAGE = 200  # the API's largest page


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


def inaturalist_frame(
    records: Iterable[Mapping[str, Any]],
    ranks: Iterable[str] | None = ("species",),
    skip_obscured: bool = True,
) -> pd.DataFrame:
    """iNaturalist observation records as ``species``, ``lon``, ``lat`` rows.

    Records without a taxon or coordinates are dropped, as are those whose
    taxon isn't at one of ``ranks`` (``None`` keeps every rank) and, with
    ``skip_obscured``, those whose coordinates iNaturalist has obscured.
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
        rows.append({"species": taxon["name"], "lon": point[0], "lat": point[1]})
    if any(dropped.values()):
        logger.info("Dropped iNaturalist records: %s", dropped)
    return pd.DataFrame(rows, columns=["species", "lon", "lat"])


def fetch_inaturalist(max_results: int | None = None, **query: Any) -> list[dict[str, Any]]:
    """Every observation matching ``query``, paging by id past the API's
    10,000-result limit (pyinaturalist keeps to the rate limit)."""
    try:
        from pyinaturalist import get_observations
    except ImportError as err:
        raise ImportError(
            "Fetching from iNaturalist needs pyinaturalist: pip install 'samplesar[inat]'"
        ) from err
    query = {k: v for k, v in query.items() if k not in ("page", "order", "order_by", "id_above")}
    per_page = int(query.pop("per_page", PER_PAGE))
    records: list[dict[str, Any]] = []
    last_id = None
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
