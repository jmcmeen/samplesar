"""Occurrences from GBIF, through pygbif."""

from __future__ import annotations

import logging
import warnings
from collections.abc import Iterable, Mapping
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

PAGE = 300  # the occurrence search API's largest page
OFFSET_LIMIT = 100_000  # the search API serves no records beyond this offset
SPECIES_LEVEL = ("SPECIES", "SUBSPECIES", "VARIETY", "FORM")


def gbif_frame(
    records: Iterable[Mapping[str, Any]],
    ranks: Iterable[str] | None = SPECIES_LEVEL,
    max_uncertainty: float | None = None,
) -> pd.DataFrame:
    """GBIF occurrence records as ``species``, ``lon``, ``lat`` rows.

    A record needs coordinates and a species-level name (GBIF's ``species``
    field, which names infraspecific taxa by their species); with ``ranks``,
    its ``taxonRank`` must be one of them, and with ``max_uncertainty`` its
    ``coordinateUncertaintyInMeters`` must be given and at most that.
    """
    wanted = None if ranks is None else {r.upper() for r in ranks}
    rows = []
    for r in records:
        if r.get("decimalLongitude") is None or r.get("decimalLatitude") is None:
            continue
        name = r.get("species") or (r.get("scientificName") if wanted is None else None)
        if not name:
            continue
        if wanted is not None and str(r.get("taxonRank", "")).upper() not in wanted:
            continue
        if max_uncertainty is not None:
            u = r.get("coordinateUncertaintyInMeters")
            if u is None or float(u) > max_uncertainty:
                continue
        rows.append(
            {
                "species": name,
                "lon": float(r["decimalLongitude"]),
                "lat": float(r["decimalLatitude"]),
            }
        )
    return pd.DataFrame(rows, columns=["species", "lon", "lat"])


def fetch_gbif(max_results: int | None = None, **query: Any) -> list[dict[str, Any]]:
    """Occurrences matching ``query`` from GBIF's occurrence search API, a
    page at a time, up to the API's 100,000-record limit."""
    try:
        from pygbif import occurrences
    except ImportError as err:
        raise ImportError("Fetching from GBIF needs pygbif: pip install 'samplesar[gbif]'") from err
    if max_results is not None and max_results < 1:
        raise ValueError(f"max_results must be at least 1, got {max_results}.")
    query = {
        "hasCoordinate": True,
        "hasGeospatialIssue": False,
        "occurrenceStatus": "PRESENT",
        **query,
    }
    query.pop("offset", None)
    query.pop("limit", None)
    limit = OFFSET_LIMIT if max_results is None else min(max_results, OFFSET_LIMIT)
    records: list[dict[str, Any]] = []
    while len(records) < limit:
        size = min(PAGE, limit - len(records))
        page = occurrences.search(limit=size, offset=len(records), **query)
        results = page.get("results", [])
        records.extend(results)
        if len(records) == len(results):  # the first page says how many match
            count = page.get("count", 0)
            if count > OFFSET_LIMIT and (max_results is None or max_results > OFFSET_LIMIT):
                warnings.warn(
                    f"{count:,} GBIF records match, but the search API serves only the first "
                    f"{OFFSET_LIMIT:,}; narrow the query, or use GBIF's download service.",
                    stacklevel=3,
                )
        logger.info("Fetched %d GBIF records", len(records))
        if page.get("endOfRecords", True) or not results:
            break
    return records[:limit]
