"""A local copy of iNaturalist's open-data export, and samples drawn from it."""

from __future__ import annotations

import http.client
import json
import logging
import re
import shutil
import time
import urllib.request
from collections.abc import Iterable
from datetime import date
from pathlib import Path
from typing import Any

from samplesar._http import request

logger = logging.getLogger(__name__)

BUCKET_URL = "https://inaturalist-open-data.s3.amazonaws.com"
FILES = ("observations.csv.gz", "taxa.csv.gz")
PARQUET = "observations.parquet"
CHUNK = 8 * 1024 * 1024


def _duckdb() -> Any:
    try:
        import duckdb
    except ImportError as err:
        raise ImportError(
            "iNaturalist open data needs DuckDB: pip install 'samplesar[opendata]'"
        ) from err
    return duckdb


def _remote(url: str) -> tuple[int, str]:
    """The remote file's size and version (its ETag, else Last-Modified)."""
    with urllib.request.urlopen(request(url, method="HEAD")) as response:
        length = response.headers.get("Content-Length")
        if length is None:
            raise OSError(f"{url} gave no Content-Length; can't check the download.")
        version = response.headers.get("ETag") or response.headers.get("Last-Modified") or ""
        return int(length), version


def _download(url: str, path: Path, retries: int = 5) -> bool:
    """Fetch ``url`` to ``path``, resuming a partial download of the same
    version and skipping a complete one; returns whether anything was
    fetched. The version is kept beside the file (``<name>.source``), so a
    newer export is fetched afresh instead of appended to the old one."""
    source = path.with_name(path.name + ".source")
    fetched = False
    for attempt in range(retries + 1):
        try:
            size, version = _remote(url)
            known = json.loads(source.read_text()) if source.exists() else None
            same = known == {"size": size, "version": version}
            have = path.stat().st_size if path.exists() else 0
            if same and have == size:
                return fetched
            if not same or have > size:
                path.unlink(missing_ok=True)
                have = 0
                source.write_text(json.dumps({"size": size, "version": version}))
            headers = {}
            if have:
                headers["Range"] = f"bytes={have}-"
                if version:
                    headers["If-Range"] = version  # the whole file if it changed meanwhile
            with urllib.request.urlopen(request(url, headers=headers)) as response:
                resumed = have and response.status == 206
                with path.open("ab" if resumed else "wb") as out:
                    while chunk := response.read(CHUNK):
                        out.write(chunk)
            fetched = True
            if path.stat().st_size == size:
                logger.info("Downloaded %s (%.1f GB)", path.name, size / 1e9)
                return fetched
            raise OSError(f"{path.name} is {path.stat().st_size} bytes; expected {size}.")
        except (OSError, http.client.HTTPException) as err:
            if attempt == retries:
                raise OSError(f"Downloading {url} failed: {err}") from err
            logger.warning("Download of %s interrupted (%s); resuming", path.name, err)
            time.sleep(min(60, 5 * 2**attempt))
    return fetched


def _sql_text(value: str | Path) -> str:
    """A string as a SQL literal."""
    return "'" + str(value).replace("'", "''") + "'"


def build_parquet(folder: str | Path, memory_limit: str = "8GB") -> Path:
    """Join the observations with the taxa into one Parquet file, grouped by
    1° cell of latitude and longitude so a query for a region reads little.

    Parameters
    ----------
    folder : str or Path
        Holding ``observations.csv.gz`` and ``taxa.csv.gz``.
    memory_limit : str
        DuckDB's memory limit, such as ``"8GB"``; it spills to ``folder``
        beyond it.

    Returns
    -------
    Path
        ``observations.parquet`` in ``folder``.
    """
    if not re.fullmatch(r"\d+(\.\d+)?\s*[KMGT]i?B", memory_limit):
        raise ValueError(f"memory_limit must look like '8GB', got {memory_limit!r}.")
    duckdb = _duckdb()
    folder = Path(folder)
    out, tmp = folder / PARQUET, folder / "duckdb-tmp"
    partial = out.with_suffix(".parquet.partial")
    con = duckdb.connect()
    try:
        con.execute(f"SET temp_directory = {_sql_text(tmp)}")
        con.execute(f"SET memory_limit = {_sql_text(memory_limit)}")
        con.execute("SET preserve_insertion_order = false")
        con.execute(
            f"""
            COPY (
                SELECT
                    o.observation_uuid, o.observer_id,
                    o.latitude, o.longitude, o.positional_accuracy,
                    o.taxon_id, t.name, t.rank, t.rank_level, t.ancestry,
                    o.quality_grade, o.observed_on, o.anomaly_score
                FROM read_csv({_sql_text(folder / "observations.csv.gz")}, delim='\t', header=true,
                              types={{'observed_on': 'DATE', 'positional_accuracy': 'DOUBLE',
                                      'anomaly_score': 'DOUBLE', 'taxon_id': 'BIGINT',
                                      'latitude': 'DOUBLE', 'longitude': 'DOUBLE'}}) AS o
                LEFT JOIN read_csv({_sql_text(folder / "taxa.csv.gz")}, delim='\t', header=true,
                                   types={{'taxon_id': 'BIGINT'}}) AS t USING (taxon_id)
                ORDER BY floor(o.latitude), floor(o.longitude)
            ) TO {_sql_text(partial)} (FORMAT parquet, COMPRESSION zstd, ROW_GROUP_SIZE 1000000)
            """
        )
        partial.replace(out)  # only a finished file takes the final name
    finally:
        con.close()
        partial.unlink(missing_ok=True)
        shutil.rmtree(tmp, ignore_errors=True)
    return out


def download_open_data(
    folder: str | Path,
    *,
    base_url: str = BUCKET_URL,
    build: bool = True,
    rebuild: bool = False,
    memory_limit: str = "8GB",
) -> Path:
    """Download iNaturalist's open-data export and build a local Parquet file.

    Fetches ``observations.csv.gz`` (about 14 GB in late 2026) and
    ``taxa.csv.gz`` from iNaturalist's public open-data bucket into
    ``folder``, resuming a partial download and skipping files already
    current (a newer export is fetched afresh), then joins them into
    ``observations.parquet`` (about 12 GB), unless it's already built from
    these files:
    one row per observation with its taxon's name, rank and ancestry,
    grouped by 1° cell so that a query for a region reads little of it.
    Build samples from it with
    [`sources.open_data`][samplesar.sources.open_data].

    The export holds the observations with openly licensed photos, not all
    of iNaturalist. Obscured observations carry their public, randomized
    coordinates and aren't marked; ``positional_accuracy`` is there to filter
    on.

    Parameters
    ----------
    folder : str or Path
        Where the files go (created if needed). Needs about 30 GB free.
    base_url : str
        The bucket's address.
    build : bool
        Build the Parquet file after downloading.
    rebuild : bool
        Build it even if it's newer than the downloads.
    memory_limit : str
        DuckDB's memory limit for the build; it spills to ``folder`` beyond
        it.

    Returns
    -------
    Path
        The Parquet file (or the folder, if ``build`` is false).

    Raises
    ------
    ImportError
        If DuckDB isn't installed (``pip install 'samplesar[opendata]'``).
    OSError
        If a download keeps failing.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    if build:
        _duckdb()  # fail before a long download, not after it
    fetched = [_download(f"{base_url.rstrip('/')}/{name}", folder / name) for name in FILES]
    if not build:
        return folder
    out = folder / PARQUET
    current = out.exists() and all(
        out.stat().st_mtime >= (folder / name).stat().st_mtime for name in FILES
    )
    if current and not any(fetched) and not rebuild:
        return out
    return build_parquet(folder, memory_limit=memory_limit)


def open_data_frame(
    path: str | Path,
    *,
    bbox: tuple[float, float, float, float] | None = None,
    taxon_id: int | None = None,
    quality_grade: str | Iterable[str] | None = "research",
    ranks: Iterable[str] | None = ("species", "subspecies", "variety", "form"),
    since: date | str | None = None,
    until: date | str | None = None,
    max_accuracy: float | None = None,
) -> Any:
    """The rows of the Parquet file matching the filters, as ``species``,
    ``lon``, ``lat``, ``date``, with infraspecific taxa named as their species."""
    duckdb = _duckdb()
    where: list[str] = []
    params: list[Any] = []
    if bbox is not None:
        west, south, east, north = (float(v) for v in bbox)
        if not (south <= north and -90 <= south and north <= 90):
            raise ValueError(f"bbox must be (west, south, east, north), got {bbox!r}.")
        where.append("latitude BETWEEN ? AND ?")
        params += [south, north]
        if west <= east:
            where.append("longitude BETWEEN ? AND ?")
            params += [west, east]
        else:  # across the antimeridian
            where.append("(longitude >= ? OR longitude <= ?)")
            params += [west, east]
    if taxon_id is not None:
        where.append("(taxon_id = ? OR ('/' || ancestry || '/') LIKE ?)")
        params += [int(taxon_id), f"%/{int(taxon_id)}/%"]
    if quality_grade is not None:
        grades = [quality_grade] if isinstance(quality_grade, str) else list(quality_grade)
        if not grades:
            raise ValueError("quality_grade is empty; give grades, or None for every grade.")
        where.append(f"quality_grade IN ({', '.join('?' * len(grades))})")
        params += grades
    if ranks is not None:
        ranks = list(ranks)
        if not ranks:
            raise ValueError("ranks is empty; give ranks, or None for every rank.")
        where.append(f"rank IN ({', '.join('?' * len(ranks))})")
        params += ranks
    if since is not None:
        where.append("observed_on >= CAST(? AS DATE)")
        params.append(str(since))
    if until is not None:
        where.append("observed_on <= CAST(? AS DATE)")
        params.append(str(until))
    if max_accuracy is not None:
        where.append("positional_accuracy IS NOT NULL AND positional_accuracy <= ?")
        params.append(float(max_accuracy))
    where.append("latitude IS NOT NULL AND longitude IS NOT NULL AND name IS NOT NULL")
    query = f"""
        SELECT
            CASE WHEN rank IN ('subspecies', 'variety', 'form')
                 THEN array_to_string(string_split(name, ' ')[1:2], ' ')
                 ELSE name END AS species,
            longitude AS lon, latitude AS lat, CAST(observed_on AS TIMESTAMP) AS date
        FROM read_parquet(?)
        WHERE {" AND ".join(where)}
    """
    return duckdb.execute(query, [str(path), *params]).df()
