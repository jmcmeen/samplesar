"""A local copy of iNaturalist's open-data export, and samples drawn from it."""

from __future__ import annotations

import logging
import time
import urllib.request
from collections.abc import Iterable
from datetime import date
from pathlib import Path
from typing import Any

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


def _remote_size(url: str) -> int:
    request = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(request) as response:
        return int(response.headers["Content-Length"])


def _download(url: str, path: Path, retries: int = 5) -> None:
    """Fetch ``url`` to ``path``, resuming a partial file and skipping a
    complete one (same size as the remote)."""
    size = _remote_size(url)
    for attempt in range(retries + 1):
        have = path.stat().st_size if path.exists() else 0
        if have == size:
            return
        if have > size:  # the remote file changed: start again
            path.unlink()
            have = 0
        request = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
        try:
            with (
                urllib.request.urlopen(request) as response,
                path.open("ab" if have else "wb") as out,
            ):
                if have and response.status != 206:  # no range support: start again
                    out.truncate(0)
                while chunk := response.read(CHUNK):
                    out.write(chunk)
            logger.info("Downloaded %s (%.1f GB)", path.name, path.stat().st_size / 1e9)
        except OSError as err:
            if attempt == retries:
                raise
            logger.warning("Download of %s interrupted (%s); resuming", path.name, err)
            time.sleep(min(60, 5 * 2**attempt))
    if path.stat().st_size != size:
        raise OSError(f"{path.name} is {path.stat().st_size} bytes; expected {size}.")


def build_parquet(folder: Path, memory_limit: str = "8GB") -> Path:
    """Join the observations with the taxa into one Parquet file, grouped by
    1° cell of latitude and longitude so a query for a region reads little."""
    duckdb = _duckdb()
    folder = Path(folder)
    out, tmp = folder / PARQUET, folder / "duckdb-tmp"
    con = duckdb.connect()
    con.execute(f"SET temp_directory = '{tmp}'")
    con.execute(f"SET memory_limit = '{memory_limit}'")
    con.execute("SET preserve_insertion_order = false")
    partial = out.with_suffix(".parquet.partial")
    con.execute(
        f"""
        COPY (
            SELECT
                o.observation_uuid, o.observer_id,
                o.latitude, o.longitude, o.positional_accuracy,
                o.taxon_id, t.name, t.rank, t.rank_level, t.ancestry,
                o.quality_grade, o.observed_on, o.anomaly_score
            FROM read_csv('{folder / "observations.csv.gz"}', delim='\t', header=true,
                          types={{'observed_on': 'DATE', 'positional_accuracy': 'DOUBLE',
                                  'anomaly_score': 'DOUBLE', 'taxon_id': 'BIGINT',
                                  'latitude': 'DOUBLE', 'longitude': 'DOUBLE'}}) AS o
            LEFT JOIN read_csv('{folder / "taxa.csv.gz"}', delim='\t', header=true,
                               types={{'taxon_id': 'BIGINT'}}) AS t USING (taxon_id)
            ORDER BY floor(o.latitude), floor(o.longitude)
        ) TO '{partial}' (FORMAT parquet, COMPRESSION zstd, ROW_GROUP_SIZE 1000000)
        """
    )
    partial.replace(out)  # only a finished file takes the final name
    for leftover in tmp.glob("*"):
        leftover.unlink()
    if tmp.exists():
        tmp.rmdir()
    return out


def download_open_data(
    folder: str | Path, *, base_url: str = BUCKET_URL, build: bool = True, memory_limit: str = "8GB"
) -> Path:
    """Download iNaturalist's open-data export and build a local Parquet file.

    Fetches ``observations.csv.gz`` (about 14 GB in late 2026) and
    ``taxa.csv.gz`` from iNaturalist's public open-data bucket into
    ``folder``, resuming a partial download and skipping files already
    complete, then joins them into ``observations.parquet`` (about 12 GB):
    one row per observation with its taxon's name, rank and ancestry,
    grouped by 1° cell so that a query for a region reads little of it.
    Build samples from it with
    [`Sample.from_open_data`][samplesar.Sample.from_open_data].

    The export holds the observations with openly licensed photos, not all
    of iNaturalist. Obscured observations carry their public, randomised
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
    for name in FILES:
        _download(f"{base_url.rstrip('/')}/{name}", folder / name)
    if not build:
        return folder
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
    ``lon``, ``lat``, with infraspecific taxa named as their species."""
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
        where.append(f"quality_grade IN ({', '.join('?' * len(grades))})")
        params += grades
    if ranks is not None:
        ranks = list(ranks)
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
            longitude AS lon, latitude AS lat
        FROM read_parquet(?)
        WHERE {" AND ".join(where)}
    """
    return duckdb.execute(query, [str(path), *params]).df()
