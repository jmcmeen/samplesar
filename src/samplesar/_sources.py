"""Observation sources; see samplesar.sources."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from samplesar._sample import Sample, _parse_crs, _validate


def dataframe(
    df: pd.DataFrame,
    species_col: str = "species",
    lon_col: str = "lon",
    lat_col: str = "lat",
    crs: Any = None,
) -> Sample:
    """Build a sample from a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        One row per observation. Other columns are dropped.
    species_col : str
        The column holding the species name.
    lon_col : str
        The column holding the longitude in decimal degrees, or the x
        (easting) coordinate when ``crs`` is projected.
    lat_col : str
        The column holding the latitude in decimal degrees, or the y
        (northing) coordinate when ``crs`` is projected.
    crs : str, int or pyproj.CRS, optional
        The coordinates' reference system, as pyproj accepts it
        (``"EPSG:32617"``, ``32617``, a WKT string). By default WGS84
        longitude and latitude. Coordinates in another geographic system
        are converted to WGS84. Coordinates in a projected system are
        converted to WGS84 for ``data``, and areas and distances are
        measured in that system, so a grid laid out in it stays straight.

    Returns
    -------
    Sample

    Raises
    ------
    KeyError
        If a column is missing.
    ValueError
        If there are no rows, a species is missing, a coordinate is
        missing, not numeric or out of range, or ``crs`` isn't a
        coordinate reference system.
    """
    renamed = df.rename(columns={species_col: "species", lon_col: "lon", lat_col: "lat"})
    parsed = _parse_crs(crs)
    projected = parsed if parsed is not None and parsed.is_projected else None
    return Sample(_validate(renamed, parsed), crs=projected)


def geodataframe(gdf: Any, species_col: str = "species") -> Sample:
    """Build a sample from a GeoDataFrame of point observations.

    The geometries' coordinates and the frame's CRS are read directly, so
    geopandas isn't a requirement of samplesar; the frame just needs a
    point ``geometry`` and a ``crs``.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        One point per observation. A frame without a CRS is taken to be
        in WGS84 longitude and latitude.
    species_col : str
        The column holding the species name.

    Returns
    -------
    Sample

    Raises
    ------
    ValueError
        If a geometry isn't a point, or as [`sources.dataframe`][samplesar.sources.dataframe].
    KeyError
        If the species column is missing.
    """
    geometry = gdf.geometry
    if (geometry.isna() | geometry.is_empty).any():
        raise ValueError("Some geometries are missing or empty; every row needs a point.")
    kinds = set(geometry.geom_type)
    if kinds - {"Point"}:
        raise ValueError(f"Every geometry must be a point; found {', '.join(sorted(kinds))}.")
    if species_col not in gdf.columns:
        raise KeyError(f"Missing column {species_col!r}.")
    frame = pd.DataFrame(
        {
            "species": np.asarray(gdf[species_col]),
            "lon": np.asarray(geometry.x),
            "lat": np.asarray(geometry.y),
        }
    )
    return dataframe(frame, crs=gdf.crs)


def csv(path: str | Path, **kwargs: str) -> Sample:
    """Build a sample from a CSV file.

    Parameters
    ----------
    path : str or Path
        The CSV file, one row per observation.
    **kwargs : str
        ``species_col``, ``lon_col`` and ``lat_col``, as in
        [`sources.dataframe`][samplesar.sources.dataframe]. Nothing else is passed to
        the CSV reader.

    Returns
    -------
    Sample

    Raises
    ------
    KeyError
        If a column is missing.
    ValueError
        As in [`sources.dataframe`][samplesar.sources.dataframe].
    TypeError
        If a keyword other than the three column names is given.
    """
    return dataframe(pd.read_csv(path), **kwargs)


def inaturalist(
    observations: Iterable[Mapping[str, Any]] | None = None,
    *,
    ranks: Iterable[str] | None = ("species", "subspecies", "variety", "form"),
    skip_obscured: bool = True,
    max_results: int | None = None,
    **query: Any,
) -> Sample:
    """Build a sample from iNaturalist observations.

    Either pass observation records already fetched (the ``results`` of
    pyinaturalist's ``get_observations``, or the API's JSON), or give a
    query and they're fetched through pyinaturalist (``pip install
    'samplesar[inat]'``), paging past the API's 10,000-result limit.

    Parameters
    ----------
    observations : iterable of dict, optional
        Observation records. If omitted, ``query`` is fetched.
    ranks : iterable of str or None
        Keep only taxa at these ranks; by default species and below, so
        a genus-level record doesn't count as a species. A subspecies,
        variety or form counts as its species. ``None`` keeps every rank.
    skip_obscured : bool
        Drop observations whose coordinates iNaturalist has obscured
        (randomised within a cell about 0.2° across), which would distort
        any hull.
    max_results : int, optional
        Stop fetching after this many records (only with a query).
    **query
        Search parameters for ``get_observations``, such as
        ``place_id``, ``taxon_id``, ``user_id`` or ``quality_grade``.

    Returns
    -------
    Sample

    Raises
    ------
    ValueError
        If no usable observations remain, both records and a query (or
        ``max_results``) are given, or ``max_results`` or ``per_page`` is
        below 1.
    ImportError
        If a query needs fetching and pyinaturalist isn't installed.

    Examples
    --------
    >>> sample = samplesar.sources.inaturalist(  # doctest: +SKIP
    ...     place_id=130630, taxon_id=20978, quality_grade="research"
    ... )
    """
    from samplesar._inat import fetch_inaturalist, inaturalist_frame

    if observations is None:
        observations = fetch_inaturalist(max_results=max_results, **query)
    elif query or max_results is not None:
        raise ValueError("Give observations or a query (and max_results), not both.")
    frame = inaturalist_frame(observations, ranks=ranks, skip_obscured=skip_obscured)
    if frame.empty:
        raise ValueError(
            "No usable iNaturalist observations (see the filters on ranks and obscured)."
        )
    return dataframe(frame)


def open_data(
    path: str | Path,
    *,
    bbox: tuple[float, float, float, float] | None = None,
    taxon_id: int | None = None,
    quality_grade: str | Iterable[str] | None = "research",
    ranks: Iterable[str] | None = ("species", "subspecies", "variety", "form"),
    since: date | str | None = None,
    until: date | str | None = None,
    max_accuracy: float | None = None,
    boundary: Any = None,
) -> Sample:
    """Build a sample from a local copy of iNaturalist's open data.

    Queries the Parquet file [`sources.download_open_data`][samplesar.sources.download_open_data]
    builds, with DuckDB (``pip install 'samplesar[opendata]'``), so a
    region or taxon comes back in moments without the API.

    Parameters
    ----------
    path : str or Path
        ``observations.parquet``.
    bbox : tuple of float, optional
        (west, south, east, north) in degrees; west greater than east
        crosses the antimeridian.
    taxon_id : int, optional
        An iNaturalist taxon: it and everything under it (by ancestry),
        so ``20979`` is every frog.
    quality_grade : str, iterable of str or None
        By default research grade only; ``None`` keeps every grade.
    ranks : iterable of str or None
        By default species and below, a subspecies, variety or form
        counting as its species; ``None`` keeps every rank.
    since, until : date or str, optional
        Observed on or after, and on or before, these dates (ISO).
    max_accuracy : float, optional
        Keep observations whose positional accuracy is given and at most
        this many metres. The export doesn't mark obscured observations,
        whose public coordinates are randomised; this is the filter
        available.
    boundary : polygon, optional
        Keep only observations inside this region (or on its edge): a
        shapely polygon or multipolygon in WGS84, or a GeoDataFrame or
        GeoSeries in any CRS, such as
        [`regions.admin`][samplesar.regions.admin] returns. Its bounding
        box narrows what's read, and ``bbox`` narrows it further.

    Returns
    -------
    Sample

    Raises
    ------
    ValueError
        If no observations match, ``bbox`` isn't (west, south, east,
        north), ``quality_grade`` or ``ranks`` is empty, or ``boundary``
        isn't polygonal.
    ImportError
        If DuckDB isn't installed.

    Examples
    --------
    >>> frogs = samplesar.sources.open_data(  # doctest: +SKIP
    ...     "observations.parquet", bbox=(-85.0, 35.0, -81.5, 37.0), taxon_id=20979
    ... )
    """
    from samplesar._boundary import as_geometry, clip_bbox, inside
    from samplesar._opendata import open_data_frame

    region = None if boundary is None else as_geometry(boundary)
    if region is not None:
        bbox = clip_bbox(region, bbox)
    frame = open_data_frame(
        path,
        bbox=bbox,
        taxon_id=taxon_id,
        quality_grade=quality_grade,
        ranks=ranks,
        since=since,
        until=until,
        max_accuracy=max_accuracy,
    )
    if region is not None and not frame.empty:
        frame = frame[inside(region, frame["lon"], frame["lat"])].reset_index(drop=True)
    if frame.empty:
        raise ValueError("No observations in the open data match these filters.")
    return dataframe(frame)


def gbif(
    records: Iterable[Mapping[str, Any]] | None = None,
    *,
    ranks: Iterable[str] | None = ("SPECIES", "SUBSPECIES", "VARIETY", "FORM"),
    max_uncertainty: float | None = None,
    boundary: Any = None,
    max_results: int | None = None,
    **query: Any,
) -> Sample:
    """Build a sample from GBIF occurrences.

    GBIF gathers occurrences from museums, surveys, eBird, iNaturalist and
    many other sources. Either pass records already fetched (the ``results``
    of pygbif's ``occurrences.search``, or the API's JSON), or give a query
    and they're fetched through pygbif (``pip install 'samplesar[gbif]'``),
    a page at a time, up to the search API's 100,000-record limit (a warning
    says when a query matches more: narrow it, or use GBIF's download
    service). A query asks by default for records with coordinates, without
    geospatial issues and marked present.

    Parameters
    ----------
    records : iterable of dict, optional
        Occurrence records. If omitted, ``query`` is fetched. Records marked
        absent or carrying a geospatial issue are dropped, as a query
        leaves them out.
    ranks : iterable of str or None
        Keep only records identified at these ranks; by default species and
        below, named by their species (GBIF's ``species`` field). ``None``
        keeps every rank.
    max_uncertainty : float, optional
        Keep records whose coordinate uncertainty is given and at most this
        many metres.
    boundary : polygon, optional
        Keep only records inside this region (as [`Sample.within`][samplesar.Sample.within]
        takes); a query is narrowed to its bounding box (only by latitude if
        the region crosses the antimeridian). ``max_results`` counts records
        fetched, before this filter.
    max_results : int, optional
        Stop fetching after this many records (only with a query).
    **query
        Search parameters for GBIF's occurrence search, such as
        ``taxonKey``, ``country`` (ISO 3166-1 alpha-2), ``year``,
        ``basisOfRecord`` or ``datasetKey``.

    Returns
    -------
    Sample

    Raises
    ------
    ValueError
        If no usable records remain, both records and a query (or
        ``max_results``) are given, or ``max_results`` is below 1.
    ImportError
        If a query needs fetching and pygbif isn't installed.

    Examples
    --------
    >>> frogs = samplesar.sources.gbif(taxonKey=952, country="CR")  # doctest: +SKIP
    """
    from samplesar._boundary import as_geometry, inside
    from samplesar._gbif import fetch_gbif, gbif_frame

    region = None if boundary is None else as_geometry(boundary)
    if records is None:
        if region is not None:
            west, south, east, north = region.bounds
            if east - west <= 180:  # a region split at the antimeridian spans the globe
                query = {"decimalLongitude": f"{west},{east}", **query}
            query = {"decimalLatitude": f"{south},{north}", **query}
        records = fetch_gbif(max_results=max_results, **query)
    elif query or max_results is not None:
        raise ValueError("Give records or a query (and max_results), not both.")
    frame = gbif_frame(records, ranks=ranks, max_uncertainty=max_uncertainty)
    if region is not None and not frame.empty:
        frame = frame[inside(region, frame["lon"], frame["lat"])].reset_index(drop=True)
    if frame.empty:
        raise ValueError(
            "No usable GBIF records (see the filters on ranks, uncertainty and boundary)."
        )
    return dataframe(frame)
