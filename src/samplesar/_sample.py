"""Point observations of species, the input every nesting strategy starts from."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd
from shapely.geometry import MultiPoint

COLUMNS = ("species", "lon", "lat")


def _validate(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"Missing column(s): {missing}. Need {list(COLUMNS)}.")
    out = df.loc[:, list(COLUMNS)].reset_index(drop=True)
    if len(out) == 0:
        raise ValueError("Sample has no observations")
    if out["species"].isna().any():
        raise ValueError("species contains missing values")
    lon = pd.to_numeric(out["lon"], errors="coerce")
    lat = pd.to_numeric(out["lat"], errors="coerce")
    if lon.isna().any() or lat.isna().any():
        raise ValueError("lon and lat must be numeric with no missing values")
    if ((lat < -90) | (lat > 90)).any():
        raise ValueError("lat must be within [-90, 90]")
    if ((lon < -180) | (lon > 180)).any():
        raise ValueError("lon must be within [-180, 180]")
    out["lon"] = lon.astype(float)
    out["lat"] = lat.astype(float)
    out["species"] = out["species"].astype(str)
    return out


@dataclass(frozen=True)
class Sample:
    """A set of georeferenced species observations.

    Build one with [`from_df`][samplesar.Sample.from_df] or
    [`from_csv`][samplesar.Sample.from_csv] rather than directly: they check
    and normalise the columns.

    Attributes
    ----------
    data : pd.DataFrame
        One row per observation, with columns ``species`` (str), ``lon`` and
        ``lat`` (float, decimal degrees). Several observations may share a
        point. Treat it as read-only: the points and other derived values
        are computed once, so build a new sample rather than editing it.
    """

    data: pd.DataFrame

    @classmethod
    def from_df(
        cls,
        df: pd.DataFrame,
        species_col: str = "species",
        lon_col: str = "lon",
        lat_col: str = "lat",
    ) -> Sample:
        """Build a sample from a DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            One row per observation. Other columns are dropped.
        species_col : str
            The column holding the species name.
        lon_col : str
            The column holding the longitude, in decimal degrees.
        lat_col : str
            The column holding the latitude, in decimal degrees.

        Returns
        -------
        Sample

        Raises
        ------
        KeyError
            If a column is missing.
        ValueError
            If there are no rows, a species is missing, or a coordinate is
            missing, not numeric or out of range.
        """
        renamed = df.rename(columns={species_col: "species", lon_col: "lon", lat_col: "lat"})
        return cls(_validate(renamed))

    @classmethod
    def from_csv(cls, path: str | Path, **kwargs: str) -> Sample:
        """Build a sample from a CSV file.

        Parameters
        ----------
        path : str or Path
            The CSV file, one row per observation.
        **kwargs : str
            ``species_col``, ``lon_col`` and ``lat_col``, as in
            [`from_df`][samplesar.Sample.from_df]. Nothing else is passed to
            the CSV reader.

        Returns
        -------
        Sample

        Raises
        ------
        KeyError
            If a column is missing.
        ValueError
            As in [`from_df`][samplesar.Sample.from_df].
        TypeError
            If a keyword other than the three column names is given.
        """
        return cls.from_df(pd.read_csv(path), **kwargs)

    @property
    def n_observations(self) -> int:
        """The number of observations."""
        return len(self.data)

    @property
    def n_species(self) -> int:
        """The number of distinct species."""
        return int(self.data["species"].nunique())

    @cached_property
    def points(self) -> np.ndarray:
        """Distinct (lon, lat) points, shape (n_points, 2), in first-seen order.

        Computed once and read-only.
        """
        points = self.data[["lon", "lat"]].drop_duplicates().to_numpy(dtype=float)
        points.flags.writeable = False
        return points

    @property
    def n_points(self) -> int:
        """The number of distinct points."""
        return len(self.points)

    @property
    def area(self) -> float:
        """Area of the convex hull of the points, in square degrees.

        Zero when fewer than three distinct, non-collinear points remain.
        Square degrees aren't a physical area; they are fine for ordering
        nested samples and for fitting within one sample, which is what the
        nesting strategies need. A projected area is planned.
        """
        return hull_area(self.points)

    def subset(self, point_indices: Sequence[int] | np.ndarray) -> Sample:
        """The observations at some of the sample's points.

        Parameters
        ----------
        point_indices : sequence of int or np.ndarray
            Rows of [`points`][samplesar.Sample.points] to keep.

        Returns
        -------
        Sample
            Every observation at those points.
        """
        keep = self.points[list(point_indices)]
        mask = pd.MultiIndex.from_frame(self.data[["lon", "lat"]]).isin([tuple(p) for p in keep])
        return Sample(self.data.loc[mask].reset_index(drop=True))

    def __repr__(self) -> str:
        return (
            f"Sample(n_observations={self.n_observations}, n_points={self.n_points}, "
            f"n_species={self.n_species}, area={self.area:.6g})"
        )


def hull_area(points: np.ndarray) -> float:
    """Convex hull area of a set of points, in square degrees.

    Parameters
    ----------
    points : np.ndarray
        Shape (n, 2): longitude and latitude.

    Returns
    -------
    float
        Zero for fewer than three points or collinear points.
    """
    pts = np.asarray(points, dtype=float)
    if len(pts) < 3:
        return 0.0
    return float(MultiPoint(pts).convex_hull.area)


def point_species(sample: Sample) -> tuple[list[np.ndarray], int]:
    """The species codes found at each of the sample's points, and the number
    of species."""
    data = sample.data
    codes, uniques = pd.factorize(data["species"])
    lookup = {(x, y): i for i, (x, y) in enumerate(map(tuple, sample.points))}
    which = np.array([lookup[(x, y)] for x, y in zip(data["lon"], data["lat"], strict=True)])
    grouped = pd.Series(codes).groupby(which).unique()
    return [np.asarray(grouped[i], dtype=int) for i in range(sample.n_points)], len(uniques)
