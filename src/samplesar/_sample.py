"""Point observations of species, the input every nesting strategy starts from."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
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

    Each row of ``data`` is one observation with columns ``species``, ``lon``,
    and ``lat`` (decimal degrees). Several observations may share a point.

    Build one with :meth:`from_df` or :meth:`from_csv`.
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
        """Build a sample from a DataFrame, renaming columns as needed."""
        renamed = df.rename(columns={species_col: "species", lon_col: "lon", lat_col: "lat"})
        return cls(_validate(renamed))

    @classmethod
    def from_csv(cls, path: str | Path, **kwargs: str) -> Sample:
        """Build a sample from a CSV file; ``kwargs`` are the column names for :meth:`from_df`."""
        return cls.from_df(pd.read_csv(path), **kwargs)

    @property
    def n_observations(self) -> int:
        return len(self.data)

    @property
    def n_species(self) -> int:
        return int(self.data["species"].nunique())

    @property
    def points(self) -> np.ndarray:
        """Distinct (lon, lat) points, shape (n_points, 2), in first-seen order."""
        return self.data[["lon", "lat"]].drop_duplicates().to_numpy(dtype=float)

    @property
    def n_points(self) -> int:
        return len(self.points)

    @property
    def area(self) -> float:
        """Area of the convex hull of the points, in square degrees.

        Zero when fewer than three distinct, non-collinear points remain.
        Square degrees are not a physical area; they are fine for ordering
        nested samples and for fitting within one sample, which is what the
        nesting strategies need. A projected area is planned.
        """
        return hull_area(self.points)

    def subset(self, point_indices: Sequence[int] | np.ndarray) -> Sample:
        """The observations at the given rows of :attr:`points`."""
        keep = self.points[list(point_indices)]
        mask = pd.MultiIndex.from_frame(self.data[["lon", "lat"]]).isin([tuple(p) for p in keep])
        return Sample(self.data.loc[mask].reset_index(drop=True))

    def __repr__(self) -> str:
        return (
            f"Sample(n_observations={self.n_observations}, n_points={self.n_points}, "
            f"n_species={self.n_species}, area={self.area:.6g})"
        )


def hull_area(points: np.ndarray) -> float:
    """Convex hull area of an (n, 2) array of (lon, lat) points, in square degrees."""
    pts = np.asarray(points, dtype=float)
    if len(pts) < 3:
        return 0.0
    return float(MultiPoint([tuple(p) for p in pts]).convex_hull.area)
