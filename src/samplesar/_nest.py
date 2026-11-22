"""Nested samples: a sequence of ever-smaller subsets of one sample."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from samplesar._sample import Sample, hull_area


@dataclass(frozen=True)
class Nest:
    """A nested sequence of sub-samples of one [`Sample`][samplesar.Sample].

    Made by a nesting strategy such as [`random_nest`][samplesar.random_nest]. Strategies differ only
    in which points they drop at each step.

    Attributes
    ----------
    sample : Sample
        The full sample.
    steps : tuple of np.ndarray
        ``steps[i]`` holds the rows of ``sample.points`` that remain at step
        ``i``. ``steps[0]`` is every point and each later step is a subset of
        the one before.
    """

    sample: Sample
    steps: tuple[np.ndarray, ...]

    @property
    def n_steps(self) -> int:
        """The number of steps, including the full sample."""
        return len(self.steps)

    def sample_at(self, step: int) -> Sample:
        """The observations remaining at a step, as a [`Sample`][samplesar.Sample]."""
        return self.sample.subset(self.steps[step])

    def to_frame(self) -> pd.DataFrame:
        """The nest as a table.

        Returns
        -------
        pd.DataFrame
            One row per step: ``step``, ``n_points``, ``area`` (convex hull,
            square degrees) and ``species`` (distinct species remaining).
        """
        points = self.sample.points
        species = self.sample.data["species"].to_numpy()
        keys = pd.MultiIndex.from_frame(self.sample.data[["lon", "lat"]])
        rows = []
        for i, idx in enumerate(self.steps):
            kept = points[idx]
            mask = keys.isin([tuple(p) for p in kept])
            rows.append(
                {
                    "step": i,
                    "n_points": len(idx),
                    "area": hull_area(kept),
                    "species": int(len(np.unique(species[mask]))),
                }
            )
        return pd.DataFrame(rows)

    def sar_data(self) -> pd.DataFrame:
        """The species-area data sars fits.

        Returns
        -------
        pd.DataFrame
            ``area`` (convex hull, square degrees) and ``species`` columns,
            one row per step, without the zero-area steps at the end.
        """
        frame = self.to_frame()
        return frame.loc[frame["area"] > 0, ["area", "species"]].reset_index(drop=True)


def random_nest(sample: Sample, min_points: int = 3, seed: int | None = None) -> Nest:
    """Nest a sample by removing one point at a time, uniformly at random.

    The baseline every other strategy is compared against.

    Parameters
    ----------
    sample : Sample
        The sample to nest.
    min_points : int
        Stop when this many points remain.
    seed : int, optional
        Seed for the random choices, for a reproducible nest.

    Returns
    -------
    Nest

    Raises
    ------
    ValueError
        If the sample has fewer than ``min_points`` points.
    """
    n = sample.n_points
    if n < min_points:
        raise ValueError(f"Sample has {n} points; need at least {min_points}")
    rng = np.random.default_rng(seed)
    remaining = np.arange(n)
    steps = [remaining.copy()]
    while len(remaining) > min_points:
        drop = rng.integers(len(remaining))
        remaining = np.delete(remaining, drop)
        steps.append(remaining.copy())
    return Nest(sample=sample, steps=tuple(steps))
