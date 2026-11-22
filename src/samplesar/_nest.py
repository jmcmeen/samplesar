"""Nested samples: a sequence of ever-smaller subsets of one sample."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd

from samplesar._registry import register_strategy
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
    strategy : str
        The registered strategy that made the nest, such as ``"hull_area"``;
        empty for a nest built directly.
    options : mapping
        The options the strategy was called with, as given (defaults left
        out).
    """

    sample: Sample
    steps: tuple[np.ndarray, ...]
    strategy: str = ""
    options: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    @property
    def label(self) -> str:
        """The strategy and its options, such as ``hull_area(keep='smallest')``.

        ``"nest"`` for a nest built directly rather than by a strategy.
        """
        name = self.strategy or "nest"
        if not self.options:
            return name
        return f"{name}({', '.join(f'{k}={v!r}' for k, v in self.options.items())})"

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

    def fit(self, model: str = "power", **kwargs: Any) -> Any:
        """Fit one of sars' models to the nest's species-area data.

        The areas are convex hull areas in square degrees, so a parameter
        scaled by area (``c`` of the power model, for example) is in those
        units.

        Parameters
        ----------
        model : str
            A sars model: ``"power"`` calls ``sars.sar_power``, ``"loga"``
            ``sars.sar_loga``, and so on.
        **kwargs
            Passed to the sars function.

        Returns
        -------
        object
            sars' fit (``SARFit`` for a single model).

        Raises
        ------
        ValueError
            If sars has no such model, or the nest has fewer than three steps
            with a positive area to fit.
        """
        from samplesar._evaluate import fit_nest

        return fit_nest(self, model, **kwargs)


def check_size(sample: Sample, min_points: int) -> None:
    """Raise ValueError unless ``1 <= min_points <= sample.n_points``."""
    if min_points < 1:
        raise ValueError(f"min_points must be at least 1, got {min_points}")
    if sample.n_points < min_points:
        raise ValueError(f"Sample has {sample.n_points} points; need at least {min_points}")


@register_strategy("random")
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
        If ``min_points`` is below 1 or the sample has fewer points.
    """
    check_size(sample, min_points)
    n = sample.n_points
    rng = np.random.default_rng(seed)
    remaining = np.arange(n)
    steps = [remaining.copy()]
    while len(remaining) > min_points:
        drop = rng.integers(len(remaining))
        remaining = np.delete(remaining, drop)
        steps.append(remaining.copy())
    return Nest(sample=sample, steps=tuple(steps))
