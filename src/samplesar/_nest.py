"""Nested samples: a sequence of ever-smaller subsets of one sample."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from samplesar._registry import Options, register_strategy
from samplesar._sample import Sample, hull_area, point_species


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
        out), read-only.
    """

    sample: Sample
    steps: tuple[np.ndarray, ...]
    strategy: str = ""
    options: Mapping[str, Any] = field(default_factory=Options)

    @property
    def label(self) -> str:
        """The strategy and its options, such as ``hull_area(keep='smallest')``.

        ``"nest"`` for a nest built directly rather than by a strategy.
        """
        name = self.strategy or "nest"
        if not self.options:
            return name
        shown = (
            f"{k}={v.item() if isinstance(v, np.generic) else v!r}" for k, v in self.options.items()
        )
        return f"{name}({', '.join(shown)})"

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
        species, n_species = point_species(self.sample)
        # How many of the step's points hold each species, updated by the
        # points each step drops (or adds), not recounted.
        holders = np.zeros(n_species, dtype=int)
        previous = np.empty(0, dtype=int)
        rows = []
        for i, idx in enumerate(self.steps):
            for change, sign in (
                (np.setdiff1d(previous, idx), -1),
                (np.setdiff1d(idx, previous), 1),
            ):
                if len(change):
                    np.add.at(holders, np.concatenate([species[p] for p in change]), sign)
            previous = np.asarray(idx)
            rows.append(
                {
                    "step": i,
                    "n_points": len(idx),
                    "area": hull_area(points[idx]),
                    "species": int(np.count_nonzero(holders)),
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


def nest_from_order(sample: Sample, order: Sequence[int] | np.ndarray, min_points: int = 3) -> Nest:
    """Nest a sample by removing its points in a given order.

    Any ordering, from a heuristic, an optimiser or an experiment's design,
    becomes a [`Nest`][samplesar.Nest] that the rest of the package can
    evaluate. One point goes per step, in ``order``, until ``min_points``
    remain or the order runs out; the order needn't name every point.

    Parameters
    ----------
    sample : Sample
        The sample to nest.
    order : sequence of int or np.ndarray
        Rows of ``sample.points``, in the order to remove them.
    min_points : int
        Stop when this many points remain, even if the order goes on.

    Returns
    -------
    Nest
        With ``strategy`` ``"order"``.

    Raises
    ------
    ValueError
        If ``order`` repeats a point or names one outside the sample, isn't
        a sequence of whole numbers, or ``min_points`` is below 1 or above the
        sample's size.
    """
    check_size(sample, min_points)
    order = np.asarray(order)
    if order.ndim != 1 or (order.size and not np.issubdtype(order.dtype, np.integer)):
        raise ValueError("order must be a one-dimensional sequence of point indices.")
    n = sample.n_points
    if order.size and (order.min() < 0 or order.max() >= n):
        raise ValueError(f"order names a point outside the sample's {n} points.")
    if len(np.unique(order)) < len(order):
        raise ValueError("order repeats a point.")
    remaining = np.arange(n)
    steps = [remaining.copy()]
    for point in order[: n - min_points]:
        remaining = remaining[remaining != point]
        steps.append(remaining.copy())
    return Nest(sample=sample, steps=tuple(steps), strategy="order")
