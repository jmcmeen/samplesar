"""Nested samples: a sequence of ever-smaller subsets of one sample."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
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
        return f"{name}({', '.join(f'{k}={_short(v)}' for k, v in self.options.items())})"

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

    def to_dict(self) -> dict[str, Any]:
        """The nest as plain data, ready for JSON.

        Holds the observations, the strategy and its options, and the steps,
        stored as the points each step removes rather than in full. Options
        that are tuples or arrays are stored as lists.

        Returns
        -------
        dict

        Raises
        ------
        ValueError
            If the steps aren't nested (only a nest built by hand can be
            like that), or an option isn't a plain value JSON can hold (a
            ``Generator`` seed, say).
        """
        removed = []
        for before, after in zip(self.steps, self.steps[1:], strict=False):
            if not np.isin(after, before).all():
                raise ValueError("Only a nest whose steps are nested can be saved.")
            removed.append(np.setdiff1d(before, after).tolist())
        data = self.sample.data
        return {
            "format": FORMAT,
            "version": FORMAT_VERSION,
            "samplesar": _package_version(),
            "strategy": self.strategy,
            "options": {k: _plain(k, v) for k, v in self.options.items()},
            "observations": {
                "species": data["species"].tolist(),
                "lon": data["lon"].tolist(),
                "lat": data["lat"].tolist(),
            },
            "first": np.asarray(self.steps[0]).tolist(),
            "removed": removed,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Nest:
        """Rebuild a nest from [`to_dict`][samplesar.Nest.to_dict]'s output.

        Parameters
        ----------
        data : mapping
            As written by ``to_dict`` (or read back from its JSON).

        Returns
        -------
        Nest
            Each step's points in the order of the first step.

        Raises
        ------
        ValueError
            If the data isn't a saved nest, is from a newer format version,
            or its steps don't fit its sample.
        """
        missing = {"format", "version", "observations", "first", "removed"} - set(data)
        if data.get("format") != FORMAT or missing:
            raise ValueError("Not a saved samplesar nest.")
        version = data["version"]
        if not isinstance(version, int) or isinstance(version, bool):
            raise ValueError(f"The saved format version must be a whole number, got {version!r}.")
        if version > FORMAT_VERSION:
            raise ValueError(
                f"Saved with nest format {version}; this samplesar reads up to "
                f"{FORMAT_VERSION}. Upgrade samplesar to load it."
            )
        sample = Sample.from_df(pd.DataFrame(data["observations"]))
        remaining = _indices(data["first"])
        if remaining.size and (remaining.min() < 0 or remaining.max() >= sample.n_points):
            raise ValueError("The saved steps name points outside the saved sample.")
        if len(np.unique(remaining)) < len(remaining):
            raise ValueError("The saved first step repeats a point.")
        steps = [remaining]
        for gone in data["removed"]:
            gone = _indices(gone)
            if not np.isin(gone, remaining).all():
                raise ValueError("A saved step removes a point that isn't there.")
            remaining = remaining[~np.isin(remaining, gone)]
            steps.append(remaining)
        return cls(
            sample=sample,
            steps=tuple(steps),
            strategy=data.get("strategy", ""),
            options=Options(data.get("options", {})),
        )

    def to_json(self, path: str | Path) -> None:
        """Save the nest to a JSON file (see [`to_dict`][samplesar.Nest.to_dict]).

        Parameters
        ----------
        path : str or Path
            The file to write.
        """
        Path(path).write_text(json.dumps(self.to_dict()))

    @classmethod
    def from_json(cls, path: str | Path) -> Nest:
        """Load a nest saved with [`to_json`][samplesar.Nest.to_json].

        Parameters
        ----------
        path : str or Path
            The file to read.

        Returns
        -------
        Nest

        Raises
        ------
        ValueError
            As [`from_dict`][samplesar.Nest.from_dict].
        """
        return cls.from_dict(json.loads(Path(path).read_text()))

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


def _short(value: Any) -> str:
    """An option value for a label: numpy scalars as plain numbers, and long
    sequences (a removal order, say) abbreviated with their length."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, np.ndarray | list | tuple) and len(value) > 6:
        head = ", ".join(
            repr(v.item() if isinstance(v, np.generic) else v) for v in list(value)[:3]
        )
        return f"[{head}, ... ({len(value)})]"
    return repr(value)


def check_size(sample: Sample, min_points: int) -> None:
    """Raise ValueError unless ``1 <= min_points <= sample.n_points``."""
    if min_points < 1:
        raise ValueError(f"min_points must be at least 1, got {min_points}")
    if sample.n_points < min_points:
        raise ValueError(f"Sample has {sample.n_points} points; need at least {min_points}")


FORMAT = "samplesar.nest"
FORMAT_VERSION = 1


def _package_version() -> str:
    from samplesar import __version__

    return __version__


def _indices(values: Any) -> np.ndarray:
    """Saved point indices as an int array; anything else isn't a saved nest."""
    array = np.asarray(values)
    if array.size == 0:
        return np.empty(0, dtype=int)
    if array.ndim != 1 or not np.issubdtype(array.dtype, np.integer):
        raise ValueError("The saved steps must be lists of whole-number point indices.")
    return array.astype(int)


def _plain(name: str, value: Any) -> Any:
    """An option value as plain JSON data (tuples and arrays become lists),
    or a ValueError naming the option if it can't be stored."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray | tuple | list):
        return [_plain(name, v) for v in value]
    if isinstance(value, Mapping):
        return {str(k): _plain(name, v) for k, v in value.items()}
    if value is None or isinstance(value, bool | int | float | str):
        return value
    raise ValueError(
        f"Option {name}={value!r} can't be saved as JSON; pass a plain value (a seed as an "
        "int, not a Generator) to save the nest."
    )


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


@register_strategy("order")
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
        Rows of ``sample.points``, in the order to remove them. Floats are
        accepted if they're whole numbers.
    min_points : int
        Stop when this many points remain, even if the order goes on.

    Returns
    -------
    Nest
        Registered as ``"order"``, so ``samplesar.nest(sample, "order",
        order=...)`` is the same.

    Raises
    ------
    ValueError
        If ``order`` repeats a point or names one outside the sample, isn't
        a sequence of whole numbers, or ``min_points`` is below 1 or above the
        sample's size.
    """
    check_size(sample, min_points)
    order = np.asarray(order)
    if order.ndim != 1:
        raise ValueError("order must be a one-dimensional sequence of point indices.")
    if order.size and not np.issubdtype(order.dtype, np.integer):
        if not np.issubdtype(order.dtype, np.floating) or not np.all(order == np.round(order)):
            raise ValueError("order must hold whole numbers: indices of the sample's points.")
        order = order.astype(int)
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
    return Nest(sample=sample, steps=tuple(steps))
