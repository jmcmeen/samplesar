"""Evaluating sampling designs by fitting their species-area data with sars."""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Mapping
from typing import Any

import pandas as pd
import sars

from samplesar._nest import Nest

MIN_FIT_POINTS = 3


def fit_nest(nest: Nest, model: str = "power", name: str | None = None, **kwargs: Any) -> Any:
    """Fit one of sars' models to a nest's species-area data.

    The areas are convex hull areas in km², so a parameter scaled by area
    (``c`` of the power model, for example) is per km².

    Parameters
    ----------
    nest : Nest
        The nest to fit.
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
    func = getattr(sars, f"sar_{model}", None)
    if not callable(func):
        raise ValueError(f"sars has no model {model!r} (no sars.sar_{model})")
    data = nest.sar_data()
    if len(data) < MIN_FIT_POINTS:
        which = f"Nest {name!r}" if name is not None else "The nest"
        raise ValueError(
            f"{which} has {len(data)} step(s) with a positive area; a fit needs at least "
            f"{MIN_FIT_POINTS}."
        )
    return func(data, **kwargs)


def compare_nests(
    nests: Mapping[str, Nest] | Iterable[Nest], model: str = "power", **kwargs: Any
) -> pd.DataFrame:
    """Fit the same sars model to several nests and compare the fits.

    Use it to compare sampling strategies on one sample: how steep each one
    makes the curve, and how well the model fits. The areas are convex hull
    areas in km².

    Parameters
    ----------
    nests : mapping of str to Nest, or sequence of Nest
        The nests to compare: by name, or a sequence named by each nest's
        [`label`][samplesar.Nest.label], such as ``hull_area(keep='smallest')``.
        Repeated labels (two unseeded random nests, say) get ``#2``, ``#3``.
    model : str
        A sars model with one fit per dataset, such as ``"power"`` or
        ``"loga"``.
    **kwargs
        Passed to the sars function for every nest.

    Returns
    -------
    pd.DataFrame
        One row per nest, indexed by name: ``n_steps``, ``n_fitted`` (the
        steps with a positive area, which are fitted), the model's
        parameters (``c`` and ``z`` for the power model), ``r_squared``,
        ``aicc``, ``converged`` and ``at_bounds``: the parameters that
        stopped on a bound of sars' search, whose values are the bound's and
        not estimates (empty when none; missing values with a sars too old
        to tell).

    Warns
    -----
    UserWarning
        If a fit's parameter stopped on a bound, naming the nests.

    Raises
    ------
    ValueError
        If ``nests`` is empty, sars has no such model, a nest has too few
        steps to fit (the message names it), or the fit has no parameters to
        compare (a multi-model average, for example).

    Examples
    --------
    >>> nests = {  # doctest: +SKIP
    ...     "random": samplesar.random_nest(sample, seed=0),
    ...     "peel": samplesar.hull_peel_nest(sample),
    ... }
    >>> samplesar.compare_nests(nests)  # doctest: +SKIP
    """
    if not isinstance(nests, Mapping):
        named: dict[str, Nest] = {}
        seen: dict[str, int] = {}
        for nest in nests:
            seen[nest.label] = seen.get(nest.label, 0) + 1
            count = seen[nest.label]
            named[nest.label if count == 1 else f"{nest.label} #{count}"] = nest
        nests = named
    if not nests:
        raise ValueError("No nests to compare.")
    rows = []
    for name, nest in nests.items():
        fit = fit_nest(nest, model, name=name, **kwargs)
        if not hasattr(fit, "params"):
            raise ValueError(
                f"sars.sar_{model} returns {type(fit).__name__}, not a single fit with "
                "parameters; choose a single model such as 'power'."
            )
        rows.append(
            {
                "nest": name,
                "n_steps": nest.n_steps,
                "n_fitted": len(nest.sar_data()),
                **fit.params,
                "r_squared": fit.r_squared,
                "aicc": fit.aicc,
                "converged": fit.converged,
                "at_bounds": _at_bounds(fit),
            }
        )
    table = pd.DataFrame(rows).set_index("nest")
    bounded = [f"{name} ({', '.join(b)})" for name, b in table["at_bounds"].items() if b]
    if bounded:
        warnings.warn(
            f"Parameters stopped on a bound of the fit, so their values are the bound's, "
            f"not estimates: {'; '.join(bounded)}.",
            stacklevel=2,
        )
    return table


def _at_bounds(fit: Any) -> tuple[str, ...] | None:
    """The fit's parameters on a bound (sars' ``SARFit.at_bounds``), or None
    with a sars that can't tell."""
    names = getattr(fit, "at_bounds", None)
    return None if names is None else tuple(names)
