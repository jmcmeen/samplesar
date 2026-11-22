"""Evaluating sampling designs by fitting their species-area data with sars."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd
import sars

from samplesar._nest import Nest


def fit_nest(nest: Nest, model: str = "power", **kwargs: Any) -> Any:
    """Fit one of sars' models to a nest's species-area data.

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
        If sars has no such model.
    """
    func = getattr(sars, f"sar_{model}", None)
    if not callable(func):
        raise ValueError(f"sars has no model {model!r} (no sars.sar_{model})")
    return func(nest.sar_data(), **kwargs)


def compare_nests(nests: Mapping[str, Nest], model: str = "power") -> pd.DataFrame:
    """Fit the same sars model to several nests and compare the fits.

    Use it to compare sampling strategies on one sample: how steep each one
    makes the curve, and how well the model fits.

    Parameters
    ----------
    nests : mapping of str to Nest
        The nests to compare, by name (for example, one per strategy).
    model : str
        A sars model with one fit per dataset, such as ``"power"`` or
        ``"loga"``.

    Returns
    -------
    pd.DataFrame
        One row per nest, indexed by name: ``n_steps``, the model's
        parameters (``c`` and ``z`` for the power model), ``r_squared``,
        ``aicc`` and ``converged``.

    Raises
    ------
    ValueError
        If sars has no such model, or its fit has no parameters to compare
        (a multi-model average, for example).

    Examples
    --------
    >>> nests = {  # doctest: +SKIP
    ...     "random": samplesar.random_nest(sample, seed=0),
    ...     "peel": samplesar.hull_peel_nest(sample),
    ... }
    >>> samplesar.compare_nests(nests)  # doctest: +SKIP
    """
    rows = []
    for name, nest in nests.items():
        fit = fit_nest(nest, model)
        if not hasattr(fit, "params"):
            raise ValueError(
                f"sars.sar_{model} returns {type(fit).__name__}, not a single fit with "
                "parameters; choose a single model such as 'power'."
            )
        rows.append(
            {
                "nest": name,
                "n_steps": nest.n_steps,
                **fit.params,
                "r_squared": fit.r_squared,
                "aicc": fit.aicc,
                "converged": fit.converged,
            }
        )
    return pd.DataFrame(rows).set_index("nest")
