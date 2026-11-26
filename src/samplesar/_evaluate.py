"""Evaluating sampling designs by fitting their species-area data with sars."""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
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


def compare(
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
        not estimates (empty when none; missing, ``pd.NA``, for a fit
        object that doesn't report it).

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
    ...     "random": samplesar.strategies.random(sample, seed=0),
    ...     "peel": samplesar.strategies.hull_peel(sample),
    ... }
    >>> samplesar.evaluate.compare(nests)  # doctest: +SKIP
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
    bounded = [
        f"{name} ({', '.join(b)})"
        for name, b in table["at_bounds"].items()
        if isinstance(b, tuple) and b
    ]
    if bounded:
        warnings.warn(
            f"Parameters stopped on a bound of the fit, so their values are the bound's, "
            f"not estimates: {'; '.join(bounded)}.",
            stacklevel=2,
        )
    return table


def _at_bounds(fit: Any) -> Any:
    """The fit's parameters on a bound (sars' ``SARFit.at_bounds``), or a
    missing value (``pd.NA``) for a fit object that doesn't report it."""
    names = getattr(fit, "at_bounds", None)
    return pd.NA if names is None else tuple(names)


def baseline(
    sample: Any,
    model: str = "power",
    n: int = 99,
    seed: int = 0,
    strategy: str = "random",
    **options: Any,
) -> pd.DataFrame:
    """Fit a model to many seeded random nests of a sample: a baseline.

    Steps of one nest share most of their points, so one nest's R² and AICc
    aren't independent evidence. Compare a strategy's fit with the spread of
    fits from random nests of the same sample instead (see
    [`evaluate.against_baseline`][samplesar.evaluate.against_baseline]).

    Parameters
    ----------
    sample : Sample
        The sample to nest.
    model : str
        A sars model with one fit per dataset, such as ``"power"``.
    n : int
        How many random nests.
    seed : int
        The first seed; nest ``i`` uses ``seed + i``.
    strategy : str
        A registered strategy that takes a ``seed``, such as ``"random"``
        (S1) or ``"hull_random"`` (S2).
    **options
        Passed to the strategy, such as ``min_points``.

    Returns
    -------
    pd.DataFrame
        One row per nest, as [`evaluate.compare`][samplesar.evaluate.compare]
        gives, indexed by seed.

    Raises
    ------
    ValueError
        If ``n`` is below 1, or ``strategy`` isn't a registered strategy
        that takes a seed.
    """
    import inspect

    from samplesar._registry import _REGISTRY, run

    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}.")
    found = _REGISTRY.get(strategy)
    if found is None or "seed" not in inspect.signature(found.original).parameters:
        raise ValueError(
            f"strategy must be a registered strategy that takes a seed, got {strategy!r}."
        )
    nests = {str(seed + i): run(sample, strategy, seed=seed + i, **options) for i in range(n)}
    with warnings.catch_warnings():
        # Bounds are reported per row instead; other warnings still show.
        warnings.filterwarnings("ignore", message="Parameters stopped on a bound")
        table = compare(nests, model=model)
    table.index = table.index.astype(int)
    table.index.name = "seed"
    return table


def against_baseline(nest: Nest, baseline: pd.DataFrame, model: str = "power") -> pd.DataFrame:
    """Where a nest's fit falls among a baseline's.

    Parameters
    ----------
    nest : Nest
        The nest to place.
    baseline : pd.DataFrame
        From [`evaluate.baseline`][samplesar.evaluate.baseline], for the same
        sample and model.
    model : str
        The model the baseline was fitted with.

    Returns
    -------
    pd.DataFrame
        One row per parameter and for ``r_squared``: the nest's ``value``,
        the baseline's ``mean``, ``sd``, ``q025`` and ``q975``, ``n`` (the
        baseline fits used, those with a finite value; failed fits are left
        out) and ``p``, the two-sided rank p-value
        ``(1 + #as-extreme) / (n + 1)`` (as extreme: at least as far from
        the baseline's median), missing if the nest's own value isn't
        finite. Its ``attrs``
        record ``at_bounds_share``, the share of baseline fits with a
        parameter on a bound, whose values are bounds, not estimates.

    Raises
    ------
    ValueError
        If the baseline is empty, lacks the model's parameters or has no
        finite value of one, or the model gives no single fit with
        parameters.
    """
    if baseline.empty:
        raise ValueError("The baseline is empty.")
    fit = fit_nest(nest, model)
    if not hasattr(fit, "params"):
        raise ValueError(
            f"sars.sar_{model} returns {type(fit).__name__}, not a single fit with "
            "parameters; choose a single model such as 'power'."
        )
    names = [*fit.params, "r_squared"]
    missing = [p for p in names if p not in baseline.columns]
    if missing:
        raise ValueError(f"The baseline has no {', '.join(missing)}: fitted with another model?")
    rows = {}
    for name in names:
        value = float(fit.params[name] if name in fit.params else fit.r_squared)
        values = baseline[name].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if len(values) == 0:
            raise ValueError(f"No baseline fit has a finite {name}.")
        center = np.median(values)
        extreme = np.count_nonzero(np.abs(values - center) >= abs(value - center))
        rows[name] = {
            "value": value,
            "mean": values.mean(),
            "sd": values.std(ddof=1) if len(values) > 1 else np.nan,
            "q025": np.quantile(values, 0.025),
            "q975": np.quantile(values, 0.975),
            "n": len(values),
            "p": (1 + extreme) / (len(values) + 1) if np.isfinite(value) else np.nan,
        }
    table = pd.DataFrame.from_dict(rows, orient="index")
    if "at_bounds" in baseline.columns:
        known = baseline["at_bounds"].dropna()
        table.attrs["at_bounds_share"] = float(known.map(bool).mean()) if len(known) else np.nan
    return table


def _seeded(strategy: str) -> bool:
    """Whether a registered strategy takes a seed; raises if it isn't registered."""
    import inspect

    from samplesar._registry import _REGISTRY

    found = _REGISTRY.get(strategy)
    if found is None:
        raise ValueError(f"No strategy {strategy!r}; known: {', '.join(_REGISTRY)}.")
    return "seed" in inspect.signature(found.original).parameters


def ensemble(
    sample: Any,
    strategy: str = "hull_random",
    n: int = 99,
    seed: int = 0,
    points: int = 50,
    **options: Any,
) -> pd.DataFrame:
    """Average many seeded nests of a sample into one curve with a band.

    One random nest is one draw: its curve wanders with the luck of its
    removals. This runs ``n`` nests of a random strategy, reads each one's
    species count at the same areas, and gives the mean and spread there:
    the strategy's expected curve, and how far a single nest strays from it.

    Parameters
    ----------
    sample : Sample
        The sample to nest.
    strategy : str
        A registered strategy that takes a ``seed``, such as
        ``"hull_random"``, ``"hull_weighted"`` or ``"random"``.
    n : int
        How many nests.
    seed : int
        The first seed; nest ``i`` uses ``seed + i``.
    points : int
        How many areas to read the curves at, evenly spaced on a log scale
        from the smallest positive area any nest reaches to the largest.
    **options
        Passed to the strategy, such as ``power`` or ``min_points``.

    Returns
    -------
    pd.DataFrame
        One row per area: ``area`` (km²), ``mean``, ``median``, ``q025`` and
        ``q975`` (of the nests' species counts there) and ``n``, the nests
        whose areas span it. A nest's count at an area is interpolated
        between its steps on log area, and isn't extrapolated beyond them.
        Its ``attrs`` record ``strategy``, ``options``, ``seeds``.

    Raises
    ------
    ValueError
        If ``n`` or ``points`` is below 1, ``strategy`` isn't a registered
        strategy that takes a seed, or no nest has two steps with a positive
        area.

    Examples
    --------
    >>> band = samplesar.evaluate.ensemble(sample, "hull_random", n=99)  # doctest: +SKIP
    >>> ax.fill_between(band["area"], band["q025"], band["q975"], alpha=0.3)  # doctest: +SKIP
    >>> ax.plot(band["area"], band["mean"])  # doctest: +SKIP
    """
    from samplesar._registry import run

    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}.")
    if points < 1:
        raise ValueError(f"points must be at least 1, got {points}.")
    if not _seeded(strategy):
        raise ValueError(
            f"strategy must be a registered strategy that takes a seed, got {strategy!r}."
        )
    curves = []
    for i in range(n):
        data = run(sample, strategy, seed=seed + i, **options).sar_data()
        # One species count per area, in increasing area: steps that share
        # an area (points removed inside the hull) keep their largest count.
        by_area = data.groupby("area")["species"].max()
        if len(by_area) >= 2:
            curves.append((np.log(by_area.index.to_numpy(float)), by_area.to_numpy(float)))
    if not curves:
        raise ValueError("No nest has two steps with a positive area to make a curve from.")
    low = min(c[0][0] for c in curves)
    high = max(c[0][-1] for c in curves)
    grid = np.linspace(low, high, points) if points > 1 else np.array([high])
    values = np.full((len(curves), len(grid)), np.nan)
    for row, (log_area, species) in enumerate(curves):
        inside = (grid >= log_area[0] - 1e-12) & (grid <= log_area[-1] + 1e-12)
        values[row, inside] = np.interp(grid[inside], log_area, species)
    covered = np.isfinite(values).sum(axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # areas no nest spans stay NaN
        table = pd.DataFrame(
            {
                "area": np.exp(grid),
                "mean": np.nanmean(values, axis=0),
                "median": np.nanmedian(values, axis=0),
                "q025": np.nanquantile(values, 0.025, axis=0),
                "q975": np.nanquantile(values, 0.975, axis=0),
                "n": covered,
            }
        )
    table.attrs.update(strategy=strategy, options=dict(options), seeds=list(range(seed, seed + n)))
    return table


def bootstrap(
    sample: Any,
    strategy: str = "hull_peel",
    model: str = "power",
    n: int = 99,
    seed: int = 0,
    **options: Any,
) -> pd.DataFrame:
    """Fit a model to nests of bootstrap resamples of a sample's points.

    A deterministic strategy gives one nest, and one fit, per sample, with
    nothing to say how much either owes to the particular points observed.
    This draws the sample's points with replacement ``n`` times (each
    resample holds about 63% of the distinct points, with all the
    observations at them), nests each resample with the same strategy, and
    fits each nest: the spread of the fits is the fit's sensitivity to which
    places happened to be sampled. Steps within a nest aren't independent,
    so read the spread as a description, not as a confidence interval of
    the usual kind.

    Parameters
    ----------
    sample : Sample
        The sample to resample.
    strategy : str
        A registered strategy. One that takes a ``seed`` is given
        ``seed + i`` for resample ``i``, unless ``seed`` is in ``options``.
    model : str
        A sars model with one fit per dataset, such as ``"power"``.
    n : int
        How many resamples.
    seed : int
        Seed for the resampling.
    **options
        Passed to the strategy.

    Returns
    -------
    pd.DataFrame
        One row per resample that could be nested and fitted, as
        [`evaluate.compare`][samplesar.evaluate.compare] gives, indexed by
        ``resample`` (0 to ``n - 1``), with ``n_points``, the distinct
        points drawn. A resample is left out when a layer-peeling strategy
        can't take a layer from it, or its nest has fewer than three steps
        with an area to fit. Works with
        [`evaluate.against_baseline`][samplesar.evaluate.against_baseline]
        as a baseline does.

    Raises
    ------
    ValueError
        If ``n`` is below 1, ``strategy`` isn't registered or rejects its
        options, or no resample could be nested and fitted.
    """
    from samplesar._nest import NoStepError
    from samplesar._registry import run

    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}.")
    seeded = _seeded(strategy) and "seed" not in options
    rng = np.random.default_rng(seed)
    nests, sizes = {}, {}
    for i in range(n):
        drawn = np.unique(rng.integers(sample.n_points, size=sample.n_points))
        own = {**options, "seed": seed + i} if seeded else options
        try:
            nest = run(sample.subset(drawn), strategy, **own)
        except NoStepError:
            continue
        if len(nest.sar_data()) < MIN_FIT_POINTS:
            continue
        nests[str(i)] = nest
        sizes[i] = len(drawn)
    if not nests:
        raise ValueError(f"No resample could be nested and fitted with {strategy!r}.")
    with warnings.catch_warnings():
        # Bounds are reported per row instead; other warnings still show.
        warnings.filterwarnings("ignore", message="Parameters stopped on a bound")
        table = compare(nests, model=model)
    table.index = table.index.astype(int)
    table.index.name = "resample"
    table.insert(0, "n_points", pd.Series(sizes))
    return table
