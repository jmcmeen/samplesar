"""The registry of sampling strategies, so any of them can be run by name."""

from __future__ import annotations

import dataclasses
import functools
import inspect
from collections.abc import Callable
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, ParamSpec

if TYPE_CHECKING:
    from samplesar._nest import Nest
    from samplesar._sample import Sample

P = ParamSpec("P")


@dataclasses.dataclass(frozen=True)
class _Strategy:
    name: str
    func: Callable[..., Nest]
    description: str


_REGISTRY: dict[str, _Strategy] = {}


def register_strategy(
    name: str, description: str | None = None
) -> Callable[[Callable[P, Nest]], Callable[P, Nest]]:
    """Register a sampling strategy, so it can be run by name.

    Decorate a function that takes a [`Sample`][samplesar.Sample] first, any
    options as keywords, and returns a [`Nest`][samplesar.Nest]. Once
    registered it runs through [`nest`][samplesar.nest], is listed by
    [`strategies`][samplesar.strategies], and every nest it returns records
    the strategy's name and the options it was called with.

    Parameters
    ----------
    name : str
        The name to run it by, such as ``"hull_area"``.
    description : str, optional
        One line for [`strategies`][samplesar.strategies]; by default the
        first line of the function's docstring.

    Returns
    -------
    callable
        A decorator that registers the function and returns it, wrapped to
        record the name and options on its nests.

    Raises
    ------
    ValueError
        If another function is already registered under ``name``.

    Examples
    --------
    >>> @samplesar.register_strategy("every_other")  # doctest: +SKIP
    ... def every_other_nest(sample, min_points=3):
    ...     \"\"\"Drop every other point at each step.\"\"\"
    ...     ...
    """

    def decorate(func: Callable[P, Nest]) -> Callable[P, Nest]:
        signature = inspect.signature(func)

        @functools.wraps(func)
        def run(*args: P.args, **kwargs: P.kwargs) -> Nest:
            from samplesar._nest import Nest

            result = func(*args, **kwargs)
            if not isinstance(result, Nest):
                raise TypeError(f"Strategy {name!r} returned {type(result).__name__}, not a Nest.")
            given = signature.bind(*args, **kwargs).arguments
            options = dict(list(given.items())[1:])  # everything but the sample
            return dataclasses.replace(result, strategy=name, options=MappingProxyType(options))

        existing = _REGISTRY.get(name)
        if existing is not None and existing.func.__qualname__ != func.__qualname__:
            raise ValueError(f"A strategy named {name!r} is already registered.")
        doc = inspect.getdoc(func) or ""
        _REGISTRY[name] = _Strategy(name, run, description or doc.split("\n", 1)[0])
        return run

    return decorate


def strategies() -> dict[str, str]:
    """The registered sampling strategies.

    Returns
    -------
    dict of str to str
        Each strategy's name, as [`nest`][samplesar.nest] takes it, and a
        one-line description, in the order they were registered.
    """
    return {s.name: s.description for s in _REGISTRY.values()}


def nest(sample: Sample, strategy: str = "random", **options: Any) -> Nest:
    """Nest a sample with any registered strategy, by name.

    ``samplesar.nest(sample, "hull_area", keep="smallest")`` is the same as
    ``samplesar.hull_area_nest(sample, keep="smallest")``; naming the
    strategy as a string lets a script or a config file choose it.

    Parameters
    ----------
    sample : Sample
        The sample to nest.
    strategy : str
        A name from [`strategies`][samplesar.strategies].
    **options
        The strategy's options, such as ``keep``, ``min_points`` or
        ``seed``.

    Returns
    -------
    Nest
        With its ``strategy`` and ``options`` recorded (see
        [`Nest`][samplesar.Nest]).

    Raises
    ------
    ValueError
        If no strategy has that name, or the strategy rejects its options.
    TypeError
        If an option isn't one the strategy takes.
    """
    found = _REGISTRY.get(strategy)
    if found is None:
        raise ValueError(f"No strategy {strategy!r}; known: {', '.join(_REGISTRY)}.")
    return found.func(sample, **options)
