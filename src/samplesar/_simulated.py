"""A simulated landscape of observations, for examples, teaching and tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from samplesar._sample import Sample
from samplesar._sources import dataframe

EARTH_KM_PER_DEGREE = 111.32


def simulated(
    n_points: int = 600,
    n_species: int = 60,
    seed: int = 0,
    side_km: float = 80.0,
    center: tuple[float, float] = (-83.5, 35.6),
    clustered: float = 0.5,
) -> Sample:
    """Simulate observations over a square of mountain country.

    No real observations ship with samplesar, so the examples start here: a
    landscape small enough to nest in a second, with the features of real
    observation data that the strategies respond to.

    - **Terrain.** A ridge runs from the south-west corner to the
      north-east, about 1,800 m at its crest, above valleys near 300 m.
    - **Species.** Each has a home (a point, a radius of a few to a few tens
      of km) and an elevation it prefers, so richness varies over the
      square and with height, and some species are rare. A place where no
      species turned up records the one best suited to it.
    - **Effort.** A share of the points pile up around a few places (a town,
      trailheads), the rest are scattered; the piles were visited first and
      the scattered points later, between 2005 and 2024.

    Parameters
    ----------
    n_points : int
        How many places were visited.
    n_species : int
        How many species live in the square (some may go unobserved).
    seed : int
        Seed for the whole simulation: the same seed gives the same sample.
    side_km : float
        The side of the square, in km.
    center : tuple of float
        The (longitude, latitude) of the square's center.
    clustered : float
        The share of points piled around a few places, from 0 (all
        scattered at random) to 1.

    Returns
    -------
    Sample
        With an ``elevation`` (meters) and a ``date`` for every observation.

    Raises
    ------
    ValueError
        If ``n_points`` or ``n_species`` isn't a whole number of at least
        1, ``seed`` isn't a whole number, ``side_km`` isn't positive (or is
        over 1,000), ``clustered`` is outside 0 to 1, or ``center`` is
        within 10° of the antimeridian or 15° of a pole.

    Examples
    --------
    >>> sample = samplesar.sources.simulated(seed=1)
    >>> sample.n_points
    600
    """
    for name, count in (("n_points", n_points), ("n_species", n_species)):
        if isinstance(count, bool) or not isinstance(count, int | np.integer) or count < 1:
            raise ValueError(f"{name} must be a whole number of at least 1, got {count!r}.")
    if isinstance(seed, bool) or not isinstance(seed, int | np.integer):
        raise ValueError(f"seed must be a whole number, got {seed!r}.")
    if not (np.isfinite(side_km) and 0 < side_km <= 1000):
        raise ValueError(f"side_km must be positive and at most 1,000, got {side_km}.")
    if not 0 <= clustered <= 1:
        raise ValueError(f"clustered must be between 0 and 1, got {clustered}.")
    lon0, lat0 = (float(v) for v in center)
    if not (-170 <= lon0 <= 170 and -75 <= lat0 <= 75):
        raise ValueError(
            "center must be a (longitude, latitude) within 170° of the prime meridian and "
            f"75° of the equator, got {center}."
        )
    rng = np.random.default_rng(seed)
    half = side_km / 2.0

    # Where people went: piles around a few places, and scattered points.
    n_piled = round(n_points * clustered)
    places = rng.uniform(-0.7 * half, 0.7 * half, (5, 2))
    piled = places[rng.integers(len(places), size=n_piled)] + rng.normal(
        0, 0.04 * side_km, (n_piled, 2)
    )
    scattered = rng.uniform(-half, half, (n_points - n_piled, 2))
    xy = np.clip(np.vstack([piled, scattered]), -half, half)

    # Terrain: height falls off with distance from the south-west to north-east diagonal.
    from_ridge = (xy[:, 1] - xy[:, 0]) / np.sqrt(2.0)
    elevation = 300.0 + 1500.0 * np.exp(-((from_ridge / (0.18 * side_km)) ** 2))
    elevation = np.round(elevation + rng.normal(0, 40.0, n_points), 1)

    # When: the piles between 2005 and 2014, scattered points from 2010 on.
    start = np.datetime64("2005-01-01")
    day = np.concatenate(
        [rng.integers(0, 3650, n_piled), rng.integers(1826, 7300, n_points - n_piled)]
    )

    # Species: a home range and a preferred elevation each.
    homes = rng.uniform(-half, half, (n_species, 2))
    reach = np.minimum(rng.lognormal(np.log(0.15 * side_km), 0.7, n_species), side_km)
    prefers = rng.uniform(300.0, 1800.0, n_species)
    tolerance = rng.uniform(150.0, 600.0, n_species)
    distance = np.hypot(xy[:, None, 0] - homes[None, :, 0], xy[:, None, 1] - homes[None, :, 1])
    suited = np.exp(-(((elevation[:, None] - prefers[None, :]) / tolerance[None, :]) ** 2))
    chance = 0.9 * suited * (distance <= reach[None, :])
    seen = rng.random((n_points, n_species)) < chance
    # Every place has at least one record: the species best suited to it
    # among those whose range reaches it, or of them all where none does.
    empty = ~seen.any(axis=1)
    within = distance[empty] <= reach[None, :]
    reachable = np.where(within.any(axis=1, keepdims=True), within, True)
    seen[empty, np.argmax(np.where(reachable, suited[empty], -1.0), axis=1)] = True

    place, species = np.nonzero(seen)
    lat = lat0 + xy[:, 1] / EARTH_KM_PER_DEGREE
    lon = lon0 + xy[:, 0] / (EARTH_KM_PER_DEGREE * np.cos(np.radians(lat0)))
    width = len(str(n_species))
    frame = pd.DataFrame(
        {
            "species": [f"Species {s + 1:0{width}d}" for s in species],
            "lon": np.round(lon, 6)[place],
            "lat": np.round(lat, 6)[place],
            "elevation": elevation[place],
            "date": (start + day.astype("timedelta64[D]"))[place].astype("datetime64[ns]"),
        }
    )
    return dataframe(frame, elevation_col="elevation", date_col="date")
