"""Strategies that rank every point once: radial, sweep, density_peel, grow."""

import numpy as np
import pandas as pd
import pytest

import samplesar
from samplesar._ranked import growth_order, kernel_density

UTM_17N = "EPSG:32617"
ORIGIN = np.array([500_000.0, 4_000_000.0])

STRATEGIES = [
    samplesar.strategies.radial,
    samplesar.strategies.sweep,
    samplesar.strategies.density_peel,
    samplesar.strategies.grow,
]


def km_sample(xy_km, species=None):
    """A sample laid out in UTM zone 17N from km offsets, so xy is known."""
    xy = np.asarray(xy_km, dtype=float) * 1000.0 + ORIGIN
    if species is None:
        species = [f"sp{i % 4}" for i in range(len(xy))]
    frame = pd.DataFrame({"species": species, "lon": xy[:, 0], "lat": xy[:, 1]})
    return samplesar.sources.dataframe(frame, crs=UTM_17N)


@pytest.fixture
def sample():
    rng = np.random.default_rng(1)
    return km_sample(rng.uniform(0, 10, (60, 2)))


def removed(nest):
    """The point each step removed, in order."""
    return [int(np.setdiff1d(a, b)[0]) for a, b in zip(nest.steps, nest.steps[1:], strict=False)]


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_nested_one_point_per_step_and_reproducible(sample, strategy):
    nest = strategy(sample, min_points=5)
    assert len(nest.steps[-1]) == 5
    for before, after in zip(nest.steps, nest.steps[1:], strict=False):
        assert set(after) < set(before) and len(before) - len(after) == 1
    again = strategy(sample, min_points=5)
    assert all(np.array_equal(a, b) for a, b in zip(nest.steps, again.steps, strict=True))
    assert nest.strategy in samplesar.strategies.available()


def test_radial_removes_the_farthest_from_the_centre(sample):
    nest = samplesar.strategies.radial(sample)
    centre = sample.xy.mean(axis=0)
    for step, point in zip(nest.steps, removed(nest), strict=False):
        distance = np.hypot(*(sample.xy[step] - centre).T)
        assert point == step[np.argmax(distance)]


def test_radial_around_a_given_centre():
    sample = km_sample([[0, 0], [1, 0], [3, 0], [6, 0], [10, 0]])
    lon, lat = sample.points[1]  # lon and lat of the second point
    nest = samplesar.strategies.radial(sample, centre=(lon, lat), min_points=1)
    order = removed(nest)
    assert [tuple(np.round(sample.xy[i] - sample.xy[1], 6)) for i in order][:2] == [
        (9.0, 0.0),
        (5.0, 0.0),
    ]
    assert nest.steps[-1].tolist() == [1]


def test_sweep_moves_along_the_bearing():
    sample = km_sample([[x, y] for x in range(4) for y in range(3)])
    east = samplesar.strategies.sweep(sample, bearing=90.0, min_points=1)
    xs = sample.xy[:, 0][removed(east)]
    assert np.all(np.diff(xs) >= -1e-9)
    north = samplesar.strategies.sweep(sample, bearing=0.0, min_points=1)
    ys = sample.xy[:, 1][removed(north)]
    assert np.all(np.diff(ys) >= -1e-9)
    south = samplesar.strategies.sweep(sample, bearing=180.0, min_points=1)
    assert np.all(np.diff(sample.xy[:, 1][removed(south)]) <= 1e-9)


def test_density_peel_removes_isolated_points_first():
    rng = np.random.default_rng(2)
    cluster = rng.normal(0, 0.2, (30, 2))
    outliers = np.array([[8.0, 8.0], [-6.0, 5.0]])
    sample = km_sample(np.vstack([cluster, outliers]))
    nest = samplesar.strategies.density_peel(sample, bandwidth_km=0.5)
    first_two = {tuple(np.round(sample.xy[i] - ORIGIN / 1000, 3)) for i in removed(nest)[:2]}
    assert first_two == {(8.0, 8.0), (-6.0, 5.0)}


def test_kernel_density_matches_brute_force(sample):
    xy = np.asarray(sample.xy)
    d2 = np.sum((xy[:, None] - xy[None]) ** 2, axis=2)
    brute = np.exp(-0.5 * d2 / 1.5**2)
    brute[d2 > (5 * 1.5) ** 2] = 0
    assert kernel_density(xy, 1.5) == pytest.approx(brute.sum(axis=1))


def dense_prim(xy, seed):
    n = len(xy)
    added = np.zeros(n, dtype=bool)
    best = np.full(n, np.inf)
    best[seed] = 0.0
    order = []
    for _ in range(n):
        candidates = np.where(~added)[0]
        point = candidates[np.argmin(best[candidates])]
        added[point] = True
        order.append(point)
        best = np.minimum(best, np.hypot(*(xy - xy[point]).T))
    return order


@pytest.mark.parametrize("seed", [0, 7, 33])
def test_growth_order_is_prims(sample, seed):
    xy = np.asarray(sample.xy)
    assert growth_order(xy, seed).tolist() == dense_prim(xy, seed)


def test_growth_on_collinear_points():
    xy = np.array([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0], [6.0, 0.0], [10.0, 0.0]])
    assert growth_order(xy, 2).tolist() == [2, 1, 0, 3, 4]


def test_grow_leaves_the_seed_last():
    sample = km_sample([[0, 0], [1, 0], [3, 0], [6, 0], [10, 0], [10, 1]])
    lon, lat = sample.points[0]
    nest = samplesar.strategies.grow(sample, seed=(lon, lat), min_points=1)
    assert nest.steps[-1].tolist() == [int(np.argmin(np.hypot(*(sample.xy - sample.xy[0]).T)))]
    assert len(nest.steps[1]) == 5


@pytest.mark.parametrize(
    ("call", "match"),
    [
        (lambda s: samplesar.strategies.radial(s, centre=(500.0, 0.0)), "longitude, latitude"),
        (lambda s: samplesar.strategies.sweep(s, bearing=float("nan")), "bearing"),
        (lambda s: samplesar.strategies.density_peel(s, bandwidth_km=0.0), "positive"),
        (lambda s: samplesar.strategies.grow(s, min_points=0), "min_points"),
    ],
)
def test_bad_arguments(sample, call, match):
    with pytest.raises(ValueError, match=match):
        call(sample)


def test_run_by_name(sample):
    nest = samplesar.strategies.run(sample, "sweep", bearing=45.0)
    assert nest.strategy == "sweep" and nest.options["bearing"] == 45.0


def test_growth_orders_every_point_at_shared_positions():
    """Distinct points can share a projected position (lon 180 and -180):
    each is still ordered, next to its twin."""
    xy = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [3.0, 1.0], [5.0, 0.0], [5.0, 2.0]])
    order = growth_order(xy, 0)
    assert sorted(order.tolist()) == list(range(6))
    assert order.tolist()[:3] == [0, 1, 2]
    assert growth_order(xy, 2).tolist()[:2] == [2, 1]


def test_grow_across_the_antimeridian():
    lon = [179.9, 180.0, -180.0, 179.95, 179.8, 179.85]
    lat = [0.0, 0.1, 0.1, 0.2, 0.05, -0.1]
    frame = pd.DataFrame({"species": list("abcdef"), "lon": lon, "lat": lat})
    sample = samplesar.sources.dataframe(frame)
    nest = samplesar.strategies.grow(sample, seed=(179.9, 0.0), min_points=1)
    assert len(nest.steps) == sample.n_points
    seed_point = int(np.argmin(np.hypot(*(sample.points - [179.9, 0.0]).T)))
    assert nest.steps[-1].tolist() == [seed_point]


def test_growth_on_a_long_line_is_fast_and_ordered():
    xy = np.column_stack([np.arange(20_000.0), np.arange(20_000.0) * 0.5])
    assert growth_order(xy, 0).tolist() == list(range(20_000))


def test_density_in_blocks_matches_one_pass(monkeypatch):
    from samplesar import _ranked

    rng = np.random.default_rng(4)
    xy = rng.uniform(0, 5, (300, 2))
    whole = kernel_density(xy, 0.7)
    monkeypatch.setattr(_ranked, "BLOCK", 37)
    assert kernel_density(xy, 0.7) == pytest.approx(whole)


@pytest.mark.parametrize("bearing", [0.0, 90.0, 180.0, 270.0])
def test_sweep_ties_go_by_index_at_real_coordinates(bearing):
    """On a UTM grid, points in one row or column tie exactly; after the
    round trip through longitude and latitude they differ only by rounding,
    and the tie still goes to the lowest index."""
    sample = km_sample([[x * 0.1, y * 0.1] for y in range(6) for x in range(6)])
    nest = samplesar.strategies.sweep(sample, bearing=bearing, min_points=1)
    first_six = removed(nest)[:6]
    assert first_six == sorted(first_six)
