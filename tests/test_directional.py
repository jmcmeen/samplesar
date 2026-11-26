"""Directional strategies: band, sector, and radial's ellipses."""

import numpy as np
import pandas as pd
import pytest

import samplesar
from samplesar._ranked import _to_xy

UTM_17N = "EPSG:32617"
ORIGIN = np.array([500_000.0, 4_000_000.0])


def km_sample(xy_km, species=None):
    xy = np.asarray(xy_km, dtype=float) * 1000.0 + ORIGIN
    if species is None:
        species = [f"sp{i % 4}" for i in range(len(xy))]
    frame = pd.DataFrame({"species": species, "lon": xy[:, 0], "lat": xy[:, 1]})
    return samplesar.sources.dataframe(frame, crs=UTM_17N)


@pytest.fixture
def sample():
    rng = np.random.default_rng(5)
    return km_sample(rng.uniform(-10, 10, (80, 2)))


def removed(nest):
    return [int(np.setdiff1d(a, b)[0]) for a, b in zip(nest.steps, nest.steps[1:], strict=False)]


def offsets(sample):
    return sample.xy - sample.xy.mean(axis=0)


@pytest.mark.parametrize(
    ("strategy", "options"),
    [
        (samplesar.strategies.band, {}),
        (samplesar.strategies.band, {"bearing": 37.0}),
        (samplesar.strategies.sector, {}),
        (samplesar.strategies.sector, {"sectors": 6, "start": 45.0, "clockwise": False}),
        (samplesar.strategies.radial, {"stretch": 3.0, "bearing": 20.0}),
    ],
)
def test_nested_one_point_per_step_and_reproducible(sample, strategy, options):
    nest = strategy(sample, min_points=5, **options)
    assert len(nest.steps[-1]) == 5
    for before, after in zip(nest.steps, nest.steps[1:], strict=False):
        assert set(after) < set(before) and len(before) - len(after) == 1
    assert removed(nest) == removed(strategy(sample, min_points=5, **options))
    assert nest.strategy in samplesar.strategies.available()


@pytest.mark.parametrize("bearing", [0.0, 90.0, 37.0, 250.0])
def test_band_removes_the_farthest_along_the_bearing(sample, bearing):
    nest = samplesar.strategies.band(sample, bearing=bearing)
    east, north = np.sin(np.radians(bearing)), np.cos(np.radians(bearing))
    off = offsets(sample)
    along = np.abs(off[:, 0] * east + off[:, 1] * north)
    for step, point in zip(nest.steps, removed(nest), strict=False):
        assert along[point] == pytest.approx(along[step].max())


def test_band_keeps_the_span_across_the_bearing(sample):
    # Closing in from east and west: the north-south extent of what's left
    # is the extent of a strip, not of a disc.
    nest = samplesar.strategies.band(sample, bearing=90.0, min_points=20)
    disc = samplesar.strategies.radial(sample, min_points=20)
    y = sample.xy[:, 1]
    assert np.ptp(y[nest.steps[-1]]) > 1.5 * np.ptp(y[disc.steps[-1]])
    x = sample.xy[:, 0]
    assert np.ptp(x[nest.steps[-1]]) < 0.5 * np.ptp(y[nest.steps[-1]])


def test_band_around_a_given_line():
    sample = km_sample([(0, y) for y in range(9)] + [(1, 4.5)])
    # The line at y = 1 km: the points go from the far (north) end.
    lon, lat = samplesar.sources.dataframe(
        pd.DataFrame({"species": ["a"], "lon": [ORIGIN[0]], "lat": [ORIGIN[1] + 1000.0]}),
        crs=UTM_17N,
    ).points[0]
    nest = samplesar.strategies.band(sample, center=(lon, lat), min_points=3)
    assert removed(nest) == [8, 7, 6, 5, 9, 4, 3]
    assert sorted(nest.steps[-1]) == [0, 1, 2]


def latitude_gradient():
    """A 9 by 9 grid, 1 km apart; each row (a latitude) has its own species."""
    xy = [(x, y) for y in range(9) for x in range(9)]
    return km_sample(xy, species=[f"row{y}" for _, y in xy])


def test_a_latitude_band_loses_a_species_per_row_from_both_ends():
    nest = samplesar.strategies.band(latitude_gradient(), bearing=0.0, min_points=9)
    table = nest.to_frame()
    # Rows go in pairs from the two ends (ties by index: the south one first).
    richness = table.set_index("n_points")["species"]
    assert richness[81] == 9
    assert richness[72] == 8  # the southern row gone
    assert richness[63] == 7  # and the northern
    assert richness[9] == 1
    rows = sorted({int(round(y)) for y in offsets(nest.sample)[nest.steps[-1], 1]})
    assert rows == [0]  # the middle row is the last left


def test_a_longitude_band_keeps_every_row_and_its_species():
    nest = samplesar.strategies.band(latitude_gradient(), bearing=90.0, min_points=9)
    table = nest.to_frame()
    assert table["species"].eq(9).all()  # the gradient survives to the end
    assert table["n_points"].iloc[-1] == 9
    assert table["area"].iloc[0] == pytest.approx(64.0, rel=2e-3)  # UTM's scale, corrected
    assert table["area"].iloc[-1] == pytest.approx(0.0, abs=1e-9)  # one column


@pytest.mark.parametrize("stretch", [0.25, 1.0, 3.0])
@pytest.mark.parametrize("bearing", [0.0, 60.0])
def test_radial_ellipses_remove_the_farthest_scaled_point(sample, stretch, bearing):
    nest = samplesar.strategies.radial(sample, stretch=stretch, bearing=bearing)
    east, north = np.sin(np.radians(bearing)), np.cos(np.radians(bearing))
    off = offsets(sample)
    along = off[:, 0] * east + off[:, 1] * north
    across = off[:, 0] * north - off[:, 1] * east
    distance = np.hypot(along / stretch, across)
    for step, point in zip(nest.steps, removed(nest), strict=False):
        assert distance[point] == pytest.approx(distance[step].max())


def test_radial_without_stretch_is_unchanged(sample):
    plain = samplesar.strategies.radial(sample)
    assert removed(plain) == removed(samplesar.strategies.radial(sample, stretch=1.0, bearing=77.0))
    distance = np.hypot(*offsets(sample).T)
    assert removed(plain) == list(np.argsort(-distance, kind="stable")[: len(removed(plain))])


def test_a_stretched_radial_shrinks_faster_across_its_bearing(sample):
    nest = samplesar.strategies.radial(sample, stretch=4.0, bearing=0.0, min_points=20)
    left = sample.xy[nest.steps[-1]]
    assert np.ptp(left[:, 1]) > 2 * np.ptp(left[:, 0])  # long north-south
    assert dict(nest.options) == {"stretch": 4.0, "bearing": 0.0, "min_points": 20}


def compass():
    """Two rings of eight points on the compass bearings 22.5, 67.5, ...,
    at 2 km and 1 km, and one at the center."""
    bearings = np.radians(np.arange(8) * 45.0 + 22.5)
    ring = np.column_stack([np.sin(bearings), np.cos(bearings)])
    return km_sample(np.vstack([2 * ring, ring, [[0.0, 0.0]]]))


def test_quadrants_go_clockwise_from_the_north_east_farthest_first():
    nest = samplesar.strategies.sector(compass(), min_points=1)
    # NE holds bearings 22.5 and 67.5: outer ring (0, 1), then inner (8, 9).
    assert removed(nest)[:4] == [0, 1, 8, 9]
    assert removed(nest)[4:8] == [2, 3, 10, 11]  # SE
    assert removed(nest)[8:12] == [4, 5, 12, 13]  # SW
    assert removed(nest)[12:] == [6, 7, 14, 15]  # NW
    assert list(nest.steps[-1]) == [16]  # the center belongs to the last sector


def test_sectors_counterclockwise_from_a_start():
    nest = samplesar.strategies.sector(
        compass(), start=180.0, sectors=8, clockwise=False, min_points=1
    )
    # Counterclockwise from south: bearing 157.5 first, then 112.5, ...
    assert removed(nest)[:6] == [3, 11, 2, 10, 1, 9]
    assert removed(nest)[-2:] == [4, 12] and list(nest.steps[-1]) == [16]


def test_a_point_on_a_sector_line_is_in_the_later_sector():
    # Symmetric about the origin, so the default center is the origin.
    sample = km_sample(
        [(0, 1), (1, 0), (0, -1), (-1, 0), (0.5, 0.5), (-0.5, -0.5), (0, 0.1), (0, -0.1)]
    )
    nest = samplesar.strategies.sector(sample, min_points=1)
    # Due north (bearing 0) opens the first quadrant, due east (90) the
    # second, due south the third and due west the fourth.
    assert [*removed(nest), int(nest.steps[-1][0])] == [0, 4, 6, 1, 2, 5, 7, 3]


@pytest.mark.parametrize(
    ("sectors", "start", "clockwise"),
    [
        (4, 0.0, True),
        (7, 0.0, True),
        (7, 30.0, False),
        (13, 200.0, True),
        (np.int64(3), -45.0, False),
    ],
)
def test_sector_order_by_brute_force(sample, sectors, start, clockwise):
    center = (float(sample.points[:, 0].mean()), float(sample.points[:, 1].mean()))
    nest = samplesar.strategies.sector(
        sample, start=start, sectors=sectors, clockwise=clockwise, center=center, min_points=1
    )
    off = sample.xy - _to_xy(sample, *center)
    bearing = np.degrees(np.arctan2(off[:, 0], off[:, 1]))
    turned = np.mod(bearing - start if clockwise else start - bearing, 360.0)
    which = (turned // (360.0 / int(sectors))).astype(int)
    distance = np.hypot(off[:, 0], off[:, 1])
    order = [*removed(nest), int(nest.steps[-1][0])]
    assert order == sorted(range(sample.n_points), key=lambda i: (which[i], -distance[i], i))
    assert isinstance(nest.options["sectors"], int | np.integer)


@pytest.mark.parametrize("sectors", [7, 11, 13, 14])
def test_a_point_on_any_sector_line_opens_that_sector(sectors):
    # One point on each line between sectors, all at 1 km, and the center.
    lines = np.radians(np.arange(sectors) * 360.0 / sectors)
    sample = km_sample(np.vstack([np.column_stack([np.sin(lines), np.cos(lines)]), [[0, 0]]]))
    origin = samplesar.sources.dataframe(
        pd.DataFrame({"species": ["a"], "lon": [ORIGIN[0]], "lat": [ORIGIN[1]]}), crs=UTM_17N
    ).points[0]
    nest = samplesar.strategies.sector(
        sample, sectors=sectors, center=(origin[0], origin[1]), min_points=1
    )
    assert removed(nest) == list(range(sectors))
    back = samplesar.strategies.sector(
        sample, sectors=sectors, clockwise=False, center=(origin[0], origin[1]), min_points=1
    )
    assert removed(back) == [0, *range(sectors - 1, 0, -1)]


def test_points_all_at_the_center():
    sample = km_sample([(0, 0), (0, 0.0)] + [(1e-12, 0)], species=["a", "b", "c"])
    nest = samplesar.strategies.sector(sample, min_points=1)
    assert nest.n_steps == sample.n_points


def test_one_sector_is_radial(sample):
    whole = samplesar.strategies.sector(sample, sectors=1)
    assert removed(whole) == removed(samplesar.strategies.radial(sample))


def test_losing_one_side_first_changes_the_curve():
    # Species are confined to the north-east; losing that quadrant first
    # drops them at once, losing it last keeps them to the end.
    rng = np.random.default_rng(2)
    xy = rng.uniform(-10, 10, (200, 2))
    off = xy - xy.mean(axis=0)
    north_east = (off[:, 0] > 0) & (off[:, 1] > 0)
    species = np.where(north_east, [f"ne{i % 10}" for i in range(200)], "everywhere")
    sample = km_sample(xy, species=list(species))
    first = samplesar.strategies.sector(sample, start=0.0).to_frame()
    last = samplesar.strategies.sector(sample, start=90.0).to_frame()
    half = len(first) // 2
    assert first["species"].iloc[half] == 1
    assert last["species"].iloc[half] == 11


@pytest.mark.parametrize(
    ("strategy", "options"),
    [
        (samplesar.strategies.band, {"bearing": float("inf")}),
        (samplesar.strategies.band, {"center": (0.0, 100.0)}),
        (samplesar.strategies.band, {"min_points": 0}),
        (samplesar.strategies.sector, {"start": float("nan")}),
        (samplesar.strategies.sector, {"sectors": 0}),
        (samplesar.strategies.sector, {"sectors": 2.5}),
        (samplesar.strategies.sector, {"sectors": True}),
        (samplesar.strategies.radial, {"stretch": 0.0}),
        (samplesar.strategies.radial, {"stretch": float("inf")}),
        (samplesar.strategies.radial, {"stretch": 1e-320}),
        (samplesar.strategies.radial, {"bearing": float("nan")}),
        (samplesar.strategies.radial, {"bearing": "90"}),
        (samplesar.strategies.radial, {"stretch": True}),
        (samplesar.strategies.band, {"bearing": None}),
        (samplesar.strategies.sweep, {"bearing": "east"}),
        (samplesar.strategies.sector, {"start": "north"}),
        (samplesar.strategies.sector, {"clockwise": "false"}),
    ],
)
def test_bad_options(sample, strategy, options):
    with pytest.raises(ValueError):
        strategy(sample, **options)
