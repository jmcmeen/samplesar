"""Sample.thin, and concave_peel's ratio schedule."""

import warnings

import numpy as np
import pandas as pd
import pytest
import shapely
from shapely.geometry import MultiPoint

import samplesar

UTM_17N = "EPSG:32617"
ORIGIN = np.array([500_000.0, 4_000_000.0])


def km_sample(xy_km, species=None):
    xy = np.asarray(xy_km, dtype=float) * 1000.0 + ORIGIN
    if species is None:
        species = [f"sp{i % 4}" for i in range(len(xy))]
    frame = pd.DataFrame({"species": species, "lon": xy[:, 0], "lat": xy[:, 1]})
    return samplesar.sources.dataframe(frame, crs=UTM_17N)


def piled():
    """A 10 km square sampled evenly, with 200 more points piled into one
    corner kilometer."""
    rng = np.random.default_rng(0)
    return km_sample(np.vstack([rng.uniform(0, 10, (100, 2)), rng.uniform(0, 1, (200, 2))]))


def cells(sample, km):
    xy = sample.xy
    return [tuple(c) for c in np.floor((xy - piled().xy.min(axis=0)) / km + 1e-9).astype(int)]


def test_one_point_per_cell_evens_out_a_pile():
    sample = piled()
    thinned = sample.thin(1.0)
    kept = cells(thinned, 1.0)
    assert len(set(kept)) == len(kept) == len(set(cells(sample, 1.0)))
    # The pile's kilometer held two thirds of the points; now it holds one.
    assert thinned.n_points < 100
    assert kept.count((0, 0)) == 1
    assert thinned.crs == sample.crs and np.allclose(thinned.area, sample.area, rtol=0.2)


def test_the_point_nearest_each_cells_center_is_kept():
    sample = km_sample([(0, 0), (0.5, 0.5), (0.9, 0.1), (1.2, 0.2), (1.5, 0.5), (0.45, 0.5)])
    thinned = sample.thin(1.0)
    # Cell (0, 0): (0.5, 0.5) is at the center; cell (1, 0): (1.5, 0.5).
    assert np.allclose(thinned.xy, sample.xy[[1, 4]])


def test_ties_for_the_center_go_to_the_lowest_index():
    # Mirror images about the cell's center, laid out in the sample's own crs.
    sample = km_sample([(0, 0), (0.25, 0.5), (0.75, 0.5), (0.5, 0.25), (0.5, 0.75)])
    kept = sample.thin(1.0)
    assert kept.n_points == 1
    assert np.hypot(*(kept.xy[0] - sample.xy[0])) == pytest.approx(np.hypot(0.25, 0.5), abs=0.3)


def test_cells_are_true_kilometers_in_a_distorting_crs():
    # Web Mercator at 60° N stretches lengths twofold: a grid of points 0.6 km
    # apart on the ground is 1.2 "km" apart on the projection.
    lon = 10 + np.arange(8) * 0.6 / 55.8  # km per degree of longitude at 60° N
    lat = 60 + np.arange(8) * 0.6 / 111.4
    frame = pd.DataFrame({"species": "a", "lon": np.repeat(lon, 8), "lat": np.tile(lat, 8)})
    geopandas = pytest.importorskip("geopandas")
    gdf = geopandas.GeoDataFrame(
        frame[["species"]], geometry=geopandas.points_from_xy(frame["lon"], frame["lat"]), crs=4326
    ).to_crs(3857)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # the crs distorts areas: the point here
        sample = samplesar.sources.geodataframe(gdf)
        assert sample.area_scale == pytest.approx(4.0, rel=0.05)
        # 1 km cells hold about (1 / 0.6)² = 2.8 points each.
        assert 64 / 4.5 < sample.thin(1.0).n_points < 64 / 1.8


def test_a_seed_keeps_a_random_point_reproducibly():
    sample = piled()
    a, b, c = sample.thin(2.0, seed=1), sample.thin(2.0, seed=1), sample.thin(2.0, seed=2)
    assert np.array_equal(a.points, b.points)
    assert not np.array_equal(a.points, c.points)
    assert a.n_points == sample.thin(2.0).n_points  # the same cells, other points
    assert not np.array_equal(a.points, sample.thin(2.0).points)


def test_dropped_points_take_their_observations_and_columns_survive():
    frame = pd.DataFrame(
        {
            "species": ["a", "b", "c", "d"],
            "lon": ORIGIN[0] + np.array([500.0, 500.0, 600.0, 5500.0]),
            "lat": ORIGIN[1] + np.array([500.0, 500.0, 100.0, 500.0]),
            "alt": [10.0, 10.0, 20.0, 30.0],
        }
    )
    sample = samplesar.sources.dataframe(frame, crs=UTM_17N, elevation_col="alt")
    thinned = sample.thin(1.0)
    # Cells start at the points' south-west corner, (0.5, 0.1) km: the first
    # cell's center is (1.0, 0.6), nearer to a and b's point than to c's.
    assert sorted(thinned.data["species"]) == ["a", "b", "d"]
    assert thinned.elevations.tolist() == [10.0, 30.0]
    land = shapely.box(*sample.points.min(axis=0) - 1, *sample.points.max(axis=0) + 1)
    assert sample.with_mask(land).thin(1.0).mask is not None


def test_large_cells_and_small_cells():
    sample = piled()
    assert sample.thin(1000.0).n_points == 1
    assert sample.thin(1e-6).n_points == sample.n_points


def test_a_cell_too_small_to_count():
    with pytest.raises(ValueError, match="too small"):
        piled().thin(1e-300)


@pytest.mark.parametrize("km", [0, -1, float("nan"), float("inf"), "1", True, None])
def test_bad_cell_sizes(km):
    with pytest.raises(ValueError, match="km must be a positive number"):
        piled().thin(km)


def test_a_thinned_sample_nests_more_evenly():
    sample = piled()
    plain = samplesar.strategies.hull_peel(sample, min_points=3)
    thinned = samplesar.strategies.hull_peel(sample.thin(1.0), min_points=3)
    # Unthinned, the last layers are all inside the pile's kilometer.
    assert plain.to_frame()["area"].iloc[-1] < 1.0 < thinned.to_frame()["area"].iloc[-1]


def l_shape():
    rng = np.random.default_rng(5)
    arm_a = rng.uniform((0, 0), (10, 2), (120, 2))
    arm_b = rng.uniform((0, 2), (2, 10), (100, 2))
    return km_sample(np.vstack([arm_a, arm_b]))


def test_a_constant_pair_is_the_single_ratio():
    sample = l_shape()
    one = samplesar.strategies.concave_peel(sample, ratio=0.3)
    pair = samplesar.strategies.concave_peel(sample, ratio=(0.3, 0.3))
    assert all(np.array_equal(a, b) for a, b in zip(one.steps, pair.steps, strict=True))
    assert pair.label == "concave_peel(ratio=(0.3, 0.3))"


def test_each_layer_is_peeled_at_the_scheduled_ratio():
    sample = l_shape()
    nest = samplesar.strategies.concave_peel(sample, ratio=(0.8, 0.1), min_points=5)
    xy, n = sample.xy, sample.n_points
    for before, after in zip(nest.steps, nest.steps[1:], strict=False):
        done = (n - len(before)) / (n - 5)
        hull = shapely.concave_hull(MultiPoint(xy[before]), ratio=0.8 - 0.7 * done)
        on = shapely.distance(hull.exterior, shapely.points(xy[before])) <= 1e-9 * 15
        assert np.array_equal(after, before[~on])
    loose = samplesar.strategies.concave_peel(sample, ratio=0.8, min_points=5)
    tight = samplesar.strategies.concave_peel(sample, ratio=0.1, min_points=5)
    assert np.array_equal(nest.steps[1], loose.steps[1])  # it starts as the loose one does
    assert tight.n_steps <= nest.n_steps <= loose.n_steps + tight.n_steps


def test_a_schedule_saves_and_loads(tmp_path):
    nest = samplesar.strategies.concave_peel(l_shape(), ratio=(0.6, 0.2))
    path = tmp_path / "nest.json"
    nest.to_json(path)
    back = samplesar.Nest.from_json(path)
    assert back.options["ratio"] == [0.6, 0.2] and back.n_steps == nest.n_steps
    # Saved options hold lists, so the pair is printed with brackets after loading.
    assert nest.label == "concave_peel(ratio=(0.6, 0.2))"
    assert back.label == "concave_peel(ratio=[0.6, 0.2])"
    again = samplesar.strategies.run(back.sample, back.strategy, **back.options)
    assert all(np.array_equal(a, b) for a, b in zip(again.steps, nest.steps, strict=True))
    as_floats = samplesar.strategies.concave_peel(l_shape(), ratio=[np.float64(0.6), 0.2])
    assert as_floats.n_steps == nest.n_steps


@pytest.mark.parametrize("ratio", [(0.5, 1.5), (-0.1, 0.5), (0.5,), (0.1, 0.2, 0.3), "0.3", None])
def test_bad_ratios(ratio):
    with pytest.raises(ValueError, match="ratio"):
        samplesar.strategies.concave_peel(l_shape(), ratio=ratio)
