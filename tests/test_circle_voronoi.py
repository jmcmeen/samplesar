"""The enclosing-circle strategy and Voronoi density peeling."""

import itertools

import numpy as np
import pandas as pd
import pytest
import shapely

import samplesar
from samplesar._hull import _circle_of, enclosing_circle
from samplesar._ranked import voronoi_density

UTM_17N = "EPSG:32617"
ORIGIN = np.array([500_000.0, 4_000_000.0])


def km_sample(xy_km, species=None):
    xy = np.asarray(xy_km, dtype=float) * 1000.0 + ORIGIN
    if species is None:
        species = [f"sp{i % 4}" for i in range(len(xy))]
    frame = pd.DataFrame({"species": species, "lon": xy[:, 0], "lat": xy[:, 1]})
    return samplesar.sources.dataframe(frame, crs=UTM_17N)


def removed(nest):
    return [int(np.setdiff1d(a, b)[0]) for a, b in zip(nest.steps, nest.steps[1:], strict=False)]


def brute_circle(points):
    """The smallest enclosing circle by trying every pair and triple."""
    best = (np.inf, None)
    for size in (2, 3):
        for chosen in itertools.combinations(range(len(points)), size):
            center, radius = _circle_of(points[list(chosen)])
            if radius < best[0] and (np.hypot(*(points - center).T) <= radius * (1 + 1e-9)).all():
                best = (radius, center)
    return best[1], best[0]


@pytest.mark.parametrize("seed", range(8))
def test_the_enclosing_circle_is_the_smallest(seed):
    rng = np.random.default_rng(seed)
    points = rng.normal(0, 1, (int(rng.integers(3, 25)), 2)) * rng.uniform(0.1, 50)
    center, radius = enclosing_circle(points)
    want_center, want_radius = brute_circle(points)
    assert radius == pytest.approx(want_radius, rel=1e-9)
    assert center == pytest.approx(want_center, abs=1e-6 * want_radius)


def test_the_enclosing_circle_far_from_the_origin():
    rng = np.random.default_rng(2)
    points = rng.uniform(0, 1, (12, 2))
    near_center, near_radius = enclosing_circle(points)
    center, radius = enclosing_circle(points + [500_000.0, 4_000_000.0])
    assert radius == pytest.approx(near_radius, rel=1e-9)
    assert center - [500_000.0, 4_000_000.0] == pytest.approx(near_center, abs=1e-8)


def test_enclosing_circles_of_few_or_awkward_points():
    assert enclosing_circle(np.array([[3.0, 4.0]]))[1] == 0.0
    center, radius = enclosing_circle(np.array([[0.0, 0.0], [2.0, 0.0]]))
    assert center.tolist() == [1.0, 0.0] and radius == 1.0
    line = np.array([[0.0, 0.0], [1.0, 1.0], [3.0, 3.0], [2.0, 2.0]])
    center, radius = enclosing_circle(line)
    assert center == pytest.approx([1.5, 1.5]) and radius == pytest.approx(np.hypot(1.5, 1.5))
    same = np.array([[1.0, 1.0]] * 4)
    assert enclosing_circle(same)[1] == 0.0
    # An obtuse triangle: the circle rests on its two far corners only.
    center, radius = enclosing_circle(np.array([[0.0, 0.0], [10.0, 0.0], [5.0, 1.0]]))
    assert center == pytest.approx([5.0, 0.0]) and radius == pytest.approx(5.0)
    with pytest.raises(ValueError):
        enclosing_circle(np.empty((0, 2)))


def test_circle_removes_the_resting_point_that_shrinks_it_most():
    rng = np.random.default_rng(4)
    sample = km_sample(rng.uniform(0, 10, (30, 2)))
    nest = samplesar.strategies.circle(sample, min_points=4)
    xy = sample.xy
    for step, point in zip(nest.steps, removed(nest), strict=False):
        center, radius = brute_circle(xy[step])
        on = step[np.hypot(*(xy[step] - center).T) >= radius * (1 - 1e-9)]
        assert point in on
        after = {int(v): brute_circle(xy[step[step != v]])[1] for v in on}
        assert after[point] == pytest.approx(min(after.values()), rel=1e-9)
    assert len(nest.steps[-1]) == 4 and nest.strategy == "circle"
    again = samplesar.strategies.run(sample, "circle", min_points=4)
    assert removed(again) == removed(nest)
    radii = [brute_circle(xy[step])[1] for step in nest.steps]
    assert all(b <= a * (1 + 1e-9) for a, b in zip(radii, radii[1:], strict=False))


def test_a_long_arm_goes_before_the_bulk():
    # A 5 by 5 block with an arm reaching 20 km east.
    block = [(x, y) for x in range(5) for y in range(5)]
    arm = [(x, 2) for x in range(6, 21, 2)]
    sample = km_sample(block + arm)
    nest = samplesar.strategies.circle(sample, min_points=20)
    gone = removed(nest)
    assert 32 in gone[:2]  # the arm's tip is on the first circle
    assert set(range(25, 33)) <= set(gone[:12])  # the arm is gone before the block is cut far
    # radial, around the mean, cuts the block's far corners while the arm's base remains.
    radial = removed(samplesar.strategies.radial(sample, min_points=20))
    assert not set(range(25, 33)) <= set(radial[:12])


def test_circle_ties_go_to_the_lowest_index():
    sample = km_sample([(0, 0), (4, 0), (4, 4), (0, 4), (2, 2), (1, 2), (3, 2)])
    assert removed(samplesar.strategies.circle(sample, min_points=6)) == [0]


def test_voronoi_cells_of_a_grid_and_an_outlier():
    grid = [(x, y) for x in range(5) for y in range(5)]
    density = voronoi_density(np.array(grid, dtype=float))
    by_point = dict(zip(grid, density, strict=True))
    assert by_point[(2, 2)] == pytest.approx(1.0)  # an inner cell is 1 km²
    assert by_point[(0, 2)] == pytest.approx(2.0)  # an edge cell is half inside the hull
    assert by_point[(0, 0)] == pytest.approx(4.0)  # a corner cell a quarter
    areas = 1.0 / density
    assert areas.sum() == pytest.approx(16.0)  # the cells tile the hull


def test_points_at_one_place_share_its_cell():
    xy = np.array([(0, 0), (2, 0), (2, 2), (0, 2), (1, 1), (1, 1), (1, 1)], dtype=float)
    density = voronoi_density(xy)
    assert density[4] == density[5] == density[6]
    assert (1.0 / density[:4]).sum() + 3.0 / density[4] == pytest.approx(4.0)
    assert voronoi_density(np.array([(0, 0), (1, 1), (2, 2)], dtype=float)).tolist() == [0, 0, 0]


def test_voronoi_peeling_removes_the_point_standing_for_the_most_ground():
    rng = np.random.default_rng(3)
    dense = rng.uniform(0, 2, (60, 2))
    sparse = rng.uniform(0, 20, (15, 2))
    sample = km_sample(np.vstack([dense, sparse]))
    nest = samplesar.strategies.density_peel(sample, method="voronoi", min_points=10)
    density = voronoi_density(np.asarray(sample.xy))
    order = removed(nest)
    assert order == list(np.lexsort((np.arange(75), density))[: len(order)])
    # The scattered points go before the cluster.
    assert np.mean(np.array(order[:12]) >= 60) > 0.8
    for before, after in zip(nest.steps, nest.steps[1:], strict=False):
        assert set(after) < set(before)
    assert dict(nest.options) == {"method": "voronoi", "min_points": 10}


def test_kernel_peeling_is_unchanged_and_options_are_checked():
    rng = np.random.default_rng(1)
    sample = km_sample(rng.uniform(0, 10, (40, 2)))
    plain = samplesar.strategies.density_peel(sample)
    assert removed(plain) == removed(samplesar.strategies.density_peel(sample, method="kernel"))
    assert removed(plain) != removed(samplesar.strategies.density_peel(sample, method="voronoi"))
    with pytest.raises(ValueError, match="method"):
        samplesar.strategies.density_peel(sample, method="knn")
    with pytest.raises(ValueError, match="bandwidth_km is only used"):
        samplesar.strategies.density_peel(sample, method="voronoi", bandwidth_km=2.0)


def test_voronoi_cells_match_the_points_nearest_ground():
    # Each cell's area, estimated by which point a fine grid of the hull is nearest to.
    rng = np.random.default_rng(8)
    xy = rng.uniform(0, 10, (12, 2))
    hull = shapely.MultiPoint(xy).convex_hull
    gx, gy = np.meshgrid(np.linspace(0, 10, 400), np.linspace(0, 10, 400))
    cells = np.column_stack([gx.ravel(), gy.ravel()])
    cells = cells[shapely.contains_xy(hull, cells[:, 0], cells[:, 1])]
    nearest = np.argmin(((cells[:, None, :] - xy[None, :, :]) ** 2).sum(axis=2), axis=1)
    estimate = np.bincount(nearest, minlength=12) * (10 / 399) ** 2
    assert 1.0 / voronoi_density(xy) == pytest.approx(estimate, rel=0.05, abs=0.05)


def test_a_tight_cluster_beside_open_ground_keeps_its_order():
    # 60 points over 100 km and 40 within a meter: cell areas span eleven
    # orders of magnitude, and every point must still go in order of its own.
    rng = np.random.default_rng(0)
    xy = np.vstack([rng.uniform(0, 100, (60, 2)), 50 + rng.uniform(0, 0.001, (40, 2))])
    sample = km_sample(xy)
    density = voronoi_density(np.asarray(sample.xy))
    assert density.max() / density.min() > 1e9
    nest = samplesar.strategies.density_peel(sample, method="voronoi", min_points=3)
    order = removed(nest)
    assert order == list(np.argsort(density, kind="stable")[: len(order)])


def test_voronoi_cells_of_three_points_and_near_duplicates():
    three = np.array([(0, 0), (4, 0), (0, 4)], dtype=float)
    assert (1.0 / voronoi_density(three)).sum() == pytest.approx(8.0)
    near = np.array([(0, 0), (4, 0), (4, 4), (0, 4), (2, 2), (2 + 1e-7, 2)], dtype=float)
    density = voronoi_density(near)
    assert np.isfinite(density).all() and (1.0 / density).sum() == pytest.approx(16.0)


def test_circle_on_a_cluster_a_few_centimeters_across():
    # Far from the projection's origin, a tiny circle must still find every
    # point it rests on.
    rng = np.random.default_rng(6)
    for trial in range(20):
        offsets = rng.uniform(0, 1e-5, (6, 2))  # a 1 cm box, in km
        sample = km_sample(offsets)
        nest = samplesar.strategies.circle(sample, min_points=5)
        local = offsets - offsets.mean(axis=0)
        center, radius = brute_circle(local)
        on = np.flatnonzero(np.hypot(*(local - center).T) >= radius * (1 - 1e-6))
        after = {int(v): brute_circle(np.delete(local, v, axis=0))[1] for v in on}
        best = min(after.values())
        assert after[removed(nest)[0]] == pytest.approx(best, rel=1e-5), trial


def test_circle_with_shared_positions_a_line_and_few_points():
    doubled = km_sample([(0, 0), (0, 0), (4, 0), (4, 4), (0, 4), (2, 2), (1, 3)])
    nest = samplesar.strategies.circle(doubled, min_points=1)
    assert nest.n_steps == doubled.n_points  # coincident observations are one point
    line = km_sample([(x, 0) for x in range(6)])
    order = removed(samplesar.strategies.circle(line, min_points=1))
    assert order[0] in (0, 5) and len(order) == 5
    # Each end in turn: the circle on a line rests on its two ends.
    assert sorted(order[:2]) in ([0, 1], [0, 5], [4, 5])
    pair = km_sample([(0, 0), (1, 1)])
    assert removed(samplesar.strategies.circle(pair, min_points=1)) == [0]
