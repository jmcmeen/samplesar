"""Strategies that remove points from the convex hull inwards."""

import numpy as np
import pandas as pd
import pytest

import samplesar
from samplesar._hull import hull_vertices


def make_sample(points, species=None):
    points = np.asarray(points, dtype=float)
    if species is None:
        species = [f"sp{i % 5}" for i in range(len(points))]
    return samplesar.Sample.from_df(
        pd.DataFrame({"species": species, "lon": points[:, 0], "lat": points[:, 1]})
    )


@pytest.fixture
def sample():
    rng = np.random.default_rng(0)
    n = 40
    return make_sample(
        np.column_stack([rng.uniform(-82.3, -82.2, n), rng.uniform(36.5, 36.6, n)]),
        species=rng.choice([f"sp{i}" for i in range(12)], size=n),
    )


def grid(k):
    xs, ys = np.meshgrid(np.arange(k), np.arange(k))
    return np.column_stack([xs.ravel(), ys.ravel()]) * 0.01


def assert_nested(nest):
    for before, after in zip(nest.steps, nest.steps[1:], strict=False):
        assert set(after) < set(before)
    frame = nest.to_frame()
    assert (np.diff(frame["area"]) <= 1e-12).all()
    assert (np.diff(frame["species"]) <= 0).all()


def removed(nest):
    """The point(s) each step removes, with the step they were removed from."""
    return [
        (before, np.setdiff1d(before, after))
        for before, after in zip(nest.steps, nest.steps[1:], strict=False)
    ]


def test_hull_vertices_skip_points_on_edges():
    points = np.array([[0, 0], [1, 0], [2, 0], [2, 2], [0, 2], [1, 1]], dtype=float)
    assert list(hull_vertices(points, np.arange(6))) == [0, 2, 3, 4]


def test_hull_vertices_of_collinear_points_are_the_ends():
    points = np.array([[0, 0], [1, 1], [3, 3], [2, 2]], dtype=float)
    assert sorted(hull_vertices(points, np.arange(4))) == [0, 2]


class TestHullRandomNest:
    def test_nested_one_point_at_a_time(self, sample):
        nest = samplesar.hull_random_nest(sample, seed=1)
        assert nest.n_steps == sample.n_points - 3 + 1
        assert all(
            len(after) == len(before) - 1
            for before, after in zip(nest.steps, nest.steps[1:], strict=False)
        )
        assert_nested(nest)

    def test_only_hull_vertices_are_removed(self, sample):
        points = sample.points
        for before, gone in removed(samplesar.hull_random_nest(sample, seed=2)):
            assert gone[0] in hull_vertices(points, before)

    def test_seed_is_reproducible(self, sample):
        a = samplesar.hull_random_nest(sample, seed=7)
        b = samplesar.hull_random_nest(sample, seed=7)
        c = samplesar.hull_random_nest(sample, seed=8)
        assert all(np.array_equal(x, y) for x, y in zip(a.steps, b.steps, strict=True))
        assert any(not np.array_equal(x, y) for x, y in zip(a.steps, c.steps, strict=True))

    def test_interior_point_outlasts_the_corners(self):
        # A centre point is never a vertex while three or more corners remain.
        corners = [[0, 0], [1, 0], [1, 1], [0, 1]]
        nest = samplesar.hull_random_nest(make_sample([*corners, [0.5, 0.5]]), seed=0)
        assert 4 in nest.steps[-1]

    def test_min_points(self, sample):
        nest = samplesar.hull_random_nest(sample, min_points=10, seed=0)
        assert len(nest.steps[-1]) == 10

    def test_runs_down_to_one_point(self, sample):
        nest = samplesar.hull_random_nest(sample, min_points=1, seed=0)
        assert len(nest.steps[-1]) == 1


class TestHullPeelNest:
    def test_peels_a_grid_ring_by_ring(self):
        nest = samplesar.hull_peel_nest(make_sample(grid(7)), min_points=1)
        assert [len(s) for s in nest.steps] == [49, 25, 9, 1]
        assert_nested(nest)

    def test_stops_before_leaving_too_few(self):
        nest = samplesar.hull_peel_nest(make_sample(grid(7)))
        assert [len(s) for s in nest.steps] == [49, 25, 9]

    def test_each_step_removes_the_whole_boundary(self, sample):
        points = sample.points
        for before, gone in removed(samplesar.hull_peel_nest(sample)):
            vertices = set(hull_vertices(points, before))
            assert vertices <= set(gone)
            # What's left is strictly inside: none of it is a vertex of the old hull.
            assert not vertices & set(np.setdiff1d(before, gone))

    def test_collinear_remainder_is_one_layer(self):
        # After the square, the three collinear points form a degenerate hull:
        # all of them are on it, so peeling would leave nothing.
        square = [[0, 0], [4, 0], [4, 4], [0, 4]]
        line = [[1, 2], [2, 2], [3, 2]]
        nest = samplesar.hull_peel_nest(make_sample([*square, *line]), min_points=1)
        assert [len(s) for s in nest.steps] == [7, 3]


@pytest.mark.parametrize("strategy", [samplesar.hull_random_nest, samplesar.hull_peel_nest])
def test_rejects_bad_sizes(strategy, sample):
    with pytest.raises(ValueError, match="at least 3"):
        strategy(make_sample([[0, 0], [1, 1]]))
    with pytest.raises(ValueError, match="min_points must be at least 1"):
        strategy(sample, min_points=0)


@pytest.mark.parametrize("strategy", [samplesar.hull_random_nest, samplesar.hull_peel_nest])
def test_sar_data_fits_with_sars(strategy, sample):
    import sars

    data = strategy(sample).sar_data()
    assert (data["area"] > 0).all()
    assert len(data) >= 2
    assert np.isfinite(sars.sar_power(data).params["z"])


@pytest.mark.parametrize("seed", range(5))
def test_hull_vertices_match_brute_force(seed):
    """A point is a vertex exactly when removing it shrinks the hull."""
    rng = np.random.default_rng(seed)
    points = rng.uniform(0, 1, size=(12, 2))
    idx = np.arange(len(points))
    full = samplesar.hull_area(points)
    brute = [i for i in idx if samplesar.hull_area(points[idx != i]) < full - 1e-12]
    assert sorted(hull_vertices(points, idx)) == brute


def test_peeling_points_all_on_the_hull_is_an_error():
    angles = np.linspace(0, 2 * np.pi, 10, endpoint=False)
    circle = make_sample(np.column_stack([np.cos(angles), np.sin(angles)]))
    with pytest.raises(ValueError, match="No layer can be peeled"):
        samplesar.hull_peel_nest(circle)
