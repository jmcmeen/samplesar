"""Strategies that remove points from the convex hull inwards."""

import numpy as np
import pandas as pd
import pytest

import samplesar
from samplesar._hull import hull_vertices

STRATEGIES = [
    samplesar.hull_random_nest,
    samplesar.hull_peel_nest,
    samplesar.hull_area_nest,
    samplesar.hull_species_nest,
    samplesar.hull_distance_nest,
]


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


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_rejects_bad_sizes(strategy, sample):
    with pytest.raises(ValueError, match="at least 3"):
        strategy(make_sample([[0, 0], [1, 1]]))
    with pytest.raises(ValueError, match="min_points must be at least 1"):
        strategy(sample, min_points=0)


@pytest.mark.parametrize("strategy", STRATEGIES)
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


def brute_force_choice(sample, before, score):
    """The vertex a rule-based strategy should remove from ``before``: highest
    score, then farthest from the vertices' centroid, then lowest index, with
    scores and distances compared to 9 significant figures."""
    points = sample.points
    vertices = hull_vertices(points, before)
    centre = points[vertices].mean(axis=0)

    def rounded(x):
        return float(f"{x:.9g}")

    return min(
        vertices,
        key=lambda v: (
            -rounded(score(before[before != v])),
            -rounded(np.hypot(*(points[v] - centre))),
            v,
        ),
    )


def shifted(points, by=(-80.1, 35.2)):
    return np.asarray(points, dtype=float) + by


class TestHullAreaNest:
    @pytest.mark.parametrize("keep", ["largest", "smallest"])
    def test_matches_brute_force(self, sample, keep):
        sign = 1 if keep == "largest" else -1
        nest = samplesar.hull_area_nest(sample, keep=keep)
        assert nest.n_steps == sample.n_points - 3 + 1
        assert_nested(nest)
        for before, gone in removed(nest):
            expected = brute_force_choice(
                sample, before, lambda rest: sign * samplesar.hull_area(sample.points[rest])
            )
            assert gone[0] == expected

    def test_known_answer(self):
        # A unit square with a point just inside corner 0: dropping corner 0
        # loses almost nothing, so "largest" takes it first; "smallest" takes a
        # corner whose loss halves the square.
        points = [[0, 0], [1, 0], [1, 1], [0, 1], [0.05, 0.05]]
        largest = samplesar.hull_area_nest(make_sample(points), keep="largest")
        assert 0 not in largest.steps[1]
        smallest = samplesar.hull_area_nest(make_sample(points), keep="smallest")
        assert samplesar.hull_area(np.asarray(points, float)[smallest.steps[1]]) < 0.6

    def test_ties_go_to_the_lowest_index(self):
        # The four corners of a square are interchangeable.
        square = make_sample([[0, 0], [1, 0], [1, 1], [0, 1]])
        assert list(samplesar.hull_area_nest(square).steps[1]) == [1, 2, 3]

    def test_area_ties_go_to_the_farthest_vertex(self):
        # Dropping vertex 2 or 3 leaves the same area (1.0); 3 is farther from
        # the vertices' centroid, so it goes, though its index is higher.
        quad = make_sample([[0, 0], [0, 2], [1, 1], [1, 0]])
        nest = samplesar.hull_area_nest(quad, keep="largest")
        assert list(nest.steps[1]) == [0, 1, 2]

    @pytest.mark.parametrize("keep", ["largest", "smallest"])
    def test_reproducible_wherever_the_sample_lies(self, keep):
        # A symmetric hexagon: its tied vertices get areas that differ in the
        # last bits at real coordinates, which must not change the choice.
        hexagon = [[0, 0], [0.3, 0], [0.3, 0.1], [0, 0.1], [0.15, 0.3], [0.15, -0.2], [0.1, 0.05]]
        here = samplesar.hull_area_nest(make_sample(hexagon), keep=keep)
        again = samplesar.hull_area_nest(make_sample(hexagon), keep=keep)
        there = samplesar.hull_area_nest(make_sample(shifted(hexagon)), keep=keep)
        for a, b, c in zip(here.steps, again.steps, there.steps, strict=True):
            assert np.array_equal(a, b) and np.array_equal(a, c)

    def test_rejects_unknown_keep(self, sample):
        with pytest.raises(ValueError, match="keep must be"):
            samplesar.hull_area_nest(sample, keep="biggest")


def species_at(sample, rest):
    """Distinct species at the given point rows, recounted from scratch."""
    return sample.subset(rest).n_species


@pytest.fixture
def shared_points_sample():
    """Several observations at most points, so species group per point."""
    rng = np.random.default_rng(4)
    points = rng.uniform(0, 0.1, size=(25, 2))
    pick = rng.integers(0, 25, size=90)
    return make_sample(points[pick], species=rng.choice([f"sp{i}" for i in range(20)], size=90))


class TestHullSpeciesNest:
    @pytest.mark.parametrize("keep", ["most", "fewest"])
    @pytest.mark.parametrize("which", ["sample", "shared_points_sample"])
    def test_matches_brute_force(self, request, which, keep):
        sample = request.getfixturevalue(which)
        sign = 1 if keep == "most" else -1
        nest = samplesar.hull_species_nest(sample, keep=keep)
        assert nest.n_steps == sample.n_points - 3 + 1
        assert_nested(nest)
        for before, gone in removed(nest):
            expected = brute_force_choice(
                sample, before, lambda rest: sign * species_at(sample, rest)
            )
            assert gone[0] == expected

    def test_shared_points_count_once(self):
        # Two observations at one point: its species are lost together.
        df = pd.DataFrame(
            {
                "species": ["a", "b", "a", "c", "c", "d"],
                "lon": [0.0, 0.0, 1.0, 1.0, 0.0, 0.5],
                "lat": [0.0, 0.0, 0.0, 1.0, 1.0, 0.5],
            }
        )
        sample = samplesar.Sample.from_df(df)
        # Points: (0,0) holds a and b, (1,0) a, (1,1) c, (0,1) c, (0.5,0.5) d.
        # Dropping (1,0) or (1,1) or (0,1) loses nothing; (0,0) loses b.
        # Points 1, 2 and 3 tie on species and on distance from the centre, so
        # the lowest index goes.
        most = samplesar.hull_species_nest(sample, keep="most")
        assert list(most.steps[1]) == [0, 2, 3, 4]
        fewest = samplesar.hull_species_nest(sample, keep="fewest")
        assert list(fewest.steps[1]) == [1, 2, 3, 4]

    @pytest.mark.parametrize("keep", ["most", "fewest"])
    def test_reproducible_wherever_the_sample_lies(self, shared_points_sample, keep):
        data = shared_points_sample.data
        moved = samplesar.Sample.from_df(
            data.assign(lon=data["lon"] - 80.1, lat=data["lat"] + 35.2)
        )
        here = samplesar.hull_species_nest(shared_points_sample, keep=keep)
        again = samplesar.hull_species_nest(shared_points_sample, keep=keep)
        there = samplesar.hull_species_nest(moved, keep=keep)
        for a, b, c in zip(here.steps, again.steps, there.steps, strict=True):
            assert np.array_equal(a, b) and np.array_equal(a, c)

    def test_rejects_unknown_keep(self, sample):
        with pytest.raises(ValueError, match="keep must be"):
            samplesar.hull_species_nest(sample, keep="many")


class TestHullDistanceNest:
    @pytest.mark.parametrize("remove", ["farthest", "closest"])
    def test_matches_brute_force(self, sample, remove):
        sign = 1 if remove == "farthest" else -1
        points = sample.points
        nest = samplesar.hull_distance_nest(sample, remove=remove)
        assert nest.n_steps == sample.n_points - 3 + 1
        assert_nested(nest)
        for before, gone in removed(nest):
            vertices = hull_vertices(points, before)
            centre = points[vertices].mean(axis=0)
            expected = min(vertices, key=lambda v: (-sign * np.hypot(*(points[v] - centre)), v))
            assert gone[0] == expected

    def test_known_answer(self):
        # A square with one corner pulled far out: "farthest" takes it first.
        points = [[0, 0], [1, 0], [1, 1], [0, 1], [5, 5], [0.5, 0.5]]
        far = samplesar.hull_distance_nest(make_sample(points))
        assert 4 not in far.steps[1]
        near = samplesar.hull_distance_nest(make_sample(points), remove="closest")
        assert 4 in near.steps[-1]

    def test_ties_go_to_the_lowest_index(self):
        square = make_sample([[0, 0], [1, 0], [1, 1], [0, 1]])
        assert list(samplesar.hull_distance_nest(square).steps[1]) == [1, 2, 3]

    def test_rejects_unknown_remove(self, sample):
        with pytest.raises(ValueError, match="remove must be"):
            samplesar.hull_distance_nest(sample, remove="outer")
