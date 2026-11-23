"""Strategies that remove points from the convex hull inwards."""

import numpy as np
import pandas as pd
import pytest
import shapely

import samplesar
from samplesar._hull import hull_vertices

STRATEGIES = [
    samplesar.strategies.hull_random,
    samplesar.strategies.hull_weighted,
    samplesar.strategies.hull_peel,
    samplesar.strategies.hull_area,
    samplesar.strategies.hull_species,
    samplesar.strategies.hull_distance,
]


def make_sample(points, species=None, crs=None):
    points = np.asarray(points, dtype=float)
    if species is None:
        species = [f"sp{i % 5}" for i in range(len(points))]
    return samplesar.sources.dataframe(
        pd.DataFrame({"species": species, "lon": points[:, 0], "lat": points[:, 1]}), crs=crs
    )


@pytest.fixture
def sample():
    rng = np.random.default_rng(0)
    n = 40
    return make_sample(
        np.column_stack([rng.uniform(-82.3, -82.2, n), rng.uniform(36.5, 36.6, n)]),
        species=rng.choice([f"sp{i}" for i in range(12)], size=n),
    )


UTM_17N = "EPSG:32617"


def grid(k, spacing=100.0):
    """A k-by-k grid of points laid out in UTM zone 17N, in meters."""
    xs, ys = np.meshgrid(np.arange(k), np.arange(k))
    return np.column_stack([xs.ravel(), ys.ravel()]) * spacing + [500_000.0, 4_000_000.0]


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


def test_hull_vertices_skip_edge_points_rounding_pushed_out():
    points = np.array([[0, 0], [1, -1e-13], [2, 0], [2, 2], [0, 2]], dtype=float)
    assert list(hull_vertices(points, np.arange(5))) == [0, 2, 3, 4]


def test_a_utm_grid_has_only_its_corners_as_vertices():
    sample = make_sample(grid(7), crs=UTM_17N)
    vertices = hull_vertices(sample.xy, np.arange(sample.n_points))
    assert sorted(vertices) == [0, 6, 42, 48]


def test_a_row_collinear_but_for_rounding_has_two_ends():
    row = make_sample(grid(7)[:7], crs=UTM_17N)
    assert sorted(hull_vertices(row.xy, np.arange(row.n_points))) == [0, 6]
    points = np.array([[0, 0], [1, 1e-13], [2, -1e-13], [3, 0], [1.5, 2e-13]])
    assert sorted(hull_vertices(points, np.arange(5))) == [0, 3]


def test_a_real_bend_is_kept():
    # A corner well off the line through its neighbors stays a vertex.
    points = np.array([[0, 0], [1, -1e-3], [2, 0], [2, 2], [0, 2]], dtype=float)
    assert list(hull_vertices(points, np.arange(5))) == [0, 1, 2, 3, 4]


def test_hull_vertices_of_collinear_points_are_the_ends():
    points = np.array([[0, 0], [1, 1], [3, 3], [2, 2]], dtype=float)
    assert sorted(hull_vertices(points, np.arange(4))) == [0, 2]


class TestHullRandomNest:
    def test_nested_one_point_at_a_time(self, sample):
        nest = samplesar.strategies.hull_random(sample, seed=1)
        assert nest.n_steps == sample.n_points - 3 + 1
        assert all(
            len(after) == len(before) - 1
            for before, after in zip(nest.steps, nest.steps[1:], strict=False)
        )
        assert_nested(nest)

    def test_only_hull_vertices_are_removed(self, sample):
        points = sample.xy
        for before, gone in removed(samplesar.strategies.hull_random(sample, seed=2)):
            assert gone[0] in hull_vertices(points, before)

    def test_seed_is_reproducible(self, sample):
        a = samplesar.strategies.hull_random(sample, seed=7)
        b = samplesar.strategies.hull_random(sample, seed=7)
        c = samplesar.strategies.hull_random(sample, seed=8)
        assert all(np.array_equal(x, y) for x, y in zip(a.steps, b.steps, strict=True))
        assert any(not np.array_equal(x, y) for x, y in zip(a.steps, c.steps, strict=True))

    def test_interior_point_outlasts_the_corners(self):
        # A center point is never a vertex while three or more corners remain.
        corners = [[0, 0], [1, 0], [1, 1], [0, 1]]
        nest = samplesar.strategies.hull_random(make_sample([*corners, [0.5, 0.5]]), seed=0)
        assert 4 in nest.steps[-1]

    def test_min_points(self, sample):
        nest = samplesar.strategies.hull_random(sample, min_points=10, seed=0)
        assert len(nest.steps[-1]) == 10

    def test_runs_down_to_one_point(self, sample):
        nest = samplesar.strategies.hull_random(sample, min_points=1, seed=0)
        assert len(nest.steps[-1]) == 1


class TestHullPeelNest:
    def test_peels_a_grid_ring_by_ring(self):
        nest = samplesar.strategies.hull_peel(make_sample(grid(7), crs=UTM_17N), min_points=1)
        assert [len(s) for s in nest.steps] == [49, 25, 9, 1]
        assert_nested(nest)

    def test_stops_before_leaving_too_few(self):
        nest = samplesar.strategies.hull_peel(make_sample(grid(7), crs=UTM_17N))
        assert [len(s) for s in nest.steps] == [49, 25, 9]

    def test_each_step_removes_the_whole_boundary(self, sample):
        points = sample.xy
        for before, gone in removed(samplesar.strategies.hull_peel(sample)):
            vertices = set(hull_vertices(points, before))
            assert vertices <= set(gone)
            # What's left is strictly inside: none of it is a vertex of the old hull.
            assert not vertices & set(np.setdiff1d(before, gone))

    def test_collinear_remainder_is_one_layer(self):
        # After the square, the three collinear points form a degenerate hull:
        # all of them are on it, so peeling would leave nothing.
        square = [[0, 0], [4, 0], [4, 4], [0, 4]]
        line = [[1, 2], [2, 2], [3, 2]]
        nest = samplesar.strategies.hull_peel(make_sample([*square, *line]), min_points=1)
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
        samplesar.strategies.hull_peel(circle)


def brute_force_choice(sample, before, score):
    """The vertex a rule-based strategy should remove from ``before``: highest
    score, then farthest from the vertices' centroid, then lowest index, with
    scores and distances compared to 9 significant figures."""
    points = sample.xy
    vertices = hull_vertices(points, before)
    center = points[vertices].mean(axis=0)

    def rounded(x):
        return float(f"{x:.9g}")

    return min(
        vertices,
        key=lambda v: (
            -rounded(score(before[before != v])),
            -rounded(np.hypot(*(points[v] - center))),
            v,
        ),
    )


def at_real_coordinates(offsets, center=(-80.1, 35.2)):
    """Offsets (in degrees) placed around a real location. A shape mirrored
    east-west about its center stays mirrored once projected, so mirror pairs
    tie exactly in theory but differ in their last bits in floating point."""
    return np.asarray(offsets, dtype=float) + center


class TestHullAreaNest:
    @pytest.mark.parametrize("keep", ["largest", "smallest"])
    def test_matches_brute_force(self, sample, keep):
        sign = 1 if keep == "largest" else -1
        nest = samplesar.strategies.hull_area(sample, keep=keep)
        assert nest.n_steps == sample.n_points - 3 + 1
        assert_nested(nest)
        for before, gone in removed(nest):
            expected = brute_force_choice(
                sample, before, lambda rest: sign * samplesar.hull_area(sample.xy[rest])
            )
            assert gone[0] == expected

    def test_known_answer(self):
        # A small square with a point just inside corner 0: dropping corner 0
        # loses almost nothing, so "largest" takes it first; "smallest" takes a
        # corner whose loss halves the square.
        points = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0.05, 0.05]]) * 0.01
        largest = samplesar.strategies.hull_area(make_sample(points), keep="largest")
        assert 0 not in largest.steps[1]
        smallest = samplesar.strategies.hull_area(make_sample(points), keep="smallest").to_frame()
        assert smallest["area"][1] / smallest["area"][0] < 0.6

    def test_ties_go_to_the_lowest_index(self):
        # The four corners of a square centered on (0, 0), where the projection
        # is symmetric, are interchangeable.
        square = make_sample(np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * 0.01)
        assert list(samplesar.strategies.hull_area(square).steps[1]) == [1, 2, 3]

    def test_area_ties_go_to_the_farthest_vertex(self):
        # Dropping vertex 2 or 3 leaves the same area (1.0); 3 is farther from
        # the vertices' centroid, so it goes, though its index is higher.
        quad = make_sample(np.array([[0, 0], [0, 2], [1, 1], [1, 0]]) * 0.001)
        nest = samplesar.strategies.hull_area(quad, keep="largest")
        assert list(nest.steps[1]) == [0, 1, 2]

    @pytest.mark.parametrize("keep", ["largest", "smallest"])
    def test_ties_hold_at_real_coordinates(self, keep):
        # A hexagon mirrored east-west, at a real location: its mirror pairs'
        # areas differ only in the last bits, and the tie rule must still pick.
        hexagon = [
            [-0.15, 0],
            [0.15, 0],
            [0.15, 0.1],
            [-0.15, 0.1],
            [0, 0.3],
            [0, -0.2],
            [-0.05, 0.05],
        ]
        sample = make_sample(at_real_coordinates(hexagon))
        sign = 1 if keep == "largest" else -1
        nest = samplesar.strategies.hull_area(sample, keep=keep)
        again = samplesar.strategies.hull_area(sample, keep=keep)
        assert all(np.array_equal(a, b) for a, b in zip(nest.steps, again.steps, strict=True))
        for before, gone in removed(nest):
            expected = brute_force_choice(
                sample, before, lambda rest: sign * samplesar.hull_area(sample.xy[rest])
            )
            assert gone[0] == expected

    def test_rejects_unknown_keep(self, sample):
        with pytest.raises(ValueError, match="keep must be"):
            samplesar.strategies.hull_area(sample, keep="biggest")


# A coastline for the 40-point sample: land with an inlet reaching in from the east.
COAST = shapely.box(-82.35, 36.45, -82.15, 36.65).difference(
    shapely.box(-82.25, 36.53, -82.1, 36.57)
)


class TestHullMaskedAreaNest:
    @pytest.mark.parametrize("keep", ["largest", "smallest"])
    def test_matches_brute_force(self, sample, keep):
        masked = sample.with_mask(COAST)
        sign = 1 if keep == "largest" else -1
        nest = samplesar.strategies.hull_masked_area(masked, keep=keep)
        assert nest.n_steps == sample.n_points - 3 + 1
        assert_nested(nest)
        for before, gone in removed(nest):
            expected = brute_force_choice(
                masked, before, lambda rest: sign * masked.hull_area_of(rest)
            )
            assert gone[0] == expected

    def test_known_answer(self):
        # A square of land with a point far out to sea: losing that point costs
        # most of the hull but almost no land, so hull_masked_area takes it
        # first, while hull_area keeps it and takes a corner.
        points = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0.5, 0.5], [3, 0.5]]) * 0.01
        land = shapely.box(-0.001, -0.001, 0.011, 0.011)
        sample = make_sample(points)
        masked = samplesar.strategies.hull_masked_area(sample.with_mask(land))
        assert 5 not in masked.steps[1]
        assert 5 in samplesar.strategies.hull_area(sample).steps[1]

    def test_a_mask_covering_every_hull_chooses_as_hull_area(self, sample):
        everywhere = sample.with_mask(shapely.box(-83, 36, -82, 37))
        masked = samplesar.strategies.hull_masked_area(everywhere)
        plain = samplesar.strategies.hull_area(sample)
        assert all(np.array_equal(a, b) for a, b in zip(masked.steps, plain.steps, strict=True))

    def test_the_mask_changes_the_choices(self, sample):
        masked = samplesar.strategies.hull_masked_area(sample.with_mask(COAST))
        plain = samplesar.strategies.hull_area(sample)
        assert any(not np.array_equal(a, b) for a, b in zip(masked.steps, plain.steps, strict=True))

    def test_is_reproducible(self, sample):
        masked = sample.with_mask(COAST)
        first = samplesar.strategies.hull_masked_area(masked, keep="smallest")
        again = samplesar.strategies.run(masked, "hull_masked_area", keep="smallest")
        assert all(np.array_equal(a, b) for a, b in zip(first.steps, again.steps, strict=True))

    def test_needs_a_mask(self, sample):
        with pytest.raises(ValueError, match="needs a sample with a mask"):
            samplesar.strategies.hull_masked_area(sample)

    def test_rejects_unknown_keep(self, sample):
        with pytest.raises(ValueError, match="keep must be"):
            samplesar.strategies.hull_masked_area(sample.with_mask(COAST), keep="biggest")


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
        nest = samplesar.strategies.hull_species(sample, keep=keep)
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
        sample = samplesar.sources.dataframe(df)
        # Points: (0,0) holds a and b, (1,0) a, (1,1) c, (0,1) c, (0.5,0.5) d.
        # Dropping (1,0) or (1,1) or (0,1) loses nothing; (0,0) loses b.
        # Points 1, 2 and 3 tie on species and on distance from the center, so
        # the lowest index goes.
        most = samplesar.strategies.hull_species(sample, keep="most")
        assert list(most.steps[1]) == [0, 2, 3, 4]
        fewest = samplesar.strategies.hull_species(sample, keep="fewest")
        assert list(fewest.steps[1]) == [1, 2, 3, 4]

    @pytest.mark.parametrize("keep", ["most", "fewest"])
    def test_reproducible(self, shared_points_sample, keep):
        here = samplesar.strategies.hull_species(shared_points_sample, keep=keep)
        again = samplesar.strategies.hull_species(shared_points_sample, keep=keep)
        assert all(np.array_equal(a, b) for a, b in zip(here.steps, again.steps, strict=True))

    def test_rejects_unknown_keep(self, sample):
        with pytest.raises(ValueError, match="keep must be"):
            samplesar.strategies.hull_species(sample, keep="many")


class TestHullDistanceNest:
    @pytest.mark.parametrize("remove", ["farthest", "closest"])
    def test_matches_brute_force(self, sample, remove):
        sign = 1 if remove == "farthest" else -1
        points = sample.xy
        nest = samplesar.strategies.hull_distance(sample, remove=remove)
        assert nest.n_steps == sample.n_points - 3 + 1
        assert_nested(nest)
        for before, gone in removed(nest):
            vertices = hull_vertices(points, before)
            center = points[vertices].mean(axis=0)
            expected = min(
                vertices,
                key=lambda v: (-sign * float(f"{np.hypot(*(points[v] - center)):.9g}"), v),
            )
            assert gone[0] == expected

    def test_known_answer(self):
        # A square with one corner pulled far out: "farthest" takes it first.
        points = [[0, 0], [1, 0], [1, 1], [0, 1], [5, 5], [0.5, 0.5]]
        far = samplesar.strategies.hull_distance(make_sample(points))
        assert 4 not in far.steps[1]
        near = samplesar.strategies.hull_distance(make_sample(points), remove="closest")
        assert 4 in near.steps[-1]

    def test_ties_go_to_the_lowest_index(self):
        square = make_sample(np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * 0.01)
        assert list(samplesar.strategies.hull_distance(square).steps[1]) == [1, 2, 3]

    @pytest.mark.parametrize("remove", ["farthest", "closest"])
    def test_ties_hold_at_real_coordinates(self, remove):
        # A square mirrored east-west at a real location: the two southern
        # corners tie on distance (to the last bits), so the lower index goes.
        square = [[-0.005, -0.005], [0.005, -0.005], [0.005, 0.005], [-0.005, 0.005], [0.001, 0.0]]
        nest = samplesar.strategies.hull_distance(
            make_sample(at_real_coordinates(square)), remove=remove
        )
        points = nest.sample.xy
        for before, gone in removed(nest):
            vertices = hull_vertices(points, before)
            center = points[vertices].mean(axis=0)
            sign = 1 if remove == "farthest" else -1
            expected = min(
                vertices,
                key=lambda v: (-sign * float(f"{np.hypot(*(points[v] - center)):.9g}"), v),
            )
            assert gone[0] == expected

    def test_rejects_unknown_remove(self, sample):
        with pytest.raises(ValueError, match="remove must be"):
            samplesar.strategies.hull_distance(sample, remove="outer")


class TestConcavePeelNest:
    def l_shape(self):
        # An L of points: the convex hull spans the empty corner, a concave
        # hull follows the L.
        arm = [[x, y] for x in np.arange(0, 10) for y in (0, 1)]
        leg = [[x, y] for x in (0, 1) for y in np.arange(2, 10)]
        inner = [[0.5, 0.5], [5, 0.5], [0.5, 5]]
        return make_sample(np.array(arm + leg + inner, dtype=float) * 0.001)

    def test_nested_and_registered(self):
        sample = self.l_shape()
        nest = samplesar.strategies.concave_peel(sample, ratio=0.2, min_points=1)
        assert_nested(nest)
        assert nest.label == "concave_peel(ratio=0.2, min_points=1)"
        by_name = samplesar.strategies.run(sample, "concave_peel", ratio=0.2, min_points=1)
        assert all(np.array_equal(a, b) for a, b in zip(nest.steps, by_name.steps, strict=True))

    def test_ratio_one_is_hull_peeling(self, sample):
        concave = samplesar.strategies.concave_peel(sample, ratio=1.0)
        convex = samplesar.strategies.hull_peel(sample)
        assert [len(s) for s in concave.steps] == [len(s) for s in convex.steps]

    def test_follows_the_outline(self):
        # On the L, a tight concave hull's first layer is the whole outline:
        # only the three points inside it (indices 36 to 38) remain. The
        # convex hull's first layer leaves far more.
        sample = self.l_shape()
        concave = samplesar.strategies.concave_peel(sample, ratio=0.05, min_points=1)
        assert list(concave.steps[1]) == [36, 37, 38]
        convex = samplesar.strategies.hull_peel(sample, min_points=1)
        assert len(convex.steps[1]) > 3

    def test_each_layer_is_the_concave_hull_boundary(self):
        # Brute force: each removed set is the points on shapely's concave
        # hull of the step before, within the shared tolerance.
        import shapely
        from shapely.geometry import MultiPoint

        sample = self.l_shape()
        nest = samplesar.strategies.concave_peel(sample, ratio=0.5, min_points=1)
        xy = sample.xy
        for before, gone in removed(nest):
            hull = shapely.concave_hull(MultiPoint(xy[before]), ratio=0.5)
            x0, y0, x1, y1 = hull.bounds
            near = shapely.distance(hull.exterior, shapely.points(xy[before])) <= 1e-9 * np.hypot(
                x1 - x0, y1 - y0
            )
            assert set(gone) == set(before[near])

    @pytest.mark.parametrize("ratio", [-0.1, 1.5])
    def test_bad_ratio(self, sample, ratio):
        with pytest.raises(ValueError, match="ratio must be between 0 and 1"):
            samplesar.strategies.concave_peel(sample, ratio=ratio)


def test_hull_weighted_with_power_zero_is_hull_random(sample):
    weighted = samplesar.strategies.hull_weighted(sample, power=0, seed=3)
    plain = samplesar.strategies.hull_random(sample, seed=3)
    assert all(np.array_equal(a, b) for a, b in zip(weighted.steps, plain.steps, strict=True))


def test_hull_weighted_removes_a_far_outlier_first():
    """A square of close points and one vertex 50 km away: with power 2 the
    outlier carries almost all the weight, so it goes first for nearly
    every seed."""
    square = grid(4, spacing=100.0)
    far = square[:1] + [50_000.0, 50_000.0]
    sample = make_sample(np.vstack([square, far]), crs=UTM_17N)
    outlier = int(np.argmax(sample.xy.sum(axis=1)))
    firsts = [
        int(np.setdiff1d(*samplesar.strategies.hull_weighted(sample, power=2, seed=s).steps[:2])[0])
        for s in range(50)
    ]
    assert firsts.count(outlier) >= 49


def test_hull_weighted_only_removes_hull_vertices(sample):
    nest = samplesar.strategies.hull_weighted(sample, power=1, seed=0)
    for before, after in zip(nest.steps, nest.steps[1:], strict=False):
        (gone,) = np.setdiff1d(before, after)
        assert gone in hull_vertices(sample.xy, before)


@pytest.mark.parametrize("power", [-1.0, float("inf"), float("nan")])
def test_hull_weighted_bad_power(sample, power):
    with pytest.raises(ValueError, match="power"):
        samplesar.strategies.hull_weighted(sample, power=power)


def test_hull_weighted_is_reproducible_with_a_power(sample):
    one = samplesar.strategies.hull_weighted(sample, power=2, seed=11)
    two = samplesar.strategies.hull_weighted(sample, power=2, seed=11)
    assert all(np.array_equal(a, b) for a, b in zip(one.steps, two.steps, strict=True))


def test_hull_weighted_first_choice_follows_the_weights():
    """Brute force on a small sample: over many seeds, the first vertex
    removed is drawn in proportion to its isolation squared."""
    from scipy.spatial.distance import cdist

    rng = np.random.default_rng(5)
    points = np.vstack([rng.uniform(0, 1000, (8, 2)), [[3000.0, 0.0], [0.0, 2500.0]]])
    sample = make_sample(points + [500_000.0, 4_000_000.0], crs=UTM_17N)
    xy = np.asarray(sample.xy)
    vertices = hull_vertices(xy, np.arange(sample.n_points))
    d = cdist(xy[vertices], xy)
    d[d == 0] = np.inf
    expected = d.min(axis=1) ** 2 / (d.min(axis=1) ** 2).sum()
    n = 3000
    firsts = [
        int(
            np.setdiff1d(
                *samplesar.strategies.hull_weighted(sample, power=2, seed=s, min_points=9).steps[:2]
            )[0]
        )
        for s in range(n)
    ]
    observed = np.array([firsts.count(int(v)) for v in vertices]) / n
    assert observed == pytest.approx(expected, abs=0.03)


def test_hull_weighted_with_a_huge_power_takes_the_most_isolated():
    """Isolation below 1 km raised to a power of 400 underflows unless it's
    scaled first; scaled, the most isolated vertex always goes."""
    pts = grid(4, spacing=100.0)
    pts = np.vstack([pts, pts[:1] + [-250.0, -250.0]])
    sample = make_sample(pts, crs=UTM_17N)
    corner = int(np.argmin(sample.xy.sum(axis=1)))
    for s in range(10):
        nest = samplesar.strategies.hull_weighted(sample, power=400, seed=s)
        assert int(np.setdiff1d(*nest.steps[:2])[0]) == corner


@pytest.mark.parametrize("look", [16, 1])
def test_isolation_matches_brute_force(sample, monkeypatch, look):
    from scipy.spatial.distance import cdist

    from samplesar._hull import _Isolation

    monkeypatch.setattr(_Isolation, "LOOK", look)  # 1 forces the fallback
    xy = np.asarray(sample.xy)
    isolation = _Isolation(xy)
    remaining = np.arange(0, sample.n_points, 3)  # most points gone: forces the fallback too
    vertices = hull_vertices(xy, remaining)
    d = cdist(xy[vertices], xy[remaining])
    d[d == 0] = np.inf
    assert isolation(remaining, vertices) == pytest.approx(d.min(axis=1))


def test_grid_peel_with_small_cells_is_hull_peel():
    sample = make_sample(grid(6), crs=UTM_17N)
    cells = samplesar.strategies.grid_peel(sample, cell_km=0.01)
    points = samplesar.strategies.hull_peel(sample)
    assert all(np.array_equal(a, b) for a, b in zip(cells.steps, points.steps, strict=True))


def test_grid_peel_known_layers():
    """A 6x6 grid at 100 m in 200 m cells is 3x3 cells of 4 points: one
    layer of 8 cells comes off, leaving the 4 points of the middle cell."""
    sample = make_sample(grid(6), crs=UTM_17N)
    nest = samplesar.strategies.grid_peel(sample, cell_km=0.2)
    assert [len(s) for s in nest.steps] == [36, 4]
    middle = sample.xy[nest.steps[-1]] - sample.xy.min(axis=0)
    assert np.all((middle > 0.19) & (middle < 0.31))


def test_grid_peel_removes_whole_cells(sample):
    from samplesar._hull import grid_cells

    nest = samplesar.strategies.grid_peel(sample, cell_km=1.0, min_points=1)
    cell = grid_cells(np.asarray(sample.xy), 1.0)
    assert_nested(nest)
    for before, after in zip(nest.steps, nest.steps[1:], strict=False):
        gone = set(cell[np.setdiff1d(before, after)])
        assert not gone & set(cell[after])


@pytest.mark.parametrize(("cell_km", "match"), [(0.0, "positive"), (1e6, "No layer")])
def test_grid_peel_errors(sample, cell_km, match):
    with pytest.raises(ValueError, match=match):
        samplesar.strategies.grid_peel(sample, cell_km=cell_km)


def test_grid_peel_is_reproducible_and_respects_min_points(sample):
    one = samplesar.strategies.grid_peel(sample, cell_km=0.5, min_points=5)
    two = samplesar.strategies.grid_peel(sample, cell_km=0.5, min_points=5)
    assert all(np.array_equal(a, b) for a, b in zip(one.steps, two.steps, strict=True))
    assert len(one.steps[-1]) >= 5
    with pytest.raises(ValueError, match="min_points"):
        samplesar.strategies.grid_peel(sample, cell_km=0.5, min_points=0)


def test_grid_peel_layers_by_brute_force(sample):
    """Each step removes exactly the cells whose point-mean lies on the
    convex hull of the remaining cells' means (checked with shapely)."""
    import shapely

    from samplesar._hull import grid_cells

    xy = np.asarray(sample.xy)
    cell = grid_cells(xy, 0.8)
    means = np.array([xy[cell == c].mean(axis=0) for c in range(cell.max() + 1)])
    nest = samplesar.strategies.grid_peel(sample, cell_km=0.8, min_points=1)
    for before, after in zip(nest.steps, nest.steps[1:], strict=False):
        cells = np.unique(cell[before])
        hull = shapely.convex_hull(shapely.multipoints(means[cells]))
        on_hull = {
            int(c)
            for c in cells
            if hull.boundary.distance(shapely.Point(means[c])) <= 1e-9 * max(1.0, hull.length)
        }
        assert set(np.unique(cell[np.setdiff1d(before, after)]).tolist()) == on_hull
