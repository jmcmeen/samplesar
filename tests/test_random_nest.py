import numpy as np
import pandas as pd
import pytest

import samplesar


@pytest.fixture
def sample():
    rng = np.random.default_rng(0)
    n = 40
    df = pd.DataFrame(
        {
            "species": rng.choice([f"sp{i}" for i in range(12)], size=n),
            "lon": rng.uniform(-82.3, -82.2, size=n),
            "lat": rng.uniform(36.5, 36.6, size=n),
        }
    )
    return samplesar.Sample.from_df(df)


def test_sample_summary(sample):
    assert sample.n_observations == 40
    assert sample.n_points == 40
    assert 1 <= sample.n_species <= 12
    assert sample.area > 0


def test_sample_validation():
    with pytest.raises(KeyError, match="Missing column"):
        samplesar.Sample.from_df(pd.DataFrame({"species": ["a"], "x": [1.0], "lat": [1.0]}))
    with pytest.raises(ValueError, match="lat"):
        samplesar.Sample.from_df(pd.DataFrame({"species": ["a"], "lon": [0.0], "lat": [95.0]}))


def test_sample_column_renames():
    df = pd.DataFrame({"taxon": ["a", "b"], "x": [0.0, 1.0], "y": [0.0, 1.0]})
    s = samplesar.Sample.from_df(df, species_col="taxon", lon_col="x", lat_col="y")
    assert list(s.data.columns) == ["species", "lon", "lat"]


def test_random_nest_is_nested_and_monotone(sample):
    nest = samplesar.random_nest(sample, seed=1)
    assert nest.n_steps == sample.n_points - 3 + 1
    for before, after in zip(nest.steps, nest.steps[1:], strict=False):
        assert set(after) < set(before)
        assert len(after) == len(before) - 1
    frame = nest.to_frame()
    assert list(frame.columns) == ["step", "n_points", "area", "species"]
    assert (np.diff(frame["n_points"]) == -1).all()
    assert (np.diff(frame["area"]) <= 1e-12).all()
    assert (np.diff(frame["species"]) <= 0).all()


def test_random_nest_seed_is_reproducible(sample):
    a = samplesar.random_nest(sample, seed=7)
    b = samplesar.random_nest(sample, seed=7)
    c = samplesar.random_nest(sample, seed=8)
    assert all(np.array_equal(x, y) for x, y in zip(a.steps, b.steps, strict=True))
    assert any(not np.array_equal(x, y) for x, y in zip(a.steps, c.steps, strict=True))


def test_random_nest_rejects_small_sample():
    df = pd.DataFrame({"species": ["a", "b"], "lon": [0.0, 1.0], "lat": [0.0, 1.0]})
    with pytest.raises(ValueError, match="at least 3"):
        samplesar.random_nest(samplesar.Sample.from_df(df))


@pytest.mark.parametrize("min_points", [0, -1])
def test_random_nest_rejects_min_points_below_one(sample, min_points):
    with pytest.raises(ValueError, match="min_points must be at least 1"):
        samplesar.random_nest(sample, min_points=min_points)


def test_sar_data_fits_with_sars(sample):
    sars = pytest.importorskip("sars")
    data = samplesar.random_nest(sample, seed=3).sar_data()
    assert list(data.columns) == ["area", "species"]
    assert (data["area"] > 0).all()
    fit = sars.sar_power(data)
    assert fit.converged
    assert 0.0 <= fit.r_squared <= 1.0


def recount(nest):
    """to_frame's species column, recounted from scratch for each step."""
    return [nest.sample_at(i).n_species for i in range(nest.n_steps)]


@pytest.mark.parametrize("seed", range(3))
def test_to_frame_species_match_a_recount(seed):
    rng = np.random.default_rng(seed)
    points = rng.uniform(0, 0.1, size=(30, 2))
    pick = rng.integers(0, 30, size=120)
    sample = samplesar.Sample.from_df(
        pd.DataFrame(
            {
                "species": rng.choice([f"sp{i}" for i in range(25)], size=120),
                "lon": points[pick, 0],
                "lat": points[pick, 1],
            }
        )
    )
    for nest in (samplesar.random_nest(sample, seed=seed), samplesar.hull_peel_nest(sample)):
        frame = nest.to_frame()
        assert list(frame["species"]) == recount(nest)
        assert list(frame["n_points"]) == [len(s) for s in nest.steps]


def test_to_frame_handles_steps_that_are_not_nested(sample):
    # Built by hand, a step may add points back; counts still follow the step.
    steps = (np.arange(10), np.arange(5), np.arange(3, 12))
    nest = samplesar.Nest(sample=sample, steps=steps)
    assert list(nest.to_frame()["species"]) == recount(nest)


class TestNestFromOrder:
    def test_removes_in_order(self, sample):
        order = [5, 0, 17, 3]
        nest = samplesar.nest_from_order(sample, order)
        assert nest.n_steps == 5
        for before, after, point in zip(nest.steps, nest.steps[1:], order, strict=False):
            assert set(before) - set(after) == {point}
        assert nest.label == "order(order=[5, 0, 17, 3])"
        by_name = samplesar.nest(sample, "order", order=order)
        assert all(np.array_equal(a, b) for a, b in zip(nest.steps, by_name.steps, strict=True))

    def test_known_answer(self):
        df = pd.DataFrame(
            {"species": list("abcde"), "lon": [0.0, 1, 2, 3, 4], "lat": [0.0, 1, 0, 1, 0]}
        )
        nest = samplesar.nest_from_order(samplesar.Sample.from_df(df), [4, 0], min_points=1)
        assert [list(s) for s in nest.steps] == [[0, 1, 2, 3, 4], [0, 1, 2, 3], [1, 2, 3]]
        assert list(nest.to_frame()["species"]) == [5, 4, 3]

    def test_long_orders_are_abbreviated_in_labels(self, sample):
        order = np.random.default_rng(0).permutation(sample.n_points)
        label = samplesar.nest_from_order(sample, order).label
        assert label.startswith("order(order=[") and label.endswith(f", ... ({sample.n_points})])")

    def test_whole_number_floats_are_accepted(self, sample):
        a = samplesar.nest_from_order(sample, np.array([3.0, 1.0]))
        b = samplesar.nest_from_order(sample, [3, 1])
        assert all(np.array_equal(x, y) for x, y in zip(a.steps, b.steps, strict=True))

    def test_stops_at_min_points(self, sample):
        order = np.random.default_rng(0).permutation(sample.n_points)
        nest = samplesar.nest_from_order(sample, order, min_points=10)
        assert len(nest.steps[-1]) == 10
        assert set(nest.steps[-1]) == set(order[-10:])

    def test_reproduces_another_nest(self, sample):
        hull = samplesar.hull_area_nest(sample)
        order = [
            int(np.setdiff1d(a, b)[0]) for a, b in zip(hull.steps, hull.steps[1:], strict=False)
        ]
        again = samplesar.nest_from_order(sample, order)
        assert all(np.array_equal(a, b) for a, b in zip(hull.steps, again.steps, strict=True))

    def test_empty_order_is_the_whole_sample(self, sample):
        assert samplesar.nest_from_order(sample, []).n_steps == 1

    @pytest.mark.parametrize(
        ("order", "match"),
        [
            ([1, 1], "repeats"),
            ([0, 40], "outside"),
            ([-1], "outside"),
            ([0.5], "whole numbers"),
            ([[0, 1]], "one-dimensional"),
        ],
    )
    def test_bad_orders(self, sample, order, match):
        with pytest.raises(ValueError, match=match):
            samplesar.nest_from_order(sample, order)
