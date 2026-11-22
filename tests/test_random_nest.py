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
