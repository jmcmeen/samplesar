"""sources.simulated: a made-up landscape for the examples."""

import numpy as np
import pytest

import samplesar


def test_the_same_seed_gives_the_same_sample():
    a = samplesar.sources.simulated(seed=3)
    b = samplesar.sources.simulated(seed=3)
    c = samplesar.sources.simulated(seed=4)
    assert a.data.equals(b.data) and not a.data.equals(c.data)
    assert a.n_points == 600 and 20 < a.n_species <= 60


def test_every_point_has_a_record_an_elevation_and_a_date():
    sample = samplesar.sources.simulated(n_points=200, n_species=30, seed=1)
    assert sample.n_points == 200
    assert np.isfinite(sample.elevations).all() and not np.isnat(sample.dates).any()
    assert 150 < sample.elevations.min() < 500 and 1500 < sample.elevations.max() < 2000
    assert sample.dates.min() >= np.datetime64("2005-01-01")
    assert sample.dates.max() < np.datetime64("2025-01-01")
    # 80 km on a side, give or take the points' reach.
    assert 4000 < sample.area < 6500


def test_the_square_can_be_moved_and_sized():
    sample = samplesar.sources.simulated(n_points=150, side_km=20, center=(10.0, 60.0), seed=2)
    lon, lat = sample.points.T
    assert 9.8 < lon.min() and lon.max() < 10.2 and 59.9 < lat.min() and lat.max() < 60.1
    assert 250 < sample.area < 420


def test_clustered_points_pile_up():
    def nearest(sample):
        from scipy.spatial import cKDTree

        return np.median(cKDTree(sample.xy).query(sample.xy, k=2)[0][:, 1])

    even = samplesar.sources.simulated(clustered=0.0, seed=5)
    piled = samplesar.sources.simulated(clustered=1.0, seed=5)
    assert nearest(piled) < 0.6 * nearest(even)


def test_it_nests_by_every_kind_of_strategy():
    sample = samplesar.sources.simulated(n_points=120, seed=6)
    for name, options in (
        ("hull_peel", {}),
        ("elevation", {}),
        ("temporal", {}),
        ("hull_rarity", {}),
        ("split", {"parts": 3, "seed": 0}),
    ):
        nest = samplesar.strategies.run(sample, name, **options)
        assert np.isfinite(nest.fit("power").params["z"]), name


@pytest.mark.parametrize(
    "options",
    [
        {"n_points": 0},
        {"n_species": 0},
        {"side_km": 0},
        {"clustered": 1.5},
        {"center": (0.0, 89.0)},
        {"center": (179.5, 0.0)},
        {"n_points": 10.5},
        {"n_points": float("nan")},
        {"seed": None},
        {"side_km": 5000},
    ],
)
def test_bad_arguments(options):
    with pytest.raises(ValueError):
        samplesar.sources.simulated(**options)


def test_the_default_landscape_is_pinned():
    # The examples and notebooks quote these: a change here means re-running them.
    sample = samplesar.sources.simulated(seed=1)
    assert (sample.n_points, sample.n_species, sample.n_observations) == (600, 49, 1023)
    fit = samplesar.strategies.hull_peel(sample.merge_within(20)).fit("power")
    assert fit.params["z"] == pytest.approx(0.465, abs=0.003)


def test_a_place_with_no_luck_records_a_species_whose_range_reaches_it():
    sample = samplesar.sources.simulated(n_points=400, n_species=8, seed=3)
    assert sample.n_points == 400  # every place has a record
