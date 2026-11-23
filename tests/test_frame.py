"""Nest.to_frame, kept incrementally, against a recount of every step."""

import numpy as np
import pandas as pd
import pytest
import shapely

import samplesar


def recount(nest):
    """Each step's areas and species, computed from scratch."""
    rows = []
    for idx in nest.steps:
        area, whole = nest.sample.areas_of(idx)
        rows.append((area, whole, nest.sample.subset(idx).n_species if len(idx) else 0))
    return np.array(rows)


@pytest.fixture(scope="module")
def sample():
    rng = np.random.default_rng(3)
    n = 120
    frame = pd.DataFrame(
        {
            "species": rng.choice([f"sp{i}" for i in range(30)], n),
            "lon": rng.uniform(-82.3, -82.2, n),
            "lat": rng.uniform(36.5, 36.6, n),
        }
    )
    return samplesar.sources.dataframe(frame)


COAST = shapely.box(-82.35, 36.45, -82.15, 36.65).difference(
    shapely.box(-82.25, 36.53, -82.1, 36.57)
)


@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize(
    "strategy",
    [
        "random",
        "hull_random",
        "hull_peel",
        "grid_peel",
        "concave_peel",
        "hull_area",
        "hull_masked_area",
        "hull_species",
        "hull_distance",
        "radial",
        "sweep",
        "grow",
    ],
)
def test_every_step_matches_a_recount(sample, strategy, masked):
    if masked:
        sample = sample.with_mask(COAST)
    elif strategy == "hull_masked_area":
        pytest.skip("hull_masked_area needs a mask")
    options = {"seed": 0} if strategy in ("random", "hull_random") else {}
    if strategy == "grid_peel":
        options = {"cell_km": 0.5}
    nest = samplesar.strategies.run(sample, strategy, **options)
    frame = nest.to_frame()
    expected = recount(nest)
    np.testing.assert_allclose(frame["area"], expected[:, 0], rtol=1e-9, atol=1e-12)
    if masked:
        np.testing.assert_allclose(frame["hull_area"], expected[:, 1], rtol=1e-9, atol=1e-12)
    np.testing.assert_array_equal(frame["species"], expected[:, 2])


@pytest.mark.parametrize("masked", [False, True])
def test_steps_that_add_points_back(sample, masked):
    """A nest built by hand needn't be nested: points can come back."""
    if masked:
        sample = sample.with_mask(COAST)
    steps = (
        np.arange(120),
        np.arange(60),
        np.arange(30, 90),  # adds 30 points, drops 30
        np.array([5, 70, 100, 110]),
        np.arange(120),  # everything back
        np.array([0, 1]),  # no polygon
        np.array([0, 1, 2, 3]),
    )
    nest = samplesar.Nest(sample, steps)
    frame = nest.to_frame()
    expected = recount(nest)
    np.testing.assert_allclose(frame["area"], expected[:, 0], rtol=1e-9, atol=1e-12)
    if masked:
        np.testing.assert_allclose(frame["hull_area"], expected[:, 1], rtol=1e-9, atol=1e-12)
    np.testing.assert_array_equal(frame["species"], expected[:, 2])


def test_collinear_steps_have_no_area():
    # A line in UTM meters: straight but for the rounding of the round trip
    # through longitude and latitude, so its hull has (next to) no area.
    line = samplesar.sources.dataframe(
        pd.DataFrame(
            {
                "species": list("abcde"),
                "x": [500_000.0, 500_100.0, 500_200.0, 500_300.0, 500_150.0],
                "y": [4_000_000.0] * 4 + [4_000_050.0],
            }
        ),
        lon_col="x",
        lat_col="y",
        crs="EPSG:32617",
    )
    nest = samplesar.Nest(line, (np.arange(5), np.arange(4), np.arange(3)))
    frame = nest.to_frame()
    assert frame["area"][0] > 1e-3
    assert (frame["area"][1:] < 1e-9).all()
    np.testing.assert_allclose(frame["area"], recount(nest)[:, 0], rtol=1e-9, atol=1e-15)
