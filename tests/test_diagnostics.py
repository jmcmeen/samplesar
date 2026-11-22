"""Spatial diagnostics of nests."""

import numpy as np
import pandas as pd
import pytest

import samplesar


def square_with_centre():
    # A 1 km square in UTM with a point at its centre.
    x = [500_000, 501_000, 501_000, 500_000, 500_500]
    y = [4_000_000, 4_000_000, 4_001_000, 4_001_000, 4_000_500]
    return samplesar.Sample.from_df(
        pd.DataFrame({"species": list("abcde"), "lon": x, "lat": y}), crs="EPSG:32617"
    )


def test_nest_diagnostics_on_a_known_shape():
    sample = square_with_centre()
    nest = samplesar.nest_from_order(sample, [2], min_points=3)  # drop one corner
    table = samplesar.nest_diagnostics(nest)
    first, second = table.iloc[0], table.iloc[1]
    assert first["compactness"] == pytest.approx(np.pi / 4, rel=1e-3)  # a square
    assert first["area_share"] == pytest.approx(1.0)
    assert second["area_share"] == pytest.approx(0.5, rel=1e-6)  # half the square
    assert first["perimeter"] == pytest.approx(4 * 0.9996, rel=1e-3)
    assert first["density"] == pytest.approx(5 / first["area"])
    assert first["centroid_shift"] == pytest.approx(0.0, abs=1e-9)
    assert second["centroid_shift"] > 0
    assert list(table["area"]) == pytest.approx(list(nest.to_frame()["area"]))


def test_steps_without_a_polygon():
    sample = square_with_centre()
    table = samplesar.nest_diagnostics(samplesar.random_nest(sample, min_points=1, seed=0))
    last = table.iloc[-1]
    assert last["n_points"] == 1 and np.isnan(last["compactness"]) and last["area"] == 0


def test_residual_autocorrelation():
    rng = np.random.default_rng(0)
    n = 60
    sample = samplesar.Sample.from_df(
        pd.DataFrame(
            {
                "species": rng.choice([f"s{i}" for i in range(20)], n),
                "lon": rng.uniform(-82.3, -82.2, n),
                "lat": rng.uniform(36.5, 36.6, n),
            }
        )
    )
    result = samplesar.residual_autocorrelation(samplesar.hull_peel_nest(sample))
    assert set(result) == {"lag1", "durbin_watson", "n"}
    assert -1 <= result["lag1"] <= 1 and 0 <= result["durbin_watson"] <= 4
    # One-at-a-time nests have strongly autocorrelated residuals.
    smooth = samplesar.residual_autocorrelation(samplesar.random_nest(sample, seed=0))
    assert smooth["lag1"] > 0.5 and smooth["durbin_watson"] < 1
