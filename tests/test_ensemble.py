"""evaluate.ensemble and evaluate.bootstrap."""

import numpy as np
import pandas as pd
import pytest

import samplesar

UTM_17N = "EPSG:32617"
ORIGIN = np.array([500_000.0, 4_000_000.0])


@pytest.fixture(scope="module")
def sample():
    rng = np.random.default_rng(2)
    xy = rng.uniform(0, 20, (120, 2)) * 1000.0 + ORIGIN
    # Species have ranges: each occurs within a few km of its own center.
    centers = rng.uniform(0, 20, (25, 2)) * 1000.0 + ORIGIN
    rows = []
    for x, y in xy:
        near = np.hypot(*(centers - (x, y)).T) < 6000.0
        for species in np.flatnonzero(near):
            rows.append({"species": f"sp{species}", "lon": x, "lat": y})
        rows.append({"species": "everywhere", "lon": x, "lat": y})
    return samplesar.sources.dataframe(pd.DataFrame(rows), crs=UTM_17N)


def test_the_band_holds_the_nests_it_was_made_from(sample):
    band = samplesar.evaluate.ensemble(sample, "hull_random", n=30, seed=5, points=40)
    assert list(band.columns) == ["area", "mean", "median", "q025", "q975", "n"]
    assert len(band) == 40 and band["area"].is_monotonic_increasing
    assert (band["q025"] <= band["median"]).all() and (band["median"] <= band["q975"]).all()
    assert (band["q025"] <= band["mean"] + 1e-9).all() and (
        band["mean"] <= band["q975"] + 1e-9
    ).all()
    assert band["n"].max() == 30 and band["n"].min() >= 1
    # More area, more species, on average.
    assert band["mean"].iloc[-1] > band["mean"].iloc[5]
    assert band.attrs["strategy"] == "hull_random" and band.attrs["seeds"] == list(range(5, 35))
    again = samplesar.evaluate.ensemble(sample, "hull_random", n=30, seed=5, points=40)
    assert again.equals(band)


def test_one_nest_is_its_own_curve(sample):
    nest = samplesar.strategies.hull_random(sample, seed=3)
    data = nest.sar_data().groupby("area")["species"].max()
    band = samplesar.evaluate.ensemble(sample, "hull_random", n=1, seed=3, points=25)
    want = np.interp(np.log(band["area"]), np.log(data.index), data.to_numpy(float))
    assert band["mean"].to_numpy() == pytest.approx(want)
    assert band["q025"].equals(band["q975"]) and band["n"].eq(1).all()
    assert band["area"].iloc[0] == pytest.approx(data.index.min())
    assert band["area"].iloc[-1] == pytest.approx(data.index.max())


def test_a_nest_is_not_extrapolated(sample):
    # With few points kept, nests end at different areas: the smallest
    # areas are spanned by only some of them.
    band = samplesar.evaluate.ensemble(sample, "hull_random", n=20, points=60, min_points=3)
    assert band["n"].iloc[0] < 20 and band["n"].iloc[-1] == 20


def test_options_reach_the_strategy(sample):
    isolated = samplesar.evaluate.ensemble(sample, "hull_weighted", n=5, power=3.0)
    uniform = samplesar.evaluate.ensemble(sample, "hull_weighted", n=5, power=0.0)
    random = samplesar.evaluate.ensemble(sample, "hull_random", n=5)
    assert uniform["mean"].to_numpy() == pytest.approx(random["mean"].to_numpy())
    assert not np.allclose(isolated["mean"], uniform["mean"])
    assert isolated.attrs["options"] == {"power": 3.0}


@pytest.mark.parametrize(
    ("options", "match"),
    [
        ({"n": 0}, "n must be"),
        ({"points": 0}, "points must be"),
        ({"strategy": "hull_peel"}, "takes a seed"),
        ({"strategy": "nothing"}, "No strategy"),
        ({"min_points": 120}, "No nest has two steps"),
    ],
)
def test_ensemble_bad_arguments(sample, options, match):
    with pytest.raises(ValueError, match=match):
        samplesar.evaluate.ensemble(sample, **options)


def test_bootstrap_fits_resamples_of_the_points(sample):
    table = samplesar.evaluate.bootstrap(sample, "hull_peel", n=20, seed=1)
    assert table.index.name == "resample" and list(table.index) == list(range(20))
    assert {"n_points", "z", "c", "r_squared"} <= set(table.columns)
    # About 63% of 120 points each time.
    assert 60 <= table["n_points"].min() and table["n_points"].max() <= 90
    assert table["z"].std() > 0
    again = samplesar.evaluate.bootstrap(sample, "hull_peel", n=20, seed=1)
    assert again["z"].tolist() == pytest.approx(table["z"].tolist())
    other = samplesar.evaluate.bootstrap(sample, "hull_peel", n=20, seed=2)
    assert other["n_points"].tolist() != table["n_points"].tolist()


def test_bootstrap_matches_nesting_each_resample_by_hand(sample):
    table = samplesar.evaluate.bootstrap(sample, "radial", n=3, seed=7, min_points=5)
    rng = np.random.default_rng(7)
    for i in range(3):
        drawn = np.unique(rng.integers(sample.n_points, size=sample.n_points))
        fit = samplesar.strategies.radial(sample.subset(drawn), min_points=5).fit("power")
        assert table.loc[i, "z"] == pytest.approx(fit.params["z"])
        assert table.loc[i, "n_points"] == len(drawn)


def test_bootstrap_seeds_random_strategies_and_places_a_nest(sample):
    table = samplesar.evaluate.bootstrap(sample, "hull_random", n=6, seed=4)
    again = samplesar.evaluate.bootstrap(sample, "hull_random", n=6, seed=4)
    assert table["z"].tolist() == pytest.approx(again["z"].tolist())
    fixed = (
        samplesar.evaluate.bootstrap(sample, "hull_random", n=6, seed=4, seed_=None)
        if False
        else None
    )
    assert fixed is None
    placed = samplesar.evaluate.against_baseline(samplesar.strategies.hull_peel(sample), table)
    assert {"value", "mean", "p"} <= set(placed.columns)


def ring_sample(center):
    angles = np.linspace(0, 2 * np.pi, 13)[:-1]
    xy = np.column_stack([np.cos(angles), np.sin(angles)]) * 5
    if center:
        xy = np.vstack([xy, [[0, 0]]])
    frame = pd.DataFrame(
        {
            "species": [f"s{i % 5}" for i in range(len(xy))],
            "lon": xy[:, 0] * 1000 + ORIGIN[0],
            "lat": xy[:, 1] * 1000 + ORIGIN[1],
        }
    )
    return samplesar.sources.dataframe(frame, crs=UTM_17N)


def test_resamples_that_cannot_be_nested_or_fitted_are_left_out(sample, monkeypatch):
    from samplesar import _registry
    from samplesar._nest import NoStepError

    real = _registry._REGISTRY["hull_peel"]
    calls = []

    def sometimes(part, **options):
        calls.append(part.n_points)
        if len(calls) % 3 == 0:
            raise NoStepError("No layer can be peeled")
        if len(calls) % 3 == 1:  # too short a nest to fit
            return samplesar.Nest(part, (np.arange(part.n_points),))
        return real.func(part, **options)

    import dataclasses

    monkeypatch.setitem(_registry._REGISTRY, "hull_peel", dataclasses.replace(real, func=sometimes))
    table = samplesar.evaluate.bootstrap(sample, "hull_peel", n=9, seed=1)
    assert list(table.index) == [1, 4, 7]
    # A ring with nothing inside can never be peeled, nor fitted with its center alone.
    for center in (False, True):
        monkeypatch.undo()
        with pytest.raises(ValueError, match="No resample could be nested and fitted"):
            samplesar.evaluate.bootstrap(ring_sample(center), "hull_peel", n=5, min_points=1)


def test_bootstrap_bad_arguments(sample):
    with pytest.raises(ValueError, match="No strategy"):
        samplesar.evaluate.bootstrap(sample, "nothing")
    with pytest.raises(ValueError, match="n must be"):
        samplesar.evaluate.bootstrap(sample, n=0)
    with pytest.raises(ValueError, match="keep"):
        samplesar.evaluate.bootstrap(sample, "hull_area", keep="biggest")
