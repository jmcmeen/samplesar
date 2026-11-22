"""Fitting nests with sars to compare sampling strategies."""

import numpy as np
import pandas as pd
import pytest

import samplesar


@pytest.fixture
def sample():
    rng = np.random.default_rng(0)
    n = 60
    return samplesar.Sample.from_df(
        pd.DataFrame(
            {
                "species": rng.choice([f"sp{i}" for i in range(15)], size=n),
                "lon": rng.uniform(-82.3, -82.2, n),
                "lat": rng.uniform(36.5, 36.6, n),
            }
        )
    )


def test_fit_is_the_sars_fit_of_the_sar_data(sample):
    import sars

    nest = samplesar.random_nest(sample, seed=0)
    fit = nest.fit()
    direct = sars.sar_power(nest.sar_data())
    assert fit.params == pytest.approx(direct.params)
    assert nest.fit("loga").params == pytest.approx(sars.sar_loga(nest.sar_data()).params)


def test_compare_nests(sample):
    nests = {
        "random": samplesar.random_nest(sample, seed=0),
        "peel": samplesar.hull_peel_nest(sample),
        "farthest": samplesar.hull_distance_nest(sample),
    }
    table = samplesar.compare_nests(nests)
    assert list(table.index) == ["random", "peel", "farthest"]
    assert {"n_steps", "n_fitted", "c", "z", "r_squared", "aicc", "converged"} <= set(table.columns)
    assert table.loc["random", "n_fitted"] == len(nests["random"].sar_data())
    assert table.loc["peel", "n_steps"] == nests["peel"].n_steps
    assert table.loc["random", "z"] == pytest.approx(nests["random"].fit().params["z"])


def test_unknown_model(sample):
    nest = samplesar.random_nest(sample, seed=0)
    with pytest.raises(ValueError, match="sars has no model 'nope'"):
        nest.fit("nope")
    with pytest.raises(ValueError, match="sars has no model"):
        samplesar.compare_nests({"a": nest}, model="nope")


def test_compare_needs_a_single_model(sample, monkeypatch):
    # Like sars.sar_average, which returns a model average with no params
    # (a stand-in: the real one fits every model and is slow).
    import sars

    class Averaged:
        pass

    monkeypatch.setattr(sars, "sar_averaged", lambda data: Averaged(), raising=False)
    nest = samplesar.random_nest(sample, seed=0)
    with pytest.raises(ValueError, match="returns Averaged, not a single fit"):
        samplesar.compare_nests({"a": nest}, model="averaged")


def test_compare_needs_nests():
    with pytest.raises(ValueError, match="No nests to compare"):
        samplesar.compare_nests({})


def test_too_few_steps_to_fit_names_the_nest():
    # One inner layer: the peeled nest has two steps, too few to fit.
    square = [[0, 0], [4, 0], [4, 4], [0, 4]]
    inner = [[1, 1], [3, 1], [2, 3]]
    pts = np.array(square + inner, dtype=float) * 0.01
    sample = samplesar.Sample.from_df(
        pd.DataFrame({"species": list("abcdefg"), "lon": pts[:, 0], "lat": pts[:, 1]})
    )
    peel = samplesar.hull_peel_nest(sample)
    assert peel.n_steps == 2
    with pytest.raises(ValueError, match="The nest has 2 step"):
        peel.fit()
    with pytest.raises(ValueError, match="Nest 'peel' has 2 step"):
        samplesar.compare_nests({"random": samplesar.random_nest(sample, seed=0), "peel": peel})


def test_compare_passes_options_to_sars(sample, monkeypatch):
    import sars

    seen = {}

    def fake(data, **kwargs):
        seen.update(kwargs)
        return sars.sar_power(data)

    monkeypatch.setattr(sars, "sar_fake", fake, raising=False)
    samplesar.compare_nests({"a": samplesar.random_nest(sample, seed=0)}, model="fake", flag=1)
    assert seen == {"flag": 1}


def test_fits_at_a_bound_are_flagged(sample, monkeypatch):
    import sars

    nests = {"a": samplesar.random_nest(sample, seed=0), "b": samplesar.hull_peel_nest(sample)}
    real = sars.sar_power

    def bounded_for_a(data):
        fit = real(data)
        stuck = len(data) == nests["a"].n_steps
        return type(
            "Fit",
            (),
            {
                **vars(fit),
                "params": fit.params,
                "r_squared": fit.r_squared,
                "aicc": fit.aicc,
                "converged": fit.converged,
                "at_bounds": ("z",) if stuck else (),
            },
        )()

    monkeypatch.setattr(sars, "sar_power", bounded_for_a)
    with pytest.warns(UserWarning, match=r"stopped on a bound.*a \(z\)"):
        table = samplesar.compare_nests(nests)
    assert table.loc["a", "at_bounds"] == ("z",) and table.loc["b", "at_bounds"] == ()


def test_an_old_sars_leaves_at_bounds_unknown(sample, monkeypatch):
    import sars

    real = sars.sar_power

    def no_attribute(data):
        fit = real(data)
        return type(
            "Fit",
            (),
            {
                "params": fit.params,
                "r_squared": fit.r_squared,
                "aicc": fit.aicc,
                "converged": fit.converged,
            },
        )()

    monkeypatch.setattr(sars, "sar_power", no_attribute)
    table = samplesar.compare_nests({"a": samplesar.random_nest(sample, seed=0)})
    assert table.loc["a", "at_bounds"] is pd.NA
