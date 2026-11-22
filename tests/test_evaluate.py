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
    assert {"n_steps", "c", "z", "r_squared", "aicc", "converged"} <= set(table.columns)
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
