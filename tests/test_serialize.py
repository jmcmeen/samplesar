"""Saving and loading nests."""

import json

import numpy as np
import pandas as pd
import pytest

import samplesar


@pytest.fixture
def sample():
    rng = np.random.default_rng(0)
    points = rng.uniform(-82.3, -82.2, size=(30, 2))
    pick = rng.integers(0, 30, size=80)
    return samplesar.Sample.from_df(
        pd.DataFrame(
            {
                "species": rng.choice([f"sp{i}" for i in range(15)], size=80),
                "lon": points[pick, 0],
                "lat": points[pick, 1] + 118.8,
            }
        )
    )


def same(a, b):
    assert a.label == b.label and a.strategy == b.strategy
    assert dict(a.options) == dict(b.options)
    assert all(np.array_equal(x, y) for x, y in zip(a.steps, b.steps, strict=True))
    pd.testing.assert_frame_equal(a.sample.data, b.sample.data)
    pd.testing.assert_frame_equal(a.to_frame(), b.to_frame())


@pytest.mark.parametrize(
    "make",
    [
        lambda s: samplesar.random_nest(s, seed=np.int64(4)),
        lambda s: samplesar.hull_peel_nest(s),
        lambda s: samplesar.nest(s, "hull_area", keep="smallest", min_points=5),
        lambda s: samplesar.nest_from_order(s, [3, 1, 4]),
        lambda s: samplesar.Nest(sample=s, steps=(np.arange(s.n_points),)),
    ],
)
def test_round_trip(sample, tmp_path, make):
    nest = make(sample)
    same(samplesar.Nest.from_dict(nest.to_dict()), nest)
    path = tmp_path / "nest.json"
    nest.to_json(path)
    same(samplesar.Nest.from_json(path), nest)


def test_steps_are_stored_as_removals(sample):
    data = samplesar.random_nest(sample, seed=0).to_dict()
    assert data["format"] == "samplesar.nest" and data["version"] == 1
    assert len(data["first"]) == sample.n_points
    assert all(len(gone) == 1 for gone in data["removed"])
    json.dumps(data)


def test_numpy_options_are_saved_plainly(sample):
    data = samplesar.random_nest(sample, seed=np.int64(4)).to_dict()
    assert data["options"] == {"seed": 4} and type(data["options"]["seed"]) is int


def test_steps_that_are_not_nested_cannot_be_saved(sample):
    nest = samplesar.Nest(sample=sample, steps=(np.arange(5), np.arange(3, 8)))
    with pytest.raises(ValueError, match="nested"):
        nest.to_dict()


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"format": "other"}, "Not a saved samplesar nest"),
        ({"version": 99}, "format 99"),
        ({"first": [0, 1, 1000]}, "outside the saved sample"),
        ({"removed": [[1000]]}, "isn't there"),
    ],
)
def test_bad_saved_data(sample, change, match):
    data = {**samplesar.random_nest(sample, seed=0).to_dict(), **change}
    with pytest.raises(ValueError, match=match):
        samplesar.Nest.from_dict(data)
