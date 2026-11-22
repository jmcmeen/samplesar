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
    return samplesar.sources.dataframe(
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
        lambda s: samplesar.strategies.random(s, seed=np.int64(4)),
        lambda s: samplesar.strategies.hull_peel(s),
        lambda s: samplesar.strategies.run(s, "hull_area", keep="smallest", min_points=5),
        lambda s: samplesar.strategies.from_order(s, [3, 1, 4]),
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
    data = samplesar.strategies.random(sample, seed=0).to_dict()
    assert data["format"] == "samplesar.nest" and data["version"] == 2
    assert len(data["first"]) == sample.n_points
    assert all(len(gone) == 1 for gone in data["removed"])
    json.dumps(data)


def test_numpy_options_are_saved_plainly(sample):
    data = samplesar.strategies.random(sample, seed=np.int64(4)).to_dict()
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
        ({"first": [0, 0, 1]}, "repeats a point"),
        ({"first": [0.5, 1]}, "whole-number point indices"),
        ({"version": "2"}, "whole number"),
        ({"centre": [1]}, "centre or crs is unusable"),
        ({"crs": "garbage"}, "centre or crs is unusable"),
        ({"observations": None, "first": None}, None),
    ],
)
def test_bad_saved_data(sample, change, match):
    data = {**samplesar.strategies.random(sample, seed=0).to_dict(), **change}
    if match is None:  # keys missing altogether
        data = {k: v for k, v in data.items() if change.get(k, 1) is not None}
        match = "Not a saved samplesar nest"
    with pytest.raises(ValueError, match=match):
        samplesar.Nest.from_dict(data)


def test_options_that_json_cant_hold(sample):
    nest = samplesar.strategies.random(sample, seed=np.random.default_rng(1))
    with pytest.raises(ValueError, match="Option seed=Generator"):
        nest.to_dict()


def test_nested_options_with_numpy_values(sample):
    nest = samplesar.Nest(sample=sample, steps=(np.arange(sample.n_points),), strategy="x")
    from samplesar._registry import Options

    nest = samplesar.Nest(
        sample=sample,
        steps=nest.steps,
        strategy="x",
        options=Options({"weights": {"a": np.float64(0.5)}, "pair": (1, 2)}),
    )
    saved = nest.to_dict()["options"]
    assert saved == {"weights": {"a": 0.5}, "pair": [1, 2]}
    json.dumps(saved)


def test_round_trip_keeps_the_crs_and_centre(tmp_path):
    x = [500_000, 501_000, 501_000, 500_000, 500_700]
    y = [4_000_000, 4_000_000, 4_001_000, 4_001_000, 4_000_300]
    utm = samplesar.sources.dataframe(
        pd.DataFrame({"species": list("abcde"), "lon": x, "lat": y}), crs="EPSG:32617"
    )
    for sample in (
        utm,
        utm.subset([0, 1, 2, 4]),
        samplesar.sources.dataframe(utm.data).subset([0, 1, 2, 4]),
    ):
        nest = samplesar.strategies.hull_peel(sample, min_points=1)
        again = samplesar.Nest.from_dict(nest.to_dict())
        assert again.sample.crs == sample.crs and again.sample.centre == sample.centre
        pd.testing.assert_frame_equal(again.to_frame(), nest.to_frame())


def test_reads_format_version_1(sample):
    data = samplesar.strategies.random(sample, seed=0).to_dict()
    old = {k: v for k, v in data.items() if k not in ("centre", "crs")} | {"version": 1}
    assert samplesar.Nest.from_dict(old).n_steps == sample.n_points - 2
