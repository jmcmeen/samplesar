"""Saving and loading nests."""

import dataclasses
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
    assert data["format"] == "samplesar.nest" and data["version"] == 3
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
        ({"center": [1]}, "center, crs or mask is unusable"),
        ({"center": None, "centre": [1], "version": 2}, "center, crs or mask is unusable"),
        ({"crs": "garbage"}, "center, crs or mask is unusable"),
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


def test_round_trip_keeps_the_crs_and_center(tmp_path):
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
        assert again.sample.crs == sample.crs and again.sample.center == sample.center
        pd.testing.assert_frame_equal(again.to_frame(), nest.to_frame())


def test_reads_format_version_1(sample):
    data = samplesar.strategies.random(sample, seed=0).to_dict()
    old = {k: v for k, v in data.items() if k not in ("center", "crs")} | {"version": 1}
    assert samplesar.Nest.from_dict(old).n_steps == sample.n_points - 2


def test_a_version_2_file_with_a_centre_key_loads():
    """Before version 3 the sample's center was saved under "centre"."""
    frame = pd.DataFrame(
        {"species": ["a", "b", "c", "d"], "lon": [0.0, 1.0, 0.0, 1.0], "lat": [0.0, 0.0, 1.0, 1.0]}
    )
    sample = dataclasses.replace(samplesar.sources.dataframe(frame), center=(0.5, 0.5))
    nest = samplesar.strategies.random(sample, seed=0, min_points=3)
    data = nest.to_dict()
    data["version"] = 2
    data["centre"] = data.pop("center")
    loaded = samplesar.Nest.from_dict(json.loads(json.dumps(data)))
    assert loaded.sample.center == (0.5, 0.5)


def test_a_version_2_centre_option_is_read_as_center():
    frame = pd.DataFrame(
        {"species": list("abcdef"), "lon": [0.0, 1, 2, 0, 1, 2], "lat": [0.0, 0, 0, 1, 1, 1]}
    )
    nest = samplesar.strategies.radial(samplesar.sources.dataframe(frame), center=(1.0, 0.5))
    data = nest.to_dict()
    data["version"] = 2
    data["options"] = {"centre": data["options"].pop("center")}
    loaded = samplesar.Nest.from_dict(json.loads(json.dumps(data)))
    rerun = samplesar.strategies.run(loaded.sample, loaded.strategy, **loaded.options)
    assert all(np.array_equal(a, b) for a, b in zip(rerun.steps, nest.steps, strict=True))
