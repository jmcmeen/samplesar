"""Elevations and dates on a sample, and the strategies that rank by them."""

import json

import numpy as np
import pandas as pd
import pytest

import samplesar
from samplesar._gbif import gbif_frame
from samplesar._inat import inaturalist_frame

UTM_17N = "EPSG:32617"
ORIGIN = np.array([500_000.0, 4_000_000.0])


def frame(n=30, seed=0):
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0, 10, (n, 2)) * 1000.0 + ORIGIN
    return pd.DataFrame(
        {
            "species": [f"sp{i % 5}" for i in range(n)],
            "lon": xy[:, 0],
            "lat": xy[:, 1],
            "alt": rng.uniform(200, 2000, n).round(1),
            "seen": pd.Timestamp("2020-01-01") + pd.to_timedelta(rng.permutation(n), unit="D"),
        }
    )


@pytest.fixture
def sample():
    return samplesar.sources.dataframe(frame(), crs=UTM_17N, elevation_col="alt", date_col="seen")


def removed(nest):
    return [int(np.setdiff1d(a, b)[0]) for a, b in zip(nest.steps, nest.steps[1:], strict=False)]


def test_columns_are_kept_only_when_asked_for():
    data = frame().rename(columns={"alt": "elevation", "seen": "date"})
    plain = samplesar.sources.dataframe(data, crs=UTM_17N)
    assert list(plain.data.columns) == ["species", "lon", "lat"]
    with pytest.raises(ValueError, match="no elevations"):
        _ = plain.elevations
    with pytest.raises(ValueError, match="no dates"):
        _ = plain.dates
    kept = samplesar.sources.dataframe(
        data, crs=UTM_17N, elevation_col="elevation", date_col="date"
    )
    assert list(kept.data.columns) == ["species", "lon", "lat", "elevation", "date"]


def test_a_named_column_wins_over_one_already_called_elevation():
    data = frame().assign(elevation=-1.0)
    made = samplesar.sources.dataframe(data, crs=UTM_17N, elevation_col="alt")
    assert made.elevations.min() >= 200


def test_per_point_values_are_the_mean_elevation_and_first_date():
    rows = pd.DataFrame(
        {
            "species": ["a", "b", "c", "a"],
            "lon": [10.0, 10.0, 10.0, 11.0],
            "lat": [50.0, 50.0, 50.0, 50.0],
            "elevation": [100.0, 300.0, np.nan, np.nan],
            "date": ["2021-06-01", "2019-03-05T12:00:00+02:00", None, None],
        }
    )
    made = samplesar.sources.dataframe(rows, elevation_col="elevation", date_col="date")
    assert made.elevations[0] == 200.0 and np.isnan(made.elevations[1])
    assert made.dates[0] == np.datetime64("2019-03-05T10:00:00")  # UTC
    assert np.isnat(made.dates[1])


def test_csv_takes_the_column_names(tmp_path):
    path = tmp_path / "obs.csv"
    degrees = frame().assign(lon=lambda f: f.index * 0.01, lat=lambda f: 35 + f.index * 0.02)
    degrees.to_csv(path, index=False)
    made = samplesar.sources.csv(path, elevation_col="alt", date_col="seen")
    assert made.elevations.shape == (30,) and made.dates.dtype == "datetime64[ns]"


@pytest.mark.parametrize(
    ("column", "values", "error"),
    [
        ("alt", ["high"] * 30, ValueError),
        ("alt", [np.inf] * 30, ValueError),
        ("seen", ["not a date"] * 30, ValueError),
    ],
)
def test_bad_values(column, values, error):
    data = frame()
    data[column] = values
    with pytest.raises(error):
        samplesar.sources.dataframe(data, crs=UTM_17N, elevation_col="alt", date_col="seen")


def test_a_missing_named_column():
    with pytest.raises(KeyError, match="height"):
        samplesar.sources.dataframe(frame(), crs=UTM_17N, elevation_col="height")


def test_subsets_merges_and_saved_nests_keep_them(sample, tmp_path):
    part = sample.subset(range(10))
    assert np.array_equal(part.elevations, sample.elevations[:10])
    assert np.array_equal(part.dates, sample.dates[:10])
    assert "elevation" in sample.merge_within(1.0).data.columns
    nest = samplesar.strategies.elevation(sample)
    path = tmp_path / "nest.json"
    nest.to_json(path)
    saved = json.loads(path.read_text())
    assert saved["version"] == 3  # still read by releases without these columns
    back = samplesar.Nest.from_json(path)
    assert np.allclose(back.sample.elevations, sample.elevations)
    assert np.array_equal(back.sample.dates, sample.dates)
    assert removed(back) == removed(nest)


def test_missing_values_survive_saving(tmp_path):
    data = frame()
    data.loc[3, "alt"] = np.nan
    data.loc[4, "seen"] = pd.NaT
    made = samplesar.sources.dataframe(data, crs=UTM_17N, elevation_col="alt", date_col="seen")
    nest = samplesar.strategies.random(made, seed=1)
    path = tmp_path / "nest.json"
    nest.to_json(path)
    back = samplesar.Nest.from_json(path).sample
    assert np.isnan(back.elevations[3]) and np.isnat(back.dates[4])
    assert np.allclose(np.delete(back.elevations, 3), np.delete(made.elevations, 3))


def test_with_elevations(sample):
    plain = samplesar.sources.dataframe(frame(), crs=UTM_17N)
    heights = np.arange(plain.n_points, dtype=float)
    made = plain.with_elevations(heights)
    assert np.array_equal(made.elevations, heights)
    assert made.crs == plain.crs and np.allclose(made.xy, plain.xy)
    assert np.array_equal(sample.with_elevations(heights).elevations, heights)  # replaced
    for bad in (heights[:-1], np.full(plain.n_points, np.inf), ["x"] * plain.n_points):
        with pytest.raises(ValueError):
            plain.with_elevations(bad)


def test_with_elevations_reaches_every_observation_at_a_point():
    rows = pd.DataFrame(
        {"species": ["a", "b", "a"], "lon": [10.0, 10.0, 11.0], "lat": [50.0, 50.0, 50.0]}
    )
    made = samplesar.sources.dataframe(rows).with_elevations([5.0, 9.0])
    assert made.data["elevation"].tolist() == [5.0, 5.0, 9.0]


@pytest.mark.parametrize("remove", ["highest", "lowest", "outer"])
def test_elevation_removes_by_height(sample, remove):
    nest = samplesar.strategies.elevation(sample, remove=remove, min_points=4)
    height = sample.elevations
    middle = np.median(height)
    for step, point in zip(nest.steps, removed(nest), strict=False):
        if remove == "highest":
            assert height[point] == height[step].max()
        elif remove == "lowest":
            assert height[point] == height[step].min()
        else:
            assert abs(height[point] - middle) == np.abs(height[step] - middle).max()
    assert len(nest.steps[-1]) == 4 and nest.strategy == "elevation"


def test_an_elevation_band_around_a_given_height(sample):
    nest = samplesar.strategies.elevation(sample, remove="outer", around=1500.0, min_points=5)
    left = sample.elevations[nest.steps[-1]]
    gone = np.delete(sample.elevations, nest.steps[-1])
    assert np.abs(left - 1500).max() <= np.abs(gone - 1500).min()


def cone():
    """A cone 1,000 m high and 10 km in radius, sampled on rings; one species
    per 250 m belt."""
    rows = []
    for radius in (1.0, 3.0, 5.0, 7.0, 9.0):
        height = 1000.0 * (1 - radius / 10.0)
        for angle in np.radians(np.arange(0, 360, 30)):
            rows.append(
                {
                    "species": f"belt{int(height // 250)}",
                    "lon": ORIGIN[0] + 1000 * radius * np.cos(angle),
                    "lat": ORIGIN[1] + 1000 * radius * np.sin(angle),
                    "elevation": height,
                }
            )
    return samplesar.sources.dataframe(pd.DataFrame(rows), crs=UTM_17N, elevation_col="elevation")


def test_on_a_cone_from_the_valley_up_shrinks_toward_the_summit():
    nest = samplesar.strategies.elevation(cone(), remove="lowest", min_points=12)
    table = nest.to_frame()
    # Rings go from the outside in: 12 points each, at 100, 300, 500, 700 and 900 m.
    by_points = table.set_index("n_points")
    radii = {60: 9.0, 48: 7.0, 36: 5.0, 24: 3.0, 12: 1.0}
    for n, radius in radii.items():
        dodecagon = 6 * radius**2 * np.sin(np.radians(30))  # 12 points on a circle
        assert by_points.loc[n, "area"] == pytest.approx(dodecagon, rel=2e-3)
    assert by_points.loc[[60, 48, 36, 24, 12], "species"].tolist() == [4, 3, 2, 2, 1]


def test_on_a_cone_from_the_summit_down_keeps_the_area_and_loses_the_top():
    nest = samplesar.strategies.elevation(cone(), remove="highest", min_points=12)
    table = nest.to_frame().set_index("n_points")
    # The outer ring holds the hull throughout: area never changes.
    assert table["area"].min() == pytest.approx(table["area"].max())
    assert table.loc[[60, 48, 36, 24, 12], "species"].tolist() == [4, 3, 3, 2, 1]


def test_elevation_needs_every_point_to_have_one():
    data = frame()
    data.loc[2, "alt"] = np.nan
    made = samplesar.sources.dataframe(data, crs=UTM_17N, elevation_col="alt")
    with pytest.raises(ValueError, match="1 of the sample's 30 points have no elevation"):
        samplesar.strategies.elevation(made)
    with pytest.raises(ValueError, match="no elevations"):
        samplesar.strategies.elevation(samplesar.sources.dataframe(data, crs=UTM_17N))


@pytest.mark.parametrize(
    "options",
    [{"remove": "up"}, {"around": 100.0}, {"remove": "outer", "around": float("nan")}],
)
def test_elevation_bad_options(sample, options):
    with pytest.raises(ValueError):
        samplesar.strategies.elevation(sample, **options)


@pytest.mark.parametrize("remove", ["newest", "oldest"])
def test_temporal_removes_by_first_date(sample, remove):
    nest = samplesar.strategies.temporal(sample, remove=remove, min_points=4)
    when = sample.dates
    for step, point in zip(nest.steps, removed(nest), strict=False):
        assert when[point] == (when[step].max() if remove == "newest" else when[step].min())
    assert nest.strategy == "temporal"


def test_temporal_runs_back_through_a_survey():
    # Three trips: each reached farther east and found a new species.
    rows = []
    for trip, year in enumerate((2001, 2005, 2011)):
        for k in range(4):
            rows.append(
                {
                    "species": f"found{trip}",
                    "lon": ORIGIN[0] + 1000.0 * (trip * 5 + k % 2),
                    "lat": ORIGIN[1] + 1000.0 * (k // 2) * 3,
                    "date": f"{year}-07-0{k + 1}",
                }
            )
    # A later revisit to the first site doesn't make it newer.
    rows.append({**rows[0], "species": "late", "date": "2015-01-01"})
    made = samplesar.sources.dataframe(pd.DataFrame(rows), crs=UTM_17N, date_col="date")
    nest = samplesar.strategies.temporal(made, min_points=4)
    table = nest.to_frame().set_index("n_points")
    assert table.loc[[12, 8, 4], "species"].tolist() == [4, 3, 2]
    assert table.loc[[12, 8, 4], "area"].tolist() == pytest.approx([33.0, 18.0, 3.0], rel=2e-3)
    assert sorted(nest.steps[-1]) == [0, 1, 2, 3]


def test_dates_a_second_apart_are_told_apart():
    data = frame(n=6)
    data["seen"] = pd.Timestamp("1900-01-01")
    data.loc[5, "seen"] = pd.Timestamp("2025-01-01")
    data.loc[2, "seen"] = pd.Timestamp("1900-01-01 00:00:01")
    made = samplesar.sources.dataframe(data, crs=UTM_17N, date_col="seen")
    assert removed(samplesar.strategies.temporal(made, min_points=1))[:3] == [5, 2, 0]


def test_temporal_needs_every_point_to_have_a_date():
    data = frame()
    data.loc[2, "seen"] = pd.NaT
    made = samplesar.sources.dataframe(data, crs=UTM_17N, date_col="seen")
    with pytest.raises(ValueError, match="have no date"):
        samplesar.strategies.temporal(made)
    with pytest.raises(ValueError, match="remove"):
        samplesar.strategies.temporal(made, remove="first")


def test_gbif_records_carry_elevation_and_date():
    records = [
        {
            "species": "Aus bus",
            "decimalLongitude": 1.0,
            "decimalLatitude": 2.0,
            "taxonRank": "SPECIES",
            "elevation": 450.0,
            "eventDate": "2020-05-01T10:00:00",
        },
        {
            "species": "Aus cus",
            "decimalLongitude": 1.5,
            "decimalLatitude": 2.0,
            "taxonRank": "SPECIES",
            "eventDate": "2019-03-02/2019-03-09",
        },
        {
            "species": "Aus dus",
            "decimalLongitude": 1.7,
            "decimalLatitude": 2.0,
            "taxonRank": "SPECIES",
            "elevation": "unknown",
            "eventDate": "sometime",
        },
    ]
    table = gbif_frame(records)
    assert table["elevation"].tolist()[0] == 450.0 and table["elevation"].isna().tolist() == [
        False,
        True,
        True,
    ]
    assert table["date"].tolist()[:2] == [
        pd.Timestamp("2020-05-01 10:00:00"),
        pd.Timestamp("2019-03-02"),
    ]
    assert pd.isna(table["date"][2])
    made = samplesar.sources.dataframe(table, elevation_col="elevation", date_col="date")
    assert made.elevations[0] == 450.0 and np.isnat(made.dates[2])


def test_inaturalist_records_carry_their_date():
    records = [
        {
            "taxon": {"name": "Aus bus", "rank": "species"},
            "geojson": {"coordinates": [1.0, 2.0]},
            "observed_on": "2021-04-03",
        },
        {
            "taxon": {"name": "Aus cus", "rank": "species"},
            "geojson": {"coordinates": [1.5, 2.0]},
        },
    ]
    table = inaturalist_frame(records)
    assert table["date"][0] == pd.Timestamp("2021-04-03") and pd.isna(table["date"][1])
