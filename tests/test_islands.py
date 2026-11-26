"""Regions as sampling units: designs.islands."""

import numpy as np
import pandas as pd
import pytest
import shapely
from shapely.geometry import box

import samplesar
from samplesar._islands import geodesic_area


def observations():
    """Three boxes of 1, 2 and 4 degrees on a side holding 2, 4 and 8
    species, plus one observation outside them all."""
    rng = np.random.default_rng(0)
    rows = []
    for name, west, side, n_species in (("small", 0, 1, 2), ("mid", 10, 2, 4), ("big", 20, 4, 8)):
        for i in range(40):
            lon, lat = rng.uniform(west + 0.05, west + side - 0.05), rng.uniform(0.05, side - 0.05)
            rows.append({"species": f"{name}{i % n_species}", "lon": lon, "lat": lat})
    rows.append({"species": "stray", "lon": 50.0, "lat": 50.0})
    return samplesar.sources.dataframe(pd.DataFrame(rows))


REGIONS = {"small": box(0, 0, 1, 1), "mid": box(10, 0, 12, 2), "big": box(20, 0, 24, 4)}


def test_one_row_per_region():
    made = samplesar.designs.islands(observations(), REGIONS)
    table = made.to_frame()
    assert table["region"].tolist() == ["small", "mid", "big"]
    assert table["species"].tolist() == [2, 4, 8]
    assert table["n_points"].tolist() == [40, 40, 40]
    assert table["n_observations"].tolist() == [40, 40, 40]
    # A square degree at the equator is about 110.6 by 111.3 km.
    assert table["area"].tolist() == pytest.approx([12_309, 49_222, 196_648], rel=2e-3)
    assert "region_area" not in table.columns
    assert made.label == "islands (3 regions)"
    assert isinstance(made, samplesar.Islands)


def test_geodesic_area_is_true_area_at_any_latitude():
    high = geodesic_area(box(0, 60, 1, 61))
    assert high == pytest.approx(111.4 * 55.0, rel=0.02)  # a degree of longitude at 60.5°
    assert geodesic_area(shapely.Polygon()) == 0.0
    # Counterclockwise or clockwise, the area is positive.
    assert geodesic_area(shapely.Polygon(box(0, 60, 1, 61).exterior.coords[::-1])) == pytest.approx(
        high
    )


def test_fitting_the_regions():
    made = samplesar.designs.islands(observations(), REGIONS)
    assert made.sar_data().columns.tolist() == ["area", "species"]
    fit = made.fit("power")
    # Species double as area quadruples: z = 0.5.
    assert fit.params["z"] == pytest.approx(0.5, abs=0.01)
    with pytest.raises(ValueError, match="no model"):
        made.fit("nothing")


def test_empty_regions_are_listed_but_not_fitted():
    regions = {**REGIONS, "empty": box(30, 0, 31, 1)}
    made = samplesar.designs.islands(observations(), regions)
    table = made.to_frame()
    assert table.loc[3, ["species", "n_points", "n_observations"]].tolist() == [0, 0, 0]
    assert len(made.sar_data()) == 3
    assert len(made.sar_data(min_observations=0)) == 4
    assert len(made.sar_data(min_observations=41)) == 0
    with pytest.raises(ValueError, match="0 region"):
        made.fit(min_observations=41)


def test_hull_areas_are_the_ground_the_points_span():
    made = samplesar.designs.islands(observations(), REGIONS, area="hull")
    table = made.to_frame()
    assert (table["area"] < table["region_area"]).all()
    assert (table["area"] > 0.6 * table["region_area"]).all()  # 40 points fill most of a box
    own = made.sample_of("mid")
    assert table["area"][1] == pytest.approx(own.area)
    with pytest.raises(ValueError, match="mask"):
        samplesar.designs.islands(observations(), REGIONS, area="hull", mask=box(0, 0, 30, 1))


def test_a_mask_cuts_each_regions_area():
    land = box(-1, -1, 30, 1)  # the southern strip
    made = samplesar.designs.islands(observations(), REGIONS, mask=land)
    table = made.to_frame()
    assert (table["area"] / table["region_area"]).tolist() == pytest.approx(
        [1.0, 0.5, 0.25], rel=1e-3
    )
    assert table["species"].tolist() == [2, 4, 8]  # the mask changes areas only


def test_a_regions_sample_is_ready_to_nest():
    made = samplesar.designs.islands(observations(), REGIONS)
    own = made.sample_of("big")
    assert own.n_points == 40 and own.n_species == 8 and own.mask is not None
    nest = samplesar.strategies.hull_peel(own)
    assert nest.to_frame()["area"].iloc[0] <= geodesic_area(REGIONS["big"])
    with pytest.raises(KeyError, match="nowhere"):
        made.sample_of("nowhere")
    with pytest.raises(ValueError, match="no observations"):
        samplesar.designs.islands(observations(), {"empty": box(30, 0, 31, 1)}).sample_of("empty")


def test_overlapping_regions_each_count_their_observations():
    regions = {"big": REGIONS["big"], "west half": box(20, 0, 22, 4)}
    table = samplesar.designs.islands(observations(), regions).to_frame()
    assert table["n_points"][0] == 40 and 0 < table["n_points"][1] < 40


def test_several_observations_at_a_point():
    frame = pd.DataFrame(
        {"species": ["a", "b", "a", "c"], "lon": [0.5, 0.5, 0.5, 0.6], "lat": [0.5, 0.5, 0.5, 0.5]}
    )
    made = samplesar.designs.islands(samplesar.sources.dataframe(frame), [box(0, 0, 1, 1)])
    table = made.to_frame()
    assert table.loc[0].tolist()[2:] == [3, 2, 4]
    assert table["region"][0] == "region 0"


def test_regions_from_a_sequence_with_names():
    made = samplesar.designs.islands(observations(), list(REGIONS.values()), names=["a", "b", "c"])
    assert made.to_frame()["region"].tolist() == ["a", "b", "c"]


def test_regions_from_a_geodataframe():
    geopandas = pytest.importorskip("geopandas")
    gdf = geopandas.GeoDataFrame(
        {"park": list(REGIONS), "id": [7, 8, 9]}, geometry=list(REGIONS.values()), crs=4326
    ).to_crs(3857)
    by_column = samplesar.designs.islands(observations(), gdf, names="park")
    assert by_column.to_frame()["region"].tolist() == ["small", "mid", "big"]
    assert by_column.to_frame()["species"].tolist() == [2, 4, 8]
    by_index = samplesar.designs.islands(observations(), gdf.set_index("id"))
    assert by_index.to_frame()["region"].tolist() == ["7", "8", "9"]
    with pytest.raises(KeyError, match="reserve"):
        samplesar.designs.islands(observations(), gdf, names="reserve")
    series = samplesar.designs.islands(observations(), gdf.geometry, names=["x", "y", "z"])
    assert series.to_frame()["species"].tolist() == [2, 4, 8]


@pytest.mark.parametrize(
    ("regions", "options", "match"),
    [
        ({}, {}, "No regions"),
        ([], {}, "No regions"),
        (box(0, 0, 1, 1), {}, "mapping of name"),
        ([box(0, 0, 1, 1)] * 2, {"names": ["a", "a"]}, "repeated: a"),
        ([box(0, 0, 1, 1)] * 2, {"names": ["a"]}, "2 regions but 1 names"),
        ([box(0, 0, 1, 1)], {"names": "park"}, "needs a GeoDataFrame"),
        (REGIONS, {"names": ["a", "b", "c"]}, "already names them"),
        (REGIONS, {"area": "convex"}, "area must be"),
        ({"line": shapely.LineString([(0, 0), (1, 1)])}, {}, "polygon"),
        (REGIONS, {"mask": shapely.Point(0, 0)}, "polygon"),
    ],
)
def test_bad_arguments(regions, options, match):
    with pytest.raises(ValueError, match=match):
        samplesar.designs.islands(observations(), regions, **options)


def test_a_curve_can_be_drawn():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    ax = samplesar.plot.curve(samplesar.designs.islands(observations(), REGIONS))
    assert ax.get_xlabel() == "Area (km²)" and len(ax.lines) == 1


def test_regions_work_as_a_split_too():
    # The same regions can split a nest instead: a different design.
    sample = observations().within(shapely.union_all(list(REGIONS.values())))
    labels = samplesar.strategies.partition(sample, by=list(REGIONS.values()))
    assert np.bincount(labels).tolist() == [40, 40, 40]


def test_areas_do_not_depend_on_ring_winding_and_holes_are_taken_out():
    east, west = box(10, 0, 11, 1), box(0, 0, 1, 1)
    backwards = shapely.Polygon(west.exterior.coords[::-1])
    both = shapely.MultiPolygon([east, backwards])
    assert geodesic_area(both) == pytest.approx(geodesic_area(east) + geodesic_area(west))
    hole = box(0.25, 0.25, 0.75, 0.75)
    for ring in (hole.exterior.coords[:], hole.exterior.coords[::-1]):
        ringed = shapely.Polygon(box(0, 0, 1, 1).exterior.coords, [ring])
        assert geodesic_area(ringed) == pytest.approx(
            geodesic_area(box(0, 0, 1, 1)) - geodesic_area(hole)
        )


def test_a_long_edge_along_a_parallel_is_not_a_geodesic():
    # 49° to 50° N over 12° of longitude: a great circle between the corners
    # bows north of the parallel, which would add area.
    wide = box(-116, 49, -104, 50)
    strips = sum(geodesic_area(box(west, 49, west + 1, 50)) for west in range(-116, -104))
    assert geodesic_area(wide) == pytest.approx(strips, rel=1e-4)


def test_a_region_that_crosses_itself_is_repaired():
    bowtie = shapely.Polygon([(0, 0), (1, 1), (1, 0), (0, 1)])
    made = samplesar.designs.islands(observations(), {"bowtie": bowtie, "big": REGIONS["big"]})
    table = made.to_frame()
    assert table["area"][0] == pytest.approx(geodesic_area(box(0, 0, 1, 1)) / 2, rel=0.01)
    assert 0 < table["n_points"][0] < 40


def test_unusable_regions_are_named():
    with pytest.raises(ValueError, match="Region 'gone' isn't a usable polygon"):
        samplesar.designs.islands(observations(), {"gone": shapely.Polygon()})
    with pytest.raises(ValueError, match="Region 'region 1' isn't a usable polygon"):
        samplesar.designs.islands(observations(), [box(0, 0, 1, 1), None])


def test_regions_from_an_array_or_a_series():
    shapes = list(REGIONS.values())
    for given in (np.array(shapes, dtype=object), pd.Series(shapes)):
        made = samplesar.designs.islands(observations(), given, names=list(REGIONS))
        assert made.to_frame()["species"].tolist() == [2, 4, 8]


def test_a_hull_needs_three_points_off_a_line():
    frame = pd.DataFrame(
        {
            "species": list("abcde"),
            "lon": [0.2, 0.8, 10.5, 11.0, 11.5],
            "lat": [0.5, 0.5, 0.5, 1.0, 1.5],
        }
    )
    made = samplesar.designs.islands(samplesar.sources.dataframe(frame), REGIONS, area="hull")
    area = made.to_frame()["area"].tolist()
    assert area[0] == 0.0 and area[2] == 0.0  # two points; none
    # Three points in a line of longitude and latitude bend a little on the
    # projection: a sliver, next to a region of 49,000 km².
    assert area[1] < 5.0


def test_the_table_is_computed_once_and_copies_are_independent():
    made = samplesar.designs.islands(observations(), REGIONS)
    table = made.to_frame()
    table["species"] = 0
    assert made.to_frame()["species"].tolist() == [2, 4, 8]
    assert made != samplesar.designs.islands(observations(), REGIONS)  # compared by identity
    with pytest.raises(ValueError, match="area must be"):
        samplesar.Islands(observations(), REGIONS, area="bogus")


def test_many_regions_and_points_are_searched_together():
    rng = np.random.default_rng(1)
    n = 20_000
    frame = pd.DataFrame(
        {
            "species": [f"s{i}" for i in rng.integers(0, 200, n)],
            "lon": rng.uniform(0, 30, n),
            "lat": rng.uniform(0, 30, n),
        }
    )
    cells = {f"{x},{y}": box(x, y, x + 1, y + 1) for x in range(30) for y in range(30)}
    table = samplesar.designs.islands(samplesar.sources.dataframe(frame), cells).to_frame()
    assert len(table) == 900 and table["n_observations"].sum() == n
    inside = frame[(frame["lon"] < 1) & (frame["lat"] < 1)]
    assert table["species"][0] == inside["species"].nunique()
