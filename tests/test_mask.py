"""Masks: areas measured within a region, through samples, nests, saving and plots."""

import dataclasses
import json

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
import shapely  # noqa: E402
from pyproj import Geod  # noqa: E402
from shapely.geometry import MultiPoint, Point, box  # noqa: E402

import samplesar  # noqa: E402

GEOD = Geod(ellps="WGS84")

# A C-shaped region: a square with a bay cut into its east side.
C_SHAPE = box(-0.01, -0.01, 0.11, 0.11).difference(box(0.03, 0.03, 0.2, 0.07))


def geodesic_km2(geometry):
    return abs(GEOD.geometry_area_perimeter(geometry)[0]) / 1e6 if not geometry.is_empty else 0.0


def masked_km2(lonlat, mask):
    """The expected area: the hull of the points, cut by the mask, on the ellipsoid."""
    if len(lonlat) < 3:
        return 0.0
    return geodesic_km2(MultiPoint(lonlat).convex_hull.intersection(mask))


@pytest.fixture
def sample():
    lons = [0.0, 0.1, 0.1, 0.0, 0.05, 0.02, 0.08, 0.09, 0.01, 0.06]
    lats = [0.0, 0.0, 0.1, 0.1, 0.05, 0.02, 0.09, 0.04, 0.08, 0.01]
    species = list("abcdefghij")
    return samplesar.sources.dataframe(pd.DataFrame({"species": species, "lon": lons, "lat": lats}))


def test_the_sample_area_is_its_hull_within_the_mask(sample):
    masked = sample.with_mask(C_SHAPE)
    expected = geodesic_km2(box(0, 0, 0.1, 0.1).difference(box(0.03, 0.03, 0.1, 0.07)))
    assert masked.area == pytest.approx(expected, rel=1e-4)
    assert masked.area < sample.area


def test_every_step_is_measured_within_the_mask(sample):
    masked = sample.with_mask(C_SHAPE)
    nest = samplesar.strategies.hull_peel(masked, min_points=1)
    frame = nest.to_frame()
    for idx, area in zip(nest.steps, frame["area"], strict=True):
        assert area == pytest.approx(masked_km2(sample.points[idx], C_SHAPE), rel=1e-4, abs=1e-9)
    # The whole hull's area comes alongside, as it is without a mask.
    unmasked = samplesar.strategies.hull_peel(sample, min_points=1).to_frame()
    np.testing.assert_allclose(frame["hull_area"], unmasked["area"])
    assert (frame["area"] <= frame["hull_area"] + 1e-12).all()


def test_the_nest_is_unchanged(sample):
    for strategy in ("hull_peel", "hull_area", "radial"):
        plain = samplesar.strategies.run(sample, strategy)
        masked = samplesar.strategies.run(sample.with_mask(C_SHAPE), strategy)
        assert all(np.array_equal(a, b) for a, b in zip(plain.steps, masked.steps, strict=True))


def test_fits_use_the_masked_areas(sample):
    nest = samplesar.strategies.hull_peel(sample.with_mask(C_SHAPE), min_points=1)
    data = nest.sar_data()
    frame = nest.to_frame()
    np.testing.assert_allclose(data["area"], frame.loc[frame["area"] > 0, "area"])


def test_without_a_mask_nothing_changes(sample):
    frame = samplesar.strategies.hull_peel(sample).to_frame()
    assert list(frame.columns) == ["step", "n_points", "area", "species"]
    assert sample.mask is None and sample.mask_xy is None


def test_a_mask_covering_the_hull_changes_no_area(sample):
    nest = samplesar.strategies.hull_peel(sample, min_points=1)
    covered = samplesar.strategies.hull_peel(sample.with_mask(box(-1, -1, 1, 1)), min_points=1)
    np.testing.assert_allclose(covered.to_frame()["area"], nest.to_frame()["area"], rtol=1e-9)


def test_a_lake_is_left_out(sample):
    lake = box(0.04, 0.04, 0.06, 0.06)
    land = box(-1, -1, 1, 1).difference(lake)
    assert sample.with_mask(land).area == pytest.approx(sample.area - geodesic_km2(lake), rel=1e-4)


def test_a_hull_outside_the_mask_has_no_area(sample):
    # Only the far corner is land; a hull of the other points has none.
    masked = sample.with_mask(box(0.095, 0.095, 0.2, 0.2))
    assert masked.hull_area_of([0, 1, 4, 5]) == 0.0
    assert masked.hull_area_of([0, 1]) == 0.0


def test_a_mask_across_the_antimeridian():
    frame = pd.DataFrame(
        {"species": list("abcd"), "lon": [179.9, -179.9, -179.9, 179.9], "lat": [0, 0, 0.2, 0.2]}
    )
    sample = samplesar.sources.dataframe(frame)
    # Land on the west side of the line only: half the hull.
    land = box(179.0, -1, 180.0, 1)
    assert sample.with_mask(land).area == pytest.approx(sample.area / 2, rel=1e-3)


def test_the_mask_follows_the_sample(sample):
    masked = sample.with_mask(C_SHAPE)
    assert masked.subset([0, 1, 2, 4]).mask.equals(C_SHAPE)
    assert masked.merge_within(10).mask.equals(C_SHAPE)
    assert masked.within(box(-1, -1, 1, 1)).mask.equals(C_SHAPE)
    assert masked.with_mask(None).mask is None
    assert "masked" in repr(masked) and "masked" not in repr(sample)


def test_within_can_set_the_mask(sample):
    region = box(-0.01, -0.01, 0.07, 0.11)
    inside = sample.within(region, mask=True)
    assert inside.mask.equals(region)
    assert sample.within(region).mask is None
    assert inside.n_points < sample.n_points


def test_a_geodataframe_mask_in_another_crs(sample):
    gpd = pytest.importorskip("geopandas")
    frame = gpd.GeoDataFrame(geometry=[C_SHAPE], crs=4326).to_crs(3857)
    assert sample.with_mask(frame).area == pytest.approx(sample.with_mask(C_SHAPE).area, rel=1e-6)


@pytest.mark.parametrize("bad", [Point(0, 0), shapely.Polygon(), "land"])
def test_a_mask_must_be_a_polygon(sample, bad):
    with pytest.raises(ValueError, match="boundary"):
        sample.with_mask(bad)


@pytest.mark.parametrize("lon0", [10.0, 179.5])
def test_a_world_mask_reaches_round_the_globe(lon0):
    """A mask covering the far side of the globe from the sample is cut to
    the sample's region, rather than garbled by the projection."""
    lons = (np.array([0.0, 1.0, 1.0, 0.0]) + lon0 + 180) % 360 - 180
    frame = pd.DataFrame({"species": list("abcd"), "lon": lons, "lat": [0.0, 0, 1, 1]})
    sample = samplesar.sources.dataframe(frame)
    world = box(-180, -80, 180, 80)
    assert sample.with_mask(world).area == pytest.approx(sample.area, rel=1e-6)
    lake_lon = (lon0 + 0.6 + 180) % 360 - 180
    lake = box(lake_lon - 0.1, 0.4, lake_lon + 0.1, 0.6)
    assert sample.with_mask(world.difference(lake)).area == pytest.approx(
        sample.area - geodesic_km2(lake), rel=1e-4
    )


def test_a_utm_sample_with_a_world_mask():
    frame = pd.DataFrame(
        {"species": list("abc"), "x": [500_000, 510_000, 500_000], "y": [0, 0, 10_000]}
    )
    sample = samplesar.sources.dataframe(frame, lon_col="x", lat_col="y", crs="EPSG:32617")
    assert sample.with_mask(box(-180, -89, 180, 89)).area == pytest.approx(sample.area)


def test_a_mask_away_from_the_sample(sample):
    with pytest.raises(ValueError, match="doesn't reach"):
        sample.with_mask(box(50, 50, 51, 51))


def test_an_invalid_mask_is_repaired(sample, tmp_path):
    import matplotlib.pyplot as plt

    bowtie = shapely.Polygon([(-0.01, -0.01), (0.11, 0.11), (0.11, -0.01), (-0.01, 0.11)])
    masked = sample.with_mask(shapely.MultiPolygon([bowtie, box(0.2, 0.2, 0.3, 0.3)]))
    assert masked.mask.is_valid and 0 < masked.area < sample.area
    samplesar.plot.nest(samplesar.strategies.hull_peel(masked))
    plt.close("all")


def test_a_mask_with_stray_lines_keeps_only_polygons(sample):
    spiked = shapely.Polygon([(-1, -1), (1, -1), (1, 1), (2, 1), (1, 1), (-1, 1)])
    assert sample.with_mask(spiked).mask.geom_type == "Polygon"
    assert sample.with_mask(spiked).area == pytest.approx(sample.area, rel=1e-6)


def test_plots_across_the_antimeridian_fill_the_whole_masked_hull(tmp_path):
    import matplotlib.pyplot as plt

    frame = pd.DataFrame(
        {"species": list("abcd"), "lon": [179.9, -179.9, -179.9, 179.9], "lat": [0, 0, 0.2, 0.2]}
    )
    sample = samplesar.sources.dataframe(frame).with_mask(box(-180, -80, 180, 80))
    ax = samplesar.plot.nest(samplesar.Nest(sample, (np.arange(4),)))
    filled = shapely.union_all([shapely.Polygon(p.get_path().vertices) for p in ax.patches])
    assert filled.area == pytest.approx(0.2 * 0.2, rel=1e-6)
    plt.close("all")


def test_subsets_reuse_the_projected_mask(sample):
    masked = sample.with_mask(C_SHAPE)
    _ = masked.area
    part = masked.subset([0, 1, 2, 4, 5])
    assert part._mask_plane is masked._mask_plane
    expected = masked_km2(sample.points[[0, 1, 2, 4, 5]], C_SHAPE)
    assert part.area == pytest.approx(expected, rel=1e-4)


def test_diagnostics_give_the_share_inside(sample):
    table = samplesar.evaluate.diagnostics(
        samplesar.strategies.hull_peel(sample.with_mask(C_SHAPE))
    )
    assert table.loc[0, "mask_share"] == pytest.approx(table.loc[0, "area"] / sample.area, rel=1e-9)
    assert "mask_share" not in samplesar.evaluate.diagnostics(
        samplesar.strategies.hull_peel(sample)
    )


def test_a_saved_nest_keeps_its_mask(sample, tmp_path):
    nest = samplesar.strategies.hull_peel(sample.with_mask(C_SHAPE), min_points=1)
    nest.to_json(tmp_path / "nest.json")
    again = samplesar.Nest.from_json(tmp_path / "nest.json")
    assert again.sample.mask.equals(C_SHAPE)
    pd.testing.assert_frame_equal(again.to_frame(), nest.to_frame())


def test_a_version_3_file_has_no_mask(sample):
    data = samplesar.strategies.hull_peel(sample).to_dict()
    assert data["version"] == 3 and "mask" not in data
    assert samplesar.Nest.from_dict(json.loads(json.dumps(data))).sample.mask is None
    masked = samplesar.strategies.hull_peel(sample.with_mask(C_SHAPE)).to_dict()
    assert masked["version"] == 4


def test_a_bad_saved_mask(sample):
    data = samplesar.strategies.hull_peel(sample).to_dict() | {"mask": "garbage"}
    with pytest.raises(ValueError, match="mask is unusable"):
        samplesar.Nest.from_dict(data)


def test_plots_draw_the_mask_and_fill_only_inside(sample, tmp_path):
    import matplotlib.pyplot as plt

    nest = samplesar.strategies.hull_peel(sample.with_mask(C_SHAPE), min_points=1)
    ax = samplesar.plot.nest(nest)
    filled = shapely.union_all([shapely.Polygon(p.get_path().vertices) for p in ax.patches])
    assert filled.area == pytest.approx(
        MultiPoint(sample.points).convex_hull.intersection(C_SHAPE).area, rel=1e-6
    )
    plt.close("all")
    samplesar.plot.animate(nest).save(tmp_path / "n.gif", writer="pillow", fps=10)
    plt.close("all")


def test_replace_validates_the_mask(sample):
    with pytest.raises(ValueError, match="boundary"):
        dataclasses.replace(sample, mask=Point(0, 0))


def test_tiled_areas_match_a_direct_intersection():
    """A jagged coastline of many vertices, as real masks are: the tiles that
    make masked areas fast give the same areas as cutting the whole mask."""
    rng = np.random.default_rng(0)
    angle = np.linspace(0, 2 * np.pi, 3000, endpoint=False)
    radius = 0.5 + 0.1 * rng.random(3000) + 0.15 * np.sin(7 * angle)
    coast = shapely.Polygon(np.column_stack([radius * np.cos(angle), radius * np.sin(angle)]))
    lon, lat = rng.uniform(-0.7, 0.7, (2, 300))
    sample = samplesar.sources.dataframe(
        pd.DataFrame({"species": [f"s{i % 40}" for i in range(300)], "lon": lon, "lat": lat})
    ).with_mask(coast)
    nest = samplesar.strategies.radial(sample, min_points=3)
    mask = sample.mask_xy
    for idx in nest.steps[::17]:
        hull = MultiPoint(sample.xy[idx]).convex_hull
        assert sample.hull_area_of(idx) == pytest.approx(
            mask.intersection(hull).area, rel=1e-9, abs=1e-9
        )
