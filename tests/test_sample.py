"""Samples: validation, subsets, projection, coordinate systems and hull areas."""

import numpy as np
import pandas as pd
import pytest
from pyproj import Geod

import samplesar


def frame(lons, lats, species=None):
    species = species if species is not None else [f"sp{i}" for i in range(len(lons))]
    return pd.DataFrame({"species": species, "lon": lons, "lat": lats})


def box(lon0, lat0, d=1.0):
    lons, lats = [lon0, lon0 + d, lon0 + d, lon0], [lat0, lat0, lat0 + d, lat0 + d]
    return samplesar.Sample.from_df(frame(lons, lats)), lons, lats


@pytest.mark.parametrize("lat0", [0, 45, 60, 80, -50])
def test_area_is_km2_on_the_ellipsoid(lat0):
    sample, lons, lats = box(-82.0, lat0)
    geodesic = abs(Geod(ellps="WGS84").polygon_area_perimeter(lons, lats)[0]) / 1e6
    # Straight edges on the projection versus geodesic edges: a few 1e-5.
    assert sample.area == pytest.approx(geodesic, rel=1e-4)


def test_area_shrinks_towards_the_poles():
    assert box(0, 60)[0].area == pytest.approx(box(0, 0)[0].area / 2, rel=0.02)


def test_a_sample_across_the_antimeridian():
    across = samplesar.Sample.from_df(frame([179.5, -179.5, -179.5, 179.5], [10, 10, 11, 11]))
    here = samplesar.Sample.from_df(frame([0, 1, 1, 0], [10, 10, 11, 11]))
    assert across.area == pytest.approx(here.area, rel=1e-9)
    assert abs(across.projection_centre[0]) == pytest.approx(180)


def test_a_subset_keeps_the_projection():
    sample = samplesar.Sample.from_df(frame([0, 1, 1, 0, 0.2], [0, 0, 1, 1, 0.3]))
    part = sample.subset([0, 1, 4])
    assert part.projection_centre == sample.projection_centre
    np.testing.assert_allclose(part.xy, sample.xy[[0, 1, 4]])


def test_xy_and_points_are_read_only():
    sample = box(0, 0)[0]
    for array in (sample.points, sample.xy):
        with pytest.raises(ValueError):
            array[0, 0] = 1.0


class TestCRS:
    def test_projected_coordinates_are_measured_in_their_crs(self):
        # A square 1 km a side in UTM zone 17N metres. UTM shrinks distances by
        # 0.9996 near its central meridian, so on the ground it's 1.0008 km².
        x, y = [500_000, 501_000, 501_000, 500_000], [4_000_000, 4_000_000, 4_001_000, 4_001_000]
        sample = samplesar.Sample.from_df(frame(x, y), crs="EPSG:32617")
        assert sample.area == pytest.approx(1 / 0.9996**2, rel=1e-5)
        geodesic = abs(
            Geod(ellps="WGS84").polygon_area_perimeter(sample.points[:, 0], sample.points[:, 1])[0]
        )
        assert sample.area == pytest.approx(geodesic / 1e6, rel=1e-5)
        assert sample.crs.to_epsg() == 32617
        # data holds WGS84 longitude and latitude.
        assert sample.data["lon"].between(-82, -80).all()
        assert sample.data["lat"].between(36, 37).all()
        np.testing.assert_allclose(
            sample.xy - sample.xy[0], [[0, 0], [1, 0], [1, 1], [0, 1]], atol=1e-9
        )

    def test_a_geographic_crs_is_converted_to_wgs84(self):
        sample = samplesar.Sample.from_df(
            frame([-82.0, -81.0, -81.0], [36.0, 36.0, 37.0]), crs=4269
        )
        assert sample.crs is None
        assert sample.data["lon"].iloc[0] == pytest.approx(-82.0, abs=1e-4)

    def test_a_bad_crs(self):
        with pytest.raises(ValueError, match="Not a coordinate reference system"):
            samplesar.Sample.from_df(frame([0, 1, 1], [0, 0, 1]), crs="EPSG:nope")

    def test_a_subset_keeps_the_crs(self):
        x, y = [500_000, 501_000, 501_000, 500_000], [4_000_000, 4_000_000, 4_001_000, 4_001_000]
        sample = samplesar.Sample.from_df(frame(x, y), crs="EPSG:32617")
        assert sample.subset([0, 1, 2]).area == pytest.approx(sample.area / 2, rel=1e-9)

    def test_web_mercator_areas_are_corrected(self):
        # A Web Mercator square near 60°N enlarges areas about 4x; the area is
        # corrected to match the same points measured on the default projection.
        from pyproj import Transformer

        lon, lat = [10.0, 10.02, 10.02, 10.0], [60.0, 60.0, 60.01, 60.01]
        x, y = Transformer.from_crs(4326, 3857, always_xy=True).transform(lon, lat)
        mercator = samplesar.Sample.from_df(frame(x, y), crs=3857)
        plain = samplesar.Sample.from_df(frame(lon, lat))
        assert mercator.area_scale == pytest.approx(4.0, rel=0.01)
        # Web Mercator treats WGS84 coordinates as if on a sphere, so this is
        # only good to a few 0.1% (uncorrected it would be 4x).
        assert mercator.area == pytest.approx(plain.area, rel=0.01)

    def test_a_wide_sample_in_a_distorting_crs_warns(self):
        from pyproj import Transformer

        lon, lat = [0.0, 10.0, 10.0, 0.0], [40.0, 40.0, 60.0, 60.0]
        x, y = Transformer.from_crs(4326, 3857, always_xy=True).transform(lon, lat)
        sample = samplesar.Sample.from_df(frame(x, y), crs=3857)
        with pytest.warns(UserWarning, match="distorts areas"):
            _ = sample.area

    def test_a_crs_that_isnt_for_points(self):
        with pytest.raises(ValueError, match="neither geographic nor projected"):
            samplesar.Sample.from_df(frame([0, 1, 1], [0, 0, 1]), crs="EPSG:5703")

    def test_a_numpy_centre(self):
        sample = samplesar.Sample.from_df(frame([0, 1, 1], [35, 35, 36]))
        moved = samplesar.Sample(sample.data, centre=(np.float64(0.5), np.float64(35.5)))
        assert moved.centre == (0.5, 35.5) and type(moved.centre[0]) is float
        assert moved.area == pytest.approx(sample.area, rel=1e-3)

    def test_a_bad_centre(self):
        sample = samplesar.Sample.from_df(frame([0, 1, 1], [35, 35, 36]))
        with pytest.raises(ValueError, match="centre must be"):
            samplesar.Sample(sample.data, centre=(0.0, 95.0))


class TestFromGeoDataFrame:
    geopandas = pytest.importorskip("geopandas")

    def test_points_and_crs(self):
        gpd = self.geopandas
        x, y = [500_000, 501_000, 501_000, 500_000], [4_000_000, 4_000_000, 4_001_000, 4_001_000]
        gdf = gpd.GeoDataFrame(
            {"taxon": list("abcd")}, geometry=gpd.points_from_xy(x, y), crs="EPSG:32617"
        )
        sample = samplesar.Sample.from_geodataframe(gdf, species_col="taxon")
        assert sample.n_species == 4 and sample.area == pytest.approx(1 / 0.9996**2, rel=1e-5)

    def test_no_crs_means_lon_lat(self):
        gpd = self.geopandas
        gdf = gpd.GeoDataFrame(
            {"species": list("abc")}, geometry=gpd.points_from_xy([0, 1, 1], [0, 0, 1])
        )
        assert samplesar.Sample.from_geodataframe(gdf).crs is None

    def test_only_points(self):
        gpd = self.geopandas
        from shapely.geometry import LineString, Point

        gdf = gpd.GeoDataFrame(
            {"species": ["a", "b"]}, geometry=[Point(0, 0), LineString([(0, 0), (1, 1)])]
        )
        with pytest.raises(ValueError, match="must be a point"):
            samplesar.Sample.from_geodataframe(gdf)

    def test_missing_geometries(self):
        gpd = self.geopandas
        from shapely.geometry import Point

        gdf = gpd.GeoDataFrame({"species": ["a", "b"]}, geometry=[Point(0, 0), None])
        with pytest.raises(ValueError, match="missing or empty"):
            samplesar.Sample.from_geodataframe(gdf)

    def test_missing_species_column(self):
        gpd = self.geopandas
        gdf = gpd.GeoDataFrame({"name": ["a"]}, geometry=gpd.points_from_xy([0], [0]))
        with pytest.raises(KeyError, match="species"):
            samplesar.Sample.from_geodataframe(gdf)


class TestValidation:
    def good(self, **change):
        return pd.DataFrame(
            {"species": ["a", "b", "c"], "lon": [0.0, 1.0, 1.0], "lat": [0.0, 0.0, 1.0], **change}
        )

    @pytest.mark.parametrize(
        ("change", "error", "match"),
        [
            ({"species": [None, "b", "c"]}, ValueError, "species contains missing"),
            ({"lon": ["x", 1.0, 1.0]}, ValueError, "numeric"),
            ({"lat": [np.nan, 0.0, 1.0]}, ValueError, "numeric"),
            ({"lat": [91.0, 0.0, 1.0]}, ValueError, r"lat must be within \[-90, 90\]"),
            ({"lat": [-90.5, 0.0, 1.0]}, ValueError, "lat must be within"),
            ({"lon": [180.5, 1.0, 1.0]}, ValueError, r"lon must be within \[-180, 180\]"),
        ],
    )
    def test_bad_values(self, change, error, match):
        with pytest.raises(error, match=match):
            samplesar.Sample.from_df(self.good(**change))

    def test_missing_column(self):
        with pytest.raises(KeyError, match=r"Missing column\(s\): \['lat'\]"):
            samplesar.Sample.from_df(self.good().drop(columns="lat"))

    def test_no_rows(self):
        with pytest.raises(ValueError, match="no observations"):
            samplesar.Sample.from_df(self.good().iloc[:0])

    def test_normalises(self):
        df = self.good(species=[1, 2, 3], lon=["0", "1", "1"], extra=["x", "y", "z"])
        sample = samplesar.Sample.from_df(df)
        assert list(sample.data.columns) == ["species", "lon", "lat"]
        assert sample.data["species"].tolist() == ["1", "2", "3"]
        assert sample.data["lon"].dtype == float

    def test_bounds_are_inclusive(self):
        df = pd.DataFrame({"species": ["a", "b"], "lon": [-180.0, 180.0], "lat": [-90.0, 90.0]})
        assert samplesar.Sample.from_df(df).n_observations == 2

    def test_from_csv(self, tmp_path):
        path = tmp_path / "obs.csv"
        self.good().rename(columns={"lon": "x"}).to_csv(path, index=False)
        assert samplesar.Sample.from_csv(path, lon_col="x").n_points == 3
        with pytest.raises(TypeError):
            samplesar.Sample.from_csv(path, sep=";")


class TestSubset:
    def sample(self):
        # Points 0 and 1 hold two observations each.
        return samplesar.Sample.from_df(
            pd.DataFrame(
                {
                    "species": ["a", "b", "a", "c", "d", "e"],
                    "lon": [0.0, 0.0, 1.0, 1.0, 1.0, 0.0],
                    "lat": [0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
                }
            )
        )

    def test_keeps_every_observation_at_the_points(self):
        part = self.sample().subset([0, 1])
        assert part.n_observations == 4 and part.n_points == 2
        assert sorted(part.data["species"]) == ["a", "a", "b", "c"]

    def test_points_and_counts(self):
        sample = self.sample()
        assert sample.n_observations == 6 and sample.n_points == 4 and sample.n_species == 5
        np.testing.assert_array_equal(sample.points, [[0, 0], [1, 0], [1, 1], [0, 1]])

    def test_an_empty_subset(self):
        part = self.sample().subset([])
        assert part.n_observations == 0 and part.n_points == 0 and part.area == 0.0


class TestDegenerateHulls:
    @pytest.mark.parametrize("n", [0, 1, 2])
    def test_fewer_than_three_points(self, n):
        assert samplesar.hull_area(np.zeros((n, 2)) + np.arange(n)[:, None]) == 0.0

    def test_collinear_points(self):
        assert samplesar.hull_area(np.array([[0, 0], [1, 1], [2, 2], [3, 3.0]])) == 0.0
        line = samplesar.Sample.from_df(frame([0.0, 0.1, 0.2], [0.0, 0.0, 0.0]))
        assert line.area == pytest.approx(0.0, abs=1e-9)

    def test_duplicates_dont_count(self):
        square = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [1, 1], [0, 0.0]])
        assert samplesar.hull_area(square) == pytest.approx(1.0)

    def test_interior_points_dont_change_the_area(self):
        square = np.array([[0, 0], [2, 0], [2, 2], [0, 2.0]])
        assert samplesar.hull_area(np.vstack([square, [[1, 1], [0.5, 1.5]]])) == pytest.approx(4.0)


class TestMergeWithin:
    def near(self):
        # Points 0-2 within a few metres (GPS jitter at one spot); 3 is 1 km away.
        lons = [-82.0, -82.00001, -82.00002, -81.99]
        lats = [36.5, 36.50001, 36.5, 36.5]
        return samplesar.Sample.from_df(frame(lons, lats, species=["a", "b", "c", "d"]))

    def test_merges_jitter_and_keeps_observations(self):
        sample = self.near()
        merged = sample.merge_within(10)
        assert merged.n_points == 2 and merged.n_observations == 4 and merged.n_species == 4
        # The group sits at its first-seen member's coordinates.
        np.testing.assert_array_equal(merged.points, [[-82.0, 36.5], [-81.99, 36.5]])
        assert merged.projection_centre == sample.projection_centre

    def test_a_small_distance_merges_nothing(self):
        assert self.near().merge_within(0.1).n_points == 4

    def test_links_chain(self):
        x = [500_000, 500_000.8, 500_001.6, 500_100]
        sample = samplesar.Sample.from_df(frame(x, [4_000_000] * 4), crs="EPSG:32617")
        assert sample.merge_within(1).n_points == 2

    def test_no_more_centimetre_hulls(self):
        rng = np.random.default_rng(0)
        spread = rng.uniform(-0.01, 0.01, size=(60, 2))
        jitter = rng.normal(0, 1e-7, size=(40, 2))  # 40 records at one spot
        pts = np.vstack([spread, jitter]) + [-82.2, 36.5]
        sample = samplesar.Sample.from_df(
            frame(pts[:, 0], pts[:, 1], [f"s{i % 30}" for i in range(100)])
        )
        smallest = samplesar.hull_peel_nest(sample, min_points=3).sar_data()["area"].min()
        merged = (
            samplesar.hull_peel_nest(sample.merge_within(5), min_points=3).sar_data()["area"].min()
        )
        assert smallest < 1e-6 < merged  # km²: below a square metre, then not

    def test_numpy_distances(self):
        assert self.near().merge_within(np.int64(10)).n_points == 2

    def test_a_long_chain_warns(self):
        # Records every 4 m along a 500 m line, merged within 5 m: one chain.
        x = 500_000 + np.arange(0, 500, 4.0)
        sample = samplesar.Sample.from_df(frame(x, [4_000_000] * len(x)), crs="EPSG:32617")
        with pytest.warns(UserWarning, match="chain over"):
            assert sample.merge_within(5).n_points == 1

    def test_a_distorting_crs_merges_at_true_distances(self):
        from pyproj import Transformer

        # Two points 8 m apart on the ground at 60°N, given in Web Mercator
        # (where they're about 16 m apart in its units).
        lon, lat = [10.0, 10.0 + 8 / (111_320 * np.cos(np.radians(60)))], [60.0, 60.0]
        x, y = Transformer.from_crs(4326, 3857, always_xy=True).transform(lon, lat)
        sample = samplesar.Sample.from_df(frame(x, y), crs=3857)
        assert sample.merge_within(10).n_points == 1
        assert sample.merge_within(5).n_points == 2

    @pytest.mark.parametrize("bad", [0, -1, float("nan"), "5", True])
    def test_bad_distance(self, bad):
        with pytest.raises(ValueError, match="metres must be a positive number"):
            self.near().merge_within(bad)
