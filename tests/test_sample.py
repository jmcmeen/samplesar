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
