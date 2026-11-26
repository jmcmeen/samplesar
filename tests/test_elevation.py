"""sources.dem_elevations: elevations read from a raster, or from Copernicus tiles."""

import numpy as np
import pandas as pd
import pytest

import samplesar
from samplesar import _elevation
from samplesar._elevation import glo30_tile


def write_raster(
    path,
    west,
    north,
    cell,
    values,
    crs="EPSG:4326",
    nodata=-9999.0,
    dtype="float32",
    south_up=False,
):
    """A raster whose top-left corner is (west, north); south-up, its first
    row is the southernmost."""
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import Affine, from_origin

    values = np.asarray(values, dtype=dtype)
    transform = from_origin(west, north, cell, cell)
    if south_up:
        transform = Affine(cell, 0, west, 0, cell, north - cell * values.shape[0])
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=values.shape[0],
        width=values.shape[1],
        count=1,
        dtype=dtype,
        crs=crs,
        transform=transform,
        nodata=nodata,
    ) as out:
        out.write(values, 1)
    return path


def sample_at(points):
    frame = pd.DataFrame(points, columns=["lon", "lat"]).assign(species="a")
    return samplesar.sources.dataframe(frame)


def test_tile_names_by_south_west_corner():
    assert glo30_tile(-83.5, 35.6) == "Copernicus_DSM_COG_10_N35_00_W084_00_DEM"
    assert glo30_tile(10.2, -0.5) == "Copernicus_DSM_COG_10_S01_00_E010_00_DEM"
    assert glo30_tile(-0.1, 0.0) == "Copernicus_DSM_COG_10_N00_00_W001_00_DEM"
    assert glo30_tile(179.9, 59.99) == "Copernicus_DSM_COG_10_N59_00_E179_00_DEM"
    assert glo30_tile(180.0, 10.5) == "Copernicus_DSM_COG_10_N10_00_E179_00_DEM"
    assert glo30_tile(-180.0, -10.5) == "Copernicus_DSM_COG_10_S11_00_W180_00_DEM"


def test_elevations_from_a_raster_in_degrees(tmp_path):
    # A 4 by 4 raster of 0.25° cells over (10..11, 50..51): the value is 100 * row + column.
    grid = 100 * np.arange(4)[:, None] + np.arange(4)[None, :]
    path = write_raster(tmp_path / "dem.tif", 10.0, 51.0, 0.25, grid)
    sample = sample_at([(10.1, 50.9), (10.9, 50.9), (10.1, 50.1), (10.6, 50.4)])
    heights = samplesar.sources.dem_elevations(sample, path)
    assert heights.tolist() == [0.0, 3.0, 300.0, 202.0]
    nested = samplesar.strategies.elevation(sample.with_elevations(heights), min_points=1)
    assert [
        int(np.setdiff1d(a, b)[0]) for a, b in zip(nested.steps, nested.steps[1:], strict=False)
    ] == [2, 3, 1]


def test_nodata_and_points_outside_are_missing(tmp_path):
    grid = np.array([[5.0, -9999.0], [7.0, 8.0]])
    path = write_raster(tmp_path / "dem.tif", 0.0, 2.0, 1.0, grid)
    sample = sample_at([(0.5, 1.5), (1.5, 1.5), (0.5, 0.5), (5.0, 5.0), (-0.5, 1.0)])
    heights = samplesar.sources.dem_elevations(sample, path)
    assert heights[[0, 2]].tolist() == [5.0, 7.0]
    assert np.isnan(heights[[1, 3, 4]]).all()


def test_a_raster_in_another_coordinate_system(tmp_path):
    # UTM zone 32N, 1 km cells west to east: the value is the column.
    grid = np.tile(np.arange(10.0), (10, 1))
    path = write_raster(
        tmp_path / "utm.tif", 500_000.0, 5_600_000.0, 1000.0, grid, crs="EPSG:32632"
    )
    frame = pd.DataFrame(
        {"species": "a", "lon": [500_500.0, 504_500.0, 509_500.0], "lat": [5_595_500.0] * 3}
    )
    sample = samplesar.sources.dataframe(frame, crs="EPSG:32632")
    assert samplesar.sources.dem_elevations(sample, path).tolist() == [0.0, 4.0, 9.0]


def not_found(url):
    """The error a missing tile gives: _fetch_to's, caused by an HTTP 404."""
    import urllib.error

    err = ValueError(f"Couldn't download {url}: HTTP Error 404: Not Found")
    err.__cause__ = urllib.error.HTTPError(url, 404, "Not Found", None, None)
    return err


def test_the_default_model_downloads_each_tile_once(tmp_path, monkeypatch):
    fetched = []

    def fetch(url, path):
        fetched.append(url)
        name = url.rsplit("/", 1)[-1]
        if "N35" not in name:  # only one tile has land
            raise not_found(url)
        write_raster(path, -84.0, 36.0, 0.5, [[1.0, 2.0], [3.0, 4.0]])

    monkeypatch.setattr(_elevation, "_fetch_to", fetch)
    sample = sample_at([(-83.9, 35.9), (-83.1, 35.1), (-83.5, 34.5), (-83.2, 35.8)])
    heights = samplesar.sources.dem_elevations(sample, cache_dir=tmp_path)
    assert heights[[0, 1, 3]].tolist() == [1.0, 4.0, 2.0] and np.isnan(heights[2])
    assert sorted(u.rsplit("/", 1)[-1] for u in fetched) == [
        "Copernicus_DSM_COG_10_N34_00_W084_00_DEM.tif",
        "Copernicus_DSM_COG_10_N35_00_W084_00_DEM.tif",
    ]
    assert fetched[0].startswith(
        "https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_"
    )
    # Again: the tile is read from the cache, and the sea isn't asked for.
    again = samplesar.sources.dem_elevations(sample, cache_dir=tmp_path)
    assert len(fetched) == 2 and np.array_equal(again, heights, equal_nan=True)


def test_a_failed_download_is_an_error_not_sea(tmp_path, monkeypatch):
    def fetch(url, path):
        raise ValueError(f"Couldn't download {url}: timed out")

    monkeypatch.setattr(_elevation, "_fetch_to", fetch)
    with pytest.raises(ValueError, match="timed out"):
        samplesar.sources.dem_elevations(sample_at([(1.5, 1.5)]), cache_dir=tmp_path)
    assert not list(tmp_path.glob("*.none"))


def test_bad_rasters(tmp_path):
    sample = sample_at([(1.5, 1.5)])
    with pytest.raises(ValueError, match="No elevation raster"):
        samplesar.sources.dem_elevations(sample, tmp_path / "nothing.tif")
    broken = tmp_path / "broken.tif"
    broken.write_text("not a raster")
    with pytest.raises(ValueError, match="Couldn't read elevations"):
        samplesar.sources.dem_elevations(sample, broken)


def test_without_rasterio(monkeypatch):
    import builtins

    real = builtins.__import__

    def no_rasterio(name, *args, **kwargs):
        if name == "rasterio":
            raise ImportError("No module named 'rasterio'")
        return real(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_rasterio)
    with pytest.raises(ImportError, match=r"samplesar\[elevation\]"):
        samplesar.sources.dem_elevations(sample_at([(1.5, 1.5)]))


def test_a_server_error_is_not_remembered_as_sea(tmp_path, monkeypatch):
    import urllib.error

    def fetch(url, path):
        err = ValueError(f"Couldn't download {url}: HTTP Error 503 (ticket 404)")
        err.__cause__ = urllib.error.HTTPError(url, 503, "Unavailable", None, None)
        raise err

    monkeypatch.setattr(_elevation, "_fetch_to", fetch)
    with pytest.raises(ValueError, match="503"):
        samplesar.sources.dem_elevations(sample_at([(1.5, 1.5)]), cache_dir=tmp_path)
    assert not list(tmp_path.glob("*.none"))


def glo30_like(path, south, west, value):
    """A tile laid out as GLO-30's are: half a cell north-west of its whole degrees."""
    cell = 1.0 / 20.0
    grid = np.full((20, 20), float(value))
    write_raster(path, west - cell / 2, south + 1 + cell / 2, cell, grid, nodata=None)


def test_points_within_half_a_cell_of_a_tile_edge_read_the_next_tile(tmp_path, monkeypatch):
    def fetch(url, path):
        name = url.rsplit("/", 1)[-1]
        south = int(name.split("_N")[1][:2])
        west = -int(name.split("_W")[1][:3])
        if (south, west) not in {(35, -84), (34, -84), (35, -83), (34, -83)}:
            raise not_found(url)
        glo30_like(path, south, west, 1000 * south + abs(west))

    monkeypatch.setattr(_elevation, "_fetch_to", fetch)
    points = [
        (-83.5, 35.5),  # well inside N35 W084
        (-83.5, 35.0001),  # by whole degrees N35, but below that tile's south edge
        (-83.0001, 35.5),  # by whole degrees W084, but east of that tile's east edge
        (-83.0001, 35.0001),  # both
    ]
    heights = samplesar.sources.dem_elevations(sample_at(points), cache_dir=tmp_path)
    assert heights.tolist() == [35084.0, 34084.0, 35083.0, 34083.0]


def test_a_south_up_raster_and_whole_numbers(tmp_path):
    grid = np.array([[1, 2], [3, 4]])  # first row is the southern one
    path = write_raster(
        tmp_path / "south_up.tif", 10.0, 52.0, 1.0, grid, nodata=-1, dtype="int16", south_up=True
    )
    sample = sample_at([(10.5, 50.5), (11.5, 50.5), (10.5, 51.5), (11.5, 51.5)])
    assert samplesar.sources.dem_elevations(sample, path).tolist() == [1.0, 2.0, 3.0, 4.0]


def test_a_raster_without_nodata_or_a_crs(tmp_path):
    path = write_raster(tmp_path / "plain.tif", 0.0, 1.0, 1.0, [[-9999.0]], nodata=None)
    assert samplesar.sources.dem_elevations(sample_at([(0.5, 0.5)]), path).tolist() == [-9999.0]
    bare = write_raster(tmp_path / "bare.tif", 0.0, 1.0, 1.0, [[5.0]], crs=None)
    with pytest.raises(ValueError, match="no coordinate reference system"):
        samplesar.sources.dem_elevations(sample_at([(0.5, 0.5)]), bare)


def test_a_large_raster_is_read_point_by_point(tmp_path, monkeypatch):
    grid = 100 * np.arange(4)[:, None] + np.arange(4)[None, :]
    path = write_raster(tmp_path / "dem.tif", 10.0, 51.0, 0.25, grid)
    sample = sample_at([(10.1, 50.9), (10.9, 50.1), (20.0, 20.0)])
    whole = samplesar.sources.dem_elevations(sample, path)
    monkeypatch.setattr(_elevation, "WHOLE", 4)
    by_point = samplesar.sources.dem_elevations(sample, path)
    assert np.array_equal(whole, by_point, equal_nan=True)
    assert whole[:2].tolist() == [0.0, 303.0] and np.isnan(whole[2])


def test_a_damaged_cached_tile_is_removed(tmp_path, monkeypatch):
    def fetch(url, path):
        path.write_text("half a download")

    monkeypatch.setattr(_elevation, "_fetch_to", fetch)
    with pytest.raises(ValueError, match="removed from the cache"):
        samplesar.sources.dem_elevations(sample_at([(1.5, 1.5)]), cache_dir=tmp_path)
    assert not list(tmp_path.glob("*.tif"))
