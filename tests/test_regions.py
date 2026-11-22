"""Regions from boundary files and known ecoregion datasets."""

import pytest
from shapely.geometry import box

import samplesar
from samplesar import _regions

gpd = pytest.importorskip("geopandas")


@pytest.fixture
def provinces(tmp_path):
    """A small GeoPackage of three regions, in a projected CRS."""
    frame = gpd.GeoDataFrame(
        {"CODE": ["66", "67", "45"], "NAME": ["Blue Ridge", "Ridge and Valley", "Piedmont"]},
        geometry=[box(0, 0, 1, 2), box(1, 0, 2, 2), box(2, 0, 3, 2)],
        crs=4326,
    ).to_crs(3857)
    path = tmp_path / "provinces.gpkg"
    frame.to_file(path)
    return path


def test_rows_by_column_dissolved_in_wgs84(provinces):
    region = samplesar.read_boundary(provinces, "CODE", [66, "67"])
    assert (
        region.equals_exact(box(0, 0, 2, 2), 1e-6)
        or region.symmetric_difference(box(0, 0, 2, 2)).area < 1e-9
    )


def test_values_ignore_case(provinces):
    region = samplesar.read_boundary(provinces, "NAME", ["piedmont"])
    assert region.symmetric_difference(box(2, 0, 3, 2)).area < 1e-9


def test_every_row_and_a_clip(provinces):
    assert samplesar.read_boundary(provinces).symmetric_difference(box(0, 0, 3, 2)).area < 1e-9
    clipped = samplesar.read_boundary(provinces, clip=(0, 0, 3, 1))
    assert clipped.symmetric_difference(box(0, 0, 3, 1)).area < 1e-9
    by_polygon = samplesar.read_boundary(provinces, clip=box(0.5, 0, 1.5, 2))
    assert by_polygon.symmetric_difference(box(0.5, 0, 1.5, 2)).area < 1e-9


@pytest.mark.parametrize(
    ("args", "error", "match"),
    [
        (("NOPE", ["1"]), KeyError, "No column 'NOPE'"),
        (("CODE", ["99"]), ValueError, "No row has CODE = 99. Values: 45, 66, 67"),
        (("CODE", None), ValueError, "together"),
    ],
)
def test_bad_selections(provinces, args, error, match):
    with pytest.raises(error, match=match):
        samplesar.read_boundary(provinces, *args)


def test_a_clip_that_leaves_nothing(provinces):
    with pytest.raises(ValueError, match="Nothing of the region"):
        samplesar.read_boundary(provinces, clip=(10, 10, 11, 11))


def test_a_missing_file(tmp_path):
    with pytest.raises(ValueError, match="No boundary file"):
        samplesar.read_boundary(tmp_path / "nothing.gpkg")


def test_a_url_is_downloaded_once(provinces, tmp_path, monkeypatch):
    calls = []

    def fetch(url, path):
        calls.append(url)
        path.write_bytes(provinces.read_bytes())

    monkeypatch.setattr(_regions, "_fetch_to", fetch)
    url = "https://example.org/data/provinces.gpkg"
    for _ in range(2):
        region = samplesar.read_boundary(url, "CODE", ["45"], cache_dir=tmp_path / "cache")
    assert calls == [url]
    assert region.symmetric_difference(box(2, 0, 3, 2)).area < 1e-9


def test_ecoregion_uses_the_dataset_columns(provinces, monkeypatch):
    seen = {}

    def read(source, column, values, *, clip, cache_dir):
        seen.update(source=source, column=column, values=list(values), clip=clip)
        return box(0, 0, 1, 1)

    monkeypatch.setattr(_regions, "read_boundary", read)
    samplesar.ecoregion("epa_l3", codes=["66"], clip=(0, 0, 1, 1))
    assert seen == {
        "source": _regions.ECOREGIONS["epa_l3"]["url"],
        "column": "US_L3CODE",
        "values": ["66"],
        "clip": (0, 0, 1, 1),
    }
    samplesar.ecoregion("resolve_2017", names=["Talamancan montane forests"])
    assert seen["column"] == "ECO_NAME"


@pytest.mark.parametrize(
    ("kwargs", "match"), [({"source": "wwf"}, "Unknown ecoregion source"), ({}, "codes or names")]
)
def test_ecoregion_bad_arguments(kwargs, match):
    with pytest.raises(ValueError, match=match):
        samplesar.ecoregion(**kwargs)


def test_a_single_value_and_numbers(provinces):
    assert (
        samplesar.read_boundary(provinces, "CODE", "66").symmetric_difference(box(0, 0, 1, 2)).area
        < 1e-9
    )
    assert (
        samplesar.read_boundary(provinces, "CODE", 67).symmetric_difference(box(1, 0, 2, 2)).area
        < 1e-9
    )
    assert (
        samplesar.read_boundary(provinces, "CODE", [66.0])
        .symmetric_difference(box(0, 0, 1, 2))
        .area
        < 1e-9
    )


def test_a_float_column_matches_integer_codes(tmp_path):
    path = tmp_path / "ids.gpkg"
    gpd.GeoDataFrame(
        {"ECO_ID": [66.0, 67.0]}, geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)], crs=4326
    ).to_file(path)
    assert (
        samplesar.read_boundary(path, "ECO_ID", ["66"]).symmetric_difference(box(0, 0, 1, 1)).area
        < 1e-9
    )


def test_no_values(provinces):
    with pytest.raises(ValueError, match="No values given"):
        samplesar.read_boundary(provinces, "CODE", [])


def test_a_clip_across_the_antimeridian(tmp_path):
    path = tmp_path / "pacific.gpkg"
    gpd.GeoDataFrame(
        {"N": [1, 2]}, geometry=[box(170, 0, 180, 1), box(-180, 0, -170, 1)], crs=4326
    ).to_file(path)
    clipped = samplesar.read_boundary(path, clip=(175, 0, -175, 1))
    expected = box(175, 0, 180, 1).union(box(-180, 0, -175, 1))
    assert clipped.symmetric_difference(expected).area < 1e-9


def test_a_zip_at_a_url_with_a_query(tmp_path, monkeypatch):
    import zipfile

    shp = tmp_path / "shp"
    shp.mkdir()
    gpd.GeoDataFrame({"CODE": ["1"]}, geometry=[box(0, 0, 1, 1)], crs=4326).to_file(shp / "r.shp")
    archive = tmp_path / "r.zip"
    with zipfile.ZipFile(archive, "w") as z:
        for f in shp.iterdir():
            z.write(f, f.name)

    monkeypatch.setattr(
        _regions, "_fetch_to", lambda url, path: path.write_bytes(archive.read_bytes())
    )
    cache = tmp_path / "cache"
    region = samplesar.read_boundary(
        "https://example.org/download?id=3&format=shp", cache_dir=cache
    )
    assert region.symmetric_difference(box(0, 0, 1, 1)).area < 1e-9
    assert [f.name.split("_", 1)[1] for f in cache.iterdir()] == ["download.zip"]
    samplesar.read_boundary("https://example.org/download?id=3&format=shp", cache_dir=cache)
