"""Administrative boundaries from geoBoundaries, with the service faked."""

import json

import pytest
from shapely.geometry import box, mapping

import samplesar
from samplesar import _geoboundaries as gb


def feature(name, iso, geometry, shape_id):
    return {
        "type": "Feature",
        "properties": {"shapeName": name, "shapeISO": iso, "shapeID": shape_id},
        "geometry": mapping(geometry),
    }


FILES = {
    "XYZ_ADM0": [feature("Xland", "XYZ", box(0, 0, 10, 10), "c0")],
    "XYZ_ADM1": [
        feature("North", "XY-N", box(0, 5, 10, 10), "s1"),
        feature("South", "XY-S", box(0, 0, 10, 5), "s2"),
    ],
    "XYZ_ADM2": [
        feature("Hill", "", box(0, 6, 2, 8), "d1"),
        feature("Hill", "", box(0, 1, 2, 3), "d2"),
        feature("Vale", "", box(5, 1, 7, 3), "d3"),
    ],
}


@pytest.fixture
def fake(monkeypatch, tmp_path):
    calls = []

    def fetch(url):
        calls.append(url)
        if url.startswith("https://www.geoboundaries.org/"):
            stem = f"{url.split('/')[-3]}_{url.split('/')[-2]}"
            if stem not in FILES:
                raise OSError("404")
            return json.dumps(
                {"gjDownloadURL": f"geo://{stem}", "simplifiedGeometryGeoJSON": f"geo://{stem}_s"}
            ).encode()
        stem = url.removeprefix("geo://").removesuffix("_s")
        if stem not in FILES:
            raise OSError("404")
        return json.dumps({"type": "FeatureCollection", "features": FILES[stem]}).encode()

    monkeypatch.setattr(gb, "_fetch", fetch)
    return tmp_path, calls


def test_a_country(fake):
    cache, _ = fake
    assert samplesar.boundary("xyz", cache_dir=cache).equals(box(0, 0, 10, 10))


def test_a_unit_by_name_or_iso_code(fake):
    cache, _ = fake
    assert samplesar.boundary("XYZ", 1, "south", cache_dir=cache).equals(box(0, 0, 10, 5))
    assert samplesar.boundary("XYZ", 1, "XY-N", cache_dir=cache).equals(box(0, 5, 10, 10))


def test_a_whole_level_is_dissolved(fake):
    cache, _ = fake
    assert samplesar.boundary("XYZ", 1, cache_dir=cache).equals(box(0, 0, 10, 10))


def test_repeated_names_need_a_parent(fake):
    cache, _ = fake
    with pytest.raises(ValueError, match="2 ADM2 units of XYZ are named 'Hill'"):
        samplesar.boundary("XYZ", 2, "Hill", cache_dir=cache)
    south = samplesar.boundary("XYZ", 1, "South", cache_dir=cache)
    assert samplesar.boundary("XYZ", 2, "Hill", inside=south, cache_dir=cache).equals(
        box(0, 1, 2, 3)
    )


def test_a_misspelt_name_suggests_close_ones(fake):
    cache, _ = fake
    with pytest.raises(ValueError, match="Close: Vale"):
        samplesar.boundary("XYZ", 2, "Vael", cache_dir=cache)


def test_downloads_are_cached(fake):
    cache, calls = fake
    samplesar.boundary("XYZ", 1, "North", cache_dir=cache)
    n = len(calls)
    samplesar.boundary("XYZ", 1, "South", cache_dir=cache)
    assert len(calls) == n
    samplesar.boundary("XYZ", 1, "South", simplified=True, cache_dir=cache)
    assert len(calls) == n + 1  # the simplified file is its own download


@pytest.mark.parametrize(
    ("args", "match"),
    [
        (("XY",), "alpha-3"),
        (("XYZ", 6), "level must be 0 to 5"),
        (("ABC", 1), "no ADM1 boundaries for 'ABC'"),
    ],
)
def test_bad_arguments(fake, args, match):
    cache, _ = fake
    with pytest.raises(ValueError, match=match):
        samplesar.boundary(*args, cache_dir=cache)


def test_works_with_within(fake):
    import pandas as pd

    cache, _ = fake
    sample = samplesar.Sample.from_df(
        pd.DataFrame({"species": list("abc"), "lon": [1.0, 1.0, 6.0], "lat": [2.0, 7.0, 2.0]})
    )
    south = samplesar.boundary("XYZ", 1, "South", cache_dir=cache)
    assert sorted(sample.within(south).data["species"]) == ["a", "c"]


def test_nothing_inside_the_parent(fake):
    cache, _ = fake
    with pytest.raises(ValueError, match="No ADM2 unit of XYZ lies inside"):
        samplesar.boundary("XYZ", 2, inside=box(50, 50, 51, 51), cache_dir=cache)


def test_a_broken_cached_file_is_downloaded_again(fake):
    cache, calls = fake
    samplesar.boundary("XYZ", 1, "North", cache_dir=cache)
    (cache / "XYZ_ADM1.geojson").write_text('{"type": "FeatureCo')  # cut off mid-download
    n = len(calls)
    assert samplesar.boundary("XYZ", 1, "North", cache_dir=cache).equals(box(0, 5, 10, 10))
    assert len(calls) == n + 1


def test_a_failed_download_is_a_value_error(fake, monkeypatch):
    cache, _ = fake
    samplesar.boundary("XYZ", 0, cache_dir=cache)  # metadata cached
    (cache / "XYZ_ADM0.geojson").unlink()

    def offline(url):
        raise OSError("unreachable")

    monkeypatch.setattr(gb, "_fetch", offline)
    with pytest.raises(ValueError, match="Couldn't download"):
        samplesar.boundary("XYZ", 0, cache_dir=cache)
    assert not list(cache.glob("*.partial"))


def test_invalid_rings_are_repaired(fake, monkeypatch):
    from shapely.geometry import Polygon

    cache, _ = fake
    bowtie = Polygon([(0, 0), (2, 2), (2, 0), (0, 2)])  # self-intersecting
    monkeypatch.setitem(FILES, "XYZ_ADM0", [feature("Xland", "XYZ", bowtie, "c0")])
    result = samplesar.boundary("XYZ", cache_dir=cache)
    assert result.is_valid and result.area == pytest.approx(2.0)
