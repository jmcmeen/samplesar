"""A local copy of iNaturalist's open data: download, build, and query."""

import gzip
import http.server
import json
import threading

import pytest

pytest.importorskip("duckdb")

import samplesar  # noqa: E402
from samplesar import _opendata  # noqa: E402

OBSERVATIONS = [
    # uuid, observer, lat, lon, accuracy, taxon, grade, date, anomaly
    ("u1", 1, 36.50, -82.20, 10, 101, "research", "2024-05-01", 1.0),
    ("u2", 1, 36.51, -82.21, 10, 102, "research", "2024-05-02", ""),
    ("u3", 2, 36.52, -82.22, 500, 101, "research", "2023-01-01", ""),
    ("u4", 2, 36.53, -82.23, "", 103, "needs_id", "2024-06-01", ""),
    ("u5", 3, 36.54, -82.24, 5, 104, "research", "2024-07-01", ""),
    ("u6", 3, 36.55, -82.25, 5, 105, "research", "2024-07-02", ""),
    ("u7", 4, 10.00, 179.50, 5, 101, "research", "2024-07-03", ""),
    ("u8", 4, 10.10, -179.50, 5, 102, "research", "2024-07-04", ""),
    ("u9", 5, "", "", 5, 101, "research", "2024-07-05", ""),
]
TAXA = [
    # taxon, ancestry, rank level, rank, name, active
    (20979, "48460/1/2/355675/20978", 40, "order", "Anura", "true"),
    (101, "48460/1/2/355675/20978/20979", 10, "species", "Lithobates clamitans", "true"),
    (102, "48460/1/2/355675/20978/20979", 10, "species", "Anaxyrus americanus", "true"),
    (103, "48460/1/2/355675/20978/20979", 10, "species", "Hyla chrysoscelis", "true"),
    (
        104,
        "48460/1/2/355675/20978/20979/101",
        5,
        "subspecies",
        "Lithobates clamitans melanota",
        "true",
    ),
    (105, "48460/1/2/355675/26036", 10, "species", "Plestiodon fasciatus", "true"),
]


def write_tsv(path, header, rows):
    lines = ["\t".join(header)] + ["\t".join(str(v) for v in row) for row in rows]
    path.write_bytes(gzip.compress(("\n".join(lines) + "\n").encode()))


@pytest.fixture
def export(tmp_path):
    """A tiny export in the open data's layout."""
    source = tmp_path / "bucket"
    source.mkdir()
    write_tsv(
        source / "observations.csv.gz",
        [
            "observation_uuid",
            "observer_id",
            "latitude",
            "longitude",
            "positional_accuracy",
            "taxon_id",
            "quality_grade",
            "observed_on",
            "anomaly_score",
        ],
        OBSERVATIONS,
    )
    write_tsv(
        source / "taxa.csv.gz",
        ["taxon_id", "ancestry", "rank_level", "rank", "name", "active"],
        TAXA,
    )
    return source


@pytest.fixture
def server(export):
    """Serves the export over HTTP, with Range requests, and counts requests.
    Set ``drop[name] = n`` to cut the next GET of ``name`` off after n bytes."""
    seen = []
    drop = {}

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(export), **kwargs)

        def log_message(self, *args):
            pass

        def do_GET(self):
            seen.append((self.path, self.headers.get("Range")))
            data = (export / self.path.lstrip("/")).read_bytes()
            start = 0
            if self.headers.get("Range"):
                start = int(self.headers["Range"].split("=")[1].rstrip("-"))
                self.send_response(206)
            else:
                self.send_response(200)
            self.send_header("Content-Length", str(len(data) - start))
            self.end_headers()
            cut = drop.pop(self.path.lstrip("/"), None)
            self.wfile.write(data[start:] if cut is None else data[start : start + cut])

    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", seen, drop
    httpd.shutdown()


@pytest.fixture
def parquet(server, tmp_path):
    url, *_ = server
    return samplesar.sources.download_open_data(tmp_path / "local", base_url=url)


def test_download_and_build(server, tmp_path, export):
    url, seen, _ = server
    path = samplesar.sources.download_open_data(tmp_path / "local", base_url=url)
    assert path.name == "observations.parquet" and path.exists()
    for name in ("observations.csv.gz", "taxa.csv.gz"):
        assert (tmp_path / "local" / name).read_bytes() == (export / name).read_bytes()
    assert not (tmp_path / "local" / "duckdb-tmp").exists()
    # A second run downloads nothing: the files are already complete.
    seen.clear()
    samplesar.sources.download_open_data(tmp_path / "local", base_url=url)
    assert seen == []


def test_a_partial_download_resumes(server, tmp_path, export):
    url, seen, _ = server
    local = tmp_path / "local"
    local.mkdir()
    whole = (export / "observations.csv.gz").read_bytes()
    # An earlier run stopped after 50 bytes, having recorded the version.
    (local / "observations.csv.gz").write_bytes(whole[:50])
    size, version = _opendata._remote(f"{url}/observations.csv.gz")
    (local / "observations.csv.gz.source").write_text(
        json.dumps({"size": size, "version": version})
    )
    samplesar.sources.download_open_data(local, base_url=url, build=False)
    assert (local / "observations.csv.gz").read_bytes() == whole
    assert ("/observations.csv.gz", "bytes=50-") in seen


def test_defaults_research_species_and_below(parquet):
    sample = samplesar.sources.open_data(parquet)
    species = sorted(sample.data["species"])
    # u4 is needs_id, u9 has no coordinates; u5's subspecies counts as its species.
    assert species == sorted(
        [
            "Lithobates clamitans",
            "Anaxyrus americanus",
            "Lithobates clamitans",
            "Lithobates clamitans",
            "Plestiodon fasciatus",
            "Lithobates clamitans",
            "Anaxyrus americanus",
        ]
    )


def test_taxon_includes_everything_under_it(parquet):
    frogs = samplesar.sources.open_data(parquet, taxon_id=20979)
    assert "Plestiodon fasciatus" not in set(frogs.data["species"])
    assert frogs.n_observations == 6
    one = samplesar.sources.open_data(parquet, taxon_id=101)
    assert set(one.data["species"]) == {"Lithobates clamitans"} and one.n_observations == 4


def test_bbox_dates_accuracy_and_grades(parquet):
    box = samplesar.sources.open_data(parquet, bbox=(-82.23, 36.5, -82.2, 36.53))
    assert box.n_observations == 3
    across = samplesar.sources.open_data(parquet, bbox=(179.0, 9.0, -179.0, 11.0))
    assert across.n_observations == 2
    recent = samplesar.sources.open_data(parquet, since="2024-06-01", until="2024-07-02")
    assert recent.n_observations == 2
    precise = samplesar.sources.open_data(parquet, max_accuracy=10)
    coords = set(zip(precise.data["lon"], precise.data["lat"], strict=True))
    assert (-82.22, 36.52) not in coords and precise.n_observations == 6  # u3, 500 m
    every = samplesar.sources.open_data(parquet, quality_grade=None, ranks=None)
    assert every.n_observations == 8


def test_nothing_matches(parquet):
    with pytest.raises(ValueError, match="No observations"):
        samplesar.sources.open_data(parquet, taxon_id=999)


def test_bad_bbox(parquet):
    with pytest.raises(ValueError, match="west, south, east, north"):
        samplesar.sources.open_data(parquet, bbox=(-82, 37, -81, 36))


def test_duckdb_is_needed(monkeypatch, tmp_path):
    import sys

    monkeypatch.setitem(sys.modules, "duckdb", None)
    with pytest.raises(ImportError, match=r"samplesar\[opendata\]"):
        samplesar.sources.download_open_data(tmp_path)
    with pytest.raises(ImportError, match=r"samplesar\[opendata\]"):
        _opendata.open_data_frame(tmp_path / "x.parquet")


def test_a_newer_export_replaces_the_old_file(server, tmp_path, export):
    import os

    url, *_ = server
    local = tmp_path / "local"
    samplesar.sources.download_open_data(local, base_url=url, build=False)
    # A month later the bucket holds a bigger, different export.
    newer = [*OBSERVATIONS, ("u10", 6, 36.6, -82.3, 5, 101, "research", "2026-10-01", "")]
    write_tsv(
        export / "observations.csv.gz",
        [
            "observation_uuid",
            "observer_id",
            "latitude",
            "longitude",
            "positional_accuracy",
            "taxon_id",
            "quality_grade",
            "observed_on",
            "anomaly_score",
        ],
        newer,
    )
    later = (export / "observations.csv.gz").stat().st_mtime + 86400
    os.utime(export / "observations.csv.gz", (later, later))
    samplesar.sources.download_open_data(local, base_url=url, build=False)
    assert (local / "observations.csv.gz").read_bytes() == (
        export / "observations.csv.gz"
    ).read_bytes()


def test_a_dropped_connection_resumes(server, tmp_path, export, monkeypatch):
    url, seen, drop = server
    monkeypatch.setattr(_opendata.time, "sleep", lambda s: None)
    drop["observations.csv.gz"] = 40
    samplesar.sources.download_open_data(tmp_path / "local", base_url=url, build=False)
    whole = (export / "observations.csv.gz").read_bytes()
    assert (tmp_path / "local" / "observations.csv.gz").read_bytes() == whole
    assert ("/observations.csv.gz", "bytes=40-") in seen


def test_an_unchanged_copy_isnt_rebuilt(server, tmp_path, monkeypatch):
    url, *_ = server
    local = tmp_path / "local"
    samplesar.sources.download_open_data(local, base_url=url)
    builds = []
    original = _opendata.build_parquet
    monkeypatch.setattr(
        _opendata, "build_parquet", lambda *a, **k: builds.append(1) or original(*a, **k)
    )
    samplesar.sources.download_open_data(local, base_url=url)
    assert builds == []
    samplesar.sources.download_open_data(local, base_url=url, rebuild=True)
    assert builds == [1]


def test_a_folder_with_a_quote_in_its_name(server, tmp_path):
    url, *_ = server
    path = samplesar.sources.download_open_data(tmp_path / "O'Brien data", base_url=url)
    assert samplesar.sources.open_data(path).n_observations == 7


def test_a_bad_memory_limit(server, tmp_path):
    url, *_ = server
    with pytest.raises(ValueError, match="memory_limit"):
        samplesar.sources.download_open_data(
            tmp_path / "local", base_url=url, memory_limit="8GB'; DROP"
        )


@pytest.mark.parametrize("option", [{"ranks": []}, {"quality_grade": []}])
def test_empty_filters(parquet, option):
    with pytest.raises(ValueError, match="is empty"):
        samplesar.sources.open_data(parquet, **option)


def test_a_partial_file_of_unknown_version_starts_again(server, tmp_path, export):
    url, seen, _ = server
    local = tmp_path / "local"
    local.mkdir()
    (local / "observations.csv.gz").write_bytes(b"left over from something else")
    samplesar.sources.download_open_data(local, base_url=url, build=False)
    assert (local / "observations.csv.gz").read_bytes() == (
        export / "observations.csv.gz"
    ).read_bytes()
    assert ("/observations.csv.gz", None) in seen


def test_a_boundary(parquet):
    from shapely.geometry import box

    # Around u1 and u2 only: the bbox would also hold u3, but it's outside.
    region = box(-82.215, 36.495, -82.195, 36.515)
    sample = samplesar.sources.open_data(parquet, boundary=region)
    assert sample.n_observations == 2
    narrower = samplesar.sources.open_data(
        parquet, boundary=region, bbox=(-82.205, 36.5, -82.2, 36.52)
    )
    assert narrower.n_observations == 1


@pytest.mark.parametrize(
    ("bbox", "match"),
    [
        ((-100.0, 30.0, -95.0, 40.0), "don't overlap"),
        ((-82.3, 50.0, -82.1, 51.0), "don't overlap"),
        ((1.0, 2.0, 3.0), "west, south, east, north"),
        ((-82.3, 37.0, -82.1, 36.0), "west, south, east, north"),
    ],
)
def test_a_bbox_that_doesnt_fit_the_boundary(parquet, bbox, match):
    from shapely.geometry import box

    with pytest.raises(ValueError, match=match):
        samplesar.sources.open_data(parquet, boundary=box(-82.3, 36.4, -82.1, 36.6), bbox=bbox)
