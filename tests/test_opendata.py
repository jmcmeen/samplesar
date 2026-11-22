"""A local copy of iNaturalist's open data: download, build, and query."""

import gzip
import http.server
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
    """Serves the export over HTTP, with Range requests, and counts requests."""
    seen = []

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
            self.wfile.write(data[start:])

    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", seen
    httpd.shutdown()


@pytest.fixture
def parquet(server, tmp_path):
    url, _ = server
    return samplesar.download_open_data(tmp_path / "local", base_url=url)


def test_download_and_build(server, tmp_path, export):
    url, seen = server
    path = samplesar.download_open_data(tmp_path / "local", base_url=url)
    assert path.name == "observations.parquet" and path.exists()
    for name in ("observations.csv.gz", "taxa.csv.gz"):
        assert (tmp_path / "local" / name).read_bytes() == (export / name).read_bytes()
    assert not (tmp_path / "local" / "duckdb-tmp").exists()
    # A second run downloads nothing: the files are already complete.
    seen.clear()
    samplesar.download_open_data(tmp_path / "local", base_url=url)
    assert seen == []


def test_a_partial_download_resumes(server, tmp_path, export):
    url, seen = server
    local = tmp_path / "local"
    local.mkdir()
    whole = (export / "observations.csv.gz").read_bytes()
    (local / "observations.csv.gz").write_bytes(whole[:50])
    samplesar.download_open_data(local, base_url=url, build=False)
    assert (local / "observations.csv.gz").read_bytes() == whole
    assert ("/observations.csv.gz", "bytes=50-") in seen


def test_defaults_research_species_and_below(parquet):
    sample = samplesar.Sample.from_open_data(parquet)
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
    frogs = samplesar.Sample.from_open_data(parquet, taxon_id=20979)
    assert "Plestiodon fasciatus" not in set(frogs.data["species"])
    assert frogs.n_observations == 6
    one = samplesar.Sample.from_open_data(parquet, taxon_id=101)
    assert set(one.data["species"]) == {"Lithobates clamitans"} and one.n_observations == 4


def test_bbox_dates_accuracy_and_grades(parquet):
    box = samplesar.Sample.from_open_data(parquet, bbox=(-82.23, 36.5, -82.2, 36.53))
    assert box.n_observations == 3
    across = samplesar.Sample.from_open_data(parquet, bbox=(179.0, 9.0, -179.0, 11.0))
    assert across.n_observations == 2
    recent = samplesar.Sample.from_open_data(parquet, since="2024-06-01", until="2024-07-02")
    assert recent.n_observations == 2
    precise = samplesar.Sample.from_open_data(parquet, max_accuracy=10)
    assert "u3" not in precise.data.to_string() and precise.n_observations == 6
    every = samplesar.Sample.from_open_data(parquet, quality_grade=None, ranks=None)
    assert every.n_observations == 8


def test_nothing_matches(parquet):
    with pytest.raises(ValueError, match="No observations"):
        samplesar.Sample.from_open_data(parquet, taxon_id=999)


def test_bad_bbox(parquet):
    with pytest.raises(ValueError, match="west, south, east, north"):
        samplesar.Sample.from_open_data(parquet, bbox=(-82, 37, -81, 36))


def test_duckdb_is_needed(monkeypatch, tmp_path):
    import sys

    monkeypatch.setitem(sys.modules, "duckdb", None)
    with pytest.raises(ImportError, match=r"samplesar\[opendata\]"):
        samplesar.download_open_data(tmp_path)
    with pytest.raises(ImportError, match=r"samplesar\[opendata\]"):
        _opendata.open_data_frame(tmp_path / "x.parquet")
