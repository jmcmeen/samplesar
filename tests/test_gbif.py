"""Samples from GBIF records, with pygbif faked out."""

import sys
import types

import pytest
from shapely.geometry import box

import samplesar


def record(i, species="Agalychnis callidryas", rank="SPECIES", lon=-84.0, lat=10.0, **extra):
    return {
        "key": i,
        "species": species,
        "taxonRank": rank,
        "decimalLongitude": lon + i * 1e-3,
        "decimalLatitude": lat + (i % 7) * 1e-3,
        **extra,
    }


RECORDS = [
    record(1),
    record(2, species="Dendrobates auratus"),
    record(3, species="Oophaga pumilio", rank="SUBSPECIES"),
    record(4, species=None, rank="GENUS", scientificName="Pristimantis"),
    {"key": 5, "species": "Hyla", "taxonRank": "SPECIES"},  # no coordinates
    record(6, coordinateUncertaintyInMeters=5000),
    record(7, coordinateUncertaintyInMeters=10),
]


def test_records_become_a_sample():
    sample = samplesar.sources.gbif(RECORDS)
    assert sample.n_observations == 5  # not the genus or the record without coordinates
    assert "Oophaga pumilio" in set(sample.data["species"])


def test_every_rank_and_uncertainty():
    assert samplesar.sources.gbif(RECORDS, ranks=None).n_observations == 6
    assert samplesar.sources.gbif(RECORDS, max_uncertainty=100).n_observations == 1


def test_a_boundary():
    region = box(-84.0, 9.99, -83.9985, 10.1)  # records 1 only
    assert samplesar.sources.gbif(RECORDS, boundary=region).n_observations == 1


def test_records_and_a_query_together():
    with pytest.raises(ValueError, match="not both"):
        samplesar.sources.gbif(RECORDS, country="CR")


def test_nothing_usable():
    with pytest.raises(ValueError, match="No usable GBIF records"):
        samplesar.sources.gbif([record(1, rank="GENUS")])


@pytest.fixture
def fake_pygbif(monkeypatch):
    """A pygbif serving 650 matching records, 300 a page, and logging each call."""
    pool = [record(i) for i in range(650)]
    calls = []

    def search(**params):
        calls.append(params)
        offset, limit = params["offset"], params["limit"]
        page = pool[offset : offset + limit]
        return {"count": len(pool), "endOfRecords": offset + limit >= len(pool), "results": page}

    occurrences = types.ModuleType("pygbif.occurrences")
    occurrences.search = search
    module = types.ModuleType("pygbif")
    module.occurrences = occurrences
    monkeypatch.setitem(sys.modules, "pygbif", module)
    monkeypatch.setitem(sys.modules, "pygbif.occurrences", occurrences)
    return calls


def test_fetches_every_page(fake_pygbif):
    sample = samplesar.sources.gbif(taxonKey=952, country="CR", offset=50)
    assert sample.n_observations == 650
    assert [c["offset"] for c in fake_pygbif] == [0, 300, 600]
    assert all(c["hasCoordinate"] and not c["hasGeospatialIssue"] for c in fake_pygbif)
    assert all(c["occurrenceStatus"] == "PRESENT" and c["taxonKey"] == 952 for c in fake_pygbif)


def test_max_results_and_last_page_size(fake_pygbif):
    assert samplesar.sources.gbif(country="CR", max_results=400).n_observations == 400
    assert [c["limit"] for c in fake_pygbif] == [300, 100]


def test_a_boundary_narrows_the_query(fake_pygbif):
    samplesar.sources.gbif(boundary=box(-85, 9, -83, 11), max_results=10)
    assert fake_pygbif[0]["decimalLongitude"] == "-85.0,-83.0"
    assert fake_pygbif[0]["decimalLatitude"] == "9.0,11.0"


def test_more_than_the_api_serves_warns(fake_pygbif, monkeypatch):
    from samplesar import _gbif

    monkeypatch.setattr(_gbif, "OFFSET_LIMIT", 500)
    with pytest.warns(UserWarning, match="serves only the first 500"):
        assert samplesar.sources.gbif(country="CR").n_observations == 500


def test_a_query_needs_pygbif(monkeypatch):
    monkeypatch.setitem(sys.modules, "pygbif", None)
    with pytest.raises(ImportError, match=r"samplesar\[gbif\]"):
        samplesar.sources.gbif(country="CR")
