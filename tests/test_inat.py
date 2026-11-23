"""Samples from iNaturalist records, with pyinaturalist faked out."""

import sys
import types

import pytest

import samplesar


def record(i, name="Rana sylvatica", rank="species", lon=-82.2, lat=36.5, **extra):
    return {
        "id": i,
        "taxon": {"name": name, "rank": rank},
        "geojson": {"type": "Point", "coordinates": [lon + i * 1e-3, lat + (i % 7) * 1e-3]},
        **extra,
    }


RECORDS = [
    record(1),
    record(2, name="Lithobates clamitans"),
    record(3, name="Lithobates", rank="genus"),
    record(4, name="Hyla chrysoscelis", obscured=True),
    record(5, name="Pseudacris crucifer", geoprivacy="private"),
    {"id": 6, "taxon": None, "geojson": {"coordinates": [-82.2, 36.5]}},
    {
        "id": 7,
        "taxon": {"name": "Anaxyrus americanus", "rank": "species"},
        "location": "36.51,-82.21",
    },
    {
        "id": 8,
        "taxon": {"name": "Anaxyrus fowleri", "rank": "species"},
        "location": [36.52, -82.22],
    },
    {"id": 9, "taxon": {"name": "Gastrophryne", "rank": "species"}},
]


def test_records_become_a_sample():
    sample = samplesar.sources.inaturalist(RECORDS)
    assert sorted(sample.data["species"]) == [
        "Anaxyrus americanus",
        "Anaxyrus fowleri",
        "Lithobates clamitans",
        "Rana sylvatica",
    ]
    row = sample.data.set_index("species").loc["Anaxyrus fowleri"]
    assert (row["lon"], row["lat"]) == (-82.22, 36.52)


def test_subspecies_count_as_their_species():
    records = [
        record(1, name="Lithobates sphenocephalus utricularius", rank="subspecies"),
        record(2, name="Lithobates sphenocephalus"),
    ]
    sample = samplesar.sources.inaturalist(records)
    assert sample.data["species"].tolist() == ["Lithobates sphenocephalus"] * 2


def test_filters_can_be_relaxed():
    sample = samplesar.sources.inaturalist(RECORDS, ranks=None, skip_obscured=False)
    assert {"Lithobates", "Hyla chrysoscelis", "Pseudacris crucifer"} <= set(sample.data["species"])


def test_nothing_usable():
    with pytest.raises(ValueError, match="No usable iNaturalist observations"):
        samplesar.sources.inaturalist([record(1, rank="genus")])


def test_records_and_a_query_together():
    with pytest.raises(ValueError, match="not both"):
        samplesar.sources.inaturalist(RECORDS, place_id=1)
    with pytest.raises(ValueError, match="not both"):
        samplesar.sources.inaturalist(RECORDS, max_results=2)


@pytest.fixture
def fake_pyinaturalist(monkeypatch):
    """A pyinaturalist whose get_observations serves 450 records, 200 a page,
    honoring id_above, and logs each call's parameters."""
    pool = [record(i) for i in range(1, 451)]
    calls = []

    def get_observations(**params):
        calls.append(params)
        above = params.get("id_above", 0)
        page = [r for r in pool if r["id"] > above][: params["per_page"]]
        return {"total_results": len(pool), "results": page}

    module = types.ModuleType("pyinaturalist")
    module.get_observations = get_observations
    monkeypatch.setitem(sys.modules, "pyinaturalist", module)
    return calls


def test_fetches_every_page_by_id(fake_pyinaturalist):
    sample = samplesar.sources.inaturalist(place_id=130630, page=3)
    assert sample.n_observations == 450
    assert [c.get("id_above") for c in fake_pyinaturalist] == [None, 200, 400]
    assert all(c["order_by"] == "id" and c["order"] == "asc" for c in fake_pyinaturalist)
    assert all(c["place_id"] == 130630 and "page" not in c for c in fake_pyinaturalist)


def test_max_results(fake_pyinaturalist):
    sample = samplesar.sources.inaturalist(place_id=1, max_results=250)
    assert sample.n_observations == 250
    # The last page asks only for what's still needed.
    assert [c["per_page"] for c in fake_pyinaturalist] == [200, 50]


def test_a_query_needs_pyinaturalist(monkeypatch):
    monkeypatch.setitem(sys.modules, "pyinaturalist", None)
    with pytest.raises(ImportError, match=r"samplesar\[inat\]"):
        samplesar.sources.inaturalist(place_id=1)


def test_pages_larger_than_the_api_serves_are_capped(fake_pyinaturalist):
    sample = samplesar.sources.inaturalist(place_id=1, per_page=500)
    assert sample.n_observations == 450
    assert all(c["per_page"] == 200 for c in fake_pyinaturalist)


def test_a_fetch_resumes_from_id_above(fake_pyinaturalist):
    sample = samplesar.sources.inaturalist(place_id=1, id_above=300)
    assert sample.n_observations == 150
    assert fake_pyinaturalist[0]["id_above"] == 300


@pytest.mark.parametrize(
    ("option", "match"), [({"max_results": 0}, "max_results"), ({"per_page": 0}, "per_page")]
)
def test_sizes_below_one(fake_pyinaturalist, option, match):
    with pytest.raises(ValueError, match=f"{match} must be at least 1"):
        samplesar.sources.inaturalist(place_id=1, **option)


def test_per_page_none_means_the_default(fake_pyinaturalist):
    samplesar.sources.inaturalist(place_id=1, per_page=None, max_results=10)
    assert fake_pyinaturalist[0]["per_page"] == 10
