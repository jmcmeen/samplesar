"""Downloads name samplesar as their client."""

import io
import urllib.request

import pytest

import samplesar
from samplesar import _geoboundaries, _opendata, _regions
from samplesar._http import request


def test_requests_name_samplesar():
    made = request("https://example.org/a", headers={"Range": "bytes=5-"}, method="HEAD")
    assert made.get_header("User-agent") == (
        f"samplesar/{samplesar.__version__} (+https://github.com/jmcmeen/samplesar)"
    )
    assert made.get_header("Range") == "bytes=5-" and made.get_method() == "HEAD"


class Reply(io.BytesIO):
    status = 200
    headers = {"Content-Length": "2", "ETag": "v1"}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


@pytest.fixture
def sent(monkeypatch):
    seen = []

    def urlopen(target, timeout=None):
        seen.append(target)
        return Reply(b"{}")

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    return seen


def test_every_download_sends_it(sent, tmp_path):
    # geoBoundaries answers Python's default User-Agent with 403.
    _geoboundaries._fetch("https://example.org/meta")
    _regions._fetch_to("https://example.org/file.zip", tmp_path / "file.zip")
    _opendata._remote("https://example.org/data.parquet")
    assert len(sent) == 3
    for target in sent:
        assert isinstance(target, urllib.request.Request)
        assert target.get_header("User-agent").startswith("samplesar/")
