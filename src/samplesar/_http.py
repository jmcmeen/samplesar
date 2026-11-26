"""Requests for the files samplesar downloads."""

from __future__ import annotations

import urllib.request
from collections.abc import Mapping


def request(
    url: str, headers: Mapping[str, str] | None = None, method: str | None = None
) -> urllib.request.Request:
    """A request that names samplesar as its client. Some servers
    (geoBoundaries among them) refuse Python's default ``User-Agent``."""
    from samplesar import __version__

    named = {"User-Agent": f"samplesar/{__version__}", **(headers or {})}
    return urllib.request.Request(url, headers=named, method=method)
