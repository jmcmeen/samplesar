"""Requests for the files samplesar downloads."""

from __future__ import annotations

import urllib.request
from collections.abc import Mapping

HOME = "https://github.com/jmcmeen/samplesar"


def request(
    url: str, headers: Mapping[str, str] | None = None, method: str | None = None
) -> urllib.request.Request:
    """A request that names samplesar as its client, with where to find it,
    so a server's operator who sees a problem can report it instead of
    blocking. Some servers (geoBoundaries among them) refuse Python's
    default ``User-Agent``."""
    from samplesar import __version__

    named = {"User-Agent": f"samplesar/{__version__} (+{HOME})", **(headers or {})}
    return urllib.request.Request(url, headers=named, method=method)
