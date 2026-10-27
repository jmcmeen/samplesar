"""Nested species-area relationship datasets from point observations."""

from importlib.metadata import PackageNotFoundError, version

from nestsar._nest import Nest, random_nest
from nestsar._sample import Sample, hull_area

try:
    __version__ = version("nestsar")
except PackageNotFoundError:  # running from a source tree without an install
    __version__ = "0.0.0"

__all__ = ["Nest", "Sample", "__version__", "hull_area", "random_nest"]
