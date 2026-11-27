"""Species-area relationship datasets from geometric sampling of point observations.

The API is grouped by job: :mod:`samplesar.sources` gets observations into a
:class:`Sample`, :mod:`samplesar.regions` gives regions to restrict them to,
:mod:`samplesar.strategies` nests them into a :class:`Nest`,
:mod:`samplesar.designs` divides them among regions or patches instead,
:mod:`samplesar.evaluate` fits and compares nests, and :mod:`samplesar.plot`
draws them.
"""

from importlib.metadata import PackageNotFoundError, version

from samplesar import designs, evaluate, plot, regions, sources, strategies
from samplesar._islands import Islands
from samplesar._nest import Nest
from samplesar._quadrats import Quadrats
from samplesar._sample import Sample, hull_area

try:
    __version__ = version("samplesar")
except PackageNotFoundError:  # running from a source tree without an install
    __version__ = "0.0.0"

__all__ = [
    "Islands",
    "Nest",
    "Quadrats",
    "Sample",
    "__version__",
    "designs",
    "evaluate",
    "hull_area",
    "plot",
    "regions",
    "sources",
    "strategies",
]
