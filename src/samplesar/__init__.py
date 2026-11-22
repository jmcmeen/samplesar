"""Species-area relationship datasets from geometric sampling of point observations."""

from importlib.metadata import PackageNotFoundError, version

from samplesar._evaluate import compare_nests
from samplesar._hull import (
    hull_area_nest,
    hull_distance_nest,
    hull_peel_nest,
    hull_random_nest,
    hull_species_nest,
)
from samplesar._nest import Nest, nest_from_order, random_nest
from samplesar._plot import animate_nest, plot_nest, plot_nest_curve
from samplesar._registry import nest, register_strategy, strategies
from samplesar._sample import Sample, hull_area

try:
    __version__ = version("samplesar")
except PackageNotFoundError:  # running from a source tree without an install
    __version__ = "0.0.0"

__all__ = [
    "Nest",
    "Sample",
    "__version__",
    "animate_nest",
    "compare_nests",
    "hull_area",
    "hull_area_nest",
    "hull_distance_nest",
    "hull_peel_nest",
    "hull_random_nest",
    "hull_species_nest",
    "nest",
    "nest_from_order",
    "plot_nest",
    "plot_nest_curve",
    "random_nest",
    "register_strategy",
    "strategies",
]
