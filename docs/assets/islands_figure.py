"""Draw docs/assets/islands.webp: made-up parks as sampling units, and their curve.

Run from the repository root: ``uv run python docs/assets/islands_figure.py``.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from shapely.geometry import box

import samplesar

sample = samplesar.sources.simulated(n_points=1500, seed=3)
parks = {
    "North Fork": box(-83.93, 35.72, -83.52, 35.95),
    "East Bench": box(-83.24, 35.70, -83.07, 35.95),
    "Ridge Top": box(-83.62, 35.52, -83.36, 35.68),
    "Big Valley": box(-83.30, 35.42, -83.07, 35.66),
    "Hollow": box(-83.60, 35.27, -83.42, 35.44),
    "Mill Creek": box(-83.30, 35.27, -83.20, 35.36),
    "Laurel Gap": box(-83.90, 35.28, -83.80, 35.35),
    "Cove": box(-83.76, 35.42, -83.68, 35.48),
}
islands = samplesar.designs.islands(sample, parks)

fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
samplesar.plot.islands(islands, ax=axes[0])
samplesar.plot.curve(islands, ax=axes[1])
for row in islands.to_frame().itertuples():
    axes[1].annotate(
        row.region, (row.area, row.species), fontsize=7, xytext=(4, 4), textcoords="offset points"
    )
axes[1].set_title("one point per region")
fig.tight_layout()
fig.savefig(
    Path(__file__).with_name("islands.webp"), dpi=90, pil_kwargs={"lossless": True, "method": 6}
)
