"""Draw docs/assets/mask.webp: a made-up island, nested without and with a mask.

Run from the repository root: ``uv run python docs/assets/mask_figure.py``.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import Point

import samplesar

# A lumpy island about 20 km across, with an inlet from the east.
land = shapely.union_all(
    [
        Point(x, y).buffer(r, quad_segs=24)
        for x, y, r in [
            (-0.04, 0.02, 0.06),
            (0.03, 0.04, 0.055),
            (0.0, -0.04, 0.06),
            (0.05, -0.03, 0.04),
        ]
    ]
)
# A winding inlet from the east, reaching the island's middle.
inlet = shapely.LineString([(0.13, 0.0), (0.07, 0.01), (0.03, -0.005), (-0.01, 0.01)]).buffer(
    0.012, quad_segs=8
)
island = land.difference(inlet)
rng = np.random.default_rng(3)
points = []
while len(points) < 160:
    lon, lat = rng.uniform(-0.11, 0.11, 2)
    if island.contains(Point(lon, lat)):
        points.append((lon, lat))
lon, lat = np.array(points).T
species = [f"sp{int(abs(y) * 300 + x * 100) % 45}" for x, y in points]
sample = samplesar.sources.dataframe(pd.DataFrame({"species": species, "lon": lon, "lat": lat}))
masked = sample.with_mask(island)

fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
for ax, s, title in (
    (axes[0], sample, "No mask: the whole hull counts"),
    (axes[1], masked, "With a mask: only the land in the hull counts"),
):
    samplesar.plot.nest(samplesar.Nest(s, (np.arange(s.n_points),)), ax=ax, show_removed=False)
    if s.mask is None:  # draw the coast anyway, for comparison
        x, y = island.exterior.xy
        ax.plot(x, y, color="0.45", linewidth=0.8, zorder=1)
    ax.set_title(f"{title}\narea = {s.area:,.0f} km²", fontsize=11)
    ax.set_xlim(-0.115, 0.115)
    ax.set_ylim(-0.115, 0.115)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
axes[1].annotate(
    "sea inside the hull:\nnot counted",
    xy=(0.055, 0.004),
    xytext=(0.04, 0.1),
    fontsize=10,
    arrowprops={"arrowstyle": "->", "color": "0.3"},
    bbox={"boxstyle": "round", "fc": "white", "ec": "0.7"},
)
fig.tight_layout()
fig.savefig(
    Path(__file__).with_name("mask.webp"), dpi=90, pil_kwargs={"lossless": True, "method": 6}
)
