"""Draw docs/assets/first_nest.webp: a simulated sample, one of its nests, and its curve.

Run from the repository root: ``uv run python docs/assets/first_nest_figure.py``.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import samplesar

sample = samplesar.sources.simulated(seed=1).merge_within(20)
nest = samplesar.strategies.hull_peel(sample)

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
samplesar.plot.nest(nest, step=0, ax=axes[0])
samplesar.plot.nest(nest, step=12, ax=axes[1])
samplesar.plot.curve(nest, ax=axes[2])
axes[2].set_title("each step: one point of the curve")
fig.tight_layout()
fig.savefig(
    Path(__file__).with_name("first_nest.webp"), dpi=90, pil_kwargs={"lossless": True, "method": 6}
)
