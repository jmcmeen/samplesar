"""A first nest: observations to a species-area table, a fitted curve and a map.

python examples/01_first_nest.py
"""

import matplotlib

matplotlib.use("Agg")  # save figures without opening a window
import matplotlib.pyplot as plt
from _common import output_dir

import samplesar

out = output_dir(__doc__)

# 1. Observations. A made-up landscape here; for your own data use
#    samplesar.sources.csv("observations.csv"), with columns species, lon, lat.
sample = samplesar.sources.simulated(seed=1)
print(sample)

# 2. Records made at one spot differ only by GPS jitter: merge points
#    within 20 m of each other, keeping every observation.
sample = sample.merge_within(20)

# 3. Nest it: peel the convex hull, layer by layer, from the outside in.
nest = samplesar.strategies.hull_peel(sample)
table = nest.to_frame()  # step, n_points, area (km²), species
print(table.head(8).to_string(index=False))

# 4. Fit a species-area model with sars. "power" is S = c * A^z.
fit = nest.fit("power")
print(
    f"\npower model: c = {fit.params['c']:.3g}, z = {fit.params['z']:.3f}, R² = {fit.r_squared:.3f}"
)

# 5. Draw the full sample, a later step, and the curve.
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
samplesar.plot.nest(nest, step=0, ax=axes[0])
samplesar.plot.nest(nest, step=nest.n_steps // 2, ax=axes[1])
samplesar.plot.curve(nest, ax=axes[2])
fig.tight_layout()
fig.savefig(out / "01_first_nest.png", dpi=120)
print(f"\nSaved {out / '01_first_nest.png'}")

# 6. Keep the nest: the observations, the strategy and every step, as JSON.
nest.to_json(out / "01_first_nest.json")
same = samplesar.Nest.from_json(out / "01_first_nest.json")
assert same.n_steps == nest.n_steps
