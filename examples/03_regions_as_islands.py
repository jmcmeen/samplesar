"""Regions as sampling units: one row per region, fitted as islands.

python examples/03_regions_as_islands.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _common import output_dir
from shapely.geometry import box

import samplesar

out = output_dir(__doc__)
sample = samplesar.sources.simulated(n_points=1500, seed=3)

# Regions are named polygons in longitude and latitude. Made-up "parks" of
# different sizes here. For real ones:
#   states = samplesar.regions.admin_units("USA", level=1)           # needs the network
#   parks = samplesar.regions.read_units("parks.gpkg", "UNIT_NAME")  # any boundary file
west, south, east, north = -83.93, 35.25, -83.07, 35.95
parks = {
    "North Fork": box(west, 35.70, -83.50, north),
    "Ridge Top": box(-83.60, 35.50, -83.35, 35.68),
    "Mill Creek": box(-83.30, 35.30, -83.20, 35.38),
    "Big Valley": box(-83.50, south, east, 35.60),
    "Laurel Gap": box(-83.90, 35.30, -83.80, 35.36),
    "East Bench": box(-83.25, 35.70, east, north),
    "Cove": box(-83.75, 35.42, -83.68, 35.47),
}

islands = samplesar.designs.islands(sample, parks)

# One row per region: its own area (km²), the species observed in it, and
# how much it was sampled. Effort matters: more visits, more species.
table = islands.to_frame()
print(table.round(1).to_string(index=False))

fit = islands.fit("power")
print(f"\nIsland-type power model: z = {fit.params['z']:.3f}, R² = {fit.r_squared:.3f}")

# The same observations as one nest give a different kind of curve: its
# areas are whatever hulls remain, and its steps share their points.
nest = samplesar.strategies.hull_peel(sample)
print(f"Nested (hull_peel) power model: z = {nest.fit('power').params['z']:.3f}")

fig, ax = plt.subplots(figsize=(6, 4.5))
samplesar.plot.curve(islands, ax=ax)
for row in table.itertuples():
    ax.annotate(
        row.region, (row.area, row.species), fontsize=7, xytext=(4, 4), textcoords="offset points"
    )
fig.tight_layout()
fig.savefig(out / "03_regions_as_islands.png", dpi=120)
print(f"\nSaved {out / '03_regions_as_islands.png'}")

# One region's observations, measured within the region, ready to nest.
ridge = islands.sample_of("Ridge Top")
print(f"\nRidge Top alone: {ridge.n_points} points, {ridge.n_species} species")
