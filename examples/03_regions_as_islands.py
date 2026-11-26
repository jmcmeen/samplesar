"""Regions as sampling units: one row per region, fitted as islands.

python examples/03_regions_as_islands.py
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from shapely.geometry import box

import samplesar

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument(
    "--out",
    type=Path,
    default=Path(__file__).resolve().parent / "output",
    help="folder for figures (default: examples/output)",
)
out = parser.parse_args().out
out.mkdir(parents=True, exist_ok=True)
sample = samplesar.sources.simulated(n_points=1500, seed=3)

# Regions are named polygons in longitude and latitude. Eight made-up "parks"
# of different sizes here. For real ones:
#   states = samplesar.regions.admin_units("USA", level=1)           # needs the network
#   parks = samplesar.regions.read_units("parks.gpkg", "UNIT_NAME")  # any boundary file
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

# The regions on a map, each with its points and species count, and their curve.
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
samplesar.plot.islands(islands, ax=axes[0])
samplesar.plot.curve(islands, ax=axes[1])
for row in table.itertuples():
    axes[1].annotate(
        row.region, (row.area, row.species), fontsize=7, xytext=(4, 4), textcoords="offset points"
    )
fig.tight_layout()
fig.savefig(out / "03_regions_as_islands.png", dpi=120)
print(f"\nSaved {out / '03_regions_as_islands.png'}")

# A design can also be put together by hand: pick some regions, combine
# sources, use an area you know, and type in units that have no boundary.
by_hand = samplesar.designs.islands(
    sample,
    {name: parks[name] for name in ("North Fork", "Ridge Top", "Hollow")},  # a subset
    areas={"Ridge Top": 400.0},  # an official area, in place of the outline's
    units={"Far Ridge": (1200.0, 14), "Old Mill": (25.0, 3)},  # from a checklist: km², species
)
by_hand = by_hand.add({"Cove": parks["Cove"]})  # and one more region, from anywhere
print("\nBuilt by hand:")
print(by_hand.to_frame()[["region", "area", "species", "source"]].round(1).to_string(index=False))
print(f"power model: z = {by_hand.fit('power').params['z']:.3f}")

# One region's observations, measured within the region, ready to nest.
ridge = islands.sample_of("Ridge Top")
print(f"\nRidge Top alone: {ridge.n_points} points, {ridge.n_species} species")
