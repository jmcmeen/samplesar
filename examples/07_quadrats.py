"""Equal-area patches as sampling units: quadrats in a grid or scattered.

python examples/07_quadrats.py
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

# Square patches of four sizes, each size a grid over the sample. One row
# per patch: its side, its area (km²), its species and how much it was sampled.
patches = samplesar.designs.quadrats(sample, [5, 10, 20, 40])
table = patches.to_frame()
print(patches.label)
summary = table.groupby("size_km").agg(
    patches=("region", "size"), mean_species=("species", "mean"), mean_points=("n_points", "mean")
)
print(summary.round(1).to_string())

# Every patch with observations is a row of the species-area data, so
# patches of several sizes can be fitted as islands are.
fit = patches.fit("power")
print(f"\nOne row per patch, power model: z = {fit.params['z']:.3f}")

# Patches of one size give a curve of their own by accumulation: the species
# in 1, 2, 3, ... patches against their combined area. Combining any patches
# mixes far corners of the sample; combining neighbors keeps them together,
# and finds fewer species at the same area.
anywhere = patches.accumulate(10)
neighbors = patches.accumulate(10, order="nearest")
half = len(anywhere) // 2
print(
    f"\n{half} patches of 10 km: {anywhere['species'].iloc[half - 1]:.1f} species from any "
    f"patches, {neighbors['species'].iloc[half - 1]:.1f} from neighbors"
)

# Patches can also be scattered: 6 of each size, placed at random without
# overlapping. The seed makes the placement repeatable.
scattered = samplesar.designs.quadrats(sample, [5, 10, 20], n=6, seed=1)
print(f"\nScattered: {scattered.label}, z = {scattered.fit('power').params['z']:.3f}")

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
samplesar.plot.islands(
    patches.select(table.loc[table["size_km"] == 20, "region"]), ax=axes[0], labels=False
)
axes[0].set_title("A grid of 20 km patches")
samplesar.plot.islands(scattered, ax=axes[1], labels=False)
axes[1].set_title("6 scattered patches each of 5, 10 and 20 km")
axes[2].plot(anywhere["area"], anywhere["species"], label="any 10 km patches")
axes[2].plot(neighbors["area"], neighbors["species"], label="neighboring 10 km patches")
data = patches.sar_data()
axes[2].scatter(data["area"], data["species"], s=8, color="0.6", label="one patch each")
axes[2].set_xscale("log")
axes[2].set_yscale("log")
axes[2].set_xlabel("Area (km²)")
axes[2].set_ylabel("Species")
axes[2].legend()
fig.tight_layout()
fig.savefig(out / "07_quadrats.png", dpi=120)
print(f"\nSaved {out / '07_quadrats.png'}")
