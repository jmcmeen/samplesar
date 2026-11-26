"""Clustered samples: split into patches, thin the piles, and a band from many random nests.

python examples/05_patches_and_effort.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _common import output_dir

import samplesar

out = output_dir(__doc__)

# Most points piled around a few places, as observation data are.
sample = samplesar.sources.simulated(n_points=500, clustered=0.9, seed=5).merge_within(20)

# 1. One hull around separate clusters spans the empty ground between them.
#    split nests the clusters as separate patches: areas are the patches'
#    hulls together.
one_hull = samplesar.strategies.hull_peel(sample)
patches = samplesar.strategies.split(sample, "hull_peel", parts=5, seed=0)
print(
    f"One hull: {one_hull.to_frame()['area'][0]:,.0f} km²; "
    f"five patches: {patches.to_frame()['area'][0]:,.0f} km²"
)

# 2. Thin to at most one point per 2 km cell, so a pile counts once.
thinned = sample.thin(2.0)
print(
    f"Thinning keeps {thinned.n_points} of {sample.n_points} points "
    f"and {thinned.n_species} of {sample.n_species} species"
)

# 3. A random strategy gives a different nest every time. Run many, and
#    read off the expected curve with the band one nest falls in.
band = samplesar.evaluate.ensemble(sample, "hull_random", n=49, seed=0)

# 4. A deterministic strategy gives one nest. Resample the points to see
#    how much its fit owes to which places were sampled.
fits = samplesar.evaluate.bootstrap(sample, "hull_peel", n=49, seed=0)
low, mid, high = fits["z"].quantile([0.025, 0.5, 0.975])
print(
    f"hull_peel z = {one_hull.fit().params['z']:.3f}; "
    f"over resampled points {mid:.3f} ({low:.3f} to {high:.3f})"
)

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
samplesar.plot.nest(patches, step=0, ax=axes[0])
axes[0].set_title("Five patches, each with its own hull")
axes[1].fill_between(band["area"], band["q025"], band["q975"], alpha=0.3, label="95% of nests")
axes[1].plot(band["area"], band["mean"], label="mean of 49 hull_random nests")
for name, nest in (
    ("hull_peel", one_hull),
    ("hull_peel, thinned", samplesar.strategies.hull_peel(thinned)),
):
    data = nest.sar_data()
    axes[1].plot(data["area"], data["species"], marker=".", linewidth=1, label=name)
axes[1].set_xscale("log")
axes[1].set_yscale("log")
axes[1].set_xlabel("Area (km²)")
axes[1].set_ylabel("Species")
axes[1].legend(fontsize=8)
axes[2].hist(fits["z"].dropna(), bins=15, color="C2")
axes[2].axvline(one_hull.fit().params["z"], color="k", linestyle="--")
axes[2].set_xlabel("z of hull_peel on resampled points")
fig.tight_layout()
fig.savefig(out / "05_patches_and_effort.png", dpi=120)
print(f"\nSaved {out / '05_patches_and_effort.png'}")
