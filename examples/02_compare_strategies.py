"""Several strategies on one sample, and where each falls against a random baseline.

python examples/02_compare_strategies.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _common import output_dir

import samplesar

out = output_dir(__doc__)
sample = samplesar.sources.simulated(n_points=300, seed=2).merge_within(20)

# Every strategy is a function, and can also be run by name.
print("Available:", ", ".join(samplesar.strategies.available()))

nests = {
    "hull_peel": samplesar.strategies.hull_peel(sample),
    "radial": samplesar.strategies.radial(sample),
    "density_peel": samplesar.strategies.density_peel(sample),
    "hull_species (keep most)": samplesar.strategies.hull_species(sample, keep="most"),
    "hull_rarity (rare first)": samplesar.strategies.hull_rarity(sample, remove="rare"),
    "random": samplesar.strategies.random(sample, seed=0),
}

# One row per nest: the fitted parameters and how well the model fits.
table = samplesar.evaluate.compare(nests, model="power")
print(table[["n_steps", "c", "z", "r_squared"]].round(3).to_string())

# Steps of one nest share most of their points, so its R² isn't independent
# evidence. Judge a strategy against many random nests of the same sample.
baseline = samplesar.evaluate.baseline(sample, n=49, strategy="hull_random")
for name in ("hull_peel", "hull_rarity (rare first)"):
    placed = samplesar.evaluate.against_baseline(nests[name], baseline)
    z = placed.loc["z"]
    print(
        f"\n{name}: z = {z['value']:.3f}; random hull-vertex nests give "
        f"{z['q025']:.3f} to {z['q975']:.3f} (p = {z['p']:.2f})"
    )

# The curves together.
fig, ax = plt.subplots(figsize=(7, 5))
for name, nest in nests.items():
    data = nest.sar_data()
    ax.plot(data["area"], data["species"], marker=".", markersize=3, linewidth=1, label=name)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Area (km²)")
ax.set_ylabel("Species")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(out / "02_compare_strategies.png", dpi=120)
print(f"\nSaved {out / '02_compare_strategies.png'}")
