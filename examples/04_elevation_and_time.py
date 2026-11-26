"""Nests by elevation and by date of first visit.

python examples/04_elevation_and_time.py
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

# The simulated observations carry an elevation (m) and a date. For your
# own table, name the columns:
#   samplesar.sources.csv("obs.csv", elevation_col="elev_m", date_col="observed_on")
# GBIF records keep theirs (samplesar.sources.gbif), and elevations looked
# up elsewhere can be set per point with sample.with_elevations(values).
sample = samplesar.sources.simulated(seed=4).merge_within(20)
print(
    f"Elevations {sample.elevations.min():.0f} to {sample.elevations.max():.0f} m; "
    f"first visits {sample.dates.min().astype('datetime64[D]')} to "
    f"{sample.dates.max().astype('datetime64[D]')}"
)

nests = {
    "from the summits down": samplesar.strategies.elevation(sample, remove="highest"),
    "from the valleys up": samplesar.strategies.elevation(sample, remove="lowest"),
    "closing on 1,000 m": samplesar.strategies.elevation(sample, remove="outer", around=1000),
    "back through time": samplesar.strategies.temporal(sample, remove="newest"),
}
table = samplesar.evaluate.compare(nests)
print(table[["n_steps", "z", "r_squared"]].round(3).to_string())
# Read these as diagnostics, not as the region's z. Removing the summits
# first leaves the valleys' hull almost whole, so species fall while area
# barely moves and the fitted z is huge; richness also varies with height
# for reasons that have nothing to do with area.

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for name, nest in nests.items():
    data = nest.sar_data()
    axes[0].plot(data["area"], data["species"], linewidth=1.2, label=name)
axes[0].set_xscale("log")
axes[0].set_yscale("log")
axes[0].set_xlabel("Area (km²)")
axes[0].set_ylabel("Species")
axes[0].legend(fontsize=8)

# What a nest through time means: the places known by each date.
through_time = nests["back through time"]
last_date = [sample.dates[step].max() for step in through_time.steps]
axes[1].plot(last_date, through_time.to_frame()["species"], color="C3")
axes[1].set_xlabel("Places first visited up to")
axes[1].set_ylabel("Species known")
fig.tight_layout()
fig.savefig(out / "04_elevation_and_time.png", dpi=120)
print(f"\nSaved {out / '04_elevation_and_time.png'}")
