"""Real observations: GBIF or iNaturalist records for a region, with a land mask.

This one needs the network and the optional sources:

    pip install 'samplesar[gbif,inat]' matplotlib
    python examples/06_real_observations.py --country CRI --taxon-key 952

Records are fetched from GBIF (taxon 952 is the frogs, Anura), restricted to
the country's boundary from geoBoundaries, and their areas measured on land
only. Both services ask to be credited: cite the GBIF download or the
iNaturalist observations you use, and geoBoundaries for the boundary.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import samplesar

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("--country", default="CRI", help="ISO 3166-1 alpha-3 code (default: CRI)")
parser.add_argument(
    "--taxon-key", type=int, default=952, help="GBIF taxon key (default: 952, Anura)"
)
parser.add_argument(
    "--max-results", type=int, default=3000, help="records to fetch (default: 3000)"
)
parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "output")
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=True)

# The country's outline, downloaded once and cached.
country = samplesar.regions.admin(args.country)

# Records inside it. boundary= keeps only those within the outline.
sample = samplesar.sources.gbif(
    taxonKey=args.taxon_key,
    country=None,
    boundary=country,
    max_results=args.max_results,
    max_uncertainty=1000,  # coordinates good to a km
)
print(sample)

# Measure areas on land only: a hull along a coast is partly sea.
sample = sample.with_mask(country).merge_within(20)

nest = samplesar.strategies.hull_peel(sample)
print(nest.to_frame().head(8).to_string(index=False))
fit = nest.fit("power")
print(f"\npower model: z = {fit.params['z']:.3f}, R² = {fit.r_squared:.3f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
samplesar.plot.nest(nest, step=0, ax=axes[0])
samplesar.plot.curve(nest, ax=axes[1])
fig.tight_layout()
name = f"06_{args.country.lower()}_{args.taxon_key}.png"
fig.savefig(args.out / name, dpi=120)
print(f"\nSaved {args.out / name}")

# From iNaturalist instead (research-grade frogs, taxon 20979, in a place):
#   sample = samplesar.sources.inaturalist(taxon_id=20979, place_id=6924, quality_grade="research")
