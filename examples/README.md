# Examples

Short scripts, each doing one job from observations to a table and a
figure. Each stands alone: download one, or clone the repository, and run
it with samplesar installed:

```sh
pip install samplesar
git clone https://github.com/jmcmeen/samplesar.git
python samplesar/examples/01_first_nest.py
```

Each prints its table and saves its figure in an `output` folder beside
the script (or the folder given with `--out`). They run offline on
`samplesar.sources.simulated()`, a made-up landscape of observations with
elevations and dates, so they finish in seconds; each one says where to
put your own data instead.

| Script | What it shows |
| --- | --- |
| `01_first_nest.py` | Observations to a nest, its species-area table, a fitted curve and a map |
| `02_compare_strategies.py` | Several strategies on one sample, and where each falls against a random baseline |
| `03_regions_as_islands.py` | Regions (states, parks) as sampling units: one row per region, fitted as islands |
| `04_elevation_and_time.py` | Nests by elevation and by date of first visit |
| `05_patches_and_effort.py` | Clustered samples: split into patches, thin the piles, and a band from many random nests |
| `06_real_observations.py` | iNaturalist or GBIF records for a country or state, with a land mask (needs the network) |

The same journeys, with their figures, are notebooks in the documentation:
[a first nest](https://jmcmeen.github.io/samplesar/examples/first_nest/),
[comparing strategies](https://jmcmeen.github.io/samplesar/examples/strategies/) and
[regions as islands](https://jmcmeen.github.io/samplesar/examples/islands/).
