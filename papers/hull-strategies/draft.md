# Nested species-area relationships from point observations: convex hull sampling strategies

John McMeen

*Pre-draft, 22 November 2026. Numbers come from the full run of
`analysis/run.py` (99 baseline nests, 20 simulation replicates; see
Reproducibility) and are preliminary until the analysis settles. References
are to be checked before submission.*

## Abstract

Species-area relationships (SARs) are usually built from separate islands or
plots, but the growing volume of georeferenced observations, from community
science platforms especially, invites building them from a single set of
points. A nested SAR can be made by shrinking a sample step by step and
recording area and richness at each step, but how the sample shrinks is a
choice, and the choice shapes the curve. We define ten strategies for
shrinking a sample, from random removal to removing points from the convex
hull inwards by rules on area, species and distance, and compare them on five
iNaturalist samples spanning a park, a mountain range, a state and a tropical
cloud forest, and on simulated communities. Random removal keeps the sample's
full extent until its last points and gives near-vertical curves whose
power-model slope is meaningless. Strategies that shrink the hull from the
outside in agree with one another and give slopes of 0.15 to 0.34, the range
of classic SARs, while greedy rules bias the slope. Records made at one spot
dominate the end of every nest unless points within a few metres are merged,
which in simulations largely restores the slope that clustering destroys. We recommend hull peeling of merged points, compared
against a random-vertex baseline, as the default, and provide the strategies
in the Python package samplesar.

## 1. Introduction

The species-area relationship is among the most general patterns in ecology
(Arrhenius 1921; Preston 1962; Rosenzweig 1995). Its shape, usually
summarised by the slope *z* of a power law *S = cA^z*, depends on how areas
and their species are sampled: islands, nested plots and independent plots
each give a different kind of curve (Scheiner 2003; Dengler 2009).

Community science now supplies georeferenced species observations in great
numbers. Such points don't come in plots or islands, but a nested curve can
be built from them: start with all the points, remove some, record the area
they span and the species they hold, and repeat. Each step's points are a
subset of the step before, so area and richness both fall, tracing a nested
SAR across a sample's extent.

Which points to remove at each step is the crux. Removing them at random
keeps the sample's extent intact until nearly the end, so its area barely
falls while its richness does. Removing them from the outside in, peeling the
convex hull, shrinks the area from the first step, as a nested-plot design
does. Between these lie many rules: remove the hull vertex that costs least
area, or most; that loses fewest species, or most; that lies farthest from
the centre, or closest. Each makes a different curve from the same points.
Observation data add a further problem: records cluster where people go, at
trailheads and feeders, and many records from one spot differ only by GPS
error.

Here we (i) define ten removal strategies formally, (ii) measure how much the
strategy changes the fitted SAR on real and simulated samples, (iii) test the
effect of clustered records and a remedy for it, and (iv) recommend a default.

## 2. Methods

### 2.1 Samples and nests

A sample is a set of species observations at points. Points are projected to
the plane in km, by default with a Lambert azimuthal equal-area projection of
the WGS84 ellipsoid centred on the sample, so areas are true areas wherever
the sample lies. A nest is a sequence of point sets, each a subset of the one
before; its SAR data are each step's convex hull area (km²) and species
richness, for the steps with a positive area.

### 2.2 Strategies

The strategies (Table 1) are specified formally in
[strategies.md](strategies.md), with notation, tie rules, stopping rules and
costs. In brief, S1 removes a random point; S2 a random vertex of the convex
hull; S3 peels the whole hull boundary at each step, giving the sample's
convex layers (Chazelle 1985); S4 to S9 remove the hull vertex chosen by a
greedy rule on the remaining hull's area (S4 largest, S5 smallest), the
remaining richness (S6 most, S7 fewest) or the distance from the hull's
centre (S8 farthest, S9 closest); and S10 peels the boundary of a concave
hull, which follows the outline of an irregular sample instead of spanning
its gaps. All are implemented in samplesar.

**Table 1.** The strategies.

| | Strategy | Removes at each step |
| --- | --- | --- |
| S1 | Random | a random point |
| S2 | Random hull vertex | a random vertex of the convex hull |
| S3 | Hull peeling | every point on the convex hull's boundary |
| S4 | Largest remaining area | the vertex whose loss costs least area |
| S5 | Smallest remaining area | the vertex whose loss costs most area |
| S6 | Most remaining species | the vertex whose loss costs fewest species |
| S7 | Fewest remaining species | the vertex whose loss costs most species |
| S8 | Farthest from the centre | the vertex farthest from the vertices' centroid |
| S9 | Closest to the centre | the vertex closest to it |
| S10 | Concave peeling | every point on a concave hull's boundary (ratio 0.3) |

### 2.3 Merging co-located records

Points closer than a set distance were merged into one, keeping all their
observations (single linkage on the projected points; each group placed at
its first-seen member). The main results use 10 m for the park samples, 25 m
for Monteverde, 50 m for the Great Smoky Mountains and 100 m for East
Tennessee; sensitivity to the distance was tested from 0 to 50 m on the
single-observer park sample.

### 2.4 Fitting and baselines

The power model was fitted to each nest's SAR data by nonlinear least squares
with sars, a Python implementation of the models of the R package sars
(Matthews et al. 2019). Steps of one nest share nearly all their points, so a nest's R²,
AICc and standard errors aren't independent evidence. We therefore compared
each strategy with a baseline of 99 random-vertex (S2) nests of the same
sample, reporting where its *z* falls in the baseline's distribution as a
two-sided rank p-value. Fits whose parameters stopped on a bound of the
search (the power model's *z* at 5) were flagged rather than read as
estimates.

### 2.5 Diagnostics

For each nest we computed the lag-1 autocorrelation of the fit's residuals in
step order (with the Durbin-Watson statistic), as a measure of how far the
steps are from independent observations, and the compactness of each step's
hull (Polsby-Popper, 4πA/P²).

### 2.6 Data

Five iNaturalist samples (Table 2): one observer's records in Steele Creek
Park, Bristol, Tennessee; all research-grade records in the same area; and
research-grade records of frogs in East Tennessee, birds in the Great Smoky
Mountains, and frogs around Monteverde, Costa Rica, from a local copy of
iNaturalist's open-data export (observations with openly licensed photos).
Samples larger than 1,500 points after merging were thinned to 1,500 points
at random (seeded), to keep the area-based strategies tractable.

**Table 2.** Samples (from `results/tables/datasets.csv`).

| Sample | Observations | Points | Species | Hull area (km²) | Merge (m) | Points used |
| --- | --- | --- | --- | --- | --- | --- |
| Steele Creek Park, one observer | 1,040 | 893 | 757 | 5.4 | 10 | 219 |
| Steele Creek Park area, all taxa | 4,431 | 3,992 | 1,281 | 17.3 | 10 | 1,224 |
| East Tennessee, frogs | 10,314 | 9,803 | 21 | 69,580 | 100 | 1,500 |
| Great Smoky Mountains, birds | 16,344 | 15,460 | 211 | 3,210 | 50 | 1,500 |
| Monteverde, Costa Rica, frogs | 1,309 | 1,137 | 55 | 319 | 25 | 685 |

### 2.7 Simulations

Simulated communities of 400 records of 80 species in a 2 km square (laid out
in UTM metres), each species occurring within a circular range of radius 200
to 1,000 m, were nested by every strategy with and without merging within
10 m (20 replicates each). A record took one of the species whose range
covered its point, or any species at random where none did. In each
community 0, 25% or 50% of the records were placed at one spot with 2 m of
GPS error.

## 3. Results

### 3.1 Strategies split into three groups

Random removal (S1) gave near-vertical curves (Fig. 2) and slopes of 0.7 to
5, often at the bound of the fit's search: it keeps the sample's full extent
until its last points, so area hardly changes while richness falls (Fig. 3).
The outside-in strategies without a greedy rule, random hull vertex (S2),
hull peeling (S3) and concave peeling (S10), agreed closely with one another
on every sample, with *z* between 0.15 and 0.34, the range reported for
classic SARs. Greedy rules shifted *z*: keeping species (S6) gave the lowest
slope on three of five samples (0.05 to 0.09), and keeping area (S4) or
removing the vertex closest to the centre (S9) the highest, both reaching or
nearing the bound on the single-observer sample. At Monteverde, where every
strategy but S1, S4 and S9 gave *z* between 0.14 and 0.17, the greedy rules
made little difference.

### 3.2 Clustered records, and merging them

On the single-observer park sample, merging within 5 m removed 72% of the
points (Table 2's samples ranged from 9% for East Tennessee frogs to 72%).
Without merging, the last few hundred steps of every hull nest measured hulls
under a square metre across, and every hull strategy but S4 and S9 gave
*z* ≈ 0.06 to 0.07.
Merging within 5 to 10 m raised hull peeling's *z* to 0.26 to 0.34, after
which it changed little up to 50 m (Fig. 4). The simulations isolate the
effect (Fig. 5): with no clustering, the outside-in strategies (S2, S3, S5,
S7, S8, S10) gave a median *z* of 0.42 to 0.48; with half the records at one
spot, *z* fell to 0.07 to 0.08 for the raw records, but stayed at 0.26 to
0.44 after merging within 10 m.

### 3.3 Against the random-vertex baseline

Hull peeling fell inside the random-vertex baseline's distribution on every
sample (rank p 0.09 to 0.89 against 99 baseline nests), and concave peeling
on three of five. The greedy strategies mostly fell outside it: keeping area
(S4) and removing the vertex closest to the centre (S9) at p = 0.01, the
smallest possible, on nearly every sample, and keeping species (S6) on four
of five. Greedy rules bias the slope rather than adding noise. A random-vertex
nest with a different seed itself fell anywhere from p = 0.03 to 0.90, a
reminder of how much one random nest varies.

### 3.4 Steps are far from independent

Residuals of one-at-a-time nests were strongly autocorrelated, with lag-1
autocorrelation above 0.94 in all but two cases (S6 at 0.84 and S8 at 0.46,
both on the single-observer sample). Peeling strategies, with far fewer
steps, were usually lower (hull peeling 0.60 to 0.96, concave peeling 0.12
to 0.88), so their R² and AICc overstate the evidence less, though still
substantially.

### 3.5 Hull shapes

Strategies differ in the shapes they leave (Fig. 1, Fig. 6): hull peeling,
S2 and S8 keep the hull compact and centred; S4 and S9 leave elongated hulls
that span the sample, which is why their areas stay large and their slopes
steep.

## 4. Discussion

How a nested SAR is built from points matters as much as the points. Random
removal, the obvious null, is a poor basis for a curve: it measures richness
against an area that barely changes. Shrinking the hull from the outside in
gives curves whose slopes are stable across strategies and samples, and
comparable with classic SARs. Among those, hull peeling is the natural
default: it needs no random seed or greedy rule, removes each convex layer
whole, and its residuals are among the least autocorrelated of the convex
strategies.
Concave peeling suits irregular samples, such as valleys and trail networks,
whose convex hull spans unsampled ground.

The greedy rules are biased by construction. Keeping species (S6) removes
redundant points first, so richness falls late and the slope is usually
shallow;
keeping area (S4) removes interior-adjacent vertices, so area falls late and
the slope is steep. They answer different questions, such as how robust
richness is to losing the least or most valuable ground, rather than giving
an unbiased SAR.

Clustered records are the largest practical hazard. Without merging, the
end of every nest measures GPS error, and the fitted slope collapses toward
zero. Merging within a few metres removes the artefact with little sign of
over-merging up to tens of metres; the distance should be chosen relative to
positional accuracy and reported.

**Limitations.** Areas are convex hull areas even for concave peeling. The
open-data export holds only observations with openly licensed photos, and
obscured observations carry randomised coordinates without a flag. Large
samples were thinned. Only the power model was compared, and the baseline
treats S2 as the reference.

**Next steps.** A known-answer simulation (communities with a set *z*), more
regions and taxa, plot-based data (NEON) for independent-plot SARs, multi-model
fits, searched removal orders (evaluated with `strategies.from_order`) to see how
far the greedy rules are from optimal, and further strategies: kernel-smoothed
boundaries, spatially weighted and multi-scale removal.

## Reproducibility

All strategies are in samplesar (`pip install 'samplesar[opendata]'`) with
sars for fitting. From the repository root:

```bash
uv run python -c "import samplesar, pathlib; samplesar.sources.download_open_data(pathlib.Path.home() / 'data/inaturalist-open-data')"
uv run python papers/hull-strategies/analysis/run.py \
    --opendata ~/data/inaturalist-open-data/observations.parquet \
    --local-sample path/to/steele_creek.csv
```

The script writes every table and figure to `results/`. None of them holds
coordinates of real observations; the only point maps are of a simulated
sample.

## Figures

1. `results/figures/fig1_strategies.png`: each strategy on a simulated sample, at about half its points.
2. `results/figures/fig2_curves.png`: species-area curves of every strategy, per sample.
3. `results/figures/fig3_z_by_strategy.png`: *z* by strategy and sample.
4. `results/figures/fig4_merge_sensitivity.png`: *z* against the merge distance, single-observer park sample.
5. `results/figures/fig5_simulated_clusters.png`: simulated clustering, with and without merging.
6. `results/figures/fig6_compactness.png`: hull compactness as points are removed.

## References (to verify)

- Arrhenius, O. (1921). Species and area. *Journal of Ecology*, 9, 95–99.
- Chazelle, B. (1985). On the convex layers of a planar set. *IEEE Transactions on Information Theory*, 31, 509–517.
- Dengler, J. (2009). Which function describes the species–area relationship best? A review and empirical evaluation. *Journal of Biogeography*, 36, 728–744.
- Matthews, T. J., Triantis, K. A., Whittaker, R. J., & Guilhaumon, F. (2019). sars: an R package for fitting, evaluating and comparing species–area relationship models. *Ecography*, 42, 1446–1455.
- Preston, F. W. (1962). The canonical distribution of commonness and rarity. *Ecology*, 43, 185–215.
- Rosenzweig, M. L. (1995). *Species Diversity in Space and Time*. Cambridge University Press.
- Scheiner, S. M. (2003). Six types of species-area curves. *Global Ecology and Biogeography*, 12, 441–447.
