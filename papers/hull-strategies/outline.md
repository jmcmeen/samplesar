# Nested species-area relationships from convex hull sampling strategies

Working title. Target: Methods in Ecology and Evolution, or Ecography.

## The idea

A species-area relationship is usually built from separate islands or plots.
From one set of georeferenced observations, a nested SAR can be built
instead, by shrinking the sample step by step and recording the area and
richness at each step. How the sample shrinks is a choice, and it shapes the
curve. Removing points at random keeps the sample's full extent until its
last few points; removing them from the convex hull inwards shrinks the area
from the first step. This paper defines a family of hull-based strategies,
says what each one does to the area and richness, and measures how much the
choice changes the fitted relationship.

## Questions

1. How much do the fitted parameters ($z$, $c$) and the shape of the curve
   depend on the strategy?
2. Which strategies give curves comparable to SARs from independent plots of
   the same region?
3. How robust is each strategy to the way observations are collected:
   clustered effort (trailheads, feeders), uneven density, outlying records?
4. Is there a strategy (or a small set) to recommend as the default, and
   when?

## Methods

- **Strategies.** The nine strategies, S1 to S9, defined formally in
  [strategies.md](strategies.md): random removal (the baseline), a random
  hull vertex, hull peeling, and greedy choices of the hull vertex by area,
  species and distance from the centre.
- **Area.** Convex hull areas on an equal-area projection (km²), so
  parameters compare across latitudes.
- **Fitting.** The power model and multi-model averaging with sars, on each
  nest's species-area data.
- **Inference.** Steps share most of their points, so single-nest R² and
  AICc aren't independent evidence. Compare strategies on the same sample,
  and build intervals by resampling: bootstrapped samples, and many seeds for
  the random strategies.

## Data

1. **Simulated communities with a known answer.** Point patterns
   (homogeneous, clustered, with a density gradient) carrying species placed
   so that the true SAR is known, to see which strategies recover it, and how
   observation clusters bias each.
2. **A local iNaturalist sample.** Steele Creek Park, Bristol, Tennessee:
   about 1,000 observations, 900 points, 760 species.
3. **Regional iNaturalist samples**, from a local copy of iNaturalist's
   open-data export (research grade), across several regions and taxa.
4. **Plot-based data** (NEON, then FIA as a stretch) for the independent-plot
   SARs to compare against (question 2).

## Analyses and figures

1. A schematic of the strategies on one small sample: the hull at a few
   steps for each.
2. Species-area curves of all nine strategies on the same sample (log axes).
3. Fitted $z$ and $c$ by strategy, with resampling intervals, per dataset.
4. Recovery of the known SAR in the simulations, by strategy and point
   pattern.
5. Sensitivity to clustered effort: the simulations with added clusters, and
   the real samples with and without merging co-located records.
6. Spatial diagnostics of each nest: hull regularity and coverage, and
   residual autocorrelation of the fitted curves.
7. Comparison with independent-plot SARs of the same regions.

## What the first run showed

On the Steele Creek Park sample (all nine strategies):

- **Strategies split into two groups.** Random removal (S1) and removing the
  vertex closest to the centre (S9) keep nearly the full extent until the
  end, giving near-vertical curves that drive the power model's $z$ to its
  bound. The other hull strategies give $z \approx 0.07$.
- **Co-located records dominate the end of every hull nest.** Many records
  share a spot to within GPS jitter, so the last few hundred steps measure
  centimetre-scale hulls. Merging points within a distance, or a minimum
  area, is needed before fitting (samplesar #46), and is itself a methods
  result.

## Open problems

- The rule for merging co-located records, and its distance.
- The fitness, or comparison, criterion that respects pseudo-replication.
- Fits at parameter bounds (samplesar #48).
- Whether greedy choices (S4 to S7) are close to the best orders, which can
  be checked by evaluating searched orders with `strategies.from_order`.
- Extensions: α-hulls and concave hulls, kernel-smoothed boundaries,
  spatially weighted and multi-scale removal.

## Software and reproducibility

All strategies are in samplesar (with sars for fitting); the analysis
scripts and figures will live in this folder, reading data from outside the
repository.
