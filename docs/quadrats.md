# Patches as sampling units

A nest shrinks one sample, and [regions](islands.md) are units you choose.
`samplesar.designs.quadrats` is the third design: it lays out square
patches of a set size over the sample and counts the species in each, as a
quadrat survey does.

```python
import samplesar

patches = samplesar.designs.quadrats(sample, [5, 10, 20, 40])   # sides, in km

patches.to_frame()     # region, size_km, area (km²), species, n_points, n_observations, source
patches.fit("power")   # one row per patch
```

The result is an island design whose regions are the patches, so everything
on the [regions page](islands.md) works on it: `to_frame`, `sar_data`,
`fit`, `select`, `drop`, `sample_of`, `plot.islands` and `plot.curve`.

## Laying out the patches

- **A grid** (the default): the patches of a size tile the sample's extent,
  patch against patch, centered on it. A point belongs to exactly one patch
  of each size; one on a shared edge goes to the patch east or north of it.
  Patches are named for their size and cell, `"10 km (2, 0)"` being column
  2 from the west and row 0 from the south.
- **Scattered** (`n=30`): that many patches of each size, placed at random
  within the sample's extent without overlapping one another, named
  `"10 km #1"` onward. `seed` makes the placement repeatable. Patches
  placed one after another at random leave gaps too small for the next, so
  they fill about half of the extent at most.

Each size is laid out on its own, so patches of different sizes overlap.

The patches are squares on the sample's projection. On the default
projection, which is equal-area, patches of a size all have the same area.
If the sample has a `crs` of its own they differ as far as it distorts
areas, and each is reported at its true area.

## Water and other ground that doesn't count

```python
patches = samplesar.designs.quadrats(sample, [5, 10, 20], mask=land, min_fraction=0.5)
```

With `mask`, a patch's `area` is the part of it within the mask, and
`region_area` gives the whole patch. A patch with less than `min_fraction`
of its area in the mask is left out, so a patch that is mostly sea isn't
counted as a small one.

## Two kinds of curve

**One row per patch.** `sar_data` and `fit` treat every patch with
observations as a unit, as islands are. Patches of one size all have the
same area, so this needs several sizes; `fit` says so if there is only
one.

**Accumulation.** Patches of a single size give a curve of their own: the
species in one patch, in two, in three and so on, against their combined
area.

```python
patches.accumulate(10)                    # any patches, wherever they lie
patches.accumulate(10, order="nearest")   # only neighbors
```

- `order="any"` averages over every combination of k patches. It is
  computed exactly, from the number of patches each species is in.
- `order="nearest"` starts from each patch in turn and adds the others
  nearest first, then averages the curves. Nearby patches usually share
  more species than distant ones, and then this curve lies below the
  other; the gap between them shows how much the species change across
  the sample.

Both return `size_km`, `patches`, `area` and `species`, one row per number
of patches. They end at the same point: every patch, and every species the
patches hold.

These curves aren't comparable point for point with a nest's or with each
other. A nest's steps are one growing area; an accumulation's are the sum
of separate patches, which reach more of the sample at the same area.

## Sampling effort

Patches of one size have the same area but not the same effort. The table
gives each patch's points and observations, and `min_observations` drops
thinly sampled patches from `sar_data`, `fit` and `accumulate` (by default,
those with none). To even effort out first, thin the sample:

```python
even = sample.thin(km=1, seed=0)          # at most one point per 1 km cell
patches = samplesar.designs.quadrats(even, [5, 10, 20, 40])
```
