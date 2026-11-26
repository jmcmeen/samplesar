"""hull_rarity: hull vertices chosen by how rare their species are."""

import numpy as np
import pandas as pd
import pytest

import samplesar
from samplesar._hull import hull_vertices
from samplesar._rarity import rarity_weights

UTM_17N = "EPSG:32617"
ORIGIN = np.array([500_000.0, 4_000_000.0])


def km_frame(rows):
    """Observations (x km, y km, species) laid out in UTM zone 17N."""
    frame = pd.DataFrame(rows, columns=["x", "y", "species"])
    frame["lon"] = frame["x"] * 1000.0 + ORIGIN[0]
    frame["lat"] = frame["y"] * 1000.0 + ORIGIN[1]
    return samplesar.sources.dataframe(frame[["species", "lon", "lat"]], crs=UTM_17N)


@pytest.fixture
def sample():
    rng = np.random.default_rng(3)
    xy = rng.uniform(0, 10, (40, 2))
    rows = []
    for x, y in xy:
        for species in rng.choice(12, size=rng.integers(1, 5), replace=False, p=PROBS):
            rows.extend([(x, y, f"sp{species}")] * int(rng.integers(1, 4)))
    return km_frame(rows)


PROBS = np.arange(12, 0, -1) / np.arange(12, 0, -1).sum()


def removed(nest):
    return [int(np.setdiff1d(a, b)[0]) for a, b in zip(nest.steps, nest.steps[1:], strict=False)]


def point_table(sample):
    """Per point: {species: observations}."""
    lookup = {tuple(p): i for i, p in enumerate(sample.points)}
    table = [{} for _ in lookup]
    for species, lon, lat in sample.data[["species", "lon", "lat"]].itertuples(index=False):
        held = table[lookup[(lon, lat)]]
        held[species] = held.get(species, 0) + 1
    return table


def brute_commonness(sample, table, points, measure):
    out = {}
    for species in {s for held in table for s in held}:
        holders = [p for p in points if species in table[p]]
        if measure == "occupancy":
            out[species] = len(holders)
        elif measure == "abundance":
            out[species] = sum(table[p][species] for p in holders)
        else:
            out[species] = samplesar.hull_area(sample.xy[holders])
    positive = [v for v in out.values() if v > 0]
    floor = min(positive) if positive else 1.0
    return {s: max(v, floor) for s, v in out.items()}


@pytest.mark.parametrize("measure", ["occupancy", "abundance", "range"])
@pytest.mark.parametrize("score", ["sum", "mean", "max"])
@pytest.mark.parametrize("remove", ["common", "rare"])
@pytest.mark.parametrize("update", [False, True])
def test_each_step_removes_the_brute_force_choice(sample, measure, score, remove, update):
    nest = samplesar.strategies.hull_rarity(
        sample, remove=remove, measure=measure, score=score, update=update, min_points=25
    )
    table = point_table(sample)
    combine = {"sum": sum, "mean": lambda v: sum(v) / len(v), "max": max}[score]
    whole = brute_commonness(sample, table, range(sample.n_points), measure)
    for step, point in zip(nest.steps, removed(nest), strict=False):
        common = brute_commonness(sample, table, list(step), measure) if update else whole
        vertices = hull_vertices(sample.xy, step)
        scores = np.array([combine([1.0 / common[s] for s in table[v]]) for v in vertices])
        best = scores.max() if remove == "rare" else scores.min()
        tied = vertices[np.isclose(scores, best, rtol=1e-9, atol=0)]
        assert point in tied


def test_nested_and_reproducible(sample):
    nest = samplesar.strategies.hull_rarity(sample, min_points=5)
    assert len(nest.steps[-1]) == 5
    for before, after in zip(nest.steps, nest.steps[1:], strict=False):
        assert set(after) < set(before) and len(before) - len(after) == 1
    again = samplesar.strategies.run(sample, "hull_rarity", min_points=5)
    assert removed(nest) == removed(again)
    assert nest.strategy == "hull_rarity"


def ring_with(rare_at):
    """A square's corners and center; the species 'rare' at one point only,
    'common' everywhere."""
    spots = {"sw": (0, 0), "se": (10, 0), "ne": (10, 10), "nw": (0, 10), "mid": (5, 5)}
    rows = [(x, y, "common") for x, y in spots.values()]
    rows.append((*spots[rare_at], "rare"))
    # A second ring inside, so corners can go without ending the nest.
    rows += [(x, y, "common") for x, y in [(3, 3), (7, 3), (7, 7), (3, 7)]]
    return km_frame(rows), list(spots)


def test_a_rare_species_on_the_edge_goes_first_or_last():
    sample, names = ring_with("ne")
    ne = names.index("ne")
    rare_first = samplesar.strategies.hull_rarity(sample, remove="rare", min_points=5)
    assert removed(rare_first)[0] == ne
    common_first = samplesar.strategies.hull_rarity(sample, remove="common", min_points=5)
    assert removed(common_first)[:3] == [0, 1, 3]  # the other corners; ties by index
    assert ne in common_first.steps[3]
    # Richness holds while the rare species' corner stays.
    assert list(common_first.to_frame()["species"][:4]) == [2, 2, 2, 2]
    assert list(rare_first.to_frame()["species"][:2]) == [2, 1]


def test_a_rare_species_in_the_center_is_out_of_reach():
    sample, _ = ring_with("mid")
    for remove in ("common", "rare"):
        nest = samplesar.strategies.hull_rarity(sample, remove=remove, min_points=5)
        # Every corner scores alike, so they go by index, and the center stays.
        assert removed(nest) == [0, 1, 2, 3]
        assert nest.to_frame()["species"].eq(2).all()


def test_sum_with_occupancy_is_rarity_weighted_richness():
    # One rich vertex of common species against one vertex with a single rarer one.
    rows = [(0, 0, s) for s in "abcd"] + [(10, 0, "e")]
    rows += [(10, 10, "a"), (0, 10, "a")]
    rows += [(x, y, s) for x, y in [(4, 4), (6, 4), (5, 6)] for s in "abcde"]
    sample = km_frame(rows)
    # a is at 6 points, b, c, d at 4, e at 4: (0, 0) sums 1/6 + 3/4, (10, 0) 1/4.
    by_sum = samplesar.strategies.hull_rarity(sample, remove="rare", score="sum", min_points=6)
    assert removed(by_sum)[0] == 0
    by_mean = samplesar.strategies.hull_rarity(sample, remove="rare", score="mean", min_points=6)
    assert removed(by_mean)[0] == 1


def test_power_zero_is_hull_random(sample):
    weighted = samplesar.strategies.hull_rarity(sample, power=0, seed=7)
    uniform = samplesar.strategies.hull_random(sample, seed=7)
    assert removed(weighted) == removed(uniform)


def test_a_large_power_approaches_the_deterministic_choice():
    sample, names = ring_with("ne")
    ne = names.index("ne")
    for seed in range(5):
        nest = samplesar.strategies.hull_rarity(sample, remove="rare", power=200, seed=seed)
        assert removed(nest)[0] == ne


def test_weights():
    score = np.array([1.0, 2.0, 4.0])
    assert rarity_weights(score, 1, True) == pytest.approx([1 / 7, 2 / 7, 4 / 7])
    assert rarity_weights(score, 1, False) == pytest.approx([4 / 7, 2 / 7, 1 / 7])
    assert rarity_weights(score, 0, True) == pytest.approx([1 / 3] * 3)
    assert np.isfinite(rarity_weights(score, 5000, True)).all()


def test_seeded_runs_repeat(sample):
    a = samplesar.strategies.hull_rarity(sample, power=1.5, seed=11)
    b = samplesar.strategies.hull_rarity(sample, power=1.5, seed=11)
    c = samplesar.strategies.hull_rarity(sample, power=1.5, seed=12)
    assert removed(a) == removed(b) != removed(c)


@pytest.mark.parametrize(
    "options",
    [
        {"remove": "rarest"},
        {"measure": "density"},
        {"score": "median"},
        {"power": -1},
        {"power": float("nan")},
        {"seed": 3},
        {"min_points": 0},
    ],
)
def test_bad_options(sample, options):
    with pytest.raises(ValueError):
        samplesar.strategies.hull_rarity(sample, **options)
