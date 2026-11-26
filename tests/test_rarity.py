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


PROBS = np.arange(12, 0, -1) / np.arange(12, 0, -1).sum()


@pytest.fixture
def sample():
    rng = np.random.default_rng(3)
    xy = rng.uniform(0, 10, (40, 2))
    rows = []
    for x, y in xy:
        for species in rng.choice(12, size=rng.integers(1, 5), replace=False, p=PROBS):
            rows.extend([(x, y, f"sp{species}")] * int(rng.integers(1, 4)))
    return km_frame(rows)


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
    return out


def floor_of(sample, table, measure):
    """Counts never fall below 1 while a species is present; a range of no
    area counts as the smallest real range in the whole sample."""
    if measure != "range":
        return 1.0
    whole = brute_commonness(sample, table, range(sample.n_points), measure)
    real = [v for v in whole.values() if v > 1e-9 * samplesar.hull_area(sample.xy)]
    return min(real) if real else 1.0


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
    floor = floor_of(sample, table, measure)
    for step, point in zip(nest.steps, removed(nest), strict=False):
        common = brute_commonness(sample, table, list(step), measure) if update else whole
        vertices = hull_vertices(sample.xy, step)
        scores = np.array(
            [combine([1.0 / max(common[s], floor) for s in table[v]]) for v in vertices]
        )
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
    assert removed(a) == removed(b)
    others = {
        tuple(removed(samplesar.strategies.hull_rarity(sample, power=1.5, seed=s)))
        for s in range(12, 16)
    }
    assert others != {tuple(removed(a))}
    # Without power the choice is deterministic, whatever the seed.
    plain = samplesar.strategies.hull_rarity(sample)
    assert removed(plain) == removed(samplesar.strategies.hull_rarity(sample, seed=3))


def test_updating_changes_the_nest(sample):
    for measure in ("occupancy", "abundance", "range"):
        once = samplesar.strategies.hull_rarity(sample, remove="rare", measure=measure)
        updated = samplesar.strategies.hull_rarity(
            sample, remove="rare", measure=measure, update=True
        )
        assert removed(once) != removed(updated)


def test_a_species_losing_its_range_stays_rare():
    # "small" has a 1 km² triangle outside the east edge; every other range is large.
    big = [(x, y, s) for x, y in [(0, 0), (20, 0), (20, 20), (0, 20), (10, 10)] for s in "ab"]
    small = [(21, 9, "small"), (22, 10, "small"), (21, 11, "small")]
    sample = km_frame(big + small + [(8, 8, "a"), (12, 8, "b"), (10, 12, "a")])
    nest = samplesar.strategies.hull_rarity(
        sample, remove="rare", measure="range", update=True, min_points=6
    )
    # Its three points go first: down to two points its range is nothing,
    # which counts as its old 1 km², not as the next smallest range.
    assert sorted(removed(nest)[:3]) == [5, 6, 7]


def test_a_species_in_a_line_has_no_range():
    # "row" lies on a straight grid row: collinear but for rounding.
    rows = [(x, 0, "row") for x in range(-2, 8)]
    rows += [(x, y, "wide") for x, y in [(0, 5), (5, 5), (0, -5), (5, -5), (2, 2), (3, -2)]]
    rows += [(1, 3, "mid"), (4, 3, "mid"), (2, -3, "mid")]
    sample = km_frame(rows)
    nest = samplesar.strategies.hull_rarity(sample, remove="rare", measure="range", min_points=8)
    # The ends of the row are hull vertices with the rarest species (no
    # range: it counts as "mid"'s, the smallest real one), so the row goes
    # first, from its ends.
    assert removed(nest)[0] in (0, 9)
    assert all(point < 10 for point in removed(nest)[:4])


def test_ties_go_to_the_vertex_farthest_from_the_center():
    # Every point holds the same species; one corner sticks out.
    rows = [(x, y, "a") for x, y in [(0, 0), (10, 0), (30, 30), (0, 10), (4, 4), (5, 6), (6, 4)]]
    nest = samplesar.strategies.hull_rarity(km_frame(rows), min_points=6)
    assert removed(nest) == [2]


@pytest.mark.parametrize(
    "options",
    [
        {"remove": "rarest"},
        {"measure": "density"},
        {"score": "median"},
        {"power": -1},
        {"power": float("nan")},
        {"power": True},
        {"power": "2"},
        {"min_points": 0},
    ],
)
def test_bad_options(sample, options):
    with pytest.raises(ValueError):
        samplesar.strategies.hull_rarity(sample, **options)
