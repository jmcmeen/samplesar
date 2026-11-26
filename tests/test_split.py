"""Nests of several parts: partition, split, and areas of hulls together."""

import numpy as np
import pandas as pd
import pytest
import shapely

import samplesar

UTM_17N = "EPSG:32617"
ORIGIN = np.array([500_000.0, 4_000_000.0])


def km_sample(xy_km, species=None):
    xy = np.asarray(xy_km, dtype=float) * 1000.0 + ORIGIN
    if species is None:
        species = [f"sp{i % 4}" for i in range(len(xy))]
    frame = pd.DataFrame({"species": species, "lon": xy[:, 0], "lat": xy[:, 1]})
    return samplesar.sources.dataframe(frame, crs=UTM_17N)


def clusters(n=40, centers=((0, 0), (100, 0), (0, 100)), spread=5.0, seed=0):
    rng = np.random.default_rng(seed)
    xy = np.vstack([rng.uniform(-spread, spread, (n, 2)) + c for c in centers])
    species = [f"c{i // n}s{i % 6}" for i in range(len(xy))]
    return km_sample(xy, species=species), np.repeat(np.arange(len(centers)), n)


def removed(nest):
    return [np.setdiff1d(a, b) for a, b in zip(nest.steps, nest.steps[1:], strict=False)]


def test_kmeans_finds_separate_clusters():
    sample, truth = clusters()
    labels = samplesar.strategies.partition(sample, parts=3, seed=1)
    # Numbered by first appearance, as the clusters were laid out.
    assert np.array_equal(labels, truth)
    assert np.array_equal(labels, samplesar.strategies.partition(sample, parts=3, seed=1))
    assert np.array_equal(samplesar.strategies.partition(sample, parts=1), np.zeros(120, int))


def test_grid_parts():
    xy = [(x + 0.5, y + 0.5) for y in range(4) for x in range(6)]
    sample = km_sample(xy)
    labels = samplesar.strategies.partition(sample, parts=(3, 2), by="grid")
    cells = np.array([(int(x // 2), int(y // 2)) for x, y in xy])
    assert len(np.unique(labels)) == 6
    for part in range(6):
        assert len({tuple(c) for c in cells[labels == part]}) == 1
    # A count gives the most nearly square layout: 4 -> 2 by 2.
    four = samplesar.strategies.partition(sample, parts=4, by="grid")
    assert np.bincount(four).tolist() == [6, 6, 6, 6]
    # Empty cells make no part.
    sparse = km_sample([(0, 0), (0.1, 0.1), (9, 9), (9.1, 9.2)])
    assert samplesar.strategies.partition(sparse, parts=(3, 3), by="grid").tolist() == [0, 0, 1, 1]


def test_given_labels_seeds_and_polygons():
    sample, truth = clusters()
    names = np.array(["west", "east", "north"])[truth]
    assert np.array_equal(samplesar.strategies.partition(sample, by=truth + 7), truth)
    assert np.array_equal(samplesar.strategies.partition(sample, by=names), truth)
    seeds = np.array([sample.points[truth == k].mean(axis=0) for k in range(3)])
    assert np.array_equal(samplesar.strategies.partition(sample, by=seeds), truth)
    boxes = [
        shapely.box(*sample.points[truth == k].min(axis=0), *sample.points[truth == k].max(axis=0))
        for k in (2, 0, 1)
    ]
    assert np.array_equal(samplesar.strategies.partition(sample, by=boxes), truth)
    with pytest.raises(ValueError, match="80 of the sample's 120 points lie in none"):
        samplesar.strategies.partition(sample, by=boxes[:1])


def test_polygons_from_a_geoseries():
    geopandas = pytest.importorskip("geopandas")
    sample, truth = clusters()
    boxes = [
        shapely.box(*sample.points[truth == k].min(axis=0), *sample.points[truth == k].max(axis=0))
        for k in range(3)
    ]
    series = geopandas.GeoSeries(boxes, crs=4326).to_crs(3857)
    assert np.array_equal(samplesar.strategies.partition(sample, by=series), truth)


@pytest.mark.parametrize(
    "options",
    [
        {"parts": 0},
        {"parts": 500},
        {"parts": 2.5},
        {"by": "voronoi"},
        {"by": np.zeros(7, dtype=int)},
        {"by": np.array([[500.0, 0.0]])},
        {"by": 3.5},
        {"parts": (0, 2), "by": "grid"},
    ],
)
def test_partition_bad_options(options):
    sample, _ = clusters()
    with pytest.raises(ValueError):
        samplesar.strategies.partition(sample, **options)


@pytest.mark.parametrize("pace", ["even", "turn"])
@pytest.mark.parametrize("strategy", ["hull_random", "hull_peel", "radial", "hull_area"])
def test_each_part_is_nested_by_the_inner_strategy(strategy, pace):
    sample, truth = clusters(n=25)
    nest = samplesar.strategies.split(sample, strategy, by=truth, pace=pace, min_points=4, seed=3)
    assert np.array_equal(nest.parts, truth)
    assert nest.strategy == "split" and nest.options["strategy"] == strategy
    for before, after in zip(nest.steps, nest.steps[1:], strict=False):
        assert set(after) < set(before)
        assert len(np.unique(truth[np.setdiff1d(before, after)])) == 1  # one part per step
    seeds = np.random.SeedSequence(3).generate_state(3)
    for part in range(3):
        members = np.flatnonzero(truth == part)
        options = {"seed": int(seeds[part])} if strategy == "hull_random" else {}
        alone = samplesar.strategies.run(sample.subset(members), strategy, min_points=4, **options)
        own = [step[truth[step] == part] for step in nest.steps]
        distinct = [own[0]] + [b for a, b in zip(own, own[1:], strict=False) if len(a) != len(b)]
        assert len(distinct) == alone.n_steps
        for got, want in zip(distinct, alone.steps, strict=True):
            assert np.array_equal(got, members[want])


def test_even_pace_finishes_together_and_turn_takes_turns():
    # Parts of 12, 6 and 6 points: radial takes 9, 3 and 3 steps.
    rng = np.random.default_rng(1)
    xy = np.vstack(
        [
            rng.uniform(0, 5, (12, 2)),
            rng.uniform(0, 5, (6, 2)) + (100, 0),
            rng.uniform(0, 5, (6, 2)) + (0, 100),
        ]
    )
    truth = np.repeat([0, 1, 2], [12, 6, 6])
    sample = km_sample(xy)
    even = samplesar.strategies.split(sample, "radial", by=truth, pace="even")
    turn = samplesar.strategies.split(sample, "radial", by=truth, pace="turn")
    assert even.n_steps == turn.n_steps == 1 + 9 + 3 + 3
    assert [int(truth[r[0]]) for r in removed(turn)] == [0, 1, 2] * 3 + [0] * 6
    assert [int(truth[r[0]]) for r in removed(even)] == [
        0,
        0,
        0,
        1,
        2,
        0,
        0,
        0,
        1,
        2,
        0,
        0,
        0,
        1,
        2,
    ]
    assert sorted(even.steps[-1]) == sorted(turn.steps[-1])


def test_area_is_the_parts_hulls_together_not_one_hull():
    sample, truth = clusters()
    nest = samplesar.strategies.split(sample, "hull_peel", by=truth)
    table = nest.to_frame()
    assert table["n_parts"].eq(3).all()
    for step, row in zip(nest.steps, table.itertuples(), strict=True):
        together = sum(sample.hull_area_of(step[truth[step] == k]) for k in range(3))
        assert row.area == pytest.approx(together, rel=1e-9)
        assert nest.areas_at(row.step)[0] == pytest.approx(together, rel=1e-9)
    one_hull = samplesar.strategies.hull_peel(sample).to_frame()["area"].iloc[0]
    # Three 10 km clusters 100 km apart: one hull spans the empty ground between.
    assert table["area"].iloc[0] < 0.1 * one_hull
    assert nest.sar_data()["area"].iloc[0] == table["area"].iloc[0]


def test_overlapping_hulls_count_their_ground_once():
    # Two parts whose hulls are the same square.
    square = [(0, 0), (10, 0), (10, 10), (0, 10), (5, 5)]
    sample = km_sample(square + [(x + 0.5, y + 0.5) for x, y in square[:4]] + [(5.5, 5.5)])
    nest = samplesar.Nest(sample, (np.arange(10),), parts=np.repeat([0, 1], 5))
    # The squares are offset by half a km: 10.5 by 10.5 less two corner notches.
    assert nest.to_frame()["area"].iloc[0] == pytest.approx(100 + 100 - 9.5 * 9.5, rel=2e-3)


def test_areas_within_a_mask():
    sample, truth = clusters()
    # A mask covering the western half of the first cluster only.
    lon, lat = sample.points[truth == 0].T
    mask = shapely.box(lon.min() - 0.01, lat.min() - 0.01, np.median(lon), lat.max() + 0.01)
    masked = sample.with_mask(mask)
    nest = samplesar.strategies.split(masked, "hull_random", by=truth, seed=1)
    table = nest.to_frame()
    assert (table["area"] < table["hull_area"]).all()
    for i in (0, len(table) // 2):
        step = nest.steps[i]
        inside = masked.hull_area_of(step[truth[step] == 0])
        assert table["area"].iloc[i] == pytest.approx(inside, rel=1e-6)


def test_small_parts_stay_whole_and_empty_work_is_an_error():
    sample, truth = clusters(n=10)
    labels = truth.copy()
    labels[:3] = 9  # a part of three points
    nest = samplesar.strategies.split(sample, "radial", by=labels, min_points=3)
    assert {0, 1, 2} <= set(nest.steps[-1])
    with pytest.raises(ValueError, match="nothing to nest"):
        samplesar.strategies.split(sample, "radial", by=truth, min_points=10)
    # hull_peel can't peel a part of points all on its hull: such parts stay whole.
    ring = [(np.cos(a), np.sin(a)) for a in np.linspace(0, 2 * np.pi, 9)[:-1]]
    filled, _ = clusters(n=30, centers=((50, 50),))
    both = km_sample(np.vstack([ring, filled.xy - filled.xy.mean(axis=0) + 50]))
    labels = np.repeat([0, 1], [8, 30])
    peeled = samplesar.strategies.split(both, "hull_peel", by=labels)
    assert set(range(8)) <= set(peeled.steps[-1])
    with pytest.raises(ValueError, match="No layer can be peeled"):
        samplesar.strategies.split(
            km_sample(ring + [(c + 9, s) for c, s in ring]), "hull_peel", parts=2
        )


def test_inner_options_and_errors():
    sample, truth = clusters(n=15)
    nest = samplesar.strategies.split(sample, "hull_area", by=truth, keep="smallest")
    assert dict(nest.options) == {"strategy": "hull_area", "by": truth, "keep": "smallest"} or (
        nest.options["keep"] == "smallest"
    )
    with pytest.raises(ValueError, match="No strategy"):
        samplesar.strategies.split(sample, "peel")
    with pytest.raises(ValueError, match="itself"):
        samplesar.strategies.split(sample, "split")
    with pytest.raises(ValueError, match="pace"):
        samplesar.strategies.split(sample, pace="fast")
    with pytest.raises(TypeError):
        samplesar.strategies.split(sample, "hull_peel", by=truth, keep="largest")
    with pytest.raises(ValueError, match="keep"):
        samplesar.strategies.split(sample, "hull_area", by=truth, keep="biggest")


def test_seeded_splits_repeat():
    sample, _ = clusters()
    a = samplesar.strategies.split(sample, "hull_random", parts=3, seed=5)
    b = samplesar.strategies.split(sample, "hull_random", parts=3, seed=5)
    c = samplesar.strategies.split(sample, "hull_random", parts=3, seed=6)
    same = [np.array_equal(x, y) for x, y in zip(a.steps, b.steps, strict=True)]
    assert all(same)
    assert not all(np.array_equal(x, y) for x, y in zip(a.steps, c.steps, strict=True))


def test_saving_keeps_the_parts(tmp_path):
    sample, truth = clusters(n=12)
    nest = samplesar.strategies.split(sample, "radial", parts=3, seed=2)
    path = tmp_path / "nest.json"
    nest.to_json(path)
    back = samplesar.Nest.from_json(path)
    assert np.array_equal(back.parts, nest.parts)
    assert back.to_frame()["area"].tolist() == pytest.approx(nest.to_frame()["area"].tolist())
    saved = nest.to_dict()
    assert saved["version"] == 5
    saved["parts"] = saved["parts"][:-1]
    with pytest.raises(ValueError, match="parts"):
        samplesar.Nest.from_dict(saved)
    assert samplesar.strategies.radial(sample).to_dict()["version"] == 3


def test_parts_must_fit_the_sample():
    sample, _ = clusters(n=5)
    with pytest.raises(ValueError, match="parts"):
        samplesar.Nest(sample, (np.arange(15),), parts=np.zeros(4, dtype=int))
    with pytest.raises(ValueError, match="parts"):
        samplesar.Nest(sample, (np.arange(15),), parts=np.zeros(15))


def test_diagnostics_and_plots_follow_the_parts():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    sample, truth = clusters(n=12)
    nest = samplesar.strategies.split(sample, "radial", by=truth)
    table = samplesar.evaluate.diagnostics(nest)
    assert table["area"].tolist() == pytest.approx(nest.to_frame()["area"].tolist())
    # Three separate hulls: far less compact than any one of them.
    assert table["compactness"].iloc[0] < 0.4
    ax = samplesar.plot.nest(nest)
    outlines = [line for line in ax.lines if line.get_color() == "C1"]
    assert len(outlines) == 3
    assert samplesar.plot.curve(nest) is not None
    fit = nest.fit("power")
    assert np.isfinite(fit.params["z"])
