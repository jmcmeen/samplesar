"""Equal-area patches as sampling units: designs.quadrats."""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
import shapely
from shapely.geometry import box

import samplesar
from samplesar._islands import geodesic_area


def scattered(n=400, seed=0, species=30):
    """Points scattered over about 110 by 110 km at the equator."""
    rng = np.random.default_rng(seed)
    return samplesar.sources.dataframe(
        pd.DataFrame(
            {
                "species": [f"s{i}" for i in rng.integers(0, species, n)],
                "lon": rng.uniform(0, 1, n),
                "lat": rng.uniform(0, 1, n),
            }
        )
    )


def corners():
    """Four clusters at the corners of a square about 100 km on a side,
    holding 1, 2, 3 and 4 species."""
    rows = []
    for (lon, lat), count in zip(
        [(0, 0), (0.9, 0), (0, 0.9), (0.9, 0.9)], (1, 2, 3, 4), strict=True
    ):
        for i in range(8):
            rows.append(
                {
                    "species": f"{lon}-{lat}-{i % count}",
                    "lon": lon + 0.01 * (i % 3),
                    "lat": lat + 0.01 * (i // 3),
                }
            )
    return samplesar.sources.dataframe(pd.DataFrame(rows))


def test_grid_with_a_known_answer():
    # Two columns and two rows of 60 km patches cover the 100 km square,
    # one cluster in each.
    made = samplesar.designs.quadrats(corners(), 60)
    table = made.to_frame()
    assert table["region"].tolist() == [
        "60 km (0, 0)",
        "60 km (0, 1)",
        "60 km (1, 0)",
        "60 km (1, 1)",
    ]
    assert table["species"].tolist() == [1, 3, 2, 4]
    assert table["n_points"].tolist() == [8, 8, 8, 8]
    assert table["size_km"].tolist() == [60.0] * 4
    assert table["area"].tolist() == pytest.approx([3600.0] * 4, rel=1e-5)
    assert made.label == "quadrats (4 patches, 1 size)"
    assert made.layout == "grid"
    assert made.sizes == [60.0]
    assert isinstance(made, samplesar.Quadrats)
    assert isinstance(made, samplesar.Islands)


def test_every_point_is_in_one_patch_of_each_size():
    sample = scattered()
    made = samplesar.designs.quadrats(sample, [7, 15, 40])
    for side in made.sizes:
        names = [name for name in made.regions if made.squares[name][0] == side]
        members = np.concatenate([made._members[name] for name in names])
        assert sorted(members.tolist()) == list(range(sample.n_points))


def test_membership_matches_brute_force():
    sample = scattered(150, seed=2)
    xy = sample.xy
    for made in (
        samplesar.designs.quadrats(sample, [9, 25]),
        samplesar.designs.quadrats(sample, [9, 25], n=6, seed=4),
    ):
        for name in made.regions:
            side, x0, y0 = made.squares[name]
            expected = [
                i for i, (x, y) in enumerate(xy) if x0 <= x < x0 + side and y0 <= y < y0 + side
            ]
            assert made._members[name].tolist() == expected
            # The patch's outline holds the same points, give or take its edge.
            outline = made.regions[name]
            near = outline.buffer(1e-6)
            assert {
                i for i, p in enumerate(sample.points) if near.contains(shapely.Point(p))
            } >= set(expected)
            outside = [i for i in range(sample.n_points) if i not in set(expected)]
            assert not any(
                outline.buffer(-1e-6).contains(shapely.Point(sample.points[i])) for i in outside
            )
            inner = [
                i
                for i in expected
                if outline.buffer(-1e-6).contains(shapely.Point(sample.points[i]))
            ]
            assert len(expected) - len(inner) <= 1


def test_grid_is_centered_and_covers_the_sample():
    sample = scattered()
    made = samplesar.designs.quadrats(sample, 25)
    xy = sample.xy
    x0, y0 = made.origins[25.0]
    cols = 1 + max(col for col, _ in made.cells.values())
    rows = 1 + max(row for _, row in made.cells.values())
    assert x0 < xy[:, 0].min() and x0 + cols * 25 > xy[:, 0].max()
    assert y0 < xy[:, 1].min() and y0 + rows * 25 > xy[:, 1].max()
    assert xy[:, 0].min() - x0 == pytest.approx(x0 + cols * 25 - xy[:, 0].max())
    assert xy[:, 1].min() - y0 == pytest.approx(y0 + rows * 25 - xy[:, 1].max())


def test_a_single_point_gets_one_patch():
    sample = samplesar.sources.dataframe(
        pd.DataFrame({"species": ["a"], "lon": [5.0], "lat": [5.0]})
    )
    table = samplesar.designs.quadrats(sample, 10).to_frame()
    assert table["region"].tolist() == ["10 km (0, 0)"]
    assert table["species"].tolist() == [1]


def test_random_patches_are_reproducible_and_apart():
    sample = scattered()
    made = samplesar.designs.quadrats(sample, [5, 10], n=12, seed=7)
    again = samplesar.designs.quadrats(sample, [5, 10], n=12, seed=7)
    other = samplesar.designs.quadrats(sample, [5, 10], n=12, seed=8)
    assert made.squares == again.squares
    assert made.to_frame().equals(again.to_frame())
    assert made.squares != other.squares
    assert made.layout == "random"
    assert list(made.regions)[:2] == ["5 km #1", "5 km #2"]
    assert len(made.regions) == 24
    xy = sample.xy
    low, high = xy.min(axis=0), xy.max(axis=0)
    for side in (5.0, 10.0):
        placed = [(x, y) for s, x, y in made.squares.values() if s == side]
        assert len(placed) == 12
        for i, (x, y) in enumerate(placed):
            assert low[0] <= x and x + side <= high[0]
            assert low[1] <= y and y + side <= high[1]
            for u, v in placed[:i]:
                assert abs(x - u) >= side or abs(y - v) >= side


def test_patch_areas_are_true_areas():
    sample = scattered()
    made = samplesar.designs.quadrats(sample, 20, n=5, seed=1)
    table = made.to_frame()
    assert table["area"].tolist() == pytest.approx([400.0] * 5, rel=1e-5)
    for name, outline in made.regions.items():
        assert geodesic_area(outline) == pytest.approx(table.set_index("region")["area"][name])
    # Under a CRS of the sample's own the patches are still measured truly.
    utm = samplesar.sources.dataframe(
        pd.DataFrame(
            {"species": ["a", "b", "c"], "x": [500_000, 560_000, 530_000], "y": [0, 40_000, 20_000]}
        ),
        lon_col="x",
        lat_col="y",
        crs=32631,
    )
    areas = samplesar.designs.quadrats(utm, 30).to_frame()["area"]
    assert areas.tolist() == pytest.approx([900.0] * len(areas), rel=2e-3)


def test_mask_measures_and_drops_patches():
    sample = corners()
    plain = samplesar.designs.quadrats(sample, 60)
    west_edge = plain.regions["60 km (1, 0)"].bounds[0]
    # Land covers the western column and the western quarter of the eastern one.
    land = box(-5, -5, west_edge + 0.25 * 60 / 111.3, 5)
    made = samplesar.designs.quadrats(sample, 60, mask=land)
    assert list(made.regions) == ["60 km (0, 0)", "60 km (0, 1)"]
    kept = samplesar.designs.quadrats(sample, 60, mask=land, min_fraction=0.1).to_frame()
    assert kept["region"].tolist() == [f"60 km ({c}, {r})" for c in (0, 1) for r in (0, 1)]
    assert kept["region_area"].tolist() == pytest.approx([3600.0] * 4, rel=1e-5)
    assert kept["area"].tolist() == pytest.approx([3600, 3600, 900, 900], rel=0.02)
    # A patch wholly outside the mask is left out even with min_fraction=0.
    none = samplesar.designs.quadrats(
        sample, 60, mask=box(-5, -5, west_edge - 0.01, 5), min_fraction=0
    )
    assert list(none.regions) == ["60 km (0, 0)", "60 km (0, 1)"]
    with pytest.raises(ValueError, match="within the mask"):
        samplesar.designs.quadrats(sample, 60, mask=box(50, 50, 51, 51))
    with pytest.raises(ValueError, match="fit within the mask"):
        samplesar.designs.quadrats(sample, 20, n=3, seed=0, mask=box(50, 50, 51, 51))


def test_fit_needs_more_than_one_size():
    sample = scattered()
    with pytest.raises(ValueError, match="one size"):
        samplesar.designs.quadrats(sample, 20).fit()
    made = samplesar.designs.quadrats(sample, [10, 20, 40])
    data = made.sar_data()
    assert data.columns.tolist() == ["area", "species"]
    assert len(data) == int((made.to_frame()["n_observations"] > 0).sum())
    assert 0 < made.fit("power").params["z"] < 1
    with pytest.raises(ValueError, match="one size"):
        made.select([name for name in made.regions if name.startswith("10 km")]).fit()


def test_select_drop_and_sample_of_keep_the_patches():
    sample = scattered()
    made = samplesar.designs.quadrats(sample, [20, 40])
    table = made.to_frame().set_index("region")
    some = made.select(["40 km (1, 1)", "20 km (0, 0)"])
    assert isinstance(some, samplesar.Quadrats)
    assert some.to_frame()["region"].tolist() == ["40 km (1, 1)", "20 km (0, 0)"]
    assert (
        some.to_frame()["species"].tolist()
        == table["species"][["40 km (1, 1)", "20 km (0, 0)"]].tolist()
    )
    less = made.drop(["20 km (0, 0)"])
    assert isinstance(less, samplesar.Quadrats)
    assert len(less.regions) == len(made.regions) - 1
    inside = made.sample_of("40 km (1, 1)")
    assert inside.n_points == table["n_points"]["40 km (1, 1)"]
    with pytest.raises(ValueError, match="only its own patches"):
        made.add({"more": box(0, 0, 1, 1)})


def test_plots():
    made = samplesar.designs.quadrats(scattered(), [20, 40])
    assert "with observations" in samplesar.plot.islands(made, labels=False).get_title()
    assert samplesar.plot.curve(made).get_xlabel() == "Area (km²)"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"size_km": 0}, "positive"),
        ({"size_km": [10, -1]}, "positive"),
        ({"size_km": "10"}, "number of km, got '10'"),
        ({"size_km": {10: 1}}, "got a mapping"),
        ({"size_km": [30.0000001, 30.0000002]}, "first six digits"),
        ({"size_km": 1e-300}, "cells; the most"),
        ({"size_km": 10, "n": 2, "seed": "a"}, "seed must be"),
        ({"size_km": True}, "number of km"),
        ({"size_km": []}, "No patch size"),
        ({"size_km": [10, 10.0]}, "must differ"),
        ({"size_km": float("nan")}, "positive"),
        ({"size_km": 10, "n": 0}, "positive whole number"),
        ({"size_km": 10, "n": 2.5}, "positive whole number"),
        ({"size_km": 10, "n": 10**6}, "the most patches"),
        ({"size_km": 0.1}, "cells; the most"),
        ({"size_km": 500, "n": 2}, "larger than the sample's extent"),
        ({"size_km": 50, "n": 9, "seed": 0}, "fit without overlapping"),
        ({"size_km": 10, "min_fraction": 1.5}, "from 0 to 1"),
        ({"size_km": 10, "min_fraction": "half"}, "from 0 to 1"),
        ({"size_km": 10, "mask": shapely.Point(0, 0)}, "polygon"),
    ],
)
def test_bad_arguments(kwargs, message):
    with pytest.raises(ValueError, match=message):
        samplesar.designs.quadrats(scattered(), **kwargs)


def test_needs_a_sample():
    with pytest.raises(ValueError, match="needs a Sample"):
        samplesar.designs.quadrats(None, 10)


def test_sizes_as_arrays_and_scalars():
    sample = scattered()
    for sizes in (np.array(30.0), np.float64(30), np.array([30.0]), pd.Series([30.0]), 30):
        assert samplesar.designs.quadrats(sample, sizes).sizes == [30.0]


def test_one_size_draws_without_a_fit():
    sample = scattered()
    for made in (
        samplesar.designs.quadrats(sample, 20),
        samplesar.designs.quadrats(sample, 10, n=5, seed=1),
    ):
        ax = samplesar.plot.curve(made)
        assert len(ax.lines) == 0
        assert "with observations" in samplesar.plot.islands(made, labels=False).get_title()
    # Two sizes, but observations in patches of one only: still nothing to fit.
    lone = samplesar.sources.dataframe(
        pd.DataFrame({"species": list("abc"), "lon": [0, 1, 0], "lat": [0, 0, 1]})
    )
    made = samplesar.designs.quadrats(lone, [5, 80])
    assert made._sizes_with_data() == 2
    emptied = made.select([n for n in made.regions if n.startswith("5 km") or n == "80 km (1, 1)"])
    assert len(emptied.sizes) == 2
    assert emptied.to_frame().set_index("region")["n_points"]["80 km (1, 1)"] == 0
    with pytest.raises(ValueError, match="one size"):
        emptied.fit()
    assert len(samplesar.plot.curve(emptied).lines) == 0


def test_patches_across_the_antimeridian_or_a_pole_are_refused():
    rng = np.random.default_rng(0)
    east = samplesar.sources.dataframe(
        pd.DataFrame(
            {
                "species": ["a"] * 20,
                "lon": [*rng.uniform(178, 179.9, 19), 179.95],
                "lat": rng.uniform(0, 1, 20),
            }
        )
    )
    with pytest.raises(ValueError, match="antimeridian or covers a pole"):
        samplesar.designs.quadrats(east, 50)
    polar = samplesar.sources.dataframe(
        pd.DataFrame(
            {
                "species": ["a"] * 40,
                "lon": rng.uniform(-180, 180, 40),
                "lat": rng.uniform(85, 89.9, 40),
            }
        )
    )
    with pytest.raises(ValueError, match="antimeridian or covers a pole"):
        samplesar.designs.quadrats(polar, 200)


def test_a_design_built_directly_holds_only_patches():
    made = samplesar.designs.quadrats(scattered(), 40)
    with pytest.raises(ValueError, match="only patches"):
        samplesar.Quadrats(
            sample=made.sample, regions=made.regions, squares=made.squares, units={"x": (1.0, 2)}
        )
    with pytest.raises(ValueError, match="No square"):
        samplesar.Quadrats(sample=made.sample, regions=made.regions)
