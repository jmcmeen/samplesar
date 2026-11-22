"""Running strategies by name, registering new ones, and nest labels."""

import numpy as np
import pandas as pd
import pytest

import samplesar
from samplesar import _registry


@pytest.fixture
def sample():
    rng = np.random.default_rng(0)
    n = 50
    return samplesar.Sample.from_df(
        pd.DataFrame(
            {
                "species": rng.choice(list("abcdefghij"), n),
                "lon": rng.uniform(-82.3, -82.2, n),
                "lat": rng.uniform(36.5, 36.6, n),
            }
        )
    )


@pytest.fixture
def clean_registry():
    """Undo any strategies a test registers."""
    before = dict(_registry._REGISTRY)
    yield
    _registry._REGISTRY.clear()
    _registry._REGISTRY.update(before)


def test_built_in_strategies_are_listed():
    listed = samplesar.strategies()
    assert list(listed) == [
        "random",
        "hull_random",
        "hull_peel",
        "hull_area",
        "hull_species",
        "hull_distance",
    ]
    assert listed["hull_peel"] == "Nest a sample by peeling off its whole convex hull at each step."


@pytest.mark.parametrize(
    ("name", "func", "options"),
    [
        ("random", samplesar.random_nest, {"seed": 1}),
        ("hull_random", samplesar.hull_random_nest, {"seed": 1}),
        ("hull_peel", samplesar.hull_peel_nest, {}),
        ("hull_area", samplesar.hull_area_nest, {"keep": "smallest"}),
        ("hull_species", samplesar.hull_species_nest, {"keep": "fewest"}),
        ("hull_distance", samplesar.hull_distance_nest, {"remove": "closest"}),
    ],
)
def test_by_name_matches_the_function(sample, name, func, options):
    by_name = samplesar.nest(sample, name, **options)
    direct = func(sample, **options)
    assert all(np.array_equal(a, b) for a, b in zip(by_name.steps, direct.steps, strict=True))
    for made in (by_name, direct):
        assert made.strategy == name
        assert dict(made.options) == options


def test_labels(sample):
    assert samplesar.hull_area_nest(sample).label == "hull_area"
    assert (
        samplesar.nest(sample, "hull_area", keep="smallest").label == "hull_area(keep='smallest')"
    )
    assert (
        samplesar.random_nest(sample, min_points=5, seed=2).label == "random(min_points=5, seed=2)"
    )
    built = samplesar.Nest(sample=sample, steps=(np.arange(sample.n_points),))
    assert built.label == "nest" and built.strategy == "" and dict(built.options) == {}


def test_options_are_read_only(sample):
    with pytest.raises(TypeError):
        samplesar.nest(sample, "hull_area", keep="smallest").options["keep"] = "largest"


def test_unknown_strategy_lists_the_known_ones(sample):
    with pytest.raises(ValueError, match=r"No strategy 'hull'; known: random, hull_random"):
        samplesar.nest(sample, "hull")


def test_unknown_option_is_a_type_error(sample):
    with pytest.raises(TypeError):
        samplesar.nest(sample, "hull_peel", seed=1)


def test_register_a_strategy(sample, clean_registry):
    @samplesar.register_strategy("every_other")
    def every_other_nest(sample, min_points=3):
        """Drop every other point at each step."""
        remaining = np.arange(sample.n_points)
        steps = [remaining]
        while len(remaining) // 2 >= min_points:
            remaining = remaining[::2]
            steps.append(remaining)
        return samplesar.Nest(sample=sample, steps=tuple(steps))

    assert samplesar.strategies()["every_other"] == "Drop every other point at each step."
    made = samplesar.nest(sample, "every_other", min_points=4)
    assert made.label == "every_other(min_points=4)"
    assert [len(s) for s in made.steps] == [50, 25, 13, 7]
    table = samplesar.compare_nests([made, samplesar.nest(sample, "hull_peel")])
    assert list(table.index) == ["every_other(min_points=4)", "hull_peel"]


def test_registering_a_name_twice_is_an_error(clean_registry):
    with pytest.raises(ValueError, match="already registered"):

        @samplesar.register_strategy("random")
        def another(sample):
            return None


def test_a_strategy_must_return_a_nest(sample, clean_registry):
    @samplesar.register_strategy("broken", description="Returns the wrong thing.")
    def broken(sample):
        return sample

    assert samplesar.strategies()["broken"] == "Returns the wrong thing."
    with pytest.raises(TypeError, match="returned Sample, not a Nest"):
        samplesar.nest(sample, "broken")


def test_compare_a_sequence_needs_distinct_labels(sample):
    twice = [samplesar.nest(sample, "random", seed=0), samplesar.nest(sample, "random", seed=0)]
    with pytest.raises(ValueError, match=r"share a label \(random\(seed=0\)\)"):
        samplesar.compare_nests(twice)
