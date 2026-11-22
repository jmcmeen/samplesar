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
    return samplesar.sources.dataframe(
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
    listed = samplesar.strategies.available()
    assert list(listed) == [
        "random",
        "order",
        "hull_random",
        "hull_peel",
        "hull_area",
        "hull_species",
        "hull_distance",
        "concave_peel",
        "radial",
        "sweep",
        "density_peel",
        "grow",
    ]
    assert listed["hull_peel"] == "Nest a sample by peeling off its whole convex hull at each step."


@pytest.mark.parametrize(
    ("name", "func", "options"),
    [
        ("random", samplesar.strategies.random, {"seed": 1}),
        ("hull_random", samplesar.strategies.hull_random, {"seed": 1}),
        ("hull_peel", samplesar.strategies.hull_peel, {}),
        ("hull_area", samplesar.strategies.hull_area, {"keep": "smallest"}),
        ("hull_species", samplesar.strategies.hull_species, {"keep": "fewest"}),
        ("hull_distance", samplesar.strategies.hull_distance, {"remove": "closest"}),
    ],
)
def test_by_name_matches_the_function(sample, name, func, options):
    by_name = samplesar.strategies.run(sample, name, **options)
    direct = func(sample, **options)
    assert all(np.array_equal(a, b) for a, b in zip(by_name.steps, direct.steps, strict=True))
    for made in (by_name, direct):
        assert made.strategy == name
        assert dict(made.options) == options


def test_labels(sample):
    assert samplesar.strategies.hull_area(sample).label == "hull_area"
    assert (
        samplesar.strategies.run(sample, "hull_area", keep="smallest").label
        == "hull_area(keep='smallest')"
    )
    assert (
        samplesar.strategies.random(sample, min_points=5, seed=2).label
        == "random(min_points=5, seed=2)"
    )
    built = samplesar.Nest(sample=sample, steps=(np.arange(sample.n_points),))
    assert built.label == "nest" and built.strategy == "" and dict(built.options) == {}


def test_options_are_read_only(sample):
    with pytest.raises(TypeError):
        samplesar.strategies.run(sample, "hull_area", keep="smallest").options["keep"] = "largest"


def test_unknown_strategy_lists_the_known_ones(sample):
    with pytest.raises(ValueError, match=r"No strategy 'hull'; known: random, order, hull_random"):
        samplesar.strategies.run(sample, "hull")


def test_unknown_option_is_a_type_error(sample):
    with pytest.raises(TypeError):
        samplesar.strategies.run(sample, "hull_peel", seed=1)


def test_register_a_strategy(sample, clean_registry):
    @samplesar.strategies.register("every_other")
    def every_other_nest(sample, min_points=3):
        """Drop every other point at each step."""
        remaining = np.arange(sample.n_points)
        steps = [remaining]
        while len(remaining) // 2 >= min_points:
            remaining = remaining[::2]
            steps.append(remaining)
        return samplesar.Nest(sample=sample, steps=tuple(steps))

    assert samplesar.strategies.available()["every_other"] == "Drop every other point at each step."
    made = samplesar.strategies.run(sample, "every_other", min_points=4)
    assert made.label == "every_other(min_points=4)"
    assert [len(s) for s in made.steps] == [50, 25, 13, 7]
    table = samplesar.evaluate.compare([made, samplesar.strategies.run(sample, "hull_peel")])
    assert list(table.index) == ["every_other(min_points=4)", "hull_peel"]


def test_registering_a_name_twice_is_an_error(clean_registry):
    with pytest.raises(ValueError, match="already registered"):

        @samplesar.strategies.register("random")
        def another(sample):
            return None


def test_a_strategy_must_return_a_nest(sample, clean_registry):
    @samplesar.strategies.register("broken", description="Returns the wrong thing.")
    def broken(sample):
        return sample

    assert samplesar.strategies.available()["broken"] == "Returns the wrong thing."
    with pytest.raises(TypeError, match="returned Sample, not a Nest"):
        samplesar.strategies.run(sample, "broken")


def test_compare_numbers_repeated_labels(sample):
    table = samplesar.evaluate.compare(
        [
            samplesar.strategies.random(sample),
            samplesar.strategies.random(sample),
            samplesar.strategies.hull_peel(sample),
        ]
    )
    assert list(table.index) == ["random", "random #2", "hull_peel"]


def test_nests_pickle_and_deep_copy(sample):
    import copy
    import pickle

    for made in (
        samplesar.strategies.run(sample, "hull_area", keep="smallest"),
        samplesar.Nest(sample=sample, steps=(np.arange(sample.n_points),)),
    ):
        for again in (pickle.loads(pickle.dumps(made)), copy.deepcopy(made)):
            assert again.label == made.label
            assert all(np.array_equal(a, b) for a, b in zip(again.steps, made.steps, strict=True))


def test_a_same_named_function_cannot_replace_a_built_in(clean_registry):
    with pytest.raises(ValueError, match="already registered"):

        @samplesar.strategies.register("random")
        def random_nest(sample):
            return None


def test_keyword_extras_are_recorded_flat(sample, clean_registry):
    @samplesar.strategies.register("flexible")
    def flexible(sample, **opts):
        """A strategy with free-form options."""
        return samplesar.Nest(sample=sample, steps=(np.arange(sample.n_points),))

    assert samplesar.strategies.run(sample, "flexible", k=2).label == "flexible(k=2)"


def test_an_option_may_be_called_strategy(sample, clean_registry):
    @samplesar.strategies.register("greedy")
    def greedy(sample, strategy="fast"):
        """Takes an option named strategy."""
        return samplesar.Nest(sample=sample, steps=(np.arange(sample.n_points),))

    assert dict(samplesar.strategies.run(sample, "greedy", strategy="slow").options) == {
        "strategy": "slow"
    }


def test_wrapped_summary_is_kept_whole(clean_registry):
    @samplesar.strategies.register("wordy")
    def wordy(sample):
        """Drop the point that
        loses the most area.

        More detail here.
        """

    assert samplesar.strategies.available()["wordy"] == "Drop the point that loses the most area."


def test_numpy_option_values_label_plainly(sample):
    assert samplesar.strategies.random(sample, seed=np.int64(1)).label == "random(seed=1)"
