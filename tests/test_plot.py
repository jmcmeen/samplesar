"""Plots and animations of nests."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

import samplesar  # noqa: E402


@pytest.fixture
def nest():
    rng = np.random.default_rng(0)
    n = 30
    sample = samplesar.Sample.from_df(
        pd.DataFrame(
            {
                "species": rng.choice(list("abcdefghij"), n),
                "lon": rng.uniform(-82.3, -82.2, n),
                "lat": rng.uniform(36.5, 36.6, n),
            }
        )
    )
    return samplesar.hull_peel_nest(sample)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_plot_nest(nest):
    ax = samplesar.plot_nest(nest, step=1)
    remaining, removed = ax.collections[1], ax.collections[0]
    assert len(remaining.get_offsets()) == len(nest.steps[1])
    assert len(removed.get_offsets()) == nest.sample.n_points - len(nest.steps[1])
    assert ax.patches  # the hull
    assert ax.get_title().startswith(f"step 1 of {nest.n_steps - 1}:")
    assert "km²" in ax.get_title()


def test_plot_nest_on_given_axes_without_removed(nest):
    _, ax = plt.subplots()
    assert samplesar.plot_nest(nest, step=-1, ax=ax, show_removed=False) is ax
    assert len(ax.collections) == 1


def test_plot_nest_rejects_a_missing_step(nest):
    with pytest.raises(IndexError, match="steps 0 to"):
        samplesar.plot_nest(nest, step=nest.n_steps)


def test_plot_nest_curve(nest):
    ax = samplesar.plot_nest_curve(nest)
    assert ax.get_xscale() == "log" and ax.get_yscale() == "log"
    assert len(ax.lines) == 1  # the fit
    assert len(samplesar.plot_nest_curve(nest, fit=False).lines) == 0


def test_animate_and_save_a_gif(nest, tmp_path):
    animation = samplesar.animate_nest(nest, interval=50, figsize=(4, 4))
    path = tmp_path / "nest.gif"
    animation.save(path, writer="pillow", fps=5)
    from PIL import Image

    with Image.open(path) as gif:
        assert gif.n_frames == nest.n_steps


@pytest.mark.filterwarnings("ignore:Animation was deleted without rendering")
def test_animate_in_reverse(nest):
    forward = samplesar.animate_nest(nest)
    backward = samplesar.animate_nest(nest, reverse=True)
    assert list(forward.new_frame_seq()) == list(range(nest.n_steps))
    assert list(backward.new_frame_seq()) == list(range(nest.n_steps))[::-1]


def test_animation_computes_the_table_once(nest, monkeypatch, tmp_path):
    calls = []
    original = samplesar.Nest.to_frame
    monkeypatch.setattr(samplesar.Nest, "to_frame", lambda self: calls.append(1) or original(self))
    samplesar.animate_nest(nest).save(tmp_path / "n.gif", writer="pillow", fps=10)
    assert len(calls) == 1


def test_a_sample_across_the_antimeridian_draws_in_one_piece():
    df = pd.DataFrame(
        {
            "species": list("abcde"),
            "lon": [179.5, -179.5, -179.5, 179.5, 179.9],
            "lat": [10, 10, 11, 11, 10.5],
        }
    )
    nest = samplesar.hull_peel_nest(samplesar.Sample.from_df(df), min_points=1)
    ax = samplesar.plot_nest(nest)
    left, right = ax.get_xlim()
    assert right - left < 2  # about a degree wide, not the whole globe


def test_few_enough_ticks_not_to_overlap():
    # A wide, long-coordinate sample: matplotlib's default picks 7 or more
    # longitude ticks here.
    rng = np.random.default_rng(1)
    df = pd.DataFrame(
        {
            "species": rng.choice(list("abcdef"), 40),
            "lon": rng.uniform(-82.30, -82.18, 40),
            "lat": rng.uniform(36.55, 36.57, 40),
        }
    )
    nest = samplesar.hull_peel_nest(samplesar.Sample.from_df(df), min_points=1)
    ax = samplesar.plot_nest(nest)
    ax.figure.set_size_inches(10, 4)
    for axis, lim in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
        shown = [v for v in axis.get_ticklocs() if lim[0] <= v <= lim[1]]
        assert len(shown) <= 6


def test_given_axes_keep_their_ticks(nest):
    from matplotlib.ticker import MultipleLocator

    _, ax = plt.subplots()
    locator = MultipleLocator(0.01)
    ax.xaxis.set_major_locator(locator)
    samplesar.plot_nest(nest, ax=ax)
    assert ax.xaxis.get_major_locator() is locator


def test_antimeridian_ticks_are_labelled_within_180():
    df = pd.DataFrame(
        {
            "species": list("abcde"),
            "lon": [179.5, -179.5, -179.5, 179.5, 179.9],
            "lat": [10, 10, 11, 11, 10.5],
        }
    )
    nest = samplesar.hull_peel_nest(samplesar.Sample.from_df(df), min_points=1)
    ax = samplesar.plot_nest(nest)
    labels = [ax.xaxis.get_major_formatter()(v, i) for i, v in enumerate(ax.get_xticks())]
    assert all(-180 <= float(label) < 180 for label in labels)
