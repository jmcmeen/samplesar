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


def test_animate_in_reverse(nest):
    animation = samplesar.animate_nest(nest, reverse=True)
    animation._func(0)
    assert animation._fig.axes[0].get_title().startswith(f"step {nest.n_steps - 1} of")
