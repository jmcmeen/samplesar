"""Plots and animations of nests (matplotlib, imported only when used)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from shapely.geometry import MultiPoint, Polygon

from samplesar._nest import Nest

if TYPE_CHECKING:
    from matplotlib.animation import FuncAnimation
    from matplotlib.axes import Axes


def _hull_ring(nest: Nest, step: int) -> np.ndarray | None:
    """The step's convex hull, as a closed ring of (lon, lat) points, or
    None when its points don't make a polygon."""
    idx = nest.steps[step]
    xy, lonlat = nest.sample.xy[idx], nest.sample.points[idx]
    hull = MultiPoint(xy).convex_hull
    if not isinstance(hull, Polygon):
        return None
    where = {tuple(p): i for i, p in enumerate(xy)}
    return np.array([lonlat[where[tuple(c)]] for c in hull.exterior.coords])


def _caption(nest: Nest, step: int) -> str:
    frame = nest.to_frame().iloc[step]
    return (
        f"step {step} of {nest.n_steps - 1}: {int(frame['n_points'])} points, "
        f"{int(frame['species'])} species, {frame['area']:.4g} km²"
    )


def _frame_axes(ax: Axes, nest: Nest) -> None:
    points = nest.sample.points
    pad = 0.05 * np.ptp(points, axis=0).max()
    ax.set_xlim(points[:, 0].min() - pad, points[:, 0].max() + pad)
    ax.set_ylim(points[:, 1].min() - pad, points[:, 1].max() + pad)
    # Degrees of longitude shrink with latitude: scale them so shapes look true.
    ax.set_aspect(1 / np.cos(np.radians(points[:, 1].mean())))
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")


def plot_nest(nest: Nest, step: int = 0, ax: Axes | None = None, show_removed: bool = True) -> Axes:
    """Draw one step of a nest: its points and convex hull.

    Parameters
    ----------
    nest : Nest
        The nest to draw.
    step : int
        Which step: 0 is the full sample.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; by default a new figure.
    show_removed : bool
        Also draw the points already removed, faintly.

    Returns
    -------
    matplotlib.axes.Axes

    Raises
    ------
    IndexError
        If the nest has no such step.
    """
    import matplotlib.pyplot as plt

    if not -nest.n_steps <= step < nest.n_steps:
        raise IndexError(f"The nest has steps 0 to {nest.n_steps - 1}, not {step}.")
    step %= nest.n_steps
    if ax is None:
        _, ax = plt.subplots()
    points = nest.sample.points
    kept = nest.steps[step]
    if show_removed:
        gone = np.setdiff1d(np.arange(len(points)), kept)
        ax.scatter(points[gone, 0], points[gone, 1], s=12, color="0.75", label="removed")
    ax.scatter(points[kept, 0], points[kept, 1], s=16, color="C0", label="remaining", zorder=3)
    ring = _hull_ring(nest, step)
    if ring is not None:
        ax.fill(ring[:, 0], ring[:, 1], color="C1", alpha=0.2, zorder=1)
        ax.plot(ring[:, 0], ring[:, 1], color="C1", linewidth=1.5, zorder=2)
    _frame_axes(ax, nest)
    ax.set_title(_caption(nest, step))
    return ax


def plot_nest_curve(nest: Nest, ax: Axes | None = None, fit: bool = True) -> Axes:
    """Draw a nest's species-area curve on log axes.

    Parameters
    ----------
    nest : Nest
        The nest to draw.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; by default a new figure.
    fit : bool
        Also draw the power model fitted with sars (``nest.fit()``).

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()
    data = nest.sar_data()
    ax.scatter(data["area"], data["species"], s=14, color="C0", label=nest.label)
    if fit and len(data) >= 3:
        model = nest.fit()
        areas = np.geomspace(data["area"].min(), data["area"].max(), 100)
        c, z = model.params["c"], model.params["z"]
        ax.plot(areas, c * areas**z, color="C1", label=f"power fit, z = {z:.3g}")
    from matplotlib.ticker import ScalarFormatter

    ax.set_xscale("log")
    ax.set_yscale("log")
    # Species counts read better as plain numbers than as powers of ten.
    for axis in (ax.yaxis,):
        axis.set_major_formatter(ScalarFormatter())
        axis.set_minor_formatter(ScalarFormatter())
    ax.set_xlabel("Area (km²)")
    ax.set_ylabel("Species")
    ax.legend()
    return ax


def animate_nest(
    nest: Nest,
    interval: int = 200,
    show_removed: bool = True,
    reverse: bool = False,
    **subplots: Any,
) -> FuncAnimation:
    """Animate a nest shrinking, one frame per step.

    Save it with matplotlib: ``animation.save("nest.gif", writer="pillow")``
    (pillow comes with matplotlib), or ``"nest.mp4"`` if ffmpeg is installed.
    In a notebook, ``IPython.display.HTML(animation.to_jshtml())`` plays it.

    Parameters
    ----------
    nest : Nest
        The nest to animate.
    interval : int
        Milliseconds between frames.
    show_removed : bool
        Keep the removed points on screen, faintly.
    reverse : bool
        Play from the last step to the full sample, so the sample grows.
    **subplots
        Passed to ``matplotlib.pyplot.subplots``, such as ``figsize``.

    Returns
    -------
    matplotlib.animation.FuncAnimation
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    fig, ax = plt.subplots(**subplots)
    order = list(range(nest.n_steps))[::-1] if reverse else list(range(nest.n_steps))

    def draw(frame: int) -> list[Any]:
        ax.clear()
        plot_nest(nest, order[frame], ax=ax, show_removed=show_removed)
        return list(ax.get_children())

    return FuncAnimation(fig, draw, frames=len(order), interval=interval, blit=False)
