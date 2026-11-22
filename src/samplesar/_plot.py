"""Plots and animations of nests (matplotlib, imported only when used)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from shapely.geometry import MultiPoint, Polygon

from samplesar._nest import Nest

if TYPE_CHECKING:
    from matplotlib.animation import FuncAnimation
    from matplotlib.axes import Axes


def _lonlat(nest: Nest) -> np.ndarray:
    """The sample's points with longitudes unwrapped around its centre, so a
    sample across the antimeridian draws as one piece."""
    points = np.array(nest.sample.points, dtype=float)
    lon0 = nest.sample.projection_centre[0]
    points[:, 0] = (points[:, 0] - lon0 + 180.0) % 360.0 - 180.0 + lon0
    return points


def _hull_ring(nest: Nest, step: int, lonlat_all: np.ndarray) -> np.ndarray | None:
    """The step's convex hull, as a closed ring of (lon, lat) points, or
    None when its points don't make a polygon."""
    idx = nest.steps[step]
    xy, lonlat = nest.sample.xy[idx], lonlat_all[idx]
    hull = MultiPoint(xy).convex_hull
    if not isinstance(hull, Polygon):
        return None
    where = {tuple(p): i for i, p in enumerate(xy)}
    return np.array([lonlat[where[tuple(c)]] for c in hull.exterior.coords])


def _caption(nest: Nest, step: int, table: Any) -> str:
    row = table.iloc[step]
    return (
        f"step {step} of {nest.n_steps - 1}: {int(row['n_points'])} points, "
        f"{int(row['species'])} species, {row['area']:.4g} km²"
    )


def _frame_axes(ax: Axes, points: np.ndarray, own_axes: bool) -> None:
    pad = 0.05 * np.ptp(points, axis=0).max()
    ax.set_xlim(points[:, 0].min() - pad, points[:, 0].max() + pad)
    ax.set_ylim(points[:, 1].min() - pad, points[:, 1].max() + pad)
    # Degrees of longitude shrink with latitude: scale them so shapes look true.
    ax.set_aspect(1 / np.cos(np.radians(points[:, 1].mean())))
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    if own_axes:
        # Long coordinates (-82.245) collide when matplotlib picks many ticks;
        # axes a caller passed in keep whatever ticks they set.
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.set_major_locator(MaxNLocator(5))
    if points[:, 0].min() < -180 or points[:, 0].max() > 180:
        # Across the antimeridian, longitudes were unwrapped to draw the
        # sample in one piece; label them back within [-180, 180).
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{(x + 180) % 360 - 180:g}"))


def nest(nest: Nest, step: int = 0, ax: Axes | None = None, show_removed: bool = True) -> Axes:
    """Draw one step of a nest: its points and convex hull.

    Parameters
    ----------
    nest : Nest
        The nest to draw.
    step : int
        Which step: 0 is the full sample.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on, keeping their tick locators; by default a new
        figure, with few enough ticks that long coordinates don't collide.
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
    own_axes = ax is None
    if ax is None:
        _, ax = plt.subplots()
    _draw(nest, step % nest.n_steps, ax, show_removed, nest.to_frame(), _lonlat(nest), own_axes)
    return ax


def _draw(
    nest: Nest,
    step: int,
    ax: Axes,
    show_removed: bool,
    table: Any,
    points: np.ndarray,
    own_axes: bool = True,
) -> None:
    """Draw a step, given the nest's table and unwrapped points (computed once
    for an animation rather than per frame)."""
    kept = nest.steps[step]
    if show_removed:
        gone = np.setdiff1d(np.arange(len(points)), kept)
        ax.scatter(points[gone, 0], points[gone, 1], s=12, color="0.75", label="removed")
    ax.scatter(points[kept, 0], points[kept, 1], s=16, color="C0", label="remaining", zorder=3)
    ring = _hull_ring(nest, step, points)
    if ring is not None:
        ax.fill(ring[:, 0], ring[:, 1], color="C1", alpha=0.2, zorder=1)
        ax.plot(ring[:, 0], ring[:, 1], color="C1", linewidth=1.5, zorder=2)
    _frame_axes(ax, points, own_axes)
    ax.set_title(_caption(nest, step, table))


def curve(nest: Nest, ax: Axes | None = None, fit: bool = True) -> Axes:
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
    from matplotlib.ticker import LogFormatter

    ax.set_xscale("log")
    ax.set_yscale("log")
    # Species counts read better as plain numbers than as powers of ten; this
    # labels minor ticks only when the axis spans too little to need them.
    ax.yaxis.set_major_formatter(LogFormatter(labelOnlyBase=False))
    ax.yaxis.set_minor_formatter(LogFormatter(labelOnlyBase=False, minor_thresholds=(1, 0.4)))
    ax.set_xlabel("Area (km²)")
    ax.set_ylabel("Species")
    ax.legend()
    return ax


def animate(
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
    table, points = nest.to_frame(), _lonlat(nest)

    def draw(step: int) -> list[Any]:
        ax.clear()
        _draw(nest, step, ax, show_removed, table, points)
        return list(ax.get_children())

    # The frames are the steps themselves, in the order they're shown.
    return FuncAnimation(fig, draw, frames=order, interval=interval, blit=False)
