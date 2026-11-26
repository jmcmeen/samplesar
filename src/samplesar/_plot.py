"""Plots and animations of nests (matplotlib, imported only when used)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from shapely.geometry import MultiPoint, Polygon

from samplesar._nest import Nest

if TYPE_CHECKING:
    from matplotlib.animation import FuncAnimation
    from matplotlib.axes import Axes

    from samplesar._islands import Islands


def _lonlat(nest: Nest) -> np.ndarray:
    """The sample's points with longitudes unwrapped around its center, so a
    sample across the antimeridian draws as one piece."""
    points = np.array(nest.sample.points, dtype=float)
    lon0 = nest.sample.projection_center[0]
    points[:, 0] = (points[:, 0] - lon0 + 180.0) % 360.0 - 180.0 + lon0
    return points


def _mask_lonlat(nest: Nest) -> Any:
    """The sample's mask, or None. A sample keeps only the part of its mask
    near its points, in longitudes unwrapped as ``_lonlat`` unwraps them."""
    return nest.sample.mask


def _rings(geometry: Any) -> list[np.ndarray]:
    """Every ring (exteriors and holes) of a geometry's polygons."""
    rings = []
    for polygon in getattr(geometry, "geoms", [geometry]):
        if isinstance(polygon, Polygon) and not polygon.is_empty:
            rings.append(np.asarray(polygon.exterior.coords))
            rings.extend(np.asarray(r.coords) for r in polygon.interiors)
    return rings


def _fill(ax: Axes, geometry: Any, **style: Any) -> None:
    """Fill a geometry's polygons, leaving their holes empty."""
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path
    from shapely.geometry.polygon import orient

    for polygon in getattr(geometry, "geoms", [geometry]):
        if not isinstance(polygon, Polygon) or polygon.is_empty:
            continue
        # Exteriors counterclockwise and holes clockwise, so holes stay empty.
        rings = _rings(orient(polygon, sign=1.0))
        vertices = np.concatenate(rings)
        codes = np.concatenate(
            [[Path.MOVETO] + [Path.LINETO] * (len(r) - 2) + [Path.CLOSEPOLY] for r in rings]
        )
        ax.add_patch(PathPatch(Path(vertices, codes), **style))


def _hull_rings(nest: Nest, step: int, lonlat_all: np.ndarray) -> list[np.ndarray]:
    """The step's convex hulls (one, or one per part), each a closed ring of
    (lon, lat) points; none where the points don't make a polygon."""
    rings = []
    for idx in nest.groups(step):
        xy, lonlat = nest.sample.xy[idx], lonlat_all[idx]
        hull = MultiPoint(xy).convex_hull
        if isinstance(hull, Polygon):
            where = {tuple(p): i for i, p in enumerate(xy)}
            rings.append(np.array([lonlat[where[tuple(c)]] for c in hull.exterior.coords]))
    return rings


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
    points = _lonlat(nest)
    _draw(
        nest,
        step % nest.n_steps,
        ax,
        show_removed,
        nest.to_frame(),
        points,
        own_axes,
        _mask_lonlat(nest),
    )
    return ax


def _draw(
    nest: Nest,
    step: int,
    ax: Axes,
    show_removed: bool,
    table: Any,
    points: np.ndarray,
    own_axes: bool = True,
    mask: Any = None,
) -> None:
    """Draw a step, given the nest's table, unwrapped points and mask
    (computed once for an animation rather than per frame). With a mask,
    its outline is drawn and only the hull's part inside it is filled."""
    kept = nest.steps[step]
    if show_removed:
        gone = np.setdiff1d(np.arange(len(points)), kept)
        ax.scatter(points[gone, 0], points[gone, 1], s=12, color="0.75", label="removed")
    ax.scatter(points[kept, 0], points[kept, 1], s=16, color="C0", label="remaining", zorder=3)
    if mask is not None:
        for edge in _rings(mask):
            ax.plot(edge[:, 0], edge[:, 1], color="0.45", linewidth=0.8, zorder=1)
    for ring in _hull_rings(nest, step, points):
        hull = Polygon(ring)
        _fill(
            ax,
            hull if mask is None else mask.intersection(hull),
            color="C1",
            alpha=0.2,
            zorder=1,
            linewidth=0,
        )
        ax.plot(ring[:, 0], ring[:, 1], color="C1", linewidth=1.5, zorder=2)
    _frame_axes(ax, points, own_axes)
    ax.set_title(_caption(nest, step, table))


def curve(nest: Nest | Islands, ax: Axes | None = None, fit: bool = True) -> Axes:
    """Draw a nest's species-area curve on log axes.

    Parameters
    ----------
    nest : Nest or Islands
        The nest to draw, or the regions of
        [`designs.islands`][samplesar.designs.islands].
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
    mask = _mask_lonlat(nest)

    def draw(step: int) -> list[Any]:
        ax.clear()
        _draw(nest, step, ax, show_removed, table, points, True, mask)
        return list(ax.get_children())

    # The frames are the steps themselves, in the order they're shown.
    return FuncAnimation(fig, draw, frames=order, interval=interval, blit=False)


def islands(
    islands: Islands, ax: Axes | None = None, labels: bool = True, show_outside: bool = True
) -> Axes:
    """Draw every region of an island design: its outline, its points and its species count.

    Parameters
    ----------
    islands : Islands
        From [`designs.islands`][samplesar.designs.islands].
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; by default a new figure.
    labels : bool
        Write each region's name and species count above it. Turn off
        for many small regions (counties, say).
    show_outside : bool
        Also draw, faintly, the points that lie in no region.

    Returns
    -------
    matplotlib.axes.Axes
        Regions are colored in turn; a point in two overlapping regions
        takes the color of the later one. Units given by hand have no
        boundary and aren't drawn.

    Raises
    ------
    ValueError
        If the design has no regions, only units given by hand.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patheffects import withStroke

    sample = islands.sample
    if sample is None or not islands.regions:
        raise ValueError("This design has only units given by hand: there is nothing to map.")
    own_axes = ax is None
    if ax is None:
        _, ax = plt.subplots()
    points = sample.points
    frame = islands.to_frame()
    species = dict(zip(frame["region"], frame["species"].astype(int), strict=True))
    inside = np.zeros(sample.n_points, dtype=bool)
    for members in islands._members.values():
        inside[members] = True
    if show_outside and not inside.all():
        ax.scatter(points[~inside, 0], points[~inside, 1], s=6, color="0.8", label="in no region")
    corners = [points] if show_outside or not len(islands.regions) else []
    for i, (name, region) in enumerate(islands.regions.items()):
        color = f"C{i % 10}"
        _fill(ax, region, color=color, alpha=0.12, linewidth=0, zorder=1)
        for edge in _rings(region):
            ax.plot(edge[:, 0], edge[:, 1], color=color, linewidth=1.4, zorder=2)
            corners.append(edge)
        members = islands._members[name]
        ax.scatter(points[members, 0], points[members, 1], s=8, color=color, zorder=3)
        if labels:
            # Above the region, so a small one isn't hidden under its own name.
            west, _, east, north = region.bounds
            ax.annotate(
                f"{name}: {species[name]} species",
                ((west + east) / 2, north),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                color=color,
                zorder=4,
                path_effects=[withStroke(linewidth=2.5, foreground="white")],
            )
    _frame_axes(ax, np.vstack(corners), own_axes)
    observed = sum(1 for name in islands.regions if species[name] > 0)
    count = len(islands.regions)
    title = f"{count} region{'s' if count != 1 else ''}, {observed} with observations"
    if islands.units:
        title += f" ({len(islands.units)} more given by hand, not on the map)"
    ax.set_title(title)
    return ax
