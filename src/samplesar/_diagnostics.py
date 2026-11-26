"""How a nest shrinks the sample, and how independent its steps are."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from samplesar._nest import Nest, _cover_areas


def diagnostics(nest: Nest) -> pd.DataFrame:
    """Shape and spread of each step's convex hull.

    Parameters
    ----------
    nest : Nest
        The nest to describe.

    Returns
    -------
    pd.DataFrame
        One row per step: ``step``, ``n_points``, ``area`` (km², within the
        sample's mask (see [`with_mask`][samplesar.Sample.with_mask]) if it has one),
        ``perimeter`` (km; with a distorting ``crs``, corrected by its scale
        at the sample's center, as areas are), ``compactness`` (Polsby-Popper, 4πA/P²: 1 for a
        circle, near 0 for a sliver), ``area_share`` (of the first step's
        area), ``density`` (points per km²) and ``centroid_shift`` (km from
        the first hull's centroid). Shape columns are missing where the
        step's points don't make a polygon. With a mask, also
        ``mask_share``: the share of the step's hull inside the mask. Shape
        columns describe the whole hull; ``area_share`` and ``density`` use
        ``area``. For a nest with parts (see
        [`strategies.split`][samplesar.strategies.split]) the hull is the
        parts' hulls together: the perimeter sums their outlines, and the
        centroid is that of the ground they cover.
    """
    sample = nest.sample
    masked = sample.mask is not None
    scale = sample.area_scale
    # A distorting crs stretches lengths by about the root of its areal scale.
    length = float(np.sqrt(scale))
    first = nest.footprint(0)
    origin = np.asarray(first.centroid.coords[0])
    area0 = _cover_areas(sample, first)[0] if first.area > 0 else np.nan
    rows = []
    for i, idx in enumerate(nest.steps):
        hull = nest.footprint(i)
        polygon = hull.area > 0
        area = _cover_areas(sample, hull)[0] if polygon else 0.0
        perimeter = hull.length / length if polygon else np.nan
        row = {
            "step": i,
            "n_points": len(idx),
            "area": area,
            "perimeter": perimeter,
            "compactness": 4 * np.pi * hull.area / hull.length**2 if polygon else np.nan,
            "area_share": area / area0 if polygon and area0 > 0 else np.nan,
            "density": len(idx) / area if polygon and area > 0 else np.nan,
            "centroid_shift": float(np.hypot(*(np.asarray(hull.centroid.coords[0]) - origin)))
            / length
            if len(idx)
            else np.nan,
        }
        if masked:
            row["mask_share"] = area * scale / hull.area if polygon else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def autocorrelation(nest: Nest, model: str = "power", **kwargs: Any) -> dict[str, float]:
    """How far a fit's residuals along a nest are from independent.

    Neighboring steps share all but one point, so their residuals tend to
    run together. Strong autocorrelation means the fit's standard errors, R²
    and AICc treat the steps as more independent evidence than they are.

    Parameters
    ----------
    nest : Nest
        The nest whose species-area data are fitted.
    model : str
        The sars model, as [`Nest.fit`][samplesar.Nest.fit] takes it.
    **kwargs
        Passed to the sars function.

    Returns
    -------
    dict
        ``lag1``: the lag-1 autocorrelation of the residuals in step order
        (0 for independent residuals, near 1 for a smooth run);
        ``durbin_watson``: the Durbin-Watson statistic (about 2 for
        independent residuals, near 0 for strong positive autocorrelation);
        ``n``: the number of residuals.

    Raises
    ------
    ValueError
        As [`Nest.fit`][samplesar.Nest.fit].
    """
    data = nest.sar_data()
    fit = nest.fit(model, **kwargs)
    residuals = data["species"].to_numpy(dtype=float) - np.asarray(
        fit.predict(data["area"].to_numpy(dtype=float)), dtype=float
    )
    centered = residuals - residuals.mean()
    denominator = float(np.sum(centered**2))
    if denominator == 0:
        return {"lag1": np.nan, "durbin_watson": np.nan, "n": len(residuals)}
    lag1 = float(np.sum(centered[1:] * centered[:-1]) / denominator)
    durbin_watson = float(np.sum(np.diff(residuals) ** 2) / np.sum(residuals**2))
    return {"lag1": lag1, "durbin_watson": durbin_watson, "n": len(residuals)}
