"""The hull-strategies paper's analysis: every table and figure from the data.

Usage (from the repository root):

    uv run python papers/hull-strategies/analysis/run.py \\
        --opendata ~/data/inaturalist-open-data/observations.parquet \\
        [--local-sample path/to/observations.csv] [--quick]

``--opendata`` is the Parquet file built by ``samplesar.download_open_data``.
``--local-sample`` is an optional CSV of iNaturalist observations with
``species``, ``longitude`` and ``latitude`` columns. ``--quick`` uses fewer
baseline nests and simulation replicates, for a fast check.

Writes ``results/tables/*.csv`` and ``results/figures/*.png`` beside this
folder (or under ``--out``). Nothing written holds coordinates of real observations: the only
point maps are of a simulated sample.
"""

from __future__ import annotations

import argparse
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import samplesar  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"

# The strategies compared, S1 to S9 and the concave extension (S10).
STRATEGIES = [
    ("S1", "random", {"seed": 0}),
    ("S2", "hull_random", {"seed": 0}),
    ("S3", "hull_peel", {}),
    ("S4", "hull_area", {"keep": "largest"}),
    ("S5", "hull_area", {"keep": "smallest"}),
    ("S6", "hull_species", {"keep": "most"}),
    ("S7", "hull_species", {"keep": "fewest"}),
    ("S8", "hull_distance", {"remove": "farthest"}),
    ("S9", "hull_distance", {"remove": "closest"}),
    ("S10", "concave_peel", {"ratio": 0.3}),
]

MAX_POINTS = 1500  # larger samples are thinned to this many points (seeded)


@dataclass
class Dataset:
    key: str
    label: str
    sample: samplesar.Sample
    merge_m: float  # the merge distance used for the main results


def regional() -> list[tuple[str, str, dict, float]]:
    """Regional pulls from the open data: key, label, query, merge distance."""
    return [
        ("steele_creek_all", "Steele Creek Park area, all taxa", {"bbox": (-82.26, 36.55, -82.21, 36.59)}, 10),
        ("etn_frogs", "East Tennessee, frogs", {"bbox": (-85.0, 35.0, -81.5, 37.0), "taxon_id": 20979}, 100),
        ("gsmnp_birds", "Great Smoky Mountains, birds", {"bbox": (-84.0, 35.43, -83.1, 35.79), "taxon_id": 3}, 50),
        ("monteverde_frogs", "Monteverde, Costa Rica, frogs", {"bbox": (-84.9, 10.25, -84.7, 10.40), "taxon_id": 20979}, 25),
    ]


def thin(sample: samplesar.Sample, max_points: int, seed: int = 0) -> samplesar.Sample:
    """At most ``max_points`` points, chosen at random (seeded), with all their
    observations."""
    if sample.n_points <= max_points:
        return sample
    keep = np.sort(np.random.default_rng(seed).choice(sample.n_points, max_points, replace=False))
    return sample.subset(keep)


def load_datasets(parquet: Path | None, local: Path | None) -> list[Dataset]:
    datasets = []
    if local is not None:
        sample = samplesar.Sample.from_csv(local, lon_col="longitude", lat_col="latitude")
        datasets.append(Dataset("local", "Steele Creek Park, one observer", sample, 10))
    if parquet is not None:
        for key, label, query, merge in regional():
            sample = samplesar.Sample.from_open_data(parquet, **query)
            datasets.append(Dataset(key, label, sample, merge))
    return datasets


def simulated_community(rng: np.random.Generator, cluster_share: float) -> samplesar.Sample:
    """A 2 km square (in UTM zone 17N metres) with 400 records of 80
    species. Each species occurs within a circular range of radius 200 to
    1,000 m; a record takes one of the species whose range covers its point,
    or any species at random where none does. ``cluster_share`` of the
    records are made at one spot, with 2 m of GPS error, as at a trailhead."""
    n, n_species, side = 400, 80, 2000.0
    centres = rng.uniform(0, side, size=(n_species, 2))
    radii = rng.uniform(200, 1000, size=n_species)
    clustered = int(n * cluster_share)
    spot = np.array([side / 2, side / 2])
    points = np.vstack(
        [rng.uniform(0, side, size=(n - clustered, 2)), spot + rng.normal(0, 2.0, size=(clustered, 2))]
    )
    species = []
    for p in points:
        present = np.flatnonzero(np.hypot(*(centres - p).T) <= radii)
        species.append(f"sp{rng.choice(present) if len(present) else rng.integers(n_species)}")
    origin = np.array([500_000.0, 4_040_000.0])
    df = pd.DataFrame({"species": species, "x": points[:, 0] + origin[0], "y": points[:, 1] + origin[1]})
    return samplesar.Sample.from_df(df, lon_col="x", lat_col="y", crs="EPSG:32617")


POWER_Z_BOUND = 5.0  # sars searches the power model's z within ±5


def at_bounds(fit) -> str:
    """The parameters stopped on a bound: sars' SARFit.at_bounds where the
    installed sars has it, else z at the power model's bound."""
    names = getattr(fit, "at_bounds", None)
    if names is None:
        names = ("z",) if abs(fit.params["z"]) >= POWER_Z_BOUND * (1 - 1e-6) else ()
    return ",".join(names)


def fit_row(nest: samplesar.Nest) -> dict:
    fit = nest.fit()
    auto = samplesar.residual_autocorrelation(nest)
    return {
        "n_steps": nest.n_steps,
        "n_fitted": len(nest.sar_data()),
        "c": fit.params["c"],
        "z": fit.params["z"],
        "r_squared": fit.r_squared,
        "at_bounds": at_bounds(fit),
        "lag1": auto["lag1"],
        "durbin_watson": auto["durbin_watson"],
    }


def run_strategies(sample: samplesar.Sample, baseline_n: int) -> tuple[pd.DataFrame, dict]:
    rows, nests = [], {}
    baseline = samplesar.random_baseline(sample, n=baseline_n, strategy="hull_random", seed=1000)
    for code, name, options in STRATEGIES:
        start = time.perf_counter()
        try:
            nest = samplesar.nest(sample, name, **options)
            row = {"strategy": code, "label": nest.label, **fit_row(nest)}
            placed = samplesar.compare_to_baseline(nest, baseline)
            row["z_baseline_mean"] = placed.loc["z", "mean"]
            row["z_baseline_q025"] = placed.loc["z", "q025"]
            row["z_baseline_q975"] = placed.loc["z", "q975"]
            row["z_p"] = placed.loc["z", "p"]
            nests[code] = nest
        except ValueError as err:  # e.g. no layer to peel, too few steps to fit
            row = {"strategy": code, "label": name, "error": str(err)}
        row["seconds"] = round(time.perf_counter() - start, 2)
        rows.append(row)
    return pd.DataFrame(rows), nests


def merge_sensitivity(sample: samplesar.Sample, distances: list[float]) -> pd.DataFrame:
    rows = []
    for metres in distances:
        merged = sample if metres == 0 else sample.merge_within(metres)
        merged = thin(merged, MAX_POINTS)
        for code, name, options in STRATEGIES:
            try:
                nest = samplesar.nest(merged, name, **options)
                fit = nest.fit()
                rows.append({"merge_m": metres, "n_points": merged.n_points, "strategy": code,
                             "z": fit.params["z"], "at_bounds": at_bounds(fit),
                             "min_area_km2": nest.sar_data()["area"].min()})
            except ValueError:
                rows.append({"merge_m": metres, "n_points": merged.n_points, "strategy": code})
    return pd.DataFrame(rows)


def simulations(replicates: int) -> pd.DataFrame:
    rows = []
    for share in (0.0, 0.25, 0.5):
        for rep in range(replicates):
            sample = simulated_community(np.random.default_rng(rep), share)
            for merge in (0, 10):
                used = sample if merge == 0 else sample.merge_within(merge)
                for code, name, options in STRATEGIES:
                    try:
                        z = samplesar.nest(used, name, **options).fit().params["z"]
                    except ValueError:
                        z = np.nan
                    rows.append({"cluster_share": share, "replicate": rep, "merge_m": merge,
                                 "strategy": code, "z": z})
    return pd.DataFrame(rows)


def figure_schematic(path: Path) -> None:
    """Each strategy on one simulated sample, at the step where about half
    the points remain."""
    sample = simulated_community(np.random.default_rng(7), 0.0)
    fig, axes = plt.subplots(2, 5, figsize=(20, 8.5))
    for ax, (code, name, options) in zip(axes.flat, STRATEGIES, strict=True):
        nest = samplesar.nest(sample, name, **options)
        half = min(range(nest.n_steps), key=lambda i: abs(len(nest.steps[i]) - sample.n_points / 2))
        samplesar.plot_nest(nest, half, ax=ax)
        ax.set_title(f"{code}: {nest.label}", fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    fig.suptitle("The strategies on a simulated sample, at about half its points (removed points grey)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def figure_curves(nests_by_dataset: dict, labels: dict, path: Path) -> None:
    keys = list(nests_by_dataset)
    fig, axes = plt.subplots(1, len(keys), figsize=(5.2 * len(keys), 4.6), squeeze=False)
    for ax, key in zip(axes[0], keys, strict=True):
        for code, nest in nests_by_dataset[key].items():
            d = nest.sar_data()
            ax.plot(d["area"], d["species"], ".", markersize=2.5, label=code)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(labels[key], fontsize=10)
        ax.set_xlabel("Area (km²)")
    axes[0][0].set_ylabel("Species")
    axes[0][-1].legend(fontsize=7, markerscale=3, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def figure_z(table: pd.DataFrame, labels: dict, path: Path) -> None:
    keys = list(dict.fromkeys(table["dataset"]))
    codes = [c for c, _, _ in STRATEGIES]
    fig, ax = plt.subplots(figsize=(11, 5))
    width = 0.8 / len(keys)
    for i, key in enumerate(keys):
        sub = table[table["dataset"] == key].set_index("strategy").reindex(codes)
        x = np.arange(len(codes)) + i * width
        bounded = sub["at_bounds"].fillna("").astype(str).str.len() > 0
        ax.bar(x[~bounded], sub["z"][~bounded], width, label=labels[key])
        ax.bar(x[bounded], sub["z"][bounded], width, color="0.85", hatch="//", edgecolor="0.5")
    ax.set_xticks(np.arange(len(codes)) + 0.4 - width / 2, codes)
    ax.set_ylabel("Power-model z")
    ax.set_ylim(0, 1.2)
    ax.set_title("z by strategy (hatched: stopped on sars' bound, not an estimate; cut at 1.2)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def figure_merge(sens: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for code, group in sens.groupby("strategy", sort=False):
        ax.plot(group["merge_m"], group["z"], "o-", label=code)
    ax.set_xlabel("Merge distance (m)")
    ax.set_ylabel("Power-model z")
    ax.set_ylim(0, 1.2)
    ax.set_title("Sensitivity of z to merging co-located records")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def figure_simulation(sim: pd.DataFrame, path: Path) -> None:
    summary = sim.groupby(["merge_m", "cluster_share", "strategy"], sort=False)["z"].median().reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, (merge, group) in zip(axes, summary.groupby("merge_m"), strict=True):
        for code, g in group.groupby("strategy", sort=False):
            ax.plot(g["cluster_share"], g["z"], "o-", label=code)
        ax.set_title("raw records" if merge == 0 else f"merged within {merge:g} m")
        ax.set_xlabel("Share of records at one spot")
    axes[0].set_ylabel("Median z")
    axes[0].set_ylim(0, 1.2)
    axes[1].legend(fontsize=8, ncol=2)
    fig.suptitle("Simulated clustered effort")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def figure_compactness(nests: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for code, nest in nests.items():
        d = samplesar.nest_diagnostics(nest)
        share = 1 - d["n_points"] / d["n_points"].iloc[0]
        ax.plot(share, d["compactness"], label=code, linewidth=1)
    ax.set_xlabel("Share of points removed")
    ax.set_ylabel("Hull compactness (4πA/P²)")
    ax.set_title("How each strategy shapes the hull")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--opendata", type=Path, help="observations.parquet from download_open_data")
    parser.add_argument("--local-sample", type=Path, help="CSV of iNaturalist observations")
    parser.add_argument("--quick", action="store_true", help="fewer baselines and replicates")
    parser.add_argument("--out", type=Path, default=RESULTS, help="where to write (default: results/)")
    args = parser.parse_args()
    warnings.simplefilter("ignore", UserWarning)  # bounds are recorded in the tables

    out = args.out.expanduser()
    tables, figures = out / "tables", out / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    baseline_n = 19 if args.quick else 99
    replicates = 3 if args.quick else 20

    if args.opendata is None and args.local_sample is None:
        parser.error("give --opendata, --local-sample or both")
    datasets = load_datasets(args.opendata.expanduser() if args.opendata else None,
                             args.local_sample.expanduser() if args.local_sample else None)
    summary, results, nests_by_dataset, labels = [], [], {}, {}
    for ds in datasets:
        raw = ds.sample
        used = thin(raw.merge_within(ds.merge_m), MAX_POINTS)
        within_5m = 1 - raw.merge_within(5).n_points / raw.n_points
        summary.append({"dataset": ds.key, "label": ds.label, "observations": raw.n_observations,
                        "points": raw.n_points, "species": raw.n_species, "area_km2": raw.area,
                        "share_merged_at_5m": within_5m,
                        "merge_m": ds.merge_m, "points_merged": raw.merge_within(ds.merge_m).n_points,
                        "points_used": used.n_points})
        print(f"{ds.label}: {raw.n_observations} obs, {raw.n_points} points -> {used.n_points} used")
        table, nests = run_strategies(used, baseline_n)
        table.insert(0, "dataset", ds.key)
        results.append(table)
        nests_by_dataset[ds.key], labels[ds.key] = nests, ds.label
    pd.DataFrame(summary).to_csv(tables / "datasets.csv", index=False)
    strategies = pd.concat(results, ignore_index=True)
    strategies.to_csv(tables / "strategies.csv", index=False)

    figure_schematic(figures / "fig1_strategies.png")
    figure_curves(nests_by_dataset, labels, figures / "fig2_curves.png")
    figure_z(strategies, labels, figures / "fig3_z_by_strategy.png")
    # The sensitivity analysis and the compactness figure use the local
    # sample when given (as the draft describes), else the first region.
    first = next((d for d in datasets if d.key == "local"), datasets[0])
    print(f"Merge sensitivity and compactness: {first.label}")
    sens = merge_sensitivity(first.sample, [0, 2, 5, 10, 25, 50])
    sens.insert(0, "dataset", first.key)
    sens.to_csv(tables / "merge_sensitivity.csv", index=False)
    figure_merge(sens, figures / "fig4_merge_sensitivity.png")
    sim = simulations(replicates)
    sim.to_csv(tables / "simulations.csv", index=False)
    figure_simulation(sim, figures / "fig5_simulated_clusters.png")
    figure_compactness(nests_by_dataset[first.key], figures / "fig6_compactness.png")
    print("written to", out)


if __name__ == "__main__":
    main()
