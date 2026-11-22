"""Run-time benchmark of the hull strategies, for one machine at a time.

Usage (from the repository root):

    uv run python papers/hull-strategies/analysis/benchmark.py [--quick]

Times every strategy of the analysis (S1 to S10) on simulated samples of 250
to 4,000 points, needing no data, and writes the timings, each strategy's
scaling exponent, the machine and Fig. 7 to ``results/benchmarks/<machine>/``,
named after the CPU, so runs on different machines sit side by side.
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from run import RESULTS, STRATEGIES  # noqa: E402

import samplesar  # noqa: E402

BENCHMARK_SIZES = [250, 500, 1000, 2000, 4000]


def machine() -> dict:
    import platform

    import shapely

    cpu = platform.processor() or platform.machine()
    try:  # macOS reports the chip here; Linux in /proc/cpuinfo
        import subprocess

        cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip() or cpu
    except OSError:
        pass
    if Path("/proc/cpuinfo").exists():
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    return {"cpu": cpu, "system": platform.platform(), "python": platform.python_version(),
            "numpy": np.__version__, "shapely": shapely.__version__, "samplesar": samplesar.__version__}


def benchmark(sizes: list[int], repeats: int) -> pd.DataFrame:
    """Each strategy's run time (best of ``repeats``) on simulated samples of
    ``sizes`` points: uniform points in a 2 km UTM square, 80 species."""
    rows = []
    for n in sizes:
        rng = np.random.default_rng(n)
        xy = rng.uniform(0, 2000, size=(n, 2)) + [500_000.0, 4_040_000.0]
        df = pd.DataFrame({"species": rng.choice([f"sp{i}" for i in range(80)], n), "x": xy[:, 0], "y": xy[:, 1]})
        sample = samplesar.sources.dataframe(df, lon_col="x", lat_col="y", crs="EPSG:32617")
        _ = sample.xy  # the projection is computed once, outside the timings
        for code, name, options in STRATEGIES:
            best = np.inf
            for _ in range(repeats):
                start = time.perf_counter()
                samplesar.strategies.run(sample, name, **options)
                best = min(best, time.perf_counter() - start)
                if best > 20:  # one long run is stable enough; don't repeat it
                    break
            rows.append({"strategy": code, "n_points": n, "seconds": best})
        print(f"benchmark: {n} points done")
    return pd.DataFrame(rows)


def scaling(bench: pd.DataFrame) -> pd.DataFrame:
    """Each strategy's fitted exponent b in seconds ~ n^b, from the sizes
    where it took at least a millisecond (below that, overhead dominates)."""
    rows = []
    for code, group in bench.groupby("strategy", sort=False):
        timed = group[group["seconds"] >= 1e-3]
        b = np.polyfit(np.log(timed["n_points"]), np.log(timed["seconds"]), 1)[0] if len(timed) >= 2 else np.nan
        largest = group.loc[group["n_points"].idxmax()]
        rows.append({"strategy": code, "exponent": b, "n_points": int(largest["n_points"]), "seconds": largest["seconds"]})
    return pd.DataFrame(rows)


def figure_runtime(bench: pd.DataFrame, exponents: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.5))
    slope = exponents.set_index("strategy")["exponent"]
    for code, group in bench.groupby("strategy", sort=False):
        ax.plot(group["n_points"], group["seconds"], "o-", label=f"{code} (n^{slope[code]:.1f})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Points in the sample")
    ax.set_ylabel("Seconds to build the nest (best of runs)")
    ax.set_title("Run time by strategy")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def slug(text: str) -> str:
    """A folder name from a CPU's name: lower case, words joined by hyphens."""
    text = re.sub(r"\(R\)|\(TM\)|\bCPU\b|\bProcessor\b|@.*$|\b\d+-Core\b", "", text)
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "machine"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="the three smallest sizes, one run each")
    parser.add_argument("--name", help="the machine's folder name (default: from the CPU)")
    parser.add_argument("--out", type=Path, default=RESULTS / "benchmarks",
                        help="where machine folders go (default: results/benchmarks/)")
    args = parser.parse_args()
    about = machine()
    folder = args.out.expanduser() / (args.name or slug(about["cpu"]))
    folder.mkdir(parents=True, exist_ok=True)
    sizes = BENCHMARK_SIZES[:3] if args.quick else BENCHMARK_SIZES
    bench = benchmark(sizes, repeats=1 if args.quick else 3)
    bench.to_csv(folder / "benchmark.csv", index=False)
    exponents = scaling(bench)
    exponents.to_csv(folder / "scaling.csv", index=False)
    pd.Series(about).to_csv(folder / "machine.csv", header=["value"])
    figure_runtime(bench, exponents, folder / "fig7_runtime.png")
    print("written to", folder)


if __name__ == "__main__":
    main()
