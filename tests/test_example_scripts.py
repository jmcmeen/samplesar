"""The scripts in examples/ run from start to finish and save their figures."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
# 06 fetches real observations over the network; it runs with the services faked, below.
OFFLINE = sorted(p for p in EXAMPLES.glob("0*.py") if not p.name.startswith("06"))


# Each starts an interpreter and draws a figure: seconds apiece, so "make test-fast" skips them.
@pytest.mark.slow
@pytest.mark.parametrize("script", OFFLINE, ids=[p.stem for p in OFFLINE])
def test_example_runs_and_saves_its_figure(script, tmp_path):
    pytest.importorskip("matplotlib")
    done = subprocess.run(
        [sys.executable, str(script), "--out", str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=300,
        env={**os.environ, "MPLBACKEND": "Agg"},
        check=False,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    assert (tmp_path / f"{script.stem}.png").stat().st_size > 10_000
    assert "Saved" in done.stdout


def test_there_are_examples_and_the_readme_lists_them():
    listed = (EXAMPLES / "README.md").read_text()
    scripts = sorted(p.name for p in EXAMPLES.glob("0*.py"))
    assert len(scripts) == 6
    for name in scripts:
        assert f"`{name}`" in listed


def test_the_network_example_runs_with_the_services_faked(tmp_path, monkeypatch):
    # The real script, with GBIF and geoBoundaries replaced, so a renamed
    # argument or function fails here instead of on a user's machine.
    pytest.importorskip("matplotlib")
    import runpy

    from shapely.geometry import box

    import samplesar

    simulated = samplesar.sources.simulated(n_points=150, seed=7)
    lon, lat = simulated.points.T
    outline = box(lon.min() - 0.01, lat.min() - 0.01, lon.max() + 0.01, lat.max() + 0.01)
    asked = {}

    def fake_admin(country, *args, **kwargs):
        asked["country"] = country
        return outline

    real_gbif = samplesar.sources.gbif

    def fake_gbif(records=None, **query):
        # The real function still checks the filtering arguments; the records are made up.
        made = [
            {
                "species": s,
                "decimalLongitude": x,
                "decimalLatitude": y,
                "taxonRank": "SPECIES",
                "coordinateUncertaintyInMeters": 30,
            }
            for s, x, y in simulated.data[["species", "lon", "lat"]].itertuples(index=False)
        ]
        asked["query"] = {k: v for k, v in query.items() if k != "boundary"}
        return real_gbif(
            made, boundary=query["boundary"], max_uncertainty=query.get("max_uncertainty")
        )

    monkeypatch.setattr(samplesar.regions, "admin", fake_admin)
    monkeypatch.setattr(samplesar.sources, "gbif", fake_gbif)
    monkeypatch.setattr(
        "sys.argv", ["06_real_observations.py", "--country", "XYZ", "--out", str(tmp_path)]
    )
    monkeypatch.setenv("MPLBACKEND", "Agg")
    runpy.run_path(str(EXAMPLES / "06_real_observations.py"), run_name="__main__")
    assert asked["country"] == "XYZ"
    assert asked["query"] == {"taxonKey": 952, "max_results": 3000, "max_uncertainty": 1000}
    assert (tmp_path / "06_xyz_952.png").stat().st_size > 10_000
