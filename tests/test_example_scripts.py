"""The scripts in examples/ run from start to finish and save their figures."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
# 06 fetches real observations over the network; it is checked for syntax only.
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


def test_the_network_example_compiles():
    source = (EXAMPLES / "06_real_observations.py").read_text()
    compile(source, "06_real_observations.py", "exec")
