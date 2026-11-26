"""Shared by the example scripts: where figures go."""

import argparse
from pathlib import Path


def output_dir(description: str) -> Path:
    """The folder to save into: ``--out``, or ``examples/output``."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "output",
        help="folder for figures (default: examples/output)",
    )
    out = parser.parse_args().out
    out.mkdir(parents=True, exist_ok=True)
    return out
