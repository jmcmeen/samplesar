"""Evaluating nests: fitting them with sars, comparing strategies, random
baselines, and diagnostics. Fitting one nest is [`Nest.fit`][samplesar.Nest.fit]."""

from samplesar._diagnostics import autocorrelation, diagnostics
from samplesar._evaluate import against_baseline, baseline, compare

__all__ = ["against_baseline", "autocorrelation", "baseline", "compare", "diagnostics"]
