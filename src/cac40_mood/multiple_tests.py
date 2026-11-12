"""Corrections for looking at several tests at once.

- Holm: for the small family of pre-registered primary tests. Controls the chance of even one
  false positive.
- Benjamini-Hochberg: for exploratory results. Controls the expected share of false positives
  among the results called significant, which is less strict.

Both return adjusted p-values: compare them with 0.05 as usual.
"""

from __future__ import annotations

import numpy as np


def holm(p_values: list[float]) -> list[float]:
    """Holm-adjusted p-values, in the order given.

    Sort the p-values; multiply the smallest by m, the next by m-1, and so on; an adjusted
    value may never be smaller than the one before it, and never above 1.
    """
    count = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(count)
    running_max = 0.0
    for rank, position in enumerate(order):
        running_max = max(running_max, (count - rank) * p_values[position])
        adjusted[position] = min(1.0, running_max)
    return adjusted.tolist()


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    """Benjamini-Hochberg adjusted p-values (false discovery rate), in the order given.

    Sort the p-values; multiply the k-th smallest by m/k; going from the largest down, an
    adjusted value may never be larger than the one after it.
    """
    count = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(count)
    running_min = 1.0
    for rank in range(count - 1, -1, -1):
        position = order[rank]
        running_min = min(running_min, p_values[position] * count / (rank + 1))
        adjusted[position] = running_min
    return adjusted.tolist()
