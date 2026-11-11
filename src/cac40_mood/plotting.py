"""One shared look for every figure in the project (matplotlib).

Why a module: the README and the report show figures from several phases side by side. Fixing
the colours and the chrome once keeps them consistent, and the palette below was checked for
colour-blind separation and contrast against the chart background (blue vs. orange: pass).

Rules the figures follow:
- colour is assigned by role in a fixed order (first series blue, second orange), never by rank;
- magnitude uses one hue from light to dark (the blue ramp), never a rainbow;
- grid lines and axes stay light grey so the data is the only loud thing;
- text is always dark ink, never the series colour.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

SURFACE: str = "#fcfcfb"
INK: str = "#0b0b0b"
INK_SECONDARY: str = "#52514e"
INK_MUTED: str = "#898781"
GRID: str = "#e1e0d9"
AXIS: str = "#c3c2b7"

# Categorical colours in their fixed order.
SERIES: tuple[str, ...] = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")

# Sequential ramp (one hue, light -> dark) for heatmaps.
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "blue_ramp", ["#f0efec", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
)


def apply_style() -> None:
    """Set matplotlib defaults for the whole session."""
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "figure.dpi": 110,
            "savefig.dpi": 160,
            "font.family": "sans-serif",
            "font.size": 10,
            "text.color": INK,
            "axes.labelcolor": INK_SECONDARY,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.labelcolor": INK_SECONDARY,
            "ytick.labelcolor": INK_SECONDARY,
            "lines.linewidth": 2.0,
            "lines.solid_capstyle": "round",
            "legend.frameon": False,
        }
    )


def titled(axes: Axes, title: str, subtitle: str) -> None:
    """Bold title with a one-line grey subtitle that says what the reader should see.

    Both are placed in points above the plot area, so the spacing is the same on a short line
    chart and on a tall heatmap.
    """
    axes.annotate(
        title,
        xy=(0, 1),
        xycoords="axes fraction",
        xytext=(0, 22),
        textcoords="offset points",
        fontsize=12,
        fontweight="bold",
        color=INK,
        va="bottom",
    )
    axes.annotate(
        subtitle,
        xy=(0, 1),
        xycoords="axes fraction",
        xytext=(0, 7),
        textcoords="offset points",
        fontsize=9,
        color=INK_SECONDARY,
        va="bottom",
    )


def save(figure: Figure, path: Path) -> Path:
    """Save a figure as PNG (creating the folder) and return the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight")
    return path
