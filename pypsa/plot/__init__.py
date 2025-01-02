"""
This module contains functions for plotting PyPSA networks.
"""

# TODO: make this namespace deprecated
from pypsa.plot.maps import (
    MapPlotter,
    add_legend_arrows,
    add_legend_circles,
    add_legend_lines,
    add_legend_patches,
    add_legend_semicircles,
    explore,
    iplot,
    plot,
)

__all__ = [
    "MapPlotter",
    "plot",
    "iplot",
    "explore",
    "add_legend_arrows",
    "add_legend_circles",
    "add_legend_lines",
    "add_legend_patches",
    "add_legend_semicircles",
]
