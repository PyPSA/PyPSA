"""Plotting package for PyPSA networks."""

from pypsa.common import deprecated_namespace
from pypsa.plot.maps.interactive import explore as _explore
from pypsa.plot.maps.interactive import iplot as _iplot
from pypsa.plot.maps.static import add_legend_arrows as _add_legend_arrows
from pypsa.plot.maps.static import add_legend_circles as _add_legend_circles
from pypsa.plot.maps.static import add_legend_lines as _add_legend_lines
from pypsa.plot.maps.static import add_legend_patches as _add_legend_patches
from pypsa.plot.maps.static import add_legend_semicircles as _add_legend_semicircles
from pypsa.plot.maps.static import plot as _plot

# Create wrapped versions
plot = deprecated_namespace(_plot, "pypsa.plot", deprecated_in="0.33", removed_in="1.0")  # noqa: F811
iplot = deprecated_namespace(
    _iplot, "pypsa.plot", deprecated_in="0.34", removed_in="1.0"
)  # noqa: F811
explore = deprecated_namespace(
    _explore, "pypsa.plot", deprecated_in="0.34", removed_in="1.0"
)  # noqa: F811
add_legend_arrows = deprecated_namespace(
    _add_legend_arrows, "pypsa.plot", deprecated_in="0.34", removed_in="1.0"
)  # noqa: F811
add_legend_circles = deprecated_namespace(
    _add_legend_circles, "pypsa.plot", deprecated_in="0.34", removed_in="1.0"
)  # noqa: F811
add_legend_lines = deprecated_namespace(
    _add_legend_lines, "pypsa.plot", deprecated_in="0.34", removed_in="1.0"
)  # noqa: F811
add_legend_patches = deprecated_namespace(
    _add_legend_patches, "pypsa.plot", deprecated_in="0.34", removed_in="1.0"
)  # noqa: F811
add_legend_semicircles = deprecated_namespace(
    _add_legend_semicircles, "pypsa.plot", deprecated_in="0.34", removed_in="1.0"
)  # noqa: F811

__all__ = [
    "plot",  # deprecated namespace
    "iplot",  # deprecated namespace
    "explore",  # deprecated namespace
    "add_legend_arrows",  # deprecated namespace
    "add_legend_circles",  # deprecated namespace
    "add_legend_lines",  # deprecated namespace
    "add_legend_patches",  # deprecated namespace
    "add_legend_semicircles",  # deprecated namespace
]
