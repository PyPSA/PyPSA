"""Maps plots based on statistics functions."""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal

import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure, SubFigure

from pypsa.plot.maps.static import (
    MapPlotter,
    add_legend_arrows,
    add_legend_circles,
    add_legend_lines,
    add_legend_patches,
    add_legend_semicircles,
    get_legend_representatives,
)
from pypsa.plot.statistics.base import PlotsGenerator

if TYPE_CHECKING:
    from pypsa import Network


class MapPlotGenerator(PlotsGenerator, MapPlotter):
    """Main statistics map plot accessor providing access to different plot types.

    This class combines functionality from both StatisticsPlotAccessor and NetworkMapPlotter
    to create geographic visualizations of network statistics.
    """

    _n: "Network"

    def __init__(self, n: "Network") -> None:
        """Initialize the MapPlotter with a PyPSA network.

        Parameters
        ----------
        n : Network
            PyPSA network object

        """
        PlotsGenerator.__init__(self, n)
        MapPlotter.__init__(self, n)

    def derive_statistic_parameters(
        self,
        *args: str | None,
        method_name: str = "",  # make required
    ) -> dict[str, Any]:
        """Handle default statistics kwargs based on provided plot kwargs."""
        return {}

    def plot(
        self,
        func: Callable,
        bus_carrier: str | None = None,
        ax: Axes | None = None,
        projection: Any = None,
        geomap: bool = True,
        geomap_resolution: Literal["10m", "50m", "110m"] = "50m",
        geomap_colors: dict | bool | None = None,
        boundaries: tuple[float, float, float, float] | None = None,
        title: str = "",
        carrier: str | None = None,
        transmission_flow: bool = False,
        bus_area_fraction: float = 0.02,
        branch_area_fraction: float = 0.02,
        flow_area_fraction: float = 0.02,
        draw_legend_circles: bool = True,
        draw_legend_lines: bool = True,
        draw_legend_arrows: bool = False,
        draw_legend_patches: bool = True,
        legend_circles_kw: dict | None = None,
        legend_lines_kw: dict | None = None,
        legend_arrows_kw: dict | None = None,
        legend_patches_kw: dict | None = None,
        bus_split_circles: bool = False,
        stats_kwargs: dict | None = None,
        nice_names: bool = True,
        **kwargs: Any,
    ) -> tuple[Figure | SubFigure | Any, Axes | Any]:
        """Plot network statistics on a map."""
        n = self._n
        colors = self.get_carrier_colors(nice_names=False)
        n.consistency_check_plots()
        boundaries = boundaries or self.boundaries
        (x_min, x_max, y_min, y_max) = boundaries  # type: ignore

        # Get non-transmission carriers
        # TODO solve circular import by refactoring to descriptors.py
        from pypsa.statistics.expressions import get_transmission_carriers

        trans_carriers = get_transmission_carriers(n, bus_carrier=bus_carrier).unique(
            "carrier"
        )
        non_transmission_carriers = n.carriers.index.difference(trans_carriers)

        # Get bus sizes from statistics function
        bus_sizes = func(
            bus_carrier=bus_carrier,
            groupby=["bus", "carrier"],
            carrier=list(non_transmission_carriers),
            nice_names=False,
            aggregate_across_components=True,
            **(stats_kwargs or {}),
        )
        if bus_sizes.empty:
            # TODO: this fallback case should be handled in the statistics function
            bus_sizes = (
                pd.DataFrame({"bus": [], "carrier": [], "value": []})
                .set_index(["bus", "carrier"])
                .value
            )

        # Calculate scaling factors for visual elements
        bus_size_scaling_factor = self.scaling_factor_from_area_contribution(
            bus_sizes, x_min, x_max, y_min, y_max, bus_area_fraction
        )

        # Handle transmission flows or branch widths
        if transmission_flow:
            branch_flows = n.statistics.transmission(
                groupby=False, bus_carrier=bus_carrier, nice_names=False
            )
            branch_flows = MapPlotGenerator.aggregate_flow_by_connection(
                branch_flows, n.branches()
            )
            branch_flow_scaling_factor = self.scaling_factor_from_area_contribution(
                branch_flows, x_min, x_max, y_min, y_max, flow_area_fraction
            )

            branch_flow_scaled = branch_flows * branch_flow_scaling_factor
            branch_widths_scaled = self.flow_to_width(branch_flow_scaled)
        else:
            branch_flow_scaled = {}
            branch_widths = func(
                comps=n.branch_components,
                bus_carrier=bus_carrier,
                groupby=False,
                carrier=list(trans_carriers),
                nice_names=False,
                **(stats_kwargs or {}),
            )
            branch_widths_scaling_factor = self.scaling_factor_from_area_contribution(
                branch_widths, x_min, x_max, y_min, y_max, branch_area_fraction
            )
            branch_widths_scaled = branch_widths * branch_widths_scaling_factor

        # Get branch colors from carrier colors
        branch_colors = n.branches().carrier[branch_widths_scaled.index].map(colors)

        # Set default plot arguments
        plot_args = {
            "bus_sizes": bus_sizes * bus_size_scaling_factor,
            "bus_split_circles": bus_split_circles,
            "bus_colors": colors,
            "line_flow": branch_flow_scaled.get("Line"),
            "line_widths": branch_widths_scaled.get("Line", 0),
            "line_colors": branch_colors.get("Line", "k"),
            "link_flow": branch_flow_scaled.get("Link"),
            "link_widths": branch_widths_scaled.get("Link", 0),
            "link_colors": branch_colors.get("Link", "k"),
            "transformer_flow": branch_flow_scaled.get("Transformer"),
            "transformer_widths": branch_widths_scaled.get("Transformer", 0),
            "transformer_colors": branch_colors.get("Transformer", "k"),
            "auto_scale_branches": False,
        }

        # Override with user-provided arguments
        if kwargs:
            plot_args.update(kwargs)

        # Draw the map
        self.draw_map(
            ax=ax,
            projection=projection,
            geomap=geomap,
            geomap_resolution=geomap_resolution,
            geomap_colors=geomap_colors,
            title=title,
            boundaries=boundaries,
            **plot_args,
        )

        # Get unit for legends
        unit = bus_sizes.attrs.get("unit", "")
        if unit == "carrier dependent":
            unit = ""

        # Add legends if requested
        if draw_legend_circles and hasattr(self.ax, "figure"):
            legend_representatives = get_legend_representatives(
                bus_sizes, group_on_first_level=True, base_unit=unit
            )

            if legend_representatives:
                if bus_split_circles:
                    add_legend_semicircles(
                        self.ax,  # type: ignore
                        [
                            s * bus_size_scaling_factor
                            for s, label in legend_representatives
                        ],
                        [label for s, label in legend_representatives],
                        legend_kw={
                            "bbox_to_anchor": (0, 0.9),
                            "loc": "lower left",
                            "frameon": True,
                            **(legend_circles_kw or {}),
                        },
                    )
                else:
                    add_legend_circles(
                        self.ax,  # type: ignore
                        [
                            s * bus_size_scaling_factor
                            for s, label in legend_representatives
                        ],
                        [label for s, label in legend_representatives],
                        legend_kw={
                            "bbox_to_anchor": (0, 0.9),
                            "loc": "lower left",
                            "frameon": True,
                            **(legend_circles_kw or {}),
                        },
                    )

        if draw_legend_arrows and hasattr(self.ax, "figure"):
            if not transmission_flow:
                msg = "Cannot draw arrow legend if transmission_flow is False. Use draw_legend_lines instead."
                raise ValueError(msg)

            legend_representatives = get_legend_representatives(
                branch_flows, n_significant=1, base_unit=unit
            )

            if legend_representatives:
                add_legend_arrows(
                    self.ax,  # type: ignore
                    [
                        s * branch_flow_scaling_factor
                        for s, label in legend_representatives
                    ],
                    [label for s, label in legend_representatives],
                    legend_kw={
                        "bbox_to_anchor": (0, 0.9),
                        "loc": "upper left",
                        "frameon": True,
                        **(legend_arrows_kw or {}),
                    },
                )

        if draw_legend_lines and hasattr(self.ax, "figure"):
            if transmission_flow:
                msg = "Cannot draw line legend if transmission_flow is True. Use draw_legend_arrows instead."
                raise ValueError(msg)

            legend_representatives = get_legend_representatives(
                branch_widths, n_significant=1, base_unit=unit
            )

            if legend_representatives:
                add_legend_lines(
                    self.ax,  # type: ignore
                    [
                        s * branch_widths_scaling_factor
                        for s, label in legend_representatives
                    ],
                    [label for s, label in legend_representatives],
                    legend_kw={
                        "bbox_to_anchor": (0, 0.9),
                        "loc": "upper left",
                        "frameon": True,
                        **(legend_lines_kw or {}),
                    },
                )

        if draw_legend_patches and hasattr(self.ax, "figure"):
            carriers = bus_sizes.index.get_level_values("carrier").unique()
            colors = self.get_carrier_colors(carriers, nice_names=False)
            labels = self.get_carrier_labels(carriers, nice_names=nice_names)

            add_legend_patches(
                self.ax,  # type: ignore
                colors=[colors[c] for c in carriers],
                labels=labels,
                legend_kw={
                    "bbox_to_anchor": (1, 1),
                    "loc": "upper left",
                    "frameon": False,
                    **(legend_patches_kw or {}),
                },
            )

        # Ensure ax has a figure (might be None if initialization failed)
        if self.ax is None or not hasattr(self.ax, "figure"):
            import matplotlib.pyplot as plt

            fig = plt.gcf()
            return fig, self.ax or plt.gca()

        return self.ax.figure, self.ax
