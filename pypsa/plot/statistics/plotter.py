"""Statistics Accessor."""

from __future__ import annotations

from abc import ABC
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Literal

import seaborn.objects as so
from matplotlib.axes import Axes
from matplotlib.figure import Figure, SubFigure

from pypsa.plot.statistics import (
    AreaPlotGenerator,
    BarPlotGenerator,
    LinePlotGenerator,
    MapPlotGenerator,
)

if TYPE_CHECKING:
    from pypsa import Network


class StatisticPlotter(ABC):
    """
    Create plots based on output of statistics functions.

    Passed arguments and the specified statistics function are stored and called
    later when the plot is created, depending on the plot type. Also some checks
    are performed to validated the arguments.
    """

    def __init__(self, bound_method: Callable, n: Network) -> None:
        """
        Initialize the statistic handler.

        Parameters
        ----------
        bound_method : Callable
            The bound method/ underlying statistic function to call.
        n : Network
            The network object to use for the statistic calculation.

        """
        self._bound_method = bound_method
        self._n = n

    def _chart(
        self,
        chart_type: str,
        plotter_class: type,
        plot_kwargs: dict,
        stats_kwargs: dict,
        **kwargs: Any,
    ) -> so.Plot:
        """
        Common chart generation method used by bar, line and area plots.

        Parameters
        ----------
        chart_type : str
            Type of chart ("bar", "line", or "area")
        plotter_class : type
            The plotter class to instantiate
        plot_kwargs : dict
            Dictionary of plotting parameters
        stats_kwargs : dict
            Dictionary of parameters for the statistics function
        **kwargs : Any
            Additional keyword arguments for the plot function

        Returns
        -------
        seaborn.objects.Plot
            Seaborn plot object

        """
        plotter = plotter_class(self._n)

        # Apply schema to plotting kwargs
        stats_name = self._bound_method.__name__
        plot_kwargs = plotter.apply_parameter_schema(
            stats_name, chart_type, plot_kwargs
        )

        # Derive base statistics kwargs
        base_stats_kwargs = plotter.derive_statistic_parameters(
            plot_kwargs["x"],
            plot_kwargs["y"],
            plot_kwargs["color"],
            plot_kwargs["facet_col"],
            plot_kwargs["facet_row"],
            method_name=self._bound_method.__name__,
        )

        # Add provided kwargs
        stats_kwargs.update(base_stats_kwargs)

        # Apply schema to statistics kwargs
        stats_kwargs = plotter.apply_parameter_schema(
            stats_name, chart_type, stats_kwargs
        )

        # Get statistics data and return plot
        data = self._bound_method(**stats_kwargs, nice_names=False)
        return plotter.plot(data, **plot_kwargs, **kwargs)

    def bar(
        self,
        x: str | None = None,
        y: str = "value",
        color: str | None = "carrier",
        facet_col: str | None = None,
        facet_row: str | None = None,
        stacked: bool = False,
        dodged: bool = False,
        query: str | None = None,
        nice_names: bool = True,
        carrier: Sequence[str] | str | None = None,
        bus_carrier: Sequence[str] | str | None = None,
        storage: bool | None = None,
        **kwargs: Any,
    ) -> so.Plot:
        """
        Plot statistics as bar plot.

        This function builds up on any statistics function and creates a bar plot
        based on it's output. Seaborn is used to create the plot.

        Parameters
        ----------
        x : str, default: None
            Data to show on x-axis. E.g. "carrier". Default depends on underlying
            statistics function.
        y : str, default: "value"
            Data to show on y-axis. E.g. "value".
        color : str | None, default: "carrier"
            Data to show as color. Pass None to disable color mapping.
        facet_col : str | None, default: None
            Whether to create subplots with conditional subsets of the data. See
            :class:`seaborn.objects.Plot.facet` for more information.
        facet_row : str | None, default: None
            Whether to create subplots with conditional subsets of the data. See
            :class:`seaborn.objects.Plot.facet` for more information.
        stacked : bool, default: False
            Whether to stack the bars. See :class:`seaborn.objects.Stack` for more
        dodged : bool, default: False
            Whether to doge the bars. See :class:`seaborn.objects.Doge` for more
            information.
        query : str | None, default: None
            Pandas query string to filter the data before plotting. E.g. "value > 0".
        nice_names : bool, default: True
            Whether to use nice names for components, as defined in
            ``c.static.nice_names.``
        carrier: Sequence[str] | str | None, default: None
            Filter by carrier of components. If specified, only considers assets with
            the given carrier(s). More information can be found in the
            documentation of the statistics functions.
        bus_carrier: Sequence[str] | str | None, default: None
            Filter by carrier of connected buses. If specified, only considers assets
            connected to buses with the given carrier(s). More information can be found
            in the documentation of the statistics functions.
        storage: bool | None, default: None
            Whether to include storage components in the statistics. Can only be used
            when chosen statistics function supports it (e.g. `optimal_capacity`,
            `installed_capacity`). Default is False for those functions.
        **kwargs: Any
            Additional keyword arguments for the plot function. These are passed to
            the seaborn plot object (:class:`seaborn.objects.Plot`).

        Returns
        -------
        seaborn.objects.Plot
            Seaborn plot object.

        Examples
        --------
        >>> import pypsa
        >>> n = pypsa.examples.ac_dc_meshed()
        >>> n.statistics.installed_capacity.bar(x="carrier", y="value", color=None) # doctest: +ELLIPSIS
        <seaborn._core.plot.Plot object at 0x...>

        """
        plot_kwargs = {
            "x": x,
            "y": y,
            "color": color,
            "facet_col": facet_col,
            "facet_row": facet_row,
            "stacked": stacked,
            "dodged": dodged,
            "query": query,
            "nice_names": nice_names,
        }
        stats_kwargs = {
            "carrier": carrier,
            "bus_carrier": bus_carrier,
            "storage": storage,
        }
        return self._chart(
            "bar",
            BarPlotGenerator,
            plot_kwargs,
            stats_kwargs,
            **kwargs,
        )

    def line(
        self,
        x: str | None = None,
        y: str = "value",
        color: str | None = "carrier",
        facet_col: str | None = None,
        facet_row: str | None = None,
        resample: str | None = None,
        query: str | None = None,
        nice_names: bool = True,
        carrier: Sequence[str] | str | None = None,
        bus_carrier: Sequence[str] | str | None = None,
        storage: bool | None = None,
        **kwargs: Any,
    ) -> so.Plot:
        """
        Plot statistics as line plot.

        This function builds up on any statistics function and creates a line plot
        based on it's output. Seaborn is used to create the plot.

        Parameters
        ----------
        x : str, default: None
            Data to show on x-axis. E.g. "carrier". Default depends on underlying
            statistics function.
        y : str, default: "value"
            Data to show on y-axis. E.g. "value".
            Data to show as color. Pass None to disable color mapping.
        color : str | None, default: "carrier"
            Data to show as color. Pass None to disable color mapping.
        facet_col : str | None, default: None
            Whether to create subplots with conditional subsets of the data. See
            :class:`seaborn.objects.Plot.facet` for more information.
        facet_row : str | None, default: None
            Whether to create subplots with conditional subsets of the data. See
            :class:`seaborn.objects.Plot.facet` for more information.
        resample : str | None, default: None
            Resampling frequency. If specified, the data is resampled to the given
            frequency. See :class:`pandas.DataFrame.resample` for more information.
            Aggregation is done by taking the mean.
        query : str | None, default: None
            Pandas query string to filter the data before plotting. E.g. "value > 0".
        nice_names : bool, default: True
            Whether to use nice names for components, as defined in
            ``c.static.nice_names.``
        carrier: Sequence[str] | str | None, default: None
            Filter by carrier of components. If specified, only considers assets with
            the given carrier(s). More information can be found in the
            documentation of the statistics functions.
        bus_carrier: Sequence[str] | str | None, default: None
            Filter by carrier of connected buses. If specified, only considers assets
            connected to buses with the given carrier(s). More information can be found
            in the documentation of the statistics functions.
        storage: bool | None, default: None
            Whether to include storage components in the statistics. Can only be used
            when chosen statistics function supports it (e.g. `optimal_capacity`,
            `installed_capacity`). Default is False for those functions.
        **kwargs: Any
            Additional keyword arguments for the plot function. These are passed to
            the seaborn plot object (:class:`seaborn.objects.Plot`).

        Returns
        -------
        seaborn.objects.Plot
            Seaborn plot object.

        Examples
        --------
        >>> import pypsa
        >>> n = pypsa.examples.ac_dc_meshed()
        >>> n.statistics.installed_capacity.line(x="carrier", y="value", color=None) # doctest: +ELLIPSIS
        <seaborn._core.plot.Plot object at 0x...>

        """
        # Get plotting kwargs
        plot_kwargs = {
            "x": x,
            "y": y,
            "color": color,
            "facet_col": facet_col,
            "facet_row": facet_row,
            "resample": resample,
            "query": query,
            "nice_names": nice_names,
        }
        stats_kwargs = {
            "carrier": carrier,
            "bus_carrier": bus_carrier,
            "storage": storage,
        }
        return self._chart(
            "line",
            LinePlotGenerator,
            plot_kwargs,
            stats_kwargs,
            **kwargs,
        )

    def area(
        self,
        x: str | None = None,
        y: str = "value",
        color: str | None = None,
        facet_col: str | None = None,
        facet_row: str | None = None,
        stacked: bool = False,
        dodged: bool = False,
        query: str | None = None,
        nice_names: bool = True,
        carrier: Sequence[str] | str | None = None,
        bus_carrier: Sequence[str] | str | None = None,
        storage: bool | None = None,
        **kwargs: Any,
    ) -> so.Plot:
        """
        Plot statistics as area plot.

        This function builds up on any statistics function and creates a area plot
        based on it's output. Seaborn is used to create the plot.

        Parameters
        ----------
        x : str, default: None
            Data to show on x-axis. E.g. "carrier". Default depends on underlying
            statistics function.
        y : str, default: "value"
            Data to show on y-axis. E.g. "value".
        color : str | None, default: "carrier"
            Data to show as color. Pass None to disable color mapping.
        facet_col : str | None, default: None
            Whether to create subplots with conditional subsets of the data. See
            :class:`seaborn.objects.Plot.facet` for more information.
        facet_row : str | None, default: None
            Whether to create subplots with conditional subsets of the data. See
            :class:`seaborn.objects.Plot.facet` for more information.
        stacked : bool, default: False
            Whether to stack the bars. See :class:`seaborn.objects.Stack` for more
        dodged : bool, default: False
            Whether to doge the bars. See :class:`seaborn.objects.Doge` for more
            information.
        query : str | None, default: None
            Pandas query string to filter the data before plotting. E.g. "value > 0".
        nice_names : bool, default: True
            Whether to use nice names for components, as defined in
            ``c.static.nice_names.``
        carrier: Sequence[str] | str | None, default: None
            Filter by carrier of components. If specified, only considers assets with
            the given carrier(s). More information can be found in the
            documentation of the statistics functions.
        bus_carrier: Sequence[str] | str | None, default: None
            Filter by carrier of connected buses. If specified, only considers assets
            connected to buses with the given carrier(s). More information can be found
            in the documentation of the statistics functions.
        **kwargs: Any
            Additional keyword arguments for the plot function. These are passed to
            the seaborn plot object (:class:`seaborn.objects.Plot`).
        storage: bool | None, default: None
            Whether to include storage components in the statistics. Can only be used
            when chosen statistics function supports it (e.g. `optimal_capacity`,
            `installed_capacity`). Default is False for those functions.

        Returns
        -------
        seaborn.objects.Plot
            Seaborn plot object.

        Examples
        --------
        >>> import pypsa
        >>> n = pypsa.examples.ac_dc_meshed()
        >>> n.statistics.installed_capacity.area(x="carrier", y="value", color=None) # doctest: +ELLIPSIS
        <seaborn._core.plot.Plot object at 0x...>

        """
        # Get plotting kwargs
        plot_kwargs = {
            "x": x,
            "y": y,
            "color": color,
            "facet_col": facet_col,
            "facet_row": facet_row,
            "stacked": stacked,
            "dodged": dodged,
            "query": query,
            "nice_names": nice_names,
        }
        stats_kwargs = {
            "carrier": carrier,
            "bus_carrier": bus_carrier,
            "storage": storage,
        }
        return self._chart(
            "area",
            AreaPlotGenerator,
            plot_kwargs,
            stats_kwargs,
            **kwargs,
        )

    def map(
        self,
        ax: Axes | None = None,
        projection: Any = None,
        geomap: bool = True,
        geomap_resolution: Literal["10m", "50m", "110m"] = "50m",
        geomap_colors: dict | bool | None = None,
        boundaries: tuple[float, float, float, float] | None = None,
        title: str = "",
        bus_carrier: str | None = None,
        carrier: str | None = None,
        transmission_flow: bool | None = None,
        bus_area_fraction: float = 0.02,
        branch_area_fraction: float = 0.02,
        flow_area_fraction: float = 0.02,
        draw_legend_circles: bool = True,
        draw_legend_lines: bool | None = None,
        draw_legend_arrows: bool | None = None,
        draw_legend_patches: bool = True,
        legend_circles_kw: dict | None = None,
        legend_lines_kw: dict | None = None,
        legend_arrows_kw: dict | None = None,
        legend_patches_kw: dict | None = None,
        bus_split_circles: bool | None = None,
        storage: bool | None = None,
        **kwargs: Any,
    ) -> tuple[Figure | SubFigure | Any, Axes | Any]:
        """
        Plot statistics on a geographic map.

        This function builds upon any statistics function and creates a geographical
        visualization based on its output. It uses the MapPlotGenerator to render the
        network components with sizes and colors based on the statistics results.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axis to plot on. If None, creates a new figure and axis.
        projection : cartopy.crs.Projection, optional
            Map projection to use. If None and geomap is True, uses the network's CRS.
        geomap : bool, default True
            Whether to add geographic features with cartopy.
        geomap_resolution : {'10m', '50m', '110m'}, default '50m'
            Resolution of geographic features.
        geomap_colors : dict or bool, optional
            Colors for geographic features. If True, uses defaults. If a dict, keys
            can include 'ocean', 'land', 'border', 'coastline'.
        boundaries : tuple(float, float, float, float), optional
            Plot boundaries as (xmin, xmax, ymin, ymax).
        title : str, default ""
            Plot title.
        bus_carrier : str, optional
            Filter by carrier of connected buses.
        carrier : str, optional
            Filter by carrier of components.
        transmission_flow : bool, optional
            Whether to plot transmission flows. If True, draws flow arrows instead of lines.
        bus_area_fraction : float, default 0.02
            Fraction of plot area to be covered by bus circles.
        branch_area_fraction : float, default 0.02
            Fraction of plot area to be covered by branch lines.
        flow_area_fraction : float, default 0.02
            Fraction of plot area to be covered by flow arrows.
        draw_legend_circles : bool, default True
            Whether to draw a legend for bus sizes.
        draw_legend_lines : bool, optional
            Whether to draw a legend for line widths. Only valid when transmission_flow is False.
        draw_legend_arrows : bool, optional
            Whether to draw a legend for flow arrows. Only valid when transmission_flow is True.
        draw_legend_patches : bool, default True
            Whether to draw a legend for carrier colors.
        legend_circles_kw : dict, optional
            Additional keyword arguments for the circles legend.
        legend_lines_kw : dict, optional
            Additional keyword arguments for the lines legend.
        legend_arrows_kw : dict, optional
            Additional keyword arguments for the arrows legend.
        legend_patches_kw : dict, optional
            Additional keyword arguments for the patches legend.
        bus_split_circles : bool, optional
            Whether to draw half circles for positive/negative values.
        storage : bool, optional
            Whether to show storage capacity in capacity plots. Only valid when
            chosen statistics function supports it (e.g. `optimal_capacity`,
            `installed_capacity`).
        **kwargs :
            Additional keyword arguments passed to the MapPlotGenerator.draw_map method.

        Returns
        -------
        tuple(matplotlib.figure.Figure, matplotlib.axes.Axes)
            The figure and axes of the plot.

        Examples
        --------
        >>> import pypsa
        >>> n = pypsa.examples.ac_dc_meshed()
        >>> fig, ax = n.statistics.installed_capacity.map(geomap=True, title="Installed Capacity")

        """
        plot_kwargs = {
            "ax": ax,
            "projection": projection,
            "geomap": geomap,
            "geomap_resolution": geomap_resolution,
            "geomap_colors": geomap_colors,
            "boundaries": boundaries,
            "title": title,
            "bus_carrier": bus_carrier,
            "carrier": carrier,
            "transmission_flow": transmission_flow,
            "bus_area_fraction": bus_area_fraction,
            "branch_area_fraction": branch_area_fraction,
            "flow_area_fraction": flow_area_fraction,
            "draw_legend_circles": draw_legend_circles,
            "draw_legend_lines": draw_legend_lines,
            "draw_legend_arrows": draw_legend_arrows,
            "draw_legend_patches": draw_legend_patches,
            "legend_circles_kw": legend_circles_kw,
            "legend_lines_kw": legend_lines_kw,
            "legend_arrows_kw": legend_arrows_kw,
            "legend_patches_kw": legend_patches_kw,
            "bus_split_circles": bus_split_circles,
        }

        plotter = MapPlotGenerator(self._n)

        # Apply schema to plotting kwargs
        stats_name = self._bound_method.__name__
        plot_kwargs = plotter.apply_parameter_schema(stats_name, "map", plot_kwargs)

        # Apply schema to statistics kwargs
        stats_kwargs = plotter.apply_parameter_schema(
            stats_name, "map", {"storage": storage}
        )

        # Note that instead of passing the data to the plotter, we pass the
        # statistics function. This gives the map plotter the ability to
        # determine the data for buses and branches itself.
        return plotter.plot(
            func=self._bound_method, **plot_kwargs, stats_kwargs=stats_kwargs, **kwargs
        )
