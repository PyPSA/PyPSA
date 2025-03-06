"""Statistics Accessor."""

from __future__ import annotations

import functools
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Literal

import seaborn.objects as so
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from pypsa.plot.maps import plot
from pypsa.plot.statistics import (
    AreaPlotGenerator,
    BarPlotGenerator,
    LinePlotGenerator,
    MapPlotGenerator,
)
from pypsa.statistics.expressions import StatisticsAccessor

if TYPE_CHECKING:
    from pypsa import Network

# TODO fix removed nice_names=False


class StatisticsPlotter:
    """
    Create plots based on output of statistics functions.

    Passed arguments and the specified statistics function are stored and called
    later when the plot is created, depending on the plot type. Also some checks
    are performed to validated the arguments.
    """

    def __init__(self, n: Network, statistics_function: Callable) -> None:
        """
        Initialize StatisticsPlotter.

        Parameters
        ----------
        n : pypsa.Network
            Network object.
        statistics_function : Callable
            Statistics function to be lazy evaluated. One of the functions in
            :mod:`pypsa.statistics`.

        """
        self._n = n
        self._stats_func = statistics_function

    def bar(
        self,
        x: str | None = None,
        y: str | None = None,
        color: str | None = "carrier",
        col: str | None = None,
        row: str | None = None,
        stacked: bool = False,
        dodged: bool = False,
        query: str | None = None,
        # Statistics kwargs
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
        x : str, optional
            Data to show on x-axis. E.g. "carrier". Default depends on underlying
            statistics function.
        y : str, optional
            Data to show on y-axis. E.g. "value". Default depends on underlying
            statistics function.
        color : str | None, default: "carrier"
            Data to show as color. Pass None to disable color mapping.
        col : str | None, default: None
            Whether to create subplots with conditional subsets of the data. See
            :class:`seaborn.objects.Plot.facet` for more information.
        row : str | None, default: None
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
        >>> n.plot.installed_capacity.bar(x="carrier", y="value", color=None) # doctest: +ELLIPSIS
        <seaborn._core.plot.Plot object at 0x...>

        """
        # Get plotting kwargs
        plot_kwargs = {
            k: v
            for k, v in locals().items()
            if k not in ["self", "storage", "carrier", "bus_carrier", "kwargs"]
        }

        plotter = BarPlotGenerator(self._n)

        # Derive base statistics kwargs
        # (groupby, aggregate_across_components, aggregate_time)
        stats_kwargs = plotter.derive_statistic_parameters(
            x,
            y,
            color,
            col,
            row,
            method_name=self._stats_func.__name__,
        )

        # Add provided kwargs
        stats_kwargs.update(
            storage=storage,
            carrier=carrier,
            bus_carrier=bus_carrier,
        )

        # Check and adjust both kwargs according to matching schema
        stats_kwargs, plot_kwargs = plotter.manage_parameters(
            self._stats_func.__name__, stats_kwargs, plot_kwargs
        )

        # Get statistics datawith updated kwargs and return plot
        data = self._stats_func(**stats_kwargs, nice_names=False)
        return plotter.plot(data, **plot_kwargs, **kwargs)

    def line(
        self,
        x: str | None = None,
        y: str | None = None,
        color: str | None = "carrier",
        col: str | None = None,
        row: str | None = None,
        resample: str | None = None,
        query: str | None = None,
        # Statistics kwargs
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
        x : str, optional
            Data to show on x-axis. E.g. "carrier". Default depends on underlying
            statistics function.
        y : str, optional
            Data to show on y-axis. E.g. "value". Default depends on underlying
            statistics function.
        color : str | None, default: "carrier"
            Data to show as color. Pass None to disable color mapping.
        col : str | None, default: None
            Whether to create subplots with conditional subsets of the data. See
            :class:`seaborn.objects.Plot.facet` for more information.
        row : str | None, default: None
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
        >>> n.plot.installed_capacity.line(x="carrier", y="value", color=None) # doctest: +ELLIPSIS
        <seaborn._core.plot.Plot object at 0x...>

        """
        # Get plotting kwargs
        plot_kwargs = {
            k: v
            for k, v in locals().items()
            if k not in ["self", "storage", "carrier", "bus_carrier", "kwargs"]
        }

        plotter = LinePlotGenerator(self._n)

        # Derive base statistics kwargs
        # (groupby, aggregate_across_components, aggregate_time)
        stats_kwargs = plotter.derive_statistic_parameters(
            x,
            y,
            color,
            col,
            row,
            method_name=self._stats_func.__name__,
        )

        # Add provided kwargs
        stats_kwargs.update(
            storage=storage,
            carrier=carrier,
            bus_carrier=bus_carrier,
        )

        # Check and adjust both kwargs according to matching schema
        stats_kwargs, plot_kwargs = plotter.manage_parameters(
            self._stats_func.__name__, stats_kwargs, plot_kwargs
        )

        # Get statistics datawith updated kwargs and return plot
        data = self._stats_func(**stats_kwargs, nice_names=False)
        return plotter.plot(data, **plot_kwargs, **kwargs)

    def area(
        self,
        x: str = "carrier",
        y: str = "value",
        color: str | None = None,
        col: str | None = None,
        row: str | None = None,
        stacked: bool = False,
        dodged: bool = False,
        query: str | None = None,
        # Statistics kwargs
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
        x : str, optional
            Data to show on x-axis. E.g. "carrier". Default depends on underlying
            statistics function.
        y : str, optional
            Data to show on y-axis. E.g. "value". Default depends on underlying
            statistics function.
        color : str | None, default: "carrier"
            Data to show as color. Pass None to disable color mapping.
        col : str | None, default: None
            Whether to create subplots with conditional subsets of the data. See
            :class:`seaborn.objects.Plot.facet` for more information.
        row : str | None, default: None
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
        >>> n.plot.installed_capacity.area(x="carrier", y="value", color=None) # doctest: +ELLIPSIS
        <seaborn._core.plot.Plot object at 0x...>

        """
        # Get plotting kwargs
        plot_kwargs = {
            k: v
            for k, v in locals().items()
            if k not in ["self", "storage", "carrier", "bus_carrier", "kwargs"]
        }

        plotter = AreaPlotGenerator(self._n)

        # Derive base statistics kwargs
        # (groupby, aggregate_across_components, aggregate_time)
        stats_kwargs = plotter.derive_statistic_parameters(
            x,
            y,
            color,
            col,
            row,
            method_name=self._stats_func.__name__,
        )

        # Add provided kwargs
        stats_kwargs.update(
            storage=storage,
            carrier=carrier,
            bus_carrier=bus_carrier,
        )

        # Check and adjust both kwargs according to matching schema
        stats_kwargs, plot_kwargs = plotter.manage_parameters(
            self._stats_func.__name__, stats_kwargs, plot_kwargs
        )

        # Get statistics datawith updated kwargs and return plot
        data = self._stats_func(**stats_kwargs, nice_names=False)
        return plotter.plot(data, **plot_kwargs, **kwargs)

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
        kind: str | None = None,
        # TODO: Additional stat kwargs needed?
        **kwargs: Any,
    ) -> tuple[Figure, Axes]:  # Store locals to pass same signature to plotter
        local_vars = locals()
        del local_vars["self"]
        del local_vars["kwargs"]

        plotter = MapPlotGenerator(self._n)
        return plotter.plot_statistics(
            func=self._stats_func, stats_kwargs={}, **local_vars, **kwargs
        )


def _register_plotters(cls: type[PlotAccessor]) -> type[PlotAccessor]:
    original_methods = {}

    # Store original methods
    for attr_name in dir(cls):
        if not (attr_name.startswith("_") or attr_name.endswith("__")):
            attr = getattr(cls, attr_name)
            if callable(attr) and not isinstance(attr, property):
                original_methods[attr_name] = attr

    # Now replace them with properties
    # for attr_name, original_method in original_methods.items():

    for attr_name, original_method in original_methods.items():

        def create_property(method_name: str, orig_method: Callable) -> property:
            def getter(cls: type[PlotAccessor]) -> StatisticsPlotter:
                # Get the parent class to access the original method
                parent = super(PlotAccessor, cls).__class__

                # Create a function that will properly call the original method
                @functools.wraps(orig_method)
                def method_caller(*args: Any, **kwargs: Any) -> Any:
                    # Use the original method directly from our saved dictionary
                    bound_method = original_methods[method_name].__get__(cls, parent)
                    return bound_method(*args, **kwargs)

                # Return StatisticsPlotter with our callable function
                return StatisticsPlotter(cls.n, method_caller)

            return property(getter)

        setattr(cls, attr_name, create_property(attr_name, original_method))

    return cls


@_register_plotters
class PlotAccessor(StatisticsAccessor):
    """
    Accessor for plotting statistics.

    The class inherits from StatisticsAccessor and provides the same statistic
    functions, but returns a StatisticsPlotter object instead of a DataFrame.
    """

    @functools.wraps(plot)
    def __call__(self, *args: Any, **kwargs: Any) -> Any:  # type: ignore
        """Legacy plot method."""
        return plot(self.n, *args, **kwargs)
