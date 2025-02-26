"""Statistics Accessor."""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import TYPE_CHECKING, Any, Literal

import seaborn.objects as so
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from pypsa.plot.statistics import (
    AreaPlotGenerator,
    BarPlotGenerator,
    LinePlotGenerator,
    MapPlotGenerator,
)
from pypsa.plot.statistics.maps import plot
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

    def __init__(self, n: Network, statistics_function: Callable, kwargs: dict) -> None:
        """
        Initialize StatisticsPlotter.

        Parameters
        ----------
        n : pypsa.Network
            Network object.
        statistics_function : Callable
            Statistics function to be lazy evaluated. One of the functions in
            :mod:`pypsa.statistics`.
        kwargs : dict
            Keyword arguments to be passed to the statistics function.

        """
        self._n = n
        self._stats_func = statistics_function
        self._kwargs = kwargs

    def bar(
        self,
        x: str | None = "carrier",
        y: str = "value",
        color: str | None = "carrier",
        col: str | None = None,
        row: str | None = None,
        query: str | None = None,
        **kwargs: Any,
    ) -> so.Plot:
        # Store locals to pass same signature to plotter
        local_vars = locals()
        del local_vars["self"]
        del local_vars["kwargs"]

        plotter = BarPlotGenerator(self._n)

        # Fill default kwargs if not provided
        stats_kwargs = plotter.add_default_kwargs(
            x,
            y,
            color,
            col,
            row,
            stats_kwargs=self._kwargs,
            method_name=self._stats_func.__name__,
        )

        # Check if for kwarg conflicts
        # plotter. TODO

        # Get statistics datawith updated kwargs and return plot
        data = self._stats_func(**stats_kwargs)
        return plotter.plot(data, **local_vars, **kwargs)

    def line(
        self,
        x: str = "carrier",
        y: str = "value",
        color: str | None = None,
        col: str | None = None,
        row: str | None = None,
        nice_names: bool = True,
        resample: str | None = None,
        query: str | None = None,
        **kwargs: Any,
    ) -> so.Plot:
        # Store locals to pass same signature to plotter
        local_vars = locals()
        del local_vars["self"]
        del local_vars["kwargs"]

        plotter = LinePlotGenerator(self._n)

        # Fill default kwargs if not provided
        stats_kwargs = plotter.add_default_kwargs(
            x,
            y,
            color,
            col,
            row,
            stats_kwargs=self._kwargs,
            method_name=self._stats_func.__name__,
        )

        # Check if for kwarg conflicts
        # plotter. TODO

        # Get statistics datawith updated kwargs and return plot
        data = self._stats_func(**stats_kwargs)
        return plotter.plot(data, **local_vars, **kwargs)

    def area(
        self,
        x: str = "carrier",
        y: str = "value",
        color: str | None = None,
        col: str | None = None,
        row: str | None = None,
        nice_names: bool = True,
        stacked: bool = False,
        dodged: bool = False,
        query: str | None = None,
        **kwargs: Any,
    ) -> so.Plot:
        # Store locals to pass same signature to plotter
        local_vars = locals()
        del local_vars["self"]
        del local_vars["kwargs"]

        plotter = AreaPlotGenerator(self._n)

        # Fill default kwargs if not provided
        stats_kwargs = plotter.add_default_kwargs(
            x,
            y,
            color,
            col,
            row,
            stats_kwargs=self._kwargs,
            method_name=self._stats_func.__name__,
        )

        # Check if for kwarg conflicts
        # plotter. TODO

        # Get statistics datawith updated kwargs and return plot
        data = self._stats_func(**stats_kwargs)
        return plotter.plot(data, **local_vars, **kwargs)

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
        **kwargs: Any,
    ) -> tuple[Figure, Axes]:  # Store locals to pass same signature to plotter
        local_vars = locals()
        del local_vars["self"]
        del local_vars["kwargs"]

        plotter = MapPlotGenerator(self._n)
        return plotter.plot_statistics(
            func=self._stats_func, stats_kwargs=self._kwargs, **local_vars, **kwargs
        )


def register_lazy_plotters(cls: type[PlotAccessor]) -> type[PlotAccessor]:
    """
    Class decorator to register lazy plotters for all methods in StatisticsAccessor.

    A StatisticsPlotter stores the statistics function and its arguments to be
    called later, depending on the plot type.

    Parameters
    ----------
    cls : type
        PlotAccessor class to register lazy plotters for.

    Returns
    -------
    type
        PlotAccessor class with lazy plotters registered.

    """
    for name, method in vars(StatisticsAccessor).items():
        if not name.startswith("_") and callable(method):

            def _create_plotter_method(
                name: str, method: Callable = method
            ) -> Callable:
                # TODO: Patch Docstring
                @wraps(method)
                def plotter(
                    self: PlotAccessor, *args: Any, **kwargs: Any
                ) -> StatisticsPlotter:
                    if args:
                        msg = (
                            "Positional arguments are not supported for plotting "
                            "methods."
                        )
                        raise ValueError(msg)

                    return StatisticsPlotter(
                        self.n,
                        getattr(super(PlotAccessor, self), name),
                        kwargs,
                    )

                return plotter

            setattr(cls, name, _create_plotter_method(name))
    return cls


@register_lazy_plotters
class PlotAccessor(StatisticsAccessor):
    """
    Accessor for plotting statistics.

    The class inherits from StatisticsAccessor and provides the same statistic
    functions, but returns a StatisticsPlotter object instead of a DataFrame.
    """

    @wraps(plot)
    def __call__(self, *args: Any, **kwargs: Any) -> Any:  # type: ignore
        """Legacy plot method."""
        return plot(self.n, *args, **kwargs)
