from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import wraps
from typing import TYPE_CHECKING, Any, ClassVar

import pandas as pd
import seaborn.objects as so

from pypsa.plot.maps import MapPlotter, plot

if TYPE_CHECKING:
    from pypsa import Network


class BasePlotTypeAccessor:
    """Base class containing shared statistics methods"""

    _network: Network
    _statistics: Any
    _time_aggregation: str | bool

    def __init__(self: BasePlotTypeAccessor, n: Network) -> None:
        self._network = n
        self._statistics = n.statistics
        self._time_aggregation = False

    def _to_long_format(
        self: BasePlotTypeAccessor, data: pd.DataFrame | pd.Series
    ) -> pd.DataFrame:
        """
        Convert data to long format suitable for plotting.

        Parameters
        ----------
        data : pd.DataFrame | pd.Series
            Input data from statistics functions, typically with multiindex

        Returns
        -------
        pd.DataFrame
            Long format DataFrame with multiindex levels as columns and values in 'value' column
        """
        if isinstance(data, pd.Series):
            return data.rename("value").reset_index()
        else:
            return data.fillna(0).melt(ignore_index=False).reset_index()

    def _validate(self: BasePlotTypeAccessor, data: pd.DataFrame) -> None:
        """Validation method to be implemented by subclasses"""
        pass

    def _get_carrier_colors(self, data: pd.DataFrame) -> dict:
        """Get colors for carriers from network.carriers.color"""
        if "carrier" in data.columns:
            colors = self._network.carriers.color.copy()
            colors["-"] = "gray"
            return dict(colors[data["carrier"].dropna().unique()])
        return {}

    def _create_base_plot(
        self, data: pd.DataFrame, x: str, y: str, **kwargs: Any
    ) -> so.Plot:
        """Create base plot with carrier colors"""
        colors = self._get_carrier_colors(data)
        return (
            so.Plot(data, x=x, y=y, color="carrier" if colors else None).scale(
                color=so.Nominal(colors)
            )
            if colors
            else so.Plot(data, x=x, y=y)
        )

    def _plot(self: BasePlotTypeAccessor, data: pd.DataFrame, **kwargs: Any) -> Any:
        """Plot method to be implemented by subclasses"""
        raise NotImplementedError

    def _process_data(
        self: BasePlotTypeAccessor, func: Callable, *args: Any, **kwargs: Any
    ) -> Any:
        """Common data processing pipeline"""
        plot_kwargs = kwargs.pop("plot_kwargs", {})
        data = func(*args, **kwargs)
        data = self._to_long_format(data)
        data = self._validate(data)
        return self._plot(data, **plot_kwargs)

    # Shared statistics methods
    def optimal_capacity(
        self: BasePlotTypeAccessor, groupby: Sequence[str] = ["carrier"], **kwargs: Any
    ) -> Any:
        """Plot optimal capacity"""
        return self._process_data(
            self._statistics.optimal_capacity, groupby=groupby, plot_kwargs=kwargs
        )

    def installed_capacity(
        self: BasePlotTypeAccessor, groupby: Sequence[str] = ["carrier"], **kwargs: Any
    ) -> Any:
        """Plot installed capacity"""
        return self._process_data(
            self._statistics.installed_capacity, groupby=groupby, plot_kwargs=kwargs
        )

    def supply(
        self: BasePlotTypeAccessor, groupby: Sequence[str] = ["carrier"], **kwargs: Any
    ) -> Any:
        """Plot supply data"""
        return self._process_data(
            self._statistics.supply,
            groupby=groupby,
            aggregate_time=self._time_aggregation,
            plot_kwargs=kwargs,
        )

    def withdrawal(
        self: BasePlotTypeAccessor, groupby: Sequence[str] = ["carrier"], **kwargs: Any
    ) -> Any:
        """Plot withdrawal data"""
        return self._process_data(
            self._statistics.withdrawal,
            groupby=groupby,
            aggregate_time=self._time_aggregation,
            plot_kwargs=kwargs,
        )

    def energy_balance(
        self: BasePlotTypeAccessor, groupby: Sequence[str] = ["carrier"], **kwargs: Any
    ) -> Any:
        """Plot energy balance"""
        return self._process_data(
            self._statistics.energy_balance,
            groupby=groupby,
            aggregate_time=self._time_aggregation,
            plot_kwargs=kwargs,
        )

    def transmission(
        self: BasePlotTypeAccessor, groupby: Sequence[str] = ["carrier"], **kwargs: Any
    ) -> Any:
        """Plot transmission data"""
        return self._process_data(
            self._statistics.transmission,
            groupby=groupby,
            aggregate_time=self._time_aggregation,
            plot_kwargs=kwargs,
        )

    def capacity_factor(
        self: BasePlotTypeAccessor, groupby: Sequence[str] = ["carrier"], **kwargs: Any
    ) -> Any:
        """Plot capacity factor"""
        return self._process_data(
            self._statistics.capacity_factor,
            groupby=groupby,
            aggregate_time=self._time_aggregation,
            plot_kwargs=kwargs,
        )

    def curtailment(
        self: BasePlotTypeAccessor, groupby: Sequence[str] = ["carrier"], **kwargs: Any
    ) -> Any:
        """Plot curtailment"""
        return self._process_data(
            self._statistics.curtailment,
            groupby=groupby,
            aggregate_time=self._time_aggregation,
            plot_kwargs=kwargs,
        )

    def capex(
        self: BasePlotTypeAccessor, groupby: Sequence[str] = ["carrier"], **kwargs: Any
    ) -> Any:
        """Plot capital expenditure"""
        return self._process_data(
            self._statistics.capex, groupby=groupby, plot_kwargs=kwargs
        )

    def opex(
        self: BasePlotTypeAccessor, groupby: Sequence[str] = ["carrier"], **kwargs: Any
    ) -> Any:
        """Plot operational expenditure"""
        return self._process_data(
            self._statistics.opex,
            groupby=groupby,
            aggregate_time=self._time_aggregation,
            plot_kwargs=kwargs,
        )

    def revenue(
        self: BasePlotTypeAccessor, groupby: Sequence[str] = ["carrier"], **kwargs: Any
    ) -> Any:
        """Plot revenue"""
        return self._process_data(
            self._statistics.revenue,
            groupby=groupby,
            aggregate_time=self._time_aggregation,
            plot_kwargs=kwargs,
        )

    def market_value(
        self: BasePlotTypeAccessor, groupby: Sequence[str] = ["carrier"], **kwargs: Any
    ) -> Any:
        """Plot market value"""
        return self._process_data(
            self._statistics.market_value,
            groupby=groupby,
            aggregate_time=self._time_aggregation,
            plot_kwargs=kwargs,
        )


class BarPlotter(BasePlotTypeAccessor):
    """Bar plot-specific implementation"""

    _default_orientation: ClassVar[str] = "vertical"

    def __init__(self: BarPlotter, n: Network) -> None:
        super().__init__(n)
        self._time_aggregation = "sum"

    def _validate(self: BarPlotter, data: pd.DataFrame) -> pd.DataFrame:
        """Implement bar-specific data validation"""
        if data.index.nlevels < 1:
            raise ValueError("Data must have at least one index level for bar plots")
        return data

    def _plot(
        self: BarPlotter,
        data: pd.DataFrame,
        stacked: bool = False,
        orientation: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Implement bar plotting logic with seaborn.objects"""
        orientation = orientation or self._default_orientation
        x = "value" if orientation == "horizontal" else "carrier"
        y = "carrier" if orientation == "horizontal" else "value"

        plot = self._create_base_plot(data, x=x, y=y)

        if "bus_carrier" in data.columns:
            plot = plot.facet(col="bus_carrier")

        transforms = [so.Agg()]
        if stacked:
            transforms.append(so.Dodge())

        return plot.add(so.Bar(), *transforms).label(x=x.title(), y=y.title()).plot()


class LinePlotter(BasePlotTypeAccessor):
    """Line plot-specific implementation"""

    _default_resample: ClassVar[str | None] = None

    def __init__(self: LinePlotter, n: Network) -> None:
        super().__init__(n)
        self._time_aggregation = False

    def _validate(self: LinePlotter, data: pd.DataFrame) -> pd.DataFrame:
        """Implement data validation for line plots"""
        # For time series data, ensure datetime index
        if "snapshot" in data.columns:
            try:
                data["snapshot"] = pd.to_datetime(data["snapshot"])
            except (ValueError, TypeError):
                pass
        return data

    def _plot(
        self: LinePlotter,
        data: pd.DataFrame,
        resample: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Implement line plotting logic with seaborn.objects"""
        if resample:
            data = data.set_index("snapshot").resample(resample).mean().reset_index()

        plot = self._create_base_plot(data, x="snapshot", y="value")

        if "bus_carrier" in data.columns:
            plot = plot.facet(col="bus_carrier")

        return plot.add(so.Line()).label(x="Time", y="Value").plot()


class AreaPlotter(BasePlotTypeAccessor):
    """Area plot-specific implementation"""

    _default_resample: ClassVar[str | None] = None
    _default_stacked: ClassVar[bool] = True

    def __init__(self: AreaPlotter, n: Network) -> None:
        super().__init__(n)
        self._time_aggregation = False

    def _validate(self: AreaPlotter, data: pd.DataFrame) -> pd.DataFrame:
        """Implement data validation for area plots"""
        # For time series data, ensure datetime index
        if "snapshot" in data.columns:
            try:
                data["snapshot"] = pd.to_datetime(data["snapshot"])
            except (ValueError, TypeError):
                pass
        return data

    def _plot(
        self: AreaPlotter,
        data: pd.DataFrame,
        resample: str | None = None,
        stacked: bool | None = None,
        **kwargs: Any,
    ) -> Any:
        """Implement area plotting logic with seaborn.objects"""
        resample = resample or self._default_resample
        stacked = stacked if stacked is not None else self._default_stacked

        if resample:
            data = data.set_index("snapshot").resample(resample).mean().reset_index()

        plot = self._create_base_plot(data, x="snapshot", y="value")

        if "bus_carrier" in data.columns:
            plot = plot.facet(col="bus_carrier")

        return (
            plot.add(so.Area(), so.Stack() if stacked else None)
            .label(x="", y="Value")
            .plot()
        )


class PlotAccessor:
    """Main plot accessor providing access to different plot types"""

    _n: Network
    maps: MapPlotter
    bar: BarPlotter
    line: LinePlotter
    area: AreaPlotter

    def __init__(self: PlotAccessor, n: Network) -> None:
        self._n = n
        self.maps = MapPlotter(n)
        self.bar = BarPlotter(n)
        self.line = LinePlotter(n)
        self.area = AreaPlotter(n)

    @wraps(plot)
    def __call__(self: PlotAccessor, *args: Any, **kwargs: Any) -> Any:
        """Default plot method (maps)"""
        return plot(self._n, *args, **kwargs)
