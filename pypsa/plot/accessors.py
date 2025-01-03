from __future__ import annotations

from collections.abc import Sequence
from functools import wraps
from typing import TYPE_CHECKING, Any, ClassVar

import pandas as pd
import seaborn.objects as so

from pypsa.consistency import (
    check_for_missing_carrier_colors,
    check_for_unknown_carriers,
)
from pypsa.plot.maps import MapPlotter, plot
from pypsa.statistics.expressions import StatisticsAccessor
from pypsa.utils import resample_timeseries

if TYPE_CHECKING:
    from pypsa import Network


class BasePlotTypeAccessor:
    """Base class containing shared statistics methods"""

    _network: Network
    _statistics: StatisticsAccessor
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
            df = data.rename("value").reset_index()
        else:
            df = data.fillna(0).melt(ignore_index=False).reset_index()

        return df

    def _validate(self: BasePlotTypeAccessor, data: pd.DataFrame) -> pd.DataFrame:
        """Validation method to be implemented by subclasses"""
        return data

    def _check_plotting_consistency(self: BasePlotTypeAccessor) -> None:
        for c in self._network.iterate_components():
            check_for_unknown_carriers(self._network, c, strict=True)
        check_for_missing_carrier_colors(self._network, strict=True)

    def _get_carrier_colors(self, data: pd.DataFrame) -> dict:
        """Get colors for carriers from network.carriers.color"""
        if "carrier" in data.columns:
            colors = self._network.carriers.color.copy()
            colors["-"] = "gray"
            colors[""] = "gray"
            carriers = data["carrier"].dropna().unique()
            return dict(colors[carriers])
        return {}

    def _get_carrier_labels(self, nice_name: bool = True) -> dict | None:
        """Get mapping of carrier names to nice names if requested"""
        if nice_name:
            return dict(self._network.carriers[["nice_name"]].dropna().nice_name)
        return None

    def _create_base_plot(
        self, data: pd.DataFrame, x: str, y: str, nice_name: bool = True, **kwargs: Any
    ) -> so.Plot:
        """Create base plot with carrier colors and optional nice names"""
        colors = self._get_carrier_colors(data)
        labels = self._get_carrier_labels(nice_name)

        plot = (
            (
                so.Plot(data, x=x, y=y, color="carrier" if colors else None).scale(
                    color=so.Nominal(colors)
                )
            )
            if colors
            else so.Plot(data, x=x, y=y)
        )

        if labels:
            plot = plot.scale(labels=so.Nominal(labels))

        # Add faceting for component and bus_carrier if present
        if "component" in data.columns:
            plot = plot.facet(col="component")
        if "bus_carrier" in data.columns:
            plot = plot.facet(row="bus_carrier")

        return plot

    def _plot(self: BasePlotTypeAccessor, data: pd.DataFrame, **kwargs: Any) -> Any:
        """Plot method to be implemented by subclasses"""
        raise NotImplementedError

    def _process_data(
        self: BasePlotTypeAccessor, data: pd.DataFrame | pd.Series, **kwargs: Any
    ) -> Any:
        """Common data processing pipeline"""
        self._check_plotting_consistency()
        plot_kwargs = kwargs.pop("plot_kwargs", {})
        data = self._to_long_format(data)
        data = self._validate(data)
        return self._plot(data, **plot_kwargs)

    # Shared statistics methods
    def optimal_capacity(
        self: BasePlotTypeAccessor,
        comps: str | Sequence[str] | None = None,
        groupby: Sequence[str] = ["carrier"],
        aggregate_across_components: bool = True,
        bus_carrier: str | None = None,
        nice_names: bool = True,
        storage: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Plot optimal capacity"""
        data = self._statistics.optimal_capacity(
            comps=comps,
            groupby=groupby,
            aggregate_across_components=aggregate_across_components,
            bus_carrier=bus_carrier,
            nice_names=nice_names,
            storage=storage,
        )
        return self._process_data(data, plot_kwargs=kwargs)

    def installed_capacity(
        self: BasePlotTypeAccessor,
        comps: str | Sequence[str] | None = None,
        groupby: Sequence[str] = ["carrier"],
        aggregate_across_components: bool = True,
        bus_carrier: str | None = None,
        nice_names: bool = True,
        storage: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Plot installed capacity"""
        data = self._statistics.installed_capacity(
            comps=comps,
            groupby=groupby,
            aggregate_across_components=aggregate_across_components,
            bus_carrier=bus_carrier,
            nice_names=nice_names,
            storage=storage,
        )
        return self._process_data(data, plot_kwargs=kwargs)

    def supply(
        self: BasePlotTypeAccessor,
        comps: str | Sequence[str] | None = None,
        groupby: Sequence[str] = ["carrier"],
        aggregate_across_components: bool = True,
        bus_carrier: str | None = None,
        nice_names: bool = True,
        **kwargs: Any,
    ) -> Any:
        """Plot supply data"""
        data = self._statistics.supply(
            comps=comps,
            groupby=groupby,
            aggregate_time=self._time_aggregation,
            aggregate_across_components=aggregate_across_components,
            bus_carrier=bus_carrier,
            nice_names=nice_names,
        )
        return self._process_data(data, plot_kwargs=kwargs)

    def withdrawal(
        self: BasePlotTypeAccessor,
        comps: str | Sequence[str] | None = None,
        groupby: Sequence[str] = ["carrier"],
        aggregate_across_components: bool = True,
        bus_carrier: str | None = None,
        nice_names: bool = True,
        **kwargs: Any,
    ) -> Any:
        """Plot withdrawal data"""
        data = self._statistics.withdrawal(
            comps=comps,
            groupby=groupby,
            aggregate_time=self._time_aggregation,
            aggregate_across_components=aggregate_across_components,
            bus_carrier=bus_carrier,
            nice_names=nice_names,
        )
        return self._process_data(data, plot_kwargs=kwargs)

    def energy_balance(
        self: BasePlotTypeAccessor,
        comps: str | Sequence[str] | None = None,
        groupby: Sequence[str] = ["carrier", "bus_carrier"],
        aggregate_across_components: bool = True,
        bus_carrier: str | None = None,
        nice_names: bool = True,
        **kwargs: Any,
    ) -> Any:
        """Plot energy balance"""
        data = self._statistics.energy_balance(
            comps=comps,
            groupby=groupby,
            aggregate_time=self._time_aggregation,
            aggregate_across_components=aggregate_across_components,
            bus_carrier=bus_carrier,
            nice_names=nice_names,
        )
        return self._process_data(data, plot_kwargs=kwargs)

    def transmission(
        self: BasePlotTypeAccessor,
        groupby: Sequence[str] = ["carrier"],
        aggregate_across_components: bool = True,
        bus_carrier: str | None = None,
        nice_names: bool = True,
        **kwargs: Any,
    ) -> Any:
        """Plot transmission data"""
        data = self._statistics.transmission(
            groupby=groupby,
            aggregate_time=self._time_aggregation,
            aggregate_across_components=aggregate_across_components,
            bus_carrier=bus_carrier,
            nice_names=nice_names,
        )
        return self._process_data(data, plot_kwargs=kwargs)

    def capacity_factor(
        self: BasePlotTypeAccessor,
        groupby: Sequence[str] = ["carrier"],
        aggregate_across_components: bool = True,
        bus_carrier: str | None = None,
        nice_names: bool = True,
        **kwargs: Any,
    ) -> Any:
        """Plot capacity factor"""
        data = self._statistics.capacity_factor(
            groupby=groupby,
            aggregate_time=self._time_aggregation,
            aggregate_across_components=aggregate_across_components,
            bus_carrier=bus_carrier,
            nice_names=nice_names,
        )
        return self._process_data(data, plot_kwargs=kwargs)

    def curtailment(
        self: BasePlotTypeAccessor,
        groupby: Sequence[str] = ["carrier"],
        aggregate_across_components: bool = True,
        bus_carrier: str | None = None,
        nice_names: bool = True,
        **kwargs: Any,
    ) -> Any:
        """Plot curtailment"""
        data = self._statistics.curtailment(
            groupby=groupby,
            aggregate_time=self._time_aggregation,
            aggregate_across_components=aggregate_across_components,
            bus_carrier=bus_carrier,
            nice_names=nice_names,
        )
        return self._process_data(data, plot_kwargs=kwargs)

    def capex(
        self: BasePlotTypeAccessor,
        groupby: Sequence[str] = ["carrier"],
        aggregate_across_components: bool = True,
        bus_carrier: str | None = None,
        nice_names: bool = True,
        **kwargs: Any,
    ) -> Any:
        """Plot capital expenditure"""
        data = self._statistics.capex(
            groupby=groupby,
            aggregate_across_components=aggregate_across_components,
            bus_carrier=bus_carrier,
            nice_names=nice_names,
        )
        return self._process_data(data, plot_kwargs=kwargs)

    def opex(
        self: BasePlotTypeAccessor,
        groupby: Sequence[str] = ["carrier"],
        aggregate_across_components: bool = True,
        bus_carrier: str | None = None,
        nice_names: bool = True,
        **kwargs: Any,
    ) -> Any:
        """Plot operational expenditure"""
        data = self._statistics.opex(
            groupby=groupby,
            aggregate_time=self._time_aggregation,
            aggregate_across_components=aggregate_across_components,
            bus_carrier=bus_carrier,
            nice_names=nice_names,
        )
        return self._process_data(data, plot_kwargs=kwargs)

    def revenue(
        self: BasePlotTypeAccessor,
        groupby: Sequence[str] = ["carrier"],
        aggregate_across_components: bool = True,
        bus_carrier: str | None = None,
        nice_names: bool = True,
        **kwargs: Any,
    ) -> Any:
        """Plot revenue"""
        data = self._statistics.revenue(
            groupby=groupby,
            aggregate_time=self._time_aggregation,
            aggregate_across_components=aggregate_across_components,
            bus_carrier=bus_carrier,
            nice_names=nice_names,
        )
        return self._process_data(data, plot_kwargs=kwargs)

    def expanded_capacity(
        self: BasePlotTypeAccessor,
        comps: str | Sequence[str] | None = None,
        groupby: Sequence[str] = ["carrier"],
        aggregate_across_components: bool = True,
        bus_carrier: str | None = None,
        nice_names: bool = True,
        **kwargs: Any,
    ) -> Any:
        """Plot expanded capacity"""
        data = self._statistics.expanded_capacity(
            comps=comps,
            groupby=groupby,
            aggregate_across_components=aggregate_across_components,
            bus_carrier=bus_carrier,
            nice_names=nice_names,
        )
        return self._process_data(data, plot_kwargs=kwargs)

    def expanded_capex(
        self: BasePlotTypeAccessor,
        comps: str | Sequence[str] | None = None,
        groupby: Sequence[str] = ["carrier"],
        aggregate_across_components: bool = True,
        bus_carrier: str | None = None,
        nice_names: bool = True,
        **kwargs: Any,
    ) -> Any:
        """Plot expanded capital expenditure"""
        data = self._statistics.expanded_capex(
            comps=comps,
            groupby=groupby,
            aggregate_across_components=aggregate_across_components,
            bus_carrier=bus_carrier,
            nice_names=nice_names,
        )
        return self._process_data(data, plot_kwargs=kwargs)

    def market_value(
        self: BasePlotTypeAccessor,
        comps: str | Sequence[str] | None = None,
        groupby: Sequence[str] = ["carrier"],
        aggregate_across_components: bool = True,
        bus_carrier: str | None = None,
        nice_names: bool = True,
        **kwargs: Any,
    ) -> Any:
        """Plot market value"""
        data = self._statistics.market_value(
            comps=comps,
            groupby=groupby,
            aggregate_time=self._time_aggregation,
            aggregate_across_components=aggregate_across_components,
            bus_carrier=bus_carrier,
            nice_names=nice_names,
        )
        return self._process_data(data, plot_kwargs=kwargs)


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
        # Determine x-axis column
        if "snapshot" in data.columns:
            x_col = "snapshot"
            if resample:
                data = resample_timeseries(
                    data.set_index("snapshot"), resample
                ).reset_index()
        else:
            # Use first non-value column as x-axis
            x_col = next(col for col in data.columns if col != "value")

        plot = self._create_base_plot(data, x=x_col, y="value")

        return plot.add(so.Line()).label(x=x_col.title(), y="Value").plot()


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

        # Determine x-axis column
        if "snapshot" in data.columns:
            x_col = "snapshot"
            if resample:
                data = resample_timeseries(
                    data.set_index("snapshot"), resample
                ).reset_index()
        else:
            # Use first non-value column as x-axis
            x_col = next(col for col in data.columns if col != "value")

        plot = self._create_base_plot(data, x=x_col, y="value")

        transforms = []
        if stacked:
            transforms.append(so.Stack())

        return plot.add(so.Area(), *transforms).label(x=x_col.title(), y="Value").plot()


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
