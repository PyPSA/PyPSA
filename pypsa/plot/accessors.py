from __future__ import annotations

from collections.abc import Sequence
from functools import wraps
from typing import TYPE_CHECKING, Any, ClassVar

import pandas as pd
import seaborn.objects as so
from pandas.api.types import CategoricalDtype

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
        """Validate data has required columns and types"""
        if "value" not in data.columns:
            raise ValueError("Data must contain 'value' column")

        # Convert object columns to category for better performance
        for col in data.columns:
            if isinstance(data[col].dtype, object | str) and not isinstance(
                data[col].dtype, CategoricalDtype
            ):
                data[col] = data[col].astype("category")

        return data

    def _check_plotting_consistency(self: BasePlotTypeAccessor) -> None:
        for c in self._network.iterate_components():
            check_for_unknown_carriers(self._network, c, strict=True)
        check_for_missing_carrier_colors(self._network, strict=True)

    def _get_carrier_colors(self) -> dict:
        """Get colors for carrier data with default gray colors"""
        colors = self._network.carriers.color.copy()
        # Always include default gray colors
        default_colors = {"-": "gray", "": "gray", None: "gray"}
        return {**default_colors, **colors}

    def _get_carrier_labels(self, nice_names: bool = True) -> dict:
        """Get mapping of carrier names to nice names if requested"""
        if nice_names:
            names = self._network.carriers.nice_name
            return dict(names[names != ""])
        return {}

    def _create_base_plot(
        self,
        data: pd.DataFrame,
        x: str,
        y: str,
        nice_names: bool = True,
        stacked: bool = False,
        **kwargs: Any,
    ) -> so.Plot:
        """Create base plot with dynamic color mapping and faceting"""
        # Remove x and y from categorical columns
        remaining_cols = [col for col in data.columns if col not in [x, y]]

        # Create base plot
        color_cols = {
            "carrier",
        }
        color = next((col for col in data.columns if col in color_cols), None)
        plot = so.Plot(data, x=x, y=y, color=color, **kwargs)

        colors = self._get_carrier_colors()
        plot = plot.scale(color=so.Nominal(colors))
        if nice_names:
            labels = self._get_carrier_labels(nice_names=nice_names)
            plot = plot.scale(labels=so.Nominal(labels))

        # Add color scale if we have categorical columns
        for i, col in enumerate(remaining_cols):
            if i % 2 == 0:
                plot = plot.facet(col=col)
            else:
                plot = plot.facet(row=col)
            plot = plot.share(x=False, y=False)

        return plot

    def _plot(self: BasePlotTypeAccessor, data: pd.DataFrame, **kwargs: Any) -> so.Plot:
        """Plot method to be implemented by subclasses"""
        return self._create_base_plot(data, **kwargs)

    def _process_data(
        self: BasePlotTypeAccessor,
        data: pd.DataFrame | pd.Series,
        nice_names: bool = True,
        **kwargs: Any,
    ) -> so.Plot:
        """Common data processing pipeline"""
        self._check_plotting_consistency()
        data = self._to_long_format(data)
        data = self._validate(data)
        return self._plot(data, nice_names=nice_names, **kwargs)

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
    ) -> so.Plot:
        """Plot optimal capacity"""
        data = self._statistics.optimal_capacity(
            comps=comps,
            groupby=groupby,
            aggregate_across_components=aggregate_across_components,
            bus_carrier=bus_carrier,
            storage=storage,
            nice_names=False,
        )
        return self._process_data(data, nice_names=nice_names, **kwargs)

    def installed_capacity(
        self: BasePlotTypeAccessor,
        comps: str | Sequence[str] | None = None,
        groupby: Sequence[str] = ["carrier"],
        aggregate_across_components: bool = True,
        bus_carrier: str | None = None,
        nice_names: bool = True,
        storage: bool = False,
        **kwargs: Any,
    ) -> so.Plot:
        """Plot installed capacity"""
        data = self._statistics.installed_capacity(
            comps=comps,
            groupby=groupby,
            aggregate_across_components=aggregate_across_components,
            bus_carrier=bus_carrier,
            storage=storage,
            nice_names=False,
        )
        return self._process_data(data, nice_names=nice_names, **kwargs)

    def supply(
        self: BasePlotTypeAccessor,
        comps: str | Sequence[str] | None = None,
        groupby: Sequence[str] = ["carrier"],
        aggregate_across_components: bool = True,
        bus_carrier: str | None = None,
        nice_names: bool = True,
        **kwargs: Any,
    ) -> so.Plot:
        """Plot supply data"""
        data = self._statistics.supply(
            comps=comps,
            groupby=groupby,
            aggregate_time=self._time_aggregation,
            aggregate_across_components=aggregate_across_components,
            bus_carrier=bus_carrier,
            nice_names=False,
        )
        return self._process_data(data, nice_names=nice_names, **kwargs)

    def withdrawal(
        self: BasePlotTypeAccessor,
        comps: str | Sequence[str] | None = None,
        groupby: Sequence[str] = ["carrier"],
        aggregate_across_components: bool = True,
        bus_carrier: str | None = None,
        nice_names: bool = True,
        **kwargs: Any,
    ) -> so.Plot:
        """Plot withdrawal data"""
        data = self._statistics.withdrawal(
            comps=comps,
            groupby=groupby,
            aggregate_time=self._time_aggregation,
            aggregate_across_components=aggregate_across_components,
            bus_carrier=bus_carrier,
            nice_names=False,
        )
        return self._process_data(data, nice_names=nice_names, **kwargs)

    def energy_balance(
        self: BasePlotTypeAccessor,
        comps: str | Sequence[str] | None = None,
        groupby: Sequence[str] = ["carrier", "bus_carrier"],
        aggregate_across_components: bool = True,
        bus_carrier: str | None = None,
        nice_names: bool = True,
        **kwargs: Any,
    ) -> so.Plot:
        """Plot energy balance"""
        data = self._statistics.energy_balance(
            comps=comps,
            groupby=groupby,
            aggregate_time=self._time_aggregation,
            aggregate_across_components=aggregate_across_components,
            bus_carrier=bus_carrier,
            nice_names=False,
        )
        return self._process_data(data, nice_names=nice_names, **kwargs)

    def transmission(
        self: BasePlotTypeAccessor,
        groupby: Sequence[str] = ["carrier"],
        aggregate_across_components: bool = True,
        bus_carrier: str | None = None,
        nice_names: bool = True,
        **kwargs: Any,
    ) -> so.Plot:
        """Plot transmission data"""
        data = self._statistics.transmission(
            groupby=groupby,
            aggregate_time=self._time_aggregation,
            aggregate_across_components=aggregate_across_components,
            bus_carrier=bus_carrier,
            nice_names=False,
        )
        return self._process_data(data, nice_names=nice_names, **kwargs)

    def capacity_factor(
        self: BasePlotTypeAccessor,
        groupby: Sequence[str] = ["carrier"],
        aggregate_across_components: bool = True,
        bus_carrier: str | None = None,
        nice_names: bool = True,
        **kwargs: Any,
    ) -> so.Plot:
        """Plot capacity factor"""
        data = self._statistics.capacity_factor(
            groupby=groupby,
            aggregate_time=self._time_aggregation,
            aggregate_across_components=aggregate_across_components,
            bus_carrier=bus_carrier,
            nice_names=False,
        )
        return self._process_data(data, nice_names=nice_names, **kwargs)

    def curtailment(
        self: BasePlotTypeAccessor,
        groupby: Sequence[str] = ["carrier"],
        aggregate_across_components: bool = True,
        bus_carrier: str | None = None,
        nice_names: bool = True,
        **kwargs: Any,
    ) -> so.Plot:
        """Plot curtailment"""
        data = self._statistics.curtailment(
            groupby=groupby,
            aggregate_time=self._time_aggregation,
            aggregate_across_components=aggregate_across_components,
            bus_carrier=bus_carrier,
            nice_names=False,
        )
        return self._process_data(data, nice_names=nice_names, **kwargs)

    def capex(
        self: BasePlotTypeAccessor,
        groupby: Sequence[str] = ["carrier"],
        aggregate_across_components: bool = True,
        bus_carrier: str | None = None,
        nice_names: bool = True,
        **kwargs: Any,
    ) -> so.Plot:
        """Plot capital expenditure"""
        data = self._statistics.capex(
            groupby=groupby,
            aggregate_across_components=aggregate_across_components,
            bus_carrier=bus_carrier,
            nice_names=False,
        )
        return self._process_data(data, nice_names=nice_names, **kwargs)

    def opex(
        self: BasePlotTypeAccessor,
        groupby: Sequence[str] = ["carrier"],
        aggregate_across_components: bool = True,
        bus_carrier: str | None = None,
        nice_names: bool = True,
        **kwargs: Any,
    ) -> so.Plot:
        """Plot operational expenditure"""
        data = self._statistics.opex(
            groupby=groupby,
            aggregate_time=self._time_aggregation,
            aggregate_across_components=aggregate_across_components,
            bus_carrier=bus_carrier,
            nice_names=False,
        )
        return self._process_data(data, nice_names=nice_names, **kwargs)

    def revenue(
        self: BasePlotTypeAccessor,
        groupby: Sequence[str] = ["carrier"],
        aggregate_across_components: bool = True,
        bus_carrier: str | None = None,
        nice_names: bool = True,
        **kwargs: Any,
    ) -> so.Plot:
        """Plot revenue"""
        data = self._statistics.revenue(
            groupby=groupby,
            aggregate_time=self._time_aggregation,
            aggregate_across_components=aggregate_across_components,
            bus_carrier=bus_carrier,
            nice_names=False,
        )
        return self._process_data(data, nice_names=nice_names, **kwargs)

    def expanded_capacity(
        self: BasePlotTypeAccessor,
        comps: str | Sequence[str] | None = None,
        groupby: Sequence[str] = ["carrier"],
        aggregate_across_components: bool = True,
        bus_carrier: str | None = None,
        nice_names: bool = True,
        **kwargs: Any,
    ) -> so.Plot:
        """Plot expanded capacity"""
        data = self._statistics.expanded_capacity(
            comps=comps,
            groupby=groupby,
            aggregate_across_components=aggregate_across_components,
            bus_carrier=bus_carrier,
            nice_names=False,
        )
        return self._process_data(data, nice_names=nice_names, **kwargs)

    def expanded_capex(
        self: BasePlotTypeAccessor,
        comps: str | Sequence[str] | None = None,
        groupby: Sequence[str] = ["carrier"],
        aggregate_across_components: bool = True,
        bus_carrier: str | None = None,
        nice_names: bool = True,
        **kwargs: Any,
    ) -> so.Plot:
        """Plot expanded capital expenditure"""
        data = self._statistics.expanded_capex(
            comps=comps,
            groupby=groupby,
            aggregate_across_components=aggregate_across_components,
            bus_carrier=bus_carrier,
            nice_names=False,
        )
        return self._process_data(data, nice_names=nice_names, **kwargs)

    def market_value(
        self: BasePlotTypeAccessor,
        comps: str | Sequence[str] | None = None,
        groupby: Sequence[str] = ["carrier"],
        aggregate_across_components: bool = True,
        bus_carrier: str | None = None,
        nice_names: bool = True,
        **kwargs: Any,
    ) -> so.Plot:
        """Plot market value"""
        data = self._statistics.market_value(
            comps=comps,
            groupby=groupby,
            aggregate_time=self._time_aggregation,
            aggregate_across_components=aggregate_across_components,
            bus_carrier=bus_carrier,
            nice_names=False,
        )
        return self._process_data(data, nice_names=nice_names, **kwargs)


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
    ) -> so.Plot:
        """Implement bar plotting logic with seaborn.objects"""
        orientation = orientation or self._default_orientation
        y_col = "value"
        x_col = next(col for col in data.columns if col != y_col)
        x = y_col if orientation == "horizontal" else x_col
        y = x_col if orientation == "horizontal" else y_col

        plot = self._create_base_plot(data, x=x, y=y, stacked=stacked)

        transforms = []
        if stacked:
            transforms.append(so.Stack())

        return plot.add(so.Bar(), *transforms).label(x=x.title(), y=y.title())


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
    ) -> so.Plot:
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

        return plot.add(so.Line()).label(x=x_col.title(), y="Value")


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
    ) -> so.Plot:
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

        return plot.add(so.Area(), *transforms).label(x=x_col.title(), y="Value")


class PlotAccessor:
    """Main plot accessor providing access to different plot types"""

    _n: Network
    maps: MapPlotter
    bar: BarPlotter
    line: LinePlotter
    area: AreaPlotter

    def __init__(self: PlotAccessor, n: Network) -> None:
        self._n = n
        self._base = BasePlotTypeAccessor(n)
        self.maps = MapPlotter(n)
        self.bar = BarPlotter(n)
        self.line = LinePlotter(n)
        self.area = AreaPlotter(n)

    @wraps(plot)
    def __call__(self: PlotAccessor, *args: Any, **kwargs: Any) -> Any:
        """Default plot method (maps)"""
        return plot(self._n, *args, **kwargs)
