from __future__ import annotations

from collections.abc import Callable, Sequence
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

if TYPE_CHECKING:
    from pypsa import Network


class BasePlotTypeAccessor:
    """Base class containing shared statistics methods"""

    _network: Network
    _statistics: StatisticsAccessor
    _time_aggregation: str | bool
    _default_static_x: ClassVar[str] = "carrier"  # Default for static plots
    _default_dynamic_x: ClassVar[str] = "carrier"  # Default for time series plots

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

    def _to_title(self: BasePlotTypeAccessor, s: str) -> str:
        """Convert string to title case"""
        return s.replace("_", " ").title()

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
        default_colors = {"-": "gray", None: "gray", "": "#00000000"}
        return {**default_colors, **colors}

    def _get_carrier_labels(self, nice_names: bool = True) -> dict:
        """Get mapping of carrier names to nice names if requested"""
        if nice_names:
            names = self._network.carriers.nice_name
            return dict(names[names != ""])
        return {}

    def _process_data_for_stacking(
        self: BasePlotTypeAccessor,
        data: pd.DataFrame,
        stacked_dim: str,
    ) -> pd.DataFrame:
        """
        This function processes the data for correctly stacking positive and negative values.

        In the default seaborn implementation, stacking is done by adding the contributions
        on top of each other. This means that negative values are added on top of positive
        values. In our case, we want negative values to be stacked on the negative side of
        the x/y-axis. This function firstly sorts the values of the data and then assigns a
        new negative contribution equal to the sum of all positive values. This new negative
        contribution is drawn as a transparent block. This way, the negative values are
        stacked on the negative side of the x/y-axis.
        """
        if stacked_dim not in data.columns:
            raise ValueError(f"Column {stacked_dim} not found in data")

        # Get the sum of all positive values
        remaining_columns = [
            c for c in data.columns if c != "value" and c != stacked_dim
        ]
        if not remaining_columns:
            return data

        balancing_contribution = (
            data[data["value"] > 0]
            .drop(columns=stacked_dim)
            .groupby(remaining_columns, as_index=False)
            .agg({"value": "sum"})
            .assign(**{stacked_dim: ""})
            .assign(value=lambda x: -x["value"])
        )

        return pd.concat(
            [data[data["value"] > 0], balancing_contribution, data[data["value"] < 0]]
        )

    def _base_plot(
        self: BasePlotTypeAccessor,
        data: pd.DataFrame,
        x: str,
        y: str,
        color: str | None = None,
        col: str | None = None,
        row: str | None = None,
        stacked: bool = False,
        nice_names: bool = True,
        query: str | None = None,
        **kwargs: Any,
    ) -> so.Plot:
        """Plot method to be implemented by subclasses"""
        self._check_plotting_consistency()
        ldata = self._to_long_format(data)
        if query:
            ldata = ldata.query(query)
        if stacked and color is not None:
            ldata = self._process_data_for_stacking(ldata, color)
        ldata = self._validate(ldata)

        plot = so.Plot(ldata, x=x, y=y, color=color, **kwargs)

        # Apply color scale if using carrier colors
        if color in ["carrier", "bus_carrier"]:
            colors = self._get_carrier_colors()
            plot = plot.scale(color=so.Nominal(colors))
            if nice_names:
                labels = self._get_carrier_labels(nice_names=nice_names)
                plot = plot.scale(labels=so.Nominal(labels))

        # Apply faceting if col/row specified
        if col is not None:
            plot = plot.facet(col=col)
        if row is not None:
            plot = plot.facet(row=row)
        if col is not None or row is not None:
            plot = plot.share(x=False, y=False)

        return plot

    def _plot(
        self: BasePlotTypeAccessor,
        data: pd.DataFrame,
        x: str,
        y: str,
        color: str | None = None,
        col: str | None = None,
        row: str | None = None,
        stacked: bool = False,
        dodged: bool = False,
        nice_names: bool = True,
        query: str | None = None,
        **kwargs: Any,
    ) -> so.Plot:
        """Plot method to be implemented by subclasses"""
        raise NotImplementedError

    def _derive_statistic_parameters(
        self,
        *args: str | None,  # Changed to tuple argument
        stats_opts: dict[str, Any] = {},
        support_snapshot: bool = True,
    ) -> tuple[str | Sequence[str] | Callable, bool, bool | str]:
        """
        Extract plotting specification rules including groupby columns and component aggregation.

        Parameters
        ----------
        *args : tuple of (str | None)
            Arguments representing x, y, color, col, row parameters

        Returns
        -------
        tuple
            List of groupby columns and boolean for component aggregation
        """
        if not support_snapshot and "snapshot" in args:
            raise ValueError(
                "'snapshot' level is not supported for this plot function."
            )

        filtered = ["value", "component", "snapshot"]
        filtered_cols = []
        for c in args:  # Iterate through the args tuple
            if c not in filtered and c is not None:
                filtered_cols.append(c)
        derived_groupby = list(set(filtered_cols))
        derived_agg_across = "component" not in args  # Check in args tuple
        derived_agg_time: str | bool = "snapshot" not in args  # Check in args tuple
        if derived_agg_time:
            derived_agg_time = "sum"

        # Use derived values only if not explicitly specified
        groupby = stats_opts.get("groupby", derived_groupby)
        aggregate_across_components = stats_opts.get(
            "aggregate_across_components", derived_agg_across
        )
        aggregate_time = stats_opts.get("aggregate_time", derived_agg_time)

        return groupby, aggregate_across_components, aggregate_time

    # The following functions in this class are all Front-end plotting methods
    def optimal_capacity(
        self: BasePlotTypeAccessor,
        x: str | None = None,  # Changed to optional
        y: str = "value",
        color: str | None = "carrier",
        col: str | None = None,
        row: str | None = None,
        nice_names: bool = True,
        storage: bool = False,
        query: str | None = None,
        stats_opts: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> so.Plot:
        """Plot optimal capacity"""
        x = x or self._default_static_x  # Static plot
        stats_opts = stats_opts or {}
        groupby, aggregate_across_components, aggregate_time = (
            self._derive_statistic_parameters(
                x, y, color, col, row, stats_opts=stats_opts, support_snapshot=False
            )
        )
        data = self._statistics.optimal_capacity(
            comps=stats_opts.get("comps"),
            groupby=groupby,
            aggregate_across_components=aggregate_across_components,
            bus_carrier=stats_opts.get("bus_carrier"),
            storage=storage,
            nice_names=False,
        )
        return self._plot(
            data,
            x=x,
            y=y,
            color=color,
            col=col,
            row=row,
            nice_names=nice_names,
            query=query,
            **kwargs,
        )

    def installed_capacity(
        self: BasePlotTypeAccessor,
        x: str | None = None,  # Changed to optional
        y: str = "value",
        color: str | None = "carrier",
        col: str | None = None,
        row: str | None = None,
        nice_names: bool = True,
        storage: bool = False,
        query: str | None = None,
        stats_opts: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> so.Plot:
        """Plot installed capacity"""
        x = x or self._default_static_x  # Static plot
        stats_opts = stats_opts or {}
        groupby, aggregate_across_components, aggregate_time = (
            self._derive_statistic_parameters(
                x, y, color, col, row, stats_opts=stats_opts, support_snapshot=False
            )
        )
        data = self._statistics.installed_capacity(
            comps=stats_opts.get("comps"),
            groupby=groupby,
            aggregate_across_components=aggregate_across_components,
            bus_carrier=stats_opts.get("bus_carrier"),
            storage=storage,
            nice_names=False,
        )
        return self._plot(
            data,
            x=x,
            y=y,
            color=color,
            col=col,
            row=row,
            nice_names=nice_names,
            query=query,
            **kwargs,
        )

    def supply(
        self: BasePlotTypeAccessor,
        x: str | None = None,  # Changed to optional
        y: str = "value",
        color: str | None = "carrier",
        col: str | None = None,
        row: str | None = None,
        nice_names: bool = True,
        query: str | None = None,
        stats_opts: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> so.Plot:
        """Plot supply data"""
        x = x or self._default_dynamic_x  # Dynamic plot
        stats_opts = stats_opts or {}
        groupby, aggregate_across_components, aggregate_time = (
            self._derive_statistic_parameters(
                x, y, color, col, row, stats_opts=stats_opts, support_snapshot=True
            )
        )
        data = self._statistics.supply(
            comps=stats_opts.get("comps"),
            groupby=groupby,
            aggregate_time=aggregate_time,
            aggregate_across_components=aggregate_across_components,
            bus_carrier=stats_opts.get("bus_carrier"),
            nice_names=False,
        )
        return self._plot(
            data,
            x=x,
            y=y,
            color=color,
            col=col,
            row=row,
            nice_names=nice_names,
            query=query,
            **kwargs,
        )

    def withdrawal(
        self: BasePlotTypeAccessor,
        x: str | None = None,  # Changed to optional
        y: str = "value",
        color: str | None = "carrier",
        col: str | None = None,
        row: str | None = None,
        nice_names: bool = True,
        query: str | None = None,
        stats_opts: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> so.Plot:
        """Plot withdrawal data"""
        x = x or self._default_dynamic_x  # Dynamic plot
        stats_opts = stats_opts or {}
        groupby, aggregate_across_components, aggregate_time = (
            self._derive_statistic_parameters(
                x, y, color, col, row, stats_opts=stats_opts, support_snapshot=True
            )
        )
        data = self._statistics.withdrawal(
            comps=stats_opts.get("comps"),
            groupby=groupby,
            aggregate_time=aggregate_time,
            aggregate_across_components=aggregate_across_components,
            bus_carrier=stats_opts.get("bus_carrier"),
            nice_names=False,
        )
        return self._plot(
            data,
            x=x,
            y=y,
            color=color,
            col=col,
            row=row,
            nice_names=nice_names,
            query=query,
            **kwargs,
        )

    def energy_balance(
        self: BasePlotTypeAccessor,
        x: str | None = None,  # Changed to optional
        y: str = "value",
        color: str | None = "carrier",
        col: str | None = None,
        row: str | None = None,
        nice_names: bool = True,
        query: str | None = None,
        stats_opts: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> so.Plot:
        """Plot energy balance"""
        x = x or self._default_dynamic_x  # Dynamic plot
        stats_opts = stats_opts or {}
        groupby, aggregate_across_components, aggregate_time = (
            self._derive_statistic_parameters(
                x, y, color, col, row, stats_opts=stats_opts, support_snapshot=True
            )
        )
        data = self._statistics.energy_balance(
            comps=stats_opts.get("comps"),
            groupby=groupby,
            aggregate_time=aggregate_time,
            aggregate_across_components=aggregate_across_components,
            bus_carrier=stats_opts.get("bus_carrier"),
            nice_names=False,
        )
        return self._plot(
            data,
            x=x,
            y=y,
            color=color,
            col=col,
            row=row,
            nice_names=nice_names,
            query=query,
            **kwargs,
        )

    def transmission(
        self: BasePlotTypeAccessor,
        x: str | None = None,  # Changed to optional
        y: str = "value",
        color: str | None = "carrier",
        col: str | None = None,
        row: str | None = None,
        nice_names: bool = True,
        query: str | None = None,
        stats_opts: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> so.Plot:
        """Plot transmission data"""
        x = x or self._default_dynamic_x  # Dynamic plot
        stats_opts = stats_opts or {}
        groupby, aggregate_across_components, aggregate_time = (
            self._derive_statistic_parameters(
                x, y, color, col, row, stats_opts=stats_opts, support_snapshot=True
            )
        )
        data = self._statistics.transmission(
            groupby=groupby,
            aggregate_time=aggregate_time,
            aggregate_across_components=aggregate_across_components,
            bus_carrier=stats_opts.get("bus_carrier"),
            nice_names=False,
        )
        return self._plot(
            data,
            x=x,
            y=y,
            color=color,
            col=col,
            row=row,
            nice_names=nice_names,
            query=query,
            **kwargs,
        )

    def capacity_factor(
        self: BasePlotTypeAccessor,
        x: str | None = None,  # Changed to optional
        y: str = "value",
        color: str | None = "carrier",
        col: str | None = None,
        row: str | None = None,
        nice_names: bool = True,
        query: str | None = None,
        stats_opts: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> so.Plot:
        """Plot capacity factor"""
        x = x or self._default_dynamic_x  # Dynamic plot
        stats_opts = stats_opts or {}
        groupby, aggregate_across_components, aggregate_time = (
            self._derive_statistic_parameters(
                x, y, color, col, row, stats_opts=stats_opts, support_snapshot=True
            )
        )
        data = self._statistics.capacity_factor(
            groupby=groupby,
            aggregate_time=aggregate_time,
            aggregate_across_components=aggregate_across_components,
            bus_carrier=stats_opts.get("bus_carrier"),
            nice_names=False,
        )
        return self._plot(
            data,
            x=x,
            y=y,
            color=color,
            col=col,
            row=row,
            nice_names=nice_names,
            query=query,
            **kwargs,
        )

    def curtailment(
        self: BasePlotTypeAccessor,
        x: str | None = None,  # Changed to optional
        y: str = "value",
        color: str | None = "carrier",
        col: str | None = None,
        row: str | None = None,
        nice_names: bool = True,
        query: str | None = None,
        stats_opts: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> so.Plot:
        """Plot curtailment"""
        x = x or self._default_dynamic_x  # Dynamic plot
        stats_opts = stats_opts or {}
        groupby, aggregate_across_components, aggregate_time = (
            self._derive_statistic_parameters(
                x, y, color, col, row, stats_opts=stats_opts, support_snapshot=True
            )
        )
        data = self._statistics.curtailment(
            groupby=groupby,
            aggregate_time=aggregate_time,
            aggregate_across_components=aggregate_across_components,
            bus_carrier=stats_opts.get("bus_carrier"),
            nice_names=False,
        )
        return self._plot(
            data,
            x=x,
            y=y,
            color=color,
            col=col,
            row=row,
            nice_names=nice_names,
            query=query,
            **kwargs,
        )

    def capex(
        self: BasePlotTypeAccessor,
        x: str | None = None,  # Changed to optional
        y: str = "value",
        color: str | None = "carrier",
        col: str | None = None,
        row: str | None = None,
        nice_names: bool = True,
        query: str | None = None,
        stats_opts: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> so.Plot:
        """Plot capital expenditure"""
        x = x or self._default_static_x  # Static plot
        stats_opts = stats_opts or {}
        groupby, aggregate_across_components, aggregate_time = (
            self._derive_statistic_parameters(
                x, y, color, col, row, stats_opts=stats_opts, support_snapshot=False
            )
        )
        data = self._statistics.capex(
            groupby=groupby,
            aggregate_across_components=aggregate_across_components,
            bus_carrier=stats_opts.get("bus_carrier"),
            nice_names=False,
        )
        return self._plot(
            data,
            x=x,
            y=y,
            color=color,
            col=col,
            row=row,
            nice_names=nice_names,
            query=query,
            **kwargs,
        )

    def opex(
        self: BasePlotTypeAccessor,
        x: str | None = None,  # Changed to optional
        y: str = "value",
        color: str | None = "carrier",
        col: str | None = None,
        row: str | None = None,
        nice_names: bool = True,
        query: str | None = None,
        stats_opts: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> so.Plot:
        """Plot operational expenditure"""
        x = x or self._default_dynamic_x  # Dynamic plot
        stats_opts = stats_opts or {}
        groupby, aggregate_across_components, aggregate_time = (
            self._derive_statistic_parameters(
                x, y, color, col, row, stats_opts=stats_opts, support_snapshot=True
            )
        )
        data = self._statistics.opex(
            groupby=groupby,
            aggregate_time=aggregate_time,
            aggregate_across_components=aggregate_across_components,
            bus_carrier=stats_opts.get("bus_carrier"),
            nice_names=False,
        )
        return self._plot(
            data,
            x=x,
            y=y,
            color=color,
            col=col,
            row=row,
            nice_names=nice_names,
            query=query,
            **kwargs,
        )

    def revenue(
        self: BasePlotTypeAccessor,
        x: str | None = None,  # Changed to optional
        y: str = "value",
        color: str | None = "carrier",
        col: str | None = None,
        row: str | None = None,
        nice_names: bool = True,
        query: str | None = None,
        stats_opts: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> so.Plot:
        """Plot revenue"""
        x = x or self._default_dynamic_x  # Dynamic plot
        stats_opts = stats_opts or {}
        groupby, aggregate_across_components, aggregate_time = (
            self._derive_statistic_parameters(
                x, y, color, col, row, stats_opts=stats_opts, support_snapshot=True
            )
        )
        data = self._statistics.revenue(
            groupby=groupby,
            aggregate_time=aggregate_time,
            aggregate_across_components=aggregate_across_components,
            bus_carrier=stats_opts.get("bus_carrier"),
            nice_names=False,
        )
        return self._plot(
            data,
            x=x,
            y=y,
            color=color,
            col=col,
            row=row,
            nice_names=nice_names,
            query=query,
            **kwargs,
        )

    def expanded_capacity(
        self: BasePlotTypeAccessor,
        x: str | None = None,  # Changed to optional
        y: str = "value",
        color: str | None = "carrier",
        col: str | None = None,
        row: str | None = None,
        nice_names: bool = True,
        query: str | None = None,
        stats_opts: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> so.Plot:
        """Plot expanded capacity"""
        x = x or self._default_static_x  # Static plot
        stats_opts = stats_opts or {}
        groupby, aggregate_across_components, aggregate_time = (
            self._derive_statistic_parameters(
                x, y, color, col, row, stats_opts=stats_opts, support_snapshot=False
            )
        )
        data = self._statistics.expanded_capacity(
            comps=stats_opts.get("comps"),
            groupby=groupby,
            aggregate_across_components=aggregate_across_components,
            bus_carrier=stats_opts.get("bus_carrier"),
            nice_names=False,
        )
        return self._plot(
            data,
            x=x,
            y=y,
            color=color,
            col=col,
            row=row,
            nice_names=nice_names,
            query=query,
            **kwargs,
        )

    def expanded_capex(
        self: BasePlotTypeAccessor,
        x: str | None = None,  # Changed to optional
        y: str = "value",
        color: str | None = "carrier",
        col: str | None = None,
        row: str | None = None,
        nice_names: bool = True,
        query: str | None = None,
        stats_opts: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> so.Plot:
        """Plot expanded capital expenditure"""
        x = x or self._default_static_x  # Static plot
        stats_opts = stats_opts or {}
        groupby, aggregate_across_components, aggregate_time = (
            self._derive_statistic_parameters(
                x, y, color, col, row, stats_opts=stats_opts, support_snapshot=False
            )
        )
        data = self._statistics.expanded_capex(
            comps=stats_opts.get("comps"),
            groupby=groupby,
            aggregate_across_components=aggregate_across_components,
            bus_carrier=stats_opts.get("bus_carrier"),
            nice_names=False,
        )
        return self._plot(
            data,
            x=x,
            y=y,
            color=color,
            col=col,
            row=row,
            nice_names=nice_names,
            query=query,
            **kwargs,
        )

    def market_value(
        self: BasePlotTypeAccessor,
        x: str | None = None,  # Changed to optional
        y: str = "value",
        color: str | None = "carrier",
        col: str | None = None,
        row: str | None = None,
        nice_names: bool = True,
        query: str | None = None,
        stats_opts: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> so.Plot:
        """Plot market value"""
        x = x or self._default_dynamic_x  # Dynamic plot
        stats_opts = stats_opts or {}
        groupby, aggregate_across_components, aggregate_time = (
            self._derive_statistic_parameters(
                x, y, color, col, row, stats_opts=stats_opts, support_snapshot=True
            )
        )
        data = self._statistics.market_value(
            comps=stats_opts.get("comps"),
            groupby=groupby,
            aggregate_time=aggregate_time,
            aggregate_across_components=aggregate_across_components,
            bus_carrier=stats_opts.get("bus_carrier"),
            nice_names=False,
        )
        return self._plot(
            data,
            x=x,
            y=y,
            color=color,
            col=col,
            row=row,
            nice_names=nice_names,
            query=query,
            **kwargs,
        )


class BarPlotter(BasePlotTypeAccessor):
    """Bar plot-specific implementation"""

    _default_orientation: ClassVar[str] = "vertical"
    _default_static_x: ClassVar[str] = "carrier"
    _default_dynamic_x: ClassVar[str] = "carrier"

    def __init__(self: BarPlotter, n: Network) -> None:
        super().__init__(n)
        self._time_aggregation = "sum"

    def _validate(self: BarPlotter, data: pd.DataFrame) -> pd.DataFrame:
        """Implement bar-specific data validation"""
        if data.index.nlevels < 1:
            raise ValueError("Data must have at least one index level for bar plots")
        return data

    def _plot(  # type: ignore
        self: BarPlotter,
        data: pd.DataFrame,
        x: str,  # Removed default
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
        """Implement bar plotting logic with seaborn.objects"""

        plot = self._base_plot(
            data,
            x=x,
            y=y,
            color=color,
            col=col,
            row=row,
            stacked=stacked,
            nice_names=nice_names,
            query=query,
            **kwargs,
        )

        transforms = []
        if stacked:
            transforms.append(so.Stack())
        if dodged:
            transforms.append(so.Dodge())

        return plot.add(so.Bar(), *transforms).label(
            x=self._to_title(x), y=self._to_title(y)
        )


class LinePlotter(BasePlotTypeAccessor):
    """Line plot-specific implementation"""

    _default_resample: ClassVar[str | None] = None
    _default_static_x: ClassVar[str] = "carrier"
    _default_dynamic_x: ClassVar[str] = "snapshot"

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

    def _plot(  # type: ignore
        self: LinePlotter,
        data: pd.DataFrame,
        x: str,  # Removed default
        y: str = "value",
        color: str | None = None,
        col: str | None = None,
        row: str | None = None,
        nice_names: bool = True,
        resample: str | None = None,
        query: str | None = None,
        **kwargs: Any,
    ) -> so.Plot:
        """Implement line plotting logic with seaborn.objects"""
        # Determine x-axis column
        if isinstance(data, pd.DataFrame) and set(data.columns).issubset(
            self._network.snapshots
        ):
            if resample:
                data = data.T.resample(resample).mean().T

        plot = self._base_plot(
            data,
            x=x,
            y=y,
            color=color,
            col=col,
            row=row,
            nice_names=nice_names,
            query=query,
            **kwargs,
        )

        return plot.add(so.Line()).label(x=self._to_title(x), y=self._to_title(y))


class AreaPlotter(BasePlotTypeAccessor):
    """Area plot-specific implementation"""

    _default_resample: ClassVar[str | None] = None
    _default_stacked: ClassVar[bool] = True
    _default_static_x: ClassVar[str] = "carrier"
    _default_dynamic_x: ClassVar[str] = "snapshot"

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

    def _plot(  # type: ignore
        self: AreaPlotter,
        data: pd.DataFrame,
        x: str,  # Removed default
        y: str = "value",
        color: str | None = None,
        col: str | None = None,
        row: str | None = None,
        nice_names: bool = True,
        stacked: bool = True,
        dodged: bool = False,
        query: str | None = None,
        **kwargs: Any,
    ) -> so.Plot:
        """Implement area plotting logic with seaborn.objects"""
        stacked = stacked if stacked is not None else self._default_stacked

        plot = self._base_plot(
            data,
            x=x,
            y="value",
            color=color,
            col=col,
            row=row,
            stacked=stacked,
            nice_names=nice_names,
            query=query,
            **kwargs,
        )

        transforms = []
        if stacked:
            transforms.append(so.Stack())

        return plot.add(so.Area(), *transforms).label(
            x=self._to_title(x), y=self._to_title(y)
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
        self._base = BasePlotTypeAccessor(n)
        self.maps = MapPlotter(n)
        self.bar = BarPlotter(n)
        self.line = LinePlotter(n)
        self.area = AreaPlotter(n)

    @wraps(plot)
    def __call__(self: PlotAccessor, *args: Any, **kwargs: Any) -> Any:
        """Default plot method (maps)"""
        return plot(self._n, *args, **kwargs)
