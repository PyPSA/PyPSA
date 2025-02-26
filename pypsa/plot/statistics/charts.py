"""Chart plots based on statistics functions (like bar, line, area)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

import pandas as pd
import seaborn.objects as so
from pandas.api.types import CategoricalDtype

from pypsa.consistency import (
    plotting_consistency_check,
)
from pypsa.plot.statistics.base import PlotsGenerator

if TYPE_CHECKING:
    pass


class ChartGenerator(PlotsGenerator, ABC):
    """Base class for generating charts based on statistics functions."""

    @abstractmethod
    def plot(
        self,
        data: pd.DataFrame,
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
        """Plot method to return final plot."""
        pass

    def _to_title(self, s: str) -> str:
        """Convert string to title case."""
        return s.replace("_", " ").title()

    def _validate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate data has required columns and types."""
        if "value" not in data.columns:
            raise ValueError("Data must contain 'value' column")

        # Convert object columns to category for better performance
        for col in data.columns:
            if isinstance(data[col].dtype, object | str) and not isinstance(
                data[col].dtype, CategoricalDtype
            ):
                data = data.assign(**{col: data[col].astype("category")})

        return data

    def _to_long_format(self, data: pd.DataFrame | pd.Series) -> pd.DataFrame:
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

    def _get_carrier_colors(self) -> dict:
        """Get colors for carrier data with default gray colors."""
        colors = self._n.carriers.color.copy()
        # Always include default gray colors
        default_colors = {"-": "gray", None: "gray", "": "#00000000"}
        return {**default_colors, **colors}

    def _get_carrier_labels(self, nice_names: bool = True) -> dict:
        """Get mapping of carrier names to nice names if requested."""
        if nice_names:
            names = self._n.carriers.nice_name
            return dict(names[names != ""])
        return {}

    def _process_data_for_stacking(
        self,
        data: pd.DataFrame,
        stacked_dim: str,
    ) -> pd.DataFrame:
        """
        Process data to correctly stack positive and negative values.

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
        self,
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
        """Plot method to be implemented by subclasses."""
        plotting_consistency_check(self._n)
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

    def add_default_kwargs(
        self,
        *args: str | None,
        stats_kwargs: dict | None = None,  # make required
        method_name: str | None = None,  # make required
    ) -> dict[str, Any]:
        """
        Extract plotting specification rules including groupby columns and component aggregation.

        Parameters
        ----------
        *args : tuple of (str | None)
            Arguments representing x, y, color, col, row parameters
        stats_kwargs : dict, optional
            Keyword arguments from statistics function which will be written over
        method_name : str, optional
            Name of the statistics function to allow for specific rules

        Returns
        -------
        tuple
            List of groupby columns and boolean for component aggregation

        """
        # TODO Dynamic vs static default x must be handled in the plotter

        no_time_support = [
            "optimal_capacity",
            "installed_capacity",
            "capex",
            "installed_capex",
            "expanded_capacity",
            "expanded_capex",
        ]
        if "snapshot" in args and method_name in no_time_support:
            raise ValueError(
                "'snapshot' level is not supported for this plot function."
            )

        filtered = ["value", "component", "snapshot"]
        filtered_cols = []
        for c in args:  # Iterate through the args tuple
            if c not in filtered and c is not None:
                filtered_cols.append(c)

        if "groupby" not in stats_kwargs:  # type: ignore
            stats_kwargs["groupby"] = list(set(filtered_cols))  # type: ignore

        if "aggregate_across_components" not in stats_kwargs:  # type: ignore
            stats_kwargs["aggregate_across_components"] = "component" not in args  # type: ignore

        if "aggregate_time" not in stats_kwargs and method_name not in no_time_support:  # type: ignore
            derived_agg_time: str | bool = "snapshot" not in args  # Check in args tuple
            if derived_agg_time:
                derived_agg_time = "sum"
            stats_kwargs["aggregate_time"] = derived_agg_time  # type: ignore

        return stats_kwargs  # type: ignore


class BarPlotGenerator(ChartGenerator):
    """Bar plot-specific implementation."""

    _default_orientation: ClassVar[str] = "vertical"
    _default_statChartGenerator = "carrier"
    _default_dynamic_x: ClassVar[str] = "carrier"
    time_aggregation: ClassVar[str | bool] = "sum"

    def _validate(self: BarPlotGenerator, data: pd.DataFrame) -> pd.DataFrame:
        """Implement bar-specific data validation."""
        if data.index.nlevels < 1:
            raise ValueError("Data must have at least one index level for bar plots")
        return data

    def plot(  # type: ignore
        self,
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
        """Implement bar plotting logic with seaborn.objects."""
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


class LinePlotGenerator(ChartGenerator):
    """Line plot-specific implementation."""

    _default_resample: ClassVar[str | None] = None
    _default_static_x: ClassVar[str] = "carrier"
    _default_dynamic_x: ClassVar[str] = "snapshot"
    time_aggregation: ClassVar[str | bool] = False

    def _validate(self: LinePlotGenerator, data: pd.DataFrame) -> pd.DataFrame:
        """Implement data validation for line plots."""
        # For time series data, ensure datetime index
        if "snapshot" in data.columns:
            try:
                data["snapshot"] = pd.to_datetime(data["snapshot"])
            except (ValueError, TypeError):
                pass
        return data

    def plot(
        self,
        data: pd.DataFrame,
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
        """Implement line plotting logic with seaborn.objects."""
        # Determine x-axis column
        if isinstance(data, pd.DataFrame) and set(data.columns).issubset(
            self._n.snapshots
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


class AreaPlotGenerator(ChartGenerator):
    """Area plot-specific implementation."""

    _default_resample: ClassVar[str | None] = None
    _default_stacked: ClassVar[bool] = True
    _default_static_x: ClassVar[str] = "carrier"
    _default_dynamic_x: ClassVar[str] = "snapshot"
    time_aggregation: ClassVar[str | bool] = False

    def _validate(self: AreaPlotGenerator, data: pd.DataFrame) -> pd.DataFrame:
        """Implement data validation for area plots."""
        # For time series data, ensure datetime index
        if "snapshot" in data.columns:
            try:
                data["snapshot"] = pd.to_datetime(data["snapshot"])
            except (ValueError, TypeError):
                pass
        return data

    def plot(  # type: ignore
        self,
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
        """Implement area plotting logic with seaborn.objects."""
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
