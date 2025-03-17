"""Chart plots based on statistics functions (like bar, line, area)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pandas.api.types import CategoricalDtype

from pypsa.consistency import (
    plotting_consistency_check,
)
from pypsa.plot.statistics.base import PlotsGenerator

if TYPE_CHECKING:
    pass


def facet_iter(
    g: sns.FacetGrid,
    df: pd.DataFrame,
    facet_row: str | None,
    facet_col: str | None,
    split_by_sign: bool = False,
) -> Iterator[tuple[Axes, pd.DataFrame]]:
    """
    Generator function that yields (axis, filtered_data) for each facet in a FacetGrid.

    Parameters
    ----------
    g : seaborn.FacetGrid
        The FacetGrid instance to use
    df : pandas.DataFrame
        The DataFrame to filter for each facet
    facet_row : str, optional
        Column name used for creating row facets
    facet_col : str, optional
        Column name used for creating column facets
    split_by_sign : bool, optional
        Whether to split the data by sign (positive/negative) for the y-axis
        Default is False.

    Yields
    ------
    tuple : (axis, filtered_data)
        - axis: The matplotlib Axes object for the current facet
        - filtered_data: DataFrame filtered for the current facet

    """
    # Get the facet values
    row_vals = g.row_names if g.row_names else [None]
    col_vals = g.col_names if g.col_names else [None]

    for i, row_val in enumerate(row_vals):
        for j, col_val in enumerate(col_vals):
            # Get the axis for this facet
            if g.axes.ndim == 1:
                ax = g.axes[i if len(g.axes) > 1 else j]
            else:
                ax = g.axes[i, j]

            # Filter the data for this specific facet
            facet_data = df.copy()
            if facet_row is not None and row_val is not None:
                facet_data = facet_data[facet_data[facet_row] == row_val]
            if facet_col is not None and col_val is not None:
                facet_data = facet_data[facet_data[facet_col] == col_val]

            # Skip if no data
            if not facet_data.values.size:
                continue

            if split_by_sign:
                for clip in [{"lower": 0}, {"upper": 0}]:
                    yield ax, facet_data.assign(value=facet_data["value"].clip(**clip))
            else:
                # Yield the axis and the filtered data
                yield ax, facet_data


def map_dataframe_pandas_plot(
    g: sns.FacetGrid,
    df: pd.DataFrame,
    x: str,
    y: str,
    color: str | None,
    facet_row: str | None,
    facet_col: str | None,
    stacked: bool,
    kind: str = "area",
    **kwargs: Any,
) -> sns.FacetGrid:
    """
    Handle the creation of area or bar plots for FacetGrid.

    Parameters
    ----------
    g : seaborn.FacetGrid
        The FacetGrid instance to use
    df : pandas.DataFrame
        The DataFrame in long format to plot
    x : str
        Column name to use for x-axis
    y : str
        Column name to use for y-axis
    color : str, optional
        Column name to use for color encoding
    facet_row : str, optional
        Column name used for creating row facets
    facet_col : str, optional
        Column name used for creating column facets
    stacked : bool
        Whether to create stacked charts
    kind : str, optional
        Kind of plot to create ('area' or 'bar')
    **kwargs : additional keyword arguments
        Passed to plotting function

    """
    # Store the color palette from FacetGrid for consistent colors
    color_order = g.hue_names if hasattr(g, "hue_names") else None

    split_by_sign = df["value"].min() < 0 and df["value"].max() > 0

    for ax, facet_data in facet_iter(g, df, facet_row, facet_col, split_by_sign):
        if color is not None:
            # Pivot data to have x as index, color as columns, and y as values
            pivoted = facet_data.pivot(index=x, columns=color, values=y)

            # Ensure columns are ordered according to the hue order
            if color_order:
                # Get only the columns that exist in this facet
                available_cols = [c for c in color_order if c in pivoted.columns]
                if available_cols:  # Only reorder if we have columns
                    pivoted = pivoted[available_cols]

            # Plot with pandas - no legend to avoid duplicates
            pivoted.plot(kind=kind, ax=ax, stacked=stacked, legend=False, **kwargs)
        else:
            # Simple case: just plot y vs x (no color grouping)
            facet_data.plot(
                kind=kind, x=x, y=y, ax=ax, stacked=stacked, legend=False, **kwargs
            )

        # Sort out the supporting information
        g._update_legend_data(ax)

    g._finalize_grid([x, y])

    return g


class ChartGenerator(PlotsGenerator, ABC):
    """Base class for generating charts based on statistics functions."""

    @abstractmethod
    def plot(
        self,
        data: pd.DataFrame,
        x: str = "carrier",
        y: str = "value",
        color: str | None = None,
        facet_col: str | None = None,
        facet_row: str | None = None,
        nice_names: bool = True,
        resample: str | None = None,
        query: str | None = None,
        **kwargs: Any,
    ) -> tuple[Figure, Axes | np.ndarray, sns.FacetGrid]:
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

    def _base_plot(
        self,
        data: pd.DataFrame,
        kind: str,
        x: str,
        y: str,
        color: str | None = None,
        facet_col: str | None = None,
        facet_row: str | None = None,
        stacked: bool = False,
        nice_names: bool = True,
        query: str | None = None,
        **kwargs: Any,
    ) -> tuple[Figure, Axes | np.ndarray, sns.FacetGrid]:
        """Plot method to be implemented by subclasses."""
        plotting_consistency_check(self._n, strict="all")
        ldata = self._to_long_format(data)
        if query:
            ldata = ldata.query(query)
        ldata = self._validate(ldata)

        # Always use FacetGrid for consistency
        g = sns.FacetGrid(
            ldata,
            row=facet_row,
            col=facet_col,
            hue=color,
        )

        # Handle special case for area and bar plots
        if kind == "area":
            g = map_dataframe_pandas_plot(
                g,
                ldata,
                x,
                y,
                color,
                facet_row,
                facet_col,
                stacked,
                kind="area",
                **kwargs,
            )
        elif kind == "bar":
            g = map_dataframe_pandas_plot(
                g,
                ldata,
                x,
                y,
                color,
                facet_row,
                facet_col,
                stacked,
                kind="bar",
                **kwargs,
            )
        # Other plot types remain the same
        elif kind == "scatter":
            g.map_dataframe(sns.scatterplot, x=x, y=y, **kwargs)
        elif kind == "line":
            g.map_dataframe(sns.lineplot, x=x, y=y, **kwargs)
        elif kind == "bar":
            g.map_dataframe(sns.barplot, x=x, y=y, **kwargs)
        elif kind == "box":
            g.map_dataframe(sns.boxplot, x=x, y=y, **kwargs)
        elif kind == "violin":
            g.map_dataframe(sns.violinplot, x=x, y=y, **kwargs)
        elif kind == "histogram":
            if y is None:
                g.map_dataframe(sns.histplot, x=x, **kwargs)
            else:
                g.map_dataframe(sns.histplot, x=x, y=y, **kwargs)
        else:
            raise ValueError(f"Unsupported plot type: {kind}")

        # Add legend if color is specified (for non-area plots, area plots handle this separately)
        if color is not None:
            g.add_legend()

        # Get the figure and axes from the FacetGrid
        fig = g.fig
        ax = g.axes  # This will be a 2D array of axes objects

        # If there's only one subplot, return the single Axes object for convenience
        if ax.size == 1:
            ax = ax.flat[0]

        return fig, ax, g

    def derive_statistic_parameters(
        self,
        *args: str | None,
        method_name: str = "",  # make required
    ) -> dict[str, Any]:
        """
        Extract plotting specification rules including groupby columns and component aggregation.

        Parameters
        ----------
        *args : tuple of (str | None)
            Arguments representing x, y, color, facet_col, facet_row parameters
        method_name : str, optional
            Name of the statistics function to allow for specific rules

        Returns
        -------
        tuple
            List of groupby columns and boolean for component aggregation

        """
        filtered = ["value", "component", "snapshot"]
        filtered_cols = []
        for c in args:  # Iterate through the args tuple
            if c not in filtered and c is not None:
                filtered_cols.append(c)

        stats_kwargs: dict[str, str | bool | list] = {}

        # `groupby`
        filtered_cols = list(set(filtered_cols))  # Remove duplicates
        if filtered_cols:
            stats_kwargs["groupby"] = filtered_cols

        # `aggregate_across_components`
        stats_kwargs["aggregate_across_components"] = "component" not in args

        # `aggregate_time` is only relevant for time series data
        if "snapshot" in args:
            derived_agg_time: str | bool = "snapshot" not in args  # Check in args tuple
            if derived_agg_time:
                # Convert to list since aggregate_time expects a list of strings
                stats_kwargs["aggregate_time"] = "sum"
            else:
                stats_kwargs["aggregate_time"] = False

        return stats_kwargs


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
        x: str,
        y: str,
        color: str | None = None,
        facet_col: str | None = None,
        facet_row: str | None = None,
        nice_names: bool = True,
        stacked: bool = False,
        query: str | None = None,
        **kwargs: Any,
    ) -> tuple[Figure, Axes | np.ndarray, sns.FacetGrid]:
        """Implement bar plotting logic with seaborn.objects."""
        return self._base_plot(
            data,
            kind="bar",
            x=x,
            y=y,
            color=color,
            facet_col=facet_col,
            facet_row=facet_row,
            stacked=stacked,
            nice_names=nice_names,
            query=query,
            **kwargs,
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
                data = data.assign(snapshot=pd.to_datetime(data["snapshot"]))
            except (ValueError, TypeError):
                pass
        return data

    def plot(
        self,
        data: pd.DataFrame,
        x: str = "carrier",
        y: str = "value",
        color: str | None = None,
        facet_col: str | None = None,
        facet_row: str | None = None,
        nice_names: bool = True,
        resample: str | None = None,
        query: str | None = None,
        **kwargs: Any,
    ) -> tuple[Figure, Axes | np.ndarray, sns.FacetGrid]:
        """Implement line plotting logic with seaborn.objects."""
        # Determine x-axis column
        if isinstance(data, pd.DataFrame) and set(data.columns).issubset(
            self._n.snapshots
        ):
            if resample:
                data = data.T.resample(resample).mean().T

        return self._base_plot(
            data,
            kind="line",
            x=x,
            y=y,
            color=color,
            facet_col=facet_col,
            facet_row=facet_row,
            nice_names=nice_names,
            query=query,
            **kwargs,
        )


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
                data = data.assign(snapshot=pd.to_datetime(data["snapshot"]))
            except (ValueError, TypeError):
                pass
        return data

    def plot(  # type: ignore
        self,
        data: pd.DataFrame,
        x: str,  # Removed default
        y: str = "value",
        color: str | None = None,
        facet_col: str | None = None,
        facet_row: str | None = None,
        nice_names: bool = True,
        stacked: bool = True,
        query: str | None = None,
        **kwargs: Any,
    ) -> tuple[Figure, Axes | np.ndarray, sns.FacetGrid]:
        """Implement area plotting logic with seaborn.objects."""
        stacked = stacked if stacked is not None else self._default_stacked

        return self._base_plot(
            data,
            kind="area",
            x=x,
            y=y,
            color=color,
            facet_col=facet_col,
            facet_row=facet_row,
            stacked=stacked,
            nice_names=nice_names,
            query=query,
            **kwargs,
        )
