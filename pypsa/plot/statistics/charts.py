"""Chart plots based on statistics functions (like bar, line, area)."""

from __future__ import annotations

from abc import ABC
from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure

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
                mask = facet_data[facet_row] == row_val
                facet_data = facet_data[mask]
            if facet_col is not None and col_val is not None:
                mask = facet_data[facet_col] == col_val
                facet_data = facet_data[mask]

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
    palette: dict | None = None,
    kind: str = "area",
    ylim: tuple[float, float] | None = None,
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
    palette : dict, optional
        Color palette to use for the plot
    kind : str, optional
        Kind of plot to create ('area' or 'bar')
    ylim : tuple, optional
        Y-axis limits for the plot
    **kwargs : additional keyword arguments
        Passed to plotting function

    """
    custom_case = color not in [x, y, facet_col, facet_row, None] or kind == "area"
    if custom_case:
        # Store the color palette from FacetGrid for consistent colors
        color_order = g.hue_names if hasattr(g, "hue_names") else None

        split_by_sign = df["value"].min() < 0 and df["value"].max() > 0

        if kind == "bar" and x == "value":
            kind = "barh"
            x_var, y_var = y, x
        else:
            x_var, y_var = x, y

        for ax, facet_data in facet_iter(g, df, facet_row, facet_col, split_by_sign):
            # Pivot data to have x as index, color as columns, and y as values
            if color is None:
                pivoted = facet_data.set_index(x_var)[[y_var]]
                color_dict = None
            else:
                pivoted = facet_data.pivot(index=x_var, columns=color, values=y_var)
                color_dict = palette

            # Special case of duplicate indices, e.g. carriers in groupers, but not plotted
            if not pivoted.index.is_unique:
                pivoted = pivoted.groupby(level=0).sum()

            # Ensure columns are ordered according to the hue order
            if color_order:
                # Get only the columns that exist in this facet
                available_cols = [c for c in color_order if c in pivoted.columns]
                if available_cols:  # Only reorder if we have columns
                    pivoted = pivoted[available_cols]

            # Weird behavior in pandas plotting, have to correct the ylim if None
            # https://github.com/pandas-dev/pandas/blob/c0371cedf3a9682596481dab87b43653a48da186/pandas/plotting/_matplotlib/core.py#L1817
            if y == "value" and ylim is None and len(list(ax.get_shared_y_axes())) == 0:
                ax._shared_axes["y"].join(ax, ax)  # type: ignore

            # Plot with pandas - no legend to avoid duplicates
            pivoted.plot(
                kind=kind,
                ax=ax,
                stacked=stacked,
                legend=False,
                color=color_dict,
                **kwargs,
            )
            g._update_legend_data(ax)
        g._finalize_grid([x, y])

    elif kind == "bar":
        palette = palette if color is not None else None
        g.map_dataframe(sns.barplot, x=x, y=y, hue=color, palette=palette, **kwargs)

    return g


class ChartGenerator(PlotsGenerator, ABC):
    """Base class for generating charts based on statistics functions."""

    def _to_title(self, s: str) -> str:
        """Convert string to title case."""
        return s.replace("_", " ").title()

    def _validate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate data has required columns and types."""
        if "value" not in data.columns:
            raise ValueError("Data must contain 'value' column")

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

    def plot(
        self,
        data: pd.DataFrame,
        kind: Literal["area", "bar", "scatter", "line", "box", "violin", "histogram"],
        x: str,
        y: str,
        color: str | None = None,
        facet_col: str | None = None,
        facet_row: str | None = None,
        stacked: bool = True,
        nice_names: bool = True,
        query: str | None = None,
        sharex: bool | None = None,
        sharey: bool | None = None,
        height: float = 3,
        aspect: float = 2,
        row_order: Sequence[str] | None = None,
        col_order: Sequence[str] | None = None,
        hue_order: Sequence[str] | None = None,
        hue_kws: dict[str, Any] | None = None,
        despine: bool = True,
        margin_titles: bool = False,
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
        subplot_kws: dict[str, Any] | None = None,
        gridspec_kws: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> tuple[Figure, Axes | np.ndarray, sns.FacetGrid]:
        """Plot method to be implemented by subclasses."""
        plotting_consistency_check(self._n, strict="all")
        ldata = self._to_long_format(data)
        if query:
            ldata = ldata.query(query)
        ldata = self._validate(ldata)
        palette = self.get_carrier_colors(nice_names=nice_names)

        # set shared axis to the one where "value" is plotted
        if sharex is None:
            sharex = x == "value"
        if sharey is None:
            sharey = y == "value"

        # Always use FacetGrid for consistency
        g = sns.FacetGrid(
            ldata,
            row=facet_row,
            col=facet_col,
            palette=palette,
            sharex=sharex,
            sharey=sharey,
            height=height,
            aspect=aspect,
            row_order=row_order,
            col_order=col_order,
            hue_order=hue_order,
            hue_kws=hue_kws,
            despine=despine,
            margin_titles=margin_titles,
            xlim=xlim,
            ylim=ylim,
            subplot_kws=subplot_kws,
            gridspec_kws=gridspec_kws,
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
                ylim=ylim,
                palette=palette,
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
                ylim=ylim,
                palette=palette,
                **kwargs,
            )
        # Other plot types remain the same
        elif kind == "scatter":
            g.map_dataframe(sns.scatterplot, x=x, y=y, hue=color, **kwargs)
        elif kind == "line":
            g.map_dataframe(sns.lineplot, x=x, y=y, hue=color, **kwargs)
        elif kind == "box":
            g.map_dataframe(sns.boxplot, x=x, y=y, hue=color, **kwargs)
        elif kind == "violin":
            g.map_dataframe(sns.violinplot, x=x, y=y, hue=color, **kwargs)
        elif kind == "histogram":
            if y is None:
                g.map_dataframe(sns.histplot, x=x, hue=color, **kwargs)
            else:
                g.map_dataframe(sns.histplot, x=x, y=y, hue=color, **kwargs)
        else:
            raise ValueError(f"Unsupported plot type: {kind}")

        # Add legend if color is specified (for non-area plots, area plots handle this separately)
        if color is not None:
            g.add_legend()

        # Set axis labels
        # Get unit for legends
        label = data.attrs.get("name", "Value")
        unit = data.attrs.get("unit", "")
        if unit != "carrier dependent":
            label += f" [{unit}]"
        if x == "value":
            g.set_axis_labels(x_var=label)
        elif y == "value":
            g.set_axis_labels(y_var=label)

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
