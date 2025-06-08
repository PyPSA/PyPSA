"""Chart plots based on statistics functions (like bar, line, area)."""

from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns

from pypsa.plot.statistics.base import PlotsGenerator

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

CHART_TYPES = [
    "area",
    "bar",
    "scatter",
    "line",
    "box",
    "violin",
    "histogram",
]


def facet_iter(
    g: sns.FacetGrid,
    df: pd.DataFrame,
    facet_row: str | None,
    facet_col: str | None,
    split_by_sign: bool = False,
) -> Iterator[tuple[Axes, pd.DataFrame]]:
    """Generate (axis, filtered_data) for each facet in a FacetGrid.

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
    """Handle the creation of area or bar plots for FacetGrid.

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
            msg = "Data must contain 'value' column"
            raise ValueError(msg)

        return data

    def _to_long_format(self, data: pd.DataFrame | pd.Series) -> pd.DataFrame:
        """Convert data to long format suitable for plotting.

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
        kind: str,
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
        self._n.consistency_check_plots(strict="all")
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
            msg = f"Unsupported plot type: {kind}"
            raise ValueError(msg)

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

    def _create_category_orders(
        self,
        facet_row: str | None,
        facet_col: str | None,
        color: str | None,
        row_order: Sequence[str] | None,
        col_order: Sequence[str] | None,
        color_order: Sequence[str] | None,
    ) -> dict[str, Sequence[str]]:
        """Create a filtered dictionary of category orders for Plotly Express."""
        category_orders = {}
        if facet_row is not None and row_order is not None:
            category_orders[facet_row] = row_order
        if facet_col is not None and col_order is not None:
            category_orders[facet_col] = col_order
        if color is not None and color_order is not None:
            category_orders[color] = color_order
        return category_orders

    def iplot(
        self,
        data: pd.DataFrame,
        kind: str,
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
        height: int = 500,
        width: int = 800,
        row_order: Sequence[str] | None = None,
        col_order: Sequence[str] | None = None,
        color_order: Sequence[str] | None = None,
        color_discrete_map: dict[str, str] | None = None,
        range_x: list[float] | None = None,
        range_y: list[float] | None = None,
        labels: dict[str, str] | None = None,
        title: str | None = None,
        **kwargs: Any,
    ) -> go.Figure:
        """Interactive plot method creating charts with Plotly Express."""
        self._n.consistency_check_plots(strict="all")
        ldata = self._to_long_format(data)
        if query:
            ldata = ldata.query(query)
        ldata = self._validate(ldata)

        # Get carrier colors for the plot
        carrier_colors = self.get_carrier_colors(nice_names=nice_names)

        # Set up labels dictionary for axis labels
        if labels is None:
            labels = {}

        # Get unit for legends
        data_label = data.attrs.get("name", "Value")
        unit = data.attrs.get("unit", "")
        if unit != "carrier dependent":
            data_label += f" [{unit}]"

        if x == "value":
            labels[x] = data_label
        elif y == "value":
            labels[y] = data_label

        # Handle categorical axes to avoid auto-sorting
        if x != "value" and ldata[x].dtype.name == "object":
            ldata[x] = pd.Categorical(
                ldata[x], categories=ldata[x].unique(), ordered=True
            )
        if y != "value" and ldata[y].dtype.name == "object":
            ldata.loc[:, y] = pd.Categorical(
                ldata[y], categories=ldata[y].unique(), ordered=True
            )

        # Prepare color mapping if color column is provided
        if color and color_discrete_map is None and color in ldata.columns:
            color_values = ldata[color].unique()
            color_discrete_map = {
                col: carrier_colors.get(col, "#AAAAAA") for col in color_values
            }

        # Set default title if none is provided
        if title is None:
            title = self._to_title(data.attrs.get("name", ""))

        # Create category orders dict
        category_orders = self._create_category_orders(
            facet_row, facet_col, color, row_order, col_order, color_order
        )

        # Create appropriate plot based on kind
        if kind == "bar":
            # Handle regular vs stacked bar charts
            if stacked and color is not None:
                barmode = "stack"
            else:
                barmode = "group"

            fig = px.bar(
                ldata,
                x=x,
                y=y,
                color=color,
                facet_col=facet_col,
                facet_row=facet_row,
                height=height,
                width=width,
                facet_col_wrap=kwargs.get("facet_col_wrap", 0),
                category_orders=category_orders,
                color_discrete_map=color_discrete_map,
                barmode=barmode,
                range_x=range_x,
                range_y=range_y,
                labels=labels,
                title=title,
                **{k: v for k, v in kwargs.items() if k not in ["facet_col_wrap"]},
            )

        elif kind == "line":
            fig = px.line(
                ldata,
                x=x,
                y=y,
                color=color,
                facet_col=facet_col,
                facet_row=facet_row,
                height=height,
                width=width,
                facet_col_wrap=kwargs.get("facet_col_wrap", 0),
                category_orders=category_orders,
                color_discrete_map=color_discrete_map,
                range_x=range_x,
                range_y=range_y,
                labels=labels,
                title=title,
                **{k: v for k, v in kwargs.items() if k not in ["facet_col_wrap"]},
            )

        elif kind == "area":
            kwargs = dict(
                x=x,
                y=y,
                color=color,
                facet_col=facet_col,
                facet_row=facet_row,
                height=height,
                width=width,
                facet_col_wrap=kwargs.get("facet_col_wrap", 0),
                category_orders=category_orders,
                color_discrete_map=color_discrete_map,
                range_x=range_x,
                range_y=range_y,
                labels=labels,
                title=title,
                **{k: v for k, v in kwargs.items() if k not in ["facet_col_wrap"]},
            )

            if stacked:
                pos = ldata[ldata.value > 0]
                neg = ldata[ldata.value < 0]
                positives = px.area(pos, **kwargs)
                positives.update_traces(
                    stackgroup="positive",
                    showlegend=False,
                )
                negatives = px.area(neg, **kwargs)
                negatives.update_traces(
                    stackgroup="negative",
                    showlegend=False,
                )

                # In order to not bloat the hover display with zeros, we need to
                # filter out zeros in ldata as done below. However, then the legend
                # only shows for the latest traces (ignoring the positive values).
                # To fix this, we need to add an artificial trace with the last value
                # of each color and use that for the legend.
                unique_colors = ldata[color].unique() if color else []
                artificial_zeros = pd.DataFrame(
                    {x: ldata[x].iloc[-1], y: np.nan, color: unique_colors}
                )
                if facet_col:
                    artificial_zeros[facet_col] = ldata[facet_col].iloc[-1]
                if facet_row:
                    artificial_zeros[facet_row] = ldata[facet_row].iloc[-1]

                artificials = px.area(
                    artificial_zeros,
                    **kwargs,
                )

                # Combine the figures
                fig = positives.add_traces(negatives.data).add_traces(artificials.data)
            else:
                fig = px.area(ldata, **kwargs)
            fig.update_traces(line={"width": 0})
            fig.update_layout(hovermode="x")
        else:
            msg = f"Unsupported plot type: {kind}"
            raise ValueError(msg)

        # Update layout
        fig.update_layout(
            template="plotly_white", margin={"l": 50, "r": 50, "t": 50, "b": 50}
        )

        if not sharex and sharex is not None:
            fig.update_xaxes(matches=None)
        if not sharey and sharey is not None:
            fig.update_yaxes(matches=None)

        return fig

    def derive_statistic_parameters(
        self,
        *args: Any,
        method_name: str = "",  # make required
    ) -> dict[str, Any]:
        """Extract plotting specification rules including groupby columns and component aggregation.

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
        filtered_cols = [c for c in args if c not in filtered and c is not None]

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
