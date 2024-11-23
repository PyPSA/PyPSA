# type: ignore #TODO: remove with #912
"""
Functions for plotting networks.
"""

from __future__ import annotations

import logging
import warnings
from functools import wraps
from typing import TYPE_CHECKING

import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.legend_handler import HandlerPatch
from matplotlib.patches import Circle, FancyArrow, Patch, Wedge
from shapely import linestrings

from pypsa.geo import (
    compute_bbox,
    get_projected_area_factor,
)
from pypsa.statistics import get_transmission_carriers
from pypsa.utils import deprecated_kwargs

cartopy_present = True
try:
    import cartopy
    import cartopy.mpl.geoaxes
except ImportError:
    cartopy_present = False


pltly_present = True
try:
    import plotly.graph_objects as go
    import plotly.offline as pltly
except ImportError:
    pltly_present = False

if TYPE_CHECKING:
    from pypsa.components import Network

logger = logging.getLogger(__name__)
warnings.simplefilter("always", DeprecationWarning)


def _convert_to_series(variable, index):
    if isinstance(variable, dict):
        variable = pd.Series(variable)
    elif not isinstance(variable, pd.Series):
        variable = pd.Series(variable, index=index)
    return variable


def _apply_cmap(colors, cmap, cmap_norm=None):
    if np.issubdtype(colors.dtype, np.number):
        if not isinstance(cmap, plt.Colormap):
            cmap = plt.get_cmap(cmap)
        if not cmap_norm:
            cmap_norm = plt.Normalize(vmin=colors.min(), vmax=colors.max())
        colors = colors.apply(lambda cval: cmap(cmap_norm(cval)))
    return colors


def apply_layouter(
    n: Network, layouter: nx.drawing.layout = None, inplace: bool = False
):
    """
    Automatically generate bus coordinates for the network graph according to a
    layouting function from `networkx <https://networkx.github.io/>`_.

    Parameters
    ----------
    n : pypsa.Network
        Network to generate coordinates for.
    layouter : networkx.drawing.layout function, default None
        Layouting function from `networkx <https://networkx.github.io/>`_. See
        `list <https://networkx.github.io/documentation/stable/reference/drawing.html#module-networkx.drawing.layout>`_
        of available options. By default, coordinates are determined for a
        `planar layout <https://networkx.github.io/documentation/stable/reference/generated/networkx.drawing.layout.planar_layout.html#networkx.drawing.layout.planar_layout>`_
        if the network graph is planar, otherwise for a
        `Kamada-Kawai layout <https://networkx.github.io/documentation/stable/reference/generated/networkx.drawing.layout.kamada_kawai_layout.html#networkx.drawing.layout.kamada_kawai_layout>`_.
    inplace : bool, default False
        Assign generated coordinates to the network bus coordinates
        at ``n.buses[['x', 'y']]`` if True, otherwise return them.

    Returns
    -------
    coordinates : pd.DataFrame or None
        DataFrame with x and y coordinates for each bus. Only returned if
        inplace is False.

    Examples
    --------
    >>> coords = apply_layouter(n)
    >>> apply_layouter(n, assign=True, layouter=nx.layout.spring_layout)

    """
    G = n.graph()

    if layouter is None:
        if nx.check_planarity(G)[0]:
            layouter = nx.planar_layout
        else:
            layouter = nx.kamada_kawai_layout

    coordinates = pd.DataFrame(layouter(G)).T.rename({0: "x", 1: "y"}, axis=1)

    if inplace:
        n.buses[["x", "y"]] = coordinates
        return None
    else:
        return coordinates.x, coordinates.y


class NetworkMapPlotter:
    def __init__(self, network: Network):
        self.n = network
        self.x = None  # Initialized in init_layout
        self.y = None  # Initialized in init_layout
        self.boundaries = None  # Initialized in init_boundaries
        self.ax = None  # Initialized in init_axis
        self.area_factor = 1  # Initialized in init_axis

    def init_layout(self, layouter: nx.drawing.layout = None):
        # Check if networkx layouter is given or needed to get bus positions
        is_empty = (
            (self.n.buses[["x", "y"]].isnull() | (self.n.buses[["x", "y"]] == 0))
            .all()
            .all()
        )
        if layouter or is_empty:
            self.x, self.y = apply_layouter(self.n, layouter)
        else:
            self.x, self.y = self.n.buses["x"], self.n.buses["y"]

    def init_boundaries(self, buses, boundaries, margin):
        # Set boundaries, if not given

        if boundaries is None:
            self.boundaries = sum(
                zip(*compute_bbox(self.x[buses], self.y[buses], margin)), ()
            )
        else:
            self.boundaries = boundaries

    def add_geomap_features(self, geomap=True, geomap_colors=None):
        resolution = "50m" if isinstance(geomap, bool) else geomap
        if resolution not in ["10m", "50m", "110m"]:
            msg = "Resolution has to be one of '10m', '50m', '110m'"
            raise ValueError(msg)

        if not geomap_colors:
            geomap_colors = {}
        elif not isinstance(geomap_colors, dict):
            geomap_colors = {
                "ocean": "lightblue",
                "land": "whitesmoke",
                "border": "darkgray",
                "coastline": "black",
            }

        if "land" in geomap_colors:
            self.ax.add_feature(
                cartopy.feature.LAND.with_scale(resolution),
                facecolor=geomap_colors["land"],
            )

        if "ocean" in geomap_colors:
            self.ax.add_feature(
                cartopy.feature.OCEAN.with_scale(resolution),
                facecolor=geomap_colors["ocean"],
            )

        self.ax.add_feature(
            cartopy.feature.BORDERS.with_scale(resolution),
            linewidth=0.3,
            edgecolor=geomap_colors.get("border", "black"),
        )

        self.ax.add_feature(
            cartopy.feature.COASTLINE.with_scale(resolution),
            linewidth=0.3,
            edgecolor=geomap_colors.get("coastline", "black"),
        )

    def init_axis(self, ax, projection, geomap, geomap_colors, title):
        # Set up plot (either cartopy or matplotlib)

        transform = cartopy.crs.Projection(self.n.crs)
        if geomap:
            if projection is None:
                projection = transform
            elif not isinstance(projection, cartopy.crs.Projection):
                msg = "The passed projection is not a cartopy.crs.Projection"
                raise ValueError(msg)

            if ax is None:
                self.ax = plt.axes(projection=projection)
            elif not isinstance(ax, cartopy.mpl.geoaxes.GeoAxesSubplot):
                msg = "The passed axis is not a GeoAxesSubplot. You can "
                "create one with: \nimport cartopy.crs as ccrs \n"
                "fig, ax = plt.subplots("
                'subplot_kw={"projection":ccrs.PlateCarree()})'
                raise ValueError(msg)
            else:
                self.ax = ax

            x_, y_, _ = self.ax.projection.transform_points(
                transform, self.x.values, self.y.values
            ).T
            self.x, self.y = pd.Series(x_, self.x.index), pd.Series(y_, self.y.index)

            if geomap_colors is not False:
                self.add_geomap_features(geomap, geomap_colors)

            if self.boundaries is not None:
                self.ax.set_extent(self.boundaries, crs=transform)

            self.area_factor = get_projected_area_factor(self.ax, self.n.srid)
        else:
            if ax is None:
                self.ax = plt.gca()
            else:
                self.ax = ax
            self.ax.axis(self.boundaries)
        self.ax.set_aspect("equal")
        self.ax.axis("off")
        self.ax.set_title(title)

    def add_jitter(self, jitter):
        """
        Add random jitter to data.

        Parameters
        ----------
        x : numpy.ndarray
            X data to add jitter to.
        y : numpy.ndarray
            Y data to add jitter to.
        jitter : float
            The amount of jitter to add. Function adds a random number between -jitter and
            jitter to each element in the data arrays.

        Returns
        -------
        x_jittered : numpy.ndarray
            X data with added jitter.
        y_jittered : numpy.ndarray
            Y data with added jitter.
        """
        rng = np.random.default_rng()  # Create a random number generator
        if jitter is not None:
            self.x = self.x + rng.uniform(low=-jitter, high=jitter, size=len(self.x))
            self.y = self.y + rng.uniform(low=-jitter, high=jitter, size=len(self.y))

        return self.x, self.y

    def get_multiindex_buses(
        self, sizes: pd.Series, colors: pd.Series, alpha, split_circles
    ):
        # We are drawing pies to show all the different shares
        patches = []
        for b_i in sizes.index.unique(level=0):
            s_base = sizes.loc[b_i]

            if split_circles:
                # As only half of the circle is drawn, increase area by factor 2
                s_base = s_base * 2
                s_base = (
                    s_base[s_base > 0],
                    s_base[s_base < 0],
                )
                starts = 0, 1
                scope = 180
            else:
                s_base = (s_base[s_base > 0],)
                starts = (0.25,)
                scope = 360

            for s, start in zip(s_base, starts):
                radius = abs(s.sum()) ** 0.5
                ratios = abs(s) if radius == 0.0 else s / s.sum()
                for i, ratio in ratios.items():
                    patches.append(
                        Wedge(
                            (self.x.at[b_i], self.y.at[b_i]),
                            radius,
                            scope * start,
                            scope * (start + ratio),
                            facecolor=colors[i],
                            alpha=alpha,
                        )
                    )
                    start = start + ratio
        return patches

    def get_singleindex_buses(self, sizes, colors, alpha):
        patches = []
        for b_i in sizes.index[(sizes != 0) & ~sizes.isna()]:
            radius = sizes.at[b_i] ** 0.5
            patches.append(
                Circle(
                    (self.x.at[b_i], self.y.at[b_i]),
                    radius,
                    facecolor=colors.at[b_i],
                    alpha=alpha,
                )
            )
        return patches

    @staticmethod
    def _dataframe_from_arguments(index, **kwargs):
        if any(isinstance(v, pd.Series) for v in kwargs.values()):
            return pd.DataFrame(kwargs)
        return pd.DataFrame(kwargs, index=index)

    def _flow_ds_from_arg(self, flow, component_name):
        if isinstance(flow, pd.Series):
            return flow

        if flow in self.n.snapshots:
            return self.n.dynamic(component_name).p0.loc[flow]

        if isinstance(flow, str) or callable(flow):
            return self.n.dynamic(component_name).p0.agg(flow, axis=0)

    def get_branch_collection(
        self,
        c,
        widths: pd.Series,
        colors: pd.Series,
        alpha: pd.Series,
        geometry: pd.Series,
    ):
        """
        Create a LineCollection for a single branch component.

        Parameters
        ----------
        c : Component
            Network component being plotted
        widths : float/Series
            Line widths for the component
        colors : str/Series
            Colors for the component
        alpha : float/Series
            Alpha values for the component
        geometry : bool
            Whether to use geometry data for plotting
        """
        idx = widths.index
        if not geometry:
            segments = np.asarray(
                (
                    (
                        c.static.bus0[idx].map(self.x),
                        c.static.bus0[idx].map(self.y),
                    ),
                    (
                        c.static.bus1[idx].map(self.x),
                        c.static.bus1[idx].map(self.y),
                    ),
                )
            ).transpose(2, 0, 1)
        else:
            from shapely.geometry import LineString
            from shapely.wkt import loads

            linestrings = geometry[lambda ds: ds != ""].map(loads)
            if not all(isinstance(ls, LineString) for ls in linestrings):
                msg = "The WKT-encoded geometry in the 'geometry' column must be "
                "composed of LineStrings"
                raise ValueError(msg)
            segments = np.asarray(list(linestrings.map(np.asarray)))

        return LineCollection(
            segments,
            linewidths=widths,
            antialiaseds=(1,),
            colors=colors,
            alpha=alpha,
        )

    @staticmethod
    def _directed_flow(coords, flow, color, area_factor, alpha=1):
        """
        Helper function to generate arrows from flow data.
        """
        # this funtion is used for diplaying arrows representing the network flow
        data = pd.DataFrame(
            {
                "arrowsize": flow.abs().pipe(np.sqrt).clip(lower=1e-8),
                "direction": np.sign(flow),
                "linelength": (
                    np.sqrt(
                        (coords.x1 - coords.x2) ** 2.0 + (coords.y1 - coords.y2) ** 2
                    )
                ),
            }
        )
        data = data.join(coords)
        data["arrowsize"] = data["arrowsize"].mul(area_factor)
        data["arrowtolarge"] = 1.5 * data.arrowsize > data.linelength
        # swap coords for negativ directions
        data.loc[data.direction == -1.0, ["x1", "x2", "y1", "y2"]] = data.loc[
            data.direction == -1.0, ["x2", "x1", "y2", "y1"]
        ].values
        if ((data.linelength > 0.0) & (~data.arrowtolarge)).any():
            data["arrows"] = data[(data.linelength > 0.0) & (~data.arrowtolarge)].apply(
                lambda ds: FancyArrow(
                    ds.x1,
                    ds.y1,
                    0.6 * (ds.x2 - ds.x1)
                    - ds.arrowsize * 0.75 * (ds.x2 - ds.x1) / ds.linelength,
                    0.6 * (ds.y2 - ds.y1)
                    - ds.arrowsize * 0.75 * (ds.y2 - ds.y1) / ds.linelength,
                    head_width=ds.arrowsize,
                ),
                axis=1,
            )
        data.loc[(data.linelength > 0.0) & (data.arrowtolarge), "arrows"] = data[
            (data.linelength > 0.0) & (data.arrowtolarge)
        ].apply(
            lambda ds: FancyArrow(
                ds.x1,
                ds.y1,
                0.001 * (ds.x2 - ds.x1),
                0.001 * (ds.y2 - ds.y1),
                head_width=ds.arrowsize,
            ),
            axis=1,
        )
        data = data.dropna(subset=["arrows"])
        return PatchCollection(
            data.arrows,
            color=color,
            alpha=alpha,
            edgecolors="black",
            linewidths=0.0,
            zorder=4,
        )

    def get_flow_collection(self, c, flow, widths, colors, alpha):
        """
        Create a flow arrow collection for a single branch component.

        Parameters
        ----------
        c : Component
            Network component being plotted
        flow : pd.Series
            Flow values for the component
        widths : float/Series
            Line widths to scale the arrows
        colors : str/Series
            Colors for the arrows
        alpha : float/Series
            Alpha values for the arrows
        """
        idx = widths.index
        coords = pd.DataFrame(
            {
                "x1": c.static.bus0[idx].map(self.x),
                "y1": c.static.bus0[idx].map(self.y),
                "x2": c.static.bus1[idx].map(self.x),
                "y2": c.static.bus1[idx].map(self.y),
            }
        )

        return self._directed_flow(coords, flow, colors, self.area_factor, alpha)

    @staticmethod
    def scaling_factor_from_area_contribution(
        area_contributions, x_min, x_max, y_min, y_max, target_area_fraction=0.1
    ):
        """
        Scale series for plotting so that the total area of all area contributions
        takes up approximately the specified fraction of the plot area.

        Parameters
        ----------
        area_contribution : pd.Series
            Series containing the balance values for each bus
        x_min, x_max : float
            The x-axis extent of the plot
        y_min, y_max : float
            The y-axis extent of the plot
        target_area_fraction : float, optional
            Desired fraction of plot area to be covered by all circles (default: 0.3)

        Returns
        -------
        pd.Series
            Scaled values
        """
        plot_area = (x_max - x_min) * (y_max - y_min)
        target_total_circle_area = plot_area * target_area_fraction
        current_total_area = np.sum(np.abs(area_contributions))
        return target_total_circle_area / current_total_area

    @staticmethod
    def aggregate_flow_by_connection(
        flow: pd.Series, branches: pd.DataFrame
    ) -> pd.Series:
        if flow.empty:
            return flow
        connected_buses = branches.loc[flow.index, ["bus0", "bus1"]]
        correctly_sorted = connected_buses.bus0 < connected_buses.bus1

        flow_sorted = flow.where(correctly_sorted, -flow)
        buses_sorted = connected_buses.apply(sorted, axis=1).str.join(" - ")
        flow_grouped = (
            flow_sorted.groupby(buses_sorted).transform("sum") * correctly_sorted
        )
        flow_grouped = flow_grouped[buses_sorted.drop_duplicates().index]

        return flow_grouped

    @staticmethod
    def flow_to_width(
        flow: float | pd.Series, width_factor: float = 0.2
    ) -> float | pd.Series:
        """
        Calculate the width of a line based on the flow value.

        Parameters
        ----------
        flow : float or pd.Series
            Flow values
        width_factor : float
            Ratio between the flow width and and line width (default: 0.5)
        """
        return abs(flow) ** 0.5 * width_factor * 10

    @classmethod
    def plot(
        cls,
        n: Network,
        layouter: nx.drawing.layout = None,
        boundaries: list | tuple | None = None,
        margin: float | None = 0.05,
        ax: plt.Axes = None,
        geomap: bool | str = True,
        projection: cartopy.crs.Projection = None,
        geomap_colors: dict | bool | None = None,
        title: str = "",
        jitter: float | None = None,
        branch_components: list | None = None,
        bus_sizes: float | dict | pd.Series = 2e-2,
        bus_split_circles: bool = False,
        bus_colors: str | dict | pd.Series = None,
        bus_cmap: str | plt.cm.ColorMap = None,
        bus_cmap_norm: plt.Normalize = None,
        bus_alpha: float | dict | pd.Series = 1,
        geometry=False,
        line_flow: str | callable | dict | pd.Series | Network.snapshots = None,
        line_colors: str | dict | pd.Series = "rosybrown",
        line_cmap: str | plt.cm.ColorMap = "viridis",
        line_cmap_norm: plt.Normalize = None,
        line_alpha: float | dict | pd.Series = 1,
        line_widths: float | dict | pd.Series = 1.5,
        link_flow: str | callable | dict | pd.Series | Network.snapshots = None,
        link_colors: str | dict | pd.Series = "darkseagreen",
        link_cmap: str | plt.cm.ColorMap = "viridis",
        link_cmap_norm: plt.Normalize = None,
        link_alpha: float | dict | pd.Series = 1,
        link_widths: float | dict | pd.Series = 1.5,
        transformer_flow: str | callable | dict | pd.Series | Network.snapshots = None,
        transformer_colors: str | dict | pd.Series = "orange",
        transformer_cmap: str | plt.cm.ColorMap = "viridis",
        transformer_cmap_norm: plt.Normalize = None,
        transformer_alpha: float | dict | pd.Series = 1,
        transformer_widths: float | dict | pd.Series = 1.5,
        auto_scale_flow: bool = True,
        flow=None,
    ):
        """
            Plot the network buses and lines using matplotlib and cartopy.

        Parameters
        ----------
            n : pypsa.Network
                Network to plot
            layouter : networkx.drawing.layout, default None
                Layouting function from `networkx <https://networkx.github.io/>`_ which
                overrules coordinates given in ``n.buses[['x', 'y']]``. See
                `list <https://networkx.github.io/documentation/stable/reference/drawing.html#module-networkx.drawing.layout>`_
                of available options.
            boundaries : list/tuple, default None
                Boundaries of the plot in format [x1, x2, y1, y2]
            margin : float, defaults to 0.05
                Margin at the sides as proportion of distance between max/min x, y
                Will be ignored if boundaries are given.
            ax : matplotlib.pyplot.Axes, defaults to None
                Axis to plot on. Defaults to plt.gca() if geomap is False, otherwise
                to plt.axes(projection=projection).
            geomap: bool/str, default True
                Switch to use Cartopy and draw geographical features.
                If string is passed, it will be used as a resolution argument,
                valid options are '10m', '50m' and '110m'.
            projection: cartopy.crs.Projection, defaults to None
                Define the projection of your geomap, only valid if cartopy is
                installed. If None (default) is passed the projection for cartopy
                is set to cartopy.crs.PlateCarree
            geomap_colors : dict/bool, default None
                Specify colors to paint land and sea areas in.
                If True, it defaults to `{'ocean': 'lightblue', 'land': 'whitesmoke'}`.
                If no dictionary is provided, colors are white.
                If False, no geographical features are plotted.
            title : string, default ""
                Graph title
            jitter : float, default None
                Amount of random noise to add to bus positions to distinguish
                overlapping buses
            branch_components : list, default n.branch_components
                Branch components to be plotted
            bus_sizes : float/dict/pandas.Series
                Sizes of bus points, defaults to 1e-2. If a multiindexed Series is passed,
                the function will draw pies for each bus (first index level) with
                segments of different color (second index level). Such a Series is ob-
                tained by e.g. n.generators.groupby(['bus', 'carrier']).p_nom.sum()
            bus_split_circles : bool, default False
                Draw half circles if bus_sizes is a pandas.Series with a Multiindex.
                If set to true, the upper half circle per bus then includes all positive values
                of the series, the lower half circle all negative values. Defaults to False.
            bus_colors : str/dict/pandas.Series
                Colors for the buses, defaults to "cadetblue". If bus_sizes is a
                pandas.Series with a Multiindex, bus_colors defaults to the
                n.carriers['color'] column.
            bus_cmap : plt.cm.ColorMap/str
                If bus_colors are floats, this color map will assign the colors
            bus_cmap_norm : plt.Normalize
                The norm applied to the bus_cmap
            bus_alpha : float/dict/pandas.Series
                Adds alpha channel to buses, defaults to 1
            geometry :
                # TODO

            Additional Parameters
            ---------------------
            line_flow : str/callable/dict/pandas.Series/Network.snapshots, default None
                Flow to be for each line branch. If an element of
                n.snapshots is given, the flow at this timestamp will be
                displayed. If an aggregation function is given, is will be applied
                to the total network flow via pandas.DataFrame.agg (accepts also
                function names). Otherwise flows can be specified by passing a pandas
                Series. Use the corresponding width argument to adjust size of the
                flow arrows.
            line_colors : str/pandas.Series
                Colors for the lines, defaults to 'rosybrown'.
            line_cmap : plt.cm.ColorMap/str|dict
                If line_colors are floats, this color map will assign the colors.
            line_cmap_norm : plt.Normalize
                The norm applied to the line_cmap.
            line_alpha : str/pandas.Series
                Alpha for the lines, defaults to 1.
            line_widths : dict/pandas.Series
                Widths of lines, defaults to 1.5
            link_flow : str/callable/dict/pandas.Series/Network.snapshots, default None
                Flow to be for each link branch. See line_flow for more information.
            link_colors : str/pandas.Series
                Colors for the links, defaults to 'darkseagreen'.
            link_cmap : plt.cm.ColorMap/str|dict
                If link_colors are floats, this color map will assign the colors.
            link_cmap_norm : plt.Normalize|matplotlib.colors.*Norm
                The norm applied to the link_cmap.
            link_alpha : str/pandas.Series
                Alpha for the links, defaults to 1.
            link_widths : dict/pandas.Series
                Widths of links, defaults to 1.5
            transformer_flow : str/callable/dict/pandas.Series/Network.snapshots, default None
                Flow to be for each transformer branch. See line_flow for more information.
            transformer_colors : str/pandas.Series
                Colors for the transfomer, defaults to 'orange'.
            transformer_cmap : plt.cm.ColorMap/str|dict
                If transformer_colors are floats, this color map will assign the colors.
            transformer_cmap_norm : matplotlib.colors.Normalize|matplotlib.colors.*Norm
                The norm applied to the transformer_cmap.
            transformer_alpha : str/pandas.Series
                Alpha for the transfomer, defaults to 1.
            transformer_widths : dict/pandas.Series
                Widths of transformer, defaults to 1.5

            .. deprecated:: 0.28.0
                `flow` will be deprecated, use `line_flow`, `link_flow` and `transformer_flow`
                    instead. The argument will be passed to all branches.
                `bus_norm`, `line_norm`, `link_norm` and `transformer_norm` are deprecated,
                    use `bus_cmap_norm`, `line_cmap_norm`, `link_cmap_norm` and
                    `transformer_cmap_norm` instead.
                `color_geomap` is deprecated, use `geomap_colors` instead.

        Returns
        -------
        dict: collections of matplotlib objects
            2D dictinary with the following keys:
            - 'nodes'
                - 'Bus': Collection of bus points
            - 'branches' (for each branch component)
                - 'lines': Collection of line branches
                - 'links': Collection of link branches
                - 'transformers': Collection of transformer branches
            - 'flows' (for each branch component)
                - 'lines': Collection of line flows
                - 'links': Collection of link flows
                - 'transformers': Collection of transformer flows
        """

        # Check for API changes

        # Deprecation warnings
        if flow is not None:
            if (
                line_flow is not None
                or link_flow is not None
                or transformer_flow is not None
            ):
                msg = "The `flow` argument is deprecated, use `line_flow`, `link_flow` and "
                "`transformer_flow` instead. You can't use both arguments at the same time."
                raise ValueError(msg)
            if isinstance(flow, pd.Series) and isinstance(flow.index, pd.MultiIndex):
                msg = (
                    "The `flow` argument is deprecated, use `line_flow`, `link_flow` and "
                    "`transformer_flow` instead. Multiindex Series are not supported anymore."
                )
                raise ValueError(msg)
            warnings.warn(
                "The `flow` argument is deprecated and will be removed in a future "
                "version. Use `line_flow`, `link_flow` and `transformer_flow` instead. The "
                "argument will be passed to all branches. Multiindex Series are not supported anymore.",
                DeprecationWarning,
                2,
            )
            line_flow = flow
            link_flow = flow
            transformer_flow = flow

        if margin is None:
            logger.warning(
                "The `margin` argument does support None value anymore. "
                "Falling back to the default value 0.05. This will raise "
                "an error in the future."
            )
            margin = 0.05

        # Deprecation errors
        if isinstance(line_widths, pd.Series) and isinstance(
            line_widths.index, pd.MultiIndex
        ):
            msg = (
                "Index of argument 'line_widths' is a Multiindex, "
                "this is not support since pypsa v0.17. "
                "Set differing widths with arguments 'line_widths', "
                "'link_widths' and 'transformer_widths'."
            )
            raise TypeError(msg)

        if isinstance(line_colors, pd.Series) and isinstance(
            line_colors.index, pd.MultiIndex
        ):
            msg = (
                "Index of argument 'line_colors' is a Multiindex, "
                "this is not support since pypsa v0.17. "
                "Set differing colors with arguments 'line_colors', "
                "'link_colors' and 'transformer_colors'."
            )
            raise TypeError(msg)

        # Check for ValueErrors

        if geomap:
            if not cartopy_present:
                logger.warning("Cartopy needs to be installed to use `geomap=True`.")
                geomap = False

        if not geomap and hasattr(ax, "projection"):
            msg = "The axis has a projection, but `geomap` is set to False"
            raise ValueError(msg)

        if geomap and not cartopy_present:
            logger.warning("Cartopy needs to be installed to use `geomap=True`.")
            geomap = False

        # Check if bus_sizes is a MultiIndex
        multindex_buses = isinstance(bus_sizes, pd.Series) and isinstance(
            bus_sizes.index, pd.MultiIndex
        )

        # Apply default values
        if bus_colors is None:
            if multindex_buses:
                bus_colors = n.carriers.color
            else:
                bus_colors = "cadetblue"

        # Format different input types
        bus_colors = _convert_to_series(bus_colors, n.buses.index)
        bus_sizes = _convert_to_series(bus_sizes, n.buses.index)

        # Add missing colors
        # TODO: This is not consistent, since for multiindex a ValueError is raised
        if not multindex_buses:
            bus_colors = bus_colors.reindex(n.buses.index)

        # Raise additional ValueErrors after formatting
        if multindex_buses:
            if len(bus_sizes.index.unique(level=0).difference(n.buses.index)) != 0:
                msg = "The first MultiIndex level of sizes must contain buses"
                raise ValueError(msg)
            if not bus_sizes.index.unique(level=1).isin(bus_colors.index).all():
                msg = "Colors not defined for all elements in the second MultiIndex "
                "level of sizes, please make sure that all the elements are "
                "included in colors or in n.carriers.color"
                raise ValueError(msg)

        # Apply all cmaps
        bus_colors = _apply_cmap(bus_colors, bus_cmap, bus_cmap_norm)

        # Initiate NetworkPlotter
        plotter = NetworkMapPlotter(n)
        plotter.init_layout(layouter)
        buses = (
            bus_sizes.index if not multindex_buses else bus_sizes.index.unique(level=0)
        )
        plotter.init_boundaries(buses, boundaries, margin)
        plotter.init_axis(ax, projection, geomap, geomap_colors, title)

        # Add jitter if given
        if jitter is not None:
            plotter.add_jitter(jitter)

        # Plot buses
        bus_sizes = bus_sizes.sort_index(level=0, sort_remaining=False)
        if geomap:
            bus_sizes = bus_sizes * plotter.area_factor**2
        if isinstance(bus_sizes.index, pd.MultiIndex):
            patches = plotter.get_multiindex_buses(
                bus_sizes, bus_colors, bus_alpha, bus_split_circles
            )
        else:
            patches = plotter.get_singleindex_buses(bus_sizes, bus_colors, bus_alpha)
        bus_collection = PatchCollection(patches, match_original=True, zorder=5)
        plotter.ax.add_collection(bus_collection)

        # Plot branches and flows
        if branch_components is None:
            branch_components = n.branch_components

        branch_collections = {}
        flow_collections = {}

        for c in n.iterate_components(branch_components):
            # Get branch collection
            if c.name == "Line":
                widths = line_widths
                colors = line_colors
                alpha = line_alpha
                flow = plotter._flow_ds_from_arg(line_flow, c.name)
                cmap = line_cmap
                cmap_norm = line_cmap_norm
            elif c.name == "Link":
                widths = link_widths
                colors = link_colors
                alpha = link_alpha
                flow = plotter._flow_ds_from_arg(link_flow, c.name)
                cmap = link_cmap
                cmap_norm = link_cmap_norm
            elif c.name == "Transformer":
                widths = transformer_widths
                colors = transformer_colors
                alpha = transformer_alpha
                flow = plotter._flow_ds_from_arg(transformer_flow, c.name)
                cmap = transformer_cmap
                cmap_norm = transformer_cmap_norm

            data = plotter._dataframe_from_arguments(
                c.df.index, widths=widths, colors=colors, alpha=alpha, flow=flow
            )
            data["colors"] = _apply_cmap(data.colors, cmap, cmap_norm)

            branch_coll = plotter.get_branch_collection(
                c, data.widths, data.colors, data.alpha, geometry
            )
            if branch_coll is not None:
                plotter.ax.add_collection(branch_coll)
                branch_collections[c.name] = branch_coll

            # Get flow collection if flow data exists
            if flow is not None:
                if auto_scale_flow:
                    rough_scale = sum(len(n.df(c)) for c in branch_components) + 100
                    data["flow"] = (
                        data.flow.mul(abs(widths), fill_value=0) / rough_scale
                    )
                flow_coll = plotter.get_flow_collection(
                    c,
                    flow=data.flow,
                    widths=data.widths,
                    colors=data.colors,
                    alpha=data.alpha,
                )
                if flow_coll is not None:
                    plotter.ax.add_collection(flow_coll)
                    flow_collections[c.name] = flow_coll

        return {
            "nodes": {"Bus": bus_collection},
            "branches": branch_collections,
            "flows": flow_collections,
        }

    def plot_balance_map(
        self,
        carrier: str,
        figsize: tuple = (8, 8),
        margin: float = 0.1,
        projection: cartopy.crs.Projection = cartopy.crs.PlateCarree(),
        bus_area_fraction: float = 0.02,
        flow_area_fraction: float = 0.02,
        draw_legend_circles: bool = True,
        draw_legend_arrows: bool = True,
        plot_kwargs: dict = {},
    ) -> tuple[plt.Figure, plt.Axes]:
        """
        Plot energy balance map for a given carrier showing bus sizes and transmission flows.

        Parameters
        ----------
        network : pypsa.Network
            PyPSA network to plot
        carrier : str, optional
            Energy carrier to plot, by default "H2"
        figsize : tuple, optional
            Figure dimensions (width, height), by default (8, 8)
        margin : float, optional
            Margin around the map bounds, by default 0.1
        bus_area_fraction : float, optional
            Target fraction of map area for bus scaling, by default 0.02
        flow_area_fraction : float, optional
            Target fraction of map area for flow scaling, by default 0.02
        draw_legend_circles : bool, optional
            Whether to draw the bus size legend, by default True
        draw_legend_arrows : bool, optional
            Whether to draw the flow arrows legend, by default True
        plot_kwargs : dict, optional
            Additional kwargs passed to network.plot(), by default None

        Returns
        -------
        tuple[plt.Figure, plt.Axes]
            Matplotlib figure and axes objects
        """
        n = self.n
        s = n.statistics

        # Calculate energy balance
        balance = s.energy_balance(
            bus_carrier=carrier,
            groupby=s.groupers.get_bus_and_carrier,
            nice_names=False,
        )
        transmission_carriers = get_transmission_carriers(n, bus_carrier=carrier)
        if "Link" in balance.index.get_level_values("component"):
            sub = balance.loc[["Link"]].drop(
                transmission_carriers.unique(1), level="carrier", errors="ignore"
            )
            balance = pd.concat([balance.drop("Link"), sub])
        balance = balance.groupby(["bus", "carrier"]).sum()

        # Calculate map bounds and scaling
        unique_buses = balance.index.get_level_values("bus").unique()
        (x_min, x_max), (y_min, y_max) = compute_bbox(
            n.buses.x[unique_buses], n.buses.y[unique_buses], margin
        )
        bus_size_factor = NetworkMapPlotter.scaling_factor_from_area_contribution(
            balance, x_min, x_max, y_min, y_max, target_area_fraction=bus_area_fraction
        )

        # Calculate flows
        flow = s.transmission(groupby=False, bus_carrier=carrier)
        flow = NetworkMapPlotter.aggregate_flow_by_connection(flow, n.branches())
        flow_scaling_factor = self.scaling_factor_from_area_contribution(
            flow, x_min, x_max, y_min, y_max, target_area_fraction=flow_area_fraction
        )
        flow_scaled = flow * flow_scaling_factor

        # Create plot
        fig, ax = plt.subplots(
            figsize=figsize,
            subplot_kw=dict(projection=projection),
        )

        plot_args = dict(
            bus_sizes=bus_size_factor * balance,
            bus_split_circles=True,
            ax=ax,
            line_widths=NetworkMapPlotter.flow_to_width(flow_scaled.get("Line", 0)),
            link_widths=NetworkMapPlotter.flow_to_width(flow_scaled.get("Link", 0)),
            line_flow=flow_scaled.get("Line"),
            link_flow=flow_scaled.get("Link"),
            auto_scale_flow=False,
        )
        if plot_kwargs is not None:
            plot_args.update(plot_kwargs)

        n.plot(**plot_args)

        # Add legends
        if draw_legend_circles:
            legend_representatives = get_legend_representatives(
                balance, group_on_first_level=True
            )
            add_legend_semicircles(
                ax,
                [s * bus_size_factor for s, label in legend_representatives],
                [label for s, label in legend_representatives],
                legend_kw={
                    "bbox_to_anchor": (0, 1),
                    "loc": "upper left",
                    "frameon": False,
                },
            )

        if draw_legend_arrows:
            legend_representatives = get_legend_representatives(
                flow, n_significant=1, base_unit="MWh"
            )
            add_legend_arrows(
                ax,
                [s * flow_scaling_factor * 10 for s, label in legend_representatives],
                [label for s, label in legend_representatives],
                arrow_to_tail_width=0.2,
                legend_kw={
                    "bbox_to_anchor": (0, 0.9),
                    "loc": "upper left",
                    "frameon": False,
                },
            )

        return fig, ax


@deprecated_kwargs(
    bus_norm="bus_cmap_norm",
    line_norm="line_cmap_norm",
    link_norm="link_cmap_norm",
    transformer_norm="transformer_cmap_norm",
    color_geomap="geomap_colors",
)
@wraps(NetworkMapPlotter.plot)
def plot(*args, **kwargs):
    # Get the signature from the classmethod
    plot.__doc__ = NetworkMapPlotter.plot.__doc__
    plot.__annotations__ = NetworkMapPlotter.plot.__annotations__
    return NetworkMapPlotter.plot(*args, **kwargs)


class HandlerCircle(HandlerPatch):
    """
    Legend Handler used to create circles for legend entries.

    This handler resizes the circles in order to match the same
    dimensional scaling as in the applied axis.
    """

    def create_artists(
        self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans
    ):
        fig = legend.get_figure()
        ax = legend.axes

        # take minimum to protect against too uneven x- and y-axis extents
        unit = min(np.diff(ax.transData.transform([(0, 0), (1, 1)]), axis=0)[0])
        radius = orig_handle.get_radius() * (72 / fig.dpi) * unit
        center = 5 - xdescent, 3 - ydescent
        p = plt.Circle(center, radius)
        self.update_prop(p, orig_handle, legend)
        p.set_transform(trans)
        return [p]


class WedgeHandler(HandlerPatch):
    """
    Legend Handler used to create sermi-circles for legend entries.

    This handler resizes the semi-circles in order to match the same
    dimensional scaling as in the applied axis.
    """

    def create_artists(
        self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans
    ):
        fig = legend.get_figure()
        ax = legend.axes
        center = 5 - xdescent, 3 - ydescent
        unit = min(np.diff(ax.transData.transform([(0, 0), (1, 1)]), axis=0)[0])
        r = orig_handle.r * (72 / fig.dpi) * unit
        p = Wedge(
            center=center,
            r=r,
            theta1=orig_handle.theta1,
            theta2=orig_handle.theta2,
        )
        self.update_prop(p, orig_handle, legend)
        p.set_transform(trans)
        return [p]


class HandlerArrow(HandlerPatch):
    """Handler for FancyArrow patches in legends."""

    def __init__(self, width_ratio=0.2):
        super().__init__()
        self.width_ratio = width_ratio

    def create_artists(
        self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans
    ):
        # Create a transformed arrow that fits in the legend box
        arrow = FancyArrow(
            -2,
            ydescent + height / 2,
            4,
            0,
            head_width=orig_handle._head_width,
            head_length=orig_handle._head_length,
            length_includes_head=False,
            width=orig_handle._head_width * self.width_ratio,  # Use the passed ratio
            color=orig_handle.get_facecolor(),
            **{
                k: getattr(orig_handle, f"get_{k}")()
                for k in ["edgecolor", "linewidth", "alpha"]
            },
        )
        return [arrow]


def add_legend_lines(ax, sizes, labels, colors=None, patch_kw=None, legend_kw=None):
    """
    Add a legend for lines and links.

    Parameters
    ----------
    ax : matplotlib ax
    sizes : list-like, float
        Size of the line reference; for example [3, 2, 1]
    labels : list-like, str
        Label of the line reference; for example ["30 GW", "20 GW", "10 GW"]
    colors: list-like, str
        Color of the line reference; for example ["red, "green", "blue"]
    patch_kw : defaults to {}
        Keyword arguments passed to plt.Line2D
    legend_kw : defaults to {}
        Keyword arguments passed to ax.legend
    """
    sizes = np.atleast_1d(sizes)
    labels = np.atleast_1d(labels)
    colors = [] if colors is None else np.atleast_1d(colors)
    if patch_kw is None:
        patch_kw = {}
    if legend_kw is None:
        legend_kw = {}

    if len(sizes) != len(labels):
        msg = "Sizes and labels must have the same length."
        raise ValueError(msg)
    elif len(colors) > 0 and len(sizes) != len(colors):
        msg = "Sizes, labels, and colors must have the same length."
        raise ValueError(msg)

    if len(colors) == 0:
        handles = [plt.Line2D([0], [0], linewidth=s, **patch_kw) for s in sizes]
    else:
        handles = [
            plt.Line2D([0], [0], linewidth=s, color=c, **patch_kw)
            for s, c in zip(sizes, colors)
        ]

    legend = ax.legend(handles, labels, **legend_kw)

    ax.get_figure().add_artist(legend)


def add_legend_patches(ax, colors, labels, patch_kw=None, legend_kw=None):
    """
    Add patches for color references.

    Parameters
    ----------
    ax : matplotlib ax
    colors : list-like, float
        Color of the patch; for example ["r", "g", "b"]
    labels : list-like, str
        Label of the patch; for example ["wind", "solar", "gas"]
    patch_kw : defaults to {}
        Keyword arguments passed to matplotlib.patches.Patch
    legend_kw : defaults to {}
        Keyword arguments passed to ax.legend
    """
    colors = np.atleast_1d(colors)
    labels = np.atleast_1d(labels)
    if patch_kw is None:
        patch_kw = {}
    if legend_kw is None:
        legend_kw = {}

    if len(colors) != len(labels):
        msg = "Colors and labels must have the same length."
        raise ValueError(msg)

    handles = [Patch(facecolor=c, **patch_kw) for c in colors]

    legend = ax.legend(handles, labels, **legend_kw)

    ax.get_figure().add_artist(legend)


def add_legend_circles(ax, sizes, labels, srid=4326, patch_kw=None, legend_kw=None):
    """
    Add a legend for reference circles.

    Parameters
    ----------
    ax : matplotlib ax
    sizes : list-like, float
        Size of the reference circle; for example [3, 2, 1]
    labels : list-like, str
        Label of the reference circle; for example ["30 GW", "20 GW", "10 GW"]
    patch_kw : defaults to {}
        Keyword arguments passed to matplotlib.patches.Circle
    legend_kw : defaults to {}
        Keyword arguments passed to ax.legend
    """
    sizes = np.atleast_1d(sizes)
    labels = np.atleast_1d(labels)
    if patch_kw is None:
        patch_kw = {}
    if legend_kw is None:
        legend_kw = {}

    if len(sizes) != len(labels):
        msg = "Sizes and labels must have the same length."
        raise ValueError(msg)

    if hasattr(ax, "projection"):
        area_correction = get_projected_area_factor(ax, srid) ** 2
        sizes = [s * area_correction for s in sizes]

    handles = [Circle((0, 0), radius=s**0.5, **patch_kw) for s in sizes]

    legend = ax.legend(
        handles, labels, handler_map={Circle: HandlerCircle()}, **legend_kw
    )

    ax.get_figure().add_artist(legend)


def add_legend_semicircles(ax, sizes, labels, srid=4326, patch_kw={}, legend_kw={}):
    """
    Add a legend for reference semi-circles.

    Parameters
    ----------
    ax : matplotlib ax
    sizes : list-like, float
        Size of the reference circle; for example [3, 2, 1]
    labels : list-like, str
        Label of the reference circle; for example ["30 GW", "20 GW", "10 GW"]
    patch_kw : defaults to {}
        Keyword arguments passed to matplotlib.patches.Wedges
    legend_kw : defaults to {}
        Keyword arguments passed to ax.legend
    """
    sizes = np.atleast_1d(sizes)
    labels = np.atleast_1d(labels)

    assert len(sizes) == len(labels), "Sizes and labels must have the same length."

    if hasattr(ax, "projection"):
        area_correction = get_projected_area_factor(ax, srid) ** 2
        sizes = [s * area_correction for s in sizes]

    radius = [np.sign(s) * np.abs(s) ** 0.5 for s in sizes]
    handles = [
        Wedge((0, -r / 2), r=r, theta1=0, theta2=180, **patch_kw) for r in radius
    ]

    legend = ax.legend(
        handles, labels, handler_map={Wedge: WedgeHandler()}, **legend_kw
    )

    ax.get_figure().add_artist(legend)


def add_legend_arrows(
    ax,
    sizes,
    labels,
    colors=None,
    arrow_to_tail_width=0.2,
    patch_kw=None,
    legend_kw=None,
):
    """
    Add a legend for flow arrows.

    Parameters
    ----------
    ax : matplotlib ax
    sizes : list-like, float
        Size of the reference arrows; for example [3, 2, 1]
    labels : list-like, str
        Label of the reference arrows; for example ["30 GW", "20 GW", "10 GW"]
    colors : str/list-like, default 'b'
        Color(s) of the arrows
    patch_kw : dict, optional
        Keyword arguments passed to FancyArrow
    legend_kw : dict, optional
        Keyword arguments passed to ax.legend
    """
    sizes = np.atleast_1d(sizes)
    labels = np.atleast_1d(labels)
    colors = np.atleast_1d(colors)

    if patch_kw is None:
        patch_kw = dict(linewidth=1, zorder=4)
    if legend_kw is None:
        legend_kw = {}

    if len(sizes) != len(labels):
        msg = "Sizes and labels must have the same length."
        raise ValueError(msg)

    if len(colors) == 1:
        colors = np.repeat(colors, len(sizes))
    elif len(colors) != len(sizes):
        msg = "Colors must be a single value or match length of sizes"
        raise ValueError(msg)

    # Scale sizes to be more visible in legend
    handles = [
        FancyArrow(
            0,
            0,
            1,
            0,  # Shorter arrow length
            head_width=s,
            head_length=s / 0.6,
            length_includes_head=False,
            color=c,
            **patch_kw,
        )
        for s, c in zip(sizes, colors)
    ]

    legend = ax.legend(
        handles,
        labels,
        handler_map={FancyArrow: HandlerArrow(width_ratio=arrow_to_tail_width)},
        **legend_kw,
    )
    ax.get_figure().add_artist(legend)


def round_to_significant_digits(x: float, n: int = 2) -> int | float:
    """Round a number to n significant figures."""
    if x == 0:
        return 0
    magnitude = int(np.floor(np.log10(abs(x))))
    rounded = round(x, -magnitude + (n - 1))
    return int(rounded) if rounded >= 1 else rounded


def scaled_legend_label(value: float, base_unit: str = "MWh") -> str:
    """
    Scale a value to an appropriate unit and return the scaled value and new unit.
    Ensures scaled values >= 1 are integers.
    """
    unit_scales = {
        "": 1,  # base
        "k": 1e3,  # kilo
        "M": 1e6,  # mega
        "G": 1e9,  # giga
        "T": 1e12,  # tera
        "P": 1e15,  # peta
    }

    # Extract base unit without prefix
    base_prefix = ""
    unit_name = base_unit
    for prefix in sorted(unit_scales.keys(), key=len, reverse=True):
        if base_unit.startswith(prefix):
            base_prefix = prefix
            unit_name = base_unit[len(prefix) :]
            break

    # Calculate absolute value in base units
    base_value = value * unit_scales[base_prefix]

    # Find appropriate prefix
    magnitude = np.floor(np.log10(abs(base_value))) if base_value != 0 else 0

    # Get closest unit scale that keeps value between 1 and 1000
    scales = np.array(list(unit_scales.values()))
    prefixes = list(unit_scales.keys())
    target_scale_idx = np.searchsorted(scales, 10 ** (magnitude - 3))
    if target_scale_idx >= len(scales):
        target_scale_idx = len(scales) - 1

    target_scale = scales[target_scale_idx]
    target_prefix = prefixes[target_scale_idx]

    # If base_unit already has a prefix, adjust the scale accordingly
    if base_prefix:
        # Calculate the relative scale between target and base prefix
        scale_difference = target_scale / unit_scales[base_prefix]
        scaled_value = value / scale_difference
    else:
        scaled_value = base_value / target_scale

    # Convert to integer if >= 1
    if abs(scaled_value) >= 1:
        scaled_value = int(round(scaled_value))

    return f"{scaled_value} {target_prefix}{unit_name}"


def get_legend_representatives(
    series: pd.Series,
    quantiles: list[float] = [0.6, 0.2],
    n_significant: int = 1,
    base_unit: str = "MWh",
    group_on_first_level: bool = False,
) -> list[tuple[int | float, str]]:
    """
    Get representative values from a numeric series for legend visualization,
    with automatic unit scaling. Values >= 1 are returned as integers.

    Parameters
    ----------
    series : pd.Series
        Series containing the values
    quantiles : list
        List of quantile to use assuming a uniform distribution from
        0 to the maximum value (default: [0.6, 0.2])
    n_significant : int
        Number of significant figures to round to
    base_unit : str
        Base unit of the values (default: "MWh")

    Returns
    -------
    list
        List of tuples (scaled_value, unit) for each quantile
    """
    if series.empty:
        return []
    if group_on_first_level:
        series = series.abs().groupby(level=0).sum()
    max_value = series.abs().max()
    values = [max_value * q for q in quantiles]
    rounded_values = [round_to_significant_digits(v, n_significant) for v in values]

    return [(v, scaled_legend_label(v, base_unit)) for v in rounded_values]


_token_required_mb_styles = [
    "basic",
    "streets",
    "outdoors",
    "light",
    "dark",
    "satellite",
    "satellite-streets",
]

_open__mb_styles = [
    "open-street-map",
    "white-bg",
    "carto-positron",
    "carto-darkmatter",
    "stamen-terrain",
    "stamen-toner",
    "stamen-watercolor",
]

# This function was borne out of a breakout group at the October 2017
# Munich Open Energy Modelling Initiative Workshop to hack together a
# working example of plotly for networks, see:
# https://forum.openmod-initiative.org/t/breakout-group-on-visualising-networks-with-plotly/384/7

# We thank Bryn Pickering for holding the tutorial on plotly which
# inspired the breakout group and for contributing ideas to the iplot
# function below.


def as_branch_series(ser, arg, c, n):
    ser = pd.Series(ser, index=n.df(c).index)
    if ser.isnull().any():
        msg = f"{c}_{arg}s does not specify all "
        f"entries. Missing values for {c}: {list(ser[ser.isnull()].index)}"
        raise ValueError(msg)
    return ser


def iplot(
    n,
    fig=None,
    bus_colors="cadetblue",
    bus_alpha=1,
    bus_sizes=10,
    bus_cmap=None,
    bus_colorbar=None,
    bus_text=None,
    line_colors="rosybrown",
    link_colors="darkseagreen",
    transformer_colors="orange",
    line_widths=3,
    link_widths=3,
    transformer_widths=3,
    line_text=None,
    link_text=None,
    transformer_text=None,
    layouter=None,
    title="",
    size=None,
    branch_components=None,
    iplot=True,
    jitter=None,
    mapbox=False,
    mapbox_style="open-street-map",
    mapbox_token="",
    mapbox_parameters=None,
):
    """
    Plot the network buses and lines interactively using plotly.

    Parameters
    ----------
    fig : dict, default None
        If not None, figure is built upon this fig.
    bus_colors : dict/pandas.Series
        Colors for the buses, defaults to "cadetblue". If bus_sizes is a
        pandas.Series with a Multiindex, bus_colors defaults to the
        n.carriers['color'] column.
    bus_alpha : float
        Adds alpha channel to buses, defaults to 1.
    bus_sizes : float/pandas.Series
        Sizes of bus points, defaults to 10.
    bus_cmap : plt.cm.ColorMap/str
        If bus_colors are floats, this color map will assign the colors
    bus_colorbar : dict
        Plotly colorbar, e.g. {'title' : 'my colorbar'}
    bus_text : pandas.Series
        Text for each bus, defaults to bus names
    line_colors : str/pandas.Series
        Colors for the lines, defaults to 'rosybrown'.
    link_colors : str/pandas.Series
        Colors for the links, defaults to 'darkseagreen'.
    transfomer_colors : str/pandas.Series
        Colors for the transfomer, defaults to 'orange'.
    line_widths : dict/pandas.Series
        Widths of lines, defaults to 1.5
    link_widths : dict/pandas.Series
        Widths of links, defaults to 1.5
    transformer_widths : dict/pandas.Series
        Widths of transformer, defaults to 1.5
    line_text : pandas.Series
        Text for lines, defaults to line names.
    link_text : pandas.Series
        Text for links, defaults to link names.
    tranformer_text : pandas.Series
        Text for transformers, defaults to transformer names.
    layouter : networkx.drawing.layout function, default None
        Layouting function from `networkx <https://networkx.github.io/>`_ which
        overrules coordinates given in ``n.buses[['x', 'y']]``. See
        `list <https://networkx.github.io/documentation/stable/reference/drawing.html#module-networkx.drawing.layout>`_
        of available options.
    title : string
        Graph title
    size : None|tuple
        Tuple specifying width and height of figure; e.g. (width, heigh).
    branch_components : list of str
        Branch components to be plotted, defaults to Line and Link.
    iplot : bool, default True
        Automatically do an interactive plot of the figure.
    jitter : None|float
        Amount of random noise to add to bus positions to distinguish
        overlapping buses
    mapbox : bool, default False
        Switch to use Mapbox.
    mapbox_style : str, default 'open-street-map'
        Define the mapbox layout style of the interactive plot. If this is set
        to a mapbox layout, the argument ``mapbox_token`` must be a valid Mapbox
        API access token.

        Valid open layouts are:
            open-street-map, white-bg, carto-positron, carto-darkmatter,
            stamen-terrain, stamen-toner, stamen-watercolor

        Valid mapbox layouts are:
            basic, streets, outdoors, light, dark, satellite, satellite-streets

    mapbox_token : string
        Mapbox API access token. Obtain from https://www.mapbox.com.
        Can also be included in mapbox_parameters as `accesstoken=mapbox_token`.
    mapbox_parameters : dict
        Configuration parameters of the Mapbox layout.
        E.g. {"bearing": 5, "pitch": 10, "zoom": 1, "style": 'dark'}.


    Returns
    -------
    fig: dictionary for plotly figure
    """
    if fig is None:
        fig = dict(data=[], layout={})

    if bus_text is None:
        bus_text = "Bus " + n.buses.index

    x, y = apply_layouter(n, layouter=layouter)

    rng = np.random.default_rng()  # Create a random number generator
    if jitter is not None:
        x = x + rng.uniform(low=-jitter, high=jitter, size=len(x))
        y = y + rng.uniform(low=-jitter, high=jitter, size=len(y))

    bus_trace = dict(
        x=x,
        y=y,
        text=bus_text,
        type="scatter",
        mode="markers",
        hoverinfo="text",
        opacity=bus_alpha,
        marker=dict(color=bus_colors, size=bus_sizes),
    )

    if bus_cmap is not None:
        bus_trace["marker"]["colorscale"] = bus_cmap

    if bus_colorbar is not None:
        bus_trace["marker"]["colorbar"] = bus_colorbar

    # Plot branches:
    if isinstance(line_widths, pd.Series):
        if isinstance(line_widths.index, pd.MultiIndex):
            raise TypeError(
                "Index of argument 'line_widths' is a Multiindex, "
                "this is not support since pypsa v0.17. "
                "Set differing widths with arguments 'line_widths', "
                "'link_widths' and 'transformer_widths'."
            )
    if isinstance(line_colors, pd.Series):
        if isinstance(line_colors.index, pd.MultiIndex):
            raise TypeError(
                "Index of argument 'line_colors' is a Multiindex, "
                "this is not support since pypsa v0.17. "
                "Set differing colors with arguments 'line_colors', "
                "'link_colors' and 'transformer_colors'."
            )

    if branch_components is None:
        branch_components = n.branch_components

    branch_colors = {
        "Line": line_colors,
        "Link": link_colors,
        "Transformer": transformer_colors,
    }
    branch_widths = {
        "Line": line_widths,
        "Link": link_widths,
        "Transformer": transformer_widths,
    }
    branch_text = {
        "Line": line_text,
        "Link": link_text,
        "Transformer": transformer_text,
    }

    shapes = []
    shape_traces = []

    for c in n.iterate_components(branch_components):
        b_widths = as_branch_series(branch_widths[c.name], "width", c.name, n)
        b_colors = as_branch_series(branch_colors[c.name], "color", c.name, n)
        b_text = branch_text[c.name]

        if b_text is None:
            b_text = c.name + " " + c.static.index

        x0 = c.static.bus0.map(x)
        x1 = c.static.bus1.map(x)
        y0 = c.static.bus0.map(y)
        y1 = c.static.bus1.map(y)

        for b in c.static.index:
            shapes.append(
                dict(
                    type="line",
                    opacity=0.8,
                    x0=x0[b],
                    y0=y0[b],
                    x1=x1[b],
                    y1=y1[b],
                    line=dict(color=b_colors[b], width=b_widths[b]),
                )
            )

        shape_traces.append(
            dict(
                x=0.5 * (x0 + x1),
                y=0.5 * (y0 + y1),
                text=b_text,
                type="scatter",
                mode="markers",
                hoverinfo="text",
                marker=dict(opacity=0.0),
            )
        )

    if mapbox:
        shape_traces_latlon = []
        for st in shape_traces:
            st["lon"] = st.pop("x")
            st["lat"] = st.pop("y")
            shape_traces_latlon.append(go.Scattermapbox(st))
        shape_traces = shape_traces_latlon

        shapes_mapbox = []
        for s in shapes:
            s["lon"] = [s.pop("x0"), s.pop("x1")]
            s["lat"] = [s.pop("y0"), s.pop("y1")]
            shapes_mapbox.append(go.Scattermapbox(s, mode="lines"))
        shapes = shapes_mapbox

        bus_trace["lon"] = bus_trace.pop("x")
        bus_trace["lat"] = bus_trace.pop("y")
        bus_trace = go.Scattermapbox(bus_trace)

        fig["data"].extend(shapes + shape_traces + [bus_trace])
    else:
        fig["data"].extend([bus_trace] + shape_traces)

    fig["layout"].update(dict(title=title, hovermode="closest", showlegend=False))

    if size is not None:
        if len(size) != 2:
            msg = "Parameter size must specify a tuple (width, height)."
            raise ValueError(msg)
        fig["layout"].update(dict(width=size[0], height=size[1]))

    if mapbox:
        if mapbox_token != "":
            mapbox_parameters["accesstoken"] = mapbox_token

        mapbox_parameters.setdefault("style", mapbox_style)

        if mapbox_parameters["style"] in _token_required_mb_styles:
            if "accesstoken" not in mapbox_parameters.keys():
                msg = (
                    "Using Mapbox layout styles requires a valid access token from "
                    "https://www.mapbox.com/, style which do not require a token "
                    "are:\n{', '.join(_open__mb_styles)}."
                )
                raise ValueError(msg)

        if "center" not in mapbox_parameters.keys():
            lon = (n.buses.x.min() + n.buses.x.max()) / 2
            lat = (n.buses.y.min() + n.buses.y.max()) / 2
            mapbox_parameters["center"] = dict(lat=lat, lon=lon)

        if "zoom" not in mapbox_parameters.keys():
            mapbox_parameters["zoom"] = 2

        fig["layout"]["mapbox"] = mapbox_parameters
    else:
        fig["layout"]["shapes"] = shapes

    if iplot:
        if not pltly_present:
            logger.warning("Plotly is not present, so interactive plotting won't work.")
        else:
            pltly.iplot(fig)

    return fig


def explore(
    n,
    crs=None,
    tooltip=True,
    popup=True,
    tiles="OpenStreetMap",
    components=None,
):
    """
    Create an interactive map displaying PyPSA network components using geopandas exlore() and folium.

    This function generates a Folium map showing buses, lines, links, and transformers from the provided network object.

    Parameters
    ----------
    n : PyPSA.Network object
        containing components `buses`, `links`, `lines`, `transformers`, `generators`, `loads`, and `storage_units`.
    crs : str, optional. If not specified, it will check whether `n.crs` exists and use it, else it will default to "EPSG:4326".
        Coordinate Reference System for the GeoDataFrames.
    tooltip : bool, optional, default=True
        Whether to include tooltips (on hover) for the features.
    popup : bool, optional, default=True
        Whether to include popups (on click) for the features.
    tiles : str, optional, default="OpenStreetMap"
        The tileset to use for the map. Options include "OpenStreetMap", "CartoDB Positron", and "CartoDB dark_matter".
    components : list-like, optional, default=None
        The components to plot. Default includes "Bus", "Line", "Link", "Transformer".

    Returns
    -------
    folium.Map
        A Folium map object with the PyPSA.Network components plotted.
    """
    try:
        import mapclassify  # noqa: F401
        from folium import Element, LayerControl, Map, TileLayer
    except ImportError:
        logger.warning(
            "folium and mapclassify need to be installed to use `n.explore()`."
        )
        return None

    if n.crs and crs is None:
        crs = n.crs
    else:
        crs = "EPSG:4326"

    if components is None:
        components = {"Bus", "Line", "Transformer", "Link"}

    # Map related settings
    bus_colors = mcolors.CSS4_COLORS["cadetblue"]
    line_colors = mcolors.CSS4_COLORS["rosybrown"]
    link_colors = mcolors.CSS4_COLORS["darkseagreen"]
    transformer_colors = mcolors.CSS4_COLORS["orange"]
    generator_colors = mcolors.CSS4_COLORS["purple"]
    load_colors = mcolors.CSS4_COLORS["red"]
    storage_unit_colors = mcolors.CSS4_COLORS["black"]

    # Initialize the map
    map = Map(tiles=None)

    # Add map title
    map_title = "PyPSA Network" + (f": {n.name}" if n.name else "")
    map.get_root().html.add_child(
        Element(
            f"<h4 style='position:absolute;z-index:100000;left:1vw;bottom:5px'>{map_title}</h4>"
        )
    )

    # Add tile layer legend entries
    TileLayer(tiles, name=tiles).add_to(map)

    components_possible = [
        "Bus",
        "Line",
        "Link",
        "Transformer",
        "Generator",
        "Load",
        "StorageUnit",
    ]
    components_present = []

    if not n.transformers.empty and "Transformer" in components:
        x1 = n.transformers.bus0.map(n.buses.x)
        y1 = n.transformers.bus0.map(n.buses.y)
        x2 = n.transformers.bus1.map(n.buses.x)
        y2 = n.transformers.bus1.map(n.buses.y)
        valid_rows = ~(x1.isna() | y1.isna() | x2.isna() | y2.isna())

        if num_invalid := sum(~valid_rows):
            logger.info(
                f"Omitting {num_invalid} transformers due to missing coordinates"
            )

        gdf_transformers = gpd.GeoDataFrame(
            n.transformers[valid_rows],
            geometry=linestrings(
                np.stack(
                    [
                        (x1[valid_rows], y1[valid_rows]),
                        (x2[valid_rows], y2[valid_rows]),
                    ],
                    axis=1,
                ).T
            ),
            crs=crs,
        )

        gdf_transformers[gdf_transformers.is_valid].explore(
            m=map,
            color=transformer_colors,
            tooltip=tooltip,
            popup=popup,
            name="Transformers",
        )
        components_present.append("Transformer")

    if not n.lines.empty and "Line" in components:
        x1 = n.lines.bus0.map(n.buses.x)
        y1 = n.lines.bus0.map(n.buses.y)
        x2 = n.lines.bus1.map(n.buses.x)
        y2 = n.lines.bus1.map(n.buses.y)
        valid_rows = ~(x1.isna() | y1.isna() | x2.isna() | y2.isna())

        if num_invalid := sum(~valid_rows):
            logger.info(f"Omitting {num_invalid} lines due to missing coordinates.")

        gdf_lines = gpd.GeoDataFrame(
            n.lines[valid_rows],
            geometry=linestrings(
                np.stack(
                    [
                        (x1[valid_rows], y1[valid_rows]),
                        (x2[valid_rows], y2[valid_rows]),
                    ],
                    axis=1,
                ).T
            ),
            crs=crs,
        )

        gdf_lines[gdf_lines.is_valid].explore(
            m=map, color=line_colors, tooltip=tooltip, popup=popup, name="Lines"
        )
        components_present.append("Line")

    if not n.links.empty and "Link" in components:
        x1 = n.links.bus0.map(n.buses.x)
        y1 = n.links.bus0.map(n.buses.y)
        x2 = n.links.bus1.map(n.buses.x)
        y2 = n.links.bus1.map(n.buses.y)
        valid_rows = ~(x1.isna() | y1.isna() | x2.isna() | y2.isna())

        if num_invalid := sum(~valid_rows):
            logger.info(f"Omitting {num_invalid} links due to missing coordinates.")

        gdf_links = gpd.GeoDataFrame(
            n.links[valid_rows],
            geometry=linestrings(
                np.stack(
                    [
                        (x1[valid_rows], y1[valid_rows]),
                        (x2[valid_rows], y2[valid_rows]),
                    ],
                    axis=1,
                ).T
            ),
            crs="EPSG:4326",
        )

        gdf_links[gdf_links.is_valid].explore(
            m=map, color=link_colors, tooltip=tooltip, popup=popup, name="Links"
        )
        components_present.append("Link")

    if not n.buses.empty and "Bus" in components:
        gdf_buses = gpd.GeoDataFrame(
            n.buses, geometry=gpd.points_from_xy(n.buses.x, n.buses.y), crs=crs
        )

        gdf_buses[gdf_buses.is_valid].explore(
            m=map,
            color=bus_colors,
            tooltip=tooltip,
            popup=popup,
            name="Buses",
            marker_kwds={"radius": 4},
        )
        components_present.append("Bus")

    if not n.generators.empty and "Generator" in components:
        gdf_generators = gpd.GeoDataFrame(
            n.generators,
            geometry=gpd.points_from_xy(
                n.generators.bus.map(n.buses.x), n.generators.bus.map(n.buses.y)
            ),
            crs=crs,
        )

        gdf_generators[gdf_generators.is_valid].explore(
            m=map,
            color=generator_colors,
            tooltip=tooltip,
            popup=popup,
            name="Generators",
            marker_kwds={"radius": 2.5},
        )
        components_present.append("Generator")

    if not n.loads.empty and "Load" in components:
        loads = n.loads.copy()
        loads["p_set_sum"] = n.loads_t.p_set.sum(axis=0).round(1)
        gdf_loads = gpd.GeoDataFrame(
            loads,
            geometry=gpd.points_from_xy(
                loads.bus.map(n.buses.x), loads.bus.map(n.buses.y)
            ),
            crs=crs,
        )

        gdf_loads[gdf_loads.is_valid].explore(
            m=map,
            color=load_colors,
            tooltip=tooltip,
            popup=popup,
            name="Loads",
            marker_kwds={"radius": 1.5},
        )
        components_present.append("Load")

    if not n.storage_units.empty and "StorageUnit" in components:
        gdf_storage_units = gpd.GeoDataFrame(
            n.storage_units,
            geometry=gpd.points_from_xy(
                n.storage_units.bus.map(n.buses.x), n.storage_units.bus.map(n.buses.y)
            ),
            crs=crs,
        )

        gdf_storage_units[gdf_storage_units.is_valid].explore(
            m=map,
            color=storage_unit_colors,
            tooltip=tooltip,
            popup=popup,
            name="Storage Units",
            marker_kwds={"radius": 1},
        )
        components_present.append("StorageUnit")

    if len(components_present) > 0:
        logger.info(
            f"Components rendered on the map: {', '.join(sorted(components_present))}."
        )
    if len(set(components) - set(components_present)) > 0:
        logger.info(
            f"Components omitted as they are missing or not selected: {', '.join(sorted(set(components_possible) - set(components_present)))}."
        )

    # Set the default view to the bounds of the elements in the map
    map.fit_bounds(map.get_bounds())

    # Add a Layer Control to toggle layers on and off
    LayerControl().add_to(map)

    return map
