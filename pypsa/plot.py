"""
Functions for plotting networks.
"""

from __future__ import annotations

__author__ = (
    "PyPSA Developers, see https://pypsa.readthedocs.io/en/latest/developers.html"
)
__copyright__ = (
    "Copyright 2015-2024 PyPSA Developers, see https://pypsa.readthedocs.io/en/latest/developers.html, "
    "MIT License"
)

import logging
import warnings
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.legend_handler import HandlerPatch
from matplotlib.patches import Circle, FancyArrow, Patch, Wedge

from pypsa.geo import (
    compute_bbox,
    get_projected_area_factor,
    get_projection_from_crs,
)
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


class NetworkPlotter:
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

    def _add_geomap_features(self, geomap=True, geomap_colors=None):
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

        transform = get_projection_from_crs(self.n.srid)
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
                self._add_geomap_features(geomap, geomap_colors)

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
        self.x = self.x + np.random.uniform(low=-jitter, high=jitter, size=len(self.x))
        self.y = self.y + np.random.uniform(low=-jitter, high=jitter, size=len(self.y))

        return self.x, self.y

    def get_multiindex_busses(
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

    def get_singleindex_busses(self, sizes, colors, alpha):
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

    def _flow_ds_from_arg(self, flow, component_name):
        if isinstance(flow, pd.Series):
            return flow

        if flow in self.n.snapshots:
            return self.n.pnl(component_name).p0.loc[flow]

        if isinstance(flow, str) or callable(flow):
            return self.n.pnl(component_name).p0.agg(flow, axis=0)

    def _get_branch_collection(self, c, df, geometry):
        if not geometry:
            segments = np.asarray(
                (
                    (c.df.bus0[df.index].map(self.x), c.df.bus0[df.index].map(self.y)),
                    (c.df.bus1[df.index].map(self.x), c.df.bus1[df.index].map(self.y)),
                )
            ).transpose(2, 0, 1)
        else:
            from shapely.geometry import LineString
            from shapely.wkt import loads

            linestrings = c.df.geometry[lambda ds: ds != ""].map(loads)
            if not all(isinstance(ls, LineString) for ls in linestrings):
                msg = "The WKT-encoded geometry in the 'geometry' column must be "
                "composed of LineStrings"
                raise ValueError(msg)
            segments = np.asarray(list(linestrings.map(np.asarray)))

        branch_coll = LineCollection(
            segments,
            linewidths=df.width,
            antialiaseds=(1,),
            colors=df.color,
            alpha=df.alpha,
        )

        return branch_coll

    def get_branch_collections(self, branch_components, geometry, branch_data):
        components = self.n.iterate_components(branch_components)

        branch_colls = {}
        for c in components:
            d = branch_data[c.name]
            if any([isinstance(v, pd.Series) for _, v in d.items()]):
                df = pd.DataFrame(d)
            else:
                df = pd.DataFrame(d, index=c.df.index)

            if df.empty:  # TODO: Can be removed since there are default values?
                continue

            branch_coll = self._get_branch_collection(c, df, geometry)
            branch_colls[c.name] = branch_coll

        return branch_colls

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

    def _get_flow_collection(self, c, df):
        # b_flow = .df.get("flow")
        # if b_flow is not None:
        coords = pd.DataFrame(
            {
                "x1": c.df.bus0.map(self.x),
                "y1": c.df.bus0.map(self.y),
                "x2": c.df.bus1.map(self.x),
                "y2": c.df.bus1.map(self.y),
            }
        )
        b_flow = df["flow"].mul(df.width.abs(), fill_value=0)
        # update the line width, allows to set line widths separately from flows
        # df.width.update((5 * b_flow.abs()).pipe(np.sqrt))
        flow_coll = self._directed_flow(
            coords, b_flow, df.color, self.area_factor, df.alpha
        )
        return flow_coll

    def get_flow_collections(self, branch_components, flow_data, branch_data):
        components = self.n.iterate_components(branch_components)

        flow_colls = {}
        for c in components:
            if flow_data[c.name] is None:
                continue
            else:
                flow_arg = flow_data[c.name]

            # Get general component data
            d = branch_data[c.name]
            if any([isinstance(v, pd.Series) for _, v in d.items()]):
                df = pd.DataFrame(d)
            else:
                df = pd.DataFrame(d, index=c.df.index)

            # Get flow data
            rough_scale = sum(len(self.n.df(c)) for c in branch_components) + 100
            df["flow"] = (
                self._flow_ds_from_arg(flow_arg, c.name) / rough_scale
            )  # TODO move this out

            flow_coll = self._get_flow_collection(c, df)
            flow_colls[c.name] = flow_coll

        return flow_colls


@deprecated_kwargs(
    bus_norm="bus_cmap_norm",
    line_norm="line_cmap_norm",
    link_norm="link_cmap_norm",
    transformer_norm="transformer_cmap_norm",
    color_geomap="geomap_colors",
)
def plot(
    n,
    layouter: nx.drawing.layout = None,
    boundaries: list | tuple = None,
    margin: float = 0.05,
    ax: plt.Axes = None,
    geomap: bool | str = True,
    projection: cartopy.crs.Projection = None,
    geomap_colors: dict | bool = None,
    title: str = "",
    jitter: float = None,
    branch_components: list = None,
    bus_sizes: float | dict | pd.Series = 2e-2,
    bus_split_circles: bool = False,
    bus_colors: str | dict | pd.Series = None,
    bus_cmap: str | plt.cm.ColorMap = None,
    bus_cmap_norm: plt.Normalize = None,
    bus_alpha: float | dict | pd.Series = 1,
    geometry=False,  # TODO
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
    flow=None,  # Deprecated
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
    line_colors = _convert_to_series(line_colors, n.lines.index)
    link_colors = _convert_to_series(link_colors, n.links.index)
    transformer_colors = _convert_to_series(transformer_colors, n.transformers.index)

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
    line_colors = _apply_cmap(line_colors, line_cmap, line_cmap_norm)
    link_colors = _apply_cmap(link_colors, link_cmap, link_cmap_norm)
    transformer_colors = _apply_cmap(
        transformer_colors, transformer_cmap, transformer_cmap_norm
    )

    # Initiate NetworkPlotter
    plotter = NetworkPlotter(n)
    plotter.init_layout(layouter)
    buses = bus_sizes.index if not multindex_buses else bus_sizes.index.unique(level=0)
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
        patches = plotter.get_multiindex_busses(
            bus_sizes, bus_colors, bus_alpha, bus_split_circles
        )
    else:
        patches = plotter.get_singleindex_busses(bus_sizes, bus_colors, bus_alpha)
    bus_collection = PatchCollection(patches, match_original=True, zorder=5)
    plotter.ax.add_collection(bus_collection)

    # Collect data for branches and flows
    branch_data = {
        "Line": {
            "width": line_widths,
            "color": line_colors,
            "alpha": line_alpha,
        },
        "Link": {
            "width": link_widths,
            "color": link_colors,
            "alpha": link_alpha,
        },
        "Transformer": {
            "width": transformer_widths,
            "color": transformer_colors,
            "alpha": transformer_alpha,
        },
    }
    if branch_components is None:
        branch_components = plotter.n.branch_components

    # Plot branches
    branches = plotter.get_branch_collections(
        branch_components,
        geometry,
        branch_data,
    )
    for branch in branches.values():
        plotter.ax.add_collection(branch)

    # Plot flows
    flow_data = {
        "Line": line_flow,
        "Link": link_flow,
        "Transformer": transformer_flow,
    }
    flows = plotter.get_flow_collections(
        branch_components,
        flow_data,
        branch_data,
    )
    for flow in flows.values():
        plotter.ax.add_collection(flow)

    return {"nodes": {"Bus": bus_collection}, "branches": branches, "flows": flows}


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


def add_legend_lines(ax, sizes, labels, colors=[], patch_kw={}, legend_kw={}):
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
    colors = np.atleast_1d(colors)

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


def add_legend_patches(ax, colors, labels, patch_kw={}, legend_kw={}):
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

    if len(colors) != len(labels):
        msg = "Colors and labels must have the same length."
        raise ValueError(msg)

    handles = [Patch(facecolor=c, **patch_kw) for c in colors]

    legend = ax.legend(handles, labels, **legend_kw)

    ax.get_figure().add_artist(legend)


def add_legend_circles(ax, sizes, labels, srid=4326, patch_kw={}, legend_kw={}):
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
    mapbox_parameters={},
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

    x, y = _get_coordinates(n, layouter=layouter)

    if jitter is not None:
        x, y = _add_jitter(x, y, jitter)

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
            b_text = c.name + " " + c.df.index

        x0 = c.df.bus0.map(x)
        x1 = c.df.bus1.map(x)
        y0 = c.df.bus0.map(y)
        y1 = c.df.bus1.map(y)

        for b in c.df.index:
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
