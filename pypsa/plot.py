# -*- coding: utf-8 -*-
"""
Functions for plotting networks.
"""

__author__ = (
    "PyPSA Developers, see https://pypsa.readthedocs.io/en/latest/developers.html"
)
__copyright__ = (
    "Copyright 2015-2024 PyPSA Developers, see https://pypsa.readthedocs.io/en/latest/developers.html, "
    "MIT License"
)

import logging

import networkx as nx
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.legend_handler import HandlerPatch
from matplotlib.patches import Circle, FancyArrow, Patch, Wedge

cartopy_present = True
try:
    import cartopy
    import cartopy.crs as ccrs
    import cartopy.mpl.geoaxes
except ImportError:
    cartopy_present = False

requests_present = True
try:
    import requests
except ImportError:
    requests_present = False

pltly_present = True
try:
    import plotly.graph_objects as go
    import plotly.offline as pltly
except ImportError:
    pltly_present = False


def plot(
    n,
    margin=0.05,
    ax=None,
    geomap=True,
    projection=None,
    bus_colors="cadetblue",
    bus_alpha=1,
    bus_sizes=2e-2,
    bus_cmap=None,
    bus_norm=None,
    bus_split_circles=False,
    line_colors="rosybrown",
    link_colors="darkseagreen",
    transformer_colors="orange",
    line_alpha=1,
    link_alpha=1,
    transformer_alpha=1,
    line_widths=1.5,
    link_widths=1.5,
    transformer_widths=1.5,
    line_cmap=None,
    link_cmap=None,
    transformer_cmap=None,
    line_norm=None,
    link_norm=None,
    transformer_norm=None,
    flow=None,
    branch_components=None,
    layouter=None,
    title="",
    boundaries=None,
    geometry=False,
    jitter=None,
    color_geomap=None,
):
    """
    Plot the network buses and lines using matplotlib and cartopy.

    Parameters
    ----------
    margin : float, defaults to 0.05
        Margin at the sides as proportion of distance between max/min x, y
        Will be ignored if boundaries are given.
    ax : matplotlib ax, defaults to plt.gca()
        Axis to which to plot the network
    geomap: bool/str, default True
        Switch to use Cartopy and draw geographical features.
        If string is passed, it will be used as a resolution argument,
        valid options are '10m', '50m' and '110m'.
    projection: cartopy.crs.Projection, defaults to None
        Define the projection of your geomap, only valid if cartopy is
        installed. If None (default) is passed the projection for cartopy
        is set to cartopy.crs.PlateCarree
    bus_colors : dict/pandas.Series
        Colors for the buses, defaults to "cadetblue". If bus_sizes is a
        pandas.Series with a Multiindex, bus_colors defaults to the
        n.carriers['color'] column.
    bus_alpha : float
        Adds alpha channel to buses, defaults to 1.
    bus_sizes : dict/pandas.Series
        Sizes of bus points, defaults to 1e-2. If a multiindexed Series is passed,
        the function will draw pies for each bus (first index level) with
        segments of different color (second index level). Such a Series is ob-
        tained by e.g. n.generators.groupby(['bus', 'carrier']).p_nom.sum()
    bus_cmap : plt.cm.ColorMap/str
        If bus_colors are floats, this color map will assign the colors
    bus_norm : plt.Normalize|matplotlib.colors.*Norm
        The norm applied to the bus_cmap.
    bus_split_circles : bool, default False
        Draw half circles if bus_sizes is a pandas.Series with a Multiindex.
        If set to true, the upper half circle per bus then includes all positive values
        of the series, the lower half circle all negative values. Defaults to False.
    line_colors : str/pandas.Series
        Colors for the lines, defaults to 'rosybrown'.
    link_colors : str/pandas.Series
        Colors for the links, defaults to 'darkseagreen'.
    transfomer_colors : str/pandas.Series
        Colors for the transfomer, defaults to 'orange'.
    line_alpha : str/pandas.Series
        Alpha for the lines, defaults to 1.
    link_alpha : str/pandas.Series
        Alpha for the links, defaults to 1.
    transfomer_alpha : str/pandas.Series
        Alpha for the transfomer, defaults to 1.
    line_widths : dict/pandas.Series
        Widths of lines, defaults to 1.5
    link_widths : dict/pandas.Series
        Widths of links, defaults to 1.5
    transformer_widths : dict/pandas.Series
        Widths of transformer, defaults to 1.5
    line_cmap : plt.cm.ColorMap/str|dict
        If line_colors are floats, this color map will assign the colors.
    link_cmap : plt.cm.ColorMap/str|dict
        If link_colors are floats, this color map will assign the colors.
    transformer_cmap : plt.cm.ColorMap/str|dict
        If transformer_colors are floats, this color map will assign the colors.
    line_norm : plt.Normalize|matplotlib.colors.*Norm
        The norm applied to the line_cmap.
    link_norm : plt.Normalize|matplotlib.colors.*Norm
        The norm applied to the link_cmap.
    transformer_norm : matplotlib.colors.Normalize|matplotlib.colors.*Norm
        The norm applied to the transformer_cmap.
    flow : snapshot/pandas.Series/function/string
        Flow to be displayed in the plot, defaults to None. If an element of
        n.snapshots is given, the flow at this timestamp will be
        displayed. If an aggregation function is given, is will be applied
        to the total network flow via pandas.DataFrame.agg (accepts also
        function names). Otherwise flows can be specified by passing a pandas
        Series with MultiIndex including all necessary branch components.
        Use the line_widths argument to additionally adjust the size of the
        flow arrows.
    layouter : networkx.drawing.layout function, default None
        Layouting function from `networkx <https://networkx.github.io/>`_ which
        overrules coordinates given in ``n.buses[['x', 'y']]``. See
        `list <https://networkx.github.io/documentation/stable/reference/drawing.html#module-networkx.drawing.layout>`_
        of available options.
    title : string
        Graph title
    boundaries : list of four floats
        Boundaries of the plot in format [x1, x2, y1, y2]
    branch_components : list of str
        Branch components to be plotted, defaults to Line and Link.
    jitter : None|float
        Amount of random noise to add to bus positions to distinguish
        overlapping buses
    color_geomap : dict or bool
        Specify colors to paint land and sea areas in.
        If True, it defaults to `{'ocean': 'lightblue', 'land': 'whitesmoke'}`.
        If no dictionary is provided, colors are white.
        If False, no geographical features are plotted.

    Returns
    -------
    bus_collection, branch_collection1, ... : tuple of Collections
        Collections for buses and branches.
    """

    if margin is None:
        logger.warning(
            "The `margin` argument does support None value anymore. "
            "Falling back to the default value 0.05. This will raise "
            "an error in the future."
        )
        margin = 0.05

    x, y = _get_coordinates(n, layouter=layouter)
    buses = n.buses.index
    if isinstance(bus_sizes, pd.Series):
        buses = bus_sizes.index
        if isinstance(buses, pd.MultiIndex):
            buses = buses.unique(0)

    if boundaries is None:
        boundaries = sum(
            zip(*compute_bbox_with_margins(margin, x[buses], y[buses])), ()
        )

    if geomap:
        if not cartopy_present:
            logger.warning("Cartopy needs to be installed to use `geomap=True`.")
            geomap = False
        if not requests_present:
            logger.warning("Requests needs to be installed to use `geomap=True`.")
            geomap = False

    if geomap:
        transform = get_projection_from_crs(n.srid)
        if projection is None:
            projection = transform
        else:
            assert isinstance(
                projection, cartopy.crs.Projection
            ), "The passed projection is not a cartopy.crs.Projection"

        if ax is None:
            ax = plt.axes(projection=projection)
        else:
            assert isinstance(ax, cartopy.mpl.geoaxes.GeoAxesSubplot), (
                "The passed axis is not a GeoAxesSubplot. You can "
                "create one with: \nimport cartopy.crs as ccrs \n"
                "fig, ax = plt.subplots("
                'subplot_kw={"projection":ccrs.PlateCarree()})'
            )

        x_, y_, _ = ax.projection.transform_points(transform, x.values, y.values).T
        x, y = pd.Series(x_, x.index), pd.Series(y_, y.index)

        if color_geomap is not False:
            draw_map_cartopy(ax, geomap, color_geomap)

        if boundaries is not None:
            ax.set_extent(boundaries, crs=transform)
    elif ax is None:
        ax = plt.gca()
    elif hasattr(ax, "projection"):
        raise ValueError("Axis is a geo axis, but `geomap` is set to False")
    if not geomap and boundaries:
        ax.axis(boundaries)

    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title)

    # Plot buses:

    if jitter is not None:
        x = x + np.random.uniform(low=-jitter, high=jitter, size=len(x))
        y = y + np.random.uniform(low=-jitter, high=jitter, size=len(y))

    patches = []
    if isinstance(bus_sizes, pd.Series) and isinstance(bus_sizes.index, pd.MultiIndex):
        # We are drawing pies to show all the different shares
        assert (
            len(bus_sizes.index.unique(level=0).difference(n.buses.index)) == 0
        ), "The first MultiIndex level of bus_sizes must contain buses"
        if isinstance(bus_colors, dict):
            bus_colors = pd.Series(bus_colors)
        # case bus_colors isn't a series or dict: look in n.carriers for existent colors
        if not isinstance(bus_colors, pd.Series):
            bus_colors = n.carriers.color.dropna()
        assert bus_sizes.index.unique(level=1).isin(bus_colors.index).all(), (
            "Colors not defined for all elements in the second MultiIndex "
            "level of bus_sizes, please make sure that all the elements are "
            "included in bus_colors or in n.carriers.color"
        )

        bus_sizes = bus_sizes.sort_index(level=0, sort_remaining=False)
        if geomap:
            bus_sizes = bus_sizes * projected_area_factor(ax, n.srid) ** 2

        for b_i in bus_sizes.index.unique(level=0):
            s_base = bus_sizes.loc[b_i]

            if bus_split_circles:
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
                            (x.at[b_i], y.at[b_i]),
                            radius,
                            scope * start,
                            scope * (start + ratio),
                            facecolor=bus_colors[i],
                            alpha=bus_alpha,
                        )
                    )
                    start = start + ratio
    else:
        c = pd.Series(bus_colors, index=n.buses.index)
        s = pd.Series(bus_sizes, index=n.buses.index, dtype="float")
        if geomap:
            s = s * projected_area_factor(ax, n.srid) ** 2

        if bus_cmap is not None and c.dtype is np.dtype("float"):
            if isinstance(bus_cmap, str):
                bus_cmap = plt.get_cmap(bus_cmap)
            if not bus_norm:
                bus_norm = plt.Normalize(vmin=c.min(), vmax=c.max())
            c = c.apply(lambda cval: bus_cmap(bus_norm(cval)))

        for b_i in s.index[(s != 0) & ~s.isna()]:
            radius = s.at[b_i] ** 0.5
            patches.append(
                Circle(
                    (x.at[b_i], y.at[b_i]), radius, facecolor=c.at[b_i], alpha=bus_alpha
                )
            )
    bus_collection = PatchCollection(patches, match_original=True, zorder=5)
    ax.add_collection(bus_collection)
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
    branch_alpha = {
        "Line": line_alpha,
        "Link": link_alpha,
        "Transformer": transformer_alpha,
    }
    branch_widths = {
        "Line": line_widths,
        "Link": link_widths,
        "Transformer": transformer_widths,
    }
    branch_cmap = {
        "Line": line_cmap,
        "Link": link_cmap,
        "Transformer": transformer_cmap,
    }
    branch_norm = {
        "Line": line_norm,
        "Link": link_norm,
        "Transformer": transformer_norm,
    }

    branch_collections = []
    arrow_collections = []

    if flow is not None:
        rough_scale = sum(len(n.df(c)) for c in branch_components) + 100
        flow = _flow_ds_from_arg(flow, n, branch_components) / rough_scale

    for c in n.iterate_components(branch_components):
        d = dict(
            width=branch_widths[c.name],
            color=branch_colors[c.name],
            alpha=branch_alpha[c.name],
        )
        if flow is not None and flow.get(c.name) is not None:
            d["flow"] = flow[c.name]

        if any([isinstance(v, pd.Series) for _, v in d.items()]):
            df = pd.DataFrame(d)
        else:
            df = pd.DataFrame(d, index=c.df.index)

        if df.empty:
            continue

        b_widths = df.width
        b_colors = df.color
        b_alpha = df.alpha
        b_nums = None
        b_cmap = branch_cmap[c.name]
        b_norm = branch_norm[c.name]
        b_flow = df.get("flow")

        if issubclass(b_colors.dtype.type, np.number):
            b_nums = b_colors
            b_colors = None

        if not geometry:
            segments = np.asarray(
                (
                    (c.df.bus0[df.index].map(x), c.df.bus0[df.index].map(y)),
                    (c.df.bus1[df.index].map(x), c.df.bus1[df.index].map(y)),
                )
            ).transpose(2, 0, 1)
        else:
            from shapely.geometry import LineString
            from shapely.wkt import loads

            linestrings = c.df.geometry[lambda ds: ds != ""].map(loads)
            assert all(isinstance(ls, LineString) for ls in linestrings), (
                "The WKT-encoded geometry in the 'geometry' column must be "
                "composed of LineStrings"
            )
            segments = np.asarray(list(linestrings.map(np.asarray)))

        if b_flow is not None:
            coords = pd.DataFrame(
                {
                    "x1": c.df.bus0.map(x),
                    "y1": c.df.bus0.map(y),
                    "x2": c.df.bus1.map(x),
                    "y2": c.df.bus1.map(y),
                }
            )
            b_flow = b_flow.mul(b_widths.abs(), fill_value=0)
            # update the line width, allows to set line widths separately from flows
            # b_widths.update((5 * b_flow.abs()).pipe(np.sqrt))
            area_factor = projected_area_factor(ax, n.srid)
            f_collection = directed_flow(
                coords, b_flow, b_colors, area_factor, b_cmap, b_alpha
            )
            if b_nums is not None:
                f_collection.set_array(np.asarray(b_nums))
                f_collection.set_cmap(b_cmap)
                f_collection.autoscale()
                f_collection.set(norm=b_norm)
            arrow_collections.append(f_collection)
            ax.add_collection(f_collection)

        b_collection = LineCollection(
            segments,
            linewidths=b_widths,
            antialiaseds=(1,),
            colors=b_colors,
            alpha=b_alpha,
        )

        if b_nums is not None:
            b_collection.set_array(np.asarray(b_nums))
            b_collection.set_cmap(b_cmap)
            b_collection.autoscale()
            b_collection.set(norm=b_norm)

        ax.add_collection(b_collection)
        b_collection.set_zorder(3)
        branch_collections.append(b_collection)

    return (bus_collection,) + tuple(branch_collections) + tuple(arrow_collections)


def as_branch_series(ser, arg, c, n):
    ser = pd.Series(ser, index=n.df(c).index)
    assert not ser.isnull().any(), (
        f"{c}_{arg}s does not specify all "
        f"entries. Missing values for {c}: {list(ser[ser.isnull()].index)}"
    )
    return ser


def get_projection_from_crs(crs):
    if crs == 4326:
        # if data is in latlon system, return default map with latlon system
        return ccrs.PlateCarree()
    try:
        return ccrs.epsg(crs)
    except requests.RequestException:
        logger.warning(
            "A connection to http://epsg.io/ is "
            "required for a projected coordinate reference system. "
            "Falling back to latlong."
        )
    except ValueError:
        logger.warning(
            "'{crs}' does not define a projected coordinate system. "
            "Falling back to latlong.".format(crs=crs)
        )
        return ccrs.PlateCarree()


def compute_bbox_with_margins(margin, x, y):
    """
    Helper function to compute bounding box for the plot.
    """
    # set margins
    pos = np.asarray((x, y))
    minxy, maxxy = pos.min(axis=1), pos.max(axis=1)
    xy1 = minxy - margin * (maxxy - minxy)
    xy2 = maxxy + margin * (maxxy - minxy)
    return tuple(xy1), tuple(xy2)


def projected_area_factor(ax, original_crs=4326):
    """
    Helper function to get the area scale of the current projection in
    reference to the default projection.

    The default 'original crs' is assumed to be 4326, which translates
    to the cartopy default cartopy.crs.PlateCarree()
    """
    if not hasattr(ax, "projection"):
        return 1
    x1, x2, y1, y2 = ax.get_extent()
    pbounds = get_projection_from_crs(original_crs).transform_points(
        ax.projection, np.array([x1, x2]), np.array([y1, y2])
    )

    return np.sqrt(
        abs((x2 - x1) * (y2 - y1)) / abs((pbounds[0] - pbounds[1])[:2].prod())
    )


def draw_map_cartopy(ax, geomap=True, color_geomap=None):
    resolution = "50m" if isinstance(geomap, bool) else geomap
    assert resolution in [
        "10m",
        "50m",
        "110m",
    ], "Resolution has to be one of '10m', '50m', '110m'"

    if not color_geomap:
        color_geomap = {}
    elif not isinstance(color_geomap, dict):
        color_geomap = {
            "ocean": "lightblue",
            "land": "whitesmoke",
            "border": "darkgray",
            "coastline": "black",
        }

    if "land" in color_geomap:
        ax.add_feature(
            cartopy.feature.LAND.with_scale(resolution), facecolor=color_geomap["land"]
        )

    if "ocean" in color_geomap:
        ax.add_feature(
            cartopy.feature.OCEAN.with_scale(resolution),
            facecolor=color_geomap["ocean"],
        )

    ax.add_feature(
        cartopy.feature.BORDERS.with_scale(resolution),
        linewidth=0.3,
        color=color_geomap.get("border", "k"),
    )

    ax.add_feature(
        cartopy.feature.COASTLINE.with_scale(resolution),
        linewidth=0.3,
        color=color_geomap.get("coastline", "k"),
    )


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


def add_legend_lines(ax, sizes, labels, patch_kw={}, legend_kw={}):
    """
    Add a legend for lines and links.

    Parameters
    ----------
    ax : matplotlib ax
    sizes : list-like, float
        Size of the line reference; for example [3, 2, 1]
    labels : list-like, str
        Label of the line reference; for example ["30 GW", "20 GW", "10 GW"]
    patch_kw : defaults to {}
        Keyword arguments passed to plt.Line2D
    legend_kw : defaults to {}
        Keyword arguments passed to ax.legend
    """
    sizes = np.atleast_1d(sizes)
    labels = np.atleast_1d(labels)

    assert len(sizes) == len(labels), "Sizes and labels must have the same length."

    handles = [plt.Line2D([0], [0], linewidth=s, **patch_kw) for s in sizes]

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

    assert len(colors) == len(labels), "Colors and labels must have the same length."

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

    assert len(sizes) == len(labels), "Sizes and labels must have the same length."

    if hasattr(ax, "projection"):
        area_correction = projected_area_factor(ax, srid) ** 2
        sizes = [s * area_correction for s in sizes]

    handles = [Circle((0, 0), radius=s**0.5, **patch_kw) for s in sizes]

    legend = ax.legend(
        handles, labels, handler_map={Circle: HandlerCircle()}, **legend_kw
    )

    ax.get_figure().add_artist(legend)


def _flow_ds_from_arg(flow, n, branch_components):
    if isinstance(flow, pd.Series):
        if not isinstance(flow.index, pd.MultiIndex):
            raise ValueError(
                "Argument 'flow' is a pandas.Series without "
                "a MultiIndex. Please provide a multiindexed series, with "
                "the first level being a subset of 'branch_components'."
            )
        return flow
    if flow in n.snapshots:
        return pd.concat(
            {
                c: n.pnl(c).p0.loc[flow]
                for c in branch_components
                if not n.pnl(c).p0.empty
            },
            sort=True,
        )
    if isinstance(flow, str) or callable(flow):
        return pd.concat(
            [n.pnl(c).p0 for c in branch_components],
            axis=1,
            keys=branch_components,
            sort=True,
        ).agg(flow, axis=0)


def directed_flow(coords, flow, color, area_factor=1, cmap=None, alpha=1):
    """
    Helper function to generate arrows from flow data.
    """
    # this funtion is used for diplaying arrows representing the network flow
    data = pd.DataFrame(
        {
            "arrowsize": flow.abs().pipe(np.sqrt).clip(lower=1e-8),
            "direction": np.sign(flow),
            "linelength": (
                np.sqrt((coords.x1 - coords.x2) ** 2.0 + (coords.y1 - coords.y2) ** 2)
            ),
        }
    )
    data = data.join(coords)
    if area_factor:
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
        edgecolors="k",
        linewidths=0.0,
        zorder=4,
    )


def autogenerate_coordinates(n, assign=False, layouter=None):
    """
    Automatically generate bus coordinates for the network graph according to a
    layouting function from `networkx <https://networkx.github.io/>`_.

    Parameters
    ----------
    n : pypsa.Network
    assign : bool, default False
        Assign generated coordinates to the network bus coordinates
        at ``n.buses[['x', 'y']]``.
    layouter : networkx.drawing.layout function, default None
        Layouting function from `networkx <https://networkx.github.io/>`_. See
        `list <https://networkx.github.io/documentation/stable/reference/drawing.html#module-networkx.drawing.layout>`_
        of available options. By default coordinates are determined for a
        `planar layout <https://networkx.github.io/documentation/stable/reference/generated/networkx.drawing.layout.planar_layout.html#networkx.drawing.layout.planar_layout>`_
        if the network graph is planar, otherwise for a
        `Kamada-Kawai layout <https://networkx.github.io/documentation/stable/reference/generated/networkx.drawing.layout.kamada_kawai_layout.html#networkx.drawing.layout.kamada_kawai_layout>`_.

    Returns
    -------
    coordinates : pd.DataFrame
        DataFrame containing the generated coordinates with
        buses as index and ['x', 'y'] as columns.

    Examples
    --------
    >>> autogenerate_coordinates(network)
    >>> autogenerate_coordinates(network, assign=True, layouter=nx.circle_layout)
    """
    G = n.graph()

    if layouter is None:
        if is_planar := nx.check_planarity(G)[0]:
            layouter = nx.planar_layout
        else:
            layouter = nx.kamada_kawai_layout

    coordinates = pd.DataFrame(layouter(G)).T.rename({0: "x", 1: "y"}, axis=1)

    if assign:
        n.buses[["x", "y"]] = coordinates

    return coordinates


def _get_coordinates(n, layouter=None):
    if layouter is not None or n.buses[["x", "y"]].isin([np.nan, 0]).all().all():
        coordinates = autogenerate_coordinates(n, layouter=layouter)
        return coordinates["x"], coordinates["y"]
    return n.buses["x"], n.buses["y"]


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
        x = x + np.random.uniform(low=-jitter, high=jitter, size=len(x))
        y = y + np.random.uniform(low=-jitter, high=jitter, size=len(y))

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
        assert len(size) == 2, "Parameter size must specify a tuple (width, height)."
        fig["layout"].update(dict(width=size[0], height=size[1]))

    if mapbox:
        if mapbox_token != "":
            mapbox_parameters["accesstoken"] = mapbox_token

        mapbox_parameters.setdefault("style", mapbox_style)

        if mapbox_parameters["style"] in _token_required_mb_styles:
            assert "accesstoken" in mapbox_parameters.keys(), (
                "Using Mapbox "
                "layout styles requires a valid access token from https://www.mapbox.com/, "
                f"style which do not require a token are:\n{', '.join(_open__mb_styles)}."
            )

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
